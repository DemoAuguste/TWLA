from __future__ import annotations

import os
import sys
import math
import copy
import argparse
import random
from typing import Dict, Tuple, Optional, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from datautils import get_loaders
from quantize.quantizer import UniformAffineQuantizer
from TWLA.quantize import gptq_fwrd
import lm_eval
from lm_eval import utils as lm_eval_utils
from lm_eval.models.huggingface import HFLM

LAC_PMIN = 0.5
LAC_PMAX = 1.0
LAC_PSTEP = 0.005
LAC_CHUNK_ROWS = 4096
LAC_NCALIB_BATCHES = 8


def make_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def parse_abits_set(s: str) -> Tuple[int, ...]:
    s = str(s).strip()
    if s.isdigit() and "," not in s:
        return tuple(int(ch) for ch in s)
    xs = []
    for t in s.split(","):
        t = t.strip()
        if t:
            xs.append(int(t))
    return tuple(xs)


def atomic_torch_save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def cache_paths(cache_dir: str):
    base = os.path.join(cache_dir, "baseline.pt")
    single_dir = os.path.join(cache_dir, "single")
    pair_dir = os.path.join(cache_dir, "pair")
    lac_dir = os.path.join(cache_dir, "lac")
    return base, single_dir, pair_dir, lac_dir


def single_item_path(single_dir: str, blk_i: int, ab: int) -> str:
    return os.path.join(single_dir, f"blk{int(blk_i):04d}_A{int(ab)}.pt")


def pair_item_path(pair_dir: str, i: int, bp: int, b: int) -> str:
    return os.path.join(pair_dir, f"pair{int(i - 1):04d}_{int(i):04d}_A{int(bp)}_{int(b)}.pt")


def lac_item_path(lac_dir: str, ab: int) -> str:
    return os.path.join(lac_dir, f"lac_A{int(ab)}.pt")


class QLinear(nn.Linear):
    @torch.no_grad()
    def init_act_quant(self, args) -> None:
        if int(getattr(args, "abits", 16)) >= 16:
            self.act_quant = None
            return
        lac = float(getattr(self, "act_lac", float(getattr(args, "lac", 0.9))))
        self.act_quant = UniformAffineQuantizer(
            n_bits=int(args.abits),
            symmetric=False,
            per_channel_axes=[],
            dynamic=True,
            dynamic_method=str(getattr(args, "a_dynamic_method", "per_token")),
            act_group_size=getattr(args, "act_group_size", None),
            lac=lac,
        )

    @torch.no_grad()
    def cali_begin(self, args, p_list: Iterable[float], abits: int, chunk_rows: int) -> None:
        self._cali_enabled = True
        self._cali_chunk_rows = int(chunk_rows)
        self._cali_err = {float(p): 0.0 for p in p_list}
        self._cali_cnt = 0
        self._cali_p_list = [float(p) for p in p_list]
        self._cali_default_p = float(getattr(args, "lac", self._cali_p_list[0]))

        self._cali_quant = UniformAffineQuantizer(
            n_bits=int(abits),
            symmetric=False,
            per_channel_axes=[],
            dynamic=True,
            dynamic_method=str(getattr(args, "a_dynamic_method", "per_token")),
            act_group_size=getattr(args, "act_group_size", None),
            lac=float(self._cali_p_list[0]),
        ).to(self.weight.device)
        self._cali_quant.enable = True

    @torch.no_grad()
    def cali_update(self, x: torch.Tensor) -> None:
        x2 = x.reshape(-1, x.shape[-1])
        n = int(x2.shape[0])
        chunk = int(self._cali_chunk_rows)
        out_dim = int(self.weight.shape[0])

        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            xb = x2[s:e]
            y_ref = F.linear(xb, self.weight, self.bias)

            for p in self._cali_p_list:
                self._cali_quant.lac = float(p)
                xq = self._cali_quant(xb)
                yq = F.linear(xq, self.weight, self.bias)
                sse = (yq - y_ref).float().pow(2).sum().item()
                if math.isfinite(sse):
                    self._cali_err[float(p)] += float(sse)

            self._cali_cnt += (e - s) * out_dim

    @torch.no_grad()
    def cali_end(self) -> Tuple[float, float]:
        denom = max(int(getattr(self, "_cali_cnt", 0)), 1)
        best_p: Optional[float] = None
        best_mse = float("inf")

        for p, sse in self._cali_err.items():
            mse = float(sse) / float(denom)
            if not math.isfinite(mse):
                continue
            if mse < best_mse:
                best_mse = mse
                best_p = float(p)

        self._cali_enabled = False
        if hasattr(self, "_cali_quant"):
            del self._cali_quant

        if best_p is None:
            return float(self._cali_default_p), float("inf")
        return float(best_p), float(best_mse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        init_shape = x.shape

        if hasattr(self, "L") and self.L is not None:
            device = x.device
            L = self.L.to(device) if self.L.device != device else self.L
            R = self.R.to(device) if self.R.device != device else self.R

            x = x.reshape(-1, self.dim_l, self.dim_r)
            x = L @ x @ R
            x = x.reshape(init_shape)
        if getattr(self, "_cali_enabled", False):
            self.cali_update(x)
        if getattr(self, "act_quant", None) is not None:
            x = self.act_quant(x)
        return super().forward(x)


def import_rotated_checkpoint_as_qlinear(model, load_path: str, args) -> int:
    pkg = torch.load(load_path, map_location="cpu")
    layers_pkg = pkg["layers"]
    filled = 0

    for name, layer in model.model.named_modules():
        if not isinstance(layer, nn.Linear):
            continue
        if "lm_head" in name:
            continue
        if name not in layers_pkg:
            continue

        d = layers_pkg[name]
        layer.weight.data = d["weight"].to(dtype=torch.float16)

        if d["bias"] is not None:
            if layer.bias is None:
                layer.bias = nn.Parameter(d["bias"].to(dtype=torch.float16))
            else:
                layer.bias.data = d["bias"].to(dtype=torch.float16)

        layer.__class__ = QLinear

        def _reg(k: str) -> None:
            t = d.get(k, None)
            if t is None:
                return
            t = t.to(dtype=torch.float16).contiguous()
            if hasattr(layer, k) and isinstance(getattr(layer, k), torch.Tensor):
                getattr(layer, k).data.copy_(t)
            else:
                layer.register_buffer(k, t, persistent=True)

        _reg("L")
        _reg("R")
        if int(d.get("dim_l", 0) or 0) > 0:
            layer.dim_l = int(d["dim_l"])
        if int(d.get("dim_r", 0) or 0) > 0:
            layer.dim_r = int(d["dim_r"])

        layer.init_act_quant(args)
        filled += 1

    return filled


def set_block_abits(model, args, block_idx: int, abits: int) -> None:
    a2 = copy.copy(args)
    a2.abits = int(abits)
    block = model.model.layers[int(block_idx)]
    for m in block.modules():
        if isinstance(m, QLinear):
            m.init_act_quant(a2)


@torch.no_grad()
def collect_layerwise_hidden_states_for_calibration(model, input_ids: torch.Tensor, dev: str) -> None:
    seqlen = int(model.seqlen)
    if input_ids.dim() != 2 or input_ids.size(1) != seqlen:
        raise ValueError(f"expect [bs,{seqlen}], got {tuple(input_ids.shape)}")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    bs = int(input_ids.size(0))

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    hidden = int(model.config.hidden_size)

    inps = torch.zeros((bs, seqlen, hidden), dtype=dtype, device=dev)
    cache = {"i": 0, "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["kwargs"] = kwargs
            cache["i"] += 1
            raise ValueError

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    layers[0] = Catcher(layers[0])

    for i in range(bs):
        batch_i = input_ids[i:i + 1].to(dev)
        try:
            model(batch_i)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache["kwargs"] or {}
    for k, v in list(layer_kwargs.items()):
        if isinstance(v, torch.Tensor):
            layer_kwargs[k] = v.to(dev)

    for li in range(len(layers)):
        layer = layers[li].to(dev)
        for bi in range(bs):
            outs[bi] = layer(inps[bi:bi + 1], **layer_kwargs)[0]
        layers[li] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache


def load_lac_cache(lac_dir: str, ab: int) -> Optional[Dict[str, float]]:
    pt = lac_item_path(lac_dir, ab)
    if not os.path.isfile(pt):
        return None
    pkg = torch.load(pt, map_location="cpu")
    d = pkg.get("lac", {})
    return {str(k): float(v) for k, v in d.items()}


def save_lac_cache(lac_dir: str, ab: int, lac_map: Dict[str, float]) -> None:
    os.makedirs(lac_dir, exist_ok=True)
    pkg = {"abits": int(ab), "lac": {str(k): float(v) for k, v in lac_map.items()}}
    atomic_torch_save(pkg, lac_item_path(lac_dir, ab))


def calibrate_per_layer_lac(model, dataloader, dev: str, args, abits_list: Tuple[int, ...]):
    base, _, _, lac_dir = cache_paths(str(args.dp_cache))
    _ = base  # keep cache layout consistent

    p_list = np.arange(LAC_PMIN, LAC_PMAX, LAC_PSTEP)

    name_map: Dict[int, str] = {}
    qlinears = []
    for full_name, m in model.model.named_modules():
        if isinstance(m, QLinear):
            name_map[id(m)] = str(full_name)
            qlinears.append(m)

    lac_table: Dict[Tuple[str, int], float] = {}

    for ab in abits_list:
        ab = int(ab)
        if ab >= 16:
            continue

        cached = load_lac_cache(lac_dir, ab)
        if cached:
            for full_name, p in cached.items():
                lac_table[(str(full_name), ab)] = float(p)
            continue

        a2 = copy.copy(args)
        a2.abits = ab
        for m in qlinears:
            m.init_act_quant(a2)
        for m in qlinears:
            m.cali_begin(args, p_list, abits=ab, chunk_rows=LAC_CHUNK_ROWS)

        nb = 0
        pbar = tqdm(total=LAC_NCALIB_BATCHES, desc=f"LAC A{ab}", dynamic_ncols=True)
        for batch in dataloader:
            input_ids = batch[0] if isinstance(batch, (list, tuple)) else batch
            collect_layerwise_hidden_states_for_calibration(model, input_ids, dev)
            nb += 1
            pbar.update(1)
            if nb >= LAC_NCALIB_BATCHES:
                break
        pbar.close()

        lac_for_ab: Dict[str, float] = {}
        for m in qlinears:
            best_p, _ = m.cali_end()
            full_name = name_map[id(m)]
            lac_table[(full_name, ab)] = float(best_p)
            lac_for_ab[full_name] = float(best_p)

        save_lac_cache(lac_dir, ab, lac_for_ab)

    return lac_table, name_map


def apply_lac_table(model, args, bits, lac_table, name_map) -> None:
    for blk_i, ab in enumerate(bits):
        ab = int(ab)
        block = model.model.layers[int(blk_i)]
        for m in block.modules():
            if isinstance(m, QLinear) and (getattr(m, "act_quant", None) is not None):
                full_name = name_map.get(id(m), None)
                p = None if full_name is None else lac_table.get((full_name, ab), None)
                m.act_lac = float(getattr(args, "lac", 0.9)) if p is None else float(p)
                a2 = copy.copy(args)
                a2.abits = ab
                m.init_act_quant(a2)


def load_dp_tables(dp_cache: str, abits_list: Tuple[int, ...], baseline_abits: int):
    base_path, single_dir, pair_dir, _ = cache_paths(dp_cache)

    if not os.path.isfile(base_path):
        raise RuntimeError("baseline.pt not found in dp_cache; run dp_sweep_cache.py first.")
    base_pkg = torch.load(base_path, map_location="cpu")
    base_nll = float(base_pkg["base_nll"])
    if int(base_pkg["baseline_abits"]) != int(baseline_abits):
        raise RuntimeError("baseline_abits mismatch with cache.")

    L = None
    for fn in os.listdir(single_dir):
        if fn.startswith("blk") and fn.endswith(".pt"):
            pkg = torch.load(os.path.join(single_dir, fn), map_location="cpu")
            L = max(L or 0, int(pkg["blk"]) + 1)
    if L is None:
        raise RuntimeError("single cache empty.")

    delta1 = {i: {int(baseline_abits): 0.0} for i in range(L)}
    for i in range(L):
        for b in abits_list:
            b = int(b)
            if b == int(baseline_abits):
                continue
            pt = single_item_path(single_dir, i, b)
            if not os.path.isfile(pt):
                raise RuntimeError(f"missing single cache: {pt}")
            pkg = torch.load(pt, map_location="cpu")
            delta1[i][b] = float(pkg["nll"]) - base_nll

    pair_table = {i: {int(bp): {} for bp in abits_list} for i in range(1, L)}
    for i in range(1, L):
        for bp in abits_list:
            for b in abits_list:
                pt = pair_item_path(pair_dir, i, int(bp), int(b))
                if not os.path.isfile(pt):
                    raise RuntimeError(f"missing pair cache: {pt}")
                pkg = torch.load(pt, map_location="cpu")
                pair_table[i][int(bp)][int(b)] = float(pkg["nll"])

    return base_nll, L, delta1, pair_table


def build_pairwise_interaction(base_nll: float, delta1, pair_table, abits_list: Tuple[int, ...]):
    L = len(delta1)
    k = {i: {int(bp): {} for bp in abits_list} for i in range(1, L)}
    for i in range(1, L):
        for bp in abits_list:
            for b in abits_list:
                pair_nll = float(pair_table[i][int(bp)][int(b)])
                pair_delta = pair_nll - float(base_nll)
                kij = pair_delta - float(delta1[i - 1][int(bp)]) - float(delta1[i][int(b)])
                k[i][int(bp)][int(b)] = float(kij)
    return k


def dp_allocate_bits_order2(delta1, k, abits_list: Tuple[int, ...], avg_bit_budget: float, beta: float):
    abits_list = tuple(int(x) for x in abits_list)
    L = len(delta1)
    budget = int(round(float(avg_bit_budget) * L))
    K = len(abits_list)
    INF = 1e18

    dp = [[[INF] * K for _ in range(budget + 1)] for _ in range(L)]
    prev = [[[(None, None)] * K for _ in range(budget + 1)] for _ in range(L)]

    for kb, b in enumerate(abits_list):
        if b <= budget:
            dp[0][b][kb] = float(delta1[0][b])
            prev[0][b][kb] = (0, None)

    for i in range(1, L):
        for c in range(budget + 1):
            for kb, b in enumerate(abits_list):
                if c - b < 0:
                    continue
                best = INF
                best_prev = None
                for kbp, bp in enumerate(abits_list):
                    val_prev = dp[i - 1][c - b][kbp]
                    if val_prev >= INF / 2:
                        continue
                    val = val_prev + float(delta1[i][b]) + float(beta) * float(k[i][bp][b])
                    if val < best:
                        best = val
                        best_prev = (c - b, kbp)
                dp[i][c][kb] = best
                prev[i][c][kb] = best_prev

    best_val = INF
    best_c = 0
    best_kb = 0
    for c in range(budget + 1):
        for kb in range(K):
            v = dp[L - 1][c][kb]
            if v < best_val:
                best_val = v
                best_c = c
                best_kb = kb

    bits = [0] * L
    c = int(best_c)
    kb = int(best_kb)
    for i in range(L - 1, -1, -1):
        bits[i] = int(abits_list[kb])
        pc, pkb = prev[i][c][kb]
        if pkb is None:
            break
        c, kb = int(pc), int(pkb)

    return bits, float(best_val), int(best_c)

def save_quantized_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)


def load_quantized_model(model, path: str, strict: bool = True):
    sd = torch.load(path, map_location="cpu")
    return model.load_state_dict(sd, strict=bool(strict))


@torch.no_grad()
def evaluate_ppl(model, args, datasets=("wikitext2",)) -> None:
    from eval_ppl_utils import llama_eval
    dev = "cuda:0"
    for ds in datasets:
        _, testloader = get_loaders(ds, seed=int(args.seed), seqlen=int(args.seqlen), model=args.model)
        llama_eval(model, testloader, dev, ds)


@torch.no_grad()
def evaluate_qa(model, tokenizer) -> Dict:
    auto_batch_size = 32
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=auto_batch_size)
    task_manager = lm_eval.tasks.TaskManager(include_defaults=True)
    wanted = ["hellaswag", "winogrande", "piqa", "lambada_openai", "lambada_standard", "arc_easy", "arc_challenge"]
    task_names = lm_eval_utils.pattern_match(wanted, task_manager.all_tasks)
    out = lm_eval.simple_evaluate(hflm, tasks=task_names, batch_size=auto_batch_size, task_manager=task_manager)
    return out["results"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seqlen", type=int, default=2048)

    p.add_argument("--import_rotated", type=str, required=True)

    p.add_argument("--dataset", type=str, default="wikitext2")
    p.add_argument("--nsamples", type=int, default=128)

    p.add_argument("--wbits", type=int, default=4)
    p.add_argument("--symmetric", action="store_true", default=True)
    p.add_argument("--group_size", type=int, default=None)
    p.add_argument("--swc", type=float, default=0.9)
    p.add_argument("--mse", type=float, default=None)
    p.add_argument("--act_order", action="store_true", default=False)

    p.add_argument("--abits", type=int, default=8)
    p.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    p.add_argument("--act_group_size", type=int, default=None)
    p.add_argument("--lac", type=float, default=0.9)

    p.add_argument("--dp_cache", type=str, required=True)
    p.add_argument("--dp_abits_set", type=str, default="2468")
    p.add_argument("--dp_baseline_abits", type=int, default=8)
    p.add_argument("--dp_avg_abits", type=float, default=4.0)
    p.add_argument("--dp_pair_beta", type=float, default=0.5)

    p.add_argument("--save_quant_model", type=str, default=None)
    p.add_argument("--load_quant_model", type=str, default=None)

    p.add_argument("--eval_qa", action="store_true", default=False)

    args = p.parse_args()
    make_deterministic(int(args.seed))

    abits_list = tuple(int(x) for x in parse_abits_set(args.dp_abits_set))
    baseline_abits = int(args.dp_baseline_abits)
    if baseline_abits not in abits_list:
        raise ValueError("dp_baseline_abits must be in dp_abits_set.")

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cpu", torch_dtype="auto")
    model.seqlen = int(args.seqlen)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    filled = import_rotated_checkpoint_as_qlinear(model, args.import_rotated, args)
    if filled <= 0:
        raise RuntimeError("No layers loaded from rotated checkpoint.")

    model.half()
    model.eval()

    trainloader, _ = get_loaders(
        args.dataset,
        nsamples=int(args.nsamples),
        seed=int(args.seed),
        model=args.model,
        seqlen=int(args.seqlen),
    )

    if args.load_quant_model is not None:
        info = load_quantized_model(model, args.load_quant_model, strict=False)
    else:
        gptq_fwrd.twla_fwrd(model, tokenizer, trainloader, "cuda", args)
        if args.save_quant_model is not None:
            save_quantized_model(model, args.save_quant_model)


    disable_amp = int(getattr(args, "abits", 16)) >= 16

    if disable_amp:
        for i in range(len(model.model.layers)):
            set_block_abits(model, args, i, 16)
    else:
        lac_loader, _ = get_loaders("wikitext2", seed=int(args.seed), seqlen=int(args.seqlen), model=args.model)
        lac_table, lac_name_map = calibrate_per_layer_lac(model, lac_loader, dev="cuda:0", args=args, abits_list=abits_list)

        base_nll, L, delta1, pair_table = load_dp_tables(str(args.dp_cache), abits_list, baseline_abits)
        k = build_pairwise_interaction(float(base_nll), delta1, pair_table, abits_list)
        bits, best_obj, used_budget = dp_allocate_bits_order2(
            delta1, k, abits_list, float(args.dp_avg_abits), float(args.dp_pair_beta)
        )

        for i, b in enumerate(bits):
            set_block_abits(model, args, i, int(b))
        apply_lac_table(model, args, bits, lac_table, lac_name_map)


    model.cuda().half()
    model.eval()

    evaluate_ppl(model, args)
    if bool(args.eval_qa):
        _ = evaluate_qa(model, tokenizer)


if __name__ == "__main__":
    main()
