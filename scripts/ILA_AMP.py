from __future__ import annotations

import os
import sys
import math
import queue
import argparse
import random
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import torch.nn.functional as F

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from datautils import get_loaders

from quantize.quantizer import UniformAffineQuantizer

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
        ).to(self.weight.device)
        self.act_quant.enable = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        init_shape = x.shape

        if hasattr(self, "L") and self.L is not None and hasattr(self, "R") and self.R is not None:
            device = x.device
            L = self.L.to(device) if self.L.device != device else self.L
            R = self.R.to(device) if self.R.device != device else self.R

            x = x.reshape(-1, int(self.dim_l), int(self.dim_r))
            x = L @ x @ R
            x = x.reshape(init_shape)

        # activation quant
        if getattr(self, "act_quant", None) is not None:
            x = self.act_quant(x)

        return F.linear(x, self.weight, self.bias)



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


def parse_abits_set(s: str):
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
    meta = os.path.join(cache_dir, "meta.pt")
    base = os.path.join(cache_dir, "baseline.pt")
    single_dir = os.path.join(cache_dir, "single")
    pair_dir = os.path.join(cache_dir, "pair")
    return meta, base, single_dir, pair_dir


def single_item_path(single_dir: str, blk_i: int, ab: int) -> str:
    return os.path.join(single_dir, f"blk{int(blk_i):04d}_A{int(ab)}.pt")


def pair_item_path(pair_dir: str, i: int, bp: int, b: int) -> str:
    return os.path.join(pair_dir, f"pair{int(i-1):04d}_{int(i):04d}_A{int(bp)}_{int(b)}.pt")


def _set_or_register_buffer(module: nn.Module, name: str, value: torch.Tensor) -> None:
    if value is None:
        if hasattr(module, "_buffers") and name in module._buffers:
            module._buffers.pop(name, None)
        if hasattr(module, name):
            try:
                delattr(module, name)
            except Exception:
                pass
        return

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"buffer {name} must be a torch.Tensor, got {type(value)}")

    if hasattr(module, "_buffers") and name in module._buffers:
        module._buffers[name] = value
        return

    if hasattr(module, name):
        try:
            delattr(module, name)
        except Exception:
            if hasattr(module, "_buffers"):
                module._buffers[name] = value
                return
            raise

    module.register_buffer(name, value, persistent=True)


def export_import_rotated(model, import_rotated: str, args) -> int:
    pkg = torch.load(import_rotated, map_location="cpu")
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

        L = d.get("L", None)
        R = d.get("R", None)
        if L is not None:
            _set_or_register_buffer(layer, "L", L.to(dtype=torch.float16).contiguous())
        else:
            _set_or_register_buffer(layer, "L", None)

        if R is not None:
            _set_or_register_buffer(layer, "R", R.to(dtype=torch.float16).contiguous())
        else:
            _set_or_register_buffer(layer, "R", None)

        if L is not None and R is not None:
            layer.dim_l = int(d.get("dim_l", 0) or 0)
            layer.dim_r = int(d.get("dim_r", 0) or 0)

        layer.init_act_quant(args)

        filled += 1


    return filled


def set_block_abits(model, args, block_idx: int, abits: int):
    abits = int(abits)
    a2 = copy.copy(args)
    a2.abits = abits
    block = model.model.layers[int(block_idx)]
    for m in block.modules():
        if isinstance(m, QLinear):
            m.init_act_quant(a2)


@torch.no_grad()
def llama_eval_return_nll(model, testenc, dev: str):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
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

    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
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

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    loss_fct = nn.CrossEntropyLoss(reduction="mean")

    total_nll = 0.0
    total_tokens = 0

    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        n_tokens = shift_labels.numel()
        total_nll += float(loss) * n_tokens
        total_tokens += n_tokens

    model.config.use_cache = use_cache
    del inps, outs, testenc, layer_kwargs, cache
    torch.cuda.empty_cache()

    avg_nll = total_nll / max(total_tokens, 1)
    return float(avg_nll)


def _dp_worker(task_queue, result_queue, stop_event, gpu_id: int, args, dataset: str, baseline_abits: int):
    device = f"cuda:{int(gpu_id)}"
    torch.cuda.set_device(device)
    make_deterministic(int(args.seed) + 1000 + int(gpu_id))

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cpu", torch_dtype="auto")
    model.seqlen = int(args.seqlen)
    _ = export_import_rotated(model, args.import_rotated, args)
    model.half()
    model.eval()

    _, testloader = get_loaders(dataset, seed=int(args.seed), seqlen=int(args.seqlen), model=args.model)
    L = len(model.model.layers)

    for i in range(L):
        set_block_abits(model, args, i, baseline_abits)

    while not stop_event.is_set():
        try:
            task = task_queue.get(timeout=1)
        except queue.Empty:
            continue
        if task is None:
            break

        kind = task[0]
        try:
            if kind == "single":
                _, blk_i, ab = task
                blk_i = int(blk_i)
                ab = int(ab)
                set_block_abits(model, args, blk_i, baseline_abits)
                set_block_abits(model, args, blk_i, ab)
                nll = llama_eval_return_nll(model, testloader, device)
                result_queue.put(("single", blk_i, ab, float(nll)))
                set_block_abits(model, args, blk_i, baseline_abits)

            elif kind == "pair":
                _, i, bp, b = task
                i = int(i)
                bp = int(bp)
                b = int(b)
                if not (1 <= i < L):
                    raise ValueError(f"pair index out of range: i={i}, L={L}")

                set_block_abits(model, args, i - 1, baseline_abits)
                set_block_abits(model, args, i, baseline_abits)
                set_block_abits(model, args, i - 1, bp)
                set_block_abits(model, args, i, b)

                nll = llama_eval_return_nll(model, testloader, device)
                result_queue.put(("pair", i, bp, b, float(nll)))

                set_block_abits(model, args, i - 1, baseline_abits)
                set_block_abits(model, args, i, baseline_abits)

            else:
                raise ValueError(f"unknown task kind: {kind}")

        except Exception:
            if kind == "single":
                _, blk_i, ab = task
                result_queue.put(("single", int(blk_i), int(ab), None))
            elif kind == "pair":
                _, i, bp, b = task
                result_queue.put(("pair", int(i), int(bp), int(b), None))
            else:
                result_queue.put(("unknown", None))

    del model
    torch.cuda.empty_cache()
    result_queue.put(("done", int(gpu_id)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--import_rotated", type=str, required=True)
    p.add_argument("--dp_cache", type=str, required=True)

    p.add_argument("--dataset", type=str, default="wikitext2")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--abits", type=int, default=8)  
    p.add_argument("--lac", type=float, default=0.9)
    p.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    p.add_argument("--act_group_size", type=int, default=None)


    p.add_argument("--dp_abits_set", type=str, default="2468")
    p.add_argument("--dp_baseline_abits", type=int, default=8)
    p.add_argument("--dp_ngpus", type=int, default=4)

    args = p.parse_args()
    make_deterministic(int(args.seed))
    mp.set_start_method("spawn", force=True)

    abits_list = tuple(int(x) for x in parse_abits_set(args.dp_abits_set))
    baseline_abits = int(args.dp_baseline_abits)
    if baseline_abits not in abits_list:
        raise ValueError("dp_baseline_abits must be in dp_abits_set.")

    meta_path, base_path, single_dir, pair_dir = cache_paths(str(args.dp_cache))
    os.makedirs(single_dir, exist_ok=True)
    os.makedirs(pair_dir, exist_ok=True)

    atomic_torch_save(
        {
            "_meta": {
                "model": str(args.model),
                "import_rotated": str(args.import_rotated),
                "dataset": str(args.dataset),
                "seqlen": int(args.seqlen),
                "seed": int(args.seed),
                "abits_list": [int(x) for x in abits_list],
                "baseline_abits": int(baseline_abits),
                "order": 2,
            }
        },
        meta_path,
    )

    base = None
    if os.path.isfile(base_path):
        pkg = torch.load(base_path, map_location="cpu")
        base = (float(pkg["base_nll"]), int(pkg["baseline_abits"]))

    if base is None:
        dev = "cuda:0"
        model0 = AutoModelForCausalLM.from_pretrained(args.model, device_map="cpu", torch_dtype="auto")
        model0.seqlen = int(args.seqlen)
        _ = export_import_rotated(model0, args.import_rotated, args)
        model0.half()
        model0.eval()
        _, testloader0 = get_loaders(args.dataset, seed=int(args.seed), seqlen=int(args.seqlen), model=args.model)

        Ltmp = len(model0.model.layers)
        for i in range(Ltmp):
            set_block_abits(model0, args, i, baseline_abits)

        base_nll = llama_eval_return_nll(model0, testloader0, dev)
        atomic_torch_save({"base_nll": float(base_nll), "baseline_abits": int(baseline_abits)}, base_path)
        del model0
        torch.cuda.empty_cache()
    else:
        base_nll, cached_baseline = base
        if int(cached_baseline) != baseline_abits:
            raise RuntimeError(f"baseline mismatch: cache={cached_baseline}, args={baseline_abits}")

    model_tmp = AutoModelForCausalLM.from_pretrained(args.model, device_map="cpu", torch_dtype="auto")
    model_tmp.seqlen = int(args.seqlen)
    _ = export_import_rotated(model_tmp, args.import_rotated, args)
    L = len(model_tmp.model.layers)
    del model_tmp
    torch.cuda.empty_cache()

    # -----------------------------
    # Spawn workers once (keep alive)
    # -----------------------------
    ctx = mp.get_context("spawn")
    tq = ctx.Queue()
    rq = ctx.Queue()
    ev = ctx.Event()
    workers = []
    for gid in range(int(args.dp_ngpus)):
        proc = ctx.Process(
            target=_dp_worker,
            args=(tq, rq, ev, gid, args, args.dataset, baseline_abits),
            daemon=True,
        )
        proc.start()
        workers.append(proc)

    # -----------------------------
    # Phase 1: single
    # -----------------------------
    single_tasks = []
    for i in range(L):
        for b in abits_list:
            b = int(b)
            if b == baseline_abits:
                continue
            out_pt = single_item_path(single_dir, i, b)
            if os.path.isfile(out_pt):
                continue
            single_tasks.append(("single", int(i), int(b)))

    for t in single_tasks:
        tq.put(t)

    pbar1 = tqdm(total=len(single_tasks), desc="ILA-AMP-order1", dynamic_ncols=True)
    got_single = 0
    while got_single < len(single_tasks):
        msg = rq.get()
        if msg[0] == "done":
            continue
        if msg[0] != "single":
            raise RuntimeError(f"unexpected message during single phase: {msg}")
        _, blk_i, ab, nll = msg
        got_single += 1
        if nll is None:
            raise RuntimeError(f"single failed: blk={blk_i} ab={ab}")
        atomic_torch_save(
            {"blk": int(blk_i), "ab": int(ab), "nll": float(nll)},
            single_item_path(single_dir, blk_i, ab),
        )
        pbar1.update(1)
    pbar1.close()

    # -----------------------------
    # Phase 2: pair (now single files exist, safe to prefill)
    # -----------------------------
    pair_tasks = []
    for i in range(1, L):
        for bp in abits_list:
            for b in abits_list:
                bp = int(bp)
                b = int(b)

                out_pt = pair_item_path(pair_dir, i, bp, b)
                if os.path.isfile(out_pt):
                    continue

                if bp == baseline_abits and b == baseline_abits:
                    atomic_torch_save(
                        {"i": i, "bp": bp, "b": b, "nll": float(base_nll), "filled": "baseline"},
                        out_pt,
                    )
                    continue

                if bp == baseline_abits and b != baseline_abits:
                    pt = single_item_path(single_dir, i, b)
                    pkg = torch.load(pt, map_location="cpu")
                    atomic_torch_save(
                        {"i": i, "bp": bp, "b": b, "nll": float(pkg["nll"]), "filled": "single_i"},
                        out_pt,
                    )
                    continue

                if bp != baseline_abits and b == baseline_abits:
                    pt = single_item_path(single_dir, i - 1, bp)
                    pkg = torch.load(pt, map_location="cpu")
                    atomic_torch_save(
                        {"i": i, "bp": bp, "b": b, "nll": float(pkg["nll"]), "filled": "single_i-1"},
                        out_pt,
                    )
                    continue

                pair_tasks.append(("pair", int(i), int(bp), int(b)))

    for t in pair_tasks:
        tq.put(t)

    pbar2 = tqdm(total=len(pair_tasks), desc="ILA-AMP-order2", dynamic_ncols=True)
    got_pair = 0
    while got_pair < len(pair_tasks):
        msg = rq.get()
        if msg[0] == "done":
            continue
        if msg[0] != "pair":
            raise RuntimeError(f"unexpected message during pair phase: {msg}")
        _, i, bp, b, nll = msg
        got_pair += 1
        if nll is None:
            raise RuntimeError(f"pair failed: i={i} bp={bp} b={b}")
        atomic_torch_save(
            {"i": int(i), "bp": int(bp), "b": int(b), "nll": float(nll)},
            pair_item_path(pair_dir, i, bp, b),
        )
        pbar2.update(1)
    pbar2.close()

    # -----------------------------
    # Send None once, after all tasks are done
    # -----------------------------
    for _ in workers:
        tq.put(None)

    done_workers = 0
    while done_workers < len(workers):
        msg = rq.get()
        if msg[0] == "done":
            done_workers += 1

    ev.set()
    for proc in workers:
        proc.join()

    tq.close()
    tq.join_thread()
    rq.close()
    rq.join_thread()

    print(f"DP cache ready: {args.dp_cache}")


if __name__ == "__main__":
    main()
