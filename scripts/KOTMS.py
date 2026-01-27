from __future__ import annotations

import argparse
import math
import os
import queue
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from quantize.k_preprocessor_ternary import KroneckerSmoothConfig, KroneckerSmoothPreprocessor


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


def atomic_torch_save(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def safe_filename(name: str) -> str:
    s = str(name)
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_")
    return s


def get_decompose_dim_budget(oc: int, ic: int) -> Tuple[int, int]:
    r0 = int(math.isqrt(max(1, (oc * ic) // 64)))
    r0 = max(1, min(r0, ic))
    for r in range(r0, 0, -1):
        if ic % r == 0:
            return ic // r, r
    return ic, 1


def build_kp(args: argparse.Namespace) -> KroneckerSmoothPreprocessor:
    if bool(getattr(args, "use_gmm", False)):
        cfg = KroneckerSmoothConfig(
            use_gmm_training=True,
            gmm_iters=int(args.gmm_iters),
            gmm_lr_r=float(args.gmm_lr_r),
            gmm_lr_l=float(args.gmm_lr_l),
            gmm_ortho_method=str(args.gmm_ortho_method),
            sigma0_ratio=float(args.sigma0_ratio),
            sigma1_ratio=float(args.sigma1_ratio),
            balance_w=float(args.balance_w),
            l=None,
            r=None,
        )
        return KroneckerSmoothPreprocessor(cfg)
    return KroneckerSmoothPreprocessor(KroneckerSmoothConfig(use_gmm_training=False))


class QLinear(nn.Linear):
    kp: Optional[KroneckerSmoothPreprocessor] = None

    def setup_rotation(self) -> None:
        if QLinear.kp is None:
            raise RuntimeError("QLinear.kp is not initialized.")

        ori_shape = self.weight.shape
        ori_dtype = self.weight.dtype

        w = self.weight.data.clone().cuda().float()
        oc, ic = int(w.shape[0]), int(w.shape[-1])
        _ = get_decompose_dim_budget(oc, ic)

        res = QLinear.kp._kronecker_process(w.reshape(ori_shape))
        if not isinstance(res, dict):
            raise RuntimeError("kp._kronecker_process must return a dict.")

        dim_l = int(res["l"])
        dim_r = int(res["r"])

        c = res["C"].to(device=w.device, dtype=torch.float32).contiguous()
        b = res["B"].to(device=w.device, dtype=torch.float32).contiguous()
        w_rot = res["W_rot"].to(device=w.device, dtype=torch.float32).reshape(ori_shape)

        self.weight.data = w_rot.to(self.weight)
        self.L = c.to(ori_dtype)
        self.R = b.to(ori_dtype)
        self.dim_l = dim_l
        self.dim_r = dim_r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        init_shape = x.shape
        if getattr(self, "L", None) is not None:
            device = x.device
            L = self.L if self.L.device == device else self.L.to(device)
            R = self.R if self.R.device == device else self.R.to(device)
            x = x.reshape(-1, int(self.dim_l), int(self.dim_r))
            x = L @ x @ R
            x = x.reshape(init_shape)
        return super().forward(x)


def export_rotated_checkpoint(model: nn.Module, save_path: str, meta: Dict[str, Any]) -> None:
    pkg: Dict[str, Any] = {"_meta": dict(meta), "layers": {}}
    for name, layer in model.model.named_modules():
        if not isinstance(layer, nn.Linear):
            continue
        d: Dict[str, Any] = {
            "weight": layer.weight.detach().cpu().half().contiguous(),
            "bias": None if layer.bias is None else layer.bias.detach().cpu().half().contiguous(),
            "L": None,
            "R": None,
            "dim_l": int(getattr(layer, "dim_l", 0) or 0),
            "dim_r": int(getattr(layer, "dim_r", 0) or 0),
        }
        for k in ("L", "R"):
            t = getattr(layer, k, None)
            d[k] = None if (t is None or not isinstance(t, torch.Tensor)) else t.detach().cpu().half().contiguous()
        pkg["layers"][name] = d
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(pkg, save_path)


@dataclass(frozen=True)
class LayerTask:
    name: str
    in_features: int
    out_features: int
    weight_path: str
    bias_path: str
    out_path: str


def _worker(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    stop_event: mp.Event,
    gpu_id: int,
    args: argparse.Namespace,
) -> None:
    device = torch.device(f"cuda:{int(gpu_id)}")
    torch.cuda.set_device(device)
    make_deterministic(int(args.seed) + int(gpu_id))
    QLinear.kp = build_kp(args)

    while not stop_event.is_set():
        try:
            task: Optional[Dict[str, Any]] = task_queue.get(timeout=1)
        except queue.Empty:
            continue
        if task is None:
            break

        try:
            name = str(task["name"])
            in_features = int(task["in_features"])
            out_features = int(task["out_features"])
            w_cpu = task["weight"]
            b_cpu = task.get("bias", None)
            out_path = str(task["out_path"])

            if bool(args.resume) and os.path.isfile(out_path):
                result_queue.put(("ok", name, out_path, "hit"))
                continue

            w = w_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            b = None if b_cpu is None else b_cpu.to(device=device, dtype=torch.float32, non_blocking=True)

            layer = nn.Linear(in_features, out_features, bias=(b is not None)).to(device)
            layer.weight.data = w
            if b is not None:
                layer.bias.data = b

            layer.__class__ = QLinear
            layer.setup_rotation()

            out = {
                "name": name,
                "in_features": in_features,
                "out_features": out_features,
                "weight": layer.weight.detach().cpu().half().contiguous(),
                "bias": None if layer.bias is None else layer.bias.detach().cpu().half().contiguous(),
                "L": None if getattr(layer, "L", None) is None else layer.L.detach().cpu().half().contiguous(),
                "R": None if getattr(layer, "R", None) is None else layer.R.detach().cpu().half().contiguous(),
                "dim_l": int(getattr(layer, "dim_l", 0) or 0),
                "dim_r": int(getattr(layer, "dim_r", 0) or 0),
            }
            atomic_torch_save(out, out_path)

            del layer, w, b
            torch.cuda.empty_cache()

            result_queue.put(("ok", name, out_path, "new"))
        except Exception as e:
            result_queue.put(("fail", str(task.get("name", "unknown")), repr(e)))

    result_queue.put(("done", int(gpu_id)))


def load_layer_result(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--export_rotated", type=str, required=True)
    p.add_argument("--ngpus", type=int, default=None)

    p.add_argument("--use_gmm", action="store_true", default=False)
    p.add_argument("--gmm_iters", type=int, default=100)
    p.add_argument("--gmm_lr_r", type=float, default=1e-2)
    p.add_argument("--gmm_lr_l", type=float, default=1e-2)
    p.add_argument("--gmm_ortho_method", type=str, default="cayley", choices=["cayley", "exp"])
    p.add_argument("--sigma0_ratio", type=float, default=0.25)
    p.add_argument("--sigma1_ratio", type=float, default=0.25)
    p.add_argument("--balance_w", type=float, default=0.1)

    p.add_argument("--tmp_dir", type=str, default=None)
    p.add_argument("--queue_timeout", type=float, default=10.0)

    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--clean", action="store_true", default=False)

    args = p.parse_args()
    make_deterministic(int(args.seed))
    mp.set_start_method("spawn", force=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cpu", torch_dtype="auto")
    model.eval()

    linear_layers: List[Tuple[str, nn.Module]] = [
        (name, mod) for name, mod in model.model.named_modules() if isinstance(mod, nn.Linear)
    ]
    if not linear_layers:
        raise RuntimeError("No nn.Linear found under model.model.")

    n_gpus = int(args.ngpus) if args.ngpus is not None else torch.cuda.device_count()
    if n_gpus <= 0:
        raise RuntimeError("No CUDA device found.")

    export_dir = os.path.dirname(args.export_rotated) or "."
    tmp_root = args.tmp_dir or os.path.join(export_dir, "tmp_rotate_ipc")
    task_dir = os.path.join(tmp_root, "tasks")
    result_dir = os.path.join(tmp_root, "results")
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    if bool(args.clean):
        for d in (task_dir, result_dir):
            for fn in os.listdir(d):
                if fn.endswith(".pt") or fn.endswith(".pt.tmp"):
                    try:
                        os.remove(os.path.join(d, fn))
                    except OSError:
                        pass

    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()
    stop_event: mp.Event = ctx.Event()

    workers: List[mp.Process] = []
    for gid in range(n_gpus):
        proc = ctx.Process(target=_worker, args=(task_queue, result_queue, stop_event, gid, args))
        proc.start()
        workers.append(proc)

    tasks_to_run = 0
    for name, layer in linear_layers:
        fn = f"{safe_filename(str(name))}.pt"
        out_path = os.path.join(result_dir, fn)
        if bool(args.resume) and os.path.isfile(out_path):
            continue

        task_path = os.path.join(task_dir, fn)
        payload = {
            "name": str(name),
            "in_features": int(layer.in_features),
            "out_features": int(layer.out_features),
            "weight": layer.weight.detach().cpu(),
            "bias": None if layer.bias is None else layer.bias.detach().cpu(),
            "out_path": out_path,
        }
        atomic_torch_save(payload, task_path)
        task_queue.put(payload)
        tasks_to_run += 1

    for _ in workers:
        task_queue.put(None)

    failed: List[Tuple[str, str]] = []
    pbar = tqdm(total=int(tasks_to_run), desc="KOTMS", dynamic_ncols=True)

    done_workers = 0
    timeout = float(args.queue_timeout)
    while done_workers < len(workers):
        try:
            msg = result_queue.get(timeout=timeout)
        except queue.Empty:
            dead = [i for i, p_ in enumerate(workers) if not p_.is_alive()]
            if dead:
                stop_event.set()
                for p_ in workers:
                    try:
                        p_.join(timeout=2)
                    except Exception:
                        pass
                raise RuntimeError(f"Worker(s) died unexpectedly: {dead}")
            continue

        kind = msg[0]
        if kind == "ok":
            _, _name, _out_path, tag = msg
            if tag == "new":
                pbar.update(1)
        elif kind == "fail":
            _, task_name, err = msg
            failed.append((task_name, err))
            pbar.update(1)
        elif kind == "done":
            done_workers += 1

    pbar.close()

    stop_event.set()
    for proc in workers:
        proc.join()

    task_queue.close()
    task_queue.join_thread()
    result_queue.close()
    result_queue.join_thread()

    if failed:
        raise RuntimeError(f"KOTMS failed on {len(failed)} layers. Example: {failed[0]}")

    rotated: Dict[str, Dict[str, Any]] = {}
    for name, _layer in linear_layers:
        fn = f"{safe_filename(str(name))}.pt"
        out_path = os.path.join(result_dir, fn)
        if not os.path.isfile(out_path):
            raise RuntimeError(f"Missing rotated layer result: {out_path} (layer={name}).")
        rotated[str(name)] = load_layer_result(out_path)

    for name, layer in model.model.named_modules():
        if not isinstance(layer, nn.Linear):
            continue
        if "lm_head" in name:
            continue
        if name not in rotated:
            continue

        d = rotated[name]
        layer.weight.data = d["weight"].to(dtype=layer.weight.dtype)
        if d["bias"] is not None:
            if layer.bias is None:
                layer.bias = nn.Parameter(d["bias"].to(dtype=layer.weight.dtype))
            else:
                layer.bias.data = d["bias"].to(dtype=layer.weight.dtype)

        layer.__class__ = QLinear
        if d.get("L", None) is not None:
            layer.register_buffer("L", d["L"].to(dtype=torch.float16), persistent=True)
            layer.register_buffer("R", d["R"].to(dtype=torch.float16), persistent=True)
            layer.dim_l = int(d["dim_l"])
            layer.dim_r = int(d["dim_r"])

    meta: Dict[str, Any] = {
        "model": str(args.model),
        "seed": int(args.seed),
        "stage": "KOTMS",
        "use_gmm": bool(args.use_gmm),
        "gmm_iters": int(args.gmm_iters),
        "gmm_lr_r": float(args.gmm_lr_r),
        "gmm_lr_l": float(args.gmm_lr_l),
        "gmm_ortho_method": str(args.gmm_ortho_method),
        "sigma0_ratio": float(args.sigma0_ratio),
        "sigma1_ratio": float(args.sigma1_ratio),
        "balance_w": float(args.balance_w),
        "tmp_root": str(tmp_root),
        "resume": bool(args.resume),
        "clean": bool(args.clean),
    }

    os.makedirs(export_dir, exist_ok=True)
    export_rotated_checkpoint(model, args.export_rotated, meta)
    print(f"Saved rotated checkpoint: {args.export_rotated}")


if __name__ == "__main__":
    main()
