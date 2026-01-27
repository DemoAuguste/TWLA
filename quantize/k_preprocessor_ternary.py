import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch


# ------------------------- utils -------------------------
def _pick_lr_factors_budget(oc: int, ic: int) -> Tuple[int, int]:
    """
    Pick (l, r) using the user's budget rule:
      1) r0 = floor( sqrt( (oc*ic)/4/16 ) ) = floor( sqrt( oc*ic / 64 ) )
      2) search r from r0 down to 1, find the first r s.t. ic % r == 0
      3) set l = ic // r
    This ensures r is as large as possible under the derived r0 bound.
    """
    assert oc > 0 and ic > 0
    r0 = int(math.isqrt(max(1, (oc * ic) // 64)))  # floor(sqrt(.))
    r0 = max(1, min(r0, ic))

    for r in range(r0, 0, -1):
        if ic % r == 0:
            l = ic // r
            return l, r

    return ic, 1  # fallback



# -------------------- two-sided orthogonal rotation --------------------
class TwoSidedOrthoSimple(torch.nn.Module):
    """
    Trainable two-sided orthogonal rotation:
      W2d[oc, ic] -> reshape [oc, l, r]
      right rotate by B[r,r], then left rotate by C[l,l]
      output back to [oc, ic]
    """
    def __init__(self, l: int, r: int, device, dtype, ortho_method: str = "cayley"):
        super().__init__()
        self.l, self.r = l, r
        self.ortho_method = ortho_method

        # free parameters -> skew -> Cayley/Exp -> orthogonal
        self.S_r = torch.nn.Parameter(torch.zeros(r, r, device=device, dtype=dtype))
        self.S_l = torch.nn.Parameter(torch.zeros(l, l, device=device, dtype=dtype))

    def _make_ortho(self, S: torch.Tensor) -> torch.Tensor:
        A = S - S.transpose(-1, -2)  # skew
        if self.ortho_method == "cayley":
            I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            O = torch.linalg.solve(I + A, I - A)
        elif self.ortho_method == "exp":
            O = torch.linalg.matrix_exp(A)
        else:
            raise ValueError(f"Unknown ortho_method={self.ortho_method}")
        return O

    def forward(self, W2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        oc, ic = W2d.shape
        assert ic == self.l * self.r, f"ic({ic}) must equal l*r({self.l*self.r})."

        B = self._make_ortho(self.S_r)  # [r,r]
        C = self._make_ortho(self.S_l)  # [l,l]

        W3 = W2d.view(oc, self.l, self.r)              # [oc,l,r]
        Xr = W3 @ B                                     # [oc,l,r]

        Xr_perm = Xr.permute(1, 0, 2).contiguous()      # [l,oc,r]
        Xl = (C @ Xr_perm.view(self.l, oc * self.r)).view(self.l, oc, self.r)
        Xl = Xl.permute(1, 0, 2).contiguous()           # [oc,l,r]

        X2d = Xl.view(oc, ic)
        return X2d, B, C


def train_trigauss_two_sided(
    W2d: torch.Tensor,
    iters: int = 600,
    lr_r: float = 2e-3,
    lr_l: float = 2e-3,
    ortho_method: str = "cayley",
    sigma0_ratio: float = 0.20,
    sigma1_ratio: float = 0.35,
    balance_w: float = 0.05,
    l: int = None,
    r: int = None,
) -> Dict[str, Any]:
    """
    Train B,C to maximize tri-GMM likelihood.

    If center_per_row=True:
      train on Xc = X2d - mean_row(X2d)  (aligns with per-row mean removal ternary)

    mixture: N(-c, sigma1^2), N(0, sigma0^2), N(+c, sigma1^2), equal weights 1/3.

    Returns dict:
      {
        "B": B, "C": C, "l": l, "r": r,
        "W_rot": X2d_after,
        "W_rot_c": Xc_after,
      }
    """
    device, dtype = W2d.device, W2d.dtype
    oc, ic = W2d.shape

    if l is None or r is None:
        l_auto, r_auto = _pick_lr_factors_budget(oc, ic)
        l = l_auto if l is None else l
        r = r_auto if r is None else r


    rot = TwoSidedOrthoSimple(l=l, r=r, device=device, dtype=dtype, ortho_method=ortho_method)
    opt = torch.optim.Adam([
        {"params": [rot.S_r], "lr": lr_r},
        {"params": [rot.S_l], "lr": lr_l},
    ])

    eps = 1e-10
    log2pi = math.log(2.0 * math.pi)
    target_resp = torch.tensor([1/3, 1/3, 1/3], device=device, dtype=dtype)

    for t in range(iters):
        opt.zero_grad(set_to_none=True)

        X2d, B, C = rot(W2d)
        Xc = X2d - X2d.mean(dim=1, keepdim=True)
        x = Xc

        # per-row scale proxy (detach for stability)
        with torch.no_grad():
            c = x.abs().mean(dim=1, keepdim=True).clamp_min(eps)   # [oc,1]
            sigma0 = (sigma0_ratio * c).clamp_min(eps)             # [oc,1]
            sigma1 = (sigma1_ratio * c).clamp_min(eps)             # [oc,1]

        c_full = c.expand_as(x)
        s0 = sigma0.expand_as(x)
        s1 = sigma1.expand_as(x)

        logp_neg  = -0.5 * (log2pi + 2.0 * torch.log(s1) + (x + c_full).pow(2) / s1.pow(2))
        logp_zero = -0.5 * (log2pi + 2.0 * torch.log(s0) + (x - 0.0).pow(2) / s0.pow(2))
        logp_pos  = -0.5 * (log2pi + 2.0 * torch.log(s1) + (x - c_full).pow(2) / s1.pow(2))

        logps = torch.stack([logp_neg, logp_zero, logp_pos], dim=-1)  # [oc,ic,3]
        log_mix = torch.logsumexp(logps, dim=-1) - math.log(3.0)
        nll = -(log_mix.mean())

        resp = torch.softmax(logps, dim=-1)
        resp_mean = resp.mean(dim=(0, 1))
        bal = (resp_mean - target_resp).pow(2).sum()

        loss = nll + balance_w * bal
        loss.backward()
        opt.step()

        with torch.no_grad():
            I_r = torch.eye(r, device=device, dtype=dtype)
            I_l = torch.eye(l, device=device, dtype=dtype)
            ortho_r = torch.linalg.norm(B.transpose(0, 1) @ B - I_r).item()
            ortho_l = torch.linalg.norm(C.transpose(0, 1) @ C - I_l).item()

    with torch.no_grad():
        X2d, B, C = rot(W2d)

    return {
        "B": B.detach(),
        "C": C.detach(),
        "l": l,
        "r": r,
        "W_rot": X2d.detach(),       

    }


# ------------------------- preprocessor (tri-GMM version) -------------------------
@dataclass
class KroneckerSmoothConfig:
    use_gmm_training: bool = True
    gmm_iters: int = 600
    gmm_lr_r: float = 2e-3
    gmm_lr_l: float = 2e-3
    gmm_ortho_method: str = "cayley"  
    sigma0_ratio: float = 0.25
    sigma1_ratio: float = 0.25
    balance_w: float = 0.1
    l: Optional[int] = None
    r: Optional[int] = None


class KroneckerSmoothPreprocessor:
    def __init__(self, config: KroneckerSmoothConfig):
        self.config = KroneckerSmoothConfig(**config) if isinstance(config, dict) else config

    def _kronecker_process(self, weight: torch.Tensor) -> Dict[str, Any]:
        if not (self.config.use_gmm_training):
            raise RuntimeError("use_gmm_training=False but tri-GMM rotation was requested.")
        return self._kronecker_process_trigmm(weight)

    def _kronecker_process_trigmm(self, weight: torch.Tensor) -> Dict[str, Any]:
        ori_shape, ori_dtype = weight.shape, weight.dtype
        x = weight.squeeze()

        # flatten -> [oc, ic]
        if x.dim() > 2:
            oc = x.shape[0]
            ic = x.shape[1:].numel()
            W2d = x.reshape(oc, ic).to(ori_dtype)
        else:
            oc, ic = x.shape
            W2d = x.to(ori_dtype)

        out = train_trigauss_two_sided(
            W2d=W2d,
            iters=self.config.gmm_iters,
            lr_r=self.config.gmm_lr_r,
            lr_l=self.config.gmm_lr_l,
            ortho_method=self.config.gmm_ortho_method,
            sigma0_ratio=self.config.sigma0_ratio,
            sigma1_ratio=self.config.sigma1_ratio,
            balance_w=self.config.balance_w,
            l=self.config.l,
            r=self.config.r,
        )

        # reshape back
        W_rot = out["W_rot"].reshape(ori_shape).to(ori_dtype).contiguous()

        return {
            "B": out["B"].to(ori_dtype),
            "C": out["C"].to(ori_dtype),
            "l": out["l"],
            "r": out["r"],
            "W_rot": W_rot,       
        }
