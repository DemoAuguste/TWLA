import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

index = 0


# -----------------------------
# Utilities: masked nanmean / absmean (row/col)
# -----------------------------
@torch.no_grad()
def _nan_where_like(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return x where mask==True, else NaN (device-safe)."""
    nan = torch.full_like(x, float("nan"))
    return torch.where(mask, x, nan)


@torch.no_grad()
def row_nanmean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """NaN-mean over dim=1 with fallback to 0 when a row is fully masked."""
    xm = _nan_where_like(x, mask)
    mu = torch.nanmean(xm, dim=1)
    mu = torch.where(torch.isnan(mu), torch.zeros_like(mu), mu)
    return mu  # [n]


@torch.no_grad()
def col_nanmean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """NaN-mean over dim=0 with fallback to 0 when a col is fully masked."""
    xm = _nan_where_like(x, mask)
    mu = torch.nanmean(xm, dim=0)
    mu = torch.where(torch.isnan(mu), torch.zeros_like(mu), mu)
    return mu  # [m]


@torch.no_grad()
def row_nanabsmean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """NaN-mean(abs(x)) over dim=1 with fallback to 0 when a row is fully masked."""
    xm = _nan_where_like(x, mask)
    v = torch.nanmean(torch.abs(xm), dim=1)
    v = torch.where(torch.isnan(v), torch.zeros_like(v), v)
    return v  # [n]


@torch.no_grad()
def col_nanabsmean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """NaN-mean(abs(x)) over dim=0 with fallback to 0 when a col is fully masked."""
    xm = _nan_where_like(x, mask)
    v = torch.nanmean(torch.abs(xm), dim=0)
    v = torch.where(torch.isnan(v), torch.zeros_like(v), v)
    return v  # [m]


# -----------------------------
# Ternary core: thresholding + alpha
# Δ_i = 0.75 * mean(|W_i - μ_i|)
# T_ij = 1 if > Δ_i; 0 if |.|<=Δ_i; -1 if < -Δ_i
# α_i = sum_j T_ij (W_ij - μ_i) / sum_j |T_ij|
# -----------------------------
@torch.no_grad()
def ternary_threshold(Wc: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Wc: [n, m] centered values (W - mu) with mask applied consistently outside if desired.
    delta: [n] per-row threshold.
    Returns: T in {-1, 0, 1}, masked to zero where mask==False.
    """
    d = delta[:, None]
    T = torch.zeros_like(Wc)
    T = torch.where(Wc > d, torch.ones_like(T), T)
    T = torch.where(Wc < -d, -torch.ones_like(T), T)
    return T * mask


@torch.no_grad()
def alpha_from_ternary(Wc: torch.Tensor, T: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute per-row alpha given centered Wc and ternary T."""
    num = torch.sum((T * Wc) * mask, dim=1)                      # [n]
    den = torch.sum(torch.abs(T) * mask, dim=1) + eps            # [n]
    return num / den


# -----------------------------
# Ternary residual fitting (residual decomposition)
# -----------------------------
@torch.no_grad()
def high_order_residual(x: torch.Tensor, mask: torch.Tensor, order: int = 2) -> torch.Tensor:
    """
    Greedy residual decomposition using ternary for each order:
      residual -> mu -> centered -> delta -> T -> alpha -> reconstruct
    """
    global index
    index += 1

    W = x.clone() * mask
    sum_order = torch.zeros_like(x)

    for _ in range(int(order)):
        residual = W - sum_order

        mu = row_nanmean(residual, mask)                         # [n]
        Wc = residual - mu[:, None] * mask
        delta = 0.75 * row_nanabsmean(Wc, mask)                  # [n]
        T = ternary_threshold(Wc, mask, delta)                   # [n, m]
        alpha = alpha_from_ternary(Wc, T, mask)                  # [n]

        recon = (alpha[:, None] * T + mu[:, None]) * mask
        sum_order = sum_order + recon

    return sum_order


# -----------------------------
# Alternating ternary fitting, order=1
# -----------------------------
@torch.no_grad()
def high_order_residual_alternating_order1(
    x: torch.Tensor,
    mask: torch.Tensor,
    order: int = 2,
    iter: int = 15,
) -> torch.Tensor:
    """
    Alternating refinement for order=1 ternary:
      mu <- mu + mean(R)
      delta <- 0.75 * mean(|W - mu|)
      T <- threshold by delta
      alpha <- sum(T*(W-mu))/sum(|T|)
    """
    global index
    index += 1

    W = x.clone() * mask

    mu = row_nanmean(W, mask)
    Wc = W - mu[:, None] * mask
    delta = 0.75 * row_nanabsmean(Wc, mask)
    T = ternary_threshold(Wc, mask, delta)
    alpha = alpha_from_ternary(Wc, T, mask)
    W_hat = (alpha[:, None] * T + mu[:, None]) * mask

    for _ in range(int(iter)):
        R = W - W_hat
        mu = mu + row_nanmean(R, mask)

        Wc = W - mu[:, None] * mask
        delta = 0.75 * row_nanabsmean(Wc, mask)
        T = ternary_threshold(Wc, mask, delta)
        alpha = alpha_from_ternary(Wc, T, mask)
        W_hat = (alpha[:, None] * T + mu[:, None]) * mask

    return W_hat


@torch.no_grad()
def high_order_residual_alternating_order1_x(
    x: torch.Tensor,
    mask: torch.Tensor,
    order: int = 2,
    S: Optional[torch.Tensor] = None,
    iter: int = 15,
    iter2: int = 15,
) -> torch.Tensor:
    """
    Two-stage refinement for order=1:
      1) Weight-domain alternating to obtain (mu, alpha, T)
      2) Freeze T and refine (mu, alpha) with S (X-domain objective)
    """
    global index
    index += 1

    W = x.clone() * mask

    # Stage 1: weight-domain alternating
    mu = row_nanmean(W, mask)
    Wc = W - mu[:, None] * mask
    delta = 0.75 * row_nanabsmean(Wc, mask)
    T = ternary_threshold(Wc, mask, delta)
    alpha = alpha_from_ternary(Wc, T, mask)
    W_hat = (alpha[:, None] * T + mu[:, None]) * mask

    for _ in range(int(iter)):
        R = W - W_hat
        mu = mu + row_nanmean(R, mask)

        Wc = W - mu[:, None] * mask
        delta = 0.75 * row_nanabsmean(Wc, mask)
        T = ternary_threshold(Wc, mask, delta)
        alpha = alpha_from_ternary(Wc, T, mask)
        W_hat = (alpha[:, None] * T + mu[:, None]) * mask

    # Stage 2: X-domain refinement (freeze T)
    if S is not None:
        MM = mask[:, :, None] * mask[:, None, :]
        mu_den = torch.sum(S * MM, dim=(1, 2), dtype=torch.float32) + 1e-10

        Tm = T * mask
        alpha_den = torch.sum(S * Tm[:, :, None] * Tm[:, None, :], dim=(1, 2), dtype=torch.float32) + 1e-10

        for _ in range(int(iter2)):
            mu = torch.sum(
                S * (W - alpha[:, None] * T * mask)[:, :, None] * MM,
                dim=(1, 2),
                dtype=torch.float32,
            ) / mu_den

            W_mu = W - mu[:, None] * mask
            alpha = torch.sum(
                S * Tm[:, :, None] * W_mu[:, None, :],
                dim=(1, 2),
                dtype=torch.float32,
            ) / alpha_den

    W_hat = (alpha[:, None] * T + mu[:, None]) * mask
    return W_hat


# -----------------------------
# Alternating ternary fitting, order=2 (two ternary bases -> 9 combos)
# Model: W ≈ μ + α0*T0 + α1*T1,  T0,T1 in {-1,0,1}
# Values set per-entry: α0*s0 + α1*s1, (s0,s1) ∈ {-1,0,1}^2  => 9 candidates
# -----------------------------
@torch.no_grad()
def high_order_residual_alternating_mean(
    x: torch.Tensor,
    mask: torch.Tensor,
    order: int = 2,
    num_iters: int = 15,
) -> torch.Tensor:
    """
    Order-2 ternary alternating:
      - update mu via residual mean correction
      - update alpha0/alpha1 by coordinate LS
      - update (T0, T1) by nearest among 9 candidates
    """
    assert order == 2, "This ternary order-2 function assumes order=2."

    global index
    index += 1

    W = x.clone() * mask

    # Init: first ternary on W
    mu = row_nanmean(W, mask)
    Wc = W - mu[:, None] * mask

    delta0 = 0.75 * row_nanabsmean(Wc, mask)
    T0 = ternary_threshold(Wc, mask, delta0)
    alpha0 = alpha_from_ternary(Wc, T0, mask)

    R1 = Wc - alpha0[:, None] * T0 * mask
    delta1 = 0.75 * row_nanabsmean(R1, mask)
    T1 = ternary_threshold(R1, mask, delta1)
    alpha1 = alpha_from_ternary(R1, T1, mask)

    W_hat = (mu[:, None] + alpha0[:, None] * T0 + alpha1[:, None] * T1) * mask

    for _ in range(int(num_iters)):
        R = W - W_hat
        mu = mu + row_nanmean(R, mask)
        Wc = W - mu[:, None] * mask

        den0 = torch.sum((T0 * T0) * mask, dim=1) + 1e-8
        num0 = torch.sum(T0 * (Wc - alpha1[:, None] * T1 * mask) * mask, dim=1)
        alpha0 = num0 / den0

        den1 = torch.sum((T1 * T1) * mask, dim=1) + 1e-8
        num1 = torch.sum(T1 * (Wc - alpha0[:, None] * T0 * mask) * mask, dim=1)
        alpha1 = num1 / den1

        target = Wc
        a0 = alpha0[:, None]
        a1 = alpha1[:, None]

        v0 = -a0 - a1
        v1 = -a0
        v2 = -a0 + a1
        v3 = -a1
        v4 = torch.zeros_like(a0)
        v5 = a1
        v6 = a0 - a1
        v7 = a0
        v8 = a0 + a1
        V = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7, v8], dim=2)

        idx_min = torch.argmin(torch.abs(target.unsqueeze(-1) - V), dim=-1)

        T0 = torch.zeros_like(idx_min, dtype=W.dtype)
        T1 = torch.zeros_like(idx_min, dtype=W.dtype)

        T0[(idx_min == 0) | (idx_min == 1) | (idx_min == 2)] = -1.0
        T0[(idx_min == 6) | (idx_min == 7) | (idx_min == 8)] = 1.0

        T1[(idx_min == 0) | (idx_min == 3) | (idx_min == 6)] = -1.0
        T1[(idx_min == 2) | (idx_min == 5) | (idx_min == 8)] = 1.0

        T0 = T0 * mask
        T1 = T1 * mask

        W_hat = (mu[:, None] + alpha0[:, None] * T0 + alpha1[:, None] * T1) * mask

    return W_hat


@torch.no_grad()
def high_order_residual_alternating_mean_x(
    x: torch.Tensor,
    mask: torch.Tensor,
    order: int = 2,
    S: Optional[torch.Tensor] = None,
    num_iters: int = 15,
    iter2: int = 15,
) -> torch.Tensor:
    """Order-2 ternary with an additional X-domain refinement stage when S is provided."""
    assert order == 2, "This ternary x function assumes order=2."

    global index
    index += 1

    W = x.clone() * mask

    # Weight-domain stage
    mu = row_nanmean(W, mask)
    Wc = W - mu[:, None] * mask

    delta0 = 0.75 * row_nanabsmean(Wc, mask)
    T0 = ternary_threshold(Wc, mask, delta0)
    alpha0 = alpha_from_ternary(Wc, T0, mask)

    R1 = Wc - alpha0[:, None] * T0 * mask
    delta1 = 0.75 * row_nanabsmean(R1, mask)
    T1 = ternary_threshold(R1, mask, delta1)
    alpha1 = alpha_from_ternary(R1, T1, mask)

    W_hat = (mu[:, None] + alpha0[:, None] * T0 + alpha1[:, None] * T1) * mask

    for _ in range(int(num_iters)):
        R = W - W_hat
        mu = mu + row_nanmean(R, mask)
        Wc = W - mu[:, None] * mask

        den0 = torch.sum((T0 * T0) * mask, dim=1) + 1e-8
        num0 = torch.sum(T0 * (Wc - alpha1[:, None] * T1 * mask) * mask, dim=1)
        alpha0 = num0 / den0

        den1 = torch.sum((T1 * T1) * mask, dim=1) + 1e-8
        num1 = torch.sum(T1 * (Wc - alpha0[:, None] * T0 * mask) * mask, dim=1)
        alpha1 = num1 / den1

        target = Wc
        a0 = alpha0[:, None]
        a1 = alpha1[:, None]

        v0 = -a0 - a1
        v1 = -a0
        v2 = -a0 + a1
        v3 = -a1
        v4 = torch.zeros_like(a0)
        v5 = a1
        v6 = a0 - a1
        v7 = a0
        v8 = a0 + a1
        V = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7, v8], dim=2)

        idx_min = torch.argmin(torch.abs(target.unsqueeze(-1) - V), dim=-1)

        T0 = torch.zeros_like(idx_min, dtype=W.dtype)
        T1 = torch.zeros_like(idx_min, dtype=W.dtype)

        T0[(idx_min == 0) | (idx_min == 1) | (idx_min == 2)] = -1.0
        T0[(idx_min == 6) | (idx_min == 7) | (idx_min == 8)] = 1.0

        T1[(idx_min == 0) | (idx_min == 3) | (idx_min == 6)] = -1.0
        T1[(idx_min == 2) | (idx_min == 5) | (idx_min == 8)] = 1.0

        T0 = T0 * mask
        T1 = T1 * mask

        W_hat = (mu[:, None] + alpha0[:, None] * T0 + alpha1[:, None] * T1) * mask

    # X-domain stage (freeze T0/T1)
    if S is not None:
        MM = mask[:, :, None] * mask[:, None, :]
        mu_den = torch.sum(S * MM, dim=(1, 2), dtype=torch.float32) + 1e-10

        T0m = T0 * mask
        T1m = T1 * mask
        den0 = torch.sum(S * T0m[:, :, None] * T0m[:, None, :], dim=(1, 2), dtype=torch.float32) + 1e-10
        den1 = torch.sum(S * T1m[:, :, None] * T1m[:, None, :], dim=(1, 2), dtype=torch.float32) + 1e-10

        for _ in range(int(iter2)):
            mu = torch.sum(
                S * (W - (alpha0[:, None] * T0 + alpha1[:, None] * T1) * mask)[:, :, None] * MM,
                dim=(1, 2),
                dtype=torch.float32,
            ) / mu_den

            W_mu = W - mu[:, None] * mask

            alpha0 = torch.sum(
                S * T0m[:, :, None] * (W_mu[:, None, :] - (alpha1[:, None] * T1m)[:, None, :]),
                dim=(1, 2),
                dtype=torch.float32,
            ) / den0

            alpha1 = torch.sum(
                S * T1m[:, :, None] * (W_mu[:, None, :] - (alpha0[:, None] * T0m)[:, None, :]),
                dim=(1, 2),
                dtype=torch.float32,
            ) / den1

    W_hat = (mu[:, None] + alpha0[:, None] * T0 + alpha1[:, None] * T1) * mask
    return W_hat


class Ternarization(nn.Module):
    """
    Keep class name for compatibility with your codebase.
    """
    def __init__(self, weight, groupsize=-1):
        super().__init__()
        oc, ic = weight.shape
        if groupsize == -1:
            groupsize = ic
        self.groupsize = groupsize
        self.n_groups = math.ceil(ic / groupsize)
        self.mean = 0

    def quantize(self, w, mask, order=2, groupi=0, S=None):
        if order == 2:
            w = high_order_residual_alternating_mean_x(w, mask, order=order, S=S)
        else:
            w = high_order_residual_alternating_order1_x(w, mask, order=order, S=S)
        return w
