import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np
import math
from quantize.const import CLIPMAX, CLIPMIN
import random


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        lwc=False,
        swc=None,
        lac=None,
        act_group_size=None,
        mse=None,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        init_value = 4.0
        if lwc:
            if group_size:
                dim1 = int(shape[0] * math.ceil(shape[1] / group_size))
                self.deficiency = shape[-1] % group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric  # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
        self.sigmoid = nn.Sigmoid()
        self.enable = True
        self.group_size = group_size
        self.is_weight = shape != None
        self.recorded_x_max = None
        self.let_s = None
        self.act_group_size = act_group_size
        self.lac = lac
        self.swc = swc
        self.init_singlequant_params = torch.tensor(1)
        self.block_size = 128
        # mse quantization
        self.mse = mse

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros(
                (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
            )
            x = torch.cat((x, pad_zeros), dim=1)

        if self.group_size:
            assert len(x.shape) == 2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        x_int = round_ste(x.float() / scale).half()  # avoid overflow

        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)

        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)

        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:, : -self.deficiency]
        return x_dequant


    def forward(self, x: torch.Tensor, return_no_quant=False):

        if self.dynamic_method == "per_token" :
            self.per_token_dynamic_calibration(x)
        else:
            raise NotImplementedError()
        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        return x_dequant

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1, self.group_size)
            else:
                pad_zeros = torch.zeros(
                    (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
                )
                x = torch.cat((x, pad_zeros), dim=1)
                x = x.reshape(-1, self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True).to(x.device)
        xmax = x.amax(reduce_shape, keepdim=True).to(x.device)
        if self.swc:  
            xmax = self.swc * xmax
            xmin = self.swc * xmin
        elif self.lwc:
            xmax = self.sigmoid(self.upbound_factor.to(x.device)) * xmax
            xmin = self.sigmoid(self.lowbound_factor.to(x.device)) * xmin
        elif self.lac:
            xmax = self.lac * xmax
            xmin = self.lac * xmin

        if self.mse is None:
            if self.symmetric:
                abs_max = torch.max(xmax.abs(), xmin.abs())
                scale = abs_max / (2 ** (self.n_bits - 1) - 1)
                self.scale = scale.clamp(min=CLIPMIN, max=CLIPMAX)
                zero_point = (2 ** (self.n_bits - 1) - 1) * torch.ones_like(self.scale)
            else:
                range = xmax - xmin
                scale = range / (2**self.n_bits - 1)
                self.scale = scale.clamp(min=CLIPMIN, max=CLIPMAX)
                zero_point = -(xmin) / (self.scale)
        else:
            zero_point = torch.zeros_like(xmax)
            self.scale = torch.ones_like(xmax)
            best_err = torch.full_like(xmax, float("inf"), dtype=torch.float32)
            for p in np.linspace(1, 0.4, 100):
                pxmax = xmax * p
                pxmin = xmin * p
                if self.symmetric:
                    abs_max = torch.max(pxmax.abs(), pxmin.abs())
                    s = abs_max / (2 ** (self.n_bits - 1) - 1)
                    s = s.clamp(min=CLIPMIN, max=CLIPMAX)
                    zp = (2 ** (self.n_bits - 1) - 1) * torch.ones_like(s)
                else:
                    range = pxmax - pxmin
                    s = range / (2**self.n_bits - 1)
                    s = s.clamp(min=CLIPMIN, max=CLIPMAX)
                    zp = (-(pxmin) / (s)).round()
                zp = zp.clamp(min=-CLIPMAX, max=CLIPMAX)
                err = (
                    (self.fake_quant(x, s, zp) - x)
                    .abs()
                    .pow(self.mse)
                    .mean(dim=-1, keepdim=True)
                )
                mask = err < best_err
                if mask.any():
                    best_err[mask] = err[mask].to(best_err.dtype)
                    self.scale[mask] = s[mask]
                    zero_point[mask] = zp[mask]
        self.round_zero_point = zero_point.clamp(min=-CLIPMAX, max=CLIPMAX).round()

