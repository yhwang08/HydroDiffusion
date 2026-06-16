"""Minimal version of S4D with optional profiling support."""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.models.nn import DropoutNd
import scipy.io as mlio


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, cfr, cfi, N=64, dt_min=0.0001, dt_max=0.1, lr=None, lr_dt=None, wd=None):
        super().__init__()

        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, 0, lr=lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2)) * cfr
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H) * cfi
        self.register("log_A_real", log_A_real, wd, lr)
        self.register("A_imag", A_imag, wd, lr)

    def forward(self, L):
        """
        returns: (H, L)
        """
        dt = torch.exp(self.log_dt)                           # (H)
        C = torch.view_as_complex(self.C)                     # (H, N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag   # (H, N)

        dtA = A * dt.unsqueeze(-1)                            # (H, N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H, N, L)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum('hn,hnl->hl', C, torch.exp(K)).real

        return K

    @torch.no_grad()
    def forward_profiled(self, L):
        times = {}

        def tic():
            _sync()
            return time.perf_counter()

        def toc(name, t0):
            _sync()
            times[name] = times.get(name, 0.0) + (time.perf_counter() - t0)

        t0 = tic()
        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        toc("s4kernel_materialize", t0)

        t0 = tic()
        dtA = A * dt.unsqueeze(-1)
        toc("s4kernel_dtA", t0)

        t0 = tic()
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)
        toc("s4kernel_vandermonde_grid", t0)

        t0 = tic()
        C = C * (torch.exp(dtA) - 1.0) / A
        toc("s4kernel_C_transform", t0)

        t0 = tic()
        K = 2 * torch.einsum('hn,hnl->hl', C, torch.exp(K)).real
        toc("s4kernel_einsum", t0)

        return K, times

    def register(self, name, tensor, wd, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": wd}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, add_noise=0, mult_noise=0,
                 cfr=1, cfi=1, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed
        self.add_noise = add_noise
        self.mult_noise = mult_noise

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, cfr, cfi, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):
        """Input/output shape: (B, H, L)"""
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)
    
        input_dtype = u.dtype
    
        # run kernel + FFT path in fp32 to avoid cuFFT half-precision restrictions
        with torch.amp.autocast("cuda", enabled=False):
            u32 = u.float()
            k = self.kernel(L=L).float()
    
            k_f = torch.fft.rfft(k, n=2 * L)
            u_f = torch.fft.rfft(u32, n=2 * L)
            ybar = u_f * k_f
            y = torch.fft.irfft(ybar, n=2 * L)[..., :L]
    
            y = y + u32 * self.D.float().unsqueeze(-1)
    
        y = y.to(input_dtype)
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
    
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y, None

    @torch.no_grad()
    def forward_profiled(self, u, **kwargs):
        """Profiled forward. Input/output shape: (B, H, L)"""
        times = {}
    
        def tic():
            _sync()
            return time.perf_counter()
    
        def toc(name, t0):
            _sync()
            times[name] = times.get(name, 0.0) + (time.perf_counter() - t0)
    
        input_dtype = u.dtype
    
        t0 = tic()
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)
        toc("s4d_prep", t0)
    
        # kernel in fp32
        t0 = tic()
        with torch.amp.autocast("cuda", enabled=False):
            k, kernel_times = self.kernel.forward_profiled(L=L)
            k = k.float()
        toc("s4d_kernel_total", t0)
    
        for kname, kval in kernel_times.items():
            times[kname] = times.get(kname, 0.0) + kval
    
        # FFT path in fp32
        t0 = tic()
        with torch.amp.autocast("cuda", enabled=False):
            u32 = u.float()
            k32 = k.float()
        toc("s4d_cast_fp32", t0)
    
        t0 = tic()
        k_f = torch.fft.rfft(k32, n=2 * L)
        toc("s4d_fft_kernel", t0)
    
        t0 = tic()
        u_f = torch.fft.rfft(u32, n=2 * L)
        toc("s4d_fft_input", t0)
    
        t0 = tic()
        ybar = u_f * k_f
        toc("s4d_freq_multiply", t0)
    
        t0 = tic()
        y = torch.fft.irfft(ybar, n=2 * L)[..., :L]
        toc("s4d_ifft", t0)
    
        # D term in fp32
        t0 = tic()
        y = y + u32 * self.D.float().unsqueeze(-1)
        toc("s4d_skip_D", t0)
    
        # cast back before pointwise layers
        t0 = tic()
        y = y.to(input_dtype)
        toc("s4d_cast_back", t0)
    
        t0 = tic()
        y = self.activation(y)
        y = self.dropout(y)
        toc("s4d_activation_dropout", t0)
    
        t0 = tic()
        y = self.output_linear(y)
        toc("s4d_output_linear", t0)
    
        t0 = tic()
        if not self.transposed:
            y = y.transpose(-1, -2)
        toc("s4d_post", t0)
    
        return y, None, times