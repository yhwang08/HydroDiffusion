"""S4D with optimized FFT inference via kernel + FFT caching.

Key optimizations:
- During inference, the SSM kernel K and its FFT are computed ONCE and cached.
- Subsequent calls with same L reuse cached k_f, reducing per-step cost to:
    rfft(u) + pointwise multiply + irfft  (no kernel recomputation)
- Recurrent mode available for very short sequences (L < 32).
- Scan mode removed: for large H*N it is strictly worse than FFT in both
  memory and speed (requires O(B*L*H*N) vs O(B*L*H) for FFT).
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.models.nn import DropoutNd


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _next_fast_len(n):
    """
    Smallest integer >= n whose only prime factors are {2, 3, 5}.
    cuFFT (and scipy/numpy FFT) run fastest on these 'smooth' sizes.
    E.g. n=744 (=2^3*3*31, has prime 31) -> 750 (=2*3*5^3, 0.8% overhead)
         n=730 (=2*5*73,  has prime 73) -> 750
    Much better than rounding up to the next power of 2 (=1024, 37% overhead).
    """
    m = n
    while True:
        r = m
        for p in (2, 3, 5):
            while r % p == 0:
                r //= p
        if r == 1:
            return m
        m += 1


class S4DKernel(nn.Module):
    """S4D kernel with aggressive inference caching."""

    def __init__(self, d_model, cfr, cfi, N=64, dt_min=0.0001, dt_max=0.1,
                 lr=None, lr_dt=None, wd=None):
        super().__init__()

        H = d_model
        self.H = H
        self.N = N // 2

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

        # Inference cache — stores both kernel K and its FFT k_f
        self._cache = {}  # key: L -> (K, k_f)

    # Chunk size for kernel computation. Tunes the peak memory of the
    # (H, N, chunk) intermediate: 256*64*64*8 bytes = 8 MB vs 48 MB full.
    _KERNEL_CHUNK = 64

    def _compute_kernel(self, L):
        """
        Compute the SSM convolution kernel of length L.

        The naive approach allocates (H, N, L) complex64 — for H=256, N=64,
        L=372 that is ~48 MB per layer per training step, saturating GPU
        memory bandwidth. Chunking along L keeps peak allocation at
        H*N*chunk*8 bytes = ~8 MB regardless of L.
        """
        dt = torch.exp(self.log_dt)                          # (H,)
        C  = torch.view_as_complex(self.C)                   # (H, N)
        A  = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H, N)

        dtA      = A * dt.unsqueeze(-1)                      # (H, N)
        C_scaled = C * (torch.exp(dtA) - 1.0) / A           # (H, N)

        chunk = self._KERNEL_CHUNK
        parts = []
        for start in range(0, L, chunk):
            l_chunk = torch.arange(start, min(start + chunk, L), device=A.device)
            # (H, N, chunk) — small intermediate, evicted from cache each chunk
            parts.append(
                2.0 * torch.einsum(
                    'hn,hnc->hc',
                    C_scaled,
                    torch.exp(dtA.unsqueeze(-1) * l_chunk),
                ).real
            )
        return torch.cat(parts, dim=-1)                      # (H, L)

    def forward_fft(self, L):
        """
        Return (K, k_f, fft_n) where:
          K     : (H, L)            real kernel
          k_f   : (H, fft_n//2+1)  rfft of K padded to fft_n (cached at inference)
          fft_n : int               next smooth number >= 2*L (fast cuFFT plan)
        """
        if not self.training and L in self._cache:
            return self._cache[L]

        # Smallest smooth number >= 2*L (factors only in {2,3,5}).
        # e.g. L=372 → 2L=744 (has prime 31) → fft_n=750 (0.8% overhead)
        #      L=365 → 2L=730 (has prime 73) → fft_n=750 (2.7% overhead)
        # Power-of-2 (=1024) would add 37% overhead — wrong for GPU FFT.
        fft_n = _next_fast_len(2 * L)

        K   = self._compute_kernel(L)                        # (H, L)
        k_f = torch.fft.rfft(K, n=fft_n)                    # (H, fft_n//2+1)

        if not self.training:
            self._cache[L] = (K, k_f, fft_n)

        return K, k_f, fft_n

    def get_discretized_params(self):
        """Discretized A, B, C for recurrent mode."""
        dt = torch.exp(self.log_dt)
        C  = torch.view_as_complex(self.C)
        A  = -torch.exp(self.log_A_real) + 1j * self.A_imag

        dtA        = A * dt.unsqueeze(-1)
        A_discrete = torch.exp(dtA)
        B_discrete = (A_discrete - 1.0) / A
        return A_discrete, B_discrete, C

    def invalidate_cache(self):
        """Call after an optimizer step to force kernel recomputation."""
        self._cache.clear()

    def register(self, name, tensor, wd, lr=None):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": wd}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4DOptimized(nn.Module):
    """
    S4D with two execution modes:
      'fft'       : FFT convolution — fast for all sequence lengths,
                    kernel+FFT cached at inference so cost per DDIM step
                    is just rfft(u) + multiply + irfft.
      'recurrent' : Sequential recurrence — only for L < 32.
      'auto'      : Picks recurrent for L<32, fft otherwise.
    """

    def __init__(self, d_model, d_state=64, dropout=0.0,
                 add_noise=0, mult_noise=0,
                 cfr=1, cfi=1, transposed=True, mode='auto',
                 **kernel_args):
        super().__init__()

        self.h          = d_model
        self.n          = d_state
        self.d_output   = self.h
        self.transposed = transposed
        self.mode       = mode

        self.D = nn.Parameter(torch.randn(self.h))

        self.kernel = S4DKernel(self.h, cfr, cfi, N=self.n, **kernel_args)

        self.activation    = nn.GELU()
        dropout_fn         = DropoutNd
        self.dropout       = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    # ------------------------------------------------------------------
    # Mode selection
    # ------------------------------------------------------------------
    def _select_mode(self, L):
        if self.mode != 'auto':
            return self.mode
        if not self.training and L < 32:
            return 'recurrent'
        return 'fft'

    # ------------------------------------------------------------------
    # FFT mode  (primary, optimized)
    # ------------------------------------------------------------------
    def forward_fft(self, u, L):
        u_f32 = u.float()
        # k_f is cached at inference; fft_n is the power-of-2 pad size.
        # Do NOT discard k_f — recomputing rfft(K) every call was the bottleneck.
        K, k_f, fft_n = self.kernel.forward_fft(L)

        try:
            from src.ops.fftconv import fftconv_func
            y = fftconv_func(u_f32, K, self.D)  # fused single kernel
        except ImportError:
            # Fallback: use cached k_f directly (no rfft(K) recomputation)
            u_f = torch.fft.rfft(u_f32, n=fft_n)
            y   = torch.fft.irfft(u_f * k_f, n=fft_n)[..., :L]
            y   = y + u_f32 * self.D.float().unsqueeze(-1)

        return y

    # ------------------------------------------------------------------
    # Recurrent mode  (only for very short L)
    # ------------------------------------------------------------------
    def forward_recurrent(self, u, L):
        """
        u : (B, H, L) float32
        Returns y : (B, H, L) float32
        """
        A_discrete, B_discrete, C = self.kernel.get_discretized_params()
        u_f32 = u.float()

        h = torch.zeros(u.shape[0], self.h, self.kernel.N,
                        device=u.device, dtype=torch.cfloat)
        outputs = []
        for t in range(L):
            u_t = u_f32[:, :, t]                            # (B, H)
            h   = (A_discrete.unsqueeze(0) * h
                   + B_discrete.unsqueeze(0) * u_t.unsqueeze(-1))
            y_t = (torch.einsum('bhn,hn->bh', h, C) * 2).real
            y_t = y_t + self.D.float() * u_t
            outputs.append(y_t)

        return torch.stack(outputs, dim=2)                   # (B, H, L)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, u, **kwargs):
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        input_dtype = u.dtype
        mode        = self._select_mode(L)

        with torch.amp.autocast("cuda", enabled=False):
            if mode == 'fft':
                y = self.forward_fft(u, L)
            else:
                y = self.forward_recurrent(u, L)

        y = y.to(input_dtype)
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)

        if not self.transposed:
            y = y.transpose(-1, -2)
        return y, None

    # ------------------------------------------------------------------
    # Profiled forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_profiled(self, u, **kwargs):
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
        L    = u.size(-1)
        mode = self._select_mode(L)
        toc("s4d_prep", t0)

        # store mode as metadata (not a float — will be handled by profiler.merge)
        times["__s4d_mode__"] = mode

        with torch.amp.autocast("cuda", enabled=False):
            if mode == 'fft':
                # Split kernel lookup vs convolution so we can see cache savings
                t0 = tic()
                _, k_f, fft_n = self.kernel.forward_fft(L)  # cached after first call
                toc("s4d_kernel_fft", t0)

                t0 = tic()
                u_f32 = u.float()
                u_f   = torch.fft.rfft(u_f32, n=fft_n)
                y     = torch.fft.irfft(u_f * k_f, n=fft_n)[..., :L]
                y     = y + u_f32 * self.D.float().unsqueeze(-1)
                toc("s4d_fft_conv", t0)

            else:  # recurrent
                t0 = tic()
                y = self.forward_recurrent(u, L)
                toc("s4d_recurrent", t0)

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

    # ------------------------------------------------------------------
    # Single recurrent step  (for past-state caching at inference)
    # ------------------------------------------------------------------
    def step(self, u_t, state, A_bar, B_bar, C_param):
        """
        Process ONE time step in recurrent mode.

        Parameters
        ----------
        u_t    : (B, H)      float input at current time step
        state  : (B, H, N)   complex hidden state from previous step
        A_bar, B_bar, C_param : precomputed from kernel.get_discretized_params()
                                 (call once per block, reuse for all steps)

        Returns
        -------
        y_t      : (B, H)      block output (after activation + output_linear)
        new_state: (B, H, N)   updated hidden state
        """
        # SSM recurrence: h_t = Ā·h_{t-1} + B̄·u_t
        new_state = (A_bar * state
                     + B_bar * u_t.to(torch.cfloat).unsqueeze(-1))   # (B, H, N)

        # Output: y = 2·Re(C·h) + D·u
        y_t = (2.0 * (C_param * new_state).real.sum(-1)
               + self.D * u_t.float())                                # (B, H)

        # Apply activation + dropout + output_linear per step
        # Conv1d(kernel_size=1) is identical to a linear applied per time step
        y_t = y_t.unsqueeze(-1)                       # (B, H, 1)
        y_t = self.dropout(self.activation(y_t))
        y_t = self.output_linear(y_t).squeeze(-1)     # (B, H)
        return y_t.to(u_t.dtype), new_state

    # ------------------------------------------------------------------
    # Utility: call after optimizer.step() during training
    # ------------------------------------------------------------------
    def invalidate_kernel_cache(self):
        self.kernel.invalidate_cache()


# Alias keeps your existing imports working
S4D = S4DOptimized