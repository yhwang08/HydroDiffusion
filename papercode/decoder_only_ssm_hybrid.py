"""
decoder_only_ssm_hybrid.py

Hybrid SSM inference: FFT for past encoding (parallel over L steps),
recurrent for future decoding (only H=8 steps, no FFT overhead).

KEY IDEA
--------
  decode_future_fft  (fast version): still pays fft_n=750 FFT cost per DDIM
                                      step per layer, even though H=8.
  decode_future_hybrid (this file) : future decode = 8 recurrent SSM steps
                                      per layer — same cost as LSTM decode.

encode_past_hybrid():
  1. FFT path  → inter-layer h propagation (parallel, same as encode_past_fft)
  2. Recurrent → terminal SSM hidden state at position L-2 (done ONCE, reused
                 across all num_steps × num_samples calls)

decode_future_hybrid():
  For each layer: resume SSM recurrence from cached state, run H steps.
  No FFT at all.

Everything else (forward, sample_ddim, encode_past_fft, decode_future_fft,
sample_ddim_fast_fft) is identical to decoder_only_ssm_fast.py — the hybrid
methods are purely additive.
"""

import torch
import torch.nn as nn
from models.s4.s4d_optimized import S4D as LTI
import pdb
from diffusion_utils import diffusion_params
import math
from runtime_profiler import RuntimeProfiler
import time
import torch.nn.functional as F

dropout_fn = nn.Dropout1d if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12) else nn.Dropout


class MPFourier(nn.Module):
    def __init__(self, num_channels, bandwidth=1.0):
        super().__init__()
        self.register_buffer('freqs', 2 * math.pi * torch.randn(num_channels))
        self.register_buffer('phases', 2 * math.pi * torch.rand(num_channels))

    def forward(self, t):
        t = t.to(torch.float32)
        x = t[:, None] * self.freqs[None, :] + self.phases[None, :]
        return (x.cos() * math.sqrt(2)).to(t.dtype)


class decoder_only_ssm(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        n_layers: int,
        cfg: dict,
        *,
        horizon: int = 8,
        time_emb_dim: int = 256,
        static_dim: int = 27,
        dropout: float = 0.15,
        time_full: bool = False,
    ):
        super().__init__()
        self.d_model    = d_model
        self.H          = horizon
        self.static_dim = static_dim

        self.input_proj = nn.Linear(d_input + 1, d_model)

        self.mp = MPFourier(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, d_model),
        )
        self.time_full = True

        self.blocks, self.norms, self.drops = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(
                LTI(
                    d_model,
                    dropout    = dropout,
                    transposed = True,
                    lr         = min(cfg["lr_min"], cfg["lr"]),
                    d_state    = cfg["d_state"],
                    dt_min     = cfg["min_dt"],
                    dt_max     = cfg["max_dt"],
                    lr_dt      = cfg["lr_dt"],
                    cfr        = cfg["cfr"],
                    cfi        = cfg["cfi"],
                    wd         = cfg["wd"],
                )
            )
            self.norms.append(nn.BatchNorm1d(d_model))
            self.drops.append(dropout_fn(dropout))

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # ======================================================================= #
    #  Standard forward (used for training — unchanged)                        #
    # ======================================================================= #

    def forward(
        self,
        x_past:       torch.Tensor,  # (B, L, d_input)
        noisy_future: torch.Tensor,  # (B, H, 1)
        t:            torch.Tensor,  # (B,)
        x_future:     torch.Tensor,  # (B, H-1, d_input)
        static_attr:  torch.Tensor,  # (B, static_dim)
    ) -> torch.Tensor:
        B, L, _ = x_past.shape
        H        = self.H
        device   = x_past.device

        t_feats = self.mp(t)

        all_met    = torch.cat([x_past, x_future], dim=1)
        pad_flow   = torch.zeros(B, L - 1, 1, device=device)
        all_flow   = torch.cat([pad_flow, noisy_future], dim=1)
        static_seq = static_attr[:, 0, :].unsqueeze(1).expand(-1, L + H - 1, -1)
        feats      = torch.cat([all_met, all_flow, static_seq], dim=-1)

        h = self.input_proj(feats)

        t_b        = self.time_mlp(t_feats)
        time_bias  = torch.zeros_like(h)
        if self.time_full:
            time_bias[:, L - 1:, :] = t_b.unsqueeze(1).expand(-1, self.H, -1)
        else:
            time_bias[:, L - 1, :] = t_b
        h = h + time_bias

        h = h.transpose(1, 2)
        for blk, norm, drop in zip(self.blocks, self.norms, self.drops):
            z, _ = blk(h)
            h    = norm(h + drop(z))
        h = h.transpose(1, 2)

        out = self.head(h)
        return out[:, -H:, :]

    # ======================================================================= #
    #  Standard DDIM sampler (used when no caching is requested)               #
    # ======================================================================= #

    @torch.no_grad()
    def sample_ddim(self,
                    x_past: torch.Tensor,
                    static_attributes: torch.Tensor,
                    future_pcp: torch.Tensor,
                    num_steps: int = 10,
                    eta: float = 0.0,
                    ddpm_n_steps: int = None,
                    ddpm_beta_start: float = 1e-4,
                    ddpm_beta_end: float = 2e-2,
                    ) -> torch.Tensor:
        device = x_past.device
        B, L, _ = x_past.shape
        H = future_pcp.size(1) + 1
        x = torch.randn(B, H, 1, device=device)

        if ddpm_n_steps is not None:
            betas     = torch.linspace(ddpm_beta_start, ddpm_beta_end, ddpm_n_steps, device=device)
            alpha_bar = torch.cumprod(1.0 - betas, dim=0)
            stride    = ddpm_n_steps // num_steps

            for i in range(num_steps - 1, -1, -1):
                t_idx      = min(i * stride, ddpm_n_steps - 1)
                t_idx_prev = min((i - 1) * stride, ddpm_n_steps - 1) if i > 0 else None
                t_float    = torch.full((B,), t_idx / (ddpm_n_steps - 1), device=device)

                pred   = self.forward(x_past=x_past, noisy_future=x, t=t_float,
                                      x_future=future_pcp, static_attr=static_attributes)
                ab_t   = alpha_bar[t_idx]
                x0     = (x - (1.0 - ab_t).sqrt() * pred) / ab_t.sqrt().clamp(min=1e-8)

                if i > 0:
                    ab_prev = alpha_bar[t_idx_prev]
                    if eta > 0:
                        sigma = eta * torch.sqrt(
                            ((1 - ab_prev) / (1 - ab_t)) * (1 - ab_t / ab_prev)
                        ).clamp(min=0)
                        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * pred + sigma * torch.randn_like(x)
                    else:
                        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * pred
                else:
                    x = x0
        else:
            ts = torch.linspace(0., 1., num_steps, device=device)

            for i in range(num_steps - 1, -1, -1):
                t      = ts[i].repeat(B)
                t_prev = ts[i - 1].repeat(B) if i > 0 else ts[0].repeat(B)

                pred = self.forward(x_past=x_past, noisy_future=x, t=t,
                                    x_future=future_pcp, static_attr=static_attributes)

                _, alpha_t,  sigma_t  = diffusion_params(t)
                _, alpha_tp, sigma_tp = diffusion_params(t_prev)
                alpha_t  = alpha_t.view(B, 1, 1);  sigma_t  = sigma_t.view(B, 1, 1)
                alpha_tp = alpha_tp.view(B, 1, 1);  sigma_tp = sigma_tp.view(B, 1, 1)

                x0  = alpha_t * x - sigma_t * pred
                eps = (pred + sigma_t * x0) / alpha_t

                if i > 0:
                    sigma = eta * torch.sqrt(
                        torch.clamp((sigma_tp ** 2) * (1 - (alpha_t ** 2 / alpha_tp ** 2)), min=1e-12)
                    )
                    noise = torch.randn_like(x) if eta > 0 else 0.0
                    x = alpha_tp * x0 + sigma_tp * eps + sigma * noise
                else:
                    x = x0

        return x.squeeze(-1)

    # ======================================================================= #
    #  FFT fast path (kept from decoder_only_ssm_fast — fully functional)      #
    # ======================================================================= #

    @torch.no_grad()
    def encode_past_fft(self, x_past, x_future, static_attr):
        """Cache past rfft per layer. Identical to decoder_only_ssm_fast."""
        B, L, _ = x_past.shape
        H = self.H
        device = x_past.device

        past_met   = x_past[:, :-1, :]
        past_flow  = torch.zeros(B, L - 1, 1, device=device)
        static_seq = static_attr[:, 0, :].unsqueeze(1).expand(-1, L - 1, -1)
        feats_past = torch.cat([past_met, past_flow, static_seq], dim=-1)
        h = self.input_proj(feats_past).transpose(1, 2)   # (B, d_model, L-1)

        cache = []
        for blk, norm, drop in zip(self.blocks, self.norms, self.drops):
            _, k_f, fft_n = blk.kernel.forward_fft(L + H - 1)

            past_u_f = torch.fft.rfft(
                F.pad(h.float(), (0, fft_n - (L - 1))), n=fft_n
            )

            z = torch.fft.irfft(past_u_f * k_f, n=fft_n)[..., :L - 1]
            z = z + h.float() * blk.D.float().unsqueeze(-1)
            z = blk.dropout(blk.activation(z))
            z = blk.output_linear(z)

            h = norm(h + drop(z).to(h.dtype))
            cache.append({'past_u_f': past_u_f, 'k_f': k_f, 'fft_n': fft_n})

        return cache

    @torch.no_grad()
    def decode_future_fft(self, x_past, x_future, static_attr,
                          noisy_future, t, past_fft_cache):
        """Decode future via FFT linearity. Identical to decoder_only_ssm_fast."""
        B, L, _ = x_past.shape
        H = self.H
        device = x_past.device

        future_met  = torch.cat([x_past[:, -1:, :], x_future], dim=1)
        future_flow = noisy_future
        static_seq  = static_attr[:, 0, :].unsqueeze(1).expand(-1, H, -1)
        feats_fut   = torch.cat([future_met, future_flow, static_seq], dim=-1)
        h_future    = self.input_proj(feats_fut).transpose(1, 2)

        t_b      = self.time_mlp(self.mp(t))
        h_future = h_future + t_b.unsqueeze(-1).expand(-1, -1, H)

        for blk, norm, drop, c in zip(self.blocks, self.norms, self.drops, past_fft_cache):
            k_f      = c['k_f']
            fft_n    = c['fft_n']
            past_u_f = c['past_u_f']

            future_u_f = torch.fft.rfft(
                F.pad(h_future.float(), (L - 1, fft_n - (L - 1) - H)), n=fft_n
            )
            u_f      = past_u_f + future_u_f
            z_future = torch.fft.irfft(u_f * k_f, n=fft_n)[..., L - 1: L + H - 1]
            z_future = z_future + h_future.float() * blk.D.float().unsqueeze(-1)
            z_future = blk.dropout(blk.activation(z_future))
            z_future = blk.output_linear(z_future)

            h_future = norm(h_future + drop(z_future).to(h_future.dtype))

        return self.head(h_future.transpose(1, 2))

    @torch.no_grad()
    def sample_ddim_fast_fft(self,
                             x_past, static_attributes, future_pcp,
                             num_steps=10, eta=0.0,
                             ddpm_n_steps=None, ddpm_beta_start=1e-4, ddpm_beta_end=2e-2,
                             past_fft_cache=None):
        """Fast FFT sampling (kept from decoder_only_ssm_fast)."""
        device = x_past.device
        B, L, _ = x_past.shape
        H = future_pcp.size(1) + 1
        x = torch.randn(B, H, 1, device=device)

        if past_fft_cache is None:
            past_fft_cache = self.encode_past_fft(x_past, future_pcp, static_attributes)

        if ddpm_n_steps is not None:
            betas     = torch.linspace(ddpm_beta_start, ddpm_beta_end, ddpm_n_steps, device=device)
            alpha_bar = torch.cumprod(1.0 - betas, dim=0)
            stride    = ddpm_n_steps // num_steps

            for i in range(num_steps - 1, -1, -1):
                t_idx      = min(i * stride, ddpm_n_steps - 1)
                t_idx_prev = min((i - 1) * stride, ddpm_n_steps - 1) if i > 0 else None
                t_float    = torch.full((B,), t_idx / (ddpm_n_steps - 1), device=device)

                pred = self.decode_future_fft(x_past, future_pcp, static_attributes,
                                              x, t_float, past_fft_cache)
                ab_t = alpha_bar[t_idx]
                x0   = (x - (1.0 - ab_t).sqrt() * pred) / ab_t.sqrt().clamp(min=1e-8)

                if i > 0:
                    ab_prev = alpha_bar[t_idx_prev]
                    if eta > 0:
                        sigma = eta * torch.sqrt(
                            ((1 - ab_prev) / (1 - ab_t)) * (1 - ab_t / ab_prev)
                        ).clamp(min=0)
                        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * pred + sigma * torch.randn_like(x)
                    else:
                        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * pred
                else:
                    x = x0
        else:
            ts = torch.linspace(0.0, 1.0, num_steps, device=device)

            for i in range(num_steps - 1, -1, -1):
                t      = ts[i].repeat(B)
                t_prev = ts[i - 1].repeat(B) if i > 0 else ts[0].repeat(B)

                pred = self.decode_future_fft(x_past, future_pcp, static_attributes,
                                              x, t, past_fft_cache)

                _, alpha_t,  sigma_t  = diffusion_params(t)
                _, alpha_tp, sigma_tp = diffusion_params(t_prev)
                alpha_t  = alpha_t.view(B, 1, 1);  sigma_t  = sigma_t.view(B, 1, 1)
                alpha_tp = alpha_tp.view(B, 1, 1);  sigma_tp = sigma_tp.view(B, 1, 1)

                x0  = alpha_t * x - sigma_t * pred
                eps = (pred + sigma_t * x0) / alpha_t

                if i > 0:
                    sigma = eta * torch.sqrt(
                        torch.clamp((sigma_tp ** 2) * (1 - (alpha_t ** 2 / alpha_tp ** 2)), min=1e-12)
                    )
                    noise = torch.randn_like(x) if eta > 0 else 0.0
                    x = alpha_tp * x0 + sigma_tp * eps + sigma * noise
                else:
                    x = x0

        return x.squeeze(-1)

    # ======================================================================= #
    #  Hybrid path: FFT encode past + fully vectorized recurrent future decode #
    # ======================================================================= #
    #
    #  WHY the old loop-based hybrid was slower:
    #    8 steps × 6 layers × 500 calls = 24,000 Python→GPU dispatches/batch.
    #    Each dispatch launches a tiny kernel; GPU sits idle between dispatches.
    #
    #  SOLUTION — closed-form parallel decode, zero Python loops in the hot path:
    #
    #    The full SSM output at future step t (starting from cached state_0) is:
    #
    #      y[t] = y_init[t] + y_inputs[t]
    #
    #    where:
    #      y_init[t]   = 2·Re( C · A^(t+1) · state_0 )
    #                  = einsum over N — one large matmul, no loop
    #
    #      y_inputs[t] = Σᵢ₌₀ᵗ K[t-i]·u[i]  +  D·u[t]
    #                  = lower-triangular (H×H) matmul with precomputed kernel
    #                  — K[t] = SSM impulse response, computed once per layer
    #
    #  encode_past_hybrid():
    #    - FFT → inter-layer h (parallel, identical to encode_past_fft)
    #    - Vectorized einsum → terminal SSM state (no loop, one matmul over L)
    #    - Precompute K_lags and A_pows_fut (fixed per layer, reused 500×)
    #
    #  decode_future_hybrid():
    #    - y_init   : einsum (H, d_model, N) × (B, d_model, N) → (B, d_model, H)
    #    - y_inputs : einsum (d_model, H, H) × (B, d_model, H) → (B, d_model, H)
    #    - No Python loop, no FFT
    # ======================================================================= #

    @torch.no_grad()
    def encode_past_hybrid(self, x_past, x_future, static_attr):
        """
        Encode past (0..L-2) via FFT for inter-layer h, plus vectorized einsum
        for the terminal SSM state. Precomputes all decode-time constants.

        No Python loops for state extraction — uses a single einsum over L.

        Parameters
        ----------
        x_past      : (B, L, d_input)
        x_future    : (B, H-1, d_input)
        static_attr : (B, static_dim) or (B, 1, static_dim)

        Returns
        -------
        cache : list of dicts, one per layer:
            'ssm_state'  : (B, d_model, N) cfloat — terminal SSM state at L-2
            'A_pows_fut' : (H, d_model, N) cfloat — A^1..A^H (for y_init einsum)
            'C_param'    : (d_model, N) cfloat
            'K_lags'     : (d_model, H, H) float  — causal Toeplitz kernel matrix
            'D'          : (d_model,) float
        """
        B, L, _ = x_past.shape
        H = self.H
        device = x_past.device

        # Past feature matrix: positions 0..L-2
        past_met  = x_past[:, :-1, :]
        past_flow = torch.zeros(B, L - 1, 1, device=device)
        if static_attr.dim() == 2:
            static_seq = static_attr.unsqueeze(1).expand(-1, L - 1, -1)
        else:
            static_seq = static_attr[:, 0, :].unsqueeze(1).expand(-1, L - 1, -1)
        feats_past = torch.cat([past_met, past_flow, static_seq], dim=-1)
        h = self.input_proj(feats_past).transpose(1, 2)   # (B, d_model, L-1)

        cache = []
        for blk, norm, drop in zip(self.blocks, self.norms, self.drops):
            h_f32 = h.float()

            # ── Pass 1: FFT → inter-layer h (parallel, identical to encode_past_fft) ──
            _, k_f, fft_n = blk.kernel.forward_fft(L + H - 1)
            past_u_f = torch.fft.rfft(F.pad(h_f32, (0, fft_n - (L - 1))), n=fft_n)
            z = torch.fft.irfft(past_u_f * k_f, n=fft_n)[..., :L - 1]
            z = z + h_f32 * blk.D.float().unsqueeze(-1)
            z = blk.dropout(blk.activation(z))
            z = blk.output_linear(z)
            h = norm(h + drop(z).to(h.dtype))             # (B, d_model, L-1) → next layer

            # ── Pass 2: vectorized terminal state extraction (no Python loop) ─────
            #
            # state[L-2] = Σᵢ₌₀^{L-2} A^(L-2-i) · B · h[i]
            #            = B · einsum('dnl, bdl -> bdn', A_pows_past, h_complex)
            #
            # A_pows_past[d,n,l] = A_bar[d,n]^(L-2-l)  for l=0..L-2
            #
            # Use float64 for power computation to avoid float32 rounding on
            # large exponents (A_bar^363): error drops from ~5e-4 to ~1e-6.
            A_bar, B_bar, C_param = blk.kernel.get_discretized_params()  # (d_model, N) cfloat

            l_idx       = torch.arange(L - 1, device=device, dtype=torch.float64)
            A_pows_past = A_bar.to(torch.cdouble).unsqueeze(-1) ** (L - 2 - l_idx)
            # (d_model, N, L-1) cdouble
            ssm_state = (B_bar.to(torch.cdouble).unsqueeze(0) * torch.einsum(
                'dnl,bdl->bdn',
                A_pows_past,
                h_f32.to(torch.cdouble),
            )).to(torch.cfloat)   # (B, d_model, N) cfloat

            # ── Precompute decode-time constants (fixed for all 500 future calls) ──

            # A^1..A^H for y_init: (H, d_model, N) cfloat
            t_range     = torch.arange(1, H + 1, device=device, dtype=torch.float32)
            A_pows_fut  = A_bar.unsqueeze(0) ** t_range.view(H, 1, 1).to(torch.cfloat)

            # Causal Toeplitz kernel matrix for y_inputs:
            # K_lags[d, t, i] = K[d, t-i]  for t>=i, else 0
            # K[d, lag] = SSM impulse response at lag steps
            K_H    = blk.kernel._compute_kernel(H)   # (d_model, H) real
            t_out  = torch.arange(H, device=device)
            lag    = (t_out.unsqueeze(1) - t_out.unsqueeze(0)).clamp(min=0)   # (H, H)
            mask   = (t_out.unsqueeze(1) >= t_out.unsqueeze(0)).float()       # (H, H)
            K_lags = K_H[:, lag] * mask.unsqueeze(0)                          # (d_model, H, H)

            cache.append({
                'ssm_state':  ssm_state,   # (B, d_model, N)   — changes per batch
                'A_pows_fut': A_pows_fut,  # (H, d_model, N)   — fixed per layer
                'C_param':    C_param,     # (d_model, N)       — fixed per layer
                'K_lags':     K_lags,      # (d_model, H, H)    — fixed per layer
                'D':          blk.D.float(),  # (d_model,)       — fixed per layer
            })

        return cache

    @torch.no_grad()
    def decode_future_hybrid(self, x_past, x_future, static_attr,
                              noisy_future, t, past_hybrid_cache):
        """
        Decode H future positions with zero Python loops.

        Per layer:
          y_init[t]   = 2·Re(C · A^(t+1) · state_0)   — einsum over N
          y_inputs[t] = Σᵢ K[t-i]·u[i]  +  D·u[t]     — (H,H) matmul
          z_future    = y_init + y_inputs

        Parameters
        ----------
        x_past            : (B, L, d_input)
        x_future          : (B, H-1, d_input)
        static_attr       : (B, static_dim) or (B, 1, static_dim)
        noisy_future      : (B, H, 1)   — changes each DDIM step
        t                 : (B,)        — changes each DDIM step
        past_hybrid_cache : from encode_past_hybrid()

        Returns
        -------
        pred : (B, H, 1)
        """
        B, L, _ = x_past.shape
        H = self.H
        device = x_past.device

        # Future features: positions L-1..L+H-2
        future_met  = torch.cat([x_past[:, -1:, :], x_future], dim=1)
        future_flow = noisy_future
        if static_attr.dim() == 2:
            static_seq = static_attr.unsqueeze(1).expand(-1, H, -1)
        else:
            static_seq = static_attr[:, 0, :].unsqueeze(1).expand(-1, H, -1)
        feats_fut = torch.cat([future_met, future_flow, static_seq], dim=-1)
        h_future  = self.input_proj(feats_fut).transpose(1, 2)   # (B, d_model, H)

        t_b      = self.time_mlp(self.mp(t))
        h_future = h_future + t_b.unsqueeze(-1).expand(-1, -1, H)

        for blk, norm, drop, c in zip(
            self.blocks, self.norms, self.drops, past_hybrid_cache
        ):
            ssm_state  = c['ssm_state']    # (B, d_model, N) cfloat
            A_pows_fut = c['A_pows_fut']   # (H, d_model, N) cfloat
            C_param    = c['C_param']      # (d_model, N) cfloat
            K_lags     = c['K_lags']       # (d_model, H, H) float
            D          = c['D']            # (d_model,) float

            h_f32 = h_future.float()      # (B, d_model, H)

            # ── y_init: contribution from cached past state ───────────────────
            # z0[b,d,n] = C[d,n] · state_0[b,d,n]
            # y_init[b,d,h] = 2·Re( Σₙ A^(h+1)[d,n] · z0[b,d,n] )
            z0     = C_param.unsqueeze(0) * ssm_state              # (B, d_model, N) cfloat
            y_init = 2.0 * torch.einsum(
                'hdn,bdn->bdh', A_pows_fut, z0
            ).real                                                  # (B, d_model, H)

            # ── y_inputs: causal conv of future inputs with SSM kernel ────────
            # y_inputs[b,d,t] = Σᵢ K_lags[d,t,i]·h[b,d,i]  +  D[d]·h[b,d,t]
            y_inputs = (torch.einsum('dti,bdi->bdt', K_lags, h_f32)
                        + D.view(1, -1, 1) * h_f32)               # (B, d_model, H)

            z_future = y_init + y_inputs                           # (B, d_model, H)
            z_future = blk.dropout(blk.activation(z_future))
            z_future = blk.output_linear(z_future)

            h_future = norm(h_future + drop(z_future).to(h_future.dtype))

        return self.head(h_future.transpose(1, 2))                 # (B, H, 1)

    @torch.no_grad()
    def sample_ddim_hybrid(self,
                           x_past, static_attributes, future_pcp,
                           num_steps=10, eta=0.0,
                           ddpm_n_steps=None, ddpm_beta_start=1e-4, ddpm_beta_end=2e-2,
                           past_hybrid_cache=None):
        """
        Fast DDIM sampling using hybrid past caching + recurrent future decode.

        Encode past once (FFT for inter-layer h, recurrent for terminal state),
        then decode H=8 future steps per DDIM iteration using pure recurrence.

        past_hybrid_cache : optional pre-computed cache from encode_past_hybrid().
            Pass this when generating multiple ensemble members for the same
            inputs — the terminal state is shared and cloned inside decode_future_hybrid.

        Returns
        -------
        samples : (B, H) denoised future streamflow.
        """
        device = x_past.device
        B, L, _ = x_past.shape
        H = future_pcp.size(1) + 1
        x = torch.randn(B, H, 1, device=device)

        if past_hybrid_cache is None:
            past_hybrid_cache = self.encode_past_hybrid(x_past, future_pcp, static_attributes)

        if ddpm_n_steps is not None:
            betas     = torch.linspace(ddpm_beta_start, ddpm_beta_end, ddpm_n_steps, device=device)
            alpha_bar = torch.cumprod(1.0 - betas, dim=0)
            stride    = ddpm_n_steps // num_steps

            for i in range(num_steps - 1, -1, -1):
                t_idx      = min(i * stride, ddpm_n_steps - 1)
                t_idx_prev = min((i - 1) * stride, ddpm_n_steps - 1) if i > 0 else None
                t_float    = torch.full((B,), t_idx / (ddpm_n_steps - 1), device=device)

                pred = self.decode_future_hybrid(
                    x_past, future_pcp, static_attributes, x, t_float, past_hybrid_cache
                )
                ab_t = alpha_bar[t_idx]
                x0   = (x - (1.0 - ab_t).sqrt() * pred) / ab_t.sqrt().clamp(min=1e-8)

                if i > 0:
                    ab_prev = alpha_bar[t_idx_prev]
                    if eta > 0:
                        sigma = eta * torch.sqrt(
                            ((1 - ab_prev) / (1 - ab_t)) * (1 - ab_t / ab_prev)
                        ).clamp(min=0)
                        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * pred + sigma * torch.randn_like(x)
                    else:
                        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * pred
                else:
                    x = x0
        else:
            ts = torch.linspace(0.0, 1.0, num_steps, device=device)

            for i in range(num_steps - 1, -1, -1):
                t      = ts[i].repeat(B)
                t_prev = ts[i - 1].repeat(B) if i > 0 else ts[0].repeat(B)

                pred = self.decode_future_hybrid(
                    x_past, future_pcp, static_attributes, x, t, past_hybrid_cache
                )

                _, alpha_t,  sigma_t  = diffusion_params(t)
                _, alpha_tp, sigma_tp = diffusion_params(t_prev)
                alpha_t  = alpha_t.view(B, 1, 1);  sigma_t  = sigma_t.view(B, 1, 1)
                alpha_tp = alpha_tp.view(B, 1, 1);  sigma_tp = sigma_tp.view(B, 1, 1)

                x0  = alpha_t * x - sigma_t * pred
                eps = (pred + sigma_t * x0) / alpha_t

                if i > 0:
                    sigma = eta * torch.sqrt(
                        torch.clamp((sigma_tp ** 2) * (1 - (alpha_t ** 2 / alpha_tp ** 2)), min=1e-12)
                    )
                    noise = torch.randn_like(x) if eta > 0 else 0.0
                    x = alpha_tp * x0 + sigma_tp * eps + sigma * noise
                else:
                    x = x0

        return x.squeeze(-1)
