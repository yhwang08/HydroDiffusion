"""
decoder_only_lstm_fast.py

Cached inference for decoder_only_lstm:
  - encode_past_lstm(): runs the LSTM over past (0..L-2) ONCE and caches (h_p, c_p).
  - sample_ddim_fast(): reuses the cached state for every ensemble member and every
                        DDIM step, so the expensive past encoding is never repeated.

Everything else (forward, sample_ddim) is identical to decoder_only_lstm.py.
"""

import math
import torch
import torch.nn as nn
from papercode.diffusion_utils import diffusion_params
from papercode.backbones.lstm import GenericLSTM

dropout_fn = nn.Dropout1d if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12) else nn.Dropout


class MPFourier(nn.Module):
    def __init__(self, num_channels, bandwidth=1.0):
        super().__init__()
        self.register_buffer('freqs', 2 * math.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * math.pi * torch.rand(num_channels))

    def forward(self, t):
        if t.dim() >= 2:
            t = t[..., 0]
            if t.dim() >= 2:
                t = t[:, 0]
        t = t.to(torch.float32)
        x = t[:, None] * self.freqs[None, :] + self.phases[None, :]
        return (x.cos() * math.sqrt(2.)).to(t.dtype)


class decoder_only_lstm(nn.Module):
    """
    Fast (cached) version of decoder_only_lstm.

    KEY IDEA
    --------
    In standard sample_ddim, the full LSTM sequence (past + future) is re-run for
    every ensemble member × every DDIM step.  The past segment (0..L-2) is
    identical across all of these calls — only the future segment changes.

    encode_past_lstm() runs the past segment ONCE and returns (h_p, c_p).
    sample_ddim_fast() accepts the pre-computed cache and only runs the future
    segment (H steps) per DDIM step, saving O(L/H) compute per sample.
    """

    def __init__(
        self,
        d_input: int,
        hidden_size: int,
        cfg: dict,
        *,
        horizon: int = 8,
        static_dim: int = 27,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.H           = horizon
        self.hidden_size = hidden_size
        self.static_dim  = static_dim
        self.in_dim      = d_input + 1

        self.lstm = GenericLSTM(
            input_size       = self.in_dim,
            hidden_size      = hidden_size,
            dropout          = cfg.get('dropout', 0.0),
            init_forget_bias = cfg.get('initial_forget_gate_bias', 3.0),
            batch_first      = True
        )

        self.mp = MPFourier(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, hidden_size),
        )

        self.head = nn.Linear(hidden_size, 1)

    def _build_sequence(self, x_past, noisy_future, x_future, static_attr):
        B, L, _ = x_past.shape
        H = self.H
        device = x_past.device

        all_met  = torch.cat([x_past, x_future], dim=1)
        pad_flow = torch.zeros(B, L - 1, 1, device=device)
        all_flow = torch.cat([pad_flow, noisy_future], dim=1)

        if static_attr.dim() == 2:
            static_seq = static_attr.unsqueeze(1).expand(-1, L + H - 1, -1)
        else:
            static_seq = static_attr[:, 0, :].unsqueeze(1).expand(-1, L + H - 1, -1)

        feats = torch.cat([all_met, all_flow, static_seq], dim=-1).contiguous()
        return feats, L

    def forward(self, x_past, noisy_future, t, x_future, static_attr):
        B, L, _ = x_past.shape
        H = self.H

        feats, L = self._build_sequence(x_past, noisy_future, x_future, static_attr)

        if L - 1 > 0:
            past = feats[:, :L - 1, :]
            _, (h_p, c_p) = self.lstm(past)
        else:
            h_p = feats.new_zeros(B, self.hidden_size)
            c_p = feats.new_zeros(B, self.hidden_size)

        t_emb = self.time_mlp(self.mp(t))
        h0 = h_p + t_emb
        c0 = c_p + t_emb

        fut_in  = feats[:, L - 1:, :]
        fut_out, _ = self.lstm(fut_in, init_state=(h0, c0))
        return self.head(fut_out)

    @torch.no_grad()
    def sample_ddim(self, x_past, static_attributes, future_pcp, num_steps=10, eta=0.0):
        device = x_past.device
        B, L, _ = x_past.shape
        H = future_pcp.size(1) + 1
        ts = torch.linspace(0., 1., num_steps, device=device)
        x  = torch.randn(B, H, 1, device=device)

        for i in range(num_steps - 1, -1, -1):
            t_cur  = ts[i].repeat(B)
            t_prev = ts[i - 1].repeat(B) if i > 0 else ts[0].repeat(B)
            pred = self.forward(x_past, x, t_cur, future_pcp, static_attributes)
            _, alpha_t,  sigma_t  = diffusion_params(t_cur)
            _, alpha_tp, sigma_tp = diffusion_params(t_prev)
            alpha_t  = alpha_t.view(B, 1, 1);  sigma_t  = sigma_t.view(B, 1, 1)
            alpha_tp = alpha_tp.view(B, 1, 1); sigma_tp = sigma_tp.view(B, 1, 1)
            x0  = alpha_t * x - sigma_t * pred
            eps = (pred + sigma_t * x0) / alpha_t
            if i > 0:
                sigma = eta * torch.sqrt(torch.clamp((sigma_tp**2) * (1 - (alpha_t**2 / alpha_tp**2)), min=1e-12))
                noise = torch.randn_like(x) if eta > 0 else 0.0
                x = alpha_tp * x0 + sigma_tp * eps + sigma * noise
            else:
                x = x0
        return x.squeeze(-1)

    # ------------------------------------------------------------------
    # Fast cached methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_past_lstm(self, x_past, x_future, static_attr):
        """
        Run the LSTM over the past segment (0..L-2) once and return (h_p, c_p).
        Call this once per batch before the ensemble loop.

        x_past:      (B, L, d_input)
        x_future:    (B, H-1, d_input)
        static_attr: (B, static_dim) or (B, 1, static_dim)

        Returns: dict with keys 'h_p', 'c_p', 'fut_met_static', 'L'
        """
        B, L, _ = x_past.shape
        H = self.H
        device = x_past.device

        all_met = torch.cat([x_past, x_future], dim=1)           # (B, L+H-1, d_input)
        if static_attr.dim() == 2:
            static_seq = static_attr.unsqueeze(1).expand(-1, L + H - 1, -1)
        else:
            static_seq = static_attr[:, 0, :].unsqueeze(1).expand(-1, L + H - 1, -1)

        # Build past features (no noisy_future needed — use zeros placeholder)
        pad_flow  = torch.zeros(B, L - 1, 1, device=device)
        fut_zeros = torch.zeros(B, H, 1, device=device)
        all_flow  = torch.cat([pad_flow, fut_zeros], dim=1)
        feats     = torch.cat([all_met, all_flow, static_seq], dim=-1).contiguous()

        if L - 1 > 0:
            _, (h_p, c_p) = self.lstm(feats[:, :L - 1, :])
        else:
            h_p = feats.new_zeros(B, self.hidden_size)
            c_p = feats.new_zeros(B, self.hidden_size)

        # Pre-build the future met+static columns (flow column filled per DDIM step)
        fut_met_static = torch.cat(
            [all_met[:, L - 1:, :], static_seq[:, L - 1:, :]], dim=-1
        )  # (B, H, d_input + static_dim)

        return {
            'h_p': h_p,
            'c_p': c_p,
            'fut_met_static': fut_met_static,  # (B, H, d_input+static_dim)
            'L': L,
        }

    @torch.no_grad()
    def sample_ddim_fast(
        self,
        x_past,
        static_attributes,
        future_pcp,
        num_steps=10,
        eta=0.0,
        past_state_cache=None,
    ):
        """
        DDIM sampling with cached past encoding.

        past_state_cache: output of encode_past_lstm(). If None, it is computed here
                          (but then there is no speedup vs sample_ddim).
        """
        device = x_past.device
        B, L, _ = x_past.shape
        H = future_pcp.size(1) + 1

        if past_state_cache is None:
            past_state_cache = self.encode_past_lstm(x_past, future_pcp, static_attributes)

        h_p            = past_state_cache['h_p']             # (B, hidden)
        c_p            = past_state_cache['c_p']             # (B, hidden)
        fut_met_static = past_state_cache['fut_met_static']  # (B, H, d_input+static_dim)

        ts = torch.linspace(0., 1., num_steps, device=device)
        x  = torch.randn(B, H, 1, device=device)

        for i in range(num_steps - 1, -1, -1):
            t_cur  = ts[i].repeat(B)
            t_prev = ts[i - 1].repeat(B) if i > 0 else ts[0].repeat(B)

            # Time conditioning
            t_emb = self.time_mlp(self.mp(t_cur))  # (B, hidden)
            h0 = h_p + t_emb
            c0 = c_p + t_emb

            # Build future input: [met | noisy_flow | static]
            fut_in = torch.cat([fut_met_static[:, :, :future_pcp.size(-1)],
                                 x,
                                 fut_met_static[:, :, future_pcp.size(-1):]], dim=-1)
            # (B, H, d_input + 1 + static_dim) = in_dim
            fut_out, _ = self.lstm(fut_in, init_state=(h0, c0))
            pred = self.head(fut_out)  # (B, H, 1)

            _, alpha_t,  sigma_t  = diffusion_params(t_cur)
            _, alpha_tp, sigma_tp = diffusion_params(t_prev)
            alpha_t  = alpha_t.view(B, 1, 1);  sigma_t  = sigma_t.view(B, 1, 1)
            alpha_tp = alpha_tp.view(B, 1, 1); sigma_tp = sigma_tp.view(B, 1, 1)

            x0  = alpha_t * x - sigma_t * pred
            eps = (pred + sigma_t * x0) / alpha_t

            if i > 0:
                sigma = eta * torch.sqrt(torch.clamp((sigma_tp**2) * (1 - (alpha_t**2 / alpha_tp**2)), min=1e-12))
                noise = torch.randn_like(x) if eta > 0 else 0.0
                x = alpha_tp * x0 + sigma_tp * eps + sigma * noise
            else:
                x = x0

        return x.squeeze(-1)
