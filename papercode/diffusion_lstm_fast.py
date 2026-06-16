"""
diffusion_lstm_fast.py

Cached inference for the diffusion_lstm (EncoderDecoderDiffusionWrapper with LSTM decoder):
  - encode_past(): runs the LSTM encoder over x_past ONCE and caches (h_enc, c_enc).
  - sample_ddim_fast(): reuses the cached encoder state for every ensemble member,
                        so the encoder is never re-run across 50 samples.

Everything else (forward, sample_ddim) is identical to diffusion_wrapper.py.
"""

import math
import torch
import torch.nn as nn
from papercode.diffusion_utils import diffusion_params


class MPFourier(nn.Module):
    def __init__(self, num_channels, bandwidth=1.0):
        super().__init__()
        self.register_buffer('freqs', 2 * math.pi * torch.randn(num_channels))
        self.register_buffer('phases', 2 * math.pi * torch.rand(num_channels))

    def forward(self, t):
        t = t.to(torch.float32)
        x = t[:, None] * self.freqs[None, :] + self.phases[None, :]
        return (x.cos() * math.sqrt(2)).to(t.dtype)


class DiffusionLSTMFast(nn.Module):
    """
    Fast (cached-encoder) wrapper for the diffusion LSTM model.

    KEY IDEA
    --------
    In the standard sample_ddim, self.encoder(x_past) is called inside the
    DDIM loop for every ensemble member.  But x_past never changes across
    ensemble members or DDIM steps — so the encoder output is identical every
    time.  encode_past() computes it once; sample_ddim_fast() injects the
    cached (h_enc, c_enc) directly, skipping the encoder entirely.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        hidden_size: int,
        time_emb_dim: int,
        decoder_name: str = 'lstm',
        prediction_type: str = 'velocity',
        guidance_weight: float = 0.0,
    ):
        super().__init__()
        self.encoder         = encoder
        self.decoder         = decoder
        self.hidden_size     = hidden_size
        self.time_emb_dim    = time_emb_dim
        self.decoder_name    = decoder_name
        self.prediction_type = prediction_type
        self.guidance_weight = guidance_weight

        self.mp = MPFourier(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, hidden_size),
        )

        self.proj_s_h = nn.Sequential(
            nn.Linear(27, 27 * 2), nn.SiLU(), nn.Linear(27 * 2, hidden_size)
        )
        self.proj_s_c = nn.Sequential(
            nn.Linear(27, 27 * 2), nn.SiLU(), nn.Linear(27 * 2, hidden_size)
        )

        self.output_proj = nn.Linear(hidden_size, 1)

    def forward(self, x_past, x_t, t, future_pcp, static_attributes):
        B, H, _ = x_t.shape
        _, encoded_state = self.encoder(x_past)
        t_feats = self.mp(t)
        t_emb   = self.time_mlp(t_feats)
        t_h = t_c = t_emb

        h_enc, c_enc = encoded_state
        h_0 = h_enc + t_h.unsqueeze(0)
        c_0 = c_enc + t_c.unsqueeze(0)

        dec_in  = torch.cat([x_t, future_pcp, static_attributes], dim=-1)
        dec_out, _ = self.decoder(dec_in, init_state=(h_0, c_0))
        return self.output_proj(dec_out)

    @torch.no_grad()
    def sample_ddim(self, x_past, static_attributes, future_pcp, num_steps=10, eta=0.0):
        device = x_past.device
        B = x_past.size(0)
        H = future_pcp.size(1)
        _, encoded_state = self.encoder(x_past)
        ts = torch.linspace(0., 1., num_steps, device=device)
        x  = torch.randn(B, H, 1, device=device)

        for i in range(num_steps - 1, -1, -1):
            t_cur  = ts[i].repeat(B)
            t_prev = ts[i - 1].repeat(B) if i > 0 else ts[0].repeat(B)
            t_emb  = self.time_mlp(self.mp(t_cur))
            h_enc, c_enc = encoded_state
            h_0 = h_enc + t_emb.unsqueeze(0)
            c_0 = c_enc + t_emb.unsqueeze(0)
            dec_in  = torch.cat([x, future_pcp, static_attributes], dim=-1)
            dec_out, _ = self.decoder(dec_in, init_state=(h_0, c_0))
            pred = self.output_proj(dec_out)

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
    def encode_past(self, x_past):
        """
        Run the LSTM encoder over x_past once and cache the result.
        Call this once per batch before the ensemble loop.

        x_past: (B, L, past_features)
        Returns: dict with keys 'h_enc', 'c_enc'
        """
        _, (h_enc, c_enc) = self.encoder(x_past)
        return {
            'h_enc': h_enc.clone(),
            'c_enc': c_enc.clone(),
        }

    @torch.no_grad()
    def sample_ddim_fast(
        self,
        x_past,
        static_attributes,
        future_pcp,
        num_steps=10,
        eta=0.0,
        encoder_cache=None,
    ):
        """
        DDIM sampling with cached encoder state.

        encoder_cache: output of encode_past(). If None, the encoder is run here
                       (no speedup vs sample_ddim).
        """
        device = x_past.device
        B = x_past.size(0)
        H = future_pcp.size(1)

        if encoder_cache is None:
            encoder_cache = self.encode_past(x_past)

        h_enc = encoder_cache['h_enc']  # (1, B, hidden)
        c_enc = encoder_cache['c_enc']  # (1, B, hidden)

        ts = torch.linspace(0., 1., num_steps, device=device)
        x  = torch.randn(B, H, 1, device=device)

        for i in range(num_steps - 1, -1, -1):
            t_cur  = ts[i].repeat(B)
            t_prev = ts[i - 1].repeat(B) if i > 0 else ts[0].repeat(B)

            t_emb = self.time_mlp(self.mp(t_cur))   # (B, hidden)
            h_0 = h_enc + t_emb.unsqueeze(0)         # (1, B, hidden)
            c_0 = c_enc + t_emb.unsqueeze(0)

            dec_in  = torch.cat([x, future_pcp, static_attributes], dim=-1)
            dec_out, _ = self.decoder(dec_in, init_state=(h_0, c_0))
            pred = self.output_proj(dec_out)          # (B, H, 1)

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
