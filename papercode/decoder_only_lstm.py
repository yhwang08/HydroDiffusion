import math
import torch
import torch.nn as nn
from papercode.diffusion_utils import diffusion_params
from papercode.lstm import Seq2SeqLSTM, EncoderDecoderDetLSTM
from papercode.SSM_test import HOPE, setup_optimizer
from papercode.backbones.lstm import GenericLSTM
import pdb
from runtime_profiler import RuntimeProfiler

# PyTorch 1.12+ naming
dropout_fn = nn.Dropout1d if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12) else nn.Dropout

class MPFourier(nn.Module):
    def __init__(self, num_channels, bandwidth=1.0):
        super().__init__()
        self.register_buffer('freqs', 2*math.pi*torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2*math.pi*torch.rand(num_channels))

    def forward(self, t):
        # Accept (B,), (B,1), (B,H) or (B,H,1) and reduce to (B,)
        if t.dim() >= 2:
            t = t[..., 0]
            if t.dim() >= 2:
                t = t[:, 0]
        t = t.to(torch.float32)  # (B,)
        x = t[:, None]*self.freqs[None, :] + self.phases[None, :]
        return (x.cos() * math.sqrt(2.)).to(t.dtype)  # (B, C)

class decoder_only_lstm(nn.Module):
    """
    Encoder-only LSTM with GenericLSTM:
      - LSTM input_size = d_input + 1 + static_dim (e.g., 33)
      - Run past (0..L-2) to get (h_p, c_p)
      - Add time-embedding to (h_p, c_p) at the nowcast boundary (bottleneck)
      - Run future segment (L-1..L+H-2) from (h0, c0)
    """
    def __init__(
        self,
        d_input: int,                 # dynamic forcings ONLY (no statics here)
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
        self.in_dim      = d_input + 1 # keep raw input size (e.g., 33)

        # GenericLSTM over raw features (used for both past and future segments)
        self.lstm = GenericLSTM(
            input_size        = self.in_dim,
            hidden_size       = hidden_size,
            dropout           = cfg.get('dropout', 0.0),
            init_forget_bias  = cfg.get('initial_forget_gate_bias', 3.0),
            batch_first       = True
        )

        # Time embedding -> hidden-size bias (added to BOTH h and c at bottleneck)
        self.mp = MPFourier(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, hidden_size),
        )

        # Readout
        self.head = nn.Linear(hidden_size, 1)

    def _build_sequence(self, x_past, noisy_future, x_future, static_attr):
        """
        x_past:      (B, L, d_input)
        noisy_future:(B, H, 1)
        x_future:    (B, H-1, d_input)   # excludes nowcast day
        static_attr: (B, static_dim) or (B, 1, static_dim)

        Returns feats: (B, L+H-1, d_input+1+static_dim), L
        """
        B, L, _ = x_past.shape
        H = self.H
        device = x_past.device

        all_met  = torch.cat([x_past, x_future], dim=1)           # (B, L+H-1, d_input)
        pad_flow = torch.zeros(B, L-1, 1, device=device)
        all_flow = torch.cat([pad_flow, noisy_future], dim=1)     # (B, L+H-1, 1)

        if static_attr.dim() == 2:
            static_seq = static_attr.unsqueeze(1).expand(-1, L+H-1, -1)  # (B, L+H-1, S)
        else:
            static_seq = static_attr[:, 0, :].unsqueeze(1).expand(-1, L+H-1, -1)

        feats = torch.cat([all_met, all_flow, static_seq], dim=-1).contiguous()
        return feats, L

    def forward(
        self,
        x_past:       torch.Tensor,  # (B, L, d_input)
        noisy_future: torch.Tensor,  # (B, H, 1)
        t:            torch.Tensor,  # (B,) or (B,H,1) etc.
        x_future:     torch.Tensor,  # (B, H-1, d_input)
        static_attr:  torch.Tensor,  # (B, static_dim) or (B,1,static_dim)
    ) -> torch.Tensor:
        B, L, _ = x_past.shape
        H = self.H

        feats, L = self._build_sequence(x_past, noisy_future, x_future, static_attr)  # (B, T, in_dim)
        T = feats.size(1)
        assert T == L + H - 1, "Sequence length mismatch."

        # 1) Encode past (0..L-2) with GenericLSTM (stateless call)
        if L - 1 > 0:
            past = feats[:, :L-1, :]                               # (B, L-1, in_dim)
            _, (h_p, c_p) = self.lstm(past)                        # h_p, c_p: (B, hidden)
        else:
            h_p = feats.new_zeros(B, self.hidden_size)
            c_p = feats.new_zeros(B, self.hidden_size)

        # 2) Bottleneck time conditioning: add t-embedding to (h, c) once
        t_emb = self.time_mlp(self.mp(t))                          # (B, hidden)
        h0 = h_p + t_emb
        c0 = c_p + t_emb

        # 3) Decode future segment starting from (h0, c0)
        fut_in = feats[:, L-1:, :]                                 # (B, H, in_dim)
        fut_out, _ = self.lstm(fut_in, init_state=(h0, c0))        # (B, H, hidden)

        # 4) Project to target space
        return self.head(fut_out)                                  # (B, H, 1)

    # ======================================================================= #
    #  Fast inference: past LSTM state caching                                 #
    # ======================================================================= #
    #
    #  KEY IDEA:
    #    During DDIM sampling, x_past / x_future / static_attr are FIXED across
    #    all num_steps iterations.  The past LSTM run (positions 0..L-2) therefore
    #    produces the same (h_p, c_p) every step — recomputing it is pure waste.
    #
    #    encode_past_lstm()  : run LSTM over past once, cache (h_p, c_p).
    #    decode_future_lstm(): add t_emb to cached state, run H future steps.
    #
    #    Training is completely unchanged — forward() is used as-is.
    # ======================================================================= #

    @torch.no_grad()
    def encode_past_lstm(self, x_past, x_future, static_attr):
        """
        Encode the fixed past (positions 0..L-2) and return the LSTM state.
        Call this ONCE before the DDIM loop; pass the result to decode_future_lstm.

        Parameters
        ----------
        x_past      : (B, L, d_input)
        x_future    : (B, H-1, d_input)   — needed only for static_attr shape; met values unused
        static_attr : (B, static_dim) or (B, 1, static_dim)

        Returns
        -------
        h_p, c_p : each (B, hidden_size)  — LSTM hidden and cell state after past
        """
        B, L, _ = x_past.shape
        device = x_past.device

        # Past features: positions 0..L-2, zero flow (noisy_future not yet known)
        past_met  = x_past[:, :-1, :]                                     # (B, L-1, d_input)
        past_flow = torch.zeros(B, L - 1, 1, device=device)

        if static_attr.dim() == 2:
            static_seq = static_attr.unsqueeze(1).expand(-1, L - 1, -1)
        else:
            static_seq = static_attr[:, 0, :].unsqueeze(1).expand(-1, L - 1, -1)

        past_feats = torch.cat([past_met, past_flow, static_seq], dim=-1)  # (B, L-1, in_dim)

        _, (h_p, c_p) = self.lstm(past_feats)                             # (B, hidden) each
        return h_p, c_p

    @torch.no_grad()
    def decode_future_lstm(self, x_past, x_future, static_attr,
                           noisy_future, t, h_p, c_p):
        """
        Decode H future positions using cached past LSTM state.
        Call this once per DDIM step.

        Parameters
        ----------
        x_past       : (B, L, d_input)
        x_future     : (B, H-1, d_input)
        static_attr  : (B, static_dim) or (B, 1, static_dim)
        noisy_future : (B, H, 1)          — changes every DDIM step
        t            : (B,)               — changes every DDIM step
        h_p, c_p     : cached past state from encode_past_lstm()

        Returns
        -------
        pred : (B, H, 1)
        """
        B, L, _ = x_past.shape
        H = self.H
        device = x_past.device

        # Future features: positions L-1..L+H-2
        fut_met  = torch.cat([x_past[:, -1:, :], x_future], dim=1)        # (B, H, d_input)

        if static_attr.dim() == 2:
            static_seq = static_attr.unsqueeze(1).expand(-1, H, -1)
        else:
            static_seq = static_attr[:, 0, :].unsqueeze(1).expand(-1, H, -1)

        fut_feats = torch.cat([fut_met, noisy_future, static_seq], dim=-1) # (B, H, in_dim)

        # Inject diffusion time into cached state (the only t-dependent operation)
        t_emb = self.time_mlp(self.mp(t))                                  # (B, hidden)
        h0 = h_p + t_emb
        c0 = c_p + t_emb

        fut_out, _ = self.lstm(fut_feats, init_state=(h0, c0))            # (B, H, hidden)
        return self.head(fut_out)                                          # (B, H, 1)

    @torch.no_grad()
    def sample_ddim_fast(
        self,
        x_past: torch.Tensor,
        static_attributes: torch.Tensor,
        future_pcp: torch.Tensor,
        num_steps: int = 10,
        eta: float = 0.0,
        ddpm_n_steps: int = None,
        ddpm_beta_start: float = 1e-4,
        ddpm_beta_end: float = 2e-2,
        past_state_cache: tuple = None,
    ) -> torch.Tensor:
        """
        Fast DDIM sampling with past LSTM state caching.

        Encode past once, then decode only H future steps per DDIM iteration.
        Produces identical output to sample_ddim() — caching is pure inference
        optimisation; training and model weights are unchanged.

        past_state_cache : optional (h_p, c_p) from encode_past_lstm().
            Pass this when generating multiple ensemble members for the same
            inputs to avoid recomputing the past state for every member.
        """
        device = x_past.device
        B, L, _ = x_past.shape
        H = future_pcp.size(1) + 1

        x = torch.randn(B, H, 1, device=device)

        # Encode past ONCE (or reuse provided cache)
        if past_state_cache is None:
            h_p, c_p = self.encode_past_lstm(x_past, future_pcp, static_attributes)
        else:
            h_p, c_p = past_state_cache

        # ------------------------------------------------------------------ #
        #  DDPM path
        # ------------------------------------------------------------------ #
        if ddpm_n_steps is not None:
            betas     = torch.linspace(ddpm_beta_start, ddpm_beta_end, ddpm_n_steps, device=device)
            alpha_bar = torch.cumprod(1.0 - betas, dim=0)
            stride    = ddpm_n_steps // num_steps

            for i in range(num_steps - 1, -1, -1):
                t_idx      = min(i * stride, ddpm_n_steps - 1)
                t_idx_prev = min((i - 1) * stride, ddpm_n_steps - 1) if i > 0 else None
                t_float    = torch.full((B,), t_idx / (ddpm_n_steps - 1), device=device)

                pred = self.decode_future_lstm(
                    x_past, future_pcp, static_attributes, x, t_float, h_p, c_p
                )

                ab_t = alpha_bar[t_idx]
                x0   = (x - (1.0 - ab_t).sqrt() * pred) / ab_t.sqrt().clamp(min=1e-8)

                if i > 0:
                    ab_prev = alpha_bar[t_idx_prev]
                    if eta > 0:
                        sigma_ddim = eta * torch.sqrt(
                            ((1 - ab_prev) / (1 - ab_t)) * (1 - ab_t / ab_prev)
                        ).clamp(min=0)
                        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * pred + sigma_ddim * torch.randn_like(x)
                    else:
                        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * pred
                else:
                    x = x0

        # ------------------------------------------------------------------ #
        #  Continuous cosine-logSNR path
        # ------------------------------------------------------------------ #
        else:
            ts = torch.linspace(0.0, 1.0, num_steps, device=device)

            for i in range(num_steps - 1, -1, -1):
                t      = ts[i].repeat(B)
                t_prev = ts[i - 1].repeat(B) if i > 0 else ts[0].repeat(B)

                pred = self.decode_future_lstm(
                    x_past, future_pcp, static_attributes, x, t, h_p, c_p
                )

                _, alpha_t,  sigma_t  = diffusion_params(t)
                _, alpha_tp, sigma_tp = diffusion_params(t_prev)

                alpha_t  = alpha_t.view(B, 1, 1);  sigma_t  = sigma_t.view(B, 1, 1)
                alpha_tp = alpha_tp.view(B, 1, 1);  sigma_tp = sigma_tp.view(B, 1, 1)

                x0  = alpha_t * x - sigma_t * pred
                eps = (pred + sigma_t * x0) / alpha_t

                if i > 0:
                    sigma = eta * torch.sqrt(
                        torch.clamp((sigma_tp**2) * (1 - (alpha_t**2 / alpha_tp**2)), min=1e-12)
                    )
                    noise = torch.randn_like(x) if eta > 0 else 0.0
                    x = alpha_tp * x0 + sigma_tp * eps + sigma * noise
                else:
                    x = x0

        return x.squeeze(-1)

    def _sync(self):
        """CUDA synchronise for accurate wall-clock GPU timing."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
    @torch.no_grad()
    def forward_profiled(self, x_past, noisy_future, t, x_future, static_attr, device):
        """Granular sub-step timing matching the SSM breakdown."""
        import time
        B, L, _ = x_past.shape
        H = self.H
        times = {}
    
        def tic():
            if device.type == 'cuda': torch.cuda.synchronize()
            return time.perf_counter()
        def toc(k, t0):
            if device.type == 'cuda': torch.cuda.synchronize()
            times[k] = times.get(k, 0.0) + (time.perf_counter() - t0)
    
        t0 = tic()
        feats, _ = self._build_sequence(x_past, noisy_future, x_future, static_attr)
        toc('lstm_feature_build', t0)
    
        t0 = tic()
        t_emb = self.time_mlp(self.mp(t))
        toc('lstm_time_embedding', t0)
    
        t0 = tic()
        _, (h_p, c_p) = self.lstm(feats[:, :L-1, :])
        toc('lstm_past_encode', t0)        # ? this is what replaces SSM's s4d_fft_conv on the past
    
        t0 = tic()
        h0, c0 = h_p + t_emb, c_p + t_emb
        fut_out, _ = self.lstm(feats[:, L-1:, :], init_state=(h0, c0))
        toc('lstm_future_decode', t0)      # ? sequential, cannot be parallelised
    
        t0 = tic()
        pred = self.head(fut_out)
        toc('lstm_head', t0)
    
        return pred[:, -H:, :], times


    @torch.no_grad()
    def sample_ddim(
        self,
        x_past: torch.Tensor,                 # (B, L, d_input)
        static_attributes: torch.Tensor,      # (B, static_dim) or (B,1,static_dim)
        future_pcp: torch.Tensor,             # (B, H-1, d_input)
        num_steps: int = 10,
        eta: float = 0.0,
        ddpm_n_steps: int = None,
        ddpm_beta_start: float = 1e-4,
        ddpm_beta_end: float = 2e-2,
    ) -> torch.Tensor:
        """
        DDIM sampling. Two schedule modes:

        DDPM mode  (ddpm_n_steps is not None):
          Uses the same linear beta schedule as train_ddpm.py -- matches DRUM exactly.
          Stride-based subsampling: stride = ddpm_n_steps // num_steps.
          Correct x0 recovery: x0 = (x_t - sqrt(1-ab_t)*eps) / sqrt(ab_t)

        Continuous mode (default, ddpm_n_steps is None):
          Uses the cosine logSNR schedule. Matches train_generic.py exactly.
          Velocity parameterization: x0 = alpha*x_t - sigma*v
        """
        device = x_past.device
        B, L, _ = x_past.shape
        H = future_pcp.size(1) + 1
        x = torch.randn(B, H, 1, device=device)

        # ------------------------------------------------------------------ #
        #  DDPM path: linear beta schedule, noise prediction                 #
        # ------------------------------------------------------------------ #
        if ddpm_n_steps is not None:
            betas     = torch.linspace(ddpm_beta_start, ddpm_beta_end, ddpm_n_steps, device=device)
            alpha_bar = torch.cumprod(1.0 - betas, dim=0)   # (T,)
            stride    = ddpm_n_steps // num_steps

            for i in range(num_steps - 1, -1, -1):
                t_idx      = min(i * stride, ddpm_n_steps - 1)
                t_idx_prev = min((i - 1) * stride, ddpm_n_steps - 1) if i > 0 else None

                # Normalize t the same way training does: t_int / (n_steps - 1)
                t_float = torch.full((B,), t_idx / (ddpm_n_steps - 1), device=device)

                pred = self.forward(
                    x_past       = x_past,
                    noisy_future = x,
                    t            = t_float,
                    x_future     = future_pcp,
                    static_attr  = static_attributes
                )

                # noise prediction
                ab_t     = alpha_bar[t_idx]
                x0       = (x - (1.0 - ab_t).sqrt() * pred) / ab_t.sqrt().clamp(min=1e-8)
                eps_pred = pred   # model output IS the noise estimate

                if i > 0:
                    ab_prev = alpha_bar[t_idx_prev]
                    if eta > 0:
                        sigma_ddim = eta * torch.sqrt(
                            ((1 - ab_prev) / (1 - ab_t)) * (1 - ab_t / ab_prev)
                        ).clamp(min=0)
                        noise = torch.randn_like(x)
                        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps_pred + sigma_ddim * noise
                    else:
                        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps_pred
                else:
                    x = x0

        # ------------------------------------------------------------------ #
        #  Continuous path: cosine logSNR schedule (train_generic.py style)  #
        # ------------------------------------------------------------------ #
        else:
            ts = torch.linspace(0., 1., num_steps, device=device)

            for i in range(num_steps - 1, -1, -1):
                t_cur  = ts[i].repeat(B)
                t_prev = ts[i - 1].repeat(B) if i > 0 else ts[0].repeat(B)

                pred = self.forward(
                    x_past       = x_past,
                    noisy_future = x,
                    t            = t_cur,
                    x_future     = future_pcp,
                    static_attr  = static_attributes
                )

                _, alpha_t,  sigma_t  = diffusion_params(t_cur)
                _, alpha_tp, sigma_tp = diffusion_params(t_prev)

                alpha_t  = alpha_t.view(B, 1, 1)
                sigma_t  = sigma_t.view(B, 1, 1)
                alpha_tp = alpha_tp.view(B, 1, 1)
                sigma_tp = sigma_tp.view(B, 1, 1)

                # velocity parameterization
                x0  = alpha_t * x - sigma_t * pred
                eps = (pred + sigma_t * x0) / alpha_t

                if i > 0:
                    sigma = eta * torch.sqrt(
                        torch.clamp((sigma_tp**2) * (1 - (alpha_t**2 / alpha_tp**2)), min=1e-12)
                    )
                    noise = torch.randn_like(x) if eta > 0 else 0.0
                    x = alpha_tp * x0 + sigma_tp * eps + sigma * noise
                else:
                    x = x0

        return x.squeeze(-1)


    @torch.no_grad()
    def sample_ddim_fast_fft_profiled(
        self,
        x_past: torch.Tensor,
        static_attributes: torch.Tensor,
        future_pcp: torch.Tensor,
        num_steps: int = 10,
        eta: float = 0.0,
        ddpm_n_steps: int = None,
        ddpm_beta_start: float = 1e-4,
        ddpm_beta_end: float = 2e-2,
    ):
        device = x_past.device
        B, L, _ = x_past.shape
        H = future_pcp.size(1) + 1

        profiler = RuntimeProfiler(use_cuda=(device.type == "cuda"))
        t_total = profiler.start()

        t0 = profiler.start()
        x = torch.randn(B, H, 1, device=device)
        profiler.stop("initialization", t0)

        # ------------------------------------------------------------------ #
        #  DDPM path: linear beta schedule, noise prediction (DRUM-style)    #
        # ------------------------------------------------------------------ #
        if ddpm_n_steps is not None:
            betas     = torch.linspace(ddpm_beta_start, ddpm_beta_end, ddpm_n_steps, device=device)
            alpha_bar = torch.cumprod(1.0 - betas, dim=0)
            stride    = ddpm_n_steps // num_steps

            for i in range(num_steps - 1, -1, -1):
                t_idx      = min(i * stride, ddpm_n_steps - 1)
                t_idx_prev = min((i - 1) * stride, ddpm_n_steps - 1) if i > 0 else None
                t_float    = torch.full((B,), t_idx / (ddpm_n_steps - 1), device=device)

                t0 = profiler.start()
                pred, fwd_times = self.forward_profiled(
                    x_past=x_past, noisy_future=x, t=t_cur,
                    x_future=future_pcp, static_attr=static_attributes,
                    device=device
                )
                profiler.stop("forward", t0)
                for k, v in fwd_times.items():       
                    profiler.times[k] += v
                    profiler.counts[k] += 1

                t0 = profiler.start()
                ab_t     = alpha_bar[t_idx]
                x0       = (x - (1.0 - ab_t).sqrt() * pred) / ab_t.sqrt().clamp(min=1e-8)
                eps_pred = pred

                if i > 0:
                    ab_prev = alpha_bar[t_idx_prev]
                    if eta > 0:
                        sigma_ddim = eta * torch.sqrt(
                            ((1 - ab_prev) / (1 - ab_t)) * (1 - ab_t / ab_prev)
                        ).clamp(min=0)
                        noise = torch.randn_like(x)
                        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps_pred + sigma_ddim * noise
                    else:
                        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps_pred
                else:
                    x = x0
                profiler.stop("update", t0)

        # ------------------------------------------------------------------ #
        #  Continuous path: cosine logSNR schedule (train_generic.py style)  #
        # ------------------------------------------------------------------ #
        else:
            ts = torch.linspace(0., 1., num_steps, device=device)

            for i in range(num_steps - 1, -1, -1):
                t0 = profiler.start()
                t_cur  = ts[i].repeat(B)
                t_prev = ts[i - 1].repeat(B) if i > 0 else ts[0].repeat(B)
                profiler.stop("conditioning", t0)

                t0 = profiler.start()
                pred, fwd_times = self.forward_profiled(
                    x_past=x_past, noisy_future=x, t=t_cur,
                    x_future=future_pcp, static_attr=static_attributes,device=device
                )
                profiler.stop("forward", t0)
                for k, v in fwd_times.items():
                    if isinstance(v, (int, float)):
                        profiler.times[k] = profiler.times.get(k, 0.0) + v
                        profiler.counts[k] = profiler.counts.get(k, 0) + 1
                    else:
                        profiler.metadata[k] = v

                t0 = profiler.start()
                _, alpha_t,  sigma_t  = diffusion_params(t_cur)
                _, alpha_tp, sigma_tp = diffusion_params(t_prev)

                alpha_t  = alpha_t.view(B, 1, 1)
                sigma_t  = sigma_t.view(B, 1, 1)
                alpha_tp = alpha_tp.view(B, 1, 1)
                sigma_tp = sigma_tp.view(B, 1, 1)

                x0  = alpha_t * x - sigma_t * pred
                eps = (pred + sigma_t * x0) / alpha_t

                if i > 0:
                    sigma = eta * torch.sqrt(
                        torch.clamp((sigma_tp**2) * (1 - (alpha_t**2 / alpha_tp**2)), min=1e-12)
                    )
                    noise = torch.randn_like(x) if eta > 0 else 0.0
                    x = alpha_tp * x0 + sigma_tp * eps + sigma * noise
                else:
                    x = x0
                profiler.stop("update", t0)

        profiler.stop("total", t_total)
        return x.squeeze(-1), profiler.summary()

