import torch
import torch.nn as nn
from models.s4.s4d import S4D as LTI
import pdb
from diffusion_utils import diffusion_params
import math
from runtime_profiler import RuntimeProfiler
import time

# Determine correct Dropout function based on PyTorch version
dropout_fn = nn.Dropout1d if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12) else nn.Dropout

class MPFourier(nn.Module):
    def __init__(self, num_channels, bandwidth=1.0):
        super().__init__()
        self.register_buffer('freqs', 2*math.pi*torch.randn(num_channels))
        self.register_buffer('phases', 2*math.pi*torch.rand(num_channels))

    def forward(self, t):
        # t: (B,) long/int or float tensor of diffusion step indices
        t = t.to(torch.float32)
        x = t[:, None]*self.freqs[None,:] + self.phases[None,:]
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
        self.d_model   = d_model
        self.H         = horizon
        self.static_dim = static_dim
        
        # 1) project raw input (met + noisy flow + static) to state dim
        self.input_proj = nn.Linear(d_input + 1, d_model)

        # 2) time embedding projector -> bias
        self.mp = MPFourier(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, d_model),
        )
        self.time_full = True

        # 3) S4D residual stack
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

        # 4) final head: map state to noise prediction
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def forward(
        self,
        x_past:       torch.Tensor,  # (B, L, d_input)
        noisy_future: torch.Tensor,  # (B, H, 1)
        t:     torch.Tensor,  # (B, H, 1)
        x_future:     torch.Tensor,  # (B, H-1, d_input), -1 is for excluding the nowcast day
        static_attr:  torch.Tensor,  # (B, static_dim)
    ) -> torch.Tensor:
        B, L, _ = x_past.shape
        H        = self.H
        device   = x_past.device
        
        t_feats = self.mp(t) # (B, time_emb_dim)

        # Build feature sequence: met & flow & static
        all_met    = torch.cat([x_past, x_future], dim=1)               # (B, L+H-1, d_input)
        pad_flow   = torch.zeros(B, L-1, 1, device=device)
        all_flow   = torch.cat([pad_flow, noisy_future], dim=1)         # (B, L+H-1, 1)
        static_seq = static_attr[:,0,:].unsqueeze(1).expand(-1, L+H-1, -1)   
        feats = torch.cat([all_met, all_flow, static_seq], dim=-1) # (B, L+H-1, d_input+1)

        # Project to state dimension
        h = self.input_proj(feats)  # (B, L+H, d_model)
        
        # Time bias: apply embedding at nowcast index L-1 or full horizon
        t_b = self.time_mlp(t_feats)  # (B, d_model)
        time_bias = torch.zeros_like(h)  # (B, L+H, d_model)
        if self.time_full:
            # broadcast the time embedding across all forecast steps (positions L-1 to L+H-1)
            time_bias[:, L-1:, :] = t_b.unsqueeze(1).expand(-1, self.H, -1)
        else:
            time_bias[:, L-1, :] = t_b
        h = h + time_bias
        
        # Residual S4D stack
        h = h.transpose(1, 2)  # (B, d_model, L+H)
        for blk, norm, drop in zip(self.blocks, self.norms, self.drops):
            z, _ = blk(h)
            h    = norm(h + drop(z))
        h = h.transpose(1, 2)  # (B, L+H, d_model)

        # Predict only the future window
        out = self.head(h)          # (B, L+H, 1)
        return out[:, -H:, :]       # (B, H, 1)
        
    @torch.no_grad()
    def forward_profiled(
        self,
        x_past: torch.Tensor,
        noisy_future: torch.Tensor,
        t: torch.Tensor,
        x_future: torch.Tensor,
        static_attr: torch.Tensor,
    ):
        times = {}
    
        def tic():
            self._sync()
            return time.perf_counter()
    
        def toc(name, t0):
            self._sync()
            times[name] = times.get(name, 0.0) + (time.perf_counter() - t0)
    
        B, L, _ = x_past.shape
        H = self.H
        device = x_past.device
    
        t0 = tic()
        t_feats = self.mp(t)
        toc("forward_time_fourier", t0)
    
        t0 = tic()
        all_met = torch.cat([x_past, x_future], dim=1)
        pad_flow = torch.zeros(B, L - 1, 1, device=device)
        all_flow = torch.cat([pad_flow, noisy_future], dim=1)
        static_seq = static_attr[:, 0, :].unsqueeze(1).expand(-1, L + H - 1, -1)
        feats = torch.cat([all_met, all_flow, static_seq], dim=-1)
        toc("forward_feature_build", t0)
    
        t0 = tic()
        h = self.input_proj(feats)
        toc("forward_input_proj", t0)
    
        t0 = tic()
        t_b = self.time_mlp(t_feats)
        time_bias = torch.zeros_like(h)
        if self.time_full:
            time_bias[:, L - 1:, :] = t_b.unsqueeze(1).expand(-1, self.H, -1)
        else:
            time_bias[:, L - 1, :] = t_b
        h = h + time_bias
        toc("forward_time_bias", t0)
    
        t0 = tic()
        h = h.transpose(1, 2)
        toc("forward_transpose_in", t0)
    
        ssm_blk_total = 0.0
        ssm_resnorm_total = 0.0
    
        for li, (blk, norm, drop) in enumerate(zip(self.blocks, self.norms, self.drops)):
            # profile the S4D block itself
            t1 = tic()
            z, _, blk_times = blk.forward_profiled(h)
            dt = time.perf_counter() - t1
            self._sync()
            times[f"forward_layer{li+1}_blk"] = times.get(f"forward_layer{li+1}_blk", 0.0) + dt
            ssm_blk_total += dt
            
            for key, sub_time in blk_times.items():
                if isinstance(sub_time, (int, float)):
                    times[key] = times.get(key, 0.0) + sub_time
                else:
                    times[key] = sub_time
        
            # residual + norm + dropout wrapper
            t1 = tic()
            h = norm(h + drop(z))
            dt = time.perf_counter() - t1
            self._sync()
            times[f"forward_layer{li+1}_resnorm"] = times.get(f"forward_layer{li+1}_resnorm", 0.0) + dt
            ssm_resnorm_total += dt
    
        times["forward_ssm_blk_total"] = times.get("forward_ssm_blk_total", 0.0) + ssm_blk_total
        times["forward_ssm_resnorm_total"] = times.get("forward_ssm_resnorm_total", 0.0) + ssm_resnorm_total
    
        t0 = tic()
        h = h.transpose(1, 2)
        toc("forward_transpose_out", t0)
    
        t0 = tic()
        out = self.head(h)
        out = out[:, -H:, :]
        toc("forward_head", t0)
    
        return out, times

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
                    x_past=x_past,
                    noisy_future=x,
                    t=t_float,
                    x_future=future_pcp,
                    static_attr=static_attributes
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
                t      = ts[i].repeat(B)
                t_prev = ts[i - 1].repeat(B) if i > 0 else ts[0].repeat(B)

                pred = self.forward(
                    x_past=x_past,
                    noisy_future=x,
                    t=t,
                    x_future=future_pcp,
                    static_attr=static_attributes
                )

                _, alpha_t,  sigma_t  = diffusion_params(t)
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
    def sample_ddim_profiled(self,
                    x_past: torch.Tensor,           # (B, L, d_input)
                    static_attributes: torch.Tensor,      # (B, S)
                    future_pcp: torch.Tensor,       # (B, H-1, forcing_dim[5, 15])
                    num_steps: int = 10,
                    eta: float = 0.0,
                    ddpm_n_steps: int = None,
                    ddpm_beta_start: float = 1e-4,
                    ddpm_beta_end: float = 2e-2,
                   ) -> torch.Tensor:
        """
        DDIM sampling for the decoder_only_ssm model with profiling.
        Supports both DDPM and continuous schedule modes.
        """
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
                pred, forward_times = self.forward_profiled(
                    x_past=x_past, noisy_future=x, t=t_float,
                    x_future=future_pcp, static_attr=static_attributes
                )
                profiler.stop("forward", t0)
                for k, v in forward_times.items():
                    if isinstance(v, (int, float)):
                        profiler.times[k] += v
                        profiler.counts[k] += 1
                    else:
                        profiler.metadata[k] = v

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
                t      = ts[i].repeat(B)
                t_prev = ts[i - 1].repeat(B) if i > 0 else ts[0].repeat(B)
                profiler.stop("conditioning", t0)

                t0 = profiler.start()
                pred, forward_times = self.forward_profiled(
                    x_past=x_past, noisy_future=x, t=t,
                    x_future=future_pcp, static_attr=static_attributes
                )
                profiler.stop("forward", t0)
                for k, v in forward_times.items():
                    if isinstance(v, (int, float)):
                        profiler.times[k] += v
                        profiler.counts[k] += 1
                    else:
                        profiler.metadata[k] = v

                t0 = profiler.start()
                _, alpha_t,  sigma_t  = diffusion_params(t)
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