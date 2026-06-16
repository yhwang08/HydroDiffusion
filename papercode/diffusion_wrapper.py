import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusion_utils import diffusion_params
import pdb
from runtime_profiler import RuntimeProfiler

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


class EncoderDecoderDiffusionWrapper(nn.Module):
    """
    A generic diffusion model wrapper. 
    You supply any `encoder` and `decoder` modules that follow this API:

    1. encoder(x_past) -> (h_enc, c_enc) 
    2. decoder(x_t, init_state) -> (dec_out, new_state) 
          where new_state matches what encoder returned (h_dec, c_dec)

    The wrapper then:
      1. Runs the encoder once on (x_past).
      2. At each diffusion step t_cur, computes a time‐embedding (t_emb), projects it to the hidden‐space,
         and adds them to the encoder’s output state to form (h0,c0) or S0.
      3. Feeds only the noisy scalar x_t into the decoder each step,
         using the modified hidden‐state as the initial state.
      4. Projects the decoder’s output to a single velocity scalar.
    """
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 hidden_size: int,
                 time_emb_dim: int,
                 decoder_name: str,
                 prediction_type: str,
                 guidance_weight: float = 0.0):
        """
        Inputs:
          - encoder:   any module satisfying encoder(x_past) -> state (h_enc,c_enc)
          - decoder:   any module satisfying decoder(x_t, init_state) -> (dec_out, new_state)
          - hidden_size:   the dimensionality of the encoder/decoder hidden‐state
          - time_emb_dim:  how large you want the diffusion‐step embedding to be
        """
        super().__init__()

        self.encoder       = encoder
        self.decoder       = decoder
        self.hidden_size   = hidden_size
        self.time_emb_dim  = time_emb_dim
        self.decoder_name = decoder_name
        self.prediction_type = prediction_type
        self.guidance_weight = guidance_weight
        
        self.mp = MPFourier(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, hidden_size),
        )

        # Project time embedding to hidden_dim
        #self.proj_t_h = nn.Linear(time_emb_dim, hidden_size)
        #self.proj_t_c = nn.Linear(time_emb_dim, hidden_size)
              
        # ---------- *static* → hidden‑space bias  ----------------------- #

        self.proj_s_h = nn.Sequential(
            nn.Linear(27, 27 * 2),
            nn.SiLU(),
            nn.Linear(27 * 2, hidden_size)
            )
        self.proj_s_c = nn.Sequential(
            nn.Linear(27, 27 * 2),
            nn.SiLU(),
            nn.Linear(27 * 2, hidden_size)
            )
        
        # read out
        self.output_proj = nn.Linear(hidden_size, 1)

    def forward(self,
                x_past: torch.Tensor,
                x_t: torch.Tensor,
                t: torch.Tensor,
                future_pcp: torch.Tensor,
                static_attributes: torch.Tensor
               ) -> torch.Tensor:
        """
        One forward pass during TRAINING:
          - x_past: (B, seq_len, past_feat_dim) (should already contain static attribtues in x_past)
          - x_t: (B, H, 1) the “noised target” at diffusion‐time t
          - t: (B,) continuous diffusion times in [0,1]
          - future_pcp: (B, H, 1) 
          - static_attributes: (B, H, 27) 

        Returns:
          - pred:      (B, H, 1)  the predicted velocity for each step in the horizon
        """
        B, H, _ = x_t.shape

        # Run encoder
        _, encoded_state = self.encoder(x_past)

        # Build embedding
        # time embedding & projection to hidden_dim
        t_feats = self.mp(t) # (B, time_emb_dim)
        
        t_h = None
        t_c = None
        
        if self.decoder_name == 'ssm':
            # pass raw sin/cos to the SSM. It will project internally
            t_for_ssm = t_feats # (B ,time_emb_dim)
            t_emb = None
        else:
            t_emb   = self.time_mlp(t_feats)    # (B ,time_emb_dim)
            t_seq = t_emb.unsqueeze(1).repeat(1, H, 1)   # (B, H, time_emb_dim) to concat with x_t for lstm backbone test
            t_h = t_emb  # (B, hidden_size)
            t_c = t_emb  # (B, hidden_size)
                        
        # static_attributes: (B, H, S)
        s_seq = static_attributes.mean(dim=1)   # (B, S)    
        
        s_h = self.proj_s_h(s_seq)              # (B, Hid)
        s_c = self.proj_s_c(s_seq)              # (B, Hid)   

        
        if isinstance(encoded_state, tuple) and len(encoded_state) == 2:
            # LSTM‐encoder and ssm-encoder case:
            h_enc, c_enc = encoded_state
            # each of shape (num_layers, B, hidden_size)
            # Condition
            h_0 = h_enc # todo, remove t_emb from global bias for lstm decoder test
            c_0 = c_enc
            if (t_h is not None) and (t_c is not None):
                h_0 = h_enc + t_h.unsqueeze(0) 
                c_0 = c_enc + t_c.unsqueeze(0)
                
                #h_0 = h_enc + t_h.unsqueeze(0) + s_h.unsqueeze(0) # unsqueeze t_h gives (1, B, hidden_size)
                #c_0 = c_enc + t_c.unsqueeze(0) + s_c.unsqueeze(0)

            decoder_init_state = (h_0, c_0)
            

        # Run decoder with noisy target (x_t) and hidden states from the encoder
        if self.decoder_name == 'unet':
            pred = self.decoder(x_t, t_h, h_enc, future_pcp, static_attributes)
            
        elif self.decoder_name == 'lstm':
            dec_in  = torch.cat([x_t, future_pcp, static_attributes], dim=-1) # todo, test concatenation!!
            
            #dec_in  = torch.cat([x_t, future_pcp], dim=-1) #todo, test concatenation!
            
            dec_out, _ = self.decoder(dec_in, init_state=decoder_init_state)
            
            #dec_out, _ = self.decoder(x_t, decoder_init_state) # todo, not concatenation!
            pred = self.output_proj(dec_out)  # (B, H, 1)
            
        elif self.decoder_name == 'ssm':
            # x_t, pcp, and raw fourier features are length-H token groups
            pred = self.decoder(               # (B, H, 1)  velocity
                x_past = torch.cat([x_t, future_pcp, static_attributes], dim=-1), # todo!!! test concatenation! originally x_t!!!
                t = t_for_ssm,       # raw fourier   (B ,time_emb_dim) 
                h_enc = h_enc, 
                future_pcp = future_pcp,      # (B, H, 1)
                static_attr = static_attributes.mean(dim=1)  # (B, static_dim)
            )
        
        else:
            raise ValueError(f"Unknown decoder: {self.decoder_name}")

        return pred


    def _decode_step(
        self,
        x: torch.Tensor,
        t_float: torch.Tensor,
        encoded_state,
        future_pcp: torch.Tensor,
        static_attributes: torch.Tensor,
        H: int,
        B: int,
    ) -> torch.Tensor:
        """
        Shared single-step decoder call for both DDIM sampling paths.
          x:                 (B, H, 1)  noisy sample at current step
          t_float:           (B,)       normalized diffusion time in [0, 1]
          encoded_state:     encoder output (tuple of tensors)
        Returns pred: (B, H, 1)
        """
        t_feats = self.mp(t_float)

        t_h = None
        t_c = None
        if self.decoder_name == 'ssm':
            t_for_ssm = t_feats
        else:
            t_emb = self.time_mlp(t_feats)
            t_h   = t_emb
            t_c   = t_emb

        if isinstance(encoded_state, tuple) and len(encoded_state) == 2:
            h_enc, c_enc = encoded_state
            h_0 = h_enc
            c_0 = c_enc
            if (t_h is not None) and (t_c is not None):
                h_0 = h_enc + t_h.unsqueeze(0)
                c_0 = c_enc + t_c.unsqueeze(0)
            decoder_init_state = (h_0, c_0)

        if self.decoder_name == 'unet':
            pred = self.decoder(x, t_h, h_enc, future_pcp, static_attributes)
        elif self.decoder_name == 'lstm':
            dec_in  = torch.cat([x, future_pcp, static_attributes], dim=-1)
            dec_out, _ = self.decoder(dec_in, init_state=decoder_init_state)
            pred = self.output_proj(dec_out)
        elif self.decoder_name == 'ssm':
            pred = self.decoder(
                x_past      = torch.cat([x, future_pcp, static_attributes], dim=-1),
                t           = t_for_ssm,
                h_enc       = h_enc,
                future_pcp  = future_pcp,
                static_attr = static_attributes.mean(dim=1)
            )
        else:
            raise ValueError(f"Unknown decoder: {self.decoder_name}")
        return pred

    @torch.no_grad()
    def encode_past(self, x_past: torch.Tensor):
        """
        Run the encoder once on the (fixed) past sequence and return the
        resulting state. This state does not depend on the diffusion step or
        the ensemble member, so it can be computed ONCE per batch and reused
        across all DDIM steps and all ensemble members (avoids re-running the
        encoder LSTM/SSM over the full past sequence for every sample).
        """
        _, encoded_state = self.encoder(x_past)
        return encoded_state

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
                    encoded_state=None,
                   ) -> torch.Tensor:
        """
        DDIM reverse sampling. Two schedule modes:

        DDPM mode  (ddpm_n_steps is not None AND prediction_type == 'noise'):
          Uses the same linear beta schedule as train_ddpm.py -- matches DRUM exactly.
          Stride-based timestep subsampling: stride = ddpm_n_steps // num_steps.
          Correct x0 recovery: x0 = (x_t - sqrt(1-ab_t)*eps) / sqrt(ab_t)

        Continuous mode (default, prediction_type == 'velocity' or ddpm_n_steps is None):
          Uses the cosine logSNR schedule from diffusion_params().
          Matches train_generic.py exactly.

        encoded_state : optional, output of self.encode_past(x_past) computed
          ONCE outside the ensemble loop. If None, the encoder is run here
          (backward compatible with existing callers).
        """
        device = x_past.device
        B, seq_len, _ = x_past.shape
        H = future_pcp.size(1)
        if encoded_state is None:
            _, encoded_state = self.encoder(x_past)
        x = torch.randn(B, H, 1, device=device)

        # ------------------------------------------------------------------ #
        #  DDPM path: linear beta schedule, noise prediction (DRUM-style)    #
        # ------------------------------------------------------------------ #
        if ddpm_n_steps is not None and self.prediction_type == 'noise':
            betas     = torch.linspace(ddpm_beta_start, ddpm_beta_end, ddpm_n_steps, device=device)
            alpha_bar = torch.cumprod(1.0 - betas, dim=0)   # (T,)
            stride    = ddpm_n_steps // num_steps            # maps sampling steps -> training indices

            for i in range(num_steps - 1, -1, -1):
                t_idx      = min(i * stride, ddpm_n_steps - 1)
                t_idx_prev = min((i - 1) * stride, ddpm_n_steps - 1) if i > 0 else None

                # Normalize t the same way training does: t_int / (n_steps - 1)
                t_float = torch.full((B,), t_idx / (ddpm_n_steps - 1), device=device)

                pred = self._decode_step(x, t_float, encoded_state, future_pcp, static_attributes, H, B)

                # Correct x0 recovery for noise prediction (matches DRUM)
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
                    x = x0   # final step: return clean sample directly

        # ------------------------------------------------------------------ #
        #  Continuous path: cosine logSNR schedule (train_generic.py style)  #
        # ------------------------------------------------------------------ #
        else:
            ts = torch.linspace(0., 1., num_steps, device=device)

            for i in range(num_steps - 1, -1, -1):
                t      = ts[i].repeat(B)
                t_prev = ts[i - 1].repeat(B) if i > 0 else ts[0].repeat(B)

                pred = self._decode_step(x, t, encoded_state, future_pcp, static_attributes, H, B)

                _, alpha_t,  sigma_t  = diffusion_params(t)
                _, alpha_tp, sigma_tp = diffusion_params(t_prev)

                alpha_t  = alpha_t.view(B, 1, 1)
                sigma_t  = sigma_t.view(B, 1, 1)
                alpha_tp = alpha_tp.view(B, 1, 1)
                sigma_tp = sigma_tp.view(B, 1, 1)

                if self.prediction_type == 'noise':
                    # correct formula for noise prediction under cosine schedule
                    x0       = (x - sigma_t * pred) / alpha_t.clamp(min=1e-8)
                    eps_pred = pred
                else:  # velocity (default for train_generic.py)
                    x0       = alpha_t * x - sigma_t * pred
                    eps_pred = (pred + sigma_t * x0) / alpha_t

                if i > 0:
                    sigma = eta * torch.sqrt(
                        torch.clamp((sigma_tp**2) * (1 - (alpha_t**2) / (alpha_tp**2)), min=1e-12)
                    )
                    noise = torch.randn_like(x) if eta > 0 else 0.0
                    x = alpha_tp * x0 + sigma_tp * eps_pred + sigma * noise
                else:
                    x = x0

        return x.squeeze(-1)

 
    @torch.no_grad()
    def sample_ddim_profiled(
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
        """
        Profiled version of sample_ddim. Supports both DDPM and continuous schedule paths.
        For DDPM path (ddpm_n_steps is not None), profiling is at a coarser step level.
        For continuous path, conditioning / forward / update are timed separately.
        """
        device = x_past.device
        B, seq_len, _ = x_past.shape
        H = future_pcp.size(1)

        profiler = RuntimeProfiler(use_cuda=(device.type == "cuda"))
        t_total = profiler.start()

        t0 = profiler.start()
        _, encoded_state = self.encoder(x_past)
        profiler.stop("encoder_once", t0)

        t0 = profiler.start()
        x = torch.randn(B, H, 1, device=device)
        profiler.stop("initialization", t0)

        # ------------------------------------------------------------------ #
        #  DDPM path: linear beta schedule, noise prediction (DRUM-style)    #
        # ------------------------------------------------------------------ #
        if ddpm_n_steps is not None and self.prediction_type == 'noise':
            betas     = torch.linspace(ddpm_beta_start, ddpm_beta_end, ddpm_n_steps, device=device)
            alpha_bar = torch.cumprod(1.0 - betas, dim=0)
            stride    = ddpm_n_steps // num_steps

            for i in range(num_steps - 1, -1, -1):
                t_idx      = min(i * stride, ddpm_n_steps - 1)
                t_idx_prev = min((i - 1) * stride, ddpm_n_steps - 1) if i > 0 else None
                t_float    = torch.full((B,), t_idx / (ddpm_n_steps - 1), device=device)

                t0 = profiler.start()
                pred = self._decode_step(x, t_float, encoded_state, future_pcp, static_attributes, H, B)
                profiler.stop("forward", t0)

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

                t_feats = self.mp(t)
                t_h = None
                t_c = None
                if self.decoder_name == 'ssm':
                    t_for_ssm = t_feats
                else:
                    t_emb = self.time_mlp(t_feats)
                    t_h = t_emb
                    t_c = t_emb

                if isinstance(encoded_state, tuple) and len(encoded_state) == 2:
                    h_enc, c_enc = encoded_state
                    h_0 = h_enc
                    c_0 = c_enc
                    if (t_h is not None) and (t_c is not None):
                        h_0 = h_enc + t_h.unsqueeze(0)
                        c_0 = c_enc + t_c.unsqueeze(0)
                    decoder_init_state = (h_0, c_0)
                profiler.stop("conditioning", t0)

                t0 = profiler.start()
                if self.decoder_name == 'unet':
                    pred = self.decoder(x, t_h, h_enc, future_pcp, static_attributes)
                elif self.decoder_name == 'lstm':
                    dec_in = torch.cat([x, future_pcp, static_attributes], dim=-1)
                    dec_out, _ = self.decoder(dec_in, init_state=decoder_init_state)
                    pred = self.output_proj(dec_out)
                elif self.decoder_name == 'ssm':
                    pred = self.decoder(
                        x_past=torch.cat([x, future_pcp, static_attributes], dim=-1),
                        t=t_for_ssm,
                        h_enc=h_enc,
                        future_pcp=future_pcp,
                        static_attr=static_attributes.mean(dim=1)
                    )
                else:
                    raise ValueError(f"Unknown decoder: {self.decoder_name}")
                profiler.stop("forward", t0)

                t0 = profiler.start()
                _, alpha_t,  sigma_t  = diffusion_params(t)
                _, alpha_tp, sigma_tp = diffusion_params(t_prev)

                alpha_t  = alpha_t.view(B, 1, 1)
                sigma_t  = sigma_t.view(B, 1, 1)
                alpha_tp = alpha_tp.view(B, 1, 1)
                sigma_tp = sigma_tp.view(B, 1, 1)

                if self.prediction_type == 'noise':
                    # correct formula for noise prediction under cosine schedule
                    x0       = (x - sigma_t * pred) / alpha_t.clamp(min=1e-8)
                    eps_pred = pred
                else:  # velocity
                    x0       = alpha_t * x - sigma_t * pred
                    eps_pred = (pred + sigma_t * x0) / alpha_t

                if i > 0:
                    sigma = eta * torch.sqrt(
                        torch.clamp(
                            (sigma_tp ** 2) * (1 - (alpha_t ** 2) / (alpha_tp ** 2)),
                            min=1e-12
                        )
                    )
                    noise = torch.randn_like(x) if eta > 0 else 0.0
                    x = alpha_tp * x0 + sigma_tp * eps_pred + sigma * noise
                else:
                    x = x0
                profiler.stop("update", t0)

        profiler.stop("total", t_total)
        return x.squeeze(-1), profiler.summary()
