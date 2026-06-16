import torch
import torch.nn as nn
from models.s4.s4d_optimized import S4D as LTI
import pdb
# ---------------------------------------------------------------------------
# Dropout selection (as before)
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
elif tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

class ssm_v2(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        n_layers: int,
        cfg: dict,
        *,
        time_emb_dim: int = 256,
        enc_hidden_dim: int = 256,
        static_dim: int = 27,
        dropout: float = 0.15,
        time_full: bool   = False,
        enc_full: bool    = False,
        static_full: bool = False
    ):
        super().__init__()
        self.d_model     = d_model
        self.time_full   = time_full
        self.enc_full    = enc_full
        self.static_full = static_full


        # 1) projections / embeddings
        self.input_proj = nn.Linear(d_input, d_model)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim*2),
            nn.SiLU(),
            nn.Linear(time_emb_dim*2, d_model),
        )
        
        self.enc_mlp = nn.Sequential(
            nn.Linear(enc_hidden_dim, enc_hidden_dim*2),
            nn.SiLU(),
            nn.Linear(enc_hidden_dim*2, d_model),
        )
        
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, static_dim*2),
            nn.SiLU(),
            nn.Linear(static_dim*2, d_model),
        )
        

        # residual S4D stack
        self.blocks, self.norms, self.drops = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(
                LTI(d_model,
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

        # read-out head
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.SiLU(),
            nn.Linear(d_model//2, 1),
        )
        
        #self.head = nn.Linear(d_model, 1)
        
    def forward(self,
        x_past: torch.Tensor,    # (B, L, d_input)
        t:      torch.Tensor,    # (B,)
        h_enc:  torch.Tensor,    # (B,enc_hidden) or (layers,B,enc_hidden)
        future_pcp:   torch.Tensor, # (B, H, 1)
        static_attr:  torch.Tensor, # (B, static_dim)
    ) -> torch.Tensor:
        B, L, _ = x_past.shape
        H       = future_pcp.shape[1]

        # 1) project your history
        h = self.input_proj(x_past)

        # helper to build initial-only vs full biases
        def make_bias(emb: torch.Tensor, full: bool):
            # emb: (B, d_model)
            if full:
                return emb.unsqueeze(1).expand(-1, L, -1)
            else:
                bias = torch.zeros(B, L, self.d_model, device=h.device, dtype=h.dtype)
                if emb.dim() == 2:
                    bias[:, 0, :] = emb
                else:
                    bias = emb
                return bias

        # 2a) time bias
        t_emb     = self.time_mlp(t)           # (B, d_model)
        time_bias = make_bias(t_emb, self.time_full)

        # 2b) encoder-state bias
        ''' comment out for cross attention
        if h_enc.dim() == 3:
            h_enc = h_enc[0]
        '''
        
        enc_emb   = self.enc_mlp(h_enc)        # (B, d_model)
        enc_bias  = make_bias(enc_emb, self.enc_full)

        # 2c) static bias
        static_emb  = self.static_mlp(static_attr)  # (B, d_model)
        static_bias = make_bias(static_emb, self.static_full)
        static_bias = 0 # todo! test static concatenation!!

        # 2d) precipitation bias
        # future_pcp: (B, H, 1)
        #pcp_bias = future_pcp.expand(-1, -1, self.d_model)  # (B, H, d_model)
        pcp_bias = 0 # todo! test pcp concatenation!

        # 3) add biases
        h = h + time_bias + enc_bias + static_bias + pcp_bias

        # 4) S4D residual stack
        h = h.transpose(1, 2)  # -> (B, d_model, L)
        for blk, norm, drop in zip(self.blocks, self.norms, self.drops):
            z, _ = blk(h)
            h    = norm(h + drop(z))

        # 5) read-out
        pred_out = self.head(h.transpose(1, 2)) # (B, L, 1)
        return pred_out
