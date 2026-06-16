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

class ssm_encoder(nn.Module):
    """
    SSM encoder producing:
      - h_seq: (B, L, d_model)
      - h_enc: (B, horizon, d_model)
    """
    def __init__(
        self,
        d_input: int,
        d_model: int,
        n_layers: int,
        cfg: dict,
        *,
        static_dim: int = 27,
        dropout: float = 0.1,
        pool_type: str = 'power',    # 'avg', 'power', 'attn', or 'cross'
        power_exponent: float = 2.0,
        n_heads: int = 4,
        horizon: int = 8                # **required** when using 'cross'
    ):
        super().__init__()
        self.d_model      = d_model
        self.pool_type    = pool_type
        self.power_exponent = power_exponent
        self.horizon = horizon

        # single Linear for dyn+static
        in_dim = d_input
        self.input_proj = nn.Linear(in_dim, d_model)

        ###############################
        # while not used, diffusoin ssm all aligned are trained with these weights initiated. 
        # For other models, comment these out for evaluateion!
        #self.head = nn.Sequential(
        #    nn.Linear(d_model, d_model//2),
        #    nn.SiLU(),
        #    nn.Linear(d_model//2, 1),
        #)
        ################################

        
        # S4D stack
        self.blocks = nn.ModuleList()
        self.norms  = nn.ModuleList()
        self.drops  = nn.ModuleList()
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

        # pooling heads
        if pool_type == 'attn':
            self.attn_scorer = nn.Linear(d_model, 1)
        elif pool_type == 'cross':
            # one query vector per forecast step
            self.pool_queries = nn.Parameter(torch.randn(horizon, d_model))
            self.cross_attn   = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=False
            )

    def forward(
        self,
        x: torch.Tensor,                # (B, L, d_input)
        init_state=None,                # unused
        static_attr: torch.Tensor=None  # (B, static_dim)
    ):
        B, L, _ = x.shape
        device  = x.device

        # 1) concat static
        if static_attr is not None:
            stat = static_attr.unsqueeze(1).expand(-1, L, -1)  # (B, L, static_dim)
            x    = torch.cat([x, stat], dim=-1)                # (B, L, d_input+static_dim)

        # 2) project
        h = self.input_proj(x)    # (B, L, d_model)

        # 3) S4D residual stack
        h = h.transpose(1,2)      # (B, d_model, L)
        for blk, norm, drop in zip(self.blocks, self.norms, self.drops):
            z, _ = blk(h)
            h = norm(h + drop(z))
        h_seq = h.transpose(1,2)  # (B, L, d_model)
        #pred_out = self.head(h.transpose(1, 2)) # (B, L, 1)

        # 4) pooling h_enc of shape (B, horizon, d_model)
        if self.pool_type == 'avg':
            pooled = h_seq.mean(dim=1)                   # (B, d_model)
            h_enc  = pooled.unsqueeze(1).expand(-1, self.horizon, -1)
        elif self.pool_type == 'power':
            idx = torch.arange(1, L+1, device=device, dtype=h_seq.dtype)
            w   = (idx / L)**self.power_exponent
            w   = w / w.sum()
            pooled = (h_seq * w.view(1, L, 1)).sum(dim=1)
            h_enc  = pooled.unsqueeze(1).expand(-1, self.horizon, -1)
        elif self.pool_type == 'attn':
            scores  = self.attn_scorer(h_seq)            # (B, L, 1)
            weights = torch.softmax(scores, dim=1)
            pooled  = (h_seq * weights).sum(dim=1)
            h_enc   = pooled.unsqueeze(1).expand(-1, self.horizon, -1)
        else:  # cross
            # prepare: seq_len × batch × dim
            h_t = h_seq.transpose(0,1)                   # (L, B, d_model)
            # queries: horizon × batch × dim
            q   = self.pool_queries.unsqueeze(1).expand(-1, B, -1)
            attn_out, _ = self.cross_attn(q, h_t, h_t)   # (horizon, B, d_model)
            h_enc = attn_out.transpose(0,1)              # (B, horizon, d_model)

        return h_seq, (h_enc, _)
