# papercode/zeroshot_test.py
# Batched, vectorized zero-shot evaluation with GEFS forcings.
# Produces per-basin CSV (ensemble mean) + a global NPZ with full ensemble cube (N,S,H).

'''
python -m papercode.zeroshot_test \
  --model_ckpt /data/home/yihan/diffusion_ssm/runs/run_2204_0533_seed3407/model_epoch60.pt \
  --ddpm_train_steps 1000 \
  --ddim_steps 10 \
  --note gefs_zst_ddpmtrain_ddim10
'''

# Batched, vectorized zero-shot evaluation with GEFS forcings.
# Produces per-basin CSV (ensemble mean) + a global NPZ with full ensemble cube (N,S,H).

from __future__ import annotations
import os, sys, argparse, logging, time, traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union, List, Tuple
import pdb
import json
from collections import defaultdict
import pdb


# ----------------- early GPU selection (before importing torch) ---------------
def _early_parse():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpu", type=str, default=None)
    args, _ = p.parse_known_args()
    if args.gpu is not None:
        if args.gpu.lower() == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args
_EARLY = _early_parse()

# Make repo imports work no matter where we run from
THIS = Path(__file__).resolve()
PKG = THIS.parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG))

import numpy as np
import pandas as pd
import sqlite3
import torch

from papercode.datautils import (
    load_forcing_multi, load_scalar, normalize_multi_features, load_discharge
)
from papercode.utils import get_basin_list

# Import is deferred to after argument parsing so --ssm_backend controls which
# version is loaded. Default is 'hybrid' (fastest); pass --ssm_backend fast to
# use the FFT-only version from decoder_only_ssm_fast.

# --------------------------- logging -----------------------------------------
def setup_logging(log_dir: str, note: str, level: str) -> Path:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{note}" if note else ""
    log_file = Path(log_dir) / f"zeroshot_{ts}{suffix}.log"

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return log_file

# --------------------------- CLI ---------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser()
    # paths (defaults provided)
    p.add_argument("--model_ckpt",   type=str, required=True)
    p.add_argument("--camels_root",  type=str,
                   default="/data/rdl/yihan/data/basin_dataset_public_v1p2")
    p.add_argument("--attr_db",      type=str,
                   default="/data/home/yihan/diffusion_ssm/runs/shared_h5_new/attributes.db")
    p.add_argument("--gefs_dir",     type=str,
                   default="/data/rdl/yihan/GEFS_forecasts")
    p.add_argument("--scalar",       type=str,
                   default="/home/yihan/diffusion_ssm/global_scalar.json")
    p.add_argument("--out_dir",      type=str, default="/data/home/yihan/diffusion_ssm/zeroshot_outputs")

    # logging & run control
    p.add_argument("--gpu",          type=str, default=_EARLY.gpu)
    p.add_argument("--log_dir",      type=str, default="reports")
    p.add_argument("--note",         type=str, default="")
    p.add_argument("--log_level",    type=str, default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])

    # time window & batching
    p.add_argument("--lookback",     type=int, default=365)
    p.add_argument("--horizon",      type=int, default=8)  # nowcast+7
    p.add_argument("--batch_inits",  type=int, default=512,
                   help="How many init times to process per forward pass.")
    p.add_argument("--date_start",   type=str, default=None,
                   help="Optional YYYY-MM-DD to limit processed init range.")
    p.add_argument("--date_end",     type=str, default=None,
                   help="Optional YYYY-MM-DD to limit processed init range.")
    p.add_argument("--num_workers",  type=int, default=0)

    # ensemble sampling (diffusion)
    p.add_argument("--num_samples",  type=int, default=50)
    p.add_argument("--ddim_steps",   type=int, default=10) # todo
    p.add_argument("--eta",          type=float, default=0.0)

    # model cfg that must match training
    p.add_argument("--d_model",      type=int, default=256)
    p.add_argument("--d_state",      type=int, default=256)
    p.add_argument("--n_layers",     type=int, default=6)
    p.add_argument("--time_emb_dim", type=int, default=256)
    p.add_argument("--ssm_dropout",  type=float, default=0.2)
    p.add_argument("--pool_type",    type=str, default="power")
    p.add_argument("--predict_mode", type=str, default="velocity")

    # SSM train cfg values needed by the model init
    p.add_argument("--lr",           type=float, default=3e-5)
    p.add_argument("--lr_min",       type=float, default=3e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--wd",           type=float, default=4e-5)
    p.add_argument("--warmup",       type=int,   default=1)
    p.add_argument("--lr_dt",        type=float, default=1e-3)
    p.add_argument("--min_dt",       type=float, default=1e-2)
    p.add_argument("--max_dt",       type=float, default=1e-1)
    p.add_argument("--cfi",          type=float, default=10.0)
    p.add_argument("--cfr",          type=float, default=10.0)

    # data flags used at train time
    p.add_argument("--forcing",      type=str, default="daymet",
                   choices=["daymet","nldas","maurer","all"])
    p.add_argument("--no_static",    action="store_true", default=False)
    p.add_argument("--concat_static",action="store_true", default=True)

    # basin selection
    p.add_argument("--basins",       type=str, default=None,
                   help="Optional text file (one basin per line) or comma list")
                   
    # sampling for ddpm trained model
    p.add_argument('--ddpm_train_steps', type=int, default=None,
               help='If set, uses DDPM linear beta schedule at sampling time (for DDPM-trained models)')
    p.add_argument('--ddpm_beta_start', type=float, default=1e-4)
    p.add_argument('--ddpm_beta_end',   type=float, default=2e-2)

    p.add_argument('--ssm_backend', type=str, default='hybrid',
                   choices=['hybrid', 'fast'],
                   help='"hybrid" = FFT encode + recurrent decode (fastest); '
                        '"fast"   = FFT encode + FFT decode (decoder_only_ssm_fast)')
    return p

# --------------------------- helpers -----------------------------------------
TRI_COLUMNS = [
    'prcp_nldas','prcp_maurer','prcp_daymet',
    'srad_nldas','srad_maurer','srad_daymet',
    'tmax_nldas','tmax_maurer','tmax_daymet',
    'tmin_nldas','tmin_maurer','tmin_daymet',
    'vp_nldas','vp_maurer','vp_daymet'
]
IDX_MAP = {'nldas':[0,3,6,9,12], 'maurer':[1,4,7,10,13], 'daymet':[2,5,8,11,14]}
STD_RENAME = {
    'prcp(mm/day)':'prcp','PRCP(mm/day)':'prcp',
    'srad(W/m2)':'srad','SRAD(W/m2)':'srad','srad(w/m2)':'srad',
    'tmax(C)':'tmax','Tmax(C)':'tmax','tmax(c)':'tmax',
    'tmin(C)':'tmin','Tmin(C)':'tmin','tmin(c)':'tmin',
    'vp(Pa)':'vp','Vp(Pa)':'vp','vp(pa)':'vp'
}

def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: STD_RENAME.get(c, c) for c in df.columns})

def _init_time_cols(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(dict(year=df['Year'], month=df['Mnth'],
                               day=df['Day'], hour=df['Hr']))

def _read_gefs_file(path: Union[str,Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    df = _std_cols(df)
    req = {'Year','Mnth','Day','Hr','lead','prcp','srad','tmax','tmin','vp'}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"GEFS file missing columns: {miss}")
    df['init_time'] = _init_time_cols(df)

    
    df['valid_time'] = pd.to_datetime(dict(year=df['Year'], month=df['Mnth'], day=df['Day'], hour=df['Hr']))
    # daily GEFS: init = valid - (lead) days
    df['init_time']  = df['valid_time'] - pd.to_timedelta(df['lead'], unit='D')
    
    df = df[df['lead'].between(1,7)].copy()
    df = df.sort_values(['init_time','lead'])
    # drop duplicate (init_time, lead) pairs to avoid reindex error
    df = df.drop_duplicates(subset=['init_time','lead'], keep='last')
    return df

def _normalize_5(X5: np.ndarray, source: str, scalar: Dict) -> np.ndarray:
    cols  = [f'prcp_{source}', f'srad_{source}', f'tmax_{source}', f'tmin_{source}', f'vp_{source}']
    means = np.array([scalar[f'{c}_mean'] for c in cols], dtype=np.float32)
    stds  = np.array([scalar[f'{c}_std']  for c in cols], dtype=np.float32)
    stds[stds == 0.0] = 1.0
    out = (X5.astype(np.float32) - means) / stds
    return out.astype(np.float32)

def _denorm_q(y_norm: np.ndarray, scalar: Dict) -> np.ndarray:
    mu = np.float32(scalar["QObs(mm/d)_mean"])
    sd = np.float32(scalar["QObs(mm/d)_std"])
    return (y_norm.astype(np.float32) * sd + mu).astype(np.float32)

def _load_static_27(db_path: Path, basin: str) -> np.ndarray:
    conn = sqlite3.connect(str(db_path))
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';")]
    if not tables:
        conn.close()
        raise RuntimeError(f"No tables in {db_path}")
    tbl = tables[0]
    df = pd.read_sql(f"SELECT * FROM {tbl}", conn)
    conn.close()
    id_col = 'gauge_id' if 'gauge_id' in df.columns else df.columns[0]
    df.set_index(id_col, inplace=True)
    numeric = df.select_dtypes(include=[np.number]).fillna(0.0)
    mu = numeric.mean(); sd = numeric.std().replace(0.0, 1.0)
    z = (numeric - mu) / sd
    if basin not in z.index:
        raise KeyError(f"Basin {basin} not in {db_path}")
    return z.loc[basin].iloc[:27].values.astype(np.float32)  # (27,)

def _filter_inits_by_date(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start is not None:
        df = df[df['init_time'] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df['init_time'] <= pd.Timestamp(end)]
    return df

# --------------------------- alignment check ---------------------------------
def assert_alignment(gefs_df: pd.DataFrame, df15: pd.DataFrame, one_init: pd.Timestamp):
    """
    Print/verify:
      - Past window: D-364..D (length 365)
      - Future valid_time for leads 1..7: expected D+1..D+7
    """
    if one_init not in gefs_df['init_time'].unique():
        print(f"[assert_alignment] init {one_init} not found in GEFS DF.")
        return

    g = (gefs_df[gefs_df['init_time'] == one_init]
            .sort_values('lead')[['lead','valid_time']])
    D = pd.Timestamp(one_init.date())

    fut_dates = g['valid_time'].dt.normalize().tolist()
    exp_fut   = [D + pd.Timedelta(days=k) for k in range(1, 8)]  # D+1..D+7

    #print("\n[ALIGNMENT CHECK]")
    #print("init_time:", one_init)
    #print("GEFS valid_time (leads 1..7):", fut_dates)
    #print("Expected D+1..D+7:           ", exp_fut)

    # Past window (inclusive) D-364..D
    start = D - pd.Timedelta(days=364)
    end   = D
    pw = df15.loc[start:end].index.tolist()
    
    print(f"Past window len={len(pw)}  first={pw[0] if pw else None}  last={pw[-1] if pw else None}")
    
    # Hard asserts you can enable once you trust the prints:
    # assert fut_dates == exp_fut, "Future block is not D+1..D+7"
    # assert len(pw) == 365 and pw[0].normalize() == start and pw[-1].normalize() == end, "Past window mismatch"



# --------------------------- model builder -----------------------------------
def ensure_cfg_defaults(cfg: Dict) -> Dict:
    defaults = {
        "lr": 3e-5, "lr_min": 3e-6, "weight_decay": 0.0, "wd": 4e-5,
        "warmup": 1, "lr_dt": 1e-3,
        "min_dt": 1e-2, "max_dt": 1e-1,
        "cfi": 10.0, "cfr": 10.0,
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    return cfg

# --------------------------- GEFS stats (means/stds) --------------------------
def compute_or_load_gefs_scalar(gefs_dir: Path,
                                cache_path: Path,
                                force_recompute: bool = False) -> Dict[str, float]:
    """
    Build per-variable global mean/std from all GEFS text files (leads 1..7).
    Cached to JSON. This avoids using Daymet/NLDAS stats for GEFS features.
    Returns dict with keys like 'prcp_mean', 'prcp_std', ... for {prcp,srad,tmax,tmin,vp}.
    """
    if (not force_recompute) and cache_path.exists():
        return json.loads(cache_path.read_text())

    # aggregate sums for mean/std (two-pass or Welford; here: Welford)
    vars5 = ['prcp','srad','tmax','tmin','vp']
    count = defaultdict(int)
    mean  = defaultdict(float)
    M2    = defaultdict(float)

    txts = sorted(Path(gefs_dir).glob("*.txt")) + sorted(Path(gefs_dir).glob("*.csv"))
    if not txts:
        raise FileNotFoundError(f"No GEFS files found under {gefs_dir}")

    for p in txts:
        try:
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
            else:
                df = pd.read_csv(p, sep=r"\s+", engine="python")
        except Exception:
            continue

        # standardize columns if your helper exists
        df = _std_cols(df) if '_std_cols' in globals() else df

        # keep only leads you use
        if 'lead' in df.columns:
            df = df[df['lead'].between(1,7)]

        ok = all(v in df.columns for v in vars5)
        if not ok:
            continue

        for v in vars5:
            x = pd.to_numeric(df[v], errors='coerce').dropna().to_numpy()
            for val in x:
                count[v] += 1
                delta    = val - mean[v]
                mean[v] += delta / count[v]
                delta2   = val - mean[v]
                M2[v]   += delta * delta2

    out = {}
    for v in vars5:
        n = max(count[v], 1)
        var = (M2[v] / (n - 1)) if n > 1 else 1.0
        sd  = float(np.sqrt(max(var, 1e-12)))
        out[f"{v}_mean"] = float(mean[v]) if n > 0 else 0.0
        out[f"{v}_std"]  = sd

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, indent=2))
    return out


def _normalize_5_gefs(X5: np.ndarray, gefs_scalar: Dict[str, float]) -> np.ndarray:
    """
    Normalize GEFS (prcp,srad,tmax,tmin,vp) using GEFS-derived stats.
    """
    cols  = ['prcp','srad','tmax','tmin','vp']
    means = np.array([gefs_scalar[f'{c}_mean'] for c in cols], dtype=np.float32)
    stds  = np.array([gefs_scalar[f'{c}_std']  for c in cols], dtype=np.float32)
    stds[stds <= 0.0] = 1.0
    return ((X5.astype(np.float32) - means) / stds).astype(np.float32)



def build_model(cfg: Dict) -> torch.nn.Module:
    cfg = ensure_cfg_defaults(cfg)
    dyn_in = 5 if cfg['forcing_source'] != 'all' else 15
    input_size_dyn = dyn_in if (cfg['no_static'] or not cfg['concat_static']) else (32 if dyn_in==5 else 42)
    m = decoder_only_ssm(
        d_input      = input_size_dyn,
        d_model      = cfg['d_model'],
        n_layers     = cfg['n_layers'],
        cfg          = cfg,
        horizon      = cfg['forecast_horizon'],
        time_emb_dim = cfg['time_emb_dim'],
        static_dim   = 27,
        dropout      = cfg['ssm_dropout'],
        time_full    = True
    ).to(cfg['DEVICE'])
    return m

# --------------------------- batching GEFS per basin -------------------------
def build_batched_inputs_for_basin(
    gefs_df: pd.DataFrame,
    camels_root: Path,
    basin: str,
    lookback: int,
    forcing_source: str,
    scalar: Dict
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Returns:
      x_past_batch:  (B, L, Din)  float32   # L == lookback; covers D-364..D (includes D)
      future_batch:  (B, 7, Din)  float32   # GEFS leads 1..7 → days D..D+6
      init_times:    list of init timestamps (length B)
    Only complete inits (leads 1..7 present) are kept. Vectorized.
    """
    # 1) load reanalysis (15 cols, daily)
    df15, _area = load_forcing_multi(Path(camels_root), basin)
    past_cols = TRI_COLUMNS

    # 2) find inits that have all leads 1..7 (after duplicate removal)
    grp = gefs_df.groupby('init_time')
    want = set(range(1, 8))
    valid_inits = [t for t, g in grp if set(g['lead']) == want]
    if not valid_inits:
        Din = 5 if forcing_source != 'all' else 15
        return (np.zeros((0, lookback, Din), np.float32),
                np.zeros((0, 7, Din), np.float32),
                [])

    # 3) Prepare future batch (GEFS leads 1..7)
    all_fut = []
    for t0 in valid_inits:
        g = grp.get_group(t0).sort_values('lead')
        Xg5 = g[['prcp','srad','tmax','tmin','vp']].to_numpy(dtype=np.float32)  # (7,5)

        if forcing_source == 'all':
            # keep your “3-source” convention if you wish, but for clarity here we
            # treat GEFS as the active source and fill others with their own means.
            X15 = np.zeros((7, 15), dtype=np.float32)
            names = ['prcp','srad','tmax','tmin','vp']
        
            # positions for the "active" stream (we keep GEFS in the *_nldas slots 0,3,6,9,12 for simplicity)
            bases = [0, 3, 6, 9, 12]
            for i, b in enumerate(bases):
                # Put GEFS into the active column
                X15[:, b+0] = Xg5[:, i]
                # Fill the two “other sources” with GEFS means so stats won't confuse the net
                X15[:, b+1] = float(GEFS_SCALAR[f"{names[i]}_mean"])
                X15[:, b+2] = float(GEFS_SCALAR[f"{names[i]}_mean"])
        
            # Normalize the 15-dim input using **GEFS stats for all five meteorological slots**.
            # If your normalize_multi_features expects a single set of stats, you can patch it
            # or simply standardize X15 manually variable-by-variable (shown here):
            # Standardize each 3-wide block with GEFS stats
            for i, b in enumerate(bases):
                mu  = float(GEFS_SCALAR[f"{names[i]}_mean"])
                std = float(max(GEFS_SCALAR[f"{names[i]}_std"], 1e-6))
                X15[:, b+0:b+3] = (X15[:, b+0:b+3] - mu) / std
        
            fut_n = X15.astype(np.float32)   # (7,15)
        
        else:
            # Single-source path: normalize 5 GEFS variables with **GEFS stats**
            fut_n = _normalize_5_gefs(Xg5, GEFS_SCALAR).astype(np.float32)   # (7,5) # todo, normalize with gefs stats
            #fut_n = _normalize_5(Xg5, forcing_source, scalar).astype(np.float32) # normalize with daymet states, should not do that


        all_fut.append(fut_n.astype(np.float32))

    future_batch = np.stack(all_fut, axis=0).astype(np.float32)  # (B,7,Din)

    # 4) Build past windows (MATCH TRAINING): D-364..D (includes D)
    Din = 15 if forcing_source == 'all' else 5
    xp = np.zeros((len(valid_inits), lookback, Din), dtype=np.float32)
    pick = None if forcing_source == 'all' else IDX_MAP[forcing_source]
    idx = df15.index

    for i, t0 in enumerate(valid_inits):
        t0d = pd.Timestamp(t0.date())  # D
        # include D (so window length is exactly `lookback`)
        start = t0d - pd.Timedelta(days=lookback - 1)   # D-364 when lookback=365
        end   = t0d                                     # D

        # bounds check
        if (start < idx.min()) or (end > idx.max()):
            xp[i, :, :] = np.nan
            continue

        # inclusive slice → shape (lookback, 15)
        Xpast_raw = df15[past_cols].loc[start:end].to_numpy(dtype=np.float32)
        # sanity: make sure length is correct
        if Xpast_raw.shape[0] != lookback:
            # try to realign via normalize to midnight if needed (rare)
            # but in most cases, just skip if misaligned
            xp[i, :, :] = np.nan
            continue

        Xpast_n = normalize_multi_features(Xpast_raw, 'inputs', scalar).astype(np.float32)  # (L,15)
        if pick is not None:
            Xpast_n = Xpast_n[:, pick].astype(np.float32)                                   # (L,5)
        xp[i] = Xpast_n.astype(np.float32)

    # Remove rows that failed bounds
    good = ~np.isnan(xp).any(axis=(1, 2))
    x_past_batch = xp[good].astype(np.float32)
    future_batch = future_batch[good].astype(np.float32)
    init_times   = [t for t, ok in zip(valid_inits, good) if ok]

    return x_past_batch, future_batch, init_times


# --------------------------- per-basin run (batched) -------------------------
@torch.no_grad()
def run_basin_batched(
    model: torch.nn.Module,
    basin: str,
    camels_root: Path,
    attr_db: Path,
    gefs_df: pd.DataFrame,
    scalar: Dict,
    forcing_source: str,
    lookback: int,
    horizon: int,
    batch_inits: int,
    num_samples: int,
    ddim_steps: int,
    eta: float,
    device: torch.device
) -> Tuple[pd.DataFrame, np.ndarray, List[pd.Timestamp], np.ndarray]:
    """
    Returns:
      df_mean:    (M rows) per-init dataframe (init_time, obs, lead0..lead7)
      ens_cube:   (M, S, H) ensemble cube for this basin
      init_times: list of init timestamps (len M)
      obs_vec:    (M,) observed discharge aligned to init day
    """
    # Build all inputs for this basin (vectorized)
    x_past_np, future_np, inits = build_batched_inputs_for_basin(
        gefs_df, camels_root, basin, lookback, forcing_source, scalar
    )
    if len(inits) == 0:
        cols = ["init_time","obs"]+[f"lead{k}" for k in range(horizon)]
        return pd.DataFrame(columns=cols), np.zeros((0, num_samples, horizon), np.float32), [], np.zeros((0,), np.float32)

    # Load area for obs conversion & static attrs
    _, area = load_forcing_multi(camels_root, basin)  # area from same util
    q_obs_series = load_discharge(camels_root, basin, area)
    static27 = torch.from_numpy(_load_static_27(attr_db, basin)).to(device)  # (27,)

    B = len(inits)
    H = horizon
    x_past_all = torch.from_numpy(x_past_np).to(device).float()    # (B,L,D) float32
    future_all = torch.from_numpy(future_np).to(device).float()    # (B,7,D) float32
    static_all = static27.unsqueeze(0).unsqueeze(1).repeat(B, H, 1).float()  # (B,H,27) float32

    # process in chunks to limit VRAM
    ens_out = np.zeros((B, num_samples, H), dtype=np.float32)
    for s in range(0, B, batch_inits):
        e = min(s + batch_inits, B)
        xs = x_past_all[s:e].float()
        fs = future_all[s:e].float()
        st = static_all[s:e].float()
        # encode past ONCE and reuse across all ensemble members
        with torch.no_grad():
            past_cache = getattr(model, _encode_past)(xs, fs, st)

        samp_acc = []
        for _ in range(num_samples):
            y_norm = getattr(model, _sample_ddim)(
                            xs, st, fs,
                            num_steps=ddim_steps,
                            eta=eta,
                            ddpm_n_steps=args.ddpm_train_steps,
                            ddpm_beta_start=args.ddpm_beta_start,
                            ddpm_beta_end=args.ddpm_beta_end,
                            **{_cache_kwarg: past_cache},
                        )  # (b,H), float32
            y = _denorm_q(y_norm.detach().cpu().numpy(), scalar)                   # (b,H) float32
            samp_acc.append(y.astype(np.float32))
        S = np.stack(samp_acc, axis=1)   # (b,S,H) float32
        ens_out[s:e, :, :] = S
    ''' old code block. 1 observation only. Now returns 8 obs cols with corresponding to lead time.
    # Build output mean dataframe & obs vector
    means = ens_out.mean(axis=1)         # (B,H)
    obs = []
    rows = []
    for i, t0 in enumerate(inits):
        d0 = pd.Timestamp(t0.date())
        obs_val = float(q_obs_series.loc[d0]) if d0 in q_obs_series.index else np.nan
        obs.append(np.float32(obs_val))
        rows.append({"init_time": t0, "obs": float(obs_val), **{f"lead{k}": float(means[i, k]) for k in range(H)}})
    df_mean = pd.DataFrame(rows).sort_values("init_time").reset_index(drop=True)
    return df_mean, ens_out, inits, np.array(obs, dtype=np.float32)
    '''
    means = ens_out.mean(axis=1)  # (B, H)
    obs = []
    rows = []
    
    for i, t0 in enumerate(inits):
        d0 = pd.Timestamp(t0.date())
    
        obs_row = []
        for k in range(H):
            dk = d0 + pd.Timedelta(days=k)
            obs_row.append(float(q_obs_series.loc[dk]) if dk in q_obs_series.index else np.nan)
    
        obs.append(obs_row)
    
        row = {
            "init_time": t0,
            **{f"obs_lead{k}": obs_row[k] for k in range(H)},
            **{f"lead{k}": float(means[i, k]) for k in range(H)}
        }
        rows.append(row)
    
    df_mean = pd.DataFrame(rows).sort_values("init_time").reset_index(drop=True)
    
    # obs shape: (M, H)
    return df_mean, ens_out, inits, np.array(obs, dtype=np.float32)

# --------------------------- MAIN --------------------------------------------

ap = build_argparser()
args = ap.parse_args()

if args.ssm_backend == 'hybrid':
    from papercode.decoder_only_ssm_hybrid import decoder_only_ssm
    _encode_past  = 'encode_past_hybrid'
    _sample_ddim  = 'sample_ddim_hybrid'
    _cache_kwarg  = 'past_hybrid_cache'
else:
    from papercode.decoder_only_ssm_fast import decoder_only_ssm
    _encode_past  = 'encode_past_fft'
    _sample_ddim  = 'sample_ddim_fast_fft'
    _cache_kwarg  = 'past_fft_cache'

log_path = setup_logging(args.log_dir, args.note, args.log_level)
logging.info(f"Logging to {log_path}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"CUDA available: {torch.cuda.is_available()} | Using device: {device}")
if torch.cuda.is_available():
    logging.info(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES','(unset)')}")
    logging.info(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
    try:
        logging.info(f"torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
    except Exception:
        pass
logging.info(f"PyTorch: {torch.__version__}")

# Build CFG to match training (and ensure all keys exist)
CFG: Dict = {
    "DEVICE": device,
    "model_name": "decoder_only_ssm",
    "forecast_horizon": args.horizon,
    "d_model": args.d_model,
    "d_state": args.d_state,
    "n_layers": args.n_layers,
    "time_emb_dim": args.time_emb_dim,
    "ssm_dropout": args.ssm_dropout,
    "pool_type": args.pool_type,
    "predict_mode": args.predict_mode,
    "forcing_source": args.forcing,
    "no_static": args.no_static,
    "concat_static": args.concat_static,
    # SSM train cfg used by model init:
    "lr": args.lr, "lr_min": args.lr_min,
    "weight_decay": args.weight_decay, "wd": args.wd,
    "warmup": args.warmup,
    "lr_dt": args.lr_dt, "min_dt": args.min_dt, "max_dt": args.max_dt,
    "cfi": args.cfi, "cfr": args.cfr,
}
for k, v in CFG.items():
    logging.info(f"{k}: {v if k!='DEVICE' else device.type}")
logging.info(f"DDIM sampling steps: {args.ddim_steps}")
logging.info(f"Number of ensemble samples: {args.num_samples}")
logging.info(f"DDIM eta: {args.eta}")

out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
per_basin_dir = out_root / "per_basin_csv_ddim10_optim_hybrid"; per_basin_dir.mkdir(parents=True, exist_ok=True) # todo

# build & load model
model = build_model(CFG)
state = torch.load(args.model_ckpt, map_location=device)
model.load_state_dict(state.get("model", state))
model.eval()
logging.info("Model loaded & set to eval().")

# select basins
if args.basins is None:
    basins = get_basin_list(Path(args.camels_root))
else:
    p = Path(args.basins)
    basins = [ln.strip() for ln in (p.read_text().splitlines() if p.exists() else args.basins.split(",")) if ln.strip()]
logging.info(f"Basins to process: {len(basins)}")

# global training scalar
GSCALAR = load_scalar(args.scalar)
# Build/load GEFS normalization stats once
GEFS_SCALAR = compute_or_load_gefs_scalar(
    gefs_dir = Path(args.gefs_dir),
    cache_path = Path(args.out_dir) / "gefs_scalar.json",
    force_recompute = False
)
logging.info(f"GEFS scalar loaded: {GEFS_SCALAR}")


# global collectors
G_ens: List[np.ndarray] = []
G_dates: List[pd.Timestamp] = []
#G_obs:   List[float] = [] # old code
G_obs:   List[List[float]] = []
G_bas:   List[str] = []

ok=skip=fail=0
t_all0 = time.time()

for i, b in enumerate(basins, 1):
    t0 = time.time()
    out_csv = per_basin_dir / f"{b}_preds.csv"

    if out_csv.exists():
        logging.info(f"[{i}/{len(basins)}] Basin {b} ... SKIP: file already exists -> {out_csv}")
        skip += 1
        continue

    logging.info(f"[{i}/{len(basins)}] Basin {b} ...")
    gefs_path = Path(args.gefs_dir) / f"{b}.txt"
    if not gefs_path.exists():
        logging.warning(f"[{b}] Missing GEFS file: {gefs_path}")
        fail += 1
        continue
    try:
        df = _read_gefs_file(gefs_path)
        # optional date window
        df = _filter_inits_by_date(df, args.date_start, args.date_end)
        if df.empty:
            logging.info(f"[{b}] No init rows after filters.")
            skip += 1
            continue

        # Optional: alignment print for the first available init in this basin
        try:
            first_init = df['init_time'].sort_values().iloc[0]
            df15, _ = load_forcing_multi(Path(args.camels_root), b)
            assert_alignment(df, df15, first_init)

        except Exception as _e:
            logging.warning(f"[{b}] alignment check skipped: {_e}")

        df_mean, ens_cube, inits, obs_vec = run_basin_batched(
            model=model,
            basin=b,
            camels_root=Path(args.camels_root),
            attr_db=Path(args.attr_db),
            gefs_df=df,
            scalar=GSCALAR,
            forcing_source=args.forcing,
            lookback=args.lookback,
            horizon=args.horizon,
            batch_inits=args.batch_inits,
            num_samples=args.num_samples,
            ddim_steps=args.ddim_steps,
            eta=args.eta,
            device=device
        )
    except Exception as e:
        logging.error(f"[{b}] FAILED:\n{e}\n{traceback.format_exc()}")
        fail += 1
        continue

    if df_mean.empty:
        logging.info(f"[{b}] SKIP: no complete inits.")
        skip += 1
        continue

    # save per-basin CSV (ensemble mean)
    df_mean.to_csv(out_csv, index=False)
    dt = time.time() - t0
    ms_per_init = 1000.0 * dt / len(df_mean)
    logging.info(f"[{b}] OK: {len(df_mean)} inits -> {out_csv}  "
                 f"time={dt:.1f}s  ({ms_per_init:.1f} ms/init)")

    # global accumulators
    G_ens.append(ens_cube)               # (M,S,H)
    G_dates.extend(inits)                # M
    G_obs.extend(obs_vec.tolist())       # M
    G_bas.extend([b] * len(inits))
    ok += 1

# save global NPZ
if G_ens:
    ens_arr = np.concatenate(G_ens, axis=0)   # (N,S,H)
    bas_arr = np.array(G_bas, dtype="U32")
    dts_arr = np.array(G_dates, dtype="datetime64[ns]")
    obs_arr = np.array(G_obs, dtype=float)
    note = f"_{args.note}" if args.note else ""
    npz_path = Path(args.out_dir) / f"gefs_ensembles_{args.forcing}_S{args.num_samples}{note}.npz"
    np.savez(npz_path, basins=bas_arr, dates=dts_arr, obs=obs_arr, ens=ens_arr)
    logging.info(f"[GLOBAL] Saved ensembles ? {npz_path} "
                 f"(N={ens_arr.shape[0]}, S={ens_arr.shape[1]}, H={ens_arr.shape[2]}; "
                 f"ok={ok}, skip={skip}, fail={fail})")
else:
    logging.warning(f"[GLOBAL] No predictions produced. (ok={ok}, skip={skip}, fail={fail})")

logging.info(f"Total wall clock: {(time.time()-t_all0)/60:.1f} min.")
