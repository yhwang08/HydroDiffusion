# papercode/zeroshot_test_lstm.py
# Zero-shot GEFS evaluation for a Diffusion Encoder–Decoder LSTM.
# - Per-basin CSVs with ensemble mean
# - Global NPZ with full ensemble cube (N_inits, S_samples, H_horizon)
# - Global NPZ obs saved as (N_inits, H_horizon), not just (N_inits,)
#
# Example:
'''
   python -m papercode.zeroshot_test_lstm \
     --gpu 5 \
     --model_ckpt /data/home/yihan/diffusion_ssm/runs/run_2207_1832_seed3407/best_model.pt \
     --gefs_dir /data/rdl/yihan/GEFS_forecasts \
     --camels_root /data/rdl/yihan/data/basin_dataset_public_v1p2 \
     --attr_db /data/home/yihan/diffusion_ssm/runs/shared_h5_new/attributes.db \
     --scalar /home/yihan/diffusion_ssm/global_scalar.json \
     --forcing daymet \
     --lookback 365 --horizon 8 \
     --num_samples 50 --ddim_steps 10 --eta 0.0 \
     --hidden_size 256 --time_emb_dim 256 --initial_forget_gate_bias 3.0 \
     --note gefs_zts_lstm_ddim10
'''

from __future__ import annotations
import os, sys, argparse, logging, time, traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union, List, Tuple
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

# Repo-relative imports regardless of CWD
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

# IMPORTANT: import your Diffusion Enc–Dec LSTM here.
# Expected API: model.sample_ddim(x_past, static, future, num_steps, eta) -> (B,H)
# If your module/class name differs, adjust the import and build_model accordingly.
from papercode.backbones.lstm import GenericLSTM
from papercode.diffusion_wrapper import EncoderDecoderDiffusionWrapper

# --------------------------- logging -----------------------------------------
def setup_logging(log_dir: str, note: str, level: str) -> Path:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{note}" if note else ""
    log_file = Path(log_dir) / f"zeroshot_lstm_{ts}{suffix}.log"

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
    # paths
    p.add_argument("--model_ckpt",   type=str, required=True)
    p.add_argument("--camels_root",  type=str,
                   default="/data/rdl/yihan/data/basin_dataset_public_v1p2")
    p.add_argument("--attr_db",      type=str,
                   default="/data/home/yihan/diffusion_ssm/runs/shared_h5_new/attributes.db")
    p.add_argument("--gefs_dir",     type=str, default="/data/rdl/yihan/GEFS_forecasts")
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
    p.add_argument("--horizon",      type=int, default=8)  # nowcast + 7
    p.add_argument("--batch_inits",  type=int, default=512,
                   help="How many init times to process per forward pass.")
    p.add_argument("--date_start",   type=str, default=None,
                   help="Optional YYYY-MM-DD to limit processed init range.")
    p.add_argument("--date_end",     type=str, default=None,
                   help="Optional YYYY-MM-DD to limit processed init range.")

    # diffusion sampling
    p.add_argument("--num_samples",  type=int, default=50)
    p.add_argument("--ddim_steps",   type=int, default=10)
    p.add_argument("--eta",          type=float, default=0.0)

    # data flags (must match training)
    p.add_argument("--forcing",      type=str, default="daymet",
                   choices=["daymet","nldas","maurer","all"])
    p.add_argument("--no_static",    action="store_true", default=False)
    p.add_argument("--concat_static",action="store_true", default=True)

    # LSTM hyperparams used at model init (adapted to your training build)
    p.add_argument("--d_input",      type=int, default=None, help="Override inferred Din; rarely needed")
    p.add_argument("--hidden_size",  type=int, default=256)
    p.add_argument("--n_layers_enc", type=int, default=1)  # kept for parity; wrapper may not use
    p.add_argument("--n_layers_dec", type=int, default=1)  # kept for parity; wrapper may not use
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--time_emb_dim", type=int, default=256)
    p.add_argument("--initial_forget_gate_bias", type=float, default=1.0)
    p.add_argument("--predict_mode", type=str, default="velocity")  # or "eps"

    # basin selection
    p.add_argument("--basins",       type=str, default=None,
                   help="Optional text file (one basin per line) or comma list")
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

    # valid_time is the actual date of the forecast value; init = valid - lead (days)
    df['valid_time'] = pd.to_datetime(dict(year=df['Year'], month=df['Mnth'], day=df['Day'], hour=df['Hr']))
    df['init_time']  = df['valid_time'] - pd.to_timedelta(df['lead'], unit='D')

    df = df[df['lead'].between(1,7)].copy()
    df = df.sort_values(['init_time','lead'])
    df = df.drop_duplicates(subset=['init_time','lead'], keep='last')
    return df

# Normalize GEFS (5 vars) using GEFS-derived stats (computed below)
def _normalize_5_gefs(X5: np.ndarray, gefs_scalar: Dict[str, float]) -> np.ndarray:
    cols  = ['prcp','srad','tmax','tmin','vp']
    means = np.array([gefs_scalar[f'{c}_mean'] for c in cols], dtype=np.float32)
    stds  = np.array([gefs_scalar[f'{c}_std']  for c in cols], dtype=np.float32)
    stds[stds <= 0.0] = 1.0
    return ((X5.astype(np.float32) - means) / stds).astype(np.float32)

def _denorm_q(y_norm: np.ndarray, scalar: Dict) -> np.ndarray:
    mu = np.float32(scalar["QObs(mm/d)_mean"]) ; sd = np.float32(scalar["QObs(mm/d)_std"])
    return (y_norm.astype(np.float32) * sd + mu).astype(np.float32)

def _load_static_27(db_path: Path, basin: str) -> np.ndarray:
    conn = sqlite3.connect(str(db_path))
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';")]
    if not tables:
        conn.close(); raise RuntimeError(f"No tables in {db_path}")
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
    return z.loc[basin].iloc[:27].values.astype(np.float32)

def _filter_inits_by_date(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start is not None:
        df = df[df['init_time'] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df['init_time'] <= pd.Timestamp(end)]
    return df

# --------------------------- GEFS stats (means/stds) --------------------------

def compute_or_load_gefs_scalar(gefs_dir: Path, cache_path: Path, force_recompute: bool = False) -> Dict[str, float]:
    if (not force_recompute) and cache_path.exists():
        return json.loads(cache_path.read_text())

    vars5 = ['prcp','srad','tmax','tmin','vp']
    count = defaultdict(int); mean = defaultdict(float); M2 = defaultdict(float)

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
        df = _std_cols(df) if '_std_cols' in globals() else df
        if 'lead' in df.columns:
            df = df[df['lead'].between(1,7)]
        if not all(v in df.columns for v in vars5):
            continue
        for v in vars5:
            x = pd.to_numeric(df[v], errors='coerce').dropna().to_numpy()
            for val in x:
                count[v] += 1
                delta = val - mean[v]
                mean[v] += delta / count[v]
                delta2 = val - mean[v]
                M2[v]  += delta * delta2

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

# --------------------------- model builder -----------------------------------

def build_model(cfg: Dict) -> torch.nn.Module:
    """Instantiate Diffusion Encoder–Decoder LSTM per training setup:
    encoder = GenericLSTM(input_size=Din)
    decoder = GenericLSTM(input_size=1 + Din + 27)
    wrapped with EncoderDecoderDiffusionWrapper.
    """
    # Determine Din from forcing (or override)
    dyn_in = 5 if cfg['forcing_source'] != 'all' else 15
    if cfg.get('d_input') is not None:
        dyn_in = int(cfg['d_input'])

    # Require statics for this build (decoder uses +27)
    if cfg.get('no_static', False) or (not cfg.get('concat_static', True)):
        raise ValueError("diffusion_lstm expects static features (27). Use --concat_static and avoid --no_static.")

    encoder = GenericLSTM(
        input_size = dyn_in + 27,  # concat statics to past features (matches training: 5 + 27 = 32)
        hidden_size = cfg.get('hidden_size', 256),
        dropout = cfg.get('dropout', 0.2),
        init_forget_bias = cfg.get('initial_forget_gate_bias', 1.0),
        batch_first = True,
    )

    decoder = GenericLSTM(
        input_size = 1 + dyn_in + 27,  # x_t + future dyn + statics
        hidden_size = cfg.get('hidden_size', 256),
        dropout = cfg.get('dropout', 0.2),
        init_forget_bias = cfg.get('initial_forget_gate_bias', 1.0),
        batch_first = True,
    )

    model = EncoderDecoderDiffusionWrapper(
        encoder = encoder,
        decoder = decoder,
        hidden_size = cfg.get('hidden_size', 256),
        time_emb_dim = cfg.get('time_emb_dim', 256),
        decoder_name = 'lstm',
        prediction_type = cfg.get('predict_mode', 'velocity'),
    ).to(cfg['DEVICE'])

    return model

# --------------------------- batching GEFS per basin -------------------------

def build_batched_inputs_for_basin(
    gefs_df: pd.DataFrame,
    camels_root: Path,
    basin: str,
    lookback: int,
    forcing_source: str,
    scalar: Dict,
    GEFS_SCALAR: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Returns:
      x_past_batch:  (B, L, Din)   float32   # past window D-364..D (includes D)
      future_batch:  (B, 8, Din)   float32   # day 0 from **DAYMET obs** + GEFS leads 1..7
      init_times:    list of init timestamps (length B)
    Keeps only inits with all GEFS leads 1..7 present.

    Notes
    -----
    * Day 0 (nowcast) is taken from reanalysis/obs (DAYMET) on calendar day D
      (same day as the init), then normalized with **GEFS stats** so that the
      decoder sees a consistent normalization across t=0..7.
    * For forcing_source == 'all', we keep the prior convention where the
      "active" stream occupies the NLDAS slots [0,3,6,9,12]. We place the
      normalized Daymet t=0 values into those active slots and zero the other
      two within each triad (equivalent to their mean after normalization).
    """
    df15, _area = load_forcing_multi(Path(camels_root), basin)

    grp = gefs_df.groupby('init_time')
    want = set(range(1, 8))
    valid_inits = [t for t, g in grp if set(g['lead']) == want]
    if not valid_inits:
        Din = 5 if forcing_source != 'all' else 15
        return (
            np.zeros((0, lookback, Din), np.float32),
            np.zeros((0, 8, Din), np.float32),
            []
        )

    def _day0_daymet_row(D_date: pd.Timestamp) -> np.ndarray:
        row = df15.loc[D_date, TRI_COLUMNS]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        dm_vals = row[['prcp_daymet','srad_daymet','tmax_daymet','tmin_daymet','vp_daymet']].to_numpy(dtype=np.float32)
        return dm_vals

    all_fut: List[np.ndarray] = []
    for t0 in valid_inits:
        g = grp.get_group(t0).sort_values('lead')
        Xg5 = g[['prcp','srad','tmax','tmin','vp']].to_numpy(dtype=np.float32)  # (7,5)

        if forcing_source == 'all':
            X15 = np.zeros((7, 15), dtype=np.float32)
            names = ['prcp','srad','tmax','tmin','vp']
            bases = [0, 3, 6, 9, 12]
            for i, b in enumerate(bases):
                mu  = float(GEFS_SCALAR[f"{names[i]}_mean"])
                sd  = float(max(GEFS_SCALAR[f"{names[i]}_std"], 1e-6))
                X15[:, b+0] = (Xg5[:, i] - mu) / sd
                X15[:, b+1] = 0.0
                X15[:, b+2] = 0.0
            fut_gefs_norm = X15  # (7,15)
        else:
            fut_gefs_norm = _normalize_5_gefs(Xg5, GEFS_SCALAR).astype(np.float32)  # (7,5)

        D = pd.Timestamp(t0.date())
        dm5 = _day0_daymet_row(D)  # (5,)
        if forcing_source == 'all':
            day0 = np.zeros((1, 15), dtype=np.float32)
            names = ['prcp','srad','tmax','tmin','vp']
            bases = [0, 3, 6, 9, 12]
            for i, b in enumerate(bases):
                mu  = float(GEFS_SCALAR[f"{names[i]}_mean"])
                sd  = float(max(GEFS_SCALAR[f"{names[i]}_std"], 1e-6))
                day0[:, b+0] = (dm5[i] - mu) / sd
                day0[:, b+1] = 0.0
                day0[:, b+2] = 0.0
            fut8 = np.concatenate([day0, fut_gefs_norm], axis=0)  # (8,15)
        else:
            day0 = _normalize_5_gefs(dm5[None, :], GEFS_SCALAR)   # (1,5)
            fut8 = np.concatenate([day0, fut_gefs_norm], axis=0)  # (8,5)

        all_fut.append(fut8.astype(np.float32))

    future_batch = np.stack(all_fut, axis=0).astype(np.float32)  # (B,8,Din)

    Din_past = 15 if forcing_source == 'all' else 5
    xp = np.zeros((len(valid_inits), lookback, Din_past), dtype=np.float32)
    pick = None if forcing_source == 'all' else IDX_MAP[forcing_source]
    idx = df15.index

    for i, t0 in enumerate(valid_inits):
        D = pd.Timestamp(t0.date())
        start = D - pd.Timedelta(days=lookback - 1)
        end   = D
        if (start < idx.min()) or (end > idx.max()):
            xp[i, :, :] = np.nan
            continue
        Xpast_raw = df15[TRI_COLUMNS].loc[start:end].to_numpy(dtype=np.float32)
        if Xpast_raw.shape[0] != lookback:
            xp[i, :, :] = np.nan
            continue
        Xpast_n = normalize_multi_features(Xpast_raw, 'inputs', scalar).astype(np.float32)
        if pick is not None:
            Xpast_n = Xpast_n[:, pick].astype(np.float32)
        xp[i] = Xpast_n

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
    GEFS_SCALAR: Dict[str, float],
    forcing_source: str,
    lookback: int,
    horizon: int,
    batch_inits: int,
    num_samples: int,
    ddim_steps: int,
    eta: float,
    device: torch.device
) -> Tuple[pd.DataFrame, np.ndarray, List[pd.Timestamp], np.ndarray]:
    x_past_np, future_np, inits = build_batched_inputs_for_basin(
        gefs_df, camels_root, basin, lookback, forcing_source, scalar, GEFS_SCALAR
    )

    if len(inits) == 0:
        cols = (
            ["init_time"] +
            [f"obs_lead{k}" for k in range(horizon)] +
            [f"lead{k}" for k in range(horizon)]
        )
        return (
            pd.DataFrame(columns=cols),
            np.zeros((0, num_samples, horizon), np.float32),
            [],
            np.zeros((0, horizon), np.float32)
        )

    _, area = load_forcing_multi(camels_root, basin)
    q_obs_series = load_discharge(camels_root, basin, area)

    static_dim = 27 if True else 0
    static27 = torch.from_numpy(_load_static_27(attr_db, basin)).to(device) if static_dim > 0 else None

    B = len(inits)
    H = horizon
    x_past_all = torch.from_numpy(x_past_np).to(device).float()   # (B,L,D)
    future_all = torch.from_numpy(future_np).to(device).float()   # (B,H,D)

    if static27 is not None:
        static_all = static27.unsqueeze(0).unsqueeze(1).repeat(B, H, 1).float()
    else:
        static_all = None

    L = x_past_np.shape[1]
    static_enc = static27.unsqueeze(0).unsqueeze(1).repeat(B, L, 1).float()
    x_past_all = torch.from_numpy(x_past_np).to(device).float()
    x_past_all = torch.cat([x_past_all, static_enc], dim=2)  # (B,L,Din+27)

    future_all = torch.from_numpy(future_np).to(device).float()   # (B,H,Din)

    ens_out = np.zeros((B, num_samples, H), dtype=np.float32)

    for s in range(0, B, batch_inits):
        e = min(s + batch_inits, B)
        xs = x_past_all[s:e]  # (b,L,Din+27)
        fs = future_all[s:e]  # (b,H,Din)
        st = static_all[s:e] if static_all is not None else None  # (b,H,27)

        samp_acc = []
        for _ in range(num_samples):
            y_norm = model.sample_ddim(xs, st, fs, num_steps=ddim_steps, eta=eta)  # (b,H)
            y = _denorm_q(y_norm.detach().cpu().numpy(), scalar)
            samp_acc.append(y.astype(np.float32))
        S = np.stack(samp_acc, axis=1)  # (b,S,H)
        ens_out[s:e, :, :] = S

    means = ens_out.mean(axis=1)  # (B,H)
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

    # obs shape: (B,H)
    return df_mean, ens_out, inits, np.array(obs, dtype=np.float32)

# --------------------------- MAIN --------------------------------------------

ap = build_argparser()
args = ap.parse_args()

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

# Config
CFG: Dict = {
    "DEVICE": device,
    "model_name": "diffusion_lstm",
    "forecast_horizon": args.horizon,
    "forcing_source": args.forcing,
    "no_static": args.no_static,
    "concat_static": args.concat_static,
    # LSTM/diffusion hparams
    "hidden_size": args.hidden_size,
    "n_layers_enc": args.n_layers_enc,
    "n_layers_dec": args.n_layers_dec,
    "dropout": args.dropout,
    "time_emb_dim": args.time_emb_dim,
    "initial_forget_gate_bias": args.initial_forget_gate_bias,
    "predict_mode": args.predict_mode,
}
if getattr(args, 'd_input', None) is not None:
    CFG['d_input'] = args.d_input

for k, v in CFG.items():
    logging.info(f"{k}: {v if k!='DEVICE' else device.type}")

out_root = Path(args.out_dir)
out_root.mkdir(parents=True, exist_ok=True)
per_basin_dir = out_root / "per_basin_csv_lstm"
per_basin_dir.mkdir(parents=True, exist_ok=True)

# model
model = build_model(CFG)
state = torch.load(args.model_ckpt, map_location=device)
model.load_state_dict(state.get("model", state))
model.eval()
logging.info("Model loaded & set to eval().")

# basins
if args.basins is None:
    basins = get_basin_list(Path(args.camels_root))
else:
    p = Path(args.basins)
    basins = [ln.strip() for ln in (p.read_text().splitlines() if p.exists() else args.basins.split(",")) if ln.strip()]
logging.info(f"Basins to process: {len(basins)}")

# scalars
GSCALAR = load_scalar(args.scalar)
GEFS_SCALAR = compute_or_load_gefs_scalar(
    gefs_dir=Path(args.gefs_dir),
    cache_path=Path(args.out_dir) / "gefs_scalar.json",
    force_recompute=False
)
logging.info(f"GEFS scalar loaded: {GEFS_SCALAR}")

# globals
G_ens: List[np.ndarray] = []
G_dates: List[pd.Timestamp] = []
G_obs: List[List[float]] = []
G_bas: List[str] = []

ok = skip = fail = 0
t_all0 = time.time()

for i, b in enumerate(basins, 1):
    t0 = time.time()
    logging.info(f"[{i}/{len(basins)}] Basin {b} ...")
    gefs_path = Path(args.gefs_dir) / f"{b}.txt"
    if not gefs_path.exists():
        logging.warning(f"[{b}] Missing GEFS file: {gefs_path}")
        fail += 1
        continue
    try:
        df = _read_gefs_file(gefs_path)
        df = _filter_inits_by_date(df, args.date_start, args.date_end)
        if df.empty:
            logging.info(f"[{b}] No init rows after filters.")
            skip += 1
            continue

        df_mean, ens_cube, inits, obs_vec = run_basin_batched(
            model=model,
            basin=b,
            camels_root=Path(args.camels_root),
            attr_db=Path(args.attr_db),
            gefs_df=df,
            scalar=GSCALAR,
            GEFS_SCALAR=GEFS_SCALAR,
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

    out_csv = per_basin_dir / f"{b}_preds.csv"
    df_mean.to_csv(out_csv, index=False)
    dt = time.time() - t0
    ms_per_init = 1000.0 * dt / len(df_mean)
    logging.info(f"[{b}] OK: {len(df_mean)} inits → {out_csv}  time={dt:.1f}s  ({ms_per_init:.1f} ms/init)")

    G_ens.append(ens_cube)
    G_dates.extend(inits)
    G_obs.extend(obs_vec.tolist())
    G_bas.extend([b] * len(inits))
    ok += 1

# save global NPZ
if G_ens:
    ens_arr = np.concatenate(G_ens, axis=0)   # (N,S,H)
    bas_arr = np.array(G_bas, dtype="U32")
    dts_arr = np.array(G_dates, dtype="datetime64[ns]")
    obs_arr = np.array(G_obs, dtype=float)    # (N,H)
    note = f"_{args.note}" if args.note else ""
    npz_path = Path(args.out_dir) / f"gefs_ensembles_lstm_{args.forcing}_S{args.num_samples}{note}.npz"
    np.savez(npz_path, basins=bas_arr, dates=dts_arr, obs=obs_arr, ens=ens_arr)
    logging.info(f"[GLOBAL] Saved ensembles → {npz_path} "
                 f"(N={ens_arr.shape[0]}, S={ens_arr.shape[1]}, H={ens_arr.shape[2]}; "
                 f"ok={ok}, skip={skip}, fail={fail})")
else:
    logging.warning(f"[GLOBAL] No predictions produced. (ok={ok}, skip={skip}, fail={fail})")

logging.info(f"Total wall clock: {(time.time()-t_all0)/60:.1f} min.")