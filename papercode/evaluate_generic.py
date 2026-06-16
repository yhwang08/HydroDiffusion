# cache for decoder_only_ssm and decoder_only_lstm both
import os, sys
import pickle
from pathlib import Path
import pdb

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from papercode.datasets import CamelsH5
from papercode.utils import get_basin_list
from papercode.datautils import load_scalar
#from papercode.diffusion_utils import ddim_sample


from papercode.lstm import Seq2SeqLSTM, EncoderDecoderDetLSTM

from papercode.backbones.lstm import GenericLSTM
from papercode.backbones.unet import unet
from papercode.backbones.unet_film import unet_film

from papercode.backbones.unet_film_v2 import unet_film_v2
from papercode.backbones.unet_film_v3 import unet_film_v3
from papercode.backbones.unet_attention import unet_attention
from papercode.backbones.unet_attention_film_v2 import unet_attention_film_v2
from papercode.backbones.unet_attention_film_v3 import unet_attention_film_v3

from papercode.decoder_only_lstm import decoder_only_lstm
from papercode.decoder_only_lstm_fast import decoder_only_lstm as decoder_only_lstm_fast
from papercode.diffusion_lstm_fast import DiffusionLSTMFast

from papercode.diffusion_wrapper import EncoderDecoderDiffusionWrapper


from papercode.seq2seq_ssm import seq2seq_ssm

from papercode.backbones.ssm_v1 import ssm_v1
from papercode.backbones.ssm_v2 import ssm_v2
from papercode.backbones.ssm_encoder import ssm_encoder
#from papercode.decoder_only_ssm import decoder_only_ssm
#from papercode.decoder_only_ssm_hybrid import decoder_only_ssm
from papercode.decoder_only_ssm_hybrid import decoder_only_ssm

from papercode.runtime_profiler import RuntimeProfiler


# --------------------------------------------------------------------
SCALAR = load_scalar("/home/yihan/diffusion_ssm/global_scalar.json")
PER_BASIN_NORM = False

# --------------------------------------------------------------------
def flatten_summary(model_name, summary):
    rows = []
    for k, stats in summary.items():
        if "seconds" not in stats:
            continue  # skip metadata entries like s4d_mode
        rows.append({
            "model": model_name,
            "component": k,
            "seconds": stats["seconds"],
            "calls": stats["calls"],
            "avg_per_call": stats["avg_per_call"],
            "percent_total": stats["percent_total"],
        })
    return rows
    
# --------------------------------------------------------------------   
def load_train_q_stats(train_h5_path: Path) -> dict:
    """Basin-specific mean/std if you choose PER_BASIN_NORM = True."""
    stats = {}
    with h5py.File(train_h5_path, "r") as f:
        y = f["target_data"][:]                 # (N, H)
        basins = [b.decode("ascii") for b in f["sample_2_basin"][:]]

    basin_to_y = {}
    for b, yy in zip(basins, y):
        basin_to_y.setdefault(b, []).append(yy)
    for b, ys in basin_to_y.items():
        ys = np.vstack(ys)
        stats[b] = {"mean": float(ys.mean()), "std": float(ys.std())}
    return stats

# --------------------------------------------------------------------
def _build_det_model(cfg, device):
    if cfg['forcing_source'] == 'all':
        dyn_in = 15 # depend on whether it's multisources. If single source, set it to 5. 
        in_size_dyn = dyn_in if (cfg['no_static'] or not cfg['concat_static']) else 42 # 42 for 3-source input, i.e., dyn_in=15
    else:
        dyn_in = 5 # depend on whether it's multisources. If single source, set it to 5. 
        in_size_dyn = dyn_in if (cfg['no_static'] or not cfg['concat_static']) else 32
    
    static_size  = 0 if cfg["no_static"] else (in_size_dyn - dyn_in)
    H = cfg["forecast_horizon"]

    name = cfg["model_name"]
    if name == 'seq2seq_lstm':
        return Seq2SeqLSTM(
            input_size   = in_size_dyn,
            hidden    = cfg['hidden_size'],
            horizon   = H,
            dropout   = cfg['dropout'],
        ).to(device)

    elif name == "encdec_lstm":
        return EncoderDecoderDetLSTM(
            past_features   = dyn_in,
            future_features = dyn_in,
            static_size     = static_size,
            hidden          = cfg["hidden_size"],
            horizon         = H,
            dropout         = cfg["dropout"]
        ).to(device)

    elif name == "seq2seq_ssm":
        return seq2seq_ssm(
            d_input      = in_size_dyn, 
            d_model      = cfg['d_model'],     
            n_layers     = cfg['n_layers'],      
            cfg          = cfg,                 
            horizon      = cfg['forecast_horizon'],
            time_emb_dim = cfg['time_emb_dim'],
            static_dim   = cfg.get('static_dim', 27),
            dropout      = cfg['ssm_dropout']
        ).to(device)

    else:
        raise ValueError(f"Unsupported deterministic model {name}")



def evaluate(cfg: dict):
    """
    * diffusion_lstm -> DDIM ensemble
    * all other model names -> single deterministic forward pass

    Optional profiling mode:
        cfg["profile_runtime"] = True
        cfg["profile_max_batches"] = 10
        cfg["profile_num_samples"] = 1
        cfg["profile_csv_name"] = "runtime_profile_ddim5.csv"
    """
    device   = cfg["DEVICE"]
    run_dir  = Path(cfg["run_dir"])

    # profiling switches
    profile_runtime     = bool(cfg.get("profile_runtime", True))
    profile_max_batches = int(cfg.get("profile_max_batches", 10))
    profile_num_samples = int(cfg.get("profile_num_samples", 1)) 
    profile_csv_name    = cfg.get("profile_csv_name", f"{cfg['model_name']}_runtime_profile_ddpmtrain_ddim{cfg['ddim_steps']}.csv")
    
    # AMP settings
    use_amp = cfg.get("use_amp", True) and torch.cuda.is_available()
    amp_dtype = torch.float16

    if "lstm" in cfg["model_name"].lower():
        model_pt = run_dir / "best_model.pt"
    else:
        model_pt = run_dir / "model_epoch60.pt"
    cache = run_dir / "predictions_checkpoint.npz"

    # --- data paths ---
    if cfg.get("h5_dir"):
        base     = Path(cfg["h5_dir"]) / "data"
        train_h5 = base / "train/train_data.h5"
        test_h5  = base / "test/test_data.h5"
        db_path  = Path(cfg["h5_dir"]) / "attributes.db"
    else:
        train_h5 = run_dir / "data/train/train_data.h5"
        test_h5  = run_dir / "data/test/test_data.h5"
        db_path  = run_dir / "attributes.db"

    train_stats = load_train_q_stats(train_h5)

    # fast-return only for normal prediction mode
    if (not profile_runtime) and cache.exists():
        return np.load(cache, allow_pickle=True)

    # --- build model ---
    is_diffusion = (
        cfg["model_name"] in [
            "diffusion_lstm", "diffusion_lstm_fast", "diffusion_unet", "diffusion_ssm",
            "decoder_only_ssm", "decoder_only_lstm", "decoder_only_lstm_fast",
            "diffusion_ssm_unet", "diffusion_ssm_lstm"
        ]
    )

    if is_diffusion:
        if cfg['forcing_source'] == 'all':
            dyn_in = 15
            in_size_dyn = dyn_in if (cfg['no_static'] or not cfg['concat_static']) else 42
        else:
            dyn_in = 5
            in_size_dyn = dyn_in if (cfg['no_static'] or not cfg['concat_static']) else 32

        static_size = 0 if cfg["no_static"] else (in_size_dyn - dyn_in)

        if cfg["model_name"] in ["diffusion_lstm", "diffusion_lstm_fast"]:
            encoder = GenericLSTM(
                input_size=in_size_dyn,
                hidden_size=cfg["hidden_size"],
                dropout=cfg["dropout"],
                init_forget_bias=cfg['initial_forget_gate_bias'],
                batch_first=True
            )
            decoder = GenericLSTM(
                input_size=1 + dyn_in + 27,
                hidden_size=cfg["hidden_size"],
                dropout=cfg["dropout"],
                init_forget_bias=cfg['initial_forget_gate_bias'],
                batch_first=True
            )
            if cfg["model_name"] == "diffusion_lstm_fast":
                model = DiffusionLSTMFast(
                    encoder=encoder,
                    decoder=decoder,
                    hidden_size=cfg["hidden_size"],
                    time_emb_dim=cfg.get("time_emb_dim", 256),
                    decoder_name='lstm',
                    prediction_type='velocity'
                ).to(device)
            else:
                model = EncoderDecoderDiffusionWrapper(
                    encoder=encoder,
                    decoder=decoder,
                    hidden_size=cfg["hidden_size"],
                    time_emb_dim=cfg.get("time_emb_dim", 256),
                    decoder_name='lstm',
                    prediction_type='velocity'
                ).to(device)

        elif cfg["model_name"] == "diffusion_unet":
            encoder = GenericLSTM(
                input_size=in_size_dyn,
                hidden_size=cfg['hidden_size'],
                num_layers=cfg.get('lstm_nlayers', 1),
                dropout=cfg['dropout'],
                init_forget_bias=cfg['initial_forget_gate_bias'],
                batch_first=True
            )
            decoder = unet_attention_film_v3(
                input_dim=1,
                hidden_dim=cfg["unet_nfeat"],
                static_dim=cfg.get("static_dim", 27),
                h_lstm_dim=cfg["hidden_size"],
                dropout=cfg.get("dropout", 0.1),
                attn_heads=cfg.get("attn_heads", 4),
            )
            model = EncoderDecoderDiffusionWrapper(
                encoder=encoder,
                decoder=decoder,
                hidden_size=cfg['hidden_size'],
                time_emb_dim=cfg['time_emb_dim'],
                decoder_name='unet',
                prediction_type='velocity'
            ).to(cfg['DEVICE'])

        elif cfg['model_name'] == 'diffusion_ssm':
            encoder = ssm_encoder(
                d_input=in_size_dyn,
                d_model=cfg['d_model'],
                n_layers=cfg['n_layers'],
                cfg=cfg,
                static_dim=cfg.get('static_dim', 27),
                dropout=cfg['ssm_dropout'],
                pool_type=cfg['pool_type'],
                horizon=cfg['forecast_horizon']
            )
            decoder = ssm_v2(
                d_input=1 + dyn_in + 27,
                d_model=cfg['d_model'],
                n_layers=cfg['n_layers'],
                cfg=cfg,
                time_emb_dim=cfg['time_emb_dim'],
                enc_hidden_dim=cfg['d_model'],
                static_dim=cfg.get('static_dim', 27),
                dropout=cfg['ssm_dropout'],
                time_full=False,
                enc_full=False,
                static_full=False
            )
            model = EncoderDecoderDiffusionWrapper(
                encoder=encoder,
                decoder=decoder,
                hidden_size=cfg['hidden_size'],
                time_emb_dim=cfg['time_emb_dim'],
                decoder_name='ssm',
                prediction_type='velocity'
            ).to(cfg['DEVICE'])

        elif cfg['model_name'] == 'decoder_only_ssm':
            model = decoder_only_ssm(
                d_input=in_size_dyn,
                d_model=cfg['d_model'],
                n_layers=cfg['n_layers'],
                cfg=cfg,
                horizon=cfg['forecast_horizon'],
                time_emb_dim=cfg['time_emb_dim'],
                static_dim=cfg.get('static_dim', 27),
                dropout=cfg['ssm_dropout'],
                time_full=True
            ).to(cfg['DEVICE'])

        elif cfg['model_name'] in ['decoder_only_lstm', 'decoder_only_lstm_fast']:
            cls = decoder_only_lstm_fast if cfg['model_name'] == 'decoder_only_lstm_fast' else decoder_only_lstm
            model = cls(
                d_input=in_size_dyn,
                hidden_size=cfg['hidden_size'],
                cfg=cfg,
                horizon=cfg['forecast_horizon'],
                time_emb_dim=cfg['time_emb_dim'],
                static_dim=cfg.get('static_dim', 27),
            ).to(cfg['DEVICE'])

        elif cfg['model_name'] == 'diffusion_ssm_unet':
            encoder = ssm_encoder(
                d_input=in_size_dyn,
                d_model=cfg['d_model'],
                n_layers=cfg['n_layers'],
                cfg=cfg,
                static_dim=cfg.get('static_dim', 27),
                dropout=cfg['ssm_dropout'],
            )
            decoder = unet_attention_film_v2(
                input_dim=1,
                hidden_dim=cfg["unet_nfeat"],
                static_dim=cfg.get("static_dim", 27),
                h_lstm_dim=cfg["hidden_size"],
                dropout=cfg.get("dropout", 0.1),
                attn_heads=cfg.get("attn_heads", 4),
            )
            model = EncoderDecoderDiffusionWrapper(
                encoder=encoder,
                decoder=decoder,
                hidden_size=cfg['hidden_size'],
                time_emb_dim=cfg['time_emb_dim'],
                decoder_name='unet',
                prediction_type='velocity'
            ).to(cfg['DEVICE'])

        elif cfg['model_name'] == 'diffusion_ssm_lstm':
            encoder = ssm_encoder(
                d_input=in_size_dyn,
                d_model=cfg['d_model'],
                n_layers=cfg['n_layers'],
                cfg=cfg,
                static_dim=cfg.get('static_dim', 27),
                dropout=cfg['ssm_dropout'],
            )
            decoder = GenericLSTM(
                input_size=2,
                hidden_size=cfg['hidden_size'],
                num_layers=cfg.get('lstm_nlayers', 1),
                dropout=cfg['dropout'],
                init_forget_bias=cfg['initial_forget_gate_bias'],
                batch_first=True
            )
            model = EncoderDecoderDiffusionWrapper(
                encoder=encoder,
                decoder=decoder,
                hidden_size=cfg['hidden_size'],
                time_emb_dim=cfg['time_emb_dim'],
                decoder_name='lstm',
                prediction_type='velocity'
            ).to(cfg['DEVICE'])

    else:
        model = _build_det_model(cfg, device)

    # --- load checkpoint ---
    ckpt = torch.load(model_pt, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    #model = torch.compile(model)

    ema = None
    if isinstance(ckpt, dict) and "ema" in ckpt:
        from torch_ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        ema.load_state_dict(ckpt["ema"])

    model.eval()
    
    '''
    # for s4d-optimized code only
    with torch.no_grad():
        K = model.blocks[0].kernel._compute_kernel(370)
        energy = K.pow(2).cumsum(dim=-1) / K.pow(2).sum(dim=-1, keepdim=True)
        print("90%  energy at:", (energy < 0.90).sum(dim=-1).float().mean().item())
        print("99%  energy at:", (energy < 0.99).sum(dim=-1).float().mean().item())
        print("99.9% energy at:", (energy < 0.999).sum(dim=-1).float().mean().item())
    '''
    
    # --- data loader ---
    basins_all = get_basin_list(cfg["camels_root"])
    test_ds = CamelsH5(
        test_h5,
        basins_all,
        str(db_path),
        concat_static=cfg["concat_static"],
        no_static=cfg["no_static"],
        model_name=cfg["model_name"],
        forecast_horizon=cfg["forecast_horizon"],
        include_dates=True,
    )

    from torch.utils.data import get_worker_info
    from multiprocessing import get_context

    def _worker_init_fn(_):
        wi = get_worker_info()
        if hasattr(wi, "dataset") and hasattr(wi.dataset, "_h5"):
            wi.dataset._h5 = None

    def _make_loader(dataset, batch_size, shuffle, num_workers):
        nw = int(num_workers)
        kw = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=(nw > 0),
            persistent_workers=(nw > 0),
            drop_last=False,
        )
        if nw > 0:
            kw["multiprocessing_context"] = get_context("fork")
            kw["prefetch_factor"] = 1
            kw["worker_init_fn"] = _worker_init_fn
        return DataLoader(**kw)

    test_loader = _make_loader(test_ds, cfg.get("batch_size", 256), False, cfg["num_workers"])

    # --- evaluation storage ---
    all_preds, all_tgts, all_basin_ids, all_dates, all_ens = [], [], [], [], []
    runtime_rows = []

    def _denorm(arr, b_ids):
        if PER_BASIN_NORM:
            m = torch.tensor([train_stats[b]["mean"] for b in b_ids], device=device).unsqueeze(1)
            s = torch.tensor([train_stats[b]["std"] for b in b_ids], device=device).unsqueeze(1)
        else:
            m = torch.tensor(SCALAR["QObs(mm/d)_mean"], device=device)
            s = torch.tensor(SCALAR["QObs(mm/d)_std"], device=device)
        return arr * s + m

    context_mgr = ema.average_parameters() if ema is not None else torch.no_grad()

    with context_mgr:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="batches")):
                if profile_runtime and batch_idx >= profile_max_batches:
                    break

                if cfg["no_static"]:
                    x_d, y_t, q_m, q_s, basin_batch, date_batch = batch
                    static_attrs = None
                else:
                    x_d, static_attrs, y_t, q_m, q_s, basin_batch, date_batch = batch
                    static_attrs = static_attrs.to(device)

                x_d = x_d.to(device)
                y_t = y_t.to(device)
                fh = cfg["forecast_horizon"]

                def denorm(a):
                    return _denorm(a, basin_batch)

                nldas_idx  = [0,  3,  6,  9, 12]
                maurer_idx = [1,  4,  7, 10, 13]
                daymet_idx = [2,  5,  8, 11, 14]
                idx_map = {
                    'nldas': nldas_idx,
                    'maurer': maurer_idx,
                    'daymet': daymet_idx
                }

                which = cfg['forcing_source']
                if cfg['forcing_source'] != 'all':
                    x_d = x_d[:, :, idx_map[which]]

                # --- diffusion models ---
                if is_diffusion:
                    x_past = x_d[:, :-fh, :]

                    if cfg['model_name'] in ['decoder_only_ssm', 'decoder_only_lstm', 'decoder_only_lstm_fast']:
                        future_prec = x_d[:, -fh+1:, :]
                    else:
                        future_prec = x_d[:, -fh:, :]

                    if (not cfg['no_static']) and cfg['concat_static'] and cfg['model_name'] not in ['decoder_only_ssm', 'decoder_only_lstm', 'decoder_only_lstm_fast']:
                        stat_p = static_attrs.unsqueeze(1).repeat(1, x_past.size(1), 1)
                        x_past = torch.cat([x_past, stat_p], dim=-1)

                    stat_f = static_attrs.unsqueeze(1).repeat(1, future_prec.size(1), 1)

                    ens = []
                    num_samples = profile_num_samples if profile_runtime else cfg.get("num_samples", 50)

                    # Encode past ONCE and reuse across all ensemble members.
                    # SSM: avoids re-running 6-layer S4D FFT over 365-step past 50 times.
                    # LSTM: avoids re-running past LSTM encoding 50 times.
                    use_fft_cache = (
                        cfg["model_name"] == "decoder_only_ssm"
                        and hasattr(model, "encode_past_fft")
                        and not profile_runtime
                    )
                    use_lstm_cache = (
                        cfg["model_name"] in ["decoder_only_lstm", "decoder_only_lstm_fast"]
                        and hasattr(model, "encode_past_lstm")
                        and not profile_runtime
                    )
                    if use_fft_cache:
                        with torch.no_grad():
                            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                                past_fft_cache = model.encode_past_hybrid(
                                    x_past=x_past,
                                    x_future=future_prec,
                                    static_attr=stat_f,
                                )
                        past_lstm_cache = None
                    elif use_lstm_cache:
                        with torch.no_grad():
                            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                                past_lstm_cache = model.encode_past_lstm(
                                    x_past=x_past,
                                    x_future=future_prec,
                                    static_attr=stat_f,
                                )
                        past_fft_cache = None
                    else:
                        past_fft_cache = None
                        past_lstm_cache = None

                    # Encode past ONCE for encoder-decoder diffusion models
                    # (e.g. diffusion_lstm) and reuse across all ensemble
                    # members. Avoids re-running the encoder over the full
                    # past sequence for every sample.
                    use_encdec_cache = (
                        hasattr(model, "encode_past")
                        and hasattr(model, "encoder")
                        and not profile_runtime
                    )
                    if use_encdec_cache:
                        with torch.no_grad():
                            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                                encdec_state_cache = model.encode_past(x_past=x_past)
                    else:
                        encdec_state_cache = None

                    for s_idx in range(num_samples):

                        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                            if profile_runtime:
                                if not hasattr(model, "sample_ddim"):
                                    raise AttributeError(
                                        f"{cfg['model_name']} does not have sample_ddim()."
                                    )

                                samp, summary = model.sample_ddim_fast_fft_profiled(
                                    x_past=x_past,
                                    static_attributes=stat_f,
                                    future_pcp=future_prec,
                                    num_steps=cfg.get("ddim_steps"),
                                    eta=0.0,
                                    ddpm_n_steps=cfg.get("ddpm_train_steps"),
                                    ddpm_beta_start=cfg.get("ddpm_beta_start", 1e-4),
                                    ddpm_beta_end=cfg.get("ddpm_beta_end", 2e-2),
                                )

                                if s_idx == 0:
                                    runtime_rows += flatten_summary(cfg["model_name"], summary)
                            else:
                                if use_fft_cache:
                                    samp = model.sample_ddim_hybrid(
                                        x_past=x_past,
                                        static_attributes=stat_f,
                                        future_pcp=future_prec,
                                        num_steps=cfg.get("ddim_steps"),
                                        eta=0.0,
                                        ddpm_n_steps=cfg.get("ddpm_train_steps"),
                                        ddpm_beta_start=cfg.get("ddpm_beta_start", 1e-4),
                                        ddpm_beta_end=cfg.get("ddpm_beta_end", 2e-2),
                                        past_hybrid_cache=past_fft_cache,
                                    )
                                elif use_lstm_cache:
                                    samp = model.sample_ddim_fast(
                                        x_past=x_past,
                                        static_attributes=stat_f,
                                        future_pcp=future_prec,
                                        num_steps=cfg.get("ddim_steps"),
                                        eta=0.0,
                                        ddpm_n_steps=cfg.get("ddpm_train_steps"),
                                        ddpm_beta_start=cfg.get("ddpm_beta_start", 1e-4),
                                        ddpm_beta_end=cfg.get("ddpm_beta_end", 2e-2),
                                        past_state_cache=past_lstm_cache,
                                    )
                                else:
                                    samp = model.sample_ddim(
                                        x_past=x_past,
                                        static_attributes=stat_f,
                                        future_pcp=future_prec,
                                        num_steps=cfg.get("ddim_steps"),
                                        eta=0.0,
                                        ddpm_n_steps=cfg.get("ddpm_train_steps"),
                                        ddpm_beta_start=cfg.get("ddpm_beta_start", 1e-4),
                                        ddpm_beta_end=cfg.get("ddpm_beta_end", 2e-2),
                                        encoded_state=encdec_state_cache,
                                    )
                        ens.append(samp)

                    ens = torch.stack(ens, dim=0)      # (S,B,H)
                    ens_d = denorm(ens)
                    preds = ens_d.mean(dim=0)          # (B,H)
                    all_ens.append(ens_d.cpu().numpy().transpose(1, 0, 2))

                # --- deterministic models ---
                else:
                    x_past = x_d[:, :-fh, :]
                    if cfg['model_name'] == 'encdec_lstm':
                        future_prec = x_d[:, -fh:, :]
                    elif cfg['model_name'] in ['seq2seq_ssm', 'seq2seq_lstm']:
                        future_prec = x_d[:, -fh+1:, :]

                    if (not cfg['no_static']) and cfg['concat_static']:
                        stat_p = static_attrs.unsqueeze(1).repeat(1, x_past.size(1), 1)
                        stat_f = static_attrs.unsqueeze(1).repeat(1, future_prec.size(1), 1)
                        x_past = torch.cat([x_past, stat_p], dim=-1)
                        future_prec = torch.cat([future_prec, stat_f], dim=-1)

                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        preds = model(
                            x_past,
                            future_prec,
                            None if cfg['concat_static'] else static_attrs
                        ).squeeze(-1)

                    preds = denorm(preds)
                    all_ens.append(None)

                all_preds.append(preds.cpu().numpy())
                all_tgts.append(y_t.cpu().numpy())
                all_basin_ids.extend(basin_batch)
                all_dates.extend(pd.to_datetime(date_batch))

    # --- save runtime profile if requested ---
    if profile_runtime and len(runtime_rows) > 0:
        runtime_df = pd.DataFrame(runtime_rows)
        runtime_df = (
            runtime_df.groupby(["model", "component"], as_index=False)
            .agg({
                "seconds": "sum",
                "calls": "sum"
            })
        )
        runtime_df["avg_per_call"] = runtime_df["seconds"] / runtime_df["calls"]
        total_by_model = runtime_df.groupby("model")["seconds"].transform("sum")
        runtime_df["percent_total"] = 100.0 * runtime_df["seconds"] / total_by_model

        runtime_csv = Path(profile_csv_name) if Path(profile_csv_name).is_absolute() \
              else Path("/data/home/yihan/diffusion_ssm/runtime_profile_csv_ddpmtrain_1000") / profile_csv_name
        runtime_df.to_csv(runtime_csv, index=False)
        print(f"[INFO] Saved runtime profile to {runtime_csv}")

    # --- save predictions ---
    preds_arr = np.vstack(all_preds)
    tgts_arr  = np.vstack(all_tgts)
    bas       = np.array(all_basin_ids, dtype="U32")
    dts       = np.array(all_dates, dtype="datetime64[ns]")
    ens_arr   = None if all_ens[0] is None else np.vstack(all_ens)

    SAVE = np.savez
    
    if ens_arr is not None:
        if "lstm" in cfg["model_name"].lower():
            npz_path = run_dir / f"ensembles_epochbest_ddim{cfg['ddim_steps']}.npz"
        else:
            npz_path = run_dir / f"ensembles_epoch60_ddim{cfg['ddim_steps']}_optim_hybrid.npz" #todo

        SAVE(
            npz_path,
            basins=bas,
            dates=dts,
            obs=tgts_arr,
            ens=ens_arr
        )
        print(f"[INFO] Saved FULL ENSEMBLES (N,S,H) to {npz_path}")
    else:
        if "lstm" in cfg["model_name"].lower():
            npz_path = run_dir / "deterministic_epoch30.npz"
        else:
            npz_path = run_dir / "deterministic_epoch49.npz"

        SAVE(
            npz_path,
            basins=bas,
            dates=dts,
            obs=tgts_arr,
            preds=preds_arr
        )
        print(f"[INFO] Saved deterministic predictions (N,H) to {npz_path}")