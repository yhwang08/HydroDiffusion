import os
import sys
import numpy as np
import pandas as pd

from performance_functions import (
    baseflow_index, bias, flow_duration_curve, get_quant,
    high_flows, low_flows, nse, alpha_nse, beta_nse,
    kge, stdev_rat, zero_freq, FHV, FLV, mass_balance
)

'''
python main_performance_full_evaluation.py diffusionlstm_ddim10_rerun /data/home/yihan/diffusion_ssm/runs/run_2207_1832_seed3407/ensembles_epochbest_ddim10.npz
python main_performance_full_evaluation.py hydrodiffusion_ddimpmtrain_ddim10 /data/rdl/yihan/diffusion_ssm/runs/run_2204_0533_seed3407/ensembles_epoch60_ddim10.npz
python main_performance_full_evaluation.py hydrodiffusion_ddpmtrain1000_optim_fixed /data/rdl/yihan/diffusion_ssm/runs/run_1605_0026_seed34/ensembles_epoch60_ddim10_optim_fixed.npz
python main_performance_full_evaluation.py decoder_only_lstm_fast /data/home/yihan/diffusion_ssm/runs/run_1509_0214_seed3407/ensembles_epochbest_ddim10.npz
'''

# ============================================================
# Probabilistic helper functions
# ============================================================

def crps_ensemble_batch(X, y, batch=4000):
    """Unbiased ensemble CRPS."""
    N, S = X.shape
    out = np.empty(N, dtype=np.float64)
    for i0 in range(0, N, batch):
        i1 = min(N, i0 + batch)
        Xi = X[i0:i1]
        yi = y[i0:i1, None]
        term1 = np.mean(np.abs(Xi - yi), axis=1)
        diffs = np.abs(Xi[:, :, None] - Xi[:, None, :])
        term2 = 0.5 * np.mean(diffs, axis=(1, 2))
        out[i0:i1] = term1 - term2
    return out

def quantile_reliability(ens_vec, obs_vec, qs=None):
    if qs is None:
        qs = np.linspace(0.05, 0.95, 19)
    E = np.asarray(ens_vec, float)
    y = np.asarray(obs_vec, float).ravel()
    qs = np.asarray(qs, float).ravel()
    valid = np.isfinite(y) & np.isfinite(E).all(axis=1)
    E = E[valid]; y = y[valid]
    pred_q = np.quantile(E, qs, axis=1)
    emp_freq = (y[None, :] <= pred_q).mean(axis=1)
    return pd.DataFrame({"q": qs, "emp_freq": emp_freq, "n": np.full(qs.shape, y.size, int)})

def pit_values(ens_vec, obs_vec):
    E = np.asarray(ens_vec, float)
    y = np.asarray(obs_vec, float).ravel()
    return (E <= y[:, None]).sum(axis=1) / E.shape[1]

def roc_auc_from_prob(p, y):
    order = np.argsort(-p)
    y_sorted = y[order].astype(np.float64)
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    P, N = tp[-1], fp[-1]
    if P == 0 or N == 0:
        return np.nan
    return np.trapz(y=tp/P, x=fp/N)

def event_reliability(p, y, bins=np.linspace(0,1,11)):
    inds = np.digitize(p, bins) - 1
    out = []
    for b in range(len(bins)-1):
        mask = inds == b
        if mask.any():
            out.append({
                "bin_left": bins[b], "bin_right": bins[b+1],
                "count": int(mask.sum()),
                "mean_p": float(p[mask].mean()),
                "obs_freq": float(y[mask].mean())
            })
    return pd.DataFrame(out)

# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python main_performance_full_evaluation.py <experiment_name> <npz_path>")
        # A det example: python main_performance_full_evaluation.py seq2seq_ssm /home/yihan/diffusion_ssm/runs/run_2910_1858_seed3407/deterministic_epoch49.npz
        # HydroDiffusion: python main_performance_full_evaluation.py hydrodiffusion_ddim3 /home/yihan/diffusion_ssm/runs/run_2507_2120_seed3407/ensembles_epoch60_ddim3.npz

        sys.exit(1)

    experiment = sys.argv[1]
    npz_path   = sys.argv[2]
    out_root = "analysis/ensemble_stats"
    os.makedirs(out_root, exist_ok=True)

    # ---- load npz ----
    print(f"[INFO] Loading {npz_path}")
    f = np.load(npz_path, allow_pickle=True)
    keys = list(f.keys())
    print(f"[INFO] Found keys: {keys}")
    basins = f["basins"]
    dates  = pd.to_datetime(f["dates"])
    obs    = f["obs"]
    
    if "preds" in keys:
        ens = f["preds"]
    elif "ens" in keys:
        ens = f["ens"]
    else:
        raise KeyError("NPZ must contain either 'preds' or 'ens'.")
    
    # Detect deterministic vs probabilistic
    # Deterministic = ens.ndim == 2 (N,H)
    deterministic = (ens.ndim == 2)
    print(f"[INFO] Detected {'deterministic' if deterministic else 'ensemble'} npz")
    
    if deterministic:
        # reshape to (N,1,H) for uniform handling
        ens = ens[:, None, :]
    
    N, S, H = ens.shape
    
    # ============================================================
    # Observation layout check
    # ============================================================
    print("=" * 70)
    print("[OBS ALIGNMENT CHECK]")
    print(f"  basins shape : {basins.shape}")
    print(f"  dates shape  : {dates.shape}")
    print(f"  obs shape    : {obs.shape}")
    print(f"  ens shape    : {ens.shape}")
    
    if obs.ndim == 2 and obs.shape == (N, H):
        OBS_MODE = "obs8"
        print("  OBS_MODE     : obs8 / lead-aligned observations")
        print("  Interpretation:")
        print("    obs[n, 0] = Q(init_date[n] + 0 days)")
        print("    obs[n, 1] = Q(init_date[n] + 1 day)")
        print("    ...")
        print(f"    obs[n, {H-1}] = Q(init_date[n] + {H-1} days)")
        print("  Lead-time metrics will use obs[idx, lead-1].")
    elif obs.ndim == 1 and obs.shape[0] == N:
        OBS_MODE = "obs1"
        print("  OBS_MODE     : obs1 / init-date-only observations")
        print("  Interpretation:")
        print("    obs[n] = Q(init_date[n]) only")
        print("  WARNING:")
        print("    This current code does NOT reconstruct Q(init_date + lead).")
        print("    Therefore lead > 0 metrics are NOT lead-time aligned unless you revise the obs lookup logic.")
    else:
        raise ValueError(
            f"Unsupported obs shape {obs.shape}. Expected (N,H)=({N},{H}) for obs8 "
            f"or (N,)={N} for obs1."
        )
    
    # Optional: show quick sample for sanity
    print("  Sample dates:")
    for i in range(min(3, N)):
        if OBS_MODE == "obs8":
            print(
                f"    row {i}: basin={basins[i]}, init={dates[i]}, "
                f"obs_lead0={obs[i,0]}, obs_lead{H-1}={obs[i,H-1]}"
            )
        else:
            print(
                f"    row {i}: basin={basins[i]}, init={dates[i]}, obs_init={obs[i]}"
            )
    print("=" * 70)

    # Detect deterministic vs probabilistic
    # Deterministic = ens.ndim == 2 (N,H)
    deterministic = (ens.ndim == 2)
    print(f"[INFO] Detected {'deterministic' if deterministic else 'ensemble'} npz")

    if deterministic:
        # reshape to (N,1,H) for uniform handling
        ens = ens[:, None, :]

    N, S, H = ens.shape
    unique_basins = np.unique(basins)
    basin_idx = {b: np.where(basins == b)[0] for b in unique_basins}

    # =====================================================
    # Deterministic metrics
    # =====================================================
    leadtime_stats = {lead: [] for lead in range(1, H+1)}

    for b in unique_basins:
        idx = basin_idx[b]
        for lead in range(1, H+1):
            y = obs[idx, lead-1].astype(float)
            x_mean = ens[idx, :, lead-1].mean(axis=1)
            df_single = pd.DataFrame({"qsim": x_mean.clip(min=0), "qobs": y}).dropna()
            if df_single.empty:
                continue

            obs5, sim5 = get_quant(df_single, 0.05)
            obs95, sim95 = get_quant(df_single, 0.95)
            obs0, sim0 = zero_freq(df_single)
            obsH, simH = high_flows(df_single)
            obsL, simL = low_flows(df_single)
            e_fhv = FHV(df_single, 0.1)
            e_flv = FLV(df_single, 0.3)
            e_nse = nse(df_single)
            e_nse_alpha = alpha_nse(df_single)
            e_nse_beta = beta_nse(df_single)
            e_kge, r, alpha, beta = kge(df_single)
            m_total, m_pos, m_neg = mass_balance(df_single)
            e_bias = bias(df_single)
            e_stdev = stdev_rat(df_single)
            obsFDC, simFDC = flow_duration_curve(df_single)

            leadtime_stats[lead].append({
                "basin": b, "lead_time": lead,
                "nse": e_nse, "alpha_nse": e_nse_alpha, "beta_nse": e_nse_beta,
                "kge": e_kge, "kge_r": r, "kge_alpha": alpha, "kge_beta": beta,
                "fhv_01": e_fhv, "flv": e_flv,
                "massbias_total": m_total, "massbias_pos": m_pos, "massbias_neg": m_neg,
                "bias": e_bias, "stdev": e_stdev,
                "obs5": obs5, "sim5": sim5, "obs95": obs95, "sim95": sim95,
                "obs0": obs0, "sim0": sim0, "obsL": obsL, "simL": simL,
                "obsH": obsH, "simH": simH, "obsFDC": obsFDC, "simFDC": simFDC
            })

    for lead, rows in leadtime_stats.items():
        df_stats = pd.DataFrame(rows)
        if df_stats.empty:
            continue
        mean_stats = df_stats.mean(numeric_only=True)
        median_stats = df_stats.median(numeric_only=True)
        mean_stats["basin"] = "mean"; median_stats["basin"] = "median"
        df_stats = pd.concat([df_stats, mean_stats.to_frame().T, median_stats.to_frame().T], ignore_index=True)
        out_csv = os.path.join(out_root, f"{experiment}_det_lead{lead}.csv")
        df_stats.to_csv(out_csv, index=False)
        print(f"[OK] Saved deterministic stats: {out_csv}")

    # =====================================================
    # Skip probabilistic metrics if deterministic
    # =====================================================
    if deterministic:
        print("[INFO] Deterministic run detected — skipping ensemble metrics.")
        print("All deterministic evaluation results saved.")
        return

    # =====================================================
    # Probabilistic metrics (for ensemble npz)
    # =====================================================
    basin_thresh = {b: np.nanpercentile(obs[basin_idx[b], :H].reshape(-1), 95.0) for b in unique_basins}
    prob_rows, rel_quant_rows, pit_rows, event_rel_rows = (
        {lead: [] for lead in range(1, H+1)} for _ in range(4)
    )

    for b in unique_basins:
        idx = basin_idx[b]; thr = basin_thresh[b]
        for lead in range(1, H+1):
            Y = obs[idx, lead-1].astype(float)
            X = ens[idx, :, lead-1].astype(float)

            crps_vec = crps_ensemble_batch(X, Y)
            crps_mean = float(np.nanmean(crps_vec))
            spread_std = X.std(axis=1)
            q05, q95 = np.quantile(X, [0.05, 0.95], axis=1)
            width_90 = q95 - q05

            qrel_df = quantile_reliability(X, Y); qrel_df.insert(0, "basin", b); qrel_df.insert(1, "lead", lead)
            pit_df = pd.DataFrame({"basin": b, "lead": lead, "pit": pit_values(X, Y)})

            y_evt = (Y > thr).astype(int)
            p_evt = (X > thr).mean(axis=1)
            auc = roc_auc_from_prob(p_evt, y_evt)
            rel_evt_df = event_reliability(p_evt, y_evt); rel_evt_df.insert(0, "basin", b); rel_evt_df.insert(1, "lead", lead)

            rel_quant_rows[lead].append(qrel_df)
            pit_rows[lead].append(pit_df)
            event_rel_rows[lead].append(rel_evt_df)

            prob_rows[lead].append({
                "basin": b, "lead_time": lead,
                "crps": crps_mean,
                "sharp_std": float(np.nanmean(spread_std)),
                "sharp_w90": float(np.nanmean(width_90)),
                "event_auc": auc, "event_threshold": thr, "n_samples": len(Y)
            })

    for lead in range(1, H+1):
        df_prob = pd.DataFrame(prob_rows[lead])
        if not df_prob.empty:
            mean_row = df_prob.mean(numeric_only=True)
            med_row = df_prob.median(numeric_only=True)
            mean_row["basin"], med_row["basin"] = "mean", "median"
            mean_row["lead_time"] = med_row["lead_time"] = lead
            df_prob = pd.concat([df_prob, mean_row.to_frame().T, med_row.to_frame().T])
            df_prob.to_csv(os.path.join(out_root, f"{experiment}_prob_lead{lead}.csv"), index=False)
            print(f"[OK] Saved probabilistic summary for lead {lead}")

    print("All deterministic + probabilistic evaluation results saved.")

if __name__ == "__main__":
    main()
