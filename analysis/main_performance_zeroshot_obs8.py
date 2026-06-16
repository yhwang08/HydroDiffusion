# main_performance_zeroshot_obs8.py

'''
python /data/home/yihan/diffusion_ssm/analysis/main_performance_zeroshot_obs8.py \
  --per_basin_dir /data/home/yihan/diffusion_ssm/zeroshot_outputs/per_basin_csv_lstm \
  --out_dir /data/home/yihan/diffusion_ssm/analysis/analysis/stats_gefs_lstm_all \
  --min_lead 0 \
  --max_lead 7
'''

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import pandas as pd
import pdb

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name == "analysis" else THIS_FILE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "papercode")]

from performance_functions import (
    bias, flow_duration_curve, get_quant,
    high_flows, low_flows, nse, alpha_nse, beta_nse,
    kge, stdev_rat, zero_freq, FHV, FLV, mass_balance
)


def collect_basin_csvs(per_basin_dir: Path, subset: List[str] | None) -> Dict[str, Path]:
    files = sorted(per_basin_dir.glob("*_preds.csv"))
    found = {fp.stem.replace("_preds", ""): fp for fp in files}

    if subset:
        subset = set(subset)
        found = {b: p for b, p in found.items() if b in subset}

    return found


def _to_float(x):
    try:
        return float(np.asarray(x).squeeze())
    except Exception:
        return np.nan


def _as_1d_array(x):
    return np.atleast_1d(np.asarray(x))


def compute_metrics(df_single: pd.DataFrame) -> Dict[str, float]:
    if len(df_single) < 2:
        return {}

    try:
        obs5, sim5 = get_quant(df_single, 0.05)
        obs95, sim95 = get_quant(df_single, 0.95)
        obs0, sim0 = zero_freq(df_single)
        obsH, simH = high_flows(df_single)
        obsL, simL = low_flows(df_single)

        e_fhv_01 = FHV(df_single, 0.1)
        e_flv = FLV(df_single, 0.3)

        e_nse = nse(df_single)
        e_nse_alpha = alpha_nse(df_single)
        e_nse_beta = beta_nse(df_single)
        e_kge, r, a, b = kge(df_single)

        mb_tot, mb_pos, mb_neg = mass_balance(df_single)
        e_bias = bias(df_single)
        e_stdev = stdev_rat(df_single)

    except Exception:
        return {}

    try:
        obsFDC, simFDC = flow_duration_curve(df_single)
        obsFDC = _as_1d_array(obsFDC)
        simFDC = _as_1d_array(simFDC)
        fdc_obs_p50 = float(np.percentile(obsFDC, 50)) if obsFDC.size else np.nan
        fdc_sim_p50 = float(np.percentile(simFDC, 50)) if simFDC.size else np.nan
    except Exception:
        fdc_obs_p50 = np.nan
        fdc_sim_p50 = np.nan

    return {
        "nse": _to_float(e_nse),
        "alpha_nse": _to_float(e_nse_alpha),
        "beta_nse": _to_float(e_nse_beta),
        "kge": _to_float(e_kge),
        "kge_r": _to_float(r),
        "kge_alpha": _to_float(a),
        "kge_beta": _to_float(b),
        "fhv_01": _to_float(e_fhv_01),
        "flv": _to_float(e_flv),
        "massbias_total": _to_float(mb_tot),
        "massbias_pos": _to_float(mb_pos),
        "massbias_neg": _to_float(mb_neg),
        "bias": _to_float(e_bias),
        "stdev": _to_float(e_stdev),
        "obs5": _to_float(obs5),
        "sim5": _to_float(sim5),
        "obs95": _to_float(obs95),
        "sim95": _to_float(sim95),
        "obs0": _to_float(obs0),
        "sim0": _to_float(sim0),
        "obsL": _to_float(obsL),
        "simL": _to_float(simL),
        "obsH": _to_float(obsH),
        "simH": _to_float(simH),
        "fdc_obs_p50": _to_float(fdc_obs_p50),
        "fdc_sim_p50": _to_float(fdc_sim_p50),
    }


def make_pairs(df: pd.DataFrame, lead: int) -> pd.DataFrame:
    pred_col = f"pred_lead{lead}" if f"pred_lead{lead}" in df.columns else f"lead{lead}"
    obs_col = f"obs_lead{lead}"

    qsim = pd.to_numeric(df[pred_col], errors="coerce")
    qobs = pd.to_numeric(df[obs_col], errors="coerce")

    pairs = pd.DataFrame({"qsim": qsim, "qobs": qobs})
    pairs = pairs.replace([np.inf, -np.inf], np.nan)
    pairs = pairs.dropna(subset=["qsim", "qobs"]).copy()
    pairs["qsim"] = pairs["qsim"].clip(lower=0.0)

    return pairs


def inspect_valid_leads(df: pd.DataFrame, min_lead: int, max_lead: int) -> Dict[int, int]:
    valid = {}

    df.columns = df.columns.astype(str).str.strip()

    for lead in range(min_lead, max_lead + 1):
        pred_col = f"pred_lead{lead}" if f"pred_lead{lead}" in df.columns else f"lead{lead}"
        obs_col = f"obs_lead{lead}"

        if pred_col not in df.columns or obs_col not in df.columns:
            valid[lead] = 0
            continue

        qsim = pd.to_numeric(df[pred_col], errors="coerce")
        qobs = pd.to_numeric(df[obs_col], errors="coerce")

        m = np.isfinite(qsim.to_numpy()) & np.isfinite(qobs.to_numpy())
        valid[lead] = int(m.sum())

    return valid


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--per_basin_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--basin_list", type=str, default=None)
    ap.add_argument("--min_lead", type=int, default=0)
    ap.add_argument("--max_lead", type=int, default=7)
    ap.add_argument("--min_pairs", type=int, default=2)

    args = ap.parse_args()

    per_basin_dir = Path(args.per_basin_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subset = None
    if args.basin_list:
        subset = [
            ln.strip()
            for ln in Path(args.basin_list).read_text().splitlines()
            if ln.strip()
        ]

    basin_files = collect_basin_csvs(per_basin_dir, subset)

    if not basin_files:
        print(f"[WARN] No *_preds.csv files found in {per_basin_dir}")
        return

    print(f"[INFO] Found {len(basin_files)} basin CSV files")

    rows_by_lead = {lead: [] for lead in range(args.min_lead, args.max_lead + 1)}
    valid_summary = []

    n_valid_basins = 0
    n_skipped_basins = 0

    for basin, csv_path in basin_files.items():
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.astype(str).str.strip()

            valid_counts = inspect_valid_leads(df, args.min_lead, args.max_lead)
            total_valid_pairs = sum(valid_counts.values())

            valid_summary.append({
                "basin": basin,
                "csv": str(csv_path),
                **{f"n_pairs_lead{lead}": valid_counts[lead]
                   for lead in range(args.min_lead, args.max_lead + 1)},
                "total_valid_pairs": total_valid_pairs,
            })

            if total_valid_pairs < args.min_pairs:
                n_skipped_basins += 1
                print(f"[SKIP_BASIN] {basin}: no valid predictions/observations")
                continue

            n_valid_basins += 1

            for lead in range(args.min_lead, args.max_lead + 1):
                if valid_counts[lead] < args.min_pairs:
                    continue

                pairs = make_pairs(df, lead)

                if len(pairs) < args.min_pairs:
                    continue

                mets = compute_metrics(pairs)

                if not mets:
                    continue

                mets.update({
                    "basin": basin,
                    "lead_index": lead,
                    "lead_days_ahead": lead,
                    "n_pairs": int(len(pairs)),
                })

                rows_by_lead[lead].append(mets)

        except Exception as e:
            n_skipped_basins += 1
            print(f"[FAIL_BASIN] {basin}: {e}")

    # Save valid-watershed inspection table
    df_valid = pd.DataFrame(valid_summary)
    valid_csv = out_dir / "valid_watershed_summary.csv"
    df_valid.to_csv(valid_csv, index=False)

    print(f"[INFO] Valid basins used: {n_valid_basins}")
    print(f"[INFO] Basins skipped: {n_skipped_basins}")
    print(f"[OK] Saved valid watershed summary: {valid_csv}")

    # Save per-lead metrics
    for lead, rows in rows_by_lead.items():
        df_stats = pd.DataFrame(rows)

        if df_stats.empty:
            print(f"[INFO] No valid stats for lead{lead}")
            continue

        mean_row = df_stats.mean(numeric_only=True).to_dict()
        med_row = df_stats.median(numeric_only=True).to_dict()

        mean_row.update({
            "basin": "mean",
            "lead_index": lead,
            "lead_days_ahead": lead,
            "n_pairs": int(df_stats["n_pairs"].sum()),
        })

        med_row.update({
            "basin": "median",
            "lead_index": lead,
            "lead_days_ahead": lead,
            "n_pairs": int(df_stats["n_pairs"].median()),
        })

        df_out = pd.concat(
            [df_stats, pd.DataFrame([mean_row, med_row])],
            ignore_index=True
        )

        front = ["basin", "lead_index", "lead_days_ahead", "n_pairs"]
        front = [c for c in front if c in df_out.columns]
        rest = [c for c in df_out.columns if c not in front]
        df_out = df_out[front + rest]

        out_csv = out_dir / f"gefs_eval_lead{lead}.csv"
        df_out.to_csv(out_csv, index=False)

        print(
            f"[OK] Saved {out_csv} "
            f"(basins: {df_stats['basin'].nunique()}, pairs total: {int(df_stats['n_pairs'].sum())})"
        )

    print("[DONE] All evaluation results saved.")


if __name__ == "__main__":
    main()