# papercode/datasets.py
from pathlib import PosixPath
from typing import List, Tuple
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import sqlite3

from .datautils import (
    load_attributes,
    load_discharge,
    load_forcing,
    load_forcing_multi,
    normalize_features_noprecip,
    normalize_features, 
    normalize_multi_features,
    reshape_data,
    load_scalar
)

SCALAR = load_scalar("/home/yihan/diffusion_ssm/global_scalar.json")

class CamelsTXT(Dataset):
    """PyTorch dataset for raw CAMELS text files with forcing data normalization."""

    def __init__(
        self,
        camels_root: PosixPath,
        basin: str,
        dates: List[pd.Timestamp],
        is_train: bool,
        seq_length: int = 365,
        forecast_horizon: int = 8,
        model_name: str = "lstm",
        with_attributes: bool = False,
        concat_static: bool = False,
        db_path: str = None,
        normalize_perbasin: bool = False,
    ):
        self.camels_root = camels_root
        self.basin = basin
        self.dates = dates
        self.period_start = self.dates[0]
        self.is_train = is_train
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.model_name = model_name.lower()
        self.with_attributes = with_attributes
        self.concat_static = concat_static
        self.db_path = db_path
        self.normalize_perbasin = normalize_perbasin

        # placeholders
        self.input_means = None
        self.input_stds = None
        self.output_mean = None
        self.output_std = None
        self.q_std = None

        # load & preprocess
        self.x, self.y = self._load_data()
        if self.with_attributes:
            self.attributes = self._load_attributes()
        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.with_attributes:
            if self.concat_static:
                static_rep = self.attributes.repeat((self.seq_length, 1))
                inp = torch.cat([self.x[idx], static_rep], dim=-1)
                return inp, self.y[idx]
            else:
                return self.x[idx], self.attributes, self.y[idx]
        else:
            return self.x[idx], self.y[idx]

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) Load forcing data from multiple sources and discharge
        df, area = load_forcing_multi(self.camels_root, self.basin)
        df["QObs(mm/d)"] = load_discharge(self.camels_root, self.basin, area)
    
        # 2) Slice to warmup and forecast window
        start = self.dates[0] - pd.Timedelta(days=self.seq_length - 1)
        end = self.dates[1] + pd.Timedelta(days=self.forecast_horizon)
        df = df[start:end]
    
        # 3) Extract and average PRCP columns (handles different cases)
        prcp_cols = [col for col in df.columns if col.lower().startswith("prcp") or col.lower().startswith("precip")]
        prcp = df[prcp_cols]
    
        # 4) Collect other variables from all sources
        var_list = ["srad", "tmax", "tmin", "vp"]
        other_cols = []
        for var in var_list:
            matched = [col for col in df.columns if col.lower().startswith(var)]
            other_cols.extend(matched)
    
        other = df[other_cols].values
        y = df["QObs(mm/d)"].values.reshape(-1, 1)
        print("basin: ", self.basin, area)
        
        # 5) Combine inputs
        X_raw = np.concatenate([prcp, other], axis=1)
    
        self.input_means = X_raw.mean(axis=0)
        self.input_stds = X_raw.std(axis=0)
        
        self.output_mean = y.mean()
        self.output_std = y.std()
        
        # 6) Normalize
        if self.normalize_perbasin:
            x = (X_raw - self.input_means) / self.input_stds
        else:
            x = normalize_multi_features(X_raw, 'inputs', SCALAR)
    
        # 7) Windowing
        x, y = reshape_data(x, y, seq_length=self.seq_length, horizon=self.forecast_horizon)
    
        # 8) Drop invalid if training
        if self.is_train:
            x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
            y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)
            if np.isnan(y).any():
                x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
                y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)
            self.q_std = np.std(y)
            #y = normalize_multi_features(y, 'output', SCALAR) # todo, normalize streamflow!
        # 9) To tensors
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        )

    def _load_attributes(self) -> torch.Tensor:
        # Only used if you REALLY want static attributes in CamelsTXT (rare in HDF5 workflows)
        df = load_attributes(self.db_path, [self.basin], drop_lat_lon=True)
        vals = df.loc[self.basin].values.astype(np.float32)
        return torch.from_numpy(vals)


class CamelsH5(Dataset):
    '''
    PyTorch dataset for pre-windowed CAMELS sequences stored in an HDF5 file.

    Parameters
    ----------
    cache : bool
        *True*  – read the whole H5 into memory once (fast, RAM-heavy)  
        *False* – lazy load each sample from disk (slow, RAM-light)
    '''

    def __init__(
        self,
        h5_file: PosixPath,
        basins: List[str],
        db_path: str,
        concat_static: bool = False,
        no_static: bool = False,
        model_name: str = "lstm",
        forecast_horizon: int = 8,
        include_dates: bool = False,
        perbasin_norm: bool = False,
        is_normalized: bool = True,
        cache: bool = False
    ):
        self.h5_file        = h5_file
        self.basins         = basins
        self.db_path        = db_path
        self.concat_static  = concat_static
        self.no_static      = no_static
        self.model_name     = model_name.lower()
        self.fh             = forecast_horizon
        self.include_dates  = include_dates
        self.perbasin_norm  = perbasin_norm
        self.is_normalized  = is_normalized
        self.cache          = cache

        # --- static attributes loaded once, stays the same ---
        self._load_attributes()

        # --- decide whether to preload ---
        if self.cache:
            (self.x, self.y, self.basin_map,
             self.q_means, self.q_stds, self.dates) = self._preload_h5()
            mask             = [i for i, b in enumerate(self.basin_map) if b in basins]
            self.x           = self.x[mask]
            self.y           = self.y[mask]
            self.q_means     = self.q_means[mask]
            self.q_stds      = self.q_stds[mask]
            self.basin_map   = [self.basin_map[i] for i in mask]
            self.dates       = [self.dates[i] for i in mask] if self.dates is not None else None
            self.num_samples = len(mask)
        else:
            with h5py.File(self.h5_file, "r") as f:
                raw = [b.decode("ascii") for b in f["sample_2_basin"][:]]
            self.valid_idx   = [i for i, b in enumerate(raw) if b in basins]
            self.num_samples = len(self.valid_idx)
            self._h5         = None  # opened lazily per worker process

    # ------------------------------------------------------------------
    # helper that actually loads h5 into numpy arrays (only if cache=True)
    # ------------------------------------------------------------------
    def _preload_h5(self):
        with h5py.File(self.h5_file, "r") as f:
            x       = f["input_data"][:]
            y       = f["target_data"][:]
            basins  = [b.decode("ascii") for b in f["sample_2_basin"][:]]
            q_mean  = f["q_means"][:]
            q_std   = f["q_stds"][:]
            dates   = ([d.decode("ascii") for d in f["dates"][:]]
                       if (self.include_dates and "dates" in f) else None)
        return x, y, basins, q_mean, q_std, dates

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_file, "r")
        return self._h5

    # ------------------------------------------------------------------
    # attribute loader
    # ------------------------------------------------------------------
    def _load_attributes(self):
        conn   = sqlite3.connect(self.db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()]
        if not tables:
            raise RuntimeError(f"No tables in {self.db_path}")
        tbl = tables[0]

        df_full = pd.read_sql(f"SELECT * FROM {tbl}", conn)
        conn.close()

        id_col = 'gauge_id' if 'gauge_id' in df_full.columns else df_full.columns[0]
        df_full.set_index(id_col, inplace=True)

        numeric = df_full.select_dtypes(include=[np.number]).fillna(0.0)
        means   = numeric.mean()
        stds    = numeric.std().replace(0.0, 1.0)
        normed  = (numeric - means) / stds

        df_sub = normed.loc[self.basins].iloc[:, :27]

        self.attr_means  = means
        self.attr_stds   = stds
        self.attr_names  = df_sub.columns
        self.attr_df     = df_sub

    # ------------------------------------------------------------------
    def __len__(self):
        return self.num_samples

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        # ---------- fast path if cached ----------
        if self.cache:
            x       = self.x[idx]
            y       = self.y[idx]
            basin   = self.basin_map[idx]
            q_mean  = self.q_means[idx]
            q_std   = self.q_stds[idx]
            date    = self.dates[idx] if self.include_dates else ""
        else:
            file_idx = self.valid_idx[idx]
            f        = self._get_h5()
            x        = f["input_data"][file_idx]
            y        = f["target_data"][file_idx]
            basin    = f["sample_2_basin"][file_idx].decode("ascii")
            q_mean   = f["q_means"][file_idx]
            q_std    = f["q_stds"][file_idx]
            date     = (f["dates"][file_idx].decode("ascii")
                        if (self.include_dates and "dates" in f) else "")

        # ---------- optional re-normalisation ----------
        if not self.is_normalized:
            x = (x - SCALAR["input_means"]) / SCALAR["input_stds"]
            y = (y - SCALAR["output_mean"])  / SCALAR["output_std"]

        # ---------- static attributes ----------
        attrs = None
        if not self.no_static:
            attrs_arr = self.attr_df.loc[basin].values.astype(np.float32)
            attrs     = torch.from_numpy(attrs_arr)

        # ---------- to tensors ----------
        x_t = torch.from_numpy(x.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32))
        q_s = torch.tensor(q_std,  dtype=torch.float32)
        q_m = torch.tensor(q_mean, dtype=torch.float32)

        if self.no_static:
            return x_t, y_t, q_m, q_s, basin, date
        else:
            return x_t, attrs, y_t, q_m, q_s, basin, date
