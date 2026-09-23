"""
papercode/utils.py
"""
import sys
from pathlib import Path, PosixPath
from typing import List

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import pdb

from .datasets import CamelsTXT


def create_h5_files(
    camels_root: PosixPath,
    out_file: PosixPath,
    basins: List[str],
    dates: List,
    db_path: str,
    model_name: str = 'lstm',
    is_train: bool = True,
    with_basin_str: bool = True,
    seq_length: int = 365,
    forecast_horizon: int = 8, # 7+1(nowcast)
    include_dates: bool = False
):
    if out_file.is_file():
        raise FileExistsError(f"File already exists at {out_file}")

    total_len = seq_length + forecast_horizon
    total_samples_written = 0
    total_basins_written = 0

    with h5py.File(out_file, 'w') as out_f:
        input_data = out_f.create_dataset(
              'input_data', shape=(0, total_len, 15), maxshape=(None, total_len, 15), # todo, based on if multi source
              dtype=np.float32, compression='gzip', chunks=True
          )
        target_data = out_f.create_dataset(
            'target_data', shape=(0, forecast_horizon), maxshape=(None, forecast_horizon),
            dtype=np.float32, compression='gzip', chunks=True
        )
        q_means = out_f.create_dataset(
            'q_means', shape=(0, 1), maxshape=(None, 1),
            dtype=np.float32, compression='gzip', chunks=True
        )       
        q_stds = out_f.create_dataset(
            'q_stds', shape=(0, 1), maxshape=(None, 1),
            dtype=np.float32, compression='gzip', chunks=True
        )
        if with_basin_str:
            sample_2_basin = out_f.create_dataset(
                'sample_2_basin', shape=(0,), maxshape=(None,),
                dtype='S10', compression='gzip', chunks=True
            )

        if include_dates:
            dates_ds = out_f.create_dataset(
                'dates', shape=(0,), maxshape=(None,),
                dtype='S10', compression='gzip', chunks=True
            )

        for basin in tqdm(basins, file=sys.stdout):
            ds = CamelsTXT(
                camels_root=camels_root, basin=basin, dates=dates,
                is_train=is_train, seq_length=seq_length, forecast_horizon=forecast_horizon,
                model_name=model_name, with_attributes=False,
                concat_static=False, db_path=db_path, normalize_perbasin=False
            ) # only return x and y; no static attributes from camelstxt

            x_np = ds.x.numpy()
            y_np = ds.y.numpy()
            N = x_np.shape[0]
            num_samples = N - forecast_horizon - 1 

            x_combined = []
            y_combined = []
            q_combined = []
            date_strings = []

            mean_q = ds.output_mean
            std_q = ds.output_std
            if not np.isfinite(std_q) or std_q == 0:
                print(f"WARNING: skipping basin {basin}: no usable streamflow in this split (std={std_q})")
                continue

            for i in range(num_samples):
                past = x_np[i]
                future = x_np[i:i+forecast_horizon, -1, :] # i+1:i+1+forecast_horizon, wrong!
                target = y_np[i]


                x_window = np.vstack([past, future])
                x_combined.append(x_window)
                y_combined.append(target)
                q_combined.append([mean_q, std_q])
                if include_dates:
                    dt = ds.period_start + pd.Timedelta(days=i)
                    date_strings.append(dt.strftime('%Y-%m-%d'))

            if not x_combined:
                continue

            total_basins_written += 1
            x_arr = np.stack(x_combined)
            y_arr = np.stack(y_combined)
            q_arr = np.array(q_combined)
            q_mean_arr = q_arr[:, 0:1]
            q_std_arr = q_arr[:, 1:2]
            n_valid = x_arr.shape[0]

            old = input_data.shape[0]
            new = old + n_valid

            input_data.resize((new, total_len, 15)) # todo, based on if multi source
            target_data.resize((new, forecast_horizon))
            
            q_means.resize((new, 1))
            q_stds.resize((new, 1))

            if with_basin_str:
                sample_2_basin.resize((new,))
            if include_dates:
                dates_ds.resize((new,))
            input_data[old:new] = x_arr
            target_data[old:new] = y_arr
            q_means[old:new] = q_mean_arr
            q_stds[old:new] = q_std_arr
            
            if with_basin_str:
                sample_2_basin[old:new] = np.array([basin.encode('ascii')] * n_valid)
            if include_dates:
                dates_ds[old:new] = np.array([d.encode('ascii') for d in date_strings])

            out_f.flush()
            total_samples_written += n_valid
            print(f"Saved {n_valid} samples for basin {basin}")

    print(f"Done. {total_samples_written} total samples written from {total_basins_written} basins.")



def get_basin_list(camels_root: Path = None) -> List[str]:
    """Read list of basins from text file.
    
    Returns
    -------
    List
        List containing the 8-digit basin code of all basins
    """
    if camels_root:
        basin_file = camels_root.parent / "basin_list.txt"
    else:
        basin_file = Path(__file__).absolute().parent.parent / "data/basin_list.txt"
    with basin_file.open('r') as fp:
        basins = fp.readlines()
    basins = [basin.strip() for basin in basins]
    return basins
