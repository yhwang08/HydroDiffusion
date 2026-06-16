"""
Main script for running training or evaluation.
"""

import argparse
import json
import pickle
import random
import sys
import os
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from papercode.datasets import CamelsH5
from papercode.datautils import add_camels_attributes
from papercode.utils import create_h5_files, get_basin_list

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_descriptor')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set default hyperparameters
GLOBAL_SETTINGS = {
    'batch_size': 256,
    'clip_norm': True,
    'clip_value': 1,
    'dropout': 0.5,
    'epochs': 60,
    'hidden_size': 256,
    'lstm_nlayers': 1, 
    'unet_nfeat': 64, 
    'initial_forget_gate_bias': 3,
    'log_interval': 50,
    'learning_rate': 3e-5,
    'seq_length': 365,
    'forecast_horizon': 8, # nowcast(1) + forecast(7) # 8
    'train_start': pd.to_datetime('01-10-1980', format='%d-%m-%Y'),
    'train_end':   pd.to_datetime('30-09-1990', format='%d-%m-%Y'),
    'val_start':   pd.to_datetime('01-10-1990', format='%d-%m-%Y'),
    'val_end':     pd.to_datetime('30-09-1995', format='%d-%m-%Y'),
    'test_start':  pd.to_datetime('01-10-1995', format='%d-%m-%Y'),
    'test_end':    pd.to_datetime('30-09-2005', format='%d-%m-%Y'),

}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args() -> Dict:
    """Parse CLI arguments into a configuration dictionary."""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate"])
    parser.add_argument('--camels_root', type=str, default='/data/rdl/yihan/data/basin_dataset_public_v1p2/')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--run_dir', type=str)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--cache_data', type=str2bool, default=True)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--no_static', type=str2bool, default=False)
    parser.add_argument('--concat_static', type=str2bool, default=True)
    parser.add_argument('--static_dim', type=int, default=27)
    parser.add_argument('--model_name', type=str, default='diffusion_lstm', help="Choose from ['seq2seq_lstm', 'seq2seq_ssm', 'diffusion_lstm', 'diffusion_unet', 'diffusion_ssm', 'diffusion_ssm_lstm']")
    parser.add_argument('--use_mse', type=str2bool, default=False)
    parser.add_argument('--n_splits', type=int, default=None)
    parser.add_argument('--basin_file', type=str, default=None)
    parser.add_argument('--split', type=int, default=None)
    parser.add_argument('--split_file', type=str, default=None)
    parser.add_argument('--eval_dataset', type=str, choices=['val','test'], default='val', help="Which split to score (only for mode=evaluate)")
    parser.add_argument('--epoch_num', type=int, default=60, help="Epoch number for evaluation")
    
    parser.add_argument('--h5_dir', type=str, default= '/home/yihan/diffusion_ssm/runs/shared_h5_new/',
    help="If set, skip create_h5_files and read HDF5s from this folder")
    
    parser.add_argument('--forcing_source', type=str, default="daymet")
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to draw during diffusion sampling')
    parser.add_argument('--train_mode', type=str, default='ddim',
                        choices=['ddim', 'ddpm'],
                        help='Training schedule: "ddim" for continuous cosine-logSNR (train_generic.py), '
                             '"ddpm" for discrete linear-beta (train_ddpm.py)')
    parser.add_argument('--predict_mode', type=str, default="velocity",
                        choices=["noise", "velocity"],
                        help='Prediction mode for the diffusion model: "noise" for standard noise prediction or "velocity" for velocity prediction.')
    parser.add_argument('--time_emb_dim', type=int, default=256)
    parser.add_argument('--use_amp', type=str2bool, default=True)
    # === DDIM-related arguments ===
    parser.add_argument('--ddim_steps', type=int, default=10,
                        help='Number of reverse steps to use for DDIM sampling')
                        
    # === DDPM training related arguments === 
    parser.add_argument('--ddpm_train_steps', type=int, default=None,
                    help='Number of discrete diffusion training steps for DDPM-style training')
    parser.add_argument('--ddpm_beta_start', type=float, default=1e-4,
                        help='Starting beta for DDPM linear schedule')
    parser.add_argument('--ddpm_beta_end', type=float, default=2e-2,
                        help='Ending beta for DDPM linear schedule')
    parser.add_argument('--ddpm_schedule', type=str, default='linear',
                        choices=['linear', 'cosine'],
                        help='Discrete DDPM noise schedule')

                        
    #====================================#
    # Argument for SSM    
    # Optimizer
    parser.add_argument('--lr', default=3e-5, type=float, help='Learning rate')
    parser.add_argument('--lr_min', default=0.001, type=float, help='SSM Learning rate')
    parser.add_argument('--lr_dt', default=0.0, type=float, help='dt lr')
    parser.add_argument('--min_dt', default=0.001, type=float, help='min dt')
    parser.add_argument('--max_dt', default=1, type=float, help='max dt')
    parser.add_argument('--wd', default=0.02, type=float, help='H weight decay')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay of the optimizer')
    
    # Scheduler
    parser.add_argument('--epochs', default=60, type=int, help='Training epochs')
    parser.add_argument('--warmup', default=1, type=int, help='warmup epochs')
    
    # Dataloader
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    
    # Model
    parser.add_argument('--n_layers', default=6, type=int, help='Number of layers')
    parser.add_argument('--d_model', default=256, type=int, help='Model dimension')
    parser.add_argument('--ssm_dropout', default=0.3, type=float, help='Dropout') # 0.15
    parser.add_argument('--prenorm', action='store_true', help='Prenorm')

    parser.add_argument('--d_state', default=256, type=int)
    parser.add_argument('--cfr', default=10.0, type=float)
    parser.add_argument('--cfi', default=10.0, type=float)
    
    # General
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--model', type=str, default='s4d', metavar='N', help='model name')
    parser.add_argument('--pool_type', type=str, default='power') # 'avg', 'power', or 'attn'
    #====================================#
    
    # -- Profiling ----------------------------------------------------------
    parser.add_argument('--profile_runtime',     type=str2bool, default=False)
    parser.add_argument('--profile_max_batches', type=int,      default=10)
    parser.add_argument('--profile_num_samples', type=int,      default=1)
    parser.add_argument('--profile_csv_name',    type=str,      default=None)

    cfg = vars(parser.parse_args())

    if cfg["mode"] in ["evaluate"] and cfg["run_dir"] is None:
        raise ValueError("In evaluation mode, --run_dir must be specified.")

    # Set device
    device = f"cuda:{cfg['gpu']}" if cfg["gpu"] >= 0 else "cpu"
    global DEVICE
    DEVICE = torch.device(device if torch.cuda.is_available() else "cpu")
    cfg["DEVICE"] = DEVICE

    # Merge global defaults
    cfg.update(GLOBAL_SETTINGS)

    # Convert paths
    if cfg["camels_root"] is not None:
        cfg["camels_root"] = Path(cfg["camels_root"])
    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])

    return cfg


def main():
    cfg = get_args()
    if cfg["mode"] == "train":
        if cfg["train_mode"] == "ddpm":
            from papercode.train_ddpm import train
        else:
            from papercode.train_generic import train
        train(cfg)

    elif cfg["mode"] == "evaluate":
        from papercode.evaluate_generic import evaluate
        evaluate(cfg)

    elif cfg["mode"] == "train_debug":
        from papercode.train_debug import train
        train(cfg)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
