# HydroDiffusion: Diffusion-Based Probabilistic Streamflow Forecasting with a State Space Backbone

---

**HydroDiffusion** is a **score-based diffusion model** built upon a **State Space Model (SSM)** backbone ([S4D-FT](https://doi.org/10.1029/2025WR039888)) for **probabilistic streamflow forecasting**.  

This repository contains:
- Training and evaluation pipelines for HydroDiffusion and baseline models (DiffusionLSTM, LSTM, SSM).  
- Scripts for deterministic and probabilistic performance evaluation.  
- The **HydroDiffusion checkpoint used in the paper**, provided for reproducibility in  
  ```
  runs/run_2507_2120_seed3407/model_epoch60.pt
  ```
---

## 0. Dataset Preparation

We use the [CAMELS dataset](https://ral.ucar.edu/solutions/products/camels) (Catchment Attributes and Meteorology for Large-sample Studies) to construct training, validation, and testing datasets across 531 U.S. watersheds.  
Each `.h5` file contains daily meteorological forcings (precipitation, temperature, vapor pressure, shortwave radiation) and static catchment attributes.

### (i) Download the official CAMELS dataset from the NCAR/UCAR repository:

🔗 [https://ral.ucar.edu/solutions/products/camels](https://ral.ucar.edu/solutions/products/camels)

After extraction, specify the dataset root directory in main.py.

### (ii) HDF5 datasets generated from Daymet forcings

For training and testing with **Daymet forcings**, if the folder `runs/shared_h5_new` does **not** exist, the code will automatically generate new `.h5` files from the raw CAMELS dataset specified by `--camels_root`.  

The generated files will be automatically stored under your current run directory (e.g., `runs/run_<MMDD>_<HHMM>_seed3407/`) each time a new training job is launched.

To avoid regenerating `.h5` files for every run, you can download the preprocessed Daymet dataset directly from Hugging Face:

➡️ [Download `shared_h5_new` from Hugging Face](https://huggingface.co/yhwang08/HydroDiffusion/upload/main)

and place the folder inside your project directory as: 

```
runs/shared_h5_new/
```
Note: Do not use shared_h5_new if you plan to train with forcings other than **Daymet**.

---

## 1. Training: `train.sh`

### Usage

```
./train.sh <model> <static|no_static> [gpu] [note]
```

**Arguments**

| Argument | Description |
|-----------|-------------|
| model | Model architecture name. Options include:<br>• seq2seq_lstm – (deterministic) sequence-to-sequence LSTM baseline<br>• encdec_lstm – (deterministic) encoder–decoder LSTM baseline<br>• seq2seq_ssm – (deterministic) state-space model (S4D-FT)<br>• decoder_only_ssm – HydroDiffusion model<br>• decoder_only_ssm_hybrid – a faster implementation of HydroDiffusion (hybrid FFT+recurrent inference; ~17x faster than the original implementation, numerically equivalent to decoder_only_ssm)<br>• diffusion_lstm – DiffusionLSTM_encdec <br>• decoder_only_lstm – DiffusionLSTM_dec |
| flag | Whether to include static catchment features (`static`) or not (`no_static`). Default: `static`. |
| gpu | CUDA device ID (default: `0`). |
| note | Custom tag appended to log filenames (no spaces). |


### Example

```
./train.sh decoder_only_ssm static 0 myExperiment
```

This command will:

1. Set `CUDA_VISIBLE_DEVICES=0`.  
2. Launch training using 5 dyanmic meteorological forcings from daymet and 27 static catchment attributes.  
3. Save training logs to:

   ```
   reports/global_decoder_only_ssm_static_<seed>_myExperiment.out
   ```

The run directory (printed in the log) contains checkpoints and outputs, for example:

```
Run directory is: runs/run_<MMDD>_<HHMM>_seed3407/
```

---

## 2. Testing: `test.sh`

### Usage

```
./test.sh <model> <static|no_static> [gpu] [run_dir] [note]
```

Arguments are the same as for `train.sh`, with an additional optional argument `run_dir` specifying the directory of the trained model to evaluate.

### Example

```
./test.sh decoder_only_ssm static 0 runs/run_<MMDD>_<HHMM>_seed3407 myExperiment
```

### Model Outputs

Each model evaluation produces `.npz` files that store either **ensemble forecasts** (for diffusion models) or **deterministic predictions** (for single-output models).  
The file naming convention depends on the model type:

| Model Type | Output File | Description |
|-------------|--------------|--------------|
| DiffusionLSTM (DiffusionLSTM<sub>EncDec</sub> (`diffusion_lstm`) and DiffusionLSTM<sub>Dec</sub> (`decoder_only_lstm`)) | `ensembles_epochbest.npz` | Full ensemble forecasts `(N, S, H)` generated from the best-performing epoch. |
| HydroDiffusion (`decoder_only_ssm`) | `ensembles_epoch60.npz` | Full ensemble forecasts `(N, S, H)` generated from epoch 60. |
| LSTM deterministic models (`encdec_lstm`, `seq2seq_lstm`) | `deterministic_epoch30.npz` | Mean predictions `(N, H)` from epoch 30. |
| SSM deterministic models (`seq2seq_ssm`, etc.) | `deterministic_epoch49.npz` | Mean predictions `(N, H)` from epoch 49. |

Each `.npz` file contains the following arrays:

- `basins`: basin identifiers `(N,)`  
- `dates`: corresponding datetime indices `(N,)`  
- `obs`: observed streamflow `(N, H_obs)`  
- `ens` or `preds`: predicted streamflow ensembles or deterministic outputs  

These files are saved automatically to the model’s `run_dir`, for example:

```
runs/run_0306_1938_seed3407/ensembles_epoch60.npz
```

**Notes**

- Executes `main.py evaluate_generic` using the trained checkpoint.  
- Writes outputs and metrics to `reports/` (e.g., `reports/test_decoder_only_ssm_static_3407_myExperiment.out`).  
- To locate the `run_dir`, open the training log and find the line:

  ```
  Run directory is: [run_dir]
  ```

---

## 3. Evaluation: `analysis/main_performance_full_evaluation.py`

After generating model outputs (`.npz` files) from `test.sh`, evaluate model performance using:

```
python analysis/main_performance_full_evaluation.py <experiment_name> <npz_path>
```

### Example

```
python analysis/main_performance_full_evaluation.py decoder_only_ssm runs/run_0306_1938_seed3407/ensembles_epoch60.npz
```

### Description

This script performs **comprehensive deterministic and probabilistic evaluation** of model predictions saved as `.npz` files.

- **Deterministic models**: Input `.npz` file (e.g., `deterministic_epoch30.npz`, `deterministic_epoch49.npz`)
- **Probabilistic (ensemble) models**: Input `.npz` file (e.g., `ensembles_epochbest.npz`, `ensembles_epoch60.npz`)

### Outputs

All evaluation results are saved automatically in:

```
analysis/ensemble_stats/
```

The script generates one CSV file per forecast lead time:

- `<experiment_name>_det_lead<k>.csv` — deterministic metrics  
- `<experiment_name>_prob_lead<k>.csv` — probabilistic metrics  

### Example Output Directory

```
analysis/ensemble_stats/
├── decoder_only_ssm_det_lead1.csv
├── decoder_only_ssm_det_lead2.csv
├── decoder_only_ssm_prob_lead1.csv
└── decoder_only_ssm_prob_lead2.csv
```

Each CSV file summarizes per-basin results, and includes mean and median statistics across all basins for each lead time.

## 4. HydroDiffusion Checkpoint

The **HydroDiffusion** checkpoint used in the paper is included in this repository for reproducibility.

You can find it at:

```
runs/run_2507_2120_seed3407/model_epoch60.pt
```
