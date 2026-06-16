#!/usr/bin/env bash
#
# Usage: ./test.sh <model> <static|no_static> [gpu] [run_dir] [note]
#   model : seq2seq_lstm | encdec_lstm | seq2seq_ssm | diffusion_lstm | diffusion_unet | diffusion_ssm
#   flag  : static | no_static
#   gpu   : CUDA device ID (default 0)
#   run_dir: the running directory
#   note  : optional tag (no spaces) added to logfile name

model=$1
static_flag=$2
gpu_id=${3:-0}
run_dir=$4
note=${5:-}

export CUDA_VISIBLE_DEVICES="$gpu_id"
export PYTHONPATH=$(pwd):$PYTHONPATH

nseeds=1
firstseed=3407 #5534 3407

# -------------------------------- static flags -------------------------------
if [[ "$static_flag" == "static" ]]; then
  no_static=false
  concat_static=true
elif [[ "$static_flag" == "no_static" ]]; then
  no_static=true
  concat_static=false
else
  echo "ERROR: static_flag must be 'static' or 'no_static'"
  exit 1
fi

mkdir -p reports

# ============================= main loop =====================================
for (( seed=firstseed; seed<firstseed+nseeds; seed++ )); do
  echo "=== seed=$seed | model=$model | no_static=$no_static | concat_static=$concat_static | GPU=$gpu_id | note='$note' ==="

  logfile="reports/test_${model}_${static_flag}_${seed}${note:+_${note}}.out"

  if [[ "$model" == "decoder_only_ssm" ]]; then
    # ---------------------------- SSM branch ---------------------------------
    # Optional DDPM args:
    #--forcing_source='daymet'\
    #--ddpm_train_steps=1000 \
    #--ddpm_beta_start=1e-4 \
    #--ddpm_beta_end=2e-2 \
    #--ddpm_schedule='linear' \
    python3 main.py evaluate \
      --model_name="$model" \
      --seed="$seed" \
      --gpu=0 \
      --no_static="$no_static" \
      --concat_static="$concat_static" \
      --run_dir="$run_dir" \
      --epochs=60 \
      --d_model=256 \
      --d_state=256 \
      --lr=3e-5 \
      --lr_min=3e-6 \
      --weight_decay=0.00 \
      --wd=4e-5 \
      --lr_dt=0.001 \
      --min_dt=0.01 \
      --max_dt=0.1 \
      --warmup=1 \
      --n_layer=6 \
      --ssm_dropout=0.2 \
      --cfi=10 \
      --cfr=10 \
      --pool_type='power'\
      --predict_mode='velocity'\
    > "$logfile" 2>&1 &
  elif [[ "$model" == "seq2seq_ssm" ]]; then
    python3 main.py evaluate \
      --model_name="$model" \
      --seed="$seed" \
      --gpu=0 \
      --no_static="$no_static" \
      --concat_static="$concat_static" \
      --run_dir="$run_dir" \
      --epochs=60 \
      --d_model=128 \
      --d_state=128 \
      --lr=4e-4 \
      --lr_min=4e-5 \
      --weight_decay=3e-2 \
      --wd=2e-2 \
      --lr_dt=0.001 \
      --min_dt=0.01 \
      --max_dt=0.1 \
      --warmup=0 \
      --n_layer=6 \
      --ssm_dropout=0.12 \
      --cfi=10 \
      --cfr=10 \
      --forcing_source='daymet'\
    > "$logfile" 2>&1 &
  elif [[ "$model" == "seq2seq_lstm" || "$model" == "encdec_lstm" ]]; then
    python3 main.py evaluate \
      --model_name="$model" \
      --seed="$seed" \
      --gpu=0 \
      --no_static="$no_static" \
      --concat_static="$concat_static" \
      --run_dir="$run_dir" \
      --epochs=30 \
      --forcing_source='daymet'\
    > "$logfile" 2>&1 &   
  else
    # ------------------------- other models branch ---------------------------
    python3 main.py evaluate \
      --model_name="$model" \
      --seed="$seed" \
      --gpu=0 \
      --no_static="$no_static" \
      --concat_static="$concat_static" \
      --run_dir="$run_dir" \
      --predict_mode='velocity'\
      --forcing_source='daymet'\
    > "$logfile" 2>&1 &
  fi

  echo "logs written to $logfile"
done
