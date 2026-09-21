#!/usr/bin/env bash

#SBATCH --job-name=er50_5k_unary
#SBATCH --partition=lipn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --array=0-5
#SBATCH --output=/data1/schulz/slurm-%x-%A_%a.log
#SBATCH --error=/data1/schulz/slurm-%x-%A_%a.log

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate difusco-pairs-gpu

cd "$HOME/MRF-DIFUSCO/difusco"

DIFFUSION_VALUES=(2 4 8 16 32 64)
DIFFUSION_STEPS="${DIFFUSION_VALUES[$SLURM_ARRAY_TASK_ID]}"

DATA_ROOT="/data1/schulz/MRF-DIFUSCO-data/mis_er50_5k"
SWEEP_ROOT="/data1/schulz/MRF-DIFUSCO-runs/er50_5k_diffusion_sweep"
RUN_DIR="${SWEEP_ROOT}/unary/diff_${DIFFUSION_STEPS}"
LOG_DIR="${RUN_DIR}/logs"

mkdir -p "$RUN_DIR" "$LOG_DIR"

SLURM_LOG="/data1/schulz/slurm-${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

ln -sf "$SLURM_LOG" "$LOG_DIR/slurm.log"

export WANDB_MODE=offline

{
    echo "mode=unary"
    echo "diffusion_steps=${DIFFUSION_STEPS}"
    echo "job_id=${SLURM_JOB_ID}"
    echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
    echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
    echo "hostname=$(hostname)"
    echo "start_time=$(date --iso-8601=seconds)"
    echo "git_commit=$(git -C "$HOME/MRF-DIFUSCO" rev-parse HEAD)"
    echo "dataset=${DATA_ROOT}"
    echo "environment=${CONDA_DEFAULT_ENV}"
    python --version
    python - <<'PY'
import torch
print("torch =", torch.__version__)
print("torch_cuda =", torch.version.cuda)
print("cuda_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu =", torch.cuda.get_device_name(0))
PY
} > "$RUN_DIR/metadata.txt" 2>&1

python train.py \
  --task mis \
  --storage_path "$RUN_DIR" \
  --training_split "${DATA_ROOT}/train/*.gpickle" \
  --training_split_label_dir "${DATA_ROOT}/train_labels" \
  --validation_split "${DATA_ROOT}/val/*.gpickle" \
  --validation_split_label_dir "${DATA_ROOT}/val_labels" \
  --test_split "${DATA_ROOT}/test/*.gpickle" \
  --test_split_label_dir "${DATA_ROOT}/test_labels" \
  --validation_examples 50 \
  --prediction_type unary \
  --diffusion_type categorical \
  --diffusion_steps "$DIFFUSION_STEPS" \
  --inference_diffusion_steps "$DIFFUSION_STEPS" \
  --sequential_sampling 1 \
  --parallel_sampling 1 \
  --batch_size 64 \
  --num_workers 16 \
  --num_epochs 50 \
  --n_layers 12 \
  --hidden_dim 256 \
  --project_name er50_5k_diffusion_sweep \
  --wandb_logger_name "unary-d${DIFFUSION_STEPS}" \
  --do_train \
  --do_test \
  2>&1 | tee "$LOG_DIR/train_test.log"
