#!/usr/bin/env bash

#SBATCH --job-name=er50_5k_random_pairwise_block_diffusion_lipngpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --array=0-26
#SBATCH --output=/data1/schulz/slurm-%x-%A_%a.log
#SBATCH --error=/data1/schulz/slurm-%x-%A_%a.log
#SBATCH --nodelist=lipn-rtx[1-3]
#SBATCH --chdir=/data1/schulz/MRF-DIFUSCO
set -euo pipefail

PROJECT_ROOT="/data1/schulz/MRF-DIFUSCO"
PROJECT_DIR="${PROJECT_ROOT}/difusco"

cd "$PROJECT_DIR"

ENV_ROOT="/data1/schulz/conda-envs/difusco-pairs-gpu"
PYTHON="${ENV_ROOT}/bin/python"

DIFFUSION_VALUES=(2 4 8 12 16 24 32 48 64)
SEEDS=(2023 2024 2025)

NUM_DIFFUSIONS=${#DIFFUSION_VALUES[@]}
NUM_SEEDS=${#SEEDS[@]}

DIFFUSION_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

DIFFUSION_STEPS="${DIFFUSION_VALUES[$DIFFUSION_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

DATA_ROOT="/data1/schulz/MRF-DIFUSCO-data/mis_er50_5k"
SWEEP_ROOT="/data1/schulz/MRF-DIFUSCO-runs/li0ngpu_er50_5k_diffusion_sweep"
RUN_DIR="${SWEEP_ROOT}/random_pairwise_block_diffusion/diff_${DIFFUSION_STEPS}/seed_${SEED}"
LOG_DIR="${RUN_DIR}/logs"

[[ -x "$PYTHON" ]] || { echo "Python not found: $PYTHON"; exit 1; }
[[ -d "$DATA_ROOT" ]] || { echo "Dataset not found: $DATA_ROOT"; exit 1; }

mkdir -p "$RUN_DIR" "$LOG_DIR"

SLURM_LOG="/data1/schulz/slurm-${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

ln -sf "$SLURM_LOG" "$LOG_DIR/slurm.log"

export PATH="${ENV_ROOT}/bin:${PATH}"
export XDG_CACHE_HOME="/data1/schulz/.cache"
export TORCH_HOME="/data1/schulz/.cache/torch"
export WANDB_MODE=offline

mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME"

{
    echo "seed=${SEED}"
    echo "mode=random_pairwise"
    echo "diffusion_mode=block"
    echo "diffusion_steps=${DIFFUSION_STEPS}"
    echo "job_id=${SLURM_JOB_ID}"
    echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
    echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
    echo "hostname=$(hostname)"
    echo "start_time=$(date --iso-8601=seconds)"
    echo "git_commit=$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
    echo "dataset=${DATA_ROOT}"
    echo "environment=${ENV_ROOT}"
    echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-unknown}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    "$PYTHON" --version
    "$PYTHON" - <<'PY'
import torch
print("torch =", torch.__version__)
print("torch_cuda =", torch.version.cuda)
print("cuda_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu =", torch.cuda.get_device_name(0))
PY
} > "$RUN_DIR/metadata.txt" 2>&1

"$PYTHON" train.py \
  --task mis \
  --storage_path "$RUN_DIR" \
  --training_split "${DATA_ROOT}/train/*.gpickle" \
  --training_split_label_dir "${DATA_ROOT}/train_labels" \
  --validation_split "${DATA_ROOT}/val/*.gpickle" \
  --validation_split_label_dir "${DATA_ROOT}/val_labels" \
  --test_split "${DATA_ROOT}/test/*.gpickle" \
  --test_split_label_dir "${DATA_ROOT}/test_labels" \
  --validation_examples 50 \
  --prediction_type random_pairwise \
  --pairwise_diffusion_mode block \
  --matching_generator greedy \
  --num_pairwise_matchings 32 \
  --matching_seed 0 \
  --matching_vertex_balance 0.05 \
  --matching_cache_dir /data1/schulz/MRF-DIFUSCO-data/matching_cache \
  --diffusion_type categorical \
  --diffusion_steps "$DIFFUSION_STEPS" \
  --inference_diffusion_steps "$DIFFUSION_STEPS" \
  --sequential_sampling 1 \
  --parallel_sampling 1 \
  --batch_size 64 \
  --num_workers 8 \
  --num_epochs 50 \
  --n_layers 12 \
  --hidden_dim 256 \
  --project_name er50_5k_diffusion_sweep \
  --wandb_logger_name "random-pairwise-block-d${DIFFUSION_STEPS}-s${SEED}" \
  --do_train \
  --do_test \
  --seed "$SEED" \
  2>&1 | tee "$LOG_DIR/train_test.log"
