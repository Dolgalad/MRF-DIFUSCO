#!/bin/bash
#SBATCH --job-name=mis_diffusion_graph_gaussian_er_700_800_5000_bipartite
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/users/schulz/DIFUSCO/slurm_logs/%x.out
#SBATCH --error=/users/schulz/DIFUSCO/slurm_logs/%x.err

set -euo pipefail

cd /users/schulz/DIFUSCO
mkdir -p /users/schulz/DIFUSCO/slurm_logs

source /users/schulz/miniconda3/etc/profile.d/conda.sh
conda activate difusco-clean

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Slurm CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
echo "SLURM_STEP_GPUS: ${SLURM_STEP_GPUS:-unset}"

nvidia-smi

python - <<'PY'
import os
import torch
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.cuda.device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("torch current device:", torch.cuda.current_device())
    print("torch device name:", torch.cuda.get_device_name(0))
PY

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

JOBNAME="mis_diffusion_graph_gaussian_er_700_800_5000_bipartite"
DATA_ROOT="$HOME/DIFUSCO/data/mis_er_700_800_5000"
STORAGE_ROOT="$HOME/DIFUSCO/outputs/$JOBNAME"
BIPARTITE_CACHE_DIR="/tmp/mis_er_700_800_5000_cache"

mkdir -p "$STORAGE_ROOT"


export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

srun python -u difusco/train.py \
  --task "mis" \
  --wandb_logger_name "$JOBNAME" \
  --project_name "$JOBNAME" \
  --diffusion_type "gaussian" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "$STORAGE_ROOT" \
  --training_split "$DATA_ROOT/train/*gpickle" \
  --training_split_label_dir "$DATA_ROOT/train_annotations/" \
  --validation_split "$DATA_ROOT/val/*gpickle" \
  --test_split "$DATA_ROOT/test/*gpickle" \
  --batch_size 4 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --input_representation "bipartite" \
  --bipartite_cache_dir "$BIPARTITE_CACHE_DIR" \
  --bipartite_cache_version "v1" \
  --use_activation_checkpoint
