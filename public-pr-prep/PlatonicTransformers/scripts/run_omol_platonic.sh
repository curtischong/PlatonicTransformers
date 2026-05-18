#!/bin/bash
# Reproduce the winning Platonic Transformer recipe on OMol25 (Snellius H100).
# See configs/omol.yaml for the full recipe; see README_omol.md for details.
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out
#SBATCH --job-name=omol_platonic

set -e
mkdir -p logs

# Edit these for your cluster:
DATA_PATH="${DATA_PATH:-$HOME/data/omol25}"

source ./venv/bin/activate
module load 2024 2>/dev/null || true
module load CUDA/12.6.0 2>/dev/null || true

export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=120

python mains/main_omol.py \
    --config configs/omol.yaml \
    --dataset.data_dir="${DATA_PATH}" \
    --logging.enabled=true
