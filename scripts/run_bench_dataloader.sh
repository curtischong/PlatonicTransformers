#!/bin/bash
#SBATCH --account=gusei11738
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch-shared/ebekkers/platonic-omol/training/slurm-%x-%j.out

# Dataloader-only benchmark: iterate the DataLoader without any model, to
# measure pure data-delivery throughput. Upper bound on training tok/s.
#
# Required env vars (sbatch --export):
#   MODE           dynamic | fixed
# Optional:
#   MAX_ATOMS      default 6000
#   NUM_WORKERS    default 8
#   NUM_BATCHES    default 200 (measurement batches, plus warmup)
#   WARMUP         default 20

set -euo pipefail

: "${MODE:?MODE required}"
: "${MAX_ATOMS:=6000}"
: "${NUM_WORKERS:=8}"
: "${NUM_BATCHES:=200}"
: "${WARMUP:=20}"
: "${BACKEND:=lmdb}"
: "${SPLIT:=train}"

source /scratch-shared/ebekkers/scaling-laws-venv/bin/activate
module load 2024
module load CUDA/12.6.0

export DATA_PATH=/scratch-shared/ebekkers/omol25
export PYTHONPATH=/scratch-shared/ebekkers/platonic-omol/training
export HYDRA_FULL_ERROR=1
export TMPDIR=/scratch-shared/ebekkers/tmp

cd /scratch-shared/ebekkers/platonic-omol/training

echo "=== DataLoader bench: mode=${MODE} max_atoms=${MAX_ATOMS} workers=${NUM_WORKERS} batches=${NUM_BATCHES} warmup=${WARMUP} ==="
echo "Node: $(hostname)   Date: $(date -Iseconds)"
echo "======================="

python ../scripts/bench_dataloader.py "${MODE}" "${MAX_ATOMS}" "${NUM_WORKERS}" "${NUM_BATCHES}" "${WARMUP}" "${BACKEND}" "${SPLIT}"

echo "[$(date -Iseconds)] Done."
