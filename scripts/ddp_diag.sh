#!/bin/bash
#SBATCH --account=gusei11738
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch-shared/ebekkers/logs/ddp-diag-%j.out

set -euo pipefail

echo "=== BATCH SCRIPT ==="
echo "hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"
nvidia-smi --query-gpu=index,name --format=csv,noheader

echo "=== SRUN TASKS ==="
srun bash -c '
  echo "[rank=$SLURM_PROCID node=$(hostname)]"
  echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "  SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"
  echo "  SLURM_LOCALID=${SLURM_LOCALID:-<unset>}"
  nvidia-smi -L 2>&1 | head -5
  source /scratch-shared/ebekkers/scaling-laws-venv/bin/activate
  module load 2024
  module load CUDA/12.6.0
  python -c "import torch; print(f\"  torch.cuda.is_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}\")"
'
