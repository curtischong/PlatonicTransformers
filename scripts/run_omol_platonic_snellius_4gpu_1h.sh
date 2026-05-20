#!/bin/bash
# Snellius 4× H100 DDP smoke test (1 hour). Mirrors qcczbpfn's effective
# batch (4 × 3000 atoms / 600k edges per rank = 12000 atoms / 2.4M edges per
# optimizer step) but on H100 hardware. Use this for code-vs-code audit
# against qcczbpfn (4× RTX 6000 Ada DDP, same recipe).
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out
#SBATCH --job-name=PR-omol-4gpu-snellius-1h

set -e
mkdir -p logs

DATA_PATH="${DATA_PATH:-$HOME/data/omol25}"
VENV_PATH="${VENV_PATH:-./venv}"

source "${VENV_PATH}/bin/activate"
module load 2024 2>/dev/null || true
module load CUDA/12.6.0 2>/dev/null || true

export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=120

echo "=== snellius 4× H100 DDP run ==="
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "JobID:       ${SLURM_JOB_ID}"
echo "DATA_PATH:   ${DATA_PATH}"
echo "VENV_PATH:   ${VENV_PATH}"
echo "GPUs:        ${SLURM_GPUS_ON_NODE:-?}"
echo "ntasks:      ${SLURM_NTASKS:-?}"
nvidia-smi -L || true

# CLI overrides: restore qcczbpfn's per-rank caps (the yaml ships 12k/2.4M
# for 1-GPU). 4 ranks × 3000 atoms = 12000 atoms per optimizer step.
srun python mains/main_omol.py \
    --config configs/omol.yaml \
    --dataset.data_dir="${DATA_PATH}" \
    --system.gpus=4 \
    --system.accumulate_grad_batches=1 \
    --training.max_atoms_per_batch=3000 \
    --training.max_atoms_per_batch_val=3000 \
    --training.max_edges_per_batch=600000 \
    --training.max_edges_per_batch_val=600000 \
    --logging.enabled=true

exit_code=$?
echo "=== JOB FINISHED (exit ${exit_code}) ==="
exit ${exit_code}
