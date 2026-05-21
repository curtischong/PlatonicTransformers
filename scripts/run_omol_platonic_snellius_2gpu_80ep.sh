#!/bin/bash
# Platonic Transformer (qcczbpfn recipe) — 80-epoch production run, 2× H100.
#
# Effective batch: 6000 atoms × 2 ranks = 12000 atoms / optimizer step,
# matching the qcczbpfn recipe (4× RTX 6000 Ada × 3000 = 12000) exactly.
# avg_num_nodes=26.5 keeps qcczbpfn parity (paper-default 26.5 for the neutral
# subset; Mohammad changed the yaml default to 54.9 — overridden here for
# parity).
#
# Projected wall-clock (from short-run throughput measurement at this config):
#   ~4.41 steps/sec × 80 epochs × 16612 batches/epoch = ~84 h ≈ 3.5 days
# 5-day SLURM wall gives ~1.5d headroom.
#
# Cosine LR with 1% warmup over 80 × 16612 = 1 328 960 total steps.
# All session fixes are active: DDP sampler sharding, set_epoch
# propagation, fairchem-style global-mean loss, cudnn deterministic,
# dynamo knobs, num_workers=16/rank. The 2-GPU code path is the same
# as 4-GPU; effective batch and LR schedule are unchanged.
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --time=5-00:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out
#SBATCH --job-name=PR-omol-platonic-2gpu-80ep

set -e
mkdir -p logs

DATA_PATH="${DATA_PATH:-/scratch-shared/ebekkers/omol25/open_mol}"
VENV_PATH="${VENV_PATH:-/scratch-shared/ebekkers/scaling-laws-venv-v2}"

source "${VENV_PATH}/bin/activate"
module load 2024 2>/dev/null || true
module load CUDA/12.6.0 2>/dev/null || true

export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=120

echo "=== Platonic Transformer 2× H100 80-epoch production run ==="
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "JobID:       ${SLURM_JOB_ID}"
echo "DATA_PATH:   ${DATA_PATH}"
echo "VENV_PATH:   ${VENV_PATH}"
nvidia-smi -L || true

srun python mains/main_omol.py \
    --config configs/omol.yaml \
    --dataset.data_dir="${DATA_PATH}" \
    --system.gpus=2 \
    --system.accumulate_grad_batches=1 \
    --training.epochs=80 \
    --training.max_atoms_per_batch=6000 \
    --training.max_atoms_per_batch_val=6000 \
    --training.max_edges_per_batch=1200000 \
    --training.max_edges_per_batch_val=1200000 \
    --model.avg_num_nodes=26.5 \
    --logging.enabled=true

exit_code=$?
echo "=== JOB FINISHED (exit ${exit_code}) at $(date) ==="
exit ${exit_code}
