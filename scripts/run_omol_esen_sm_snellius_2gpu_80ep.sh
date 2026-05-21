#!/bin/bash
# eSEN-sm (paper recipe) — 80-epoch production run, 2× H100.
#
# Architecture matches the released `facebook/OMol25/checkpoints/
# esen_sm_direct_all.pt` checkpoint (6.04M params from this config; the
# released 6.33M checkpoint differs only on minor non-architectural knobs).
# Direct-force prediction so torch.compile works.
#
# Effective batch: 6000 atoms × 2 ranks = 12000 atoms / optimizer step.
# (zaq6tuhv production reference used max_atoms=20000 on 1-GPU; this run
# uses 12000 to stay consistent with the Platonic 2-GPU sibling.)
#
# Projected wall-clock (from short-run throughput measurement at this config):
#   ~3.70 steps/sec × 80 epochs × 16612 batches/epoch = ~100 h ≈ 4.2 days
# 5-day SLURM wall gives ~0.8d headroom.
#
# Note: main_omol.py automatically sets `torch._dynamo.config.optimize_ddp
# = False` for (model.name=esen AND gpus>1), which works around a
# torch.compile DDPOptimizer crash on the eSCNMDBackbone
# (`AttributeError: 'tuple' object has no attribute 'meta'`). Compile
# still runs (as one graph); DDP's standard backward hooks handle
# gradient sync.
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --time=5-00:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out
#SBATCH --job-name=PR-omol-esen-sm-2gpu-80ep

set -e
mkdir -p logs

DATA_PATH="${DATA_PATH:-/scratch-shared/ebekkers/omol25/open_mol}"
VENV_PATH="${VENV_PATH:-/scratch-shared/ebekkers/scaling-laws-venv-v2}"

source "${VENV_PATH}/bin/activate"
module load 2024 2>/dev/null || true
module load CUDA/12.6.0 2>/dev/null || true

export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=120

echo "=== eSEN-sm 2× H100 80-epoch production run ==="
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "JobID:       ${SLURM_JOB_ID}"
echo "DATA_PATH:   ${DATA_PATH}"
echo "VENV_PATH:   ${VENV_PATH}"
nvidia-smi -L || true

srun python mains/main_omol.py \
    --config configs/omol_esen_sm.yaml \
    --dataset.data_dir="${DATA_PATH}" \
    --system.gpus=2 \
    --system.accumulate_grad_batches=1 \
    --training.epochs=80 \
    --training.max_atoms_per_batch=6000 \
    --training.max_atoms_per_batch_val=6000 \
    --training.max_edges_per_batch=1200000 \
    --training.max_edges_per_batch_val=1200000 \
    --logging.enabled=true

exit_code=$?
echo "=== JOB FINISHED (exit ${exit_code}) at $(date) ==="
exit ${exit_code}
