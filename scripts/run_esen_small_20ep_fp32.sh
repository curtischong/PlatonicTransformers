#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out

# eSEN "small" — fairchem's eSCN-MD backbone with autograd-derived (conservative)
# forces. Config: sphere/hidden=32, lmax=4, mmax=2, 12 layers, ~3.4M params.
#
# Recipe mirrors the PT 20-epoch runs for apples-to-apples comparison:
#   - AdamW + cosine_annealing_ws, 1% fractional warmup, lr=5e-4, wd=1e-5
#   - dynamic batching, max_atoms_per_batch=12000
#   - heldout val, val_check_interval=5000
#   - fp32 baseline precision
#
# trainer.inference_mode=false is required because eSEN derives forces via
# torch.autograd.grad and val/test cannot run under inference-mode.
#
# Usage:
#   sbatch scripts/run_esen_small_20ep_fp32.sh

set -e
mkdir -p logs

source /scratch-shared/ebekkers/scaling-laws-venv-v2/bin/activate
module load 2024
module load CUDA/12.6.0

export DATA_PATH=/scratch-shared/ebekkers/omol25
export PYTHONPATH=/scratch-shared/ebekkers/platonic-omol/training
export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=120
export TMPDIR=/scratch-shared/ebekkers/tmp
export TORCH_HOME=/scratch-shared/ebekkers/torch_cache
export TRITON_CACHE_DIR=/scratch-shared/ebekkers/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/scratch-shared/ebekkers/torch_cache/inductor

cd /scratch-shared/ebekkers/platonic-omol/training

EXP_NAME="esen-small-lmax4-l12-h32-20ep-fp32"

echo "=== Long run (eSEN small, fp32_baseline preset, 20ep): ${EXP_NAME} ==="
echo "Date:  $(date)"
echo "Job:   ${SLURM_JOB_ID:-?} on $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Branch + commit:"
git -C /scratch-shared/ebekkers/platonic-omol log --oneline -3

OVERRIDES=(
    +precision=fp32_baseline
    force_field_module=esen
    data=omol_4m
    data.datamodule.data.num_workers=16
    +data.datamodule.prefetch_factor=4
    data.datamodule.batch_size.train=64
    data.datamodule.batch_size.val=64
    data.datamodule.dynamic_batching=true
    data.datamodule.max_atoms_per_batch=2500
    data.datamodule.max_atoms_per_batch_val=2500
    +data.datamodule.max_edges_per_batch=300000
    +data.datamodule.max_edges_per_batch_val=300000
    data.datamodule.validation_mode=heldout
    data.datamodule.data.val_data_path=/scratch-shared/ebekkers/omol25/open_mol/val
    trainer.max_epochs=20
    trainer.gradient_clip_val=1
    trainer.gradient_clip_algorithm=norm
    trainer.inference_mode=false
    trainer.val_check_interval=5000
    +trainer.limit_val_batches=500
    exp_name=${EXP_NAME}
    model_name=esen
    wandb.use_wandb=True
    wandb.wandb_project=scaling-laws-symmetry
    wandb.group=esen-baseline-20ep
    seed=1
)

echo "=== Hydra parse-check ==="
python train_omol.py "${OVERRIDES[@]}" --cfg job > /dev/null
echo "Hydra config OK"

python train_omol.py "${OVERRIDES[@]}"
exit_code=$?

echo
echo "=== Done: $(date) (exit=$exit_code) ==="
exit $exit_code
