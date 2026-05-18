#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out

# eSEN-sm — paper-matching configuration (sphere/hidden=128, lmax=2, 5 layers,
# ~6.07M params). Hyperparameters were read from facebook/OMol25's released
# checkpoint esen_sm_direct_all.pt (6.33M params); the 4% param shortfall is
# from non-architectural backbone knobs (ff_type, edge_channels, etc.) that
# don't appear in the published backbone config dict.
#
# Recipe mirrors the PT 20-epoch runs and the original eSEN-small-20ep run
# (22630480) for apples-to-apples comparison against PT-2:
#   - AdamW + cosine_annealing_ws, 1% fractional warmup, lr=5e-4, wd=1e-5
#   - dynamic batching, max_atoms_per_batch=2500 (eSEN cannot fit 12000 at lmax=2)
#   - heldout val, val_check_interval=5000
#   - fp32 baseline precision
#   - torch.compile is set true in esen_sm.yaml (compile applies on stage=="fit"
#     via the omol_module hook); first-step compile takes 5-10 min on H100
#
# trainer.inference_mode=false is required because eSEN derives forces via
# torch.autograd.grad and val/test cannot run under inference-mode.
#
# Usage:
#   sbatch scripts/run_esen_sm_20ep.sh

set -e
mkdir -p logs

# Default to the largest batch that fits eSEN-sm-direct + compile on an H100
# (validated by OOM probe 22644769: max_atoms=20000 trained 20 steps with
# peak VRAM well under the 96GB limit). Override per submission with
# --export=ALL,MAX_ATOMS=...
MAX_ATOMS="${MAX_ATOMS:-20000}"
MAX_EDGES="${MAX_EDGES:-$((MAX_ATOMS * 200))}"

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

EXP_NAME="esen-sm-paper-lmax2-l5-h128-20ep-fp32-compile-n${MAX_ATOMS}"

echo "=== eSEN-sm paper config, 20ep fp32+compile: ${EXP_NAME} ==="
echo "Date:  $(date)"
echo "Job:   ${SLURM_JOB_ID:-?} on $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Branch + commit:"
git -C /scratch-shared/ebekkers/platonic-omol log --oneline -3

OVERRIDES=(
    +precision=fp32_baseline
    force_field_module=esen_sm
    data=omol_4m
    data.datamodule.data.num_workers=16
    +data.datamodule.prefetch_factor=4
    data.datamodule.batch_size.train=64
    data.datamodule.batch_size.val=64
    data.datamodule.dynamic_batching=true
    data.datamodule.max_atoms_per_batch=${MAX_ATOMS}
    data.datamodule.max_atoms_per_batch_val=${MAX_ATOMS}
    +data.datamodule.max_edges_per_batch=${MAX_EDGES}
    +data.datamodule.max_edges_per_batch_val=${MAX_EDGES}
    data.datamodule.validation_mode=heldout
    data.datamodule.data.val_data_path=/scratch-shared/ebekkers/omol25/open_mol/val
    trainer.max_epochs=20
    trainer.gradient_clip_val=1
    trainer.gradient_clip_algorithm=norm
    trainer.inference_mode=false
    trainer.val_check_interval=5000
    +trainer.limit_val_batches=500
    # +precision=fp32_baseline sets force_field_module.compile=false; override here
    # so the omol_module compile hook fires on stage=="fit". esen_sm.yaml also sets
    # compile=true but the precision preset's value wins without this explicit override.
    force_field_module.compile=true
    exp_name=${EXP_NAME}
    model_name=esen_sm
    wandb.use_wandb=True
    wandb.wandb_project=scaling-laws-symmetry
    wandb.group=esen-sm-paper-20ep
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
