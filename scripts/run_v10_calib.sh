#!/bin/bash
#SBATCH --account=gusei11738
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch-shared/ebekkers/platonic-omol/training/slurm-%x-%j.out

# v10 throughput calibration — runs 300 steps (200 warmup, 100 measurement) of the
# v10 recipe on train_4M, logs token_processed + _timestamp every step to wandb so
# steady-state tokens/sec can be parsed afterwards. Validation is disabled.
#
# Required env vars (set via sbatch --export):
#   MODEL_ID   (e.g. TT-1, PT-A)
#   SOLID      (trivial | tetrahedron)
#   HDIM       hidden_dim
#   NHEAD      nhead
#   FC         flops_coef (6 for trivial, 72 for tetra)
#   MAX_ATOMS  dynamic-batching cap (atoms per batch). Replaces bs=64 fixed.
#
# Optional env vars (defaults match calibration recipe):
#   LS_INIT    layer_scale_init_value (default 1e-4)
#   MAX_STEPS  training step cap (default 300 for calibration; override for ablations)
#   RUN_TAG    appended to exp_name in wandb (default "atoms${MAX_ATOMS}")

set -euo pipefail

: "${MODEL_ID:?MODEL_ID is required}"
: "${SOLID:?SOLID is required}"
: "${HDIM:?HDIM is required}"
: "${NHEAD:?NHEAD is required}"
: "${FC:?FC is required}"
: "${MAX_ATOMS:?MAX_ATOMS is required}"
: "${LS_INIT:=1e-4}"
: "${MAX_STEPS:=300}"
: "${DATA:=omol_4m}"
: "${DYNAMIC:=true}"
: "${COMPILE:=true}"
: "${MAX_EDGES:=0}"  # 0 = unset (no edge cap). Set positive value to enable as OOM safety net.
: "${EXTRA_HYDRA:=}"  # extra space-separated Hydra overrides appended to the command
: "${RUN_TAG:=${DATA}-atoms${MAX_ATOMS}}"

source /scratch-shared/ebekkers/scaling-laws-venv/bin/activate
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PSL_FAST_BACKENDS=1

cd /scratch-shared/ebekkers/platonic-omol/training

echo "Run: v10-calib-${MODEL_ID}  solid=${SOLID} d=${HDIM} nhead=${NHEAD} fc=${FC} max_atoms=${MAX_ATOMS} ls_init=${LS_INIT} max_steps=${MAX_STEPS} tag=${RUN_TAG}"
echo "Date: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

python train_omol.py \
    force_field_module=platoformer \
    data=${DATA} \
    data.datamodule.validation_mode=heldout \
    data.datamodule.batch_size.train=64 \
    data.datamodule.batch_size.val=64 \
    data.datamodule.dynamic_batching=${DYNAMIC} \
    data.datamodule.max_atoms_per_batch=${MAX_ATOMS} \
    data.datamodule.max_atoms_per_batch_val=${MAX_ATOMS} \
    +data.datamodule.max_edges_per_batch=$([ "${MAX_EDGES}" = "0" ] && echo "null" || echo "${MAX_EDGES}") \
    +data.datamodule.max_edges_per_batch_val=$([ "${MAX_EDGES}" = "0" ] && echo "null" || echo "${MAX_EDGES}") \
    force_field_module.compile=${COMPILE} \
    force_field_module.net.hidden_dim=${HDIM} \
    force_field_module.net.nhead=${NHEAD} \
    force_field_module.net.num_layers=10 \
    force_field_module.net.solid_name=${SOLID} \
    force_field_module.net.dense_mode=false \
    force_field_module.net.layer_scale_init_value=${LS_INIT} \
    +force_field_module.net.rope_on_values=true \
    force_field_module.net.rope_sigma=1.5 \
    force_field_module.net.freq_init=random \
    force_field_module.net.learned_freqs=true \
    force_field_module.net.attention=true \
    force_field_module.net.avg_num_nodes=26.5 \
    force_field_module.train_augmentation=o3 \
    force_field_module.flops_coef=${FC} \
    +force_field_module.optimizer.r=2.0 \
    force_field_module.optimizer.lr=5e-4 \
    force_field_module.optimizer.num_warmup_steps=100 \
    force_field_module.net.charge_emb_dim=64 \
    force_field_module.net.spin_emb_dim=64 \
    trainer.max_epochs=1 \
    +trainer.max_steps=${MAX_STEPS} \
    trainer.gradient_clip_val=1 \
    trainer.gradient_clip_algorithm=norm \
    trainer.inference_mode=false \
    trainer.val_check_interval=1000 \
    +trainer.num_sanity_val_steps=0 \
    +trainer.limit_val_batches=0 \
    +trainer.limit_test_batches=0 \
    exp_name=v10-calib-${MODEL_ID}-${RUN_TAG} \
    model_name=platoformer \
    wandb.use_wandb=True \
    wandb.wandb_project=scaling-laws-symmetry \
    wandb.group=v10-calib \
    seed=1 \
    ${EXTRA_HYDRA}

echo "Done: $(date)"
