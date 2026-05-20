#!/bin/bash
#SBATCH --account=gusei11738
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch-shared/ebekkers/platonic-omol/training/slurm-%x-%j.out

# v10 production training — full 1-epoch run on omol_full with node-local data
# staging. Before training starts we copy the 527 GB train split (+ 23 GB val)
# from shared GPFS to /scratch-local/$SLURM_JOB_ID/ (per-node scratch, not
# shared across jobs). That sidesteps the cross-job GPFS contention that
# capped the 11-job calibration at ~3 k tok/s and gives each job full
# per-node bandwidth.
#
# Required env vars (set via sbatch --export):
#   MODEL_ID   (e.g. TT-1, PT-A)
#   SOLID      (trivial | tetrahedron)
#   HDIM       hidden_dim
#   NHEAD      nhead
#   FC         flops_coef (6 for trivial, 72 for tetra)
#
# Optional env vars:
#   MAX_ATOMS      default 6000
#   LS_INIT        default 1e-4
#   MAX_EPOCHS     default 1
#   MAX_STEPS      default unset (run full epoch). Set to cap early.
#   GROUP          wandb group, default scaling-v10-{trivial|tetra}
#   MAX_EDGES      optional edge-count safety cap (0 = off)
#   SKIP_STAGE     if "1", read directly from /scratch-shared (for debugging)

set -euo pipefail

: "${MODEL_ID:?MODEL_ID is required}"
: "${SOLID:?SOLID is required}"
: "${HDIM:?HDIM is required}"
: "${NHEAD:?NHEAD is required}"
: "${FC:?FC is required}"
: "${MAX_ATOMS:=6000}"
: "${LS_INIT:=1e-4}"
: "${MAX_EPOCHS:=1}"
: "${MAX_STEPS:=}"
: "${MAX_EDGES:=0}"
: "${SKIP_STAGE:=0}"

# Pick a default wandb group from the solid.
if [ -z "${GROUP:-}" ]; then
    if [ "$SOLID" = "tetrahedron" ]; then
        GROUP="scaling-v10-tetra"
    else
        GROUP="scaling-v10-trivial"
    fi
fi

source /scratch-shared/ebekkers/scaling-laws-venv/bin/activate
module load 2024
module load CUDA/12.6.0

export PYTHONPATH=/scratch-shared/ebekkers/platonic-omol/training
export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=120
export TMPDIR=/scratch-shared/ebekkers/tmp
export TORCH_HOME=/scratch-shared/ebekkers/torch_cache
export TRITON_CACHE_DIR=/scratch-shared/ebekkers/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/scratch-shared/ebekkers/torch_cache/inductor
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PSL_FAST_BACKENDS=1  # opt1: TF32 + cuDNN autotune

# --- Data staging -----------------------------------------------------------
SRC=/scratch-shared/ebekkers/omol25/open_mol
if [ "$SKIP_STAGE" = "1" ]; then
    echo "[$(date -Iseconds)] SKIP_STAGE=1 — using shared GPFS directly."
    export DATA_PATH=/scratch-shared/ebekkers/omol25
else
    LOCAL_ROOT=/scratch-local/${SLURM_JOB_ID}/omol25
    LOCAL=$LOCAL_ROOT/open_mol
    mkdir -p "$LOCAL/train" "$LOCAL/val"
    echo "[$(date -Iseconds)] Staging dataset to $LOCAL"
    df -h /scratch-local | tail -2 || true

    # Copy train shards in parallel. Each aselmdb is ~7 GB.
    echo "[$(date -Iseconds)] Copying train shards (parallel -P 8)..."
    ls "$SRC"/train/data*.aselmdb "$SRC"/train/data*.aselmdb-lock 2>/dev/null \
        | xargs -P 8 -I{} cp {} "$LOCAL/train/"
    cp "$SRC"/train/metadata.npz "$LOCAL/train/" 2>/dev/null || true
    echo "[$(date -Iseconds)] Train staged: $(du -sh "$LOCAL/train" | cut -f1)"

    # Copy val shards (smaller, 23 GB).
    echo "[$(date -Iseconds)] Copying val shards (parallel -P 4)..."
    ls "$SRC"/val/data*.aselmdb "$SRC"/val/data*.aselmdb-lock 2>/dev/null \
        | xargs -P 4 -I{} cp {} "$LOCAL/val/"
    cp "$SRC"/val/metadata.npz "$LOCAL/val/" 2>/dev/null || true
    echo "[$(date -Iseconds)] Val staged: $(du -sh "$LOCAL/val" | cut -f1)"

    export DATA_PATH=$LOCAL_ROOT
fi

cd /scratch-shared/ebekkers/platonic-omol/training

echo "=========================================="
echo "Run: V10-${MODEL_ID}   group=${GROUP}"
echo "solid=${SOLID} d=${HDIM} nhead=${NHEAD} fc=${FC} ls=${LS_INIT}"
echo "max_atoms=${MAX_ATOMS} max_edges=${MAX_EDGES} max_epochs=${MAX_EPOCHS} max_steps=${MAX_STEPS:-<unset>}"
echo "DATA_PATH=${DATA_PATH}"
echo "Node: $(hostname)   Date: $(date -Iseconds)"
echo "=========================================="
nvidia-smi --query-gpu=name --format=csv,noheader

# Assemble optional overrides.
EXTRA=()
if [ -n "$MAX_STEPS" ]; then
    EXTRA+=("+trainer.max_steps=${MAX_STEPS}")
fi
if [ "$MAX_EDGES" != "0" ]; then
    EXTRA+=("+data.datamodule.max_edges_per_batch=${MAX_EDGES}")
    EXTRA+=("+data.datamodule.max_edges_per_batch_val=${MAX_EDGES}")
else
    EXTRA+=("+data.datamodule.max_edges_per_batch=null")
    EXTRA+=("+data.datamodule.max_edges_per_batch_val=null")
fi

python train_omol.py \
    force_field_module=platoformer \
    data=omol_full \
    data.datamodule.batch_size.train=64 \
    data.datamodule.batch_size.val=64 \
    data.datamodule.dynamic_batching=true \
    data.datamodule.max_atoms_per_batch=${MAX_ATOMS} \
    data.datamodule.max_atoms_per_batch_val=${MAX_ATOMS} \
    force_field_module.compile=true \
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
    trainer.max_epochs=${MAX_EPOCHS} \
    trainer.gradient_clip_val=1 \
    trainer.gradient_clip_algorithm=norm \
    trainer.inference_mode=false \
    trainer.val_check_interval=1000 \
    +trainer.limit_val_batches=500 \
    exp_name=V10-${MODEL_ID} \
    model_name=platoformer \
    wandb.use_wandb=True \
    wandb.wandb_project=scaling-laws-symmetry \
    wandb.group=${GROUP} \
    seed=1 \
    "${EXTRA[@]}"

echo "[$(date -Iseconds)] Training done."
