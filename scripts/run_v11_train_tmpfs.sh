#!/bin/bash
#SBATCH --account=gusei11738
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=184G
#SBATCH --time=5-00:00:00
#SBATCH --requeue
#SBATCH --output=/scratch-shared/ebekkers/platonic-omol/training/slurm-%x-%j.out

# v11 production training — stage the memmap dataset into node-local tmpfs (/tmp,
# RAM-backed on gpu_h100 diskless nodes) before training starts. Memmap size is
# 131G (train 124G + val 6.7G) vs the 378G tmpfs limit, so it fits comfortably.
#
# Rationale:
#   - gpu_h100 nodes are diskless: no NVMe. /scratch-local resolves to GPFS, not
#     actual node-local storage. The previous run_v10_train.sh staged to
#     /scratch-local → same GPFS backend, no contention relief.
#   - /tmp IS tmpfs (RAM-backed, 378 GB per node). Reads hit DRAM at 200+ GB/s
#     with zero cross-job contention. 131 GB fits in node RAM (737 GB total).
#   - Mem accounting: tmpfs usage counts against --mem. Snellius SBU slots are
#     184 GB per GPU. 131 GB stage + ~50 GB training heap fits in 184 GB, so we
#     stay in the 1-GPU slot pricing instead of being charged as 2 GPUs.
#     If OOMs appear during test-eval or large model checkpointing, bump to
#     --mem=368G (= 2 slots) — but avoid any value between 184 and 368.
#
# Required env vars (sbatch --export):
#   MODEL_ID, SOLID, HDIM, NHEAD, FC
# Optional:
#   MAX_ATOMS (default 6000), LS_INIT (default 1e-4), MAX_EPOCHS (default 1),
#   MAX_STEPS (unset = full epoch), GROUP (default scaling-v11-{trivial|tetra}),
#   MAX_EDGES (0 = off), SKIP_STAGE (1 = read directly from GPFS),
#   CHECKPOINT (resume from this .ckpt path — adds +checkpoint_path override),
#   RESUME_FROM_RUN (original wandb run id, logged into config for later stitching),
#   EXTRA_HYDRA (space-separated Hydra overrides appended to the python command).

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
: "${CHECKPOINT:=}"
: "${RESUME_FROM_RUN:=}"
: "${EXTRA_HYDRA:=}"

if [ -z "${GROUP:-}" ]; then
    if [ "$SOLID" = "tetrahedron" ]; then
        GROUP="scaling-v11-tetra"
    else
        GROUP="scaling-v11-trivial"
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
export PSL_FAST_BACKENDS=1

# --- Stage memmap to node-local tmpfs ---------------------------------------
SRC=/scratch-shared/ebekkers/omol25/open_mol_memmap
STAGE_ROOT=/tmp/${USER}-${SLURM_JOB_ID}/omol25
STAGE=${STAGE_ROOT}/open_mol_memmap

if [ "$SKIP_STAGE" = "1" ]; then
    echo "[$(date -Iseconds)] SKIP_STAGE=1 — reading memmap directly from GPFS."
    export DATA_PATH=/scratch-shared/ebekkers/omol25
else
    mkdir -p "$STAGE/train" "$STAGE/val"
    echo "[$(date -Iseconds)] Staging memmap to $STAGE"
    echo "[node] $(hostname)   /tmp: $(df -h /tmp | tail -1)"

    # 7 per-field memmaps + manifest + offsets per split. Parallel cp at -P 4
    # (tmpfs is DRAM-bound; more parallel doesn't speed it up but costs CPU).
    stage_split () {
        local split="$1"
        local src_split="$SRC/$split"
        local dst_split="$STAGE/$split"
        echo "[$(date -Iseconds)] Copying $split shards..."
        ls "$src_split"/*.bin "$src_split"/manifest.json 2>/dev/null \
            | xargs -P 4 -I{} cp {} "$dst_split/"
        echo "[$(date -Iseconds)] $split staged: $(du -sh "$dst_split" | cut -f1)"
    }
    stage_split val
    stage_split train

    echo "[$(date -Iseconds)] Stage complete. /tmp usage: $(df -h /tmp | tail -1)"
    export DATA_PATH=$STAGE_ROOT

    # Ensure the tmpfs stage is cleaned up on exit (including scancel / requeue).
    trap "echo '[cleanup] rm -rf $STAGE_ROOT'; rm -rf $STAGE_ROOT" EXIT
fi

cd /scratch-shared/ebekkers/platonic-omol/training

echo "=========================================="
echo "Run: V11-${MODEL_ID}   group=${GROUP}"
echo "solid=${SOLID} d=${HDIM} nhead=${NHEAD} fc=${FC} ls=${LS_INIT}"
echo "max_atoms=${MAX_ATOMS} max_edges=${MAX_EDGES} max_epochs=${MAX_EPOCHS} max_steps=${MAX_STEPS:-<unset>}"
echo "DATA_PATH=${DATA_PATH}"
echo "Node: $(hostname)   Date: $(date -Iseconds)"
echo "=========================================="
nvidia-smi --query-gpu=name --format=csv,noheader

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
if [ -n "$CHECKPOINT" ]; then
    EXTRA+=("+checkpoint_path=${CHECKPOINT}")
fi
if [ -n "$RESUME_FROM_RUN" ]; then
    # Log the source run id into config for later curve stitching.
    EXTRA+=("+resume_from_run=${RESUME_FROM_RUN}")
fi
# EXTRA_HYDRA is appended at the end so it can override any of the above.
if [ -n "$EXTRA_HYDRA" ]; then
    # Split on whitespace so each "+key=value" becomes its own argv element.
    read -r -a EXTRA_HYDRA_TOKS <<< "$EXTRA_HYDRA"
    EXTRA+=("${EXTRA_HYDRA_TOKS[@]}")
fi

python train_omol.py \
    force_field_module=platoformer \
    data=omol_full_memmap \
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
    exp_name=V11-${MODEL_ID} \
    model_name=platoformer \
    wandb.use_wandb=True \
    wandb.wandb_project=scaling-laws-symmetry \
    wandb.group=${GROUP} \
    seed=1 \
    "${EXTRA[@]}"

echo "[$(date -Iseconds)] Training done."
