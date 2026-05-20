#!/bin/bash
#SBATCH --account=gusei11738
#SBATCH --partition=gpu_h100
#SBATCH --time=5-00:00:00
#SBATCH --requeue
#SBATCH --output=/scratch-shared/ebekkers/platonic-omol/training/slurm-%x-%j.out

# Multi-GPU DDP variant of run_v11_train_tmpfs.sh. The existing single-GPU
# script stays authoritative for production; this one is for speed-up
# probes / continuations that want parallel GPUs on the same node.
#
# Pattern: --ntasks-per-node=1, --gres=gpu:h100:N. Single Python process
# per node, Lightning's DDPStrategy forks N DDP workers internally (via
# torch.multiprocessing). We explicitly suppress Lightning's SLURM auto-
# detection (SLURM_JOB_NAME=bash) so it uses LightningEnvironment instead
# of SLURMEnvironment — SLURMEnvironment requires tasks_per_node==devices
# which collides with srun GPU allocation on Snellius.
#
# sbatch fields set on the command line:
#   --gres=gpu:h100:N
#   --ntasks-per-node=1
#   --cpus-per-task=C        # exactly 16*N to match N-slot SBU pricing
#   --mem=M                  # exactly 184*N to stay in N-slot pricing;
#                            # going over rounds up to the next SBU slot
#                            # (Snellius charges on the max dimension)
#
# Required env vars:
#   MODEL_ID, SOLID, HDIM, NHEAD, FC
# Optional:
#   MAX_ATOMS (default 6000 per GPU — effective global batch scales with NGPUs),
#   LS_INIT (1e-4), MAX_EPOCHS (1), MAX_STEPS (unset), GROUP, MAX_EDGES (0),
#   CHECKPOINT, RESUME_FROM_RUN, EXTRA_HYDRA, SKIP_STAGE (0)

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
    if [ "$SOLID" = "tetrahedron" ]; then GROUP="scaling-v11-tetra"
    else GROUP="scaling-v11-trivial"; fi
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

# --- One-time tmpfs stage per NODE (not per task) ---------------------------
SRC=/scratch-shared/ebekkers/omol25/open_mol_memmap
STAGE_ROOT=/tmp/${USER}-${SLURM_JOB_ID}/omol25
STAGE=${STAGE_ROOT}/open_mol_memmap
if [ "$SKIP_STAGE" = "1" ]; then
    echo "[$(date -Iseconds)] SKIP_STAGE=1"
    export DATA_PATH=/scratch-shared/ebekkers/omol25
else
    mkdir -p "$STAGE/train" "$STAGE/val"
    echo "[$(date -Iseconds)] Staging memmap to $STAGE on $(hostname)"
    echo "[/tmp] $(df -h /tmp | tail -1)"
    stage_split () {
        local split="$1"
        local src_split="$SRC/$split"
        local dst_split="$STAGE/$split"
        ls "$src_split"/*.bin "$src_split"/manifest.json 2>/dev/null \
            | xargs -P 4 -I{} cp {} "$dst_split/"
        echo "[$(date -Iseconds)] $split staged: $(du -sh "$dst_split" | cut -f1)"
    }
    stage_split val
    stage_split train
    echo "[$(date -Iseconds)] Stage complete."
    export DATA_PATH=$STAGE_ROOT
    trap "echo '[cleanup] rm -rf $STAGE_ROOT'; rm -rf $STAGE_ROOT" EXIT
fi

cd /scratch-shared/ebekkers/platonic-omol/training

NGPUS=${SLURM_GPUS_ON_NODE:-1}
# Trick Lightning into using LightningEnvironment (pure-Python DDP launch)
# instead of SLURMEnvironment. Without this Lightning enforces
# devices == ntasks_per_node, which we can't satisfy on Snellius because
# srun + --gpus-per-task restricts GPU visibility per task.
export SLURM_JOB_NAME=bash
echo "=========================================="
echo "Run: V11-${MODEL_ID}  NGPUS=${NGPUS}  group=${GROUP}"
echo "solid=${SOLID} d=${HDIM} nhead=${NHEAD} fc=${FC}"
echo "max_atoms=${MAX_ATOMS} max_epochs=${MAX_EPOCHS} max_steps=${MAX_STEPS:-<unset>}"
echo "DATA_PATH=${DATA_PATH}"
echo "Node: $(hostname)   Date: $(date -Iseconds)"
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "=========================================="

EXTRA=()
if [ -n "$MAX_STEPS" ]; then EXTRA+=("+trainer.max_steps=${MAX_STEPS}"); fi
if [ "$MAX_EDGES" != "0" ]; then
    EXTRA+=("+data.datamodule.max_edges_per_batch=${MAX_EDGES}")
    EXTRA+=("+data.datamodule.max_edges_per_batch_val=${MAX_EDGES}")
else
    EXTRA+=("+data.datamodule.max_edges_per_batch=null")
    EXTRA+=("+data.datamodule.max_edges_per_batch_val=null")
fi
if [ -n "$CHECKPOINT" ]; then EXTRA+=("+checkpoint_path=${CHECKPOINT}"); fi
if [ -n "$RESUME_FROM_RUN" ]; then EXTRA+=("+resume_from_run=${RESUME_FROM_RUN}"); fi
if [ -n "$EXTRA_HYDRA" ]; then
    read -r -a ETOKS <<< "$EXTRA_HYDRA"
    EXTRA+=("${ETOKS[@]}")
fi

# Single Python process; Lightning's DDPStrategy forks N workers. No srun.
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
    trainer.devices=${NGPUS} \
    trainer.strategy=ddp \
    +trainer.use_distributed_sampler=false \
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

echo "[$(date -Iseconds)] Done."
