#!/bin/bash
# Delta launcher for the PT-2 h1920/l8 20-epoch OMol run.
#
# This mirrors the Snellius/Hydra config:
#   pt2-h1920-l8-ls1e-4-ffn2-actsin-ractsin-rs2.0-ema0.99-wd1e-5-20ep-n20000
# but uses paths available on ivi-h1/delta.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --job-name=pt2_h1920_l8_exact
#SBATCH --output=/home/thadziv/GitHub/platonic-omol/logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/platonic-omol}"
TRAINING_DIR="${TRAINING_DIR:-$REPO_ROOT/training}"
PYTHON_BIN="${PYTHON_BIN:-/home/thadziv/GitHub/erwin/erwin/bin/python}"
FAIRCHEM_SRC="${FAIRCHEM_SRC:-/home/thadziv/GitHub/fairchem/src}"

DATA_PATH="${DATA_PATH:-/home/ebekker/data/omol}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$DATA_PATH/open_mol/train_4M}"
VAL_DATA_PATH="${VAL_DATA_PATH:-$DATA_PATH/open_mol/val}"

RUN_ROOT="${RUN_ROOT:-/home/thadziv/platonic_omol_runs}"
CACHE_ROOT="${CACHE_ROOT:-/home/thadziv/platonic_omol_cache}"
SLURM_RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"

MAX_ATOMS="${MAX_ATOMS:-20000}"
MAX_EDGES="${MAX_EDGES:-4000000}"
WD="${WD:-1e-5}"
FFN_FACTOR="${FFN_FACTOR:-2}"
LAYER_SCALE="${LAYER_SCALE:-1e-4}"
ACTIVATION="${ACTIVATION:-sin}"
READOUT_ACTIVATION="${READOUT_ACTIVATION:-sin}"
NUM_LAYERS="${NUM_LAYERS:-8}"
OPTIMIZER="${OPTIMIZER:-adamw}"
ROPE_SIGMA="${ROPE_SIGMA:-2.0}"
EMA_DECAY="${EMA_DECAY:-0.99}"
SOLID_NAME="${SOLID_NAME:-tetrahedron}"
LR="${LR:-5e-4}"
WANDB_GROUP="${WANDB_GROUP:-platonic_pt2_exact_config}"
SKIP_LOSS_ABOVE="${SKIP_LOSS_ABOVE:-1000}"

mkdir -p "$REPO_ROOT/logs" "$RUN_ROOT" "$CACHE_ROOT"/{tmp,torch,triton,inductor,wandb,checkpoints}

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[error] PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [ ! -d "$TRAINING_DIR" ]; then
  echo "[error] TRAINING_DIR does not exist: $TRAINING_DIR" >&2
  exit 1
fi
if [ ! -d "$FAIRCHEM_SRC" ]; then
  echo "[error] FAIRCHEM_SRC does not exist: $FAIRCHEM_SRC" >&2
  exit 1
fi
if [ ! -d "$TRAIN_DATA_PATH" ]; then
  echo "[error] TRAIN_DATA_PATH does not exist: $TRAIN_DATA_PATH" >&2
  exit 1
fi
if [ ! -d "$VAL_DATA_PATH" ]; then
  echo "[error] VAL_DATA_PATH does not exist: $VAL_DATA_PATH" >&2
  exit 1
fi

if [ "$OPTIMIZER" = "adamw" ]; then
  OPT_OVERRIDES=(
    force_field_module.optimizer.name=adamw
    force_field_module.optimizer.scheduler_name=cosine_annealing_ws
    force_field_module.optimizer.num_warmup_steps=0.01
  )
elif [ "$OPTIMIZER" = "free" ]; then
  OPT_OVERRIDES=(
    force_field_module.optimizer.name=free
    force_field_module.optimizer.num_warmup_steps=100
    +force_field_module.optimizer.r=2.0
  )
else
  echo "[error] unknown OPTIMIZER=${OPTIMIZER}; expected adamw or free" >&2
  exit 2
fi

EMA_OVERRIDES=()
if [ "$EMA_DECAY" != "null" ]; then
  EMA_OVERRIDES=(
    +ema.decay=${EMA_DECAY}
    +ema.warmup_steps=2000
  )
fi

WD_TAG=$(echo "$WD" | sed -e 's/^0\.0$/0/' -e 's/^0$/0/')
FFN_TAG=""
if [ "$FFN_FACTOR" != "4" ]; then
  FFN_TAG="-ffn${FFN_FACTOR}"
fi
if [ "$LAYER_SCALE" = "null" ]; then
  LS_TAG="nols"
else
  LS_TAG="ls$(echo "$LAYER_SCALE" | sed -e 's/^1\.0$/1/' -e 's/^0\.0$/0/')"
fi
ACT_TAG=""
if [ "$ACTIVATION" != "gelu" ]; then
  ACT_TAG="-act${ACTIVATION}"
fi
RACT_TAG=""
if [ "$READOUT_ACTIVATION" != "null" ]; then
  RACT_TAG="-ract${READOUT_ACTIVATION}"
fi
OPT_TAG=""
if [ "$OPTIMIZER" != "adamw" ]; then
  OPT_TAG="-${OPTIMIZER}"
fi
RS_TAG=""
if [ "$ROPE_SIGMA" != "4.0" ]; then
  RS_TAG="-rs${ROPE_SIGMA}"
fi
EMA_TAG=""
if [ "$EMA_DECAY" != "null" ]; then
  EMA_TAG="-ema${EMA_DECAY}"
fi
SOLID_TAG=""
if [ "$SOLID_NAME" != "tetrahedron" ]; then
  SOLID_TAG="-solid${SOLID_NAME}"
fi

EXP_NAME="${EXP_NAME:-pt2-h1920-l${NUM_LAYERS}-${LS_TAG}${FFN_TAG}${ACT_TAG}${RACT_TAG}${OPT_TAG}${RS_TAG}${EMA_TAG}${SOLID_TAG}-wd${WD_TAG}-20ep-n${MAX_ATOMS}}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/${EXP_NAME}_${SLURM_RUN_ID}}"

export DATA_PATH
export PYTHONPATH="$FAIRCHEM_SRC:$TRAINING_DIR:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-120}"
export WANDB_DIR="${WANDB_DIR:-$CACHE_ROOT/wandb}"
export TMPDIR="${TMPDIR:-$CACHE_ROOT/tmp}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/inductor}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$CACHE_ROOT/checkpoints}"
export PYTHONUNBUFFERED=1

CUDA_PIP_LIB_DIRS="$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
import site

roots = []
for root in site.getsitepackages():
    nvidia_root = Path(root) / "nvidia"
    if nvidia_root.is_dir():
        roots.extend(str(path) for path in sorted(nvidia_root.glob("*/lib")) if path.is_dir())
print(":".join(roots))
PY
)"
if [ -n "$CUDA_PIP_LIB_DIRS" ]; then
  export LD_LIBRARY_PATH="$CUDA_PIP_LIB_DIRS:${LD_LIBRARY_PATH:-}"
fi

cd "$TRAINING_DIR"

echo "=== PT-2 h1920/l8 exact-config delta run ==="
echo "Date:       $(date -Iseconds)"
echo "Job:        ${SLURM_JOB_ID:-?} on $(hostname)"
echo "Repo:       $REPO_ROOT"
echo "Commit:     $(git -C "$REPO_ROOT" rev-parse --short HEAD)"
echo "Python:     $PYTHON_BIN"
echo "Data:       $DATA_PATH"
echo "Train:      $TRAIN_DATA_PATH"
echo "Val:        $VAL_DATA_PATH"
echo "Output:     $OUTPUT_DIR"
echo "Exp:        $EXP_NAME"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

"$PYTHON_BIN" - <<'PY'
from flash_attn import flash_attn_varlen_func
from fairchem.core.datasets import AseDBDataset
print("flash_attn_varlen_func import OK")
print("fairchem AseDBDataset import OK")
PY

OVERRIDES=(
  +precision=fp32_baseline
  force_field_module=platoformer
  data=omol_4m
  data.datamodule.data.train_data_path=${TRAIN_DATA_PATH}
  data.datamodule.data.val_data_path=${VAL_DATA_PATH}
  data.datamodule.data.pin_memory=True
  data.datamodule.data.num_workers=16
  data.datamodule.data.seed=42
  data.datamodule.data.train_size=0.9
  data.datamodule.data.shuffle=True
  +data.datamodule.prefetch_factor=4
  data.datamodule.batch_size.train=64
  data.datamodule.batch_size.val=64
  data.datamodule.dynamic_batching=true
  data.datamodule.max_atoms_per_batch=${MAX_ATOMS}
  data.datamodule.max_atoms_per_batch_val=${MAX_ATOMS}
  +data.datamodule.max_edges_per_batch=${MAX_EDGES}
  +data.datamodule.max_edges_per_batch_val=${MAX_EDGES}
  data.datamodule.validation_mode=heldout
  data.datamodule.precompute_reference_energy=true
  force_field_module.net.hidden_dim=1920
  force_field_module.net.nhead=60
  force_field_module.net.num_layers=${NUM_LAYERS}
  force_field_module.net.ffn_dim_factor=${FFN_FACTOR}
  force_field_module.net.solid_name=${SOLID_NAME}
  force_field_module.net.activation=${ACTIVATION}
  +force_field_module.net.readout_activation=${READOUT_ACTIVATION}
  force_field_module.net.dense_mode=false
  force_field_module.net.layer_scale_init_value=${LAYER_SCALE}
  +force_field_module.net.rope_on_values=true
  force_field_module.net.rope_sigma=${ROPE_SIGMA}
  force_field_module.net.freq_init=random
  force_field_module.net.learned_freqs=true
  force_field_module.net.attention=true
  force_field_module.net.avg_num_nodes=26.5
  force_field_module.net.attention_backend=flash
  force_field_module.net.chgspin_mode=add
  force_field_module.train_augmentation=o3
  force_field_module.flops_coef=72
  force_field_module.compile=true
  +force_field_module.skip_loss_above=${SKIP_LOSS_ABOVE}
  "${OPT_OVERRIDES[@]}"
  "${EMA_OVERRIDES[@]}"
  force_field_module.optimizer.lr=${LR}
  force_field_module.optimizer.lr_min=1e-6
  force_field_module.optimizer.weight_decay=${WD}
  force_field_module.optimizer.e_loss_name=per_atom_mae
  force_field_module.optimizer.f_loss_name=l2norm
  force_field_module.optimizer.e_weight=10
  force_field_module.optimizer.f_weight=10
  force_field_module.optimizer.e_weight_warmup_steps=0
  force_field_module.optimizer.num_restarts=10
  force_field_module.optimizer.amsgrad=False
  force_field_module.train_mean=0
  force_field_module.train_rmsd=1.433569
  trainer.max_epochs=20
  trainer.gradient_clip_val=1
  trainer.gradient_clip_algorithm=norm
  trainer.inference_mode=false
  trainer.val_check_interval=5000
  +trainer.limit_val_batches=500
  trainer.accelerator=gpu
  trainer.devices=auto
  trainer.strategy=auto
  trainer.fast_dev_run=false
  trainer.detect_anomaly=false
  trainer.overfit_batches=0
  trainer.check_val_every_n_epoch=1
  trainer.enable_progress_bar=false
  matmul_precision=high
  cudnn_benchmark=false
  cudnn_deterministic=true
  exp_name=${EXP_NAME}
  model_name=platoformer
  wandb.use_wandb=True
  wandb.entity=null
  wandb.wandb_project=matterformer_omol_4m
  wandb.group=${WANDB_GROUP}
  seed=1
  hydra.run.dir=${OUTPUT_DIR}/hydra
)

echo "=== Hydra parse-check ==="
"$PYTHON_BIN" train_omol.py "${OVERRIDES[@]}" --cfg job > "$OUTPUT_DIR.hydra.yaml"
echo "Hydra config OK: $OUTPUT_DIR.hydra.yaml"

if [ "${DRY_RUN_ONLY:-0}" = "1" ]; then
  echo "DRY_RUN_ONLY=1; exiting before training."
  exit 0
fi

"$PYTHON_BIN" train_omol.py "${OVERRIDES[@]}"
