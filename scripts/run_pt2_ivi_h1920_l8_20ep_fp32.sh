#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --output=/home/ebekker/platonic-omol/logs/%x-%j.out

# IVI mirror of scripts/run_pt2_h1920_l8_20ep_fp32.sh (the canonical Snellius
# hardening launcher). Same env-var interface and overrides; the only changes
# are IVI plumbing: venv path, CUDA module, DATA_PATH, no Lmod, no
# /scratch-shared.
#
# Partition + GPU type are passed via sbatch CLI so the same script targets
# all6000 (rtx_6000 / sm_86) or geodude (rtx_a5000 / sm_86). Flash 2.7.4 was
# built locally on IVI for sm_86/89 (build 168484, install_flash_attn_ivi_v2.sh).
#
# Canonical submit to reproduce yxh5y61s (Snellius 22783311) on all6000 with
# flash + maxed batch:
#   sbatch -p all6000 --account=all6000users --gres=gpu:rtx_6000:1 \
#     --export=ALL,MAX_ATOMS=8000,WD=1e-8,LAYER_SCALE=null,FFN_FACTOR=2,\
#ACTIVATION=gelu,READOUT_ACTIVATION=gelu,ROPE_SIGMA=2.0,QK_NORM=true,\
#SWIGLU=true,USE_KEY=true,CHGSPIN_MODE=off,CHGSPIN_FILM=true,\
#CHGSPIN_MIX_INIT_STD=0.014,E_WEIGHT=10,F_WEIGHT=20,EMA_DECAY=0.99 \
#     --job-name=PT2-ivi-all6000-qkn-sg-uk-flash \
#     scripts/run_pt2_ivi_h1920_l8_20ep_fp32.sh

set -e
mkdir -p /home/ebekker/platonic-omol/logs

# IVI default soft ulimit -n is 1024 — too low for DataLoader workers +
# dynamo's pipe-based AOT-repro profiler. Without this, inductor compile fails
# with OSError: [Errno 24] Too many open files mid-training and falls back to
# eager (or kills the dataloader). Raise to the hard cap. Same fix the hipster
# launcher applies.
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true

MAX_ATOMS="${MAX_ATOMS:-8000}"
MAX_EDGES="${MAX_EDGES:-$((MAX_ATOMS * 200))}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DYNAMIC_BATCHING="${DYNAMIC_BATCHING:-true}"
WD="${WD:?must set WD env var (e.g. 0.0, 1e-5, 1e-4, 1e-3)}"
HIDDEN_DIM="${HIDDEN_DIM:-1920}"
NHEAD="${NHEAD:-60}"
FFN_FACTOR="${FFN_FACTOR:-4}"
LAYER_SCALE="${LAYER_SCALE:-null}"
ACTIVATION="${ACTIVATION:-gelu}"
READOUT_ACTIVATION="${READOUT_ACTIVATION:-null}"
NUM_LAYERS="${NUM_LAYERS:-8}"
OPTIMIZER="${OPTIMIZER:-adamw}"
ROPE_SIGMA="${ROPE_SIGMA:-4.0}"
ROPE_ON_VALUES="${ROPE_ON_VALUES:-true}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
QK_NORM="${QK_NORM:-false}"
SWIGLU="${SWIGLU:-false}"
USE_KEY="${USE_KEY:-false}"
NORM_TYPE="${NORM_TYPE:-layernorm}"
PRECISION="${PRECISION:-fp32_baseline}"
RADIUS="${RADIUS:-null}"
LOCAL_GLOBAL="${LOCAL_GLOBAL:-false}"
if [ "${RADIUS}" != "null" ] && [ "${LOCAL_GLOBAL}" != "true" ]; then
    ATTENTION_BACKEND=scatter
fi
CHGSPIN_MODE="${CHGSPIN_MODE:-add}"
CHGSPIN_LAYERWISE="${CHGSPIN_LAYERWISE:-false}"
CHGSPIN_LAYERWISE_GATE="${CHGSPIN_LAYERWISE_GATE:-false}"
CHGSPIN_FILM="${CHGSPIN_FILM:-false}"
CHGSPIN_MIX_INIT_STD="${CHGSPIN_MIX_INIT_STD:-0.02}"
E_WEIGHT="${E_WEIGHT:-10}"
F_WEIGHT="${F_WEIGHT:-10}"
LR="${LR:-5e-4}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-5000}"
EMA_DECAY="${EMA_DECAY:-null}"
SOLID_NAME="${SOLID_NAME:-tetrahedron}"
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
    echo "ERROR: unknown OPTIMIZER=${OPTIMIZER}; expected 'adamw' or 'free'"; exit 2
fi
EMA_OVERRIDES=()
if [ "${EMA_DECAY}" != "null" ]; then
    EMA_OVERRIDES=(
        +ema.decay=${EMA_DECAY}
        +ema.warmup_steps=2000
    )
fi

source /home/ebekker/platonic-omol/venv/bin/activate

# IVI default cuda symlink resolves to cuda-13.1 which torch (cu128) doesn't
# link cleanly against. Pin to 12.9 (matches the venv build env).
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export DATA_PATH=/home/ebekker/data/omol
export PYTHONPATH=/home/ebekker/platonic-omol/training
export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=120
export TORCHDYNAMO_VERBOSE=1

cd /home/ebekker/platonic-omol/training

WD_TAG=$(echo "$WD" | sed -e 's/^0\.0$/0/' -e 's/^0$/0/')
FFN_TAG=""
if [ "${FFN_FACTOR}" != "4" ]; then
    FFN_TAG="-ffn${FFN_FACTOR}"
fi
if [ "${LAYER_SCALE}" = "null" ]; then
    LS_TAG="nols"
else
    LS_TAG="ls$(echo ${LAYER_SCALE} | sed -e 's/^1\.0$/1/' -e 's/^0\.0$/0/')"
fi
ACT_TAG=""
if [ "${ACTIVATION}" != "gelu" ]; then
    ACT_TAG="-act${ACTIVATION}"
fi
RACT_TAG=""
if [ "${READOUT_ACTIVATION}" != "null" ]; then
    RACT_TAG="-ract${READOUT_ACTIVATION}"
fi
OPT_TAG=""
if [ "${OPTIMIZER}" != "adamw" ]; then
    OPT_TAG="-${OPTIMIZER}"
fi
RS_TAG=""
if [ "${ROPE_SIGMA}" != "4.0" ]; then
    RS_TAG="-rs${ROPE_SIGMA}"
fi
ROV_TAG=""
if [ "${ROPE_ON_VALUES}" != "true" ]; then
    ROV_TAG="-rovF"
fi
BACKEND_TAG=""
if [ "${ATTENTION_BACKEND}" != "flash" ]; then
    BACKEND_TAG="-${ATTENTION_BACKEND}"
fi
QK_TAG=""
if [ "${QK_NORM}" = "true" ]; then
    QK_TAG="-qkn"
fi
SG_TAG=""
if [ "${SWIGLU}" = "true" ]; then
    SG_TAG="-sg"
fi
UK_TAG=""
if [ "${USE_KEY}" = "true" ]; then
    UK_TAG="-uk"
fi
NT_TAG=""
if [ "${NORM_TYPE}" != "layernorm" ]; then
    NT_TAG="-${NORM_TYPE}"
fi
CSM_TAG=""
if [ "${CHGSPIN_MODE}" != "add" ]; then
    CSM_TAG="-csM${CHGSPIN_MODE}"
fi
CSL_TAG=""
if [ "${CHGSPIN_LAYERWISE}" = "true" ]; then
    if [ "${CHGSPIN_LAYERWISE_GATE}" = "true" ]; then
        CSL_TAG="-csLg"
    else
        CSL_TAG="-csL"
    fi
fi
CSF_TAG=""
if [ "${CHGSPIN_FILM}" = "true" ]; then
    CSF_TAG="-csFiLM"
fi
CSI_TAG=""
if [ "${CHGSPIN_MIX_INIT_STD}" != "0.02" ]; then
    CSI_TAG="-csi${CHGSPIN_MIX_INIT_STD}"
fi
EW_TAG=""
if [ "${E_WEIGHT}" != "10" ]; then
    EW_TAG="-eW${E_WEIGHT}"
fi
FW_TAG=""
if [ "${F_WEIGHT}" != "10" ]; then
    FW_TAG="-fW${F_WEIGHT}"
fi
R_TAG=""
if [ "${RADIUS}" != "null" ]; then
    R_TAG="-r${RADIUS}"
fi
LG_TAG=""
if [ "${LOCAL_GLOBAL}" = "true" ]; then
    LG_TAG="-lg"
fi
EMA_TAG=""
if [ "${EMA_DECAY}" != "null" ]; then
    EMA_TAG="-ema${EMA_DECAY}"
fi
LR_TAG=""
if [ "${LR}" != "5e-4" ]; then
    LR_TAG="-lr${LR}"
fi
BS_TAG=""
if [ "${BATCH_SIZE}" != "64" ]; then
    BS_TAG="-bs${BATCH_SIZE}"
fi
if [ "${DYNAMIC_BATCHING}" != "true" ]; then
    BS_TAG="${BS_TAG}-static"
fi
SOLID_TAG=""
if [ "${SOLID_NAME}" != "tetrahedron" ]; then
    SOLID_TAG="-solid${SOLID_NAME}"
fi
PART_TAG="-${SLURM_JOB_PARTITION:-ivi}"
EXP_NAME="pt2-ivi-h${HIDDEN_DIM}-l${NUM_LAYERS}-${LS_TAG}${FFN_TAG}${ACT_TAG}${RACT_TAG}${OPT_TAG}${RS_TAG}${ROV_TAG}${BACKEND_TAG}${QK_TAG}${SG_TAG}${UK_TAG}${NT_TAG}${CSM_TAG}${CSL_TAG}${CSF_TAG}${CSI_TAG}${EW_TAG}${FW_TAG}${R_TAG}${LG_TAG}${EMA_TAG}${LR_TAG}${BS_TAG}${SOLID_TAG}-wd${WD_TAG}-20ep-n${MAX_ATOMS}${PART_TAG}"

echo "=== PT-2 IVI 20ep (max_atoms=${MAX_ATOMS}, backend=${ATTENTION_BACKEND}): ${EXP_NAME} ==="
echo "Date:  $(date)"
echo "Job:   ${SLURM_JOB_ID:-?} on $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Branch + commit:"
git -C /home/ebekker/platonic-omol log --oneline -3

if [ "${ATTENTION_BACKEND}" = "flash" ]; then
    python3 -c "from flash_attn import flash_attn_varlen_func; print('flash_attn_varlen_func import OK')"
fi

OVERRIDES=(
    +precision=${PRECISION}
    force_field_module=platoformer
    data=omol_4m
    data.datamodule.data.num_workers=16
    +data.datamodule.prefetch_factor=4
    data.datamodule.batch_size.train=${BATCH_SIZE}
    data.datamodule.batch_size.val=${BATCH_SIZE}
    data.datamodule.dynamic_batching=${DYNAMIC_BATCHING}
    data.datamodule.max_atoms_per_batch=${MAX_ATOMS}
    data.datamodule.max_atoms_per_batch_val=${MAX_ATOMS}
    +data.datamodule.max_edges_per_batch=${MAX_EDGES}
    +data.datamodule.max_edges_per_batch_val=${MAX_EDGES}
    data.datamodule.validation_mode=heldout
    data.datamodule.data.val_data_path=${DATA_PATH}/open_mol/val
    force_field_module.net.hidden_dim=${HIDDEN_DIM}
    force_field_module.net.nhead=${NHEAD}
    force_field_module.net.num_layers=${NUM_LAYERS}
    force_field_module.net.ffn_dim_factor=${FFN_FACTOR}
    force_field_module.net.solid_name=${SOLID_NAME}
    force_field_module.net.activation=${ACTIVATION}
    +force_field_module.net.readout_activation=${READOUT_ACTIVATION}
    force_field_module.net.dense_mode=false
    force_field_module.net.layer_scale_init_value=${LAYER_SCALE}
    +force_field_module.net.rope_on_values=${ROPE_ON_VALUES}
    force_field_module.net.rope_sigma=${ROPE_SIGMA}
    force_field_module.net.freq_init=random
    force_field_module.net.learned_freqs=true
    force_field_module.net.attention=true
    force_field_module.net.avg_num_nodes=26.5
    force_field_module.net.attention_backend=${ATTENTION_BACKEND}
    force_field_module.net.qk_norm=${QK_NORM}
    force_field_module.net.swiglu=${SWIGLU}
    force_field_module.net.use_key=${USE_KEY}
    force_field_module.net.norm_type=${NORM_TYPE}
    force_field_module.net.chgspin_mode=${CHGSPIN_MODE}
    force_field_module.net.chgspin_layerwise=${CHGSPIN_LAYERWISE}
    force_field_module.net.chgspin_layerwise_gate=${CHGSPIN_LAYERWISE_GATE}
    force_field_module.net.chgspin_film=${CHGSPIN_FILM}
    force_field_module.net.chgspin_mix_init_std=${CHGSPIN_MIX_INIT_STD}
    force_field_module.net.interaction_radius=${RADIUS}
    force_field_module.net.local_global=${LOCAL_GLOBAL}
    force_field_module.train_augmentation=o3
    force_field_module.flops_coef=72
    force_field_module.compile=true
    "${OPT_OVERRIDES[@]}"
    "${EMA_OVERRIDES[@]}"
    force_field_module.optimizer.lr=${LR}
    force_field_module.optimizer.weight_decay=${WD}
    force_field_module.optimizer.e_weight=${E_WEIGHT}
    force_field_module.optimizer.f_weight=${F_WEIGHT}
    force_field_module.train_rmsd=1.433569
    trainer.max_epochs=20
    trainer.gradient_clip_val=1
    trainer.gradient_clip_algorithm=norm
    trainer.inference_mode=false
    trainer.val_check_interval=${VAL_CHECK_INTERVAL}
    +trainer.limit_val_batches=500
    exp_name=${EXP_NAME}
    model_name=platoformer
    wandb.use_wandb=True
    wandb.wandb_project=scaling-laws-symmetry
    wandb.group=pt2-vs-esen-sm-direct-n${MAX_ATOMS}-ivi
    seed=1
)

if [ -n "${CKPT_PATH:-}" ]; then
    OVERRIDES+=(checkpoint_path="${CKPT_PATH}")
    echo "=== Resuming from checkpoint: ${CKPT_PATH} ==="
fi

echo "=== Hydra parse-check ==="
python train_omol.py "${OVERRIDES[@]}" --cfg job > /dev/null
echo "Hydra config OK"

python train_omol.py "${OVERRIDES[@]}"
exit_code=$?

echo
echo "=== Done: $(date) (exit=$exit_code) ==="
exit $exit_code
