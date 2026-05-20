#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out

# PT-2 20-epoch launcher matching the production "winning small recipe":
#   h1920/l8, ffn_dim_factor=2, layer_scale=1e-4, activation=sin (FFN + readout),
#   rope_sigma=2.0, EMA=0.99, wd=1e-5, fp32 baseline, AdamW + cosine_annealing_ws.
# Parametric MAX_ATOMS so we can match eSEN-sm-direct's larger feasible batch
# (validated to 20000 by OOM probe 22644769).
#
# Canonical submit for the "match-eSEN-sm-direct at batch=20000" comparison:
#   sbatch --export=ALL,MAX_ATOMS=20000,WD=1e-5,LAYER_SCALE=1e-4,FFN_FACTOR=2,\
#ACTIVATION=sin,READOUT_ACTIVATION=sin,ROPE_SIGMA=2.0,EMA_DECAY=0.99 \
#     --job-name=PT2-rs2-ema0.99-20ep-n20000 \
#     scripts/run_pt2_h1920_l8_20ep_fp32.sh

set -e
mkdir -p logs

MAX_ATOMS="${MAX_ATOMS:-12000}"
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
# LLaMA-style transformer hardening:
#   QK_NORM: RMSNorm on Q and K (head_dim) before RoPE
#   SWIGLU:  gated FFN — silu(W_gate(x)) * W_up(x) → linear2
QK_NORM="${QK_NORM:-false}"
SWIGLU="${SWIGLU:-false}"
USE_KEY="${USE_KEY:-false}"
# NORM_TYPE: "layernorm" (default) or "rmsnorm" — LLaMA-style RMSNorm in place
# of LayerNorm for the two per-block normalization layers (pre-attention,
# pre-FFN). Independent of qk_norm.
NORM_TYPE="${NORM_TYPE:-layernorm}"
# QK_DIM_FACTOR: expand Q/K head_dim by this integer factor (V unchanged unless
# V_DIM_FACTOR is also set). Only supported with ATTENTION_BACKEND=flash.
QK_DIM_FACTOR="${QK_DIM_FACTOR:-1}"
# V_DIM_FACTOR: V head_dim factor (and output projection input dim). Must be
# <= QK_DIM_FACTOR. Default 1 = V unchanged (V padded with zeros to match QK
# for flash). Set equal to QK_DIM_FACTOR for full expansion without padding.
V_DIM_FACTOR="${V_DIM_FACTOR:-1}"
# ROPE_V_INDEPENDENT: when true, V's RoPE uses its own independent learnable
# frequency bank (decoupled from Q/K's). Doubles spatial-direction coverage at
# matching head_dim. Only meaningful with rope_on_values=true (default).
ROPE_V_INDEPENDENT="${ROPE_V_INDEPENDENT:-false}"
# COMPILE_MODE: torch.compile mode. "default" (current production), "reduce-overhead"
# (reduces CPU dispatcher cost — modest gain), or "max-autotune" (aggressive fusion
# search, slower first step but ~10-20% steady-state speedup).
COMPILE_MODE="${COMPILE_MODE:-default}"
# PRECISION: Hydra preset name under configs/precision/ — "fp32_baseline" (default,
# shipped production) or "bf16_h100" (bf16-mixed + TF32 + compile, H100-only).
PRECISION="${PRECISION:-fp32_baseline}"
# Radius graph mode: when set (recommended starting value: 2.0 Å), the model
# uses sparse radius-based attention with a polynomial cutoff window.
RADIUS="${RADIUS:-null}"
# Dual-stage local→global blocks (AllScAIP-style). When true, doubles the
# number of PlatonicBlocks per logical layer. Requires RADIUS != null.
LOCAL_GLOBAL="${LOCAL_GLOBAL:-false}"
# Single-stream radius mode (RADIUS set, LOCAL_GLOBAL false) is fully sparse and
# must use the scatter backend. In local_global mode the global sub-blocks run
# full attention, so flash is kept — local sub-blocks force scatter internally.
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
# Platonic-solid symmetry group. "tetrahedron" (|G|=12) is the default equivariant
# recipe; "trivial_3" disables equivariance (|G|=1, plain Transformer in 3D).
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

source /scratch-shared/ebekkers/scaling-laws-venv-v2/bin/activate
module load 2024
module load CUDA/12.6.0

export DATA_PATH=$HOME/data/omol25
export PYTHONPATH=$HOME/projects/platonic-omol/training
export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=120
export TMPDIR=$HOME/tmp
export TORCH_HOME=$HOME/torch_cache
export TRITON_CACHE_DIR=$HOME/triton_cache
export TORCHINDUCTOR_CACHE_DIR=$HOME/torch_cache/inductor

cd $HOME/projects/platonic-omol/training

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
# Encode solid_name when non-default (tetrahedron is implicit; everything else
# gets a tag, e.g. trivial_3 → -solidtrivial_3).
SOLID_TAG=""
if [ "${SOLID_NAME}" != "tetrahedron" ]; then
    SOLID_TAG="-solid${SOLID_NAME}"
fi
QKN_TAG=""
if [ "${QK_NORM}" = "true" ]; then QKN_TAG="-qkn"; fi
SG_TAG=""
if [ "${SWIGLU}" = "true" ]; then SG_TAG="-sg"; fi
UK_TAG=""
if [ "${USE_KEY}" = "true" ]; then UK_TAG="-uk"; fi
NORMTYPE_TAG=""
if [ "${NORM_TYPE}" != "layernorm" ]; then NORMTYPE_TAG="-${NORM_TYPE}"; fi
NHEAD_TAG=""
if [ "${NHEAD}" != "60" ]; then NHEAD_TAG="-nh${NHEAD}"; fi
QKDF_TAG=""
if [ "${QK_DIM_FACTOR}" != "1" ]; then QKDF_TAG="-qkdf${QK_DIM_FACTOR}"; fi
VDF_TAG=""
if [ "${V_DIM_FACTOR}" != "1" ]; then VDF_TAG="-vdf${V_DIM_FACTOR}"; fi
RVI_TAG=""
if [ "${ROPE_V_INDEPENDENT}" = "true" ]; then RVI_TAG="-rvi"; fi
CM_TAG=""
if [ "${COMPILE_MODE}" != "default" ]; then CM_TAG="-cm${COMPILE_MODE}"; fi
EXP_NAME="pt2-h${HIDDEN_DIM}-l${NUM_LAYERS}-${LS_TAG}${FFN_TAG}${ACT_TAG}${RACT_TAG}${OPT_TAG}${RS_TAG}${ROV_TAG}${BACKEND_TAG}${CSM_TAG}${CSL_TAG}${CSF_TAG}${CSI_TAG}${EW_TAG}${FW_TAG}${R_TAG}${LG_TAG}${EMA_TAG}${LR_TAG}${BS_TAG}${QKN_TAG}${SG_TAG}${UK_TAG}${NORMTYPE_TAG}${NHEAD_TAG}${QKDF_TAG}${VDF_TAG}${RVI_TAG}${CM_TAG}${SOLID_TAG}-wd${WD_TAG}-20ep-n${MAX_ATOMS}"

echo "=== PT-2 20ep (max_atoms=${MAX_ATOMS}): ${EXP_NAME} ==="
echo "Date:  $(date)"
echo "Job:   ${SLURM_JOB_ID:-?} on $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Branch + commit:"
git -C $HOME/projects/platonic-omol log --oneline -3

python3 -c "from flash_attn import flash_attn_varlen_func; print('flash_attn_varlen_func import OK')"

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
    force_field_module.net.qk_dim_factor=${QK_DIM_FACTOR}
    force_field_module.net.v_dim_factor=${V_DIM_FACTOR}
    force_field_module.net.rope_v_independent=${ROPE_V_INDEPENDENT}
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
    # +precision=fp32_baseline sets force_field_module.compile=false; override here
    # so the omol_module compile hook fires on stage=="fit".
    force_field_module.compile=true
    force_field_module.compile_mode=${COMPILE_MODE}
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
    wandb.group=pt2-vs-esen-sm-direct-n${MAX_ATOMS}
    seed=1
)

# Optional: resume from checkpoint via CKPT_PATH env var
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
