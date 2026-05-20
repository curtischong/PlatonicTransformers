#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out

# Weight-decay sweep on the slimmer arch: hidden_dim=1920 (head_dim=32, nhead=60),
# 8 layers, layer_scale=null. AdamW + cosine_annealing_ws (1% warmup, lr→0).
# 20 epochs (~22h wall-clock at ~68 min/epoch).
#
# nhead=60 = 5 * |G|=12, the closest multiple-of-12 to flash-attn's preferred
# pow2 head count (64). hidden_dim=60*32=1920 → head_dim=32, divisible by both
# |G|=12 (=160 per group element) and nhead=60.
#
# Usage (one job per WD value, all in W&B group pt2-h1920-l8-nols-wd-sweep):
#   for WD in 0.0 1e-5 1e-4 1e-3; do
#     sbatch --export=ALL,WD=$WD \
#       --job-name="PT2-h1920-l8-nols-wd${WD}" \
#       scripts/run_pt2_h1920_l8_nols_wd_sweep_fp32.sh
#   done
# Snellius's sbatch defaults to --export=NONE, so the WD env var must be passed
# via --export=ALL,WD=... rather than the usual `WD=$WD sbatch ...` shell prefix.

set -e
mkdir -p logs

WD="${WD:?must set WD env var (e.g. 0.0, 1e-5, 1e-4, 1e-3)}"
# ffn_dim_factor: width multiplier of the FFN block (linear1: hidden→ffn, linear2: ffn→hidden).
# Default 4 matches the production sig4 recipe; set to 2 for a thinner FFN ablation.
FFN_FACTOR="${FFN_FACTOR:-4}"
# layer_scale_init_value: per-block residual gating γ. null = no LayerScale (residuals at
# full strength from init); 1e-4 = production sig4 recipe (residuals near-zero at init,
# gradually grow); 1.0 = standard pre-norm transformer (γ=1 fixed). Hydra parses "null"
# as None and "1e-4"/"1.0" as floats.
LAYER_SCALE="${LAYER_SCALE:-null}"
# Activation function for FFN blocks. Registry: gelu (default), silu, relu, mish, sin.
ACTIVATION="${ACTIVATION:-gelu}"
# Readout activation. "null" → keep legacy nn.GELU readout (preserves prior behavior).
# Set to gelu/silu/relu/mish/sin to apply that activation in the scalar+vector readout
# heads as well. chgspin_mix's F.silu is intentionally left hardcoded (problem-specific).
READOUT_ACTIVATION="${READOUT_ACTIVATION:-null}"
# Number of transformer layers. Default 8 (matches the original wd-sweep recipe);
# set to 16 for a depth ablation (~2× params, ~2× wall per step).
NUM_LAYERS="${NUM_LAYERS:-8}"
# Optimizer: adamw (default, with cosine_annealing_ws + 1% fractional warmup) or free
# (schedulefree.AdamWScheduleFree with absolute warmup_steps=100 and r=2.0; matches the
# production sig4 recipe).
OPTIMIZER="${OPTIMIZER:-adamw}"
# rope_sigma: stddev of RoPE frequency init. Default 4.0 (production sig4 recipe);
# 2.0 makes positional encoding less aggressive (smaller angles, less per-call precision risk).
ROPE_SIGMA="${ROPE_SIGMA:-4.0}"
# EMA decay. "null" = no EMA. Common values: 0.99 (very light), 0.999 (light),
# 0.9999 (moderate, ~half-epoch memory). Heavier than 0.99995 not recommended in 20-epoch runs.
EMA_DECAY="${EMA_DECAY:-null}"
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
# EMA overrides: only injected when EMA_DECAY is set (non-null). train_omol.py reads
# cfg.ema.{decay,warmup_steps} and registers an EMACallback if decay is present.
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

export DATA_PATH=/scratch-shared/ebekkers/omol25
export PYTHONPATH=/scratch-shared/ebekkers/platonic-omol/training
export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=120
export TMPDIR=/scratch-shared/ebekkers/tmp
export TORCH_HOME=/scratch-shared/ebekkers/torch_cache
export TRITON_CACHE_DIR=/scratch-shared/ebekkers/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/scratch-shared/ebekkers/torch_cache/inductor

cd /scratch-shared/ebekkers/platonic-omol/training

# Tag wd values cleanly: 0.0 → wd0, 1e-5 → wd1e-5, etc.
WD_TAG=$(echo "$WD" | sed -e 's/^0\.0$/0/' -e 's/^0$/0/')
# Encode FFN factor in name only when it deviates from the default 4
FFN_TAG=""
if [ "${FFN_FACTOR}" != "4" ]; then
    FFN_TAG="-ffn${FFN_FACTOR}"
fi
# Encode LayerScale: "nols" if null, "ls<value>" otherwise (with 1.0 → ls1)
if [ "${LAYER_SCALE}" = "null" ]; then
    LS_TAG="nols"
else
    LS_TAG="ls$(echo ${LAYER_SCALE} | sed -e 's/^1\.0$/1/' -e 's/^0\.0$/0/')"
fi
# Encode activation only when non-default
ACT_TAG=""
if [ "${ACTIVATION}" != "gelu" ]; then
    ACT_TAG="-act${ACTIVATION}"
fi
# Suffix when readout_activation overrides the legacy GELU readout
RACT_TAG=""
if [ "${READOUT_ACTIVATION}" != "null" ]; then
    RACT_TAG="-ract${READOUT_ACTIVATION}"
fi
# Encode optimizer when non-default (free)
OPT_TAG=""
if [ "${OPTIMIZER}" != "adamw" ]; then
    OPT_TAG="-${OPTIMIZER}"
fi
# Encode rope_sigma when non-default (4.0)
RS_TAG=""
if [ "${ROPE_SIGMA}" != "4.0" ]; then
    RS_TAG="-rs${ROPE_SIGMA}"
fi
# Encode EMA when enabled
EMA_TAG=""
if [ "${EMA_DECAY}" != "null" ]; then
    EMA_TAG="-ema${EMA_DECAY}"
fi
EXP_NAME="pt2-h1920-l${NUM_LAYERS}-${LS_TAG}${FFN_TAG}${ACT_TAG}${RACT_TAG}${OPT_TAG}${RS_TAG}${EMA_TAG}-wd${WD_TAG}"

echo "=== Long run (h1920/l8/nols, fp32_baseline preset, wd=${WD}): ${EXP_NAME} ==="
echo "Date:  $(date)"
echo "Job:   ${SLURM_JOB_ID:-?} on $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Branch + commit:"
git -C /scratch-shared/ebekkers/platonic-omol log --oneline -3

python3 -c "from flash_attn import flash_attn_varlen_func; print('flash_attn_varlen_func import OK')"

OVERRIDES=(
    +precision=fp32_baseline
    force_field_module=platoformer
    data=omol_4m
    data.datamodule.data.num_workers=16
    +data.datamodule.prefetch_factor=4
    data.datamodule.batch_size.train=64
    data.datamodule.batch_size.val=64
    data.datamodule.dynamic_batching=true
    data.datamodule.max_atoms_per_batch=12000
    data.datamodule.max_atoms_per_batch_val=12000
    +data.datamodule.max_edges_per_batch=2000000
    +data.datamodule.max_edges_per_batch_val=2000000
    data.datamodule.validation_mode=heldout
    data.datamodule.data.val_data_path=/scratch-shared/ebekkers/omol25/open_mol/val
    force_field_module.net.hidden_dim=1920
    force_field_module.net.nhead=60
    force_field_module.net.num_layers=${NUM_LAYERS}
    force_field_module.net.ffn_dim_factor=${FFN_FACTOR}
    force_field_module.net.solid_name=tetrahedron
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
    "${OPT_OVERRIDES[@]}"
    "${EMA_OVERRIDES[@]}"
    force_field_module.optimizer.lr=5e-4
    force_field_module.optimizer.weight_decay=${WD}
    force_field_module.train_rmsd=1.433569
    trainer.max_epochs=20
    trainer.gradient_clip_val=1
    trainer.gradient_clip_algorithm=norm
    trainer.inference_mode=false
    trainer.val_check_interval=5000
    +trainer.limit_val_batches=500
    exp_name=${EXP_NAME}
    model_name=platoformer
    wandb.use_wandb=True
    wandb.wandb_project=scaling-laws-symmetry
    wandb.group=pt2-h1920-l8-nols-cosine-wd-sweep
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
