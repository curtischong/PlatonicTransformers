#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out

# Long PT-2 run on the upstream-port-pt2 branch.
# - PT-2 recipe: hidden_dim=1728, L=12, nhead=36, sigma=1.5, bs=64,
#   ScheduleFree r=2.0, lr=5e-4, LayerScale=1e-4, train_aug=o3
# - Upstream-ported model (2-layer vector readout, eSEN charge/spin recipe)
# - attention_backend=flash (requires flash-attn built in venv)
# - 100 epochs, 5-day wall (prior PT-2 took ~3-4d at 80 epochs)
# - Validate on the official OMol25 heldout split (2.76M samples) with
#   limit_val_batches=500 during training. Full-val pass at end.
#
# Usage:
#   sbatch --job-name=PT2-upstream-add    scripts/run_pt2_upstream_long.sh add
#   sbatch --job-name=PT2-upstream-concat scripts/run_pt2_upstream_long.sh concat

set -e
mkdir -p logs

CHGSPIN_MODE=${1:?"Usage: sbatch --job-name=... $0 <add|concat>"}
case "$CHGSPIN_MODE" in
  add|concat) ;;
  *) echo "ERROR: chgspin_mode must be 'add' or 'concat', got '$CHGSPIN_MODE'"; exit 1 ;;
esac

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

cd /scratch-shared/ebekkers/platonic-omol/training

EXP_NAME="pt2-upstream-${CHGSPIN_MODE}"

echo "=== Long run: ${EXP_NAME} ==="
echo "Date:  $(date)"
echo "Job:   ${SLURM_JOB_ID:-?} on $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Branch + commit:"
git -C /scratch-shared/ebekkers/platonic-omol log --oneline -3

# Sanity-check flash-attn import before launching a 5-day job.
python3 -c "from flash_attn import flash_attn_varlen_func; print('flash_attn_varlen_func import OK')"

# Hydra parse-check: validate every override resolves, in <1s, before
# we burn an H100 allocation. Catches typos, wrong + prefix, etc. early.
COMMON_OVERRIDES=(
    force_field_module=platoformer
    data=omol_4m
    data.datamodule.batch_size.train=64
    data.datamodule.batch_size.val=64
    data.datamodule.dynamic_batching=false
    data.datamodule.validation_mode=heldout
    data.datamodule.data.val_data_path=/scratch-shared/ebekkers/omol25/open_mol/val
    force_field_module.compile=true
    force_field_module.net.hidden_dim=1728
    force_field_module.net.nhead=36
    force_field_module.net.num_layers=12
    force_field_module.net.solid_name=tetrahedron
    force_field_module.net.dense_mode=false
    force_field_module.net.layer_scale_init_value=1e-4
    +force_field_module.net.rope_on_values=true
    force_field_module.net.rope_sigma=1.5
    force_field_module.net.freq_init=random
    force_field_module.net.learned_freqs=true
    force_field_module.net.attention=true
    force_field_module.net.avg_num_nodes=26.5
    force_field_module.net.attention_backend=flash
    force_field_module.net.chgspin_mode=${CHGSPIN_MODE}
    force_field_module.train_augmentation=o3
    force_field_module.flops_coef=72
    +force_field_module.optimizer.r=2.0
    force_field_module.optimizer.lr=5e-4
    force_field_module.optimizer.num_warmup_steps=100
    # Use eSEN's published OMol-4M normalizer_rmsd (configs/allscaip/omol_4m.yml).
    # Skips compute_stats entirely (which has Lightning logging quirks in on_fit_start).
    force_field_module.train_rmsd=1.433569
    trainer.max_epochs=100
    trainer.gradient_clip_val=1
    trainer.gradient_clip_algorithm=norm
    trainer.inference_mode=false
    trainer.val_check_interval=5000
    +trainer.limit_val_batches=500
    exp_name=${EXP_NAME}
    model_name=platoformer
    wandb.use_wandb=True
    wandb.wandb_project=scaling-laws-symmetry
    wandb.group=pt2-upstream-concat-vs-add
    seed=1
)
echo "=== Hydra parse-check ==="
python train_omol.py "${COMMON_OVERRIDES[@]}" --cfg job > /dev/null
echo "Hydra config OK"

python train_omol.py "${COMMON_OVERRIDES[@]}"
exit_code=$?

echo
echo "=== Done: $(date) (exit=$exit_code) ==="
exit $exit_code
