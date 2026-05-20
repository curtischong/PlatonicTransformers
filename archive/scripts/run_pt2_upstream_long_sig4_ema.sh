#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out

# Long PT-2 run: js1hjhu9 recipe + sigma=4.0 + EMA + tiny weight decay.
# - Identical base recipe to run_pt2_upstream_long.sh (fixed bs=64, add).
# - Changes: rope_sigma 1.5 -> 4.0, weight_decay 0 -> 1e-8, EMA enabled.
# - EMA decay passed as arg. 0.9999 ~ QM9 0.99 in tau/N, 0.99999 ~ QM9 0.999.
#
# Usage:
#   sbatch --job-name=PT2-sig4-ema9999  scripts/run_pt2_upstream_long_sig4_ema.sh 0.9999
#   sbatch --job-name=PT2-sig4-ema99999 scripts/run_pt2_upstream_long_sig4_ema.sh 0.99999

set -e
mkdir -p logs

EMA_DECAY=${1:?"Usage: sbatch --job-name=... $0 <ema_decay e.g. 0.9999>"}

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

EMA_TAG=$(echo "$EMA_DECAY" | tr -d '.')
EXP_NAME="pt2-upstream-add-sig4-ema${EMA_TAG}"

echo "=== Long run (sigma=4, EMA=${EMA_DECAY}, wd=1e-8): ${EXP_NAME} ==="
echo "Date:  $(date)"
echo "Job:   ${SLURM_JOB_ID:-?} on $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Branch + commit:"
git -C /scratch-shared/ebekkers/platonic-omol log --oneline -3

python3 -c "from flash_attn import flash_attn_varlen_func; print('flash_attn_varlen_func import OK')"

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
    force_field_module.net.rope_sigma=4.0
    force_field_module.net.freq_init=random
    force_field_module.net.learned_freqs=true
    force_field_module.net.attention=true
    force_field_module.net.avg_num_nodes=26.5
    force_field_module.net.attention_backend=flash
    force_field_module.net.chgspin_mode=add
    force_field_module.train_augmentation=o3
    force_field_module.flops_coef=72
    +force_field_module.optimizer.r=2.0
    force_field_module.optimizer.lr=5e-4
    force_field_module.optimizer.num_warmup_steps=100
    force_field_module.optimizer.weight_decay=1e-8
    force_field_module.train_rmsd=1.433569
    trainer.max_epochs=100
    trainer.gradient_clip_val=1
    trainer.gradient_clip_algorithm=norm
    trainer.inference_mode=false
    trainer.val_check_interval=5000
    +trainer.limit_val_batches=500
    +ema.decay=${EMA_DECAY}
    +ema.warmup_steps=2000
    exp_name=${EXP_NAME}
    model_name=platoformer
    wandb.use_wandb=True
    wandb.wandb_project=scaling-laws-symmetry
    wandb.group=pt2-upstream-sig4-ema
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
