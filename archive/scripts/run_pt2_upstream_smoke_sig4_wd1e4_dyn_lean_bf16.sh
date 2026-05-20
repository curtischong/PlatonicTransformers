#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out

# 1-hour smoke: bf16-mixed + dynamo-tuned compile + leaner per-step overhead.
# Builds on run_pt2_upstream_smoke_sig4_wd1e4_dyn_compile_bf16.sh with these
# additional changes (in omol_module.py and via Hydra overrides):
#   - on_after_backward grad_norm logging removed (was causing ~500 host syncs / 100 steps)
#   - per-step torch.isfinite(loss).all() NaN guard removed (1 host sync/step)
#   - _log_stage_metrics now uses self.log_dict (single batched call instead of N)
#   - _accumulate_tokens batches its 3 logs into 1 log_dict + 1 log call
#   - _random_rotation runs on the batch's device with torch.where (no host sync, no H2D)
#   - Hydra: data.datamodule.data.num_workers=16, prefetch_factor=4
#
# Goal: get GPU util above the 26% the previous compile-bf16 smoke saw.
#
# Usage:
#   sbatch --job-name=PT2-sig4-wd1e4-dyn-lean-bf16 scripts/run_pt2_upstream_smoke_sig4_wd1e4_dyn_lean_bf16.sh

set -e
mkdir -p logs

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

# Dynamo tuning knobs:
export PSL_DYNAMO_TUNE=1
export PSL_COMPILE_DYNAMIC=1

# bf16 speed knobs:
export PSL_FAST_BACKENDS=1

cd /scratch-shared/ebekkers/platonic-omol/training

EXP_NAME="pt2-upstream-add-sig4-wd1e4-dyn-lean-bf16"

echo "=== SMOKE (1h, dyn-batching, sigma=4, wd=1e-4, bf16-mixed + dynamo-tuned + lean): ${EXP_NAME} ==="
echo "Date:  $(date)"
echo "Job:   ${SLURM_JOB_ID:-?} on $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Knobs: PSL_DYNAMO_TUNE=${PSL_DYNAMO_TUNE} PSL_COMPILE_DYNAMIC=${PSL_COMPILE_DYNAMIC} PSL_FAST_BACKENDS=${PSL_FAST_BACKENDS}"
echo "       trainer.precision=bf16-mixed, num_workers=16, prefetch_factor=4"
echo "Branch + commit:"
git -C /scratch-shared/ebekkers/platonic-omol log --oneline -3

python3 -c "from flash_attn import flash_attn_varlen_func; print('flash_attn_varlen_func import OK')"

COMMON_OVERRIDES=(
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
    force_field_module.optimizer.weight_decay=1e-4
    force_field_module.train_rmsd=1.433569
    trainer.max_epochs=100
    trainer.gradient_clip_val=1
    trainer.gradient_clip_algorithm=norm
    trainer.inference_mode=false
    trainer.val_check_interval=5000
    +trainer.limit_val_batches=500
    +trainer.precision=bf16-mixed
    exp_name=${EXP_NAME}
    model_name=platoformer
    wandb.use_wandb=True
    wandb.wandb_project=scaling-laws-symmetry
    wandb.group=pt2-upstream-sig4-wd-snellius
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
