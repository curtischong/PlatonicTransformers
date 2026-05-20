#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out

# Production long run: PlatoFormer sig=4, wd=1e-4, dyn-batching, bf16 path on H100.
# All precision/backend knobs flow from `+precision=bf16_h100` (configs/precision/bf16_h100.yaml):
#   trainer.precision=bf16-mixed, matmul_precision=high (TF32),
#   cudnn.benchmark=true, force_field_module.compile=true with dynamic=true,
#   plus auto-applied dynamo tunings (cache_size_limit=256, etc.).
# omol_module.py was leaned out for low per-step Python overhead (no per-param
# grad_norm logging, no per-step .all() NaN guard, log_dict batching, GPU-side
# random rotation). Use SMOKE_HOURS=1 to run a 1h variant for sanity-checking
# changes; default is the full 5d production run.
#
# Usage:
#   sbatch --job-name=PT2-sig4-wd1e4-bf16 scripts/run_pt2_long_sig4_wd1e4_bf16.sh
#   SMOKE_HOURS=1 sbatch --time=01:00:00 --job-name=PT2-sig4-wd1e4-bf16-smoke \
#       scripts/run_pt2_long_sig4_wd1e4_bf16.sh

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

cd /scratch-shared/ebekkers/platonic-omol/training

EXP_SUFFIX="${SMOKE_HOURS:+-smoke}"
EXP_NAME="pt2-sig4-wd1e4-bf16${EXP_SUFFIX}"

echo "=== Long run (sig4, wd1e4, dyn-batch, bf16_h100 preset): ${EXP_NAME} ==="
echo "Date:  $(date)"
echo "Job:   ${SLURM_JOB_ID:-?} on $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Branch + commit:"
git -C /scratch-shared/ebekkers/platonic-omol log --oneline -3

python3 -c "from flash_attn import flash_attn_varlen_func; print('flash_attn_varlen_func import OK')"

OVERRIDES=(
    +precision=bf16_h100
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
    exp_name=${EXP_NAME}
    model_name=platoformer
    wandb.use_wandb=True
    wandb.wandb_project=scaling-laws-symmetry
    wandb.group=pt2-sig4-wd1e4-precision-comparison
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
