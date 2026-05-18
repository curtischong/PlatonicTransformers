#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out

# Full-validation test of the cancelled `pt2-upstream-add-sig4-wd1e4-dyn` run
# (job 22430304, last checkpoint epoch=29, May 7 10:51).
#
# Mirrors run_pt2_upstream_long_sig4_wd1e4_dyn.sh's recipe so the model is
# instantiated with the same architecture as the saved checkpoint, but:
#   - calls test_omol.py (test-only; no trainer.fit())
#   - drops `+trainer.limit_val_batches=500` -> evaluates the full 2.76M
#     heldout val set (open_mol/val) instead of just the first 32K samples
#   - 2h walltime (single full pass, ~20-30 min on H100)
#
# Usage:
#   sbatch --job-name=PT2-sig4-wd1e4-test scripts/run_test_sig4_wd1e4_dyn_snellius.sh

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

EXP_NAME="pt2-upstream-add-sig4-wd1e4-dyn-fullval"
CKPT="/scratch-shared/ebekkers/platonic-omol/training/checkpoints/pt2-upstream-add-sig4-wd1e4-dyn/platoformer/run_22430304_params_33.7 million/last.ckpt"

echo "=== Full-val test: ${EXP_NAME} ==="
echo "Date:  $(date)"
echo "Job:   ${SLURM_JOB_ID:-?} on $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Ckpt:  ${CKPT}"
echo "Branch + commit:"
git -C /scratch-shared/ebekkers/platonic-omol log --oneline -3 || true

python3 -c "from flash_attn import flash_attn_varlen_func; print('flash_attn_varlen_func import OK')"

OVERRIDES=(
    "+checkpoint_path=${CKPT}"
    force_field_module=platoformer
    data=omol_4m
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
    trainer.gradient_clip_val=1
    trainer.gradient_clip_algorithm=norm
    trainer.inference_mode=false
    exp_name=${EXP_NAME}
    model_name=platoformer
    wandb.use_wandb=True
    wandb.wandb_project=scaling-laws-symmetry
    wandb.group=pt2-upstream-sig4-wd-snellius
    seed=1
)

echo "=== Hydra parse-check ==="
python test_omol.py "${OVERRIDES[@]}" --cfg job > /dev/null
echo "Hydra config OK"

python test_omol.py "${OVERRIDES[@]}"
exit_code=$?

echo
echo "=== Done: $(date) (exit=$exit_code) ==="
exit $exit_code
