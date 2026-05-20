#!/bin/bash
#SBATCH --job-name=smoke-pt2-upstream-flash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=logs/smoke-pt2-upstream-flash-%j.out

# Smoke test (flash backend variant) for the upstream-port-pt2 branch:
# - Same as smoke_pt2_upstream.sh but attention_backend=flash
# - Submit with --dependency=afterok:<install_flash_attn_jobid> so it
#   only runs after flash-attn is successfully built.

set -e
mkdir -p logs

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

echo "=== Smoke test (flash backend): upstream-port-pt2 ==="
echo "Date: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader
git -C /scratch-shared/ebekkers/platonic-omol log --oneline -3

# Verify flash-attn import works before launching
python3 -c "from flash_attn import flash_attn_varlen_func; print('flash_attn_varlen_func import OK')"

run_smoke() {
    local mode=$1
    echo
    echo "=========================="
    echo "  flash + chgspin_mode=$mode"
    echo "=========================="
    python train_omol.py \
        force_field_module=platoformer \
        data=omol_4m \
        data.datamodule.batch_size.train=64 \
        data.datamodule.batch_size.val=64 \
        data.datamodule.dynamic_batching=false \
        data.datamodule.data.val_data_path=/scratch-shared/ebekkers/omol25/open_mol/neutral_val \
        force_field_module.compile=true \
        force_field_module.net.hidden_dim=1728 \
        force_field_module.net.nhead=36 \
        force_field_module.net.num_layers=12 \
        force_field_module.net.solid_name=tetrahedron \
        force_field_module.net.dense_mode=false \
        force_field_module.net.layer_scale_init_value=1e-4 \
        +force_field_module.net.rope_on_values=true \
        force_field_module.net.rope_sigma=1.5 \
        force_field_module.net.freq_init=random \
        force_field_module.net.learned_freqs=true \
        force_field_module.net.attention=true \
        force_field_module.net.avg_num_nodes=26.5 \
        force_field_module.net.attention_backend=flash \
        force_field_module.net.chgspin_mode=$mode \
        force_field_module.train_augmentation=o3 \
        force_field_module.flops_coef=72 \
        +force_field_module.optimizer.r=2.0 \
        force_field_module.optimizer.lr=5e-4 \
        force_field_module.optimizer.num_warmup_steps=100 \
        trainer.max_epochs=1 \
        +trainer.max_steps=200 \
        trainer.gradient_clip_val=1 \
        trainer.gradient_clip_algorithm=norm \
        trainer.inference_mode=false \
        trainer.val_check_interval=200 \
        +trainer.limit_val_batches=10 \
        +trainer.limit_test_batches=0 \
        exp_name=smoke-pt2-flash-$mode \
        model_name=platoformer \
        wandb.use_wandb=True \
        wandb.wandb_project=scaling-laws-symmetry \
        wandb.group=upstream-port-pt2-smoke \
        seed=1
}

run_smoke add
run_smoke concat

echo
echo "=== Flash smoke test complete: $(date) ==="
