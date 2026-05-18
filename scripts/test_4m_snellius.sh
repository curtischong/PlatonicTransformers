#!/bin/bash
# Smoke test: PlatoFormer on OMol25 4M split with charge/spin additive injection
# Usage: sbatch --job-name=test-4m --account=gusei11738 --partition=gpu_h100 \
#        --gres=gpu:h100:1 --cpus-per-task=16 --mem=180G --time=01:00:00 \
#        --output=slurm-test-4m-%j.out scripts/test_4m_snellius.sh

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

echo "Run: test-4m-smoke"
echo "Date: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

python train_omol.py \
    force_field_module=platoformer \
    data=omol_4m \
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
    force_field_module.train_augmentation=o3 \
    force_field_module.flops_coef=72 \
    +force_field_module.optimizer.r=2.0 \
    force_field_module.optimizer.num_warmup_steps=100 \
    force_field_module.net.charge_emb_dim=64 \
    force_field_module.net.spin_emb_dim=64 \
    trainer.gradient_clip_val=1 \
    trainer.gradient_clip_algorithm=norm \
    trainer.inference_mode=false \
    trainer.val_check_interval=100 \
    trainer.fast_dev_run=50 \
    exp_name=test-4m-smoke \
    model_name=platoformer \
    wandb.use_wandb=False \
    seed=1

echo "Done: $(date)"
