#!/bin/bash
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --output=/home/ebekker/platonic-omol/logs/%x-%j.out

# Hipster (RTX 6000 Ada sm_89, 48GB) counterpart of run_pt2_long_sig4_wd1e4_fp32.sh.
# Same recipe, same `+precision=fp32_baseline` preset (TF32 on, no autocast,
# compile off — diagnostic showed compile wasn't a win on the fp32 path).
# Hardware-specific change: attention_backend=scatter (no flash-attn for sm_89).
#
# Companion to run_pt2_long_sig4_wd1e4_bf16_hipster.sh — both submit to the same
# W&B group `pt2-sig4-wd1e4-precision-comparison-hipster` for accuracy + speed
# overlay.
#
# Usage:
#   sbatch --job-name=PT2-sig4-wd1e4-fp32-hipster scripts/run_pt2_long_sig4_wd1e4_fp32_hipster.sh

set -e
mkdir -p /home/ebekker/platonic-omol/logs

# Hipster default soft ulimit -n is 1024 — too low for DataLoader workers + dynamo's
# pipe-based profiler. Without this, both bf16 and fp32 die in the first 30s with
# "Pin memory thread exited unexpectedly" / "pipe() failed". Raise to the hard cap.
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true

source /home/ebekker/platonic-omol/venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export DATA_PATH=/scratch/ebekker/omol
export PYTHONPATH=/home/ebekker/platonic-omol/training
export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=120

cd /home/ebekker/platonic-omol/training

EXP_SUFFIX="${SMOKE_HOURS:+-smoke}"
EXP_NAME="pt2-sig4-wd1e4-fp32-hipster${EXP_SUFFIX}"

echo "=== Long run hipster (sig4, wd1e4, dyn-batch, fp32_baseline preset): ${EXP_NAME} ==="
echo "Date:  $(date)"
echo "Job:   ${SLURM_JOB_ID:-?} on $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Branch + commit:"
git -C /home/ebekker/platonic-omol log --oneline -3

OVERRIDES=(
    +precision=fp32_baseline
    force_field_module=platoformer
    force_field_module.compile=true
    force_field_module.compile_dynamic=true
    data=omol_4m
    data.datamodule.batch_size.train=64
    data.datamodule.batch_size.val=64
    data.datamodule.dynamic_batching=true
    data.datamodule.max_atoms_per_batch=8000
    data.datamodule.max_atoms_per_batch_val=8000
    +data.datamodule.max_edges_per_batch=2000000
    +data.datamodule.max_edges_per_batch_val=2000000
    data.datamodule.validation_mode=heldout
    data.datamodule.data.val_data_path=/scratch/ebekker/omol/open_mol/val
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
    force_field_module.net.attention_backend=scatter
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
    wandb.group=pt2-sig4-wd1e4-precision-comparison-hipster
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
