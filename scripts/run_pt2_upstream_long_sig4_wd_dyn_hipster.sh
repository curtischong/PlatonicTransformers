#!/bin/bash
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --output=/home/ebekker/platonic-omol/logs/%x-%j.out

# Long PT-2 dynamic-batching run on hipster (RTX 6000 Ada, 48GB):
# its7gzf1 recipe + sigma=4.0 + weight_decay sweep. NO EMA.
# - Identical to its7gzf1 (run_pt2_upstream_long_dynamic.sh) except
#   rope_sigma 1.5 -> 4.0 and weight_decay set per arg.
# - attention_backend=scatter (flash-attn 2.7+ doesn't build for sm_89).
# - 48GB VRAM allows the same max_atoms=12000 as the snellius H100 runs.
#
# Usage:
#   sbatch --job-name=PT2-sig4-wd-1e2-hipster   scripts/run_pt2_upstream_long_sig4_wd_dyn_hipster.sh 1e-2
#   sbatch --job-name=PT2-sig4-wd-1e3-hipster   scripts/run_pt2_upstream_long_sig4_wd_dyn_hipster.sh 1e-3
#   sbatch --job-name=PT2-sig4-wd-1e4-hipster   scripts/run_pt2_upstream_long_sig4_wd_dyn_hipster.sh 1e-4
#   sbatch --job-name=PT2-sig4-wd-1e5-hipster   scripts/run_pt2_upstream_long_sig4_wd_dyn_hipster.sh 1e-5
#   sbatch --job-name=PT2-sig4-wd-1e6-hipster   scripts/run_pt2_upstream_long_sig4_wd_dyn_hipster.sh 1e-6

set -e
mkdir -p /home/ebekker/platonic-omol/logs

WD=${1:?"Usage: sbatch ... $0 <weight_decay e.g. 1e-4>"}

source /home/ebekker/platonic-omol/venv/bin/activate

# torch.compile / inductor invokes nvcc to compile fused kernels.
# hipster has 11.4 + 12.3; pin to 12.3 (cu123 is ABI-compatible with cu128 torch).
export CUDA_HOME=/usr/local/cuda-12.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export DATA_PATH=/scratch/ebekker/omol
export PYTHONPATH=/home/ebekker/platonic-omol/training
export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=120
export TORCHDYNAMO_VERBOSE=1

cd /home/ebekker/platonic-omol/training

# Tag for exp_name: "1e-4" -> "1e4"
WD_TAG=$(echo "$WD" | tr -d '.-')
EXP_NAME="pt2-upstream-add-sig4-wd${WD_TAG}-dyn-hipster"

echo "=== Long run (dyn-batching, sigma=4, no EMA, wd=${WD}): ${EXP_NAME} ==="
echo "Date:  $(date)"
echo "Job:   ${SLURM_JOB_ID:-?} on $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Branch + commit:"
git -C /home/ebekker/platonic-omol log --oneline -3

echo "attention_backend=scatter (no flash-attn dep on hipster)"

COMMON_OVERRIDES=(
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
    data.datamodule.data.val_data_path=/scratch/ebekker/omol/open_mol/val
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
    force_field_module.net.attention_backend=scatter
    force_field_module.net.chgspin_mode=add
    force_field_module.train_augmentation=o3
    force_field_module.flops_coef=72
    +force_field_module.optimizer.r=2.0
    force_field_module.optimizer.lr=5e-4
    force_field_module.optimizer.num_warmup_steps=100
    force_field_module.optimizer.weight_decay=${WD}
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
    wandb.group=pt2-upstream-sig4-wd-hipster
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
