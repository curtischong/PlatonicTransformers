#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:2
#SBATCH --time=5-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --output=logs/%x-%j.out

# Snellius PT-2 dynamic-batching run: its7gzf1 + sigma=4.0 + wd=1e-4 + 2 GPUs + 2x batch.
# Same recipe as run_pt2_upstream_long_sig4_wd1e4_dyn.sh except:
#   --gres=gpu:h100:2 (was 1)
#   max_atoms_per_batch=24000 (was 12000)  — doubles per-GPU batch
#   max_edges_per_batch=4000000 (was 2M)   — scales proportionally
# Effective batch per step = 4× the 1-GPU run (2 GPUs × 2× per-GPU batch).
# Weight decay = 1e-4 matches the 1-GPU sibling for a direct comparison.
#
# DDP pattern lifted from run_v11_train_tmpfs_multigpu.sh:
#   single Python process; Lightning's DDPStrategy forks N workers.
#   SLURM_JOB_NAME=bash forces Lightning to use LightningEnvironment instead
#   of SLURMEnvironment (which would require ntasks_per_node==devices).
#
# Usage:
#   sbatch --job-name=PT2-sig4-wd1e4-dyn-2gpu  scripts/run_pt2_upstream_long_sig4_wd1e4_dyn_2gpu.sh

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch-shared/ebekkers/platonic-omol/training

NGPUS=${SLURM_GPUS_ON_NODE:-2}
# Force LightningEnvironment instead of SLURMEnvironment (otherwise Lightning
# enforces ntasks_per_node==devices, which collides with --gpus-per-task).
export SLURM_JOB_NAME=bash

EXP_NAME="pt2-upstream-add-sig4-wd1e4-dyn-2gpu"

echo "=== Long run (2 GPU DDP, dyn-batching, sigma=4, wd=1e-4): ${EXP_NAME} ==="
echo "Date:  $(date)"
echo "Job:   ${SLURM_JOB_ID:-?} on $(hostname)"
echo "NGPUS: $NGPUS"
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "Branch + commit:"
git -C /scratch-shared/ebekkers/platonic-omol log --oneline -3

python3 -c "from flash_attn import flash_attn_varlen_func; print('flash_attn_varlen_func import OK')"

COMMON_OVERRIDES=(
    force_field_module=platoformer
    data=omol_4m
    data.datamodule.batch_size.train=64
    data.datamodule.batch_size.val=64
    data.datamodule.dynamic_batching=true
    data.datamodule.max_atoms_per_batch=24000
    data.datamodule.max_atoms_per_batch_val=24000
    +data.datamodule.max_edges_per_batch=4000000
    +data.datamodule.max_edges_per_batch_val=4000000
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
    trainer.devices=${NGPUS}
    trainer.strategy=ddp
    +trainer.use_distributed_sampler=false
    trainer.max_epochs=100
    trainer.gradient_clip_val=1
    trainer.gradient_clip_algorithm=norm
    trainer.inference_mode=false
    trainer.val_check_interval=5000
    trainer.check_val_every_n_epoch=null
    +trainer.limit_val_batches=500
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

# Single Python process; Lightning's DDPStrategy forks N workers. No srun.
python train_omol.py "${COMMON_OVERRIDES[@]}"
exit_code=$?

echo
echo "=== Done: $(date) (exit=$exit_code) ==="
exit $exit_code
