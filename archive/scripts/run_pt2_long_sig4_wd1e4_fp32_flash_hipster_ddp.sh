#!/bin/bash
#SBATCH --partition=performance
#SBATCH --time=0:45:00
#SBATCH --cpus-per-task=32
#SBATCH --output=/home/ebekker/platonic-omol/logs/%x-%j.out

# Multi-GPU DDP throughput test on hipster. Max_atoms-driven batching (no
# graph-count cap): each rank packs molecules until max_atoms_per_batch=8000
# is hit. Compare wall-clock atoms/sec against the production single-GPU
# fp32+flash baseline (run bcuxsrmo, ~1700 atoms/step at batch_size=64-bound).
#
# Usage (W&B group pt2-fp32-flash-ddp-strong-hipster):
#   DEVICES=2 sbatch \
#     --job-name="PT2-ddp-N2-hipster" \
#     --gres=gpu:rtx_6000_ada:2 \
#     --ntasks=2 \
#     --mem=360G \
#     scripts/run_pt2_long_sig4_wd1e4_fp32_flash_hipster_ddp.sh
#
# Why srun: SLURM_NTASKS=N + SLURM_PROCID per task lets Lightning's
# SLURMEnvironment auto-bind one rank per task. Each task gets --cpus-per-task=32
# (matching the single-GPU baseline).

set -e

DEVICES="${DEVICES:?must set DEVICES env var (1, 2, ...)}"

mkdir -p /home/ebekker/platonic-omol/logs

# Same fd-exhaustion fix as single-GPU scripts. srun --propagate=NOFILE below
# ensures child tasks inherit this limit.
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

EXP_NAME="pt2-fp32-flash-ddp-N${DEVICES}-maxatoms-hipster"

echo "=== Multi-GPU DDP throughput test, max_atoms-driven (per-GPU max_atoms=8000, no graph-count cap, N=${DEVICES}): ${EXP_NAME} ==="
echo "Date:  $(date)"
echo "Job:   ${SLURM_JOB_ID:-?} on $(hostname)"
echo "Node GPUs visible: $(nvidia-smi --query-gpu=index,name --format=csv,noheader | wc -l)"
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "SLURM_NTASKS=${SLURM_NTASKS:-?} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"
echo "Branch + commit:"
git -C /home/ebekker/platonic-omol log --oneline -3

OVERRIDES=(
    +precision=fp32_baseline
    force_field_module=platoformer
    force_field_module.compile=true
    force_field_module.compile_dynamic=true
    data=omol_4m
    # Effectively-unlimited graph-count cap so that max_atoms_per_batch is the
    # binding constraint. With avg 26.5 atoms/molecule, 8000 atoms ≈ 302 graphs;
    # 1000 leaves comfortable headroom and doesn't bind.
    data.datamodule.batch_size.train=1000
    data.datamodule.batch_size.val=1000
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
    +trainer.max_steps=1500
    trainer.devices=${DEVICES}
    trainer.strategy=ddp
    +trainer.num_nodes=1
    # DynamicAtomBatchSamplerForAseDB shards itself via dist.get_rank() /
    # get_world_size() — disable Lightning's auto-injection of DistributedSampler,
    # which would otherwise try (and fail) to wrap our custom batch_sampler.
    +trainer.use_distributed_sampler=false
    trainer.gradient_clip_val=1
    trainer.gradient_clip_algorithm=norm
    trainer.inference_mode=false
    trainer.val_check_interval=5000
    +trainer.limit_val_batches=0
    +trainer.limit_test_batches=0
    exp_name=${EXP_NAME}
    model_name=platoformer
    wandb.use_wandb=True
    wandb.wandb_project=scaling-laws-symmetry
    wandb.group=pt2-fp32-flash-ddp-maxatoms-hipster
    seed=1
)

echo "=== Hydra parse-check ==="
python train_omol.py "${OVERRIDES[@]}" --cfg job > /dev/null
echo "Hydra config OK"

# srun spawns one task per GPU; Lightning's SLURMEnvironment binds each rank
# to its local GPU via LOCAL_RANK. --propagate=NOFILE preserves the raised
# file-descriptor limit across the srun task barrier.
srun --propagate=NOFILE python train_omol.py "${OVERRIDES[@]}"
exit_code=$?

echo
echo "=== Done: $(date) (exit=$exit_code) ==="
exit $exit_code
