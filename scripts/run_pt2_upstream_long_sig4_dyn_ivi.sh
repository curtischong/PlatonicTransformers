#!/bin/bash
#SBATCH --job-name=PT2-sig4-dyn-ivi
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --output=/home/ebekker/platonic-omol/logs/%x-%j.out

# Long PT-2 dynamic-batching run on ivi-cluster: its7gzf1 recipe + sigma=4.0.
# - Identical to its7gzf1 (run_pt2_upstream_long_dynamic.sh) except rope_sigma 1.5 -> 4.0.
# - NO EMA, NO weight decay (matches the original its7gzf1 baseline).
#
# Partition + GPU type are NOT pinned here — pass them via sbatch CLI so the
# same script targets all6000 (rtx_6000) or geodude (rtx_a5000):
#   sbatch -p all6000 --account=all6000users --gres=gpu:rtx_6000:1 \
#          --job-name=PT2-sig4-dyn-ivi-all6000 \
#          scripts/run_pt2_upstream_long_sig4_dyn_ivi.sh
#   sbatch -p geodude --account=geodudeusers --gres=gpu:rtx_a5000:1 \
#          --job-name=PT2-sig4-dyn-ivi-geodude \
#          scripts/run_pt2_upstream_long_sig4_dyn_ivi.sh

set -e
mkdir -p /home/ebekker/platonic-omol/logs

source /home/ebekker/platonic-omol/venv/bin/activate

# torch.compile / inductor invokes nvcc to compile fused kernels; the default
# /etc/alternatives/cuda symlink resolves to cuda-13.1 which torch (cu128)
# doesn't link cleanly against. Pin to 12.9 (matches the build env).
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export DATA_PATH=/home/ebekker/data/omol
export PYTHONPATH=/home/ebekker/platonic-omol/training
export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=120
export TORCHDYNAMO_VERBOSE=1

cd /home/ebekker/platonic-omol/training

EXP_NAME="pt2-upstream-add-sig4-dyn-ivi-${SLURM_JOB_PARTITION}"

echo "=== Long run (dyn-batching, sigma=4, no EMA, wd=0): ${EXP_NAME} ==="
echo "Date:  $(date)"
echo "Job:   ${SLURM_JOB_ID:-?} on $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Branch + commit:"
git -C /home/ebekker/platonic-omol log --oneline -3

echo "attention_backend=scatter (no flash-attn dep on ivi)"

COMMON_OVERRIDES=(
    force_field_module=platoformer
    data=omol_4m
    data.datamodule.batch_size.train=64
    data.datamodule.batch_size.val=64
    data.datamodule.dynamic_batching=true
    data.datamodule.max_atoms_per_batch=4000
    data.datamodule.max_atoms_per_batch_val=4000
    +data.datamodule.max_edges_per_batch=500000
    +data.datamodule.max_edges_per_batch_val=500000
    data.datamodule.validation_mode=heldout
    data.datamodule.data.val_data_path=/home/ebekker/data/omol/open_mol/val
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
    wandb.group=pt2-upstream-sig4-ivi
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
