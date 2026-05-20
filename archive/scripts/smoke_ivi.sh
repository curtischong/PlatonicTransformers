#!/bin/bash
#SBATCH --job-name=smoke-ivi
#SBATCH --partition=geodude
#SBATCH --account=geodudeusers
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/home/ebekker/platonic-omol/logs/smoke-ivi-%j.out

# End-to-end smoke test for ivi-cluster. Run after flash-attn finishes:
#   sbatch --dependency=afterok:<flash-attn-jobid> scripts/smoke_ivi.sh
# Tiny PT-2 model, 5 train batches, 2 val batches, full flash-attn backend.

set -e
mkdir -p /home/ebekker/platonic-omol/logs

source /home/ebekker/platonic-omol/venv/bin/activate

export DATA_PATH=/home/ebekker/data/omol
export PYTHONPATH=/home/ebekker/platonic-omol/training
export HYDRA_FULL_ERROR=1

cd /home/ebekker/platonic-omol/training

echo "=== Smoke test on ivi-cluster ==="
echo "Date:  $(date)"
echo "Job:   ${SLURM_JOB_ID:-?} on $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
git -C /home/ebekker/platonic-omol log --oneline -3

echo
echo "=== Import sanity ==="
python -c "import torch, lightning, fairchem; from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cxx11abi', torch._C._GLIBCXX_USE_CXX11_ABI); print('lightning', lightning.__version__); print('atomicdata_list_to_batch OK')"
echo

echo "=== Train smoke (5 train batches, 2 val batches) ==="
python train_omol.py \
    force_field_module=platoformer \
    data=omol_4m \
    data.datamodule.batch_size.train=4 \
    data.datamodule.batch_size.val=4 \
    data.datamodule.dynamic_batching=true \
    data.datamodule.max_atoms_per_batch=2000 \
    data.datamodule.max_atoms_per_batch_val=2000 \
    data.datamodule.validation_mode=heldout \
    data.datamodule.data.val_data_path=/home/ebekker/data/omol/open_mol/val \
    data.datamodule.data.num_workers=4 \
    force_field_module.compile=false \
    force_field_module.net.hidden_dim=144 \
    force_field_module.net.nhead=12 \
    force_field_module.net.num_layers=2 \
    force_field_module.net.solid_name=tetrahedron \
    force_field_module.net.dense_mode=false \
    force_field_module.net.layer_scale_init_value=1e-4 \
    +force_field_module.net.rope_on_values=true \
    force_field_module.net.rope_sigma=1.5 \
    force_field_module.net.freq_init=random \
    force_field_module.net.learned_freqs=true \
    force_field_module.net.attention=true \
    force_field_module.net.avg_num_nodes=26.5 \
    force_field_module.net.attention_backend=scatter \
    force_field_module.net.chgspin_mode=add \
    force_field_module.train_augmentation=o3 \
    force_field_module.flops_coef=72 \
    +force_field_module.optimizer.r=2.0 \
    force_field_module.optimizer.lr=5e-4 \
    force_field_module.optimizer.num_warmup_steps=10 \
    force_field_module.train_rmsd=1.433569 \
    +trainer.limit_train_batches=5 \
    +trainer.limit_val_batches=2 \
    trainer.val_check_interval=5 \
    trainer.max_epochs=1 \
    wandb.use_wandb=false \
    exp_name=smoke-ivi
exit_code=$?

echo
echo "=== Done: $(date) (exit=$exit_code) ==="
exit $exit_code
