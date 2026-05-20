#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out

# Precision experiment: mirror the wd=0 baseline (job 22577151,
# run_pt2_h1920_l8_nols_wd_sweep_fp32.sh with WD=0) but run from a separate
# tree (/scratch-shared/ebekkers/platonic-omol-precision) that contains 5
# fp64 / no-TF32 edits in the encoder + readout path:
#   1. RoPE angles in fp64 (rope.py)
#   2. APE angles in fp64 (ape.py)
#   3. Force explicit fp32-internal RMSNorm (block.py) — no-op for this
#      recipe (norm_type defaults to layernorm), kept for symmetry.
#   4. Energy index_add_ accumulator in fp64 (model.py)
#   5. Readout subnetwork in fp64 with TF32 disabled (platoformer.py).
# Forces and energy are cast back to fp32 at PlatonicForceField boundary, so
# the loss/metric pipeline below the model is unchanged.
#
# Compare against W&B run pt2-h1920-l8-nols-cos20-wd0 (group
# pt2-h1920-l8-nols-cosine-wd-sweep) — same recipe, same seed.
#
# Submit:
#   sbatch --job-name=PT2-h1920-l8-nols-cos20-wd0-prec-fp64 \
#     scripts/run_pt2_h1920_l8_nols_wd0_precision_fp64_snellius.sh

set -e
mkdir -p logs

WD="${WD:-0.0}"
FFN_FACTOR="${FFN_FACTOR:-4}"

source /scratch-shared/ebekkers/scaling-laws-venv-v2/bin/activate
module load 2024
module load CUDA/12.6.0

export DATA_PATH=/scratch-shared/ebekkers/omol25
export PYTHONPATH=/scratch-shared/ebekkers/platonic-omol-precision/training
export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=120
export TMPDIR=/scratch-shared/ebekkers/tmp
export TORCH_HOME=/scratch-shared/ebekkers/torch_cache
export TRITON_CACHE_DIR=/scratch-shared/ebekkers/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/scratch-shared/ebekkers/torch_cache/inductor

cd /scratch-shared/ebekkers/platonic-omol-precision/training

WD_TAG=$(echo "$WD" | sed -e 's/^0\.0$/0/' -e 's/^0$/0/')
FFN_TAG=""
if [ "${FFN_FACTOR}" != "4" ]; then
    FFN_TAG="-ffn${FFN_FACTOR}"
fi
EXP_NAME="pt2-h1920-l8-nols${FFN_TAG}-wd${WD_TAG}-prec-fp64"

echo "=== Precision experiment (h1920/l8/nols, fp32_baseline + fp64 edits, wd=${WD}): ${EXP_NAME} ==="
echo "Date:  $(date)"
echo "Job:   ${SLURM_JOB_ID:-?} on $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Tree:  /scratch-shared/ebekkers/platonic-omol-precision (separate from active wd-sweep)"

python3 -c "from flash_attn import flash_attn_varlen_func; print('flash_attn_varlen_func import OK')"

OVERRIDES=(
    +precision=fp32_baseline
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
    force_field_module.net.hidden_dim=1920
    force_field_module.net.nhead=60
    force_field_module.net.num_layers=8
    force_field_module.net.ffn_dim_factor=${FFN_FACTOR}
    force_field_module.net.solid_name=tetrahedron
    force_field_module.net.dense_mode=false
    force_field_module.net.layer_scale_init_value=null
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
    force_field_module.optimizer.name=adamw
    force_field_module.optimizer.scheduler_name=cosine_annealing_ws
    force_field_module.optimizer.lr=5e-4
    force_field_module.optimizer.num_warmup_steps=0.01
    force_field_module.optimizer.weight_decay=${WD}
    force_field_module.train_rmsd=1.433569
    trainer.max_epochs=20
    trainer.gradient_clip_val=1
    trainer.gradient_clip_algorithm=norm
    trainer.inference_mode=false
    trainer.val_check_interval=5000
    +trainer.limit_val_batches=500
    exp_name=${EXP_NAME}
    model_name=platoformer
    wandb.use_wandb=True
    wandb.wandb_project=scaling-laws-symmetry
    wandb.group=pt2-h1920-l8-nols-precision-experiments
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
