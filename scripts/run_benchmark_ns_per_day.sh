#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out

# Inference throughput benchmark (ns/day, MD timestep dt=1 fs by default).
# Methodology mirrors Qu et al. 2026 (AllScAIP): single GPU, N≈1000 atoms,
# forward only, graph generation per the model's own otf_graph setting.
#
# MODEL env var selects the recipe to benchmark:
#   MODEL=platoformer  — current PT2 recipe (rs2 EMA0.99 small-FFN sin/sin)
#   MODEL=esen         — eSEN small config (the one launched as 22630480)
#   MODEL=esen_paper   — eSEN with paper-equivalent K4L2 hyperparameters:
#                        sphere/hidden=128, lmax=2, num_layers=4, max_neighbors=4.
#                        Matches fairchem's K4L2 backbone (likely the eSEN-sm
#                        6M baseline in Qu et al. Fig 4/5).
#
# Usage:
#   sbatch --job-name=bench-pt2         --export=ALL,MODEL=platoformer  scripts/run_benchmark_ns_per_day.sh
#   sbatch --job-name=bench-esen        --export=ALL,MODEL=esen         scripts/run_benchmark_ns_per_day.sh
#   sbatch --job-name=bench-esen-paper  --export=ALL,MODEL=esen_paper   scripts/run_benchmark_ns_per_day.sh

set -e
mkdir -p logs

MODEL="${MODEL:?must set MODEL env var (platoformer, esen, or esen_paper)}"
MODE="${MODE:-single}"        # single | batched. single = one molecule of N_ATOMS (paper protocol)
N_ATOMS="${N_ATOMS:-1000}"
N_WARMUP="${N_WARMUP:-10}"
N_TIMED="${N_TIMED:-50}"
DT_FS="${DT_FS:-1.0}"
COMPILE="${COMPILE:-false}"   # torch.compile around the net (set true for paper-style "fast" run)
AC="${AC:-true}"              # force_field_module.net.activation_checkpointing (eSEN only; false for fast inference)
CUDNN_BENCH="${CUDNN_BENCH:-false}"  # cudnn.benchmark = true picks faster convolution algos after warmup

source /scratch-shared/ebekkers/scaling-laws-venv-v2/bin/activate
module load 2024
module load CUDA/12.6.0

export DATA_PATH=/scratch-shared/ebekkers/omol25
export PYTHONPATH=/scratch-shared/ebekkers/platonic-omol/training
export HYDRA_FULL_ERROR=1
export TMPDIR=/scratch-shared/ebekkers/tmp
export TORCH_HOME=/scratch-shared/ebekkers/torch_cache
export TRITON_CACHE_DIR=/scratch-shared/ebekkers/triton_cache

cd /scratch-shared/ebekkers/platonic-omol/training

echo "=== Benchmark (MODEL=${MODEL}, N≈${N_ATOMS} atoms) ==="
date; hostname
nvidia-smi --query-gpu=name --format=csv,noheader

COMMON=(
    +precision=fp32_baseline
    data=omol_4m
    data.datamodule.data.num_workers=4
    +data.datamodule.prefetch_factor=2
    data.datamodule.batch_size.train=64
    data.datamodule.batch_size.val=64
    data.datamodule.dynamic_batching=true
    data.datamodule.max_atoms_per_batch=${N_ATOMS}
    data.datamodule.max_atoms_per_batch_val=${N_ATOMS}
    +data.datamodule.max_edges_per_batch=2000000
    +data.datamodule.max_edges_per_batch_val=2000000
    data.datamodule.validation_mode=heldout
    data.datamodule.data.val_data_path=/scratch-shared/ebekkers/omol25/open_mol/val
    trainer.max_epochs=1
    trainer.inference_mode=false
    +bench.mode=${MODE}
    +bench.n_atoms=${N_ATOMS}
    +bench.n_warmup=${N_WARMUP}
    +bench.n_timed=${N_TIMED}
    +bench.dt_fs=${DT_FS}
    +bench.compile=${COMPILE}
    cudnn_benchmark=${CUDNN_BENCH}
    exp_name=bench-${MODEL}
    model_name=${MODEL}
    wandb.use_wandb=False
    seed=1
)

if [ "$MODEL" = "platoformer" ]; then
    # PT2 baseline recipe (matches 77j0ulg4).
    EXTRA=(
        force_field_module=platoformer
        force_field_module.net.hidden_dim=1920
        force_field_module.net.nhead=60
        force_field_module.net.num_layers=8
        force_field_module.net.ffn_dim_factor=2
        force_field_module.net.solid_name=tetrahedron
        force_field_module.net.activation=sin
        +force_field_module.net.readout_activation=sin
        force_field_module.net.dense_mode=false
        force_field_module.net.layer_scale_init_value=1e-4
        +force_field_module.net.rope_on_values=true
        force_field_module.net.rope_sigma=2.0
        force_field_module.net.freq_init=random
        force_field_module.net.learned_freqs=true
        force_field_module.net.attention=true
        force_field_module.net.avg_num_nodes=26.5
        force_field_module.net.attention_backend=flash
        force_field_module.net.chgspin_mode=add
    )
elif [ "$MODEL" = "esen" ]; then
    EXTRA=(force_field_module=esen)
elif [ "$MODEL" = "esen_paper" ]; then
    # eSEN-sm hyperparameters read directly from facebook/OMol25 checkpoint
    # esen_sm_direct_all.pt (6.33M params, 5 blocks, sphere=128, lmax=2).
    # max_neighbors not stored as a backbone knob in the checkpoint config;
    # leaving at the esen.yaml default (30) since it is graph-construction-side.
    # direct_forces=true matches the AllScAIP Fig 5 "filled-circle direct" eSEN
    # baselines (and avoids the autograd.grad double-backward memory blow-up).
    EXTRA=(
        force_field_module=esen
        force_field_module.net.sphere_channels=128
        force_field_module.net.hidden_channels=128
        force_field_module.net.lmax=2
        force_field_module.net.num_layers=5
        force_field_module.net.direct_forces=true
        force_field_module.net.activation_checkpointing=${AC}
    )
else
    echo "ERROR: unknown MODEL=${MODEL} (expected platoformer, esen, or esen_paper)"; exit 2
fi

python benchmark_ns_per_day.py "${COMMON[@]}" "${EXTRA[@]}"
exit_code=$?
echo "=== Done: $(date) (exit=$exit_code) ==="
exit $exit_code
