#!/bin/bash
# 1-hour Snellius H100 throughput sweep: Platonic Transformer vs AllScAIP
# variant vs eSEN baseline. Each model runs forward+backward on one synthetic
# N=1000-atom molecule for 10 warmup + 50 timed steps and reports ms/step,
# atoms/sec, ns/day @1fs and peak VRAM.
#
# Usage:
#   sbatch scripts/run_benchmark_throughput.sh
# Override N or step counts via env:
#   N_ATOMS=2000 N_TIMED=100 sbatch scripts/run_benchmark_throughput.sh
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out
#SBATCH --job-name=PR-omol-throughput-bench

set -e
mkdir -p logs

VENV_PATH="${VENV_PATH:-/scratch-shared/ebekkers/scaling-laws-venv-v2}"
N_ATOMS="${N_ATOMS:-1000}"
N_WARMUP="${N_WARMUP:-10}"
N_TIMED="${N_TIMED:-50}"

source "${VENV_PATH}/bin/activate"
module load 2024 2>/dev/null || true
module load CUDA/12.6.0 2>/dev/null || true

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

echo "=== Throughput benchmark ==="
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "JobID:    ${SLURM_JOB_ID}"
echo "N atoms:  ${N_ATOMS} | warmup ${N_WARMUP} | timed ${N_TIMED}"
nvidia-smi -L || true

run_one () {
    local label="$1"; shift
    echo
    echo "##############################################################"
    echo "###  ${label}"
    echo "##############################################################"
    python scripts/benchmark_throughput.py \
        --bench.n_atoms="${N_ATOMS}" \
        --bench.n_warmup="${N_WARMUP}" \
        --bench.n_timed="${N_TIMED}" \
        "$@"
}

# 1) Platonic Transformer — qcczbpfn recipe (attention=true, dense_mode=false,
#    flash backend, no local_global). The reference for OMol25 training cost.
run_one "Platonic Transformer (qcczbpfn recipe)" \
    --config configs/omol.yaml

# 2) AllScAIP variant — same backbone but each logical layer expands to a
#    (local, global) PlatonicBlock pair. ~2x params and FLOPs.
#    Local sub-blocks force scatter; global sub-blocks can stay on flash.
run_one "AllScAIP variant (local_global=true, interaction_radius=2.0)" \
    --config configs/omol.yaml \
    --model.dense_mode=false \
    --model.local_global=true \
    --model.interaction_radius=2.0

# 3) eSEN baseline — eSCNMDBackbone + MLP_EFS_Head, conservative forces
#    (autograd.grad). Much smaller per-step compute but autograd-forces are
#    more expensive than direct forces.
run_one "eSEN baseline" \
    --config configs/omol_esen.yaml

echo
echo "=== JOB FINISHED ($(date)) ==="
