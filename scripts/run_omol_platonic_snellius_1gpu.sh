#!/bin/bash
# Snellius 1× H100 PR run — reproduces the qcczbpfn recipe on the public PR
# code path. The configs/omol.yaml defaults already target 1× H100 (max_atoms
# 12000, max_edges 2.4M, single-rank → effective batch = 12k atoms/step,
# matching qcczbpfn's 4×3000-atom DDP setup). No CLI overrides needed.
#
# This is the successor to the cancelled zdudavnw run; the difference is
# main_omol.py now sets cudnn.deterministic=True, cudnn.benchmark=False, and
# the three torch._dynamo knobs (cache_size_limit=256,
# force_parameter_static_shapes=False, capture_scalar_outputs=True) that
# qcczbpfn applied.
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out
#SBATCH --job-name=PR-omol-1gpu-snellius

set -e
mkdir -p logs

DATA_PATH="${DATA_PATH:-/scratch-shared/ebekkers/omol25/open_mol}"
VENV_PATH="${VENV_PATH:-/scratch-shared/ebekkers/scaling-laws-venv-v2}"

source "${VENV_PATH}/bin/activate"
module load 2024 2>/dev/null || true
module load CUDA/12.6.0 2>/dev/null || true

export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=120

echo "=== snellius 1× H100 run ==="
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "JobID:       ${SLURM_JOB_ID}"
echo "DATA_PATH:   ${DATA_PATH}"
echo "VENV_PATH:   ${VENV_PATH}"
nvidia-smi -L || true

python mains/main_omol.py \
    --config configs/omol.yaml \
    --dataset.data_dir="${DATA_PATH}" \
    --logging.enabled=true

exit_code=$?
echo "=== JOB FINISHED (exit ${exit_code}) ==="
exit ${exit_code}
