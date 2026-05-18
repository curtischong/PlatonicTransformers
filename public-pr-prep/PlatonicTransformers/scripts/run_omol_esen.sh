#!/bin/bash
# Reproduce the eSEN baseline on OMol25 (Snellius H100).
# Requires fairchem-core>=2.19. See configs/omol_esen.yaml + README_omol.md.
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out
#SBATCH --job-name=omol_esen

set -e
mkdir -p logs

DATA_PATH="${DATA_PATH:-$HOME/data/omol25}"

source ./venv/bin/activate
module load 2024 2>/dev/null || true
module load CUDA/12.6.0 2>/dev/null || true

# Verify fairchem-core is installed (the eSCNMDBackbone import will fail
# otherwise with a clear error from EquivariantNet).
python -c "from fairchem.core.models.uma.escn_md import eSCNMDBackbone; print('fairchem OK')"

export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=120

python mains/main_omol.py \
    --config configs/omol_esen.yaml \
    --dataset.data_dir="${DATA_PATH}" \
    --logging.enabled=true
