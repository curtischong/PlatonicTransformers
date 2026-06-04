#!/bin/bash
#Set job requirements
#SBATCH --job-name=omol-snellius
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --time=0-06:00:00
#SBATCH --mem=250G

set -euo pipefail

# Change only the #SBATCH --gpus number above. Batch caps scale to keep the global batch fixed:
# 4 GPUs -> 3000 atoms / 600000 edges per GPU
# 2 GPUs -> 6000 atoms / 1200000 edges per GPU
# 1 GPU  -> 12000 atoms / 2400000 edges per GPU
GPUS="${SLURM_GPUS_ON_NODE:-4}"

TOTAL_ATOMS_PER_STEP=12000
TOTAL_EDGES_PER_STEP=2400000
MAX_ATOMS_PER_BATCH=$((TOTAL_ATOMS_PER_STEP / GPUS))
MAX_EDGES_PER_BATCH=$((TOTAL_EDGES_PER_STEP / GPUS))

cd /projects/prjs2025/platonic-omol/
mkdir -p logs

# Path to OMOL data
export DATA_PATH=/projects/prjs2025/omol/omol

source /home/mislam1/venvs/platonic-scaling-laws/bin/activate


python - <<'PY'
import os
import torch
print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
PY

echo "GPUS=${GPUS}"
echo "MAX_ATOMS_PER_BATCH=${MAX_ATOMS_PER_BATCH}"
echo "MAX_EDGES_PER_BATCH=${MAX_EDGES_PER_BATCH}"

python mains/main_omol.py --config configs/omol.yaml \
 --dataset.data_dir=/projects/prjs2025/omol/omol \
 --training.gradient_clip_val=0.0 \
 --model.avg_num_nodes=26.5 \
 --logging.enabled=true \
 --training.epochs=80 \
 --training.max_atoms_per_batch="${MAX_ATOMS_PER_BATCH}" \
 --training.max_atoms_per_batch_val="${MAX_ATOMS_PER_BATCH}" \
 --training.max_edges_per_batch="${MAX_EDGES_PER_BATCH}" \
 --training.max_edges_per_batch_val="${MAX_EDGES_PER_BATCH}" \
 --system.gpus="${GPUS}"
