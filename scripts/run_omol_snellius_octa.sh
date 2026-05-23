#!/bin/bash
#Set job requirements
#SBATCH --job-name=omol-octa
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=70G

set -euo pipefail

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


python mains/main_omol.py --config configs/omol.yaml \
 --dataset.data_dir=/projects/prjs2025/omol/omol \
 --training.gradient_clip_val=0.0 \
 --model.avg_num_nodes=26.5 \
 --logging.enabled=true \
 --training.epochs=20 \
 --model.solid_name=octahedron \
 --model.num_layers=16 \
 --model.hidden_dim=3456 \
 --model.ffn_dim_factor=1 \
 --model.num_heads=24 \
 --training.max_atoms_per_batch=12000 \
 --training.max_atoms_per_batch_val=12000 \
 --training.max_edges_per_batch=2400000 \
 --system.gpus=1