#!/bin/bash
#SBATCH --job-name=install-flash-attn
#SBATCH --partition=staging
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/install-flash-attn-%j.out

# Build flash-attn from source against the existing scaling-laws venv.
# No GPU needed for compile; nvcc handles cross-arch.
# Run on staging partition (CPU-only, fast queue) so we don't burn GPU SBU.

set -e

mkdir -p logs

module load 2024
module load CUDA/12.6.0

export CUDA_HOME=$EBROOTCUDA
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Hopper (H100 = sm_90) is what we run on.
export TORCH_CUDA_ARCH_LIST="9.0"
# Limit parallel nvcc jobs to avoid OOM during compile.
export MAX_JOBS=2  # 4 OOM'd at 32G; reduce parallelism, increase mem
export FLASH_ATTENTION_FORCE_BUILD=TRUE

source /scratch-shared/ebekkers/scaling-laws-venv/bin/activate
export UV_CACHE_DIR=/scratch-shared/ebekkers/.uv-cache

echo "=== Build env ==="
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version | tail -2
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo

echo "=== Installing flash-attn (this will compile, ~30-60 min) ==="
time uv pip install flash-attn --no-build-isolation 2>&1

echo
echo "=== Verifying ==="
python3 -c "import flash_attn; print('flash_attn version:', flash_attn.__version__)"
python3 -c "from flash_attn import flash_attn_varlen_func; print('flash_attn_varlen_func imported OK')"
echo "DONE"
