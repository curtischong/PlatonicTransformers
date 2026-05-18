#!/bin/bash
#SBATCH --job-name=install-flash-attn-hipster
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:1
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=/home/ebekker/platonic-omol/logs/install-flash-attn-%j.out

# Build flash-attn from source on hipster (RTX 6000 Ada, sm_89).
# Hipster has GLIBC 2.28 (RHEL 8) so the prebuilt wheels (built against
# GLIBC 2.32) won't load. Source build links against system libc.

set -e
mkdir -p /home/ebekker/platonic-omol/logs

# Hipster has cuda-12.3 system-wide and gnu12 (gcc 12.4) as a module.
module load gnu12

export CUDA_HOME=/usr/local/cuda-12.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# RTX 6000 Ada and L4 are both sm_89 (Ada Lovelace).
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=2
export FLASH_ATTENTION_FORCE_BUILD=TRUE

source /home/ebekker/platonic-omol/venv/bin/activate

echo "=== Build env ==="
which nvcc
nvcc --version | tail -2
which g++
g++ --version | head -1
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cxx11abi', torch._C._GLIBCXX_USE_CXX11_ABI)"
echo

echo "=== Compiling flash-attn (~30-60 min) ==="
time pip install --no-build-isolation --force-reinstall --no-deps flash-attn 2>&1

echo
echo "=== Verifying ==="
python -c "from flash_attn import flash_attn_varlen_func; import flash_attn; print('flash_attn', flash_attn.__version__, 'OK')"
echo "DONE"
