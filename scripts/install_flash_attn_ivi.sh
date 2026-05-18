#!/bin/bash
#SBATCH --job-name=install-flash-attn-ivi
#SBATCH --partition=geodude
#SBATCH --account=geodudeusers
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/home/ebekker/platonic-omol/logs/install-flash-attn-%j.out

# Build flash-attn from source on ivi-cluster — pinned to 2.7.4.post1.
# Why pin to 2.7.4: flash-attn 2.8.x dropped Turing (sm_75) AND its setup.py
# ignores TORCH_CUDA_ARCH_LIST, hardcoding sm_80/90/100/120 with no PTX
# (so the wheel won't even JIT to sm_86 / RTX A5000). 2.7.4.post1 still
# supports sm_75 and respects TORCH_CUDA_ARCH_LIST.
#
# RHEL 8 / GLIBC 2.28 (same as hipster) so the prebuilt wheels (built
# against newer GLIBC) don't load. Source build links against system libc.
# Targets sm_75 (Turing: rtx_6000, titan_rtx) + sm_86 (Ampere: rtx_a5000,
# rtx_3090, rtx_a6000) — covers every modern ivi GPU we have access to.

set -e
mkdir -p /home/ebekker/platonic-omol/logs

module load gnu12

# torch on this venv is +cu128 (12.8); ivi has 12.9 + 13.x (no 12.3 nvcc on this node).
# 12.9 is ABI-compatible with cu12 torch builds.
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Cover both Turing and Ampere ivi GPUs in one binary.
export TORCH_CUDA_ARCH_LIST="7.5;8.6"
export MAX_JOBS=2
export FLASH_ATTENTION_FORCE_BUILD=TRUE

source /home/ebekker/platonic-omol/venv/bin/activate

echo "=== Build env ==="
which nvcc && nvcc --version | tail -2
which g++ && g++ --version | head -1
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cxx11abi', torch._C._GLIBCXX_USE_CXX11_ABI)"
echo

echo "=== Compiling flash-attn==2.7.4.post1 (~60-90 min, sm_75 + sm_86) ==="
time pip install --no-build-isolation --force-reinstall --no-deps "flash-attn==2.7.4.post1" 2>&1

echo
echo "=== Verifying ==="
python -c "from flash_attn import flash_attn_varlen_func; import flash_attn; print('flash_attn', flash_attn.__version__, 'OK')"
echo "DONE"
