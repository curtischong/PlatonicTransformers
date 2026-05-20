#!/bin/bash
# Reproduce the qcczbpfn Platonic Transformer recipe on OMol25 (hipster, 4× RTX 6000 Ada).
# Mirrors qcczbpfn's hardware/data exactly: 4-way DDP at 3000 atoms/600k edges per rank
# (= 12000 atoms / 2.4M edges per optimizer step), reading from /scratch/ebekker/omol/.
# See configs/omol.yaml for the recipe; CLI overrides below restore the 4-GPU defaults
# (the yaml defaults are tuned for 1× H100 on snellius).
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:4
#SBATCH --time=5-00:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out
#SBATCH --job-name=PR-omol-hipster-4gpu

set -e
mkdir -p logs

DATA_PATH="${DATA_PATH:-/scratch/ebekker/omol/open_mol}"
VENV_PATH="${VENV_PATH:-/home/ebekker/platonic-omol-backup/venv}"

source "${VENV_PATH}/bin/activate"

export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=120
# flash-attn on hipster's compile pipeline opens many shared libs; raise nofile.
ulimit -n 65536 || true

echo "=== hipster 4× DDP run ==="
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "JobID:       ${SLURM_JOB_ID}"
echo "DATA_PATH:   ${DATA_PATH}"
echo "VENV_PATH:   ${VENV_PATH}"
echo "GPUs:        ${SLURM_GPUS_ON_NODE:-?}"
echo "ntasks:      ${SLURM_NTASKS:-?}"
nvidia-smi -L || true

# CLI overrides restore qcczbpfn's per-rank caps (the yaml has 12k/2.4M for 1-GPU).
# Lightning DDP all-reduces grads at every backward, so 4 ranks × 3000 atoms = 12k
# atoms per optimizer step, byte-identical to qcczbpfn's setup.
srun python mains/main_omol.py \
    --config configs/omol.yaml \
    --dataset.data_dir="${DATA_PATH}" \
    --system.gpus=4 \
    --system.accumulate_grad_batches=1 \
    --training.batch_size=16 \
    --training.max_atoms_per_batch=3000 \
    --training.max_atoms_per_batch_val=3000 \
    --training.max_edges_per_batch=600000 \
    --training.max_edges_per_batch_val=600000 \
    --logging.enabled=true

exit_code=$?
echo "=== JOB FINISHED (exit ${exit_code}) ==="
exit ${exit_code}
