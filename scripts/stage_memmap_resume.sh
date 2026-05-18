#!/bin/bash
#SBATCH --account=gusei11738
#SBATCH --partition=staging
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=224G
#SBATCH --output=/scratch-shared/ebekkers/logs/memmap-resume-%j.out

# Resume the 9 train chunks that 22055213 never processed. Uses spawn start
# method for clean child Python processes. Original failure was OOM —
# 64G / 16 workers = 4G/worker was too tight for fairchem/LMDB + dirty
# memmap pages. 224G / 9 workers ≈ 25G/worker is conservative.

set -euo pipefail

source /scratch-shared/ebekkers/scaling-laws-venv/bin/activate
module load 2024

export PYTHONPATH=/scratch-shared/ebekkers/platonic-omol/training
cd /scratch-shared/ebekkers/platonic-omol

SRC_ROOT=/scratch-shared/ebekkers/omol25/open_mol
DEST_ROOT=/scratch-shared/ebekkers/omol25/open_mol_memmap

# 9 chunks that 22055213 failed to process (never printed anything).
RANGES="6354142-12708284,12708284-19062427,19062427-25416570,25416570-31770712,50833140-57187282,63541425-69895567,69895567-76249710,76249710-82603852,95312137-101666280"

echo "[$(date -Iseconds)] Resuming memmap preprocess — train split, 9 chunks"
python scripts/preprocess_memmap_resume.py "$SRC_ROOT/train" "$DEST_ROOT/train" "$RANGES" 9
echo "[$(date -Iseconds)] Done."
