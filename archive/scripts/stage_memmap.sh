#!/bin/bash
#SBATCH --account=gusei11738
#SBATCH --partition=staging
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/scratch-shared/ebekkers/logs/memmap-%j.out

# One-time LMDB → memmap preprocessing on the Snellius staging partition.
#
# Writes to /scratch-shared/ebekkers/omol25/open_mol_memmap/{train,val}/ with
# per-field flat binary files and a manifest.json. The result can be
# memmapped by OMolMemmapModule during training.
#
# Runtime estimate: ~1-2 h for train (102M samples) + ~5 min for val (2.76M)
# with 16 parallel workers.

set -euo pipefail

source /scratch-shared/ebekkers/scaling-laws-venv/bin/activate
module load 2024

export PYTHONPATH=/scratch-shared/ebekkers/platonic-omol/training
cd /scratch-shared/ebekkers/platonic-omol

mkdir -p /scratch-shared/ebekkers/logs

SRC_ROOT=/scratch-shared/ebekkers/omol25/open_mol
DEST_ROOT=/scratch-shared/ebekkers/omol25/open_mol_memmap

echo "[$(date -Iseconds)] Starting memmap preprocess"
echo "  SRC : $SRC_ROOT"
echo "  DEST: $DEST_ROOT"

# val first (small, quick sanity check)
echo "[$(date -Iseconds)] === val split ==="
python scripts/preprocess_memmap.py "$SRC_ROOT/val" "$DEST_ROOT/val" 16

# train next (large)
echo "[$(date -Iseconds)] === train split ==="
python scripts/preprocess_memmap.py "$SRC_ROOT/train" "$DEST_ROOT/train" 16

echo "[$(date -Iseconds)] Done."
du -sh "$DEST_ROOT/train" "$DEST_ROOT/val" 2>/dev/null || true
