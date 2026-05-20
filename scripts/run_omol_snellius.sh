#!/bin/bash
# Submit the OMol25 4M force-field training run on snellius.
#
# Usage:   ./scripts/run_omol_snellius.sh [1|4]
# Default: 4   (4× H100, 3000 atoms/rank × 4 = 12000 effective — qcczbpfn-matched)
# Alt:     1   (1× H100, 12000 atoms/step — same effective batch, single rank)
#
# Both call `sbatch` on the matching launcher in this directory; tweak that
# launcher to change wall-clock, partition, DATA_PATH, etc.
set -e
cd "$(dirname "$0")/.."

GPUS="${1:-4}"
case "$GPUS" in
    1) exec sbatch scripts/run_omol_platonic_snellius_1gpu.sh ;;
    4) exec sbatch scripts/run_omol_platonic_snellius_4gpu.sh ;;
    *) echo "Usage: $0 [1|4]" >&2; exit 1 ;;
esac
