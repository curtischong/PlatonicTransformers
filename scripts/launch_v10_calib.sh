#!/bin/bash
# Launch all 11 v10 calibration jobs in parallel on Snellius gpu_h100.
# Each job runs 300 steps (200 warmup + 100 measurement) with the v10 recipe.
#
# Env overrides:
#   MAX_ATOMS  dynamic-batching cap (default 6000)
#   DATA       data config name: omol_4m | omol_full (default omol_4m)
#   TAG        suffix for wandb exp_name (default derived from MAX_ATOMS+DATA)

set -euo pipefail

cd "$(dirname "$0")/.."

MAX_ATOMS=${MAX_ATOMS:-6000}
DATA=${DATA:-omol_4m}
TAG=${TAG:-${DATA}-atoms${MAX_ATOMS}}

# MODEL_ID  SOLID         HDIM  NHEAD  FC
MODELS=(
  "TT-1 trivial     336  7   6"
  "TT-2 trivial     576  12  6"
  "TT-3 trivial     864  18  6"
  "TT-4 trivial     1152 24  6"
  "TT-5 trivial     1728 36  6"
  "PT-A tetrahedron 576  12  72"
  "PT-B tetrahedron 1152 24  72"
  "PT-C tetrahedron 1728 36  72"
  "PT-D tetrahedron 2304 48  72"
  "PT-E tetrahedron 2880 60  72"
  "PT-F tetrahedron 4032 84  72"
)

echo "Launching 11 calibration jobs:  data=${DATA}  max_atoms=${MAX_ATOMS}  tag=${TAG}"

for spec in "${MODELS[@]}"; do
  read -r MID SOLID HDIM NHEAD FC <<< "$spec"
  echo "Submitting v10-calib-${MID}-${TAG}  solid=${SOLID} d=${HDIM} nhead=${NHEAD} fc=${FC}"
  sbatch \
    --job-name="v10-calib-${MID}-${TAG}" \
    --export="MODEL_ID=${MID},SOLID=${SOLID},HDIM=${HDIM},NHEAD=${NHEAD},FC=${FC},MAX_ATOMS=${MAX_ATOMS},DATA=${DATA},RUN_TAG=${TAG}" \
    scripts/run_v10_calib.sh
done

echo "All 11 jobs submitted. Monitor with: squeue -u ebekkers --name=v10-calib"
