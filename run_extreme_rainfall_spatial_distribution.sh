#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/m301257_matplotlib}"

PYTHON="${PYTHON:-/home/m/m301257/.conda/envs/xianpu/bin/python}"
SCRIPT="/work/mh1498/m301257/code_extreme_event/scripts/plot_extreme_rainfall_spatial_distribution.py"

"${PYTHON}" "${SCRIPT}" \
  --scheduler threads \
  --workers auto \
  --threads-per-worker 4 \
  "$@"
