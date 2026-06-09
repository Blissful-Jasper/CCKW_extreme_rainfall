#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/m301257_matplotlib}"

PYTHON="${PYTHON:-/home/m/m301257/.conda/envs/xianpu/bin/python}"
PROJECT_DIR="/work/mh1498/m301257/code_extreme_event"
BASE_DIR="/work/mh1498/m301257"
FIG_DIR="${PROJECT_DIR}/figures"
LAYER_DIR="${PROJECT_DIR}/data/model_layers_lat30"
MODEL_LEVEL_DIR="${PROJECT_DIR}/data/model_levels_lat30"
PRESSURE_DIR="${PROJECT_DIR}/data/pressure_levels_lat30"

PREPROCESS_SCRIPT="${PROJECT_DIR}/preprocess_lat30_wind_pressure.py"
INTERP_SCRIPT="${BASE_DIR}/code/interpolate_to_pressure_levels.py"
PLOT_SCRIPT="${PROJECT_DIR}/plot_extreme_rainfall_spatial_distribution.py"
DIFF_SCRIPT="${PROJECT_DIR}/build_p4k_cntl_difference.py"

DASK_NUM_WORKERS="${DASK_NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
TIME_BATCH_SIZE="${TIME_BATCH_SIZE:-730}"
LEVEL_PARALLELISM="${LEVEL_PARALLELISM:-4}"
DASK_INNER_SCHEDULER="${DASK_INNER_SCHEDULER:-synchronous}"
DASK_INNER_WORKERS="${DASK_INNER_WORKERS:-1}"

mkdir -p "${FIG_DIR}" "${LAYER_DIR}" "${MODEL_LEVEL_DIR}" "${PRESSURE_DIR}"

echo "Project output directory: ${PROJECT_DIR}"
echo "Model-level layer output: ${LAYER_DIR}"
echo "Merged model-level output: ${MODEL_LEVEL_DIR}"
echo "Pressure-level wind output: ${PRESSURE_DIR}"
echo "Seasonal fields and figures: ${FIG_DIR}"
echo "Preprocess level parallelism: ${LEVEL_PARALLELISM}"
echo "Preprocess time batch size: ${TIME_BATCH_SIZE}"

echo
echo "[1/5] Building 30S-30N ua/va/pfull model-level files inside the project directory..."
"${PYTHON}" "${PREPROCESS_SCRIPT}" \
  --variables pfull ua va \
  --experiments cntl p4k \
  --layer-dir "${LAYER_DIR}" \
  --output-dir "${MODEL_LEVEL_DIR}" \
  --target-lat-min -30 \
  --target-lat-max 30 \
  --target-lon-min 0 \
  --target-lon-max 358 \
  --target-step 2 \
  --grid-minmax-lat 36 \
  --time-batch-size "${TIME_BATCH_SIZE}" \
  --scheduler threads \
  --num-workers "${DASK_NUM_WORKERS}" \
  --level-parallelism "${LEVEL_PARALLELISM}" \
  --inner-scheduler "${DASK_INNER_SCHEDULER}" \
  --inner-workers "${DASK_INNER_WORKERS}"

echo
echo "[2/5] Interpolating ua/va to 850 hPa inside the project directory..."
"${PYTHON}" "${INTERP_SCRIPT}" \
  --var ua va \
  --experiments cntl p4k \
  --plev 850 \
  --data-dir "${MODEL_LEVEL_DIR}" \
  --output-dir "${PRESSURE_DIR}" \
  --lat-range -30 30

echo
echo "[3/5] Computing CNTL seasonal rainfall percentiles and 850 hPa winds..."
"${PYTHON}" "${PLOT_SCRIPT}" \
  --pr-path "${BASE_DIR}/processed_data_lat_30/2d_layers/pr_cntl/pr_2deg_interp.nc" \
  --u-path "${PRESSURE_DIR}/ua_pressure_levels_cntl.nc" \
  --v-path "${PRESSURE_DIR}/va_pressure_levels_cntl.nc" \
  --output-dir "${FIG_DIR}" \
  --fields-name "seasonal_rainfall_wind_cntl_full_domain_fields.nc" \
  --p99-fields-name "seasonal_rainfall_p99_cntl_full_domain_fields.nc" \
  --output-name "figure1_style_seasonal_rainfall_wind_icon_cntl.png" \
  --scheduler threads \
  --workers auto \
  --threads-per-worker 4 \
  --dpi 300

echo
echo "[4/5] Computing P4K seasonal rainfall percentiles and 850 hPa winds..."
"${PYTHON}" "${PLOT_SCRIPT}" \
  --pr-path "${BASE_DIR}/processed_data_lat_30/2d_layers/pr_p4k/pr_2deg_interp.nc" \
  --u-path "${PRESSURE_DIR}/ua_pressure_levels_p4k.nc" \
  --v-path "${PRESSURE_DIR}/va_pressure_levels_p4k.nc" \
  --output-dir "${FIG_DIR}" \
  --fields-name "seasonal_rainfall_wind_p4k_full_domain_fields.nc" \
  --p99-fields-name "seasonal_rainfall_p99_p4k_full_domain_fields.nc" \
  --output-name "figure1_style_seasonal_rainfall_wind_icon_p4k.png" \
  --scheduler threads \
  --workers auto \
  --threads-per-worker 4 \
  --dpi 300

echo
echo "[5/5] Building P4K-CNTL difference fields..."
"${PYTHON}" "${DIFF_SCRIPT}" \
  --cntl-fields "${FIG_DIR}/seasonal_rainfall_wind_cntl_full_domain_fields.nc" \
  --p4k-fields "${FIG_DIR}/seasonal_rainfall_wind_p4k_full_domain_fields.nc" \
  --output "${FIG_DIR}/seasonal_rainfall_wind_p4k_minus_cntl_fields.nc" \
  --cntl-p99-output "${FIG_DIR}/seasonal_rainfall_p99_cntl_full_domain_fields.nc" \
  --p4k-p99-output "${FIG_DIR}/seasonal_rainfall_p99_p4k_full_domain_fields.nc"

echo
echo "Done. Project-local outputs are under:"
echo "  ${MODEL_LEVEL_DIR}"
echo "  ${PRESSURE_DIR}"
echo "  ${FIG_DIR}"
