# CCKW Extreme Rainfall

Project-local workflow for reproducing and extending western Pacific / Maritime
Continent extreme-rainfall diagnostics from ICON C5 CNTL and P4K experiments.
The repository focuses on code and editable notebooks; large NetCDF products
are regenerated on Levante and are intentionally not versioned.

## What This Project Does

The workflow supports four connected tasks:

1. Extract daily model-level wind and pressure fields over 30S-30N on a
   2-degree latitude-longitude grid.
2. Interpolate model-level `ua` and `va` to pressure levels, especially 850 hPa.
3. Compute seasonal rainfall means, seasonal 95th/99th percentiles, 850 hPa
   seasonal winds, and P4K-CNTL changes.
4. Tune AGU-style maps and interactive notebook diagnostics for seasonal
   extreme rainfall, `Pr99-Pr95`, dominant contributing seasons, and P4K
   warming changes.

The code assumes the DKRZ Levante filesystem layout used in this project:

```text
/work/mh1498/m301257
```

## Repository Layout

```text
.
├── README.md
├── scripts/
│   ├── Xianpumap.py
│   ├── build_p4k_cntl_difference.py
│   ├── interpolate_to_pressure_levels.py
│   ├── plot_extreme_rainfall_spatial_distribution.py
│   ├── preprocess_lat30_wind_pressure.py
│   └── standardize_850hpa_wind_files.py
├── notebooks/
│   └── Python_*.ipynb
├── run_extreme_rainfall_spatial_distribution.sh
└── run_project_extreme_rainfall_pipeline.sh
```

Generated directories are ignored by Git:

```text
data/       # project-local NetCDF products and diagnostic field caches
figures/    # generated image products only
```

## Main Pipeline

Run the full project-local pipeline on a Levante compute node:

```bash
cd /work/mh1498/m301257/code_extreme_event
./run_project_extreme_rainfall_pipeline.sh
```

The pipeline has five stages:

1. `scripts/preprocess_lat30_wind_pressure.py` extracts `pfull`, `ua`, and `va` for CNTL
   and P4K from the NextGEMS/ICON catalog, using a 36S-36N HEALPix buffer before
   interpolation onto the target 30S-30N grid.
2. `scripts/interpolate_to_pressure_levels.py` interpolates model-level `ua` and `va`
   to 850 hPa.
3. `scripts/standardize_850hpa_wind_files.py` converts the interpolation output into
   explicitly named single-level 850 hPa files.
4. `scripts/plot_extreme_rainfall_spatial_distribution.py` computes CNTL seasonal
   rainfall statistics and 850 hPa winds.
5. The same plotting script computes the P4K seasonal statistics.
6. `scripts/build_p4k_cntl_difference.py` writes P4K-CNTL seasonal difference fields.

Default outputs:

```text
data/model_layers_lat30/       # per-level extraction files
data/model_levels_lat30/       # merged all-level pfull/ua/va files
data/pressure_levels_lat30/    # intermediate pressure-interpolation outputs
data/wind_850hpa_lat30/        # explicit single-level ua/va files at 850 hPa
data/seasonal_fields/          # compact seasonal rainfall/wind NetCDF caches
figures/                       # image outputs, grouped by plotting script
```

## Compute-Node Controls

The extraction step can use independent model-level parallelism while preserving
the full multi-level output:

```bash
LEVEL_PARALLELISM=8 TIME_BATCH_SIZE=730 ./run_project_extreme_rainfall_pipeline.sh
```

Useful environment variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `PYTHON` | Python executable | `/home/m/m301257/.conda/envs/xianpu/bin/python` |
| `LEVEL_PARALLELISM` | Number of independent model levels processed in parallel | `4` |
| `TIME_BATCH_SIZE` | Daily time steps loaded per level task | `730` |
| `DASK_NUM_WORKERS` | Threads used by Dask threaded computations | `SLURM_CPUS_PER_TASK` or `8` |
| `DASK_INNER_SCHEDULER` | Scheduler inside each level-parallel process | `synchronous` |
| `DASK_INNER_WORKERS` | Dask workers inside each level-parallel process | `1` |

Use lower `LEVEL_PARALLELISM` if memory or filesystem I/O becomes limiting.

## Script Inventory

| File | Role |
| --- | --- |
| `run_project_extreme_rainfall_pipeline.sh` | End-to-end Levante workflow. Builds project-local 30S-30N wind/pressure products, interpolates to 850 hPa, computes CNTL/P4K seasonal fields, and writes P4K-CNTL differences. |
| `scripts/preprocess_lat30_wind_pressure.py` | Optimized extractor for ICON C5 3-D fields. Wraps the existing `process_3d_data_optimized` workflow, supports per-level multiprocessing, skip-existing behavior, retries, Dask scheduler choices, and merging per-level NetCDF files into all-level products. |
| `scripts/interpolate_to_pressure_levels.py` | Project-local pressure interpolation utility. Interpolates model-level variables to requested pressure levels and writes the generic intermediate files consumed by the 850 hPa standardization step. |
| `scripts/standardize_850hpa_wind_files.py` | Converts generic pressure-level interpolation outputs such as `ua_pressure_levels_cntl.nc` into explicit single-level files such as `ua_850hpa_cntl_lat30.nc`. |
| `scripts/plot_extreme_rainfall_spatial_distribution.py` | Batch script for Figure-1-style seasonal rainfall maps. Computes seasonal mean precipitation, seasonal `Pr95`/`Pr99`, seasonal 850 hPa winds, stores compact field caches, and saves AGU-style PNG figures. |
| `scripts/build_p4k_cntl_difference.py` | Builds seasonal P4K-CNTL difference fields from the CNTL and P4K field-cache NetCDF files. Can also export standalone CNTL/P4K `rain_p99` files. |
| `run_extreme_rainfall_spatial_distribution.sh` | Lightweight wrapper around `scripts/plot_extreme_rainfall_spatial_distribution.py`, useful when wind pressure-level files already exist. |
| `scripts/Xianpumap.py` | Shared plotting helpers for Cartopy/GeoCAT western-Pacific maps, AGU sizing/style constants, tick formatting, colorbar helpers, and Hovmoller plotting utilities. |

## Notebook Inventory

The notebooks keep plotting code editable so figure style, levels, map extent,
and contribution diagnostics can be adjusted interactively.

| Notebook | Purpose |
| --- | --- |
| `notebooks/Python_extreme_rainfall_spatial_distribution.ipynb` | Main editable Figure-1-style notebook. Reads cached seasonal fields or recomputes them; plots CNTL/P4K seasonal rainfall, 850 hPa winds, percentiles, and warming changes. |
| `notebooks/Python_extreme_rainfall_95&99.ipynb` | Focused seasonal `Pr95`/`Pr99` map notebook using cached rainfall/wind diagnostics. |
| `notebooks/Python_rainfall_95&99.ipynb` | Direct CNTL/P4K percentile analysis from precipitation files. Computes full-period `Pr95`, `Pr99`, `Pr99-Pr95`, seasonal `Pr99`, seasonal extreme-rainfall contributions, dominant contributing seasons, and P4K-CNTL contribution changes. |
| `notebooks/Python_extreme_rainfall_wind850.ipynb` | CNTL-oriented seasonal rainfall and 850 hPa wind plotting notebook. |
| `notebooks/Python_extreme_rainfall_wind850_warming_P4k.ipynb` | P4K/warming counterpart for seasonal rainfall and 850 hPa wind plotting. |
| `notebooks/Python_DJF_mean_data_get.ipynb` | Exploratory seasonal-mean precipitation notebook comparing CNTL and P4K. |
| `notebooks/Python_check_olr_origin.ipynb` | Exploratory filtering/check notebook using `CCKWFilter` on precipitation input; retained as provenance for wave-filter checks. |

## Data Inputs

Primary precipitation inputs are expected under:

```text
/work/mh1498/m301257/processed_data_lat_30/2d_layers/pr_cntl/pr_2deg_interp.nc
/work/mh1498/m301257/processed_data_lat_30/2d_layers/pr_p4k/pr_2deg_interp.nc
```

Model-level `pfull`, `ua`, and `va` are read from the NextGEMS/ICON intake
catalog:

```text
https://data.nextgems-h2020.eu/catalog.yaml
```

The pressure interpolation stage uses the project-local
`scripts/interpolate_to_pressure_levels.py` script.

Wind-vector legends use:

```text
/work/mh1498/m301257/wave_tools/easyxp.py
```

## Minimal Commands

Generate only the CNTL Figure-1-style map when 850 hPa wind files already exist:

```bash
./run_extreme_rainfall_spatial_distribution.sh \
  --pr-path /work/mh1498/m301257/processed_data_lat_30/2d_layers/pr_cntl/pr_2deg_interp.nc \
  --u-path data/wind_850hpa_lat30/ua_850hpa_cntl_lat30.nc \
  --v-path data/wind_850hpa_lat30/va_850hpa_cntl_lat30.nc
```

Generate P4K-CNTL difference fields after CNTL and P4K caches exist:

```bash
/home/m/m301257/.conda/envs/xianpu/bin/python scripts/build_p4k_cntl_difference.py \
  --cntl-fields data/seasonal_fields/seasonal_rainfall_wind_cntl_full_domain_fields.nc \
  --p4k-fields data/seasonal_fields/seasonal_rainfall_wind_p4k_full_domain_fields.nc \
  --output data/seasonal_fields/seasonal_rainfall_wind_p4k_minus_cntl_fields.nc
```

## GitHub Policy

This repository tracks source code, notebooks, and documentation. It does not
track the generated NetCDF data products or figures by default. Regenerate them
on Levante using the commands above.
