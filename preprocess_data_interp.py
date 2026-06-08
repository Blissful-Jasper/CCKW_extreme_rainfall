#!/usr/bin/env python3
"""Build project-local lat30 model-level wind and pressure files.

This wraps the existing optimized 3-D preprocessing used by
00_preprocess_lat30_region_for_clw_cli.ipynb, but keeps every output under the
extreme-rainfall project directory. The merged files are written in the layout
expected by /work/mh1498/m301257/code/interpolate_to_pressure_levels.py:

    data/model_levels_lat30/cntl/ua_all_levels.nc
    data/model_levels_lat30/cntl/va_all_levels.nc
    data/model_levels_lat30/cntl/pfull_all_levels.nc
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import dask
import numpy as np
import xarray as xr


BASE_DIR = Path("/work/mh1498/m301257")
PROJECT_DIR = BASE_DIR / "code_extreme_event"
DEFAULT_LAYER_DIR = PROJECT_DIR / "data/model_layers_lat30"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data/model_levels_lat30"
CATALOG_URL = "https://data.nextgems-h2020.eu/catalog.yaml"

EXPERIMENTS = {
    "cntl": ("CNTL", "AMIP_CNTL"),
    "p4k": ("P4K", "AMIP_P4K"),
    "4co2": ("4CO2", "AMIP_4CO2"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process ICON C5 model-level ua/va/pfull over a target latitude "
            "band and save project-local all-level NetCDF files."
        )
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["pfull", "ua", "va"],
        help="3-D variables to process. Default: pfull ua va",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["cntl", "p4k"],
        choices=sorted(EXPERIMENTS),
        help="Experiments to process. Default: cntl p4k",
    )
    parser.add_argument("--layer-dir", type=Path, default=DEFAULT_LAYER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--catalog-url", default=CATALOG_URL)
    parser.add_argument("--target-lat-min", type=float, default=-30.0)
    parser.add_argument("--target-lat-max", type=float, default=30.0)
    parser.add_argument("--target-lon-min", type=float, default=0.0)
    parser.add_argument("--target-lon-max", type=float, default=358.0)
    parser.add_argument("--target-step", type=float, default=2.0)
    parser.add_argument(
        "--grid-minmax-lat",
        type=float,
        default=36.0,
        help="Latitude half-width used in HEALPix extraction before interpolation.",
    )
    parser.add_argument("--level-start", type=float, default=None)
    parser.add_argument("--level-stop", type=float, default=None)
    parser.add_argument("--time-batch-size", type=int, default=365)
    parser.add_argument("--memory-threshold", type=float, default=85.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--scheduler",
        choices=("threads", "synchronous", "distributed"),
        default="threads",
        help="Dask scheduler for loading/computation inside each time batch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("DASK_NUM_WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", 8))),
        help="Threads for the threaded scheduler, or workers for distributed.",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=int(os.environ.get("DASK_THREADS_PER_WORKER", 1)),
        help="Threads per worker for the distributed scheduler.",
    )
    parser.add_argument(
        "--memory-limit",
        default=os.environ.get("DASK_MEMORY_LIMIT", "auto"),
        help="Per-worker memory limit for the distributed scheduler.",
    )
    parser.add_argument(
        "--dashboard-address",
        default=os.environ.get("DASK_DASHBOARD_ADDRESS", ":0"),
        help="Dashboard address for the distributed scheduler.",
    )
    parser.add_argument(
        "--level-parallelism",
        type=int,
        default=int(os.environ.get("LEVEL_PARALLELISM", 1)),
        help=(
            "Process independent model levels in parallel processes. "
            "Use 1 for the legacy serial level loop."
        ),
    )
    parser.add_argument(
        "--inner-scheduler",
        choices=("threads", "synchronous"),
        default=os.environ.get("DASK_INNER_SCHEDULER", "synchronous"),
        help="Dask scheduler used inside each level-parallel worker.",
    )
    parser.add_argument(
        "--inner-workers",
        type=int,
        default=int(os.environ.get("DASK_INNER_WORKERS", 1)),
        help="Dask workers used inside each level-parallel worker.",
    )
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument(
        "--only-merge",
        action="store_true",
        help="Skip catalog processing and only merge existing layer files.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess existing layer files and rewrite merged all-level files.",
    )
    parser.add_argument(
        "--remove-layers-after-merge",
        action="store_true",
        help="Delete project-local per-level files after a successful merge.",
    )
    return parser


def target_values(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step))
    values = start + step * np.arange(count + 1, dtype=float)
    return np.round(values, 10)


def ensure_import_paths() -> None:
    code_dir = BASE_DIR / "code"
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


def merged_output_file(output_root: Path, var_name: str, exp_key: str) -> Path:
    return output_root / exp_key / f"{var_name}_all_levels.nc"


def merged_output_exists(args: argparse.Namespace, var_name: str, exp_key: str) -> bool:
    path = merged_output_file(args.output_dir, var_name, exp_key)
    if not path.exists():
        return False
    try:
        with xr.open_dataset(path) as ds:
            return var_name in ds.data_vars and "level" in ds.dims
    except Exception:
        return False


def pending_work(args: argparse.Namespace) -> dict[str, dict[str, tuple[str, str]]]:
    work: dict[str, dict[str, tuple[str, str]]] = {}
    for var_name in args.variables:
        experiments = {}
        for exp_key in args.experiments:
            if args.no_skip_existing or not merged_output_exists(args, var_name, exp_key):
                experiments[exp_key] = EXPERIMENTS[exp_key]
        if experiments:
            work[var_name] = experiments
    return work


def level_dim_name(da: xr.DataArray) -> str:
    for dim in ("level_full", "level_half", "level"):
        if dim in da.dims:
            return dim
    raise ValueError(f"Could not find a model-level dimension in {list(da.dims)}")


def levels_for_variable(
    catalog,
    var_name: str,
    dataset_key: str,
    level_slice: tuple[float | None, float | None],
) -> list[float]:
    da = catalog.ICON.C5[dataset_key].to_dask()[var_name].sel(time=slice("1980", "1993"))
    dim = level_dim_name(da)
    selected = da.sel({dim: slice(*level_slice)})
    return [float(value) for value in selected[dim].values]


def layer_file(layer_root: Path, var_name: str, exp_key: str, level: float) -> Path:
    exp_name = EXPERIMENTS[exp_key][0].lower()
    return layer_root / f"{var_name}_{exp_name}_layers" / f"{var_name}_lev_{int(level):03d}.nc"


@contextlib.contextmanager
def dask_runtime(args: argparse.Namespace):
    if args.scheduler != "distributed":
        dask.config.set(scheduler=args.scheduler, num_workers=max(1, args.num_workers))
        print(
            f"Dask scheduler: {args.scheduler}, threads: {max(1, args.num_workers)}",
            flush=True,
        )
        yield None
        return

    try:
        from dask.distributed import Client, LocalCluster
    except ImportError:
        dask.config.set(scheduler="threads", num_workers=max(1, args.num_workers))
        print(
            "dask.distributed is unavailable; falling back to threaded scheduler "
            f"with {max(1, args.num_workers)} threads.",
            flush=True,
        )
        yield None
        return

    cluster = LocalCluster(
        n_workers=max(1, args.num_workers),
        threads_per_worker=max(1, args.threads_per_worker),
        processes=True,
        memory_limit=args.memory_limit,
        dashboard_address=args.dashboard_address,
    )
    client = Client(cluster)
    print(
        "Dask distributed cluster: "
        f"{max(1, args.num_workers)} workers x {max(1, args.threads_per_worker)} threads "
        f"(dashboard: {client.dashboard_link})",
        flush=True,
    )
    try:
        yield client
    finally:
        client.close()
        cluster.close()


def process_single_level(task: dict) -> str:
    ensure_import_paths()

    import intake
    from process_3d_data_optimized import process_3d_variable_optimized

    dask.config.set(
        scheduler=task["inner_scheduler"],
        num_workers=max(1, int(task["inner_workers"])),
    )
    catalog = intake.open_catalog(task["catalog_url"])
    process_3d_variable_optimized(
        var_name=task["var_name"],
        experiment_name=task["exp_name"],
        dataset_key=task["dataset_key"],
        save_dir=task["layer_dir"],
        grid_dict=task["grid_dict"],
        target_lat=task["target_lat"],
        target_lon=task["target_lon"],
        catalog=catalog,
        level_slice=(task["level"], task["level"]),
        time_batch_size=task["time_batch_size"],
        memory_threshold=task["memory_threshold"],
        max_retries=task["max_retries"],
        skip_existing=task["skip_existing"],
    )
    return (
        f"{task['var_name']} {task['exp_key'].upper()} "
        f"level {int(task['level']):03d}"
    )


def process_layers_parallel(
    args: argparse.Namespace,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    grid_dict: dict,
    work: dict[str, dict[str, tuple[str, str]]],
) -> None:
    import intake

    catalog = intake.open_catalog(args.catalog_url)
    tasks = []
    level_slice = (args.level_start, args.level_stop)

    for var_name, experiments in work.items():
        for exp_key, (exp_name, dataset_key) in experiments.items():
            levels = levels_for_variable(catalog, var_name, dataset_key, level_slice)
            for level in levels:
                if (
                    not args.no_skip_existing
                    and layer_file(args.layer_dir, var_name, exp_key, level).exists()
                ):
                    print(
                        f"Layer skip existing: {var_name} {exp_key.upper()} "
                        f"level {int(level):03d}",
                        flush=True,
                    )
                    continue
                tasks.append(
                    {
                        "catalog_url": args.catalog_url,
                        "var_name": var_name,
                        "exp_key": exp_key,
                        "exp_name": exp_name,
                        "dataset_key": dataset_key,
                        "layer_dir": str(args.layer_dir),
                        "grid_dict": grid_dict,
                        "target_lat": target_lat,
                        "target_lon": target_lon,
                        "level": level,
                        "time_batch_size": args.time_batch_size,
                        "memory_threshold": args.memory_threshold,
                        "max_retries": args.max_retries,
                        "skip_existing": not args.no_skip_existing,
                        "inner_scheduler": args.inner_scheduler,
                        "inner_workers": args.inner_workers,
                    }
                )

    if not tasks:
        print("All requested per-level files already exist; skipping extraction.", flush=True)
        return

    parallelism = max(1, min(args.level_parallelism, len(tasks)))
    print(
        f"Processing {len(tasks)} model-level tasks with "
        f"{parallelism} parallel worker processes "
        f"(inner scheduler: {args.inner_scheduler}, inner workers: {args.inner_workers}).",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=parallelism) as pool:
        futures = [pool.submit(process_single_level, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), 1):
            label = future.result()
            print(f"[{idx}/{len(futures)}] Finished {label}", flush=True)


def process_layers(args: argparse.Namespace, target_lat: np.ndarray, target_lon: np.ndarray) -> None:
    import intake
    from process_3d_data_optimized import batch_process_3d_variables

    catalog = intake.open_catalog(args.catalog_url)
    grid_dict = {
        "nside": 256,
        "nest": True,
        "minmax_lat": args.grid_minmax_lat,
    }
    work = pending_work(args)

    if not work:
        print("All requested merged model-level outputs already exist; skipping extraction.", flush=True)
        return

    print(f"Layer output: {args.layer_dir}", flush=True)
    print(f"Merged output: {args.output_dir}", flush=True)
    print(
        "Target grid: "
        f"lat {target_lat[0]:.1f}..{target_lat[-1]:.1f} ({target_lat.size}), "
        f"lon {target_lon[0]:.1f}..{target_lon[-1]:.1f} ({target_lon.size})",
        flush=True,
    )
    print(
        f"HEALPix extraction half-width: {args.grid_minmax_lat:.1f} deg",
        flush=True,
    )

    if args.level_parallelism > 1:
        process_layers_parallel(args, target_lat, target_lon, grid_dict, work)
        return

    with dask_runtime(args):
        for var_name, experiments in work.items():
            print(
                f"Pending {var_name}: {', '.join(exp.upper() for exp in experiments)}",
                flush=True,
            )
            batch_process_3d_variables(
                var_names=[var_name],
                experiments=experiments,
                save_dir=str(args.layer_dir),
                grid_dict=grid_dict,
                target_lat=target_lat,
                target_lon=target_lon,
                catalog=catalog,
                level_slice=(args.level_start, args.level_stop),
                time_batch_size=args.time_batch_size,
                memory_threshold=args.memory_threshold,
                max_retries=args.max_retries,
                skip_existing=not args.no_skip_existing,
            )


def layer_files(layer_dir: Path, var_name: str) -> list[tuple[int, Path]]:
    pattern = re.compile(rf"^{re.escape(var_name)}_lev_(\d+)\.nc$")
    files: list[tuple[int, Path]] = []
    if not layer_dir.exists():
        return files
    for path in layer_dir.iterdir():
        match = pattern.match(path.name)
        if match:
            files.append((int(match.group(1)), path))
    return sorted(files)


def merge_variable_layers(
    layer_root: Path,
    output_root: Path,
    var_name: str,
    exp_key: str,
    *,
    dtype: str,
    skip_existing: bool,
    remove_layers_after_merge: bool,
) -> Path:
    exp_name = EXPERIMENTS[exp_key][0].lower()
    this_layer_dir = layer_root / f"{var_name}_{exp_name}_layers"
    output_dir = output_root / exp_key
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = merged_output_file(output_root, var_name, exp_key)

    if skip_existing and output_file.exists():
        print(f"Merge skip existing: {output_file}", flush=True)
        return output_file

    files = layer_files(this_layer_dir, var_name)
    if not files:
        raise FileNotFoundError(f"No layer files found in {this_layer_dir}")

    print(
        f"Merging {len(files)} {var_name} layers for {exp_key.upper()} -> {output_file}",
        flush=True,
    )

    arrays: list[xr.DataArray] = []
    datasets: list[xr.Dataset] = []
    for level, path in files:
        ds = xr.open_dataset(path, chunks={"time": 1000})
        datasets.append(ds)
        if var_name not in ds:
            data_vars = list(ds.data_vars)
            if len(data_vars) != 1:
                raise KeyError(f"{var_name!r} not found in {path}; variables={data_vars}")
            da = ds[data_vars[0]].rename(var_name)
        else:
            da = ds[var_name]
        arrays.append(da.expand_dims({"level": [level]}))

    try:
        merged = xr.concat(arrays, dim="level").sortby("level")
        merged = merged.astype(dtype)
        merged["level"].attrs.update(
            {
                "long_name": "model full level",
                "source": "restored from per-level file names",
            }
        )
        merged.attrs.update(
            {
                "preprocessing": "HEALPix to lat-lon, then per-level merge",
                "experiment": exp_key.upper(),
            }
        )
        merged.to_dataset(name=var_name).to_netcdf(output_file)
    finally:
        for ds in datasets:
            ds.close()

    if remove_layers_after_merge:
        for _, path in files:
            path.unlink()
        print(f"Removed merged layer files under {this_layer_dir}", flush=True)

    return output_file


def merge_all(args: argparse.Namespace) -> None:
    for exp_key in args.experiments:
        for var_name in args.variables:
            merge_variable_layers(
                args.layer_dir,
                args.output_dir,
                var_name,
                exp_key,
                dtype=args.dtype,
                skip_existing=not args.no_skip_existing,
                remove_layers_after_merge=args.remove_layers_after_merge,
            )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ensure_import_paths()

    args.layer_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_lat = target_values(args.target_lat_min, args.target_lat_max, args.target_step)
    target_lon = target_values(args.target_lon_min, args.target_lon_max, args.target_step)

    if not args.only_merge:
        process_layers(args, target_lat, target_lon)

    merge_all(args)

    print("Done. Merged model-level outputs:", flush=True)
    for exp_key in args.experiments:
        for var_name in args.variables:
            print(f"  {args.output_dir / exp_key / f'{var_name}_all_levels.nc'}", flush=True)


if __name__ == "__main__":
    main()
