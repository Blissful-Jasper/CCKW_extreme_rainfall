#!/usr/bin/env python
"""Reproduce a Figure-1-style seasonal rainfall map with 850 hPa winds.

The defaults use the local ICON CNTL data that are already present in this
workspace. To use TRMM/ERA-Interim instead, pass the precipitation, u-wind,
and v-wind files plus variable names on the command line.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/m301257_matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import dask
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from matplotlib.colors import BoundaryNorm, ListedColormap

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

try:
    from dask.distributed import Client, LocalCluster, progress
except ImportError:  # pragma: no cover - distributed is available in xianpu.
    Client = None
    LocalCluster = None
    progress = None


BASE_DIR = Path("/work/mh1498/m301257")
CODE_DIR = BASE_DIR / "code_extreme_event"

EASYXP_PATH = BASE_DIR / "wave_tools" / "easyxp.py"
_easyxp_spec = importlib.util.spec_from_file_location("easyxp_local", EASYXP_PATH)
if _easyxp_spec is None or _easyxp_spec.loader is None:
    raise ImportError(f"Cannot load quiver legend helper from {EASYXP_PATH}")
_easyxp_module = importlib.util.module_from_spec(_easyxp_spec)
_easyxp_spec.loader.exec_module(_easyxp_module)
simple_quiver_legend = _easyxp_module.simple_quiver_legend

DEFAULT_PR_PATH = (
    BASE_DIR / "processed_data_lat_30/2d_layers/pr_cntl/pr_2deg_interp.nc"
)
DEFAULT_U850_PATH = (
    CODE_DIR / "data/wind_850hpa_lat30/ua_850hpa_cntl_lat30.nc"
)
DEFAULT_V850_PATH = (
    CODE_DIR / "data/wind_850hpa_lat30/va_850hpa_cntl_lat30.nc"
)

SEASONS = ("DJF", "MAM", "JJA", "SON")
PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
DOMAIN = (94.0, 136.0, -14.0, 25.0)
MEAN_LEVELS = np.array([0.3, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0])
P95_LEVELS = np.array([0.3, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0])

MEAN_CMAP = ListedColormap(
    [
        "#fbf8cf",
        "#dff1b7",
        "#a7dba0",
        "#72c9b7",
        "#4aa6c2",
        "#2d74bd",
        "#234f9f",
        "#20306f",
    ]
)
P95_CMAP = ListedColormap(
    [
        "#ffffff",
        "#fbf8cf",
        "#dff1b7",
        "#a7dba0",
        "#72c9b7",
        "#4aa6c2",
        "#2d74bd",
        "#234f9f",
        "#20306f",
        "#17204f",
    ]
)

REGION_BOXES = {
    "PM": (99.0, 104.5, 1.0, 7.0),
    "EM": (109.0, 119.0, 0.5, 6.5),
    "WI": (99.0, 105.0, -9.0, 2.0),
    "NI": (104.5, 113.0, -5.0, 1.0),
    "SI": (104.0, 115.0, -10.0, -6.0),
    "EI": (113.0, 124.0, -5.0, 2.0),
    "NP": (120.0, 126.5, 12.0, 21.0),
    "SP": (119.0, 126.5, 5.0, 13.0),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute and plot seasonal rainfall means, DJF/JJA 95th percentiles, and 850 hPa winds."
    )
    parser.add_argument("--pr-path", type=Path, default=DEFAULT_PR_PATH)
    parser.add_argument("--u-path", type=Path, default=DEFAULT_U850_PATH)
    parser.add_argument("--v-path", type=Path, default=DEFAULT_V850_PATH)
    parser.add_argument("--pr-var", default="pr")
    parser.add_argument("--u-var", default="ua")
    parser.add_argument("--v-var", default="va")
    parser.add_argument("--plev", type=float, default=85000.0)
    parser.add_argument("--lon-min", type=float, default=DOMAIN[0], help="Plot longitude minimum.")
    parser.add_argument("--lon-max", type=float, default=DOMAIN[1], help="Plot longitude maximum.")
    parser.add_argument("--lat-min", type=float, default=DOMAIN[2], help="Plot latitude minimum.")
    parser.add_argument("--lat-max", type=float, default=DOMAIN[3], help="Plot latitude maximum.")
    parser.add_argument("--data-lon-min", type=float, default=None, help="Optional computation/output longitude minimum.")
    parser.add_argument("--data-lon-max", type=float, default=None, help="Optional computation/output longitude maximum.")
    parser.add_argument("--data-lat-min", type=float, default=None, help="Optional computation/output latitude minimum.")
    parser.add_argument("--data-lat-max", type=float, default=None, help="Optional computation/output latitude maximum.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--pr-scale",
        default="auto",
        help="Use 'auto', 1, or 86400. ICON pr defaults to kg m-2 s-1 and needs 86400.",
    )
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--memory-limit", default="auto")
    parser.add_argument("--scheduler", choices=("distributed", "threads"), default="distributed")
    parser.add_argument("--wind-step", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_DIR / "figures/plot_extreme_rainfall_spatial_distribution",
        help="Directory for image outputs only.",
    )
    parser.add_argument(
        "--fields-dir",
        type=Path,
        default=CODE_DIR / "data/seasonal_fields",
        help="Directory for generated NetCDF diagnostic fields.",
    )
    parser.add_argument(
        "--output-name",
        default="figure1_style_seasonal_rainfall_wind_icon_cntl.png",
    )
    parser.add_argument(
        "--fields-name",
        default="seasonal_rainfall_wind_cntl_full_domain_fields.nc",
    )
    parser.add_argument(
        "--p99-fields-name",
        default="seasonal_rainfall_p99_cntl_full_domain_fields.nc",
    )
    parser.add_argument("--no-save-fields", action="store_true")
    parser.add_argument("--no-region-boxes", action="store_true")
    parser.add_argument("--title", default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _existing_file(path: Path, label: str) -> Path:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _coord_name(obj: xr.Dataset | xr.DataArray, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in obj.coords or name in obj.dims:
            return name
    raise KeyError(f"Could not find any of these coordinates: {candidates}")


def _var(ds: xr.Dataset, preferred: str) -> xr.DataArray:
    if preferred in ds:
        return ds[preferred]
    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        return ds[data_vars[0]]
    raise KeyError(f"Variable {preferred!r} not found. Available variables: {data_vars}")


def subset_region(
    da: xr.DataArray,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> xr.DataArray:
    lat_name = _coord_name(da, ("lat", "latitude", "y"))
    lon_name = _coord_name(da, ("lon", "longitude", "x"))

    if float(da[lon_name].max()) > 180.0 and lon_min < 0:
        lon_min = lon_min % 360.0
        lon_max = lon_max % 360.0

    da = da.sortby(lon_name)
    lat_values = da[lat_name].values
    lat_slice = slice(lat_min, lat_max)
    if lat_values[0] > lat_values[-1]:
        lat_slice = slice(lat_max, lat_min)

    if lon_min <= lon_max:
        return da.sel({lon_name: slice(lon_min, lon_max), lat_name: lat_slice})

    left = da.sel({lon_name: slice(lon_min, float(da[lon_name].max())), lat_name: lat_slice})
    right = da.sel({lon_name: slice(float(da[lon_name].min()), lon_max), lat_name: lat_slice})
    return xr.concat([left, right], dim=lon_name)


def apply_time_range(
    da: xr.DataArray,
    start_date: str | None,
    end_date: str | None,
) -> xr.DataArray:
    if start_date is None and end_date is None:
        return da
    time_name = _coord_name(da, ("time",))
    return da.sel({time_name: slice(start_date, end_date)})


def choose_pr_scale(da: xr.DataArray, pr_scale: str) -> float:
    if pr_scale != "auto":
        return float(pr_scale)

    units = str(da.attrs.get("units", "")).lower()
    if "kg" in units and ("s-1" in units or "s**-1" in units or "/s" in units):
        return 86400.0
    if units in {"m s-1", "m/s"}:
        return 86400.0 * 1000.0
    if "mm" in units and ("day" in units or "d-1" in units):
        return 1.0

    sample = da.isel(time=slice(0, min(30, da.sizes["time"]))).mean()
    with dask.config.set(scheduler="threads"):
        sample_value = float(sample.compute())
    return 86400.0 if sample_value < 0.1 else 1.0


def select_pressure_level(da: xr.DataArray, target_plev: float) -> xr.DataArray:
    if "plev" not in da.dims:
        return da.drop_vars("plev", errors="ignore")
    if "plev" not in da.coords:
        return da

    plev = da["plev"]
    target = target_plev
    units = str(plev.attrs.get("units", "")).lower()
    if ("hpa" in units or "millibar" in units) and target > 2000.0:
        target = target / 100.0
    elif ("pa" in units and "hpa" not in units) and target < 2000.0:
        target = target * 100.0
    elif float(plev.max()) < 2000.0 and target > 2000.0:
        target = target / 100.0

    return da.sel(plev=target, method="nearest").drop_vars("plev", errors="ignore")


def grid_summary(da: xr.DataArray) -> str:
    lat_name = _coord_name(da, ("lat", "latitude", "y"))
    lon_name = _coord_name(da, ("lon", "longitude", "x"))
    lat = da[lat_name]
    lon = da[lon_name]
    return (
        f"lat {float(lat.min()):g}..{float(lat.max()):g} (n={lat.size}), "
        f"lon {float(lon.min()):g}..{float(lon.max()):g} (n={lon.size})"
    )


def make_cluster(
    scheduler: str,
    workers: str,
    threads_per_worker: int,
    memory_limit: str,
) -> tuple[Client | None, LocalCluster | None]:
    ncores = os.cpu_count() or 1
    if workers == "auto":
        cap = 32 if scheduler == "distributed" else ncores
        n_workers = max(1, min(cap, ncores // max(1, threads_per_worker)))
    else:
        n_workers = int(workers)

    if scheduler != "distributed":
        n_threads = max(1, n_workers * threads_per_worker)
        dask.config.set(scheduler="threads", num_workers=n_threads)
        print(f"Using dask threaded scheduler with {n_threads} threads.", flush=True)
        return None, None

    if LocalCluster is None or Client is None:
        n_threads = max(1, n_workers * threads_per_worker)
        print(
            f"dask.distributed is unavailable; using threaded scheduler with {n_threads} threads.",
            flush=True,
        )
        dask.config.set(scheduler="threads", num_workers=n_threads)
        return None, None

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
        memory_limit=memory_limit,
        dashboard_address=":0",
    )
    client = Client(cluster)
    print(
        "Dask cluster:",
        f"{n_workers} workers x {threads_per_worker} threads",
        f"(dashboard: {client.dashboard_link})",
        flush=True,
    )
    return client, cluster


def seasonal_tasks(
    pr: xr.DataArray,
    u850: xr.DataArray,
    v850: xr.DataArray,
) -> tuple[list[str], list[xr.DataArray]]:
    names: list[str] = []
    tasks: list[xr.DataArray] = []

    for season in SEASONS:
        rain = pr.where(pr.time.dt.season == season, drop=True)
        names.append(f"rain_mean_{season}")
        tasks.append(rain.mean("time", skipna=True).astype("float32"))

    for season in SEASONS:
        rain = pr.where(pr.time.dt.season == season, drop=True).chunk({"time": -1})
        names.append(f"rain_p95_{season}")
        tasks.append(rain.quantile(0.95, dim="time", skipna=True).drop_vars("quantile").astype("float32"))
        names.append(f"rain_p99_{season}")
        tasks.append(rain.quantile(0.99, dim="time", skipna=True).drop_vars("quantile").astype("float32"))

    for season in SEASONS:
        names.append(f"u850_{season}")
        tasks.append(u850.where(u850.time.dt.season == season, drop=True).mean("time", skipna=True).astype("float32"))
        names.append(f"v850_{season}")
        tasks.append(v850.where(v850.time.dt.season == season, drop=True).mean("time", skipna=True).astype("float32"))

    return names, tasks


def compute_all(names: list[str], tasks: list[xr.DataArray], client: Client | None) -> dict[str, xr.DataArray]:
    print(f"Computing {len(tasks)} seasonal fields...", flush=True)
    if client is not None:
        futures = client.compute(tasks)
        if progress is not None:
            progress(futures)
        values = client.gather(futures)
    else:
        with ProgressBar():
            values = dask.compute(*tasks)
    return dict(zip(names, values, strict=True))


def open_inputs(args: argparse.Namespace) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, float]:
    pr_path = _existing_file(args.pr_path, "precipitation file")
    u_path = _existing_file(args.u_path, "u-wind file")
    v_path = _existing_file(args.v_path, "v-wind file")

    pr_ds = xr.open_dataset(
        pr_path,
        chunks={"time": 365, "lat": -1, "lon": -1},
        decode_times=True,
    )
    u_ds = xr.open_dataset(
        u_path,
        chunks={"time": 365, "lat": -1, "lon": -1},
        decode_times=True,
    )
    v_ds = xr.open_dataset(
        v_path,
        chunks={"time": 365, "lat": -1, "lon": -1},
        decode_times=True,
    )

    pr = _var(pr_ds, args.pr_var)
    u = _var(u_ds, args.u_var)
    v = _var(v_ds, args.v_var)

    data_domain = (
        args.data_lon_min,
        args.data_lon_max,
        args.data_lat_min,
        args.data_lat_max,
    )
    if all(value is not None for value in data_domain):
        pr = subset_region(pr, args.data_lon_min, args.data_lon_max, args.data_lat_min, args.data_lat_max)
        u = subset_region(u, args.data_lon_min, args.data_lon_max, args.data_lat_min, args.data_lat_max)
        v = subset_region(v, args.data_lon_min, args.data_lon_max, args.data_lat_min, args.data_lat_max)
        selected_domain = (
            f"lon {args.data_lon_min:g}-{args.data_lon_max:g}, "
            f"lat {args.data_lat_min:g}-{args.data_lat_max:g}"
        )
    elif any(value is not None for value in data_domain):
        raise ValueError("Set all four data domain options, or leave all of them as None.")
    else:
        selected_domain = "full input domains"

    pr = apply_time_range(pr, args.start_date, args.end_date)
    u = apply_time_range(u, args.start_date, args.end_date)
    v = apply_time_range(v, args.start_date, args.end_date)

    u = select_pressure_level(u, args.plev)
    v = select_pressure_level(v, args.plev)

    scale = choose_pr_scale(pr, args.pr_scale)
    pr = (pr * scale).assign_attrs(units="mm day-1")

    print(f"Precipitation: {pr_path}", flush=True)
    print(f"U850: {u_path}", flush=True)
    print(f"V850: {v_path}", flush=True)
    print(f"Precipitation grid: {grid_summary(pr)}", flush=True)
    print(f"U850 grid: {grid_summary(u)}", flush=True)
    print(f"V850 grid: {grid_summary(v)}", flush=True)
    print(f"Precipitation scale factor: {scale:g}", flush=True)
    print(
        "Computation/output domain:",
        selected_domain,
        flush=True,
    )
    print(
        "Plot domain:",
        f"lon {args.lon_min:g}-{args.lon_max:g}, lat {args.lat_min:g}-{args.lat_max:g}",
        flush=True,
    )
    return pr, u, v, scale


def _lon_lat(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    lat_name = _coord_name(da, ("lat", "latitude", "y"))
    lon_name = _coord_name(da, ("lon", "longitude", "x"))
    return da[lon_name], da[lat_name]


def _turn_off_cartopy_ticks(ax: plt.Axes, labelsize: float = 7.0, spine_lw: float = 1.5) -> None:
    ax.tick_params(labelsize=labelsize, direction="out", top=False, right=False)

    from matplotlib.lines import Line2D

    try:
        ax.spines["geo"].set_visible(False)
    except KeyError:
        ax.outline_patch.set_visible(False)

    plt.draw()
    ax.xaxis.set_tick_params(top=False, which="both")
    ax.yaxis.set_tick_params(right=False, which="both")
    ax.add_artist(
        Line2D(
            [0, 0],
            [0, 1],
            transform=ax.transAxes,
            color="black",
            linewidth=spine_lw,
            clip_on=False,
        )
    )
    ax.add_artist(
        Line2D(
            [0, 1],
            [0, 0],
            transform=ax.transAxes,
            color="black",
            linewidth=spine_lw,
            clip_on=False,
        )
    )


def format_map(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines("50m", linewidth=0.7)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.35,
        color="0.55",
        alpha=0.7,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = plt.FixedLocator([100, 110, 120, 130])
    gl.ylocator = plt.FixedLocator([-10, 0, 10, 20])
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    _turn_off_cartopy_ticks(ax, labelsize=7.0, spine_lw=1.5)


def add_region_boxes(ax: plt.Axes) -> None:
    transform = ccrs.PlateCarree()
    for label, (lon0, lon1, lat0, lat1) in REGION_BOXES.items():
        ax.plot(
            [lon0, lon1, lon1, lon0, lon0],
            [lat0, lat0, lat1, lat1, lat0],
            color="black",
            linewidth=1.15,
            transform=transform,
            zorder=5,
        )
        ax.text(
            lon0 + 0.6,
            lat1 - 1.2,
            label,
            transform=transform,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
            zorder=6,
        )


def plot_panel(
    ax: plt.Axes,
    rain: xr.DataArray,
    levels: np.ndarray,
    cmap: ListedColormap,
    title: str,
    label: str,
    extent: tuple[float, float, float, float],
    u: xr.DataArray | None = None,
    v: xr.DataArray | None = None,
    wind_step: int = 2,
) -> matplotlib.contour.QuadContourSet:
    lon, lat = _lon_lat(rain)
    norm = BoundaryNorm(levels, cmap.N, extend="both")
    cf = ax.contourf(
        lon,
        lat,
        rain,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    format_map(ax, extent)
    ax.set_title(title, fontsize=9, pad=3)
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.2},
    )

    if u is not None and v is not None:
        u_lon, u_lat = _lon_lat(u)
        q = ax.quiver(
            u_lon[::wind_step],
            u_lat[::wind_step],
            u.values[::wind_step, ::wind_step],
            v.values[::wind_step, ::wind_step],
            transform=ccrs.PlateCarree(),
            color="black",
            pivot="middle",
            width=0.0022,
            headwidth=3.2,
            headlength=4.0,
            headaxislength=3.5,
            scale=80,
            zorder=4,
        )
        simple_quiver_legend(
            ax,
            q,
            reference_value=5.0,
            unit="",
            legend_location="lower right",
            box_width=0.12,
            box_height=0.10,
            text_offset=0.020,
            font_size=7,
            label_separation=0.04,
            box_facecolor="white",
            box_edgecolor="none",
            box_linewidth=0.0,
            zorder=10,
        )
    return cf


def build_fields_dataset(fields: dict[str, xr.DataArray], scale: float, plev: float) -> xr.Dataset:
    rain_mean = xr.concat(
        [fields[f"rain_mean_{season}"] for season in SEASONS],
        dim=pd.Index(SEASONS, name="season"),
    )
    rain_p95 = xr.concat(
        [fields[f"rain_p95_{season}"] for season in SEASONS],
        dim=pd.Index(SEASONS, name="season"),
    )
    rain_p99 = xr.concat(
        [fields[f"rain_p99_{season}"] for season in SEASONS],
        dim=pd.Index(SEASONS, name="season"),
    )
    u850 = xr.concat(
        [fields[f"u850_{season}"].rename({"lat": "wind_lat", "lon": "wind_lon"}) for season in SEASONS],
        dim=pd.Index(SEASONS, name="season"),
    )
    v850 = xr.concat(
        [fields[f"v850_{season}"].rename({"lat": "wind_lat", "lon": "wind_lon"}) for season in SEASONS],
        dim=pd.Index(SEASONS, name="season"),
    )

    ds = xr.Dataset(
        {
            "rain_mean": rain_mean,
            "rain_p95": rain_p95,
            "rain_p99": rain_p99,
            "u850": u850,
            "v850": v850,
        }
    )
    ds["rain_mean"].attrs["units"] = "mm day-1"
    ds["rain_p95"].attrs["units"] = "mm day-1"
    ds["rain_p99"].attrs["units"] = "mm day-1"
    ds["u850"].attrs["units"] = "m s-1"
    ds["v850"].attrs["units"] = "m s-1"
    ds.attrs["precipitation_scale_factor"] = scale
    ds.attrs["selected_wind_plev_pa"] = plev
    return ds


def make_figure(
    fields: dict[str, xr.DataArray],
    args: argparse.Namespace,
    period_label: str,
) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    extent = (args.lon_min, args.lon_max, args.lat_min, args.lat_max)

    fig = plt.figure(figsize=(7.4, 11.1), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=5,
        ncols=2,
        height_ratios=[1.0, 1.0, 0.07, 1.0, 0.07],
        hspace=0.34,
        wspace=0.12,
    )

    axes = [
        fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[3, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[3, 1], projection=ccrs.PlateCarree()),
    ]

    mean_cf = None
    for idx, season in enumerate(SEASONS):
        mean_cf = plot_panel(
            axes[idx],
            fields[f"rain_mean_{season}"],
            MEAN_LEVELS,
            MEAN_CMAP,
            f"{season} mean",
            PANEL_LABELS[idx],
            extent,
            fields[f"u850_{season}"],
            fields[f"v850_{season}"],
            args.wind_step,
        )

    p95_cf = plot_panel(
        axes[4],
        fields["rain_p95_DJF"],
        P95_LEVELS,
        P95_CMAP,
        "DJF 95th percentile",
        PANEL_LABELS[4],
        extent,
    )
    if not args.no_region_boxes:
        add_region_boxes(axes[4])

    plot_panel(
        axes[5],
        fields["rain_p95_JJA"],
        P95_LEVELS,
        P95_CMAP,
        "JJA 95th percentile",
        PANEL_LABELS[5],
        extent,
    )

    cax1 = fig.add_subplot(gs[2, :])
    cbar1 = fig.colorbar(mean_cf, cax=cax1, orientation="horizontal", ticks=MEAN_LEVELS)
    cbar1.ax.tick_params(labelsize=7, length=2)
    cbar1.set_label("precipitation (mm day$^{-1}$)", fontsize=8)

    cax2 = fig.add_subplot(gs[4, :])
    cbar2 = fig.colorbar(p95_cf, cax=cax2, orientation="horizontal", ticks=P95_LEVELS)
    cbar2.ax.tick_params(labelsize=7, length=2)
    cbar2.set_label("precipitation (mm day$^{-1}$)", fontsize=8)

    title = args.title
    if title is None:
        title = f"Seasonal rainfall and 850 hPa winds, {period_label}"
    fig.suptitle(title, fontsize=11, y=0.982)
    fig.subplots_adjust(top=0.945, bottom=0.055, left=0.07, right=0.98)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def period_label(pr: xr.DataArray) -> str:
    time_values = pr.time.values
    start = pd.to_datetime(time_values[0]).strftime("%Y-%m-%d")
    end = pd.to_datetime(time_values[-1]).strftime("%Y-%m-%d")
    return f"{start} to {end}"


def run_from_config(args: argparse.Namespace) -> tuple[Path, Path | None]:
    warnings.filterwarnings("ignore", message="pyproj unable to set PROJ database path")
    warnings.filterwarnings("ignore", message=".*getfattr.*")

    client, cluster = make_cluster(
        args.scheduler,
        args.workers,
        args.threads_per_worker,
        args.memory_limit,
    )
    try:
        pr, u850, v850, scale = open_inputs(args)
        label = period_label(pr)
        names, tasks = seasonal_tasks(pr, u850, v850)
        fields = compute_all(names, tasks, client)

        fields_path = None
        if not args.no_save_fields:
            fields_path = args.fields_dir / args.fields_name
            args.fields_dir.mkdir(parents=True, exist_ok=True)
            fields_ds = build_fields_dataset(fields, scale, args.plev)
            fields_ds.to_netcdf(fields_path)
            print(f"Saved computed fields: {fields_path}", flush=True)
            p99_fields_path = args.fields_dir / args.p99_fields_name
            fields_ds[["rain_p99"]].to_netcdf(p99_fields_path)
            print(f"Saved 99th percentile fields: {p99_fields_path}", flush=True)

        figure_path = make_figure(fields, args, label)
        print(f"Saved figure: {figure_path}", flush=True)
        return figure_path, fields_path
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()


def main() -> None:
    args = parse_args()
    run_from_config(args)


if __name__ == "__main__":
    main()
