#!/usr/bin/env python3
"""Write clearly named single-level 850 hPa wind files.

The shared pressure-interpolation script writes files named like
``ua_pressure_levels_cntl.nc`` because it can handle many pressure levels. This
project only uses one level, 850 hPa, for the seasonal wind overlays. This
helper turns those intermediate files into explicit single-level files such as
``ua_850hpa_cntl_lat30.nc``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr


PROJECT_DIR = Path("/work/mh1498/m301257/code_extreme_event")
DEFAULT_INPUT_DIR = PROJECT_DIR / "data/pressure_levels_lat30"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data/wind_850hpa_lat30"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert pressure-level wind outputs to explicit single-level 850 hPa files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variables", nargs="+", default=["ua", "va"])
    parser.add_argument("--experiments", nargs="+", default=["cntl", "p4k"])
    parser.add_argument("--plev", type=float, default=850.0)
    parser.add_argument("--lat-label", default="lat30")
    return parser


def _source_file(input_dir: Path, var_name: str, exp: str) -> Path:
    return input_dir / f"{var_name}_pressure_levels_{exp}.nc"


def _target_file(output_dir: Path, var_name: str, exp: str, plev: float, lat_label: str) -> Path:
    plev_label = f"{plev:g}".replace(".", "p")
    return output_dir / f"{var_name}_{plev_label}hpa_{exp}_{lat_label}.nc"


def _get_data_var(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    if var_name in ds:
        return ds[var_name]
    if len(ds.data_vars) == 1:
        return ds[next(iter(ds.data_vars))].rename(var_name)
    raise KeyError(f"{var_name!r} not found; available variables: {list(ds.data_vars)}")


def standardize_file(source: Path, target: Path, var_name: str, plev: float) -> None:
    if not source.exists():
        raise FileNotFoundError(source)

    with xr.open_dataset(source) as ds:
        da = _get_data_var(ds, var_name)
        if "plev" in da.coords or "plev" in da.dims:
            da = da.sel(plev=plev, method="nearest").squeeze(drop=True)
            da = da.drop_vars("plev", errors="ignore")
        da = da.rename(var_name)
        da.attrs = dict(da.attrs)
        da.attrs["selected_pressure_level_hpa"] = float(plev)
        da.attrs["description"] = f"{var_name} wind interpolated to {plev:g} hPa"
        out = da.to_dataset(name=var_name)
        out.attrs["source_file"] = str(source)
        out.attrs["selected_pressure_level_hpa"] = float(plev)

        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_netcdf(target)
    print(f"Saved {target}")


def main() -> None:
    args = build_parser().parse_args()
    for var_name in args.variables:
        for exp in args.experiments:
            source = _source_file(args.input_dir, var_name, exp)
            target = _target_file(args.output_dir, var_name, exp, args.plev, args.lat_label)
            standardize_file(source, target, var_name, args.plev)


if __name__ == "__main__":
    main()
