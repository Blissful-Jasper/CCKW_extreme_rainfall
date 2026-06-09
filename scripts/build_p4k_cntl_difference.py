#!/usr/bin/env python
"""Build project-local P4K minus CNTL seasonal rainfall and wind differences."""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

REQUIRED_DATA_VARS = ("rain_mean", "rain_p95", "rain_p99", "u850", "v850")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create P4K-CNTL difference fields from seasonal rainfall/wind caches."
    )
    parser.add_argument("--cntl-fields", type=Path, required=True)
    parser.add_argument("--p4k-fields", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cntl-p99-output", type=Path, default=None)
    parser.add_argument("--p4k-p99-output", type=Path, default=None)
    return parser.parse_args()


def _check(ds: xr.Dataset, path: Path) -> None:
    missing = [name for name in REQUIRED_DATA_VARS if name not in ds]
    if missing:
        raise KeyError(f"{path} is missing variables: {missing}")


def main() -> None:
    args = parse_args()
    cntl = xr.open_dataset(args.cntl_fields)
    p4k = xr.open_dataset(args.p4k_fields)
    _check(cntl, args.cntl_fields)
    _check(p4k, args.p4k_fields)

    diff = xr.Dataset()
    for var in REQUIRED_DATA_VARS:
        warm, base = xr.align(p4k[var], cntl[var], join="inner")
        diff[var] = (warm - base).astype("float32")
        diff[var].attrs["units"] = p4k[var].attrs.get("units", "")
        diff[var].attrs["description"] = f"P4K minus CNTL {var}"

    diff.attrs["description"] = "Seasonal P4K minus CNTL rainfall and 850 hPa wind differences"
    diff.attrs["difference"] = "P4K - CNTL"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    diff.to_netcdf(args.output)
    print(f"Saved P4K-CNTL fields: {args.output}")

    if args.cntl_p99_output is not None:
        args.cntl_p99_output.parent.mkdir(parents=True, exist_ok=True)
        cntl[["rain_p99"]].to_netcdf(args.cntl_p99_output)
        print(f"Saved CNTL P99 fields: {args.cntl_p99_output}")

    if args.p4k_p99_output is not None:
        args.p4k_p99_output.parent.mkdir(parents=True, exist_ok=True)
        p4k[["rain_p99"]].to_netcdf(args.p4k_p99_output)
        print(f"Saved P4K P99 fields: {args.p4k_p99_output}")


if __name__ == "__main__":
    main()
