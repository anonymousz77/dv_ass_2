"""Download the Jul-Sep 2019 gridMET slice used for the SciVis tasks.

All files are fetched from the gridMET THREDDS NetCDF Subset Service (the same values as the yearly <var>_2019.nc
files on the gridMET direct-download page):

1. ``<var>_2019JAS_samples.nc`` - the 10 sampled days (every 10th day from 1 July) at the native 1/24 degree (~4 km)
   resolution, for tmmx, pr, vpd, vs and th. These drive all maps and GIFs.
2. ``<var>_2019JAS_daily_stride4.nc`` - all 92 days at every 4th grid cell (~16 km) for tmmx, vpd and vs. Daily
   context only (domain means and percentiles, which a 16 km subsample estimates well).
3. ``pr_2019JAS_daily.nc`` - all 92 days of precipitation at full resolution, because storm maxima are a few cells
   wide and a 16 km subsample misses them (e.g. Imelda 528 instead of 707 mm on 18 Sep).
4. ``<var>_2019_storms_4km.nc`` - vs, th and pr at full resolution over the Gulf and south-east coast
   (24.5-38 N, 100-74 W) for the storm days that fall between the samples: 12-15 Jul (Hurricane Barry),
   4-7 Sep (Hurricane Dorian) and 16-20 Sep (Tropical Storm Imelda).

Products 3 and 4 are requested in pieces and stored with zlib compression (the server returns uncompressed NetCDF-3).

Usage:  python code/data_prep/download_gridmet.py
"""
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

NCSS = "http://thredds.northwestknowledge.net:8080/thredds/ncss/"
DAILY_CONTEXT = ["tmmx", "vpd", "vs"]
MONTHS = [("2019-07-01", "2019-07-31"), ("2019-08-01", "2019-08-31"), ("2019-09-01", "2019-09-30")]
STORM_BOX = dict(north=38, south=24.5, west=-100, east=-74)
STORM_WINDOWS = [("2019-07-12", "2019-07-15"), ("2019-09-04", "2019-09-07"), ("2019-09-16", "2019-09-20")]


def present(out: Path) -> bool:
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name} already present")
        return True
    return False


def fetch(url: str, out: Path, retries: int = 3) -> None:
    if present(out):
        return
    tmp = out.with_suffix(".part")
    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            with urllib.request.urlopen(url, timeout=900) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(out)
            print(f"[ok] {out.name}: {out.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.0f}s")
            return
        except Exception as exc:  # network hiccups on the public server
            print(f"[retry {attempt}] {out.name}: {exc}")
            time.sleep(5)
    raise RuntimeError(f"failed to download {url}")


def fetch_merged(urls: list[str], var: str, out: Path) -> None:
    """Download several time pieces, join them along the day axis and store compressed with the original packing."""
    if present(out):
        return
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for k, url in enumerate(urls):
            fetch(url, Path(tmp) / f"part{k}.nc")
            parts.append(xr.open_dataset(Path(tmp) / f"part{k}.nc").load())
        ds = xr.concat(parts, dim="day")
        enc = {k: v for k, v in parts[0][var].encoding.items() if k in ("dtype", "scale_factor", "add_offset",
                                                                        "_FillValue", "_Unsigned")}
        ds.to_netcdf(out, encoding={var: {**enc, "zlib": True, "complevel": 5}})
        for p in parts:
            p.close()
    print(f"[ok] {out.name}: {out.stat().st_size / 1e6:.1f} MB ({ds.sizes['day']} days)")


def url(agg: str, var: str, start: str, end: str, **extra) -> str:
    q = "&".join(f"{k}={v}" for k, v in extra.items())
    return (f"{NCSS}{agg}?var={var}&accept=netcdf&time_start={start}T00:00:00Z&time_end={end}T00:00:00Z"
            + (f"&{q}" if q else ""))


def main() -> None:
    config.GRIDMET_RAW.mkdir(parents=True, exist_ok=True)
    for short, (agg, var, _units) in config.GRIDMET_VARS.items():
        fetch(url(agg, var, config.SAMPLE_DATES[0], config.SAMPLE_DATES[-1], horizStride=1,
                  timeStride=config.SAMPLE_STRIDE_DAYS), config.GRIDMET_RAW / f"{short}_2019JAS_samples.nc")
        if short in DAILY_CONTEXT:
            fetch(url(agg, var, config.PERIOD_START, config.PERIOD_END, horizStride=4),
                  config.GRIDMET_RAW / f"{short}_2019JAS_daily_stride4.nc")
        if short == "pr":
            fetch_merged([url(agg, var, a, b, horizStride=1) for a, b in MONTHS], var,
                         config.GRIDMET_RAW / "pr_2019JAS_daily.nc")
        if short in ("vs", "th", "pr"):
            fetch_merged([url(agg, var, a, b, horizStride=1, **STORM_BOX) for a, b in STORM_WINDOWS], var,
                         config.GRIDMET_RAW / f"{short}_2019_storms_4km.nc")


if __name__ == "__main__":
    main()
