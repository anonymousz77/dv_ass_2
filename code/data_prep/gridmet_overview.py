"""Daily overview of the Jul-Sep 2019 gridMET slice and the 10 uniformly sampled time instances (Fig. 1).

Shows how the sampled days sit within the full quarter and writes the per-day domain statistics to
data/processed/gridmet_daily_stats_JAS2019.csv. Temperature, VPD and wind use the 92-day, every-4th-cell (~16 km)
subset, whose means and percentiles match the full grid; precipitation uses the full 4 km grid, because the daily
maximum of a storm is only a few cells wide and a 16 km subsample underestimates it (median 12 %, up to 50 %).

Usage:  python code/data_prep/gridmet_overview.py
"""
import sys
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scivis.common import LABELS, load, plt, save  # noqa: E402

EVENTS = [  # verified against the data: location/date of the domain maxima printed in the cleaning log
    ("2019-07-13", "2019-07-15", "Hurricane Barry\n(LA/AR rain)"),
    ("2019-07-18", "2019-07-21", "Central/Eastern\nheat wave"),
    ("2019-09-05", "2019-09-06", "Hurricane Dorian\n(Carolinas)"),
    ("2019-09-17", "2019-09-19", "TS Imelda\n(SE Texas rain)"),
    ("2019-09-27", "2019-09-30", "N. Rockies early\nsnow & cold"),
]


def weighted_mean(da):
    w = np.cos(np.deg2rad(da.lat))
    return da.weighted(w.fillna(0)).mean(dim=("lat", "lon"))


def daily_precip_stats() -> pd.DataFrame:
    """Daily precipitation statistics on the full 4 km grid, one day at a time to keep memory low."""
    ds = xr.open_dataset(config.GRIDMET_RAW / "pr_2019JAS_daily.nc")
    da = ds[list(ds.data_vars)[0]]
    w = np.cos(np.deg2rad(da.lat.values))[:, None]
    rows = []
    for d in da.day.values:
        f = da.sel(day=d).values
        ok = np.isfinite(f)
        i, j = np.unravel_index(np.nanargmax(f), f.shape)
        rows.append(dict(date=pd.Timestamp(d), pr_mean=(np.where(ok, f, 0) * w).sum() / (ok * w).sum(),
                         pr_max=f[i, j], pr_max_lat=float(da.lat[i]), pr_max_lon=float(da.lon[j]),
                         pr_wet_frac=(f > 1).sum() / ok.sum()))
    return pd.DataFrame(rows).set_index("date")


def main() -> None:
    stats = {}
    t = load("tmmx", "daily_stride4")
    stats["tmmx_mean"] = weighted_mean(t)
    stats["tmmx_p05"] = t.quantile(0.05, dim=("lat", "lon"))
    stats["tmmx_p95"] = t.quantile(0.95, dim=("lat", "lon"))
    v = load("vpd", "daily_stride4")
    stats["vpd_mean"] = weighted_mean(v)
    stats["vpd_p95"] = v.quantile(0.95, dim=("lat", "lon"))
    s = load("vs", "daily_stride4")
    stats["vs_mean"] = weighted_mean(s)
    stats["vs_p95"] = s.quantile(0.95, dim=("lat", "lon"))
    df = pd.DataFrame({k: val.to_series() for k, val in stats.items()})
    df.index.name = "date"
    pr = daily_precip_stats()
    pr.index = df.index
    df = pd.concat([df, pr], axis=1)
    df.drop(columns=[c for c in df.columns if c == "quantile"], errors="ignore").round(3).to_csv(
        config.DATA_PROCESSED / "gridmet_daily_stats_JAS2019.csv")

    samples = pd.to_datetime(config.SAMPLE_DATES)
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    x = df.index
    ax = axes[0]
    ax.fill_between(x, df.tmmx_p05, df.tmmx_p95, color="#f4a582", alpha=0.45, label="5th–95th percentile of grid cells")
    ax.plot(x, df.tmmx_mean, color="#b2182b", lw=1.6, label="area-weighted CONUS mean")
    ax.set_ylabel("Max. temperature (°C)")
    ax = axes[1]
    ax.bar(x, df.pr_mean, color="#2166ac", width=0.8, label="area-weighted CONUS mean")
    ax.set_ylabel("Precipitation mean\n(mm/day)")
    ax2 = ax.twinx()
    ax2.plot(x, df.pr_max, color="#542788", lw=1.1, marker=".", ms=3, label="4 km grid-cell maximum (right axis)")
    ax2.set_ylabel("Grid-cell max\n(mm/day)", color="#542788")
    ax2.tick_params(axis="y", colors="#542788")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.5, frameon=False)
    ax = axes[2]
    ax.plot(x, df.vpd_mean, color="#8c510a", lw=1.6, label="area-weighted CONUS mean")
    ax.plot(x, df.vpd_p95, color="#d8b365", lw=1.2, ls="--", label="95th percentile")
    ax.set_ylabel("Vapour pressure\ndeficit (kPa)")
    ax = axes[3]
    ax.plot(x, df.vs_mean, color="#01665e", lw=1.6, label="area-weighted CONUS mean")
    ax.plot(x, df.vs_p95, color="#5ab4ac", lw=1.2, ls="--", label="95th percentile")
    ax.set_ylabel("Wind speed\n(m/s)")
    for i, ax in enumerate(axes):
        for d in samples:
            ax.axvline(d, color="#333333", lw=0.8, ls=":", zorder=0)
        for a, b, lab in EVENTS:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1), color="#999999", alpha=0.15, lw=0)
        if i != 1:
            ax.legend(loc="upper right" if i == 0 else "upper left", fontsize=7.5, frameon=False)
        ax.grid(axis="y", lw=0.3, alpha=0.6)
    for k, (a, b, lab) in enumerate(EVENTS):
        mid = pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2 + pd.Timedelta(hours=12)
        axes[0].text(mid, 1.02 + 0.2 * (k % 2), lab.replace("\n", " ") if k % 2 else lab,
                     transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=7.5,
                     bbox=dict(boxstyle="round,pad=0.15", fc="#eeeeee", ec="none"))
    axes[3].set_xticks(samples)
    axes[3].set_xticklabels([f"t{i + 1}\n{d.strftime('%d %b')}" for i, d in enumerate(samples)], fontsize=7.5)
    axes[3].set_xlabel("Sampled time instances t1–t10 (2019); dotted lines mark them in every panel")
    fig.suptitle("gridMET daily CONUS statistics, 1 Jul – 30 Sep 2019, and the 10 uniformly sampled days (every 10 days)",
                 y=0.995)
    fig.tight_layout(rect=(0, 0.01, 1, 0.94))
    fig.text(0.995, 0.0, "Shaded bands: notable weather events. All 92 days; precipitation from the full 4 km grid, the other "
                         "variables from every 4th grid cell (~16 km).",
             ha="right", fontsize=7, color="#555555")
    save(fig, "Fig01_gridmet_daily_overview_sampling")


if __name__ == "__main__":
    main()
