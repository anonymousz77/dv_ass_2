"""Shared helpers for the SciVis tasks: loading the gridMET slice, map axes, saving figures and GIFs."""
import io
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

PC = ccrs.PlateCarree()
# Lambert conformal conic: the standard conformal projection for CONUS, so wind directions are preserved.
PROJ = ccrs.LambertConformal(central_longitude=-96, central_latitude=39, standard_parallels=(33, 45))
EXTENT = [-120.5, -73.0, 22.8, 50.2]

LABELS = {
    "tmmx": ("Daily maximum temperature", "°C"),
    "pr": ("Daily precipitation", "mm/day"),
    "vpd": ("Daily mean vapour pressure deficit", "kPa"),
    "vs": ("Daily mean wind speed (10 m)", "m/s"),
    "th": ("Daily mean wind direction (10 m, from)", "° from N"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.titlesize": 12, "figure.titleweight": "bold", "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})


def load(var: str, kind: str = "samples") -> xr.DataArray:
    """Load one gridMET variable for the Jul-Sep 2019 slice in display units (temperature in °C)."""
    ds = xr.open_dataset(config.GRIDMET_RAW / f"{var}_2019JAS_{kind}.nc")
    da = ds[list(ds.data_vars)[0]].load()
    if var in ("tmmx", "tmmn"):
        da = da - 273.15
    da.name = var
    return da


def wind_components(speed: np.ndarray, direction_from: np.ndarray):
    """Meteorological convention: direction the wind blows FROM, clockwise from north.
    u (eastward) = -s*sin(theta), v (northward) = -s*cos(theta)."""
    rad = np.deg2rad(direction_from)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def geo_vectors(u, v, lat):
    """Convert (east, north) components in m/s into the form cartopy expects for vectors given in PlateCarree.

    cartopy treats such vectors as rates of change of (longitude, latitude) in degrees, and one degree of longitude
    is only cos(lat) times as long as one degree of latitude. The eastward component is therefore divided by cos(lat)
    (otherwise diagonal winds are rotated towards the meridian, by ~10° at 45° N), and the pair is rescaled to the
    true speed, because cartopy keeps the magnitude of the vectors it rotates into the projection."""
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    ue = u / np.cos(np.deg2rad(lat))
    speed, raw = np.hypot(u, v), np.hypot(ue, v)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(raw > 0, speed / raw, 0.0)
    return ue * f, v * f


def date_label(d) -> str:
    return pd.Timestamp(d).strftime("%d %b %Y")


_PROJ_EXTENT = None


def conus_extent():
    """Bounding box (in projected metres) of the valid gridMET cells (every 4th cell is enough), plus a small margin."""
    global _PROJ_EXTENT
    if _PROJ_EXTENT is None:
        ds = xr.open_dataset(config.GRIDMET_RAW / "tmmx_2019JAS_samples.nc")
        da = ds[list(ds.data_vars)[0]].isel(day=0, lat=slice(None, None, 4), lon=slice(None, None, 4))
        lon2, lat2 = np.meshgrid(da.lon.values, da.lat.values)
        ok = np.isfinite(da.values)
        xy = PROJ.transform_points(PC, lon2[ok], lat2[ok])
        x0, x1, y0, y1 = xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()
        mx, my = 0.012 * (x1 - x0), 0.02 * (y1 - y0)
        _PROJ_EXTENT = [x0 - mx, x1 + mx, y0 - my, y1 + my]
    return _PROJ_EXTENT


def add_features(ax, lw: float = 0.35, color: str = "#222222", ocean: bool = True, extent=None) -> None:
    if ocean:
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#e9edf1", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f4f4f4", zorder=0)
    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor=color, linewidth=lw * 0.7, facecolor="none", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=color, linewidth=lw * 1.4, zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=color, linewidth=lw * 1.4, zorder=3)
    if extent is None:
        ax.set_extent(conus_extent(), crs=PROJ)
    else:
        ax.set_extent(extent, crs=PC)


def map_axes(fig, *args, extent=None, **kw):
    ax = fig.add_subplot(*args, projection=PROJ, **kw)
    add_features(ax, extent=extent)
    return ax


def gridlines(ax, labels: bool = False) -> None:
    gl = ax.gridlines(crs=PC, draw_labels=labels, linewidth=0.3, color="#888888", alpha=0.5,
                      linestyle=":", x_inline=False, y_inline=False)
    if labels:
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 7}


def mesh(ax, da: xr.DataArray, **kw):
    return ax.pcolormesh(da.lon.values, da.lat.values, np.ma.masked_invalid(da.values), transform=PC,
                         shading="nearest", rasterized=True, **kw)


def save(fig, name: str, dpi: int = config.FIG_DPI) -> Path:
    config.IMAGES.mkdir(parents=True, exist_ok=True)
    out = config.IMAGES / f"{name}.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print("saved", out.name)
    return out


def fig_to_image(fig, dpi: int = 100) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_gif(frames, name: str, ms: int = 900) -> Path:
    """Write frames (PIL images) to an animated GIF that loops forever."""
    out = config.IMAGES / f"{name}.gif"
    w = max(f.width for f in frames)
    h = max(f.height for f in frames)
    norm = []
    for f in frames:
        canvas = Image.new("RGB", (w, h), "white")
        canvas.paste(f, (0, 0))
        norm.append(canvas.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    norm[0].save(out, save_all=True, append_images=norm[1:], duration=ms, loop=0, optimize=True)
    print("saved", out.name, f"({len(frames)} frames)")
    return out


def stamp(fig, text: str) -> None:
    fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=6.5, color="#555555")


SOURCE = "Data: gridMET (Abatzoglou, 2013), 1/24° grid, Jul–Sep 2019"
