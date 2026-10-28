"""SciVis Task 3 - Quiver plots and streamlines (Arnav, BT2024233).

Vector field: gridMET daily mean wind speed (vs, m/s) and wind direction (th, degrees clockwise from north,
direction the wind blows FROM), 10 uniformly sampled days of Jul-Sep 2019.
u = -vs*sin(th) (eastward), v = -vs*cos(th) (northward). Maps use a Lambert conformal projection so that
arrow directions are preserved. Before cartopy rotates the vectors into the projection they are converted with
geo_vectors() (eastward component / cos(latitude), true speed restored), because cartopy reads vectors given in
PlateCarree as degrees of longitude and latitude per unit time.

Figures written to images/:
  Fig16  grid sampling experiments: regular strides, block-averaged vectors, jittered sampling, aliasing error
  Fig17  magnitude-proportional vs same-size (direction-only) vs square-root scaled arrows
  Fig18  streamline seeding strategies: density-based, uniform grid, random, feature-based (high speed)
  Fig19  quiver vs streamlines in a strong-flow (29 Sep) and a weak-flow (21 Jul) situation
  Fig20  quiver small multiples for all 10 time steps (+ GIF)
  Fig21  streamline small multiples for all 10 time steps (+ GIF)
  Fig22  storm close-ups at 4 km on days between the samples: Hurricane Barry, Hurricane Dorian, Tropical Storm Imelda

Usage:  python code/scivis/quiver_streamlines.py
"""
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.vector_transform import vector_scalar_to_grid
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scivis.common import (PC, PROJ, SOURCE, conus_extent, date_label, fig_to_image, geo_vectors, load, map_axes,  # noqa: E402
                           mesh, plt, save, save_gif, stamp, wind_components)

SPEED_LABEL = "Wind speed at 10 m (m/s)"
SPEED_NORM = mcolors.Normalize(0, 12)
BG_CMAP = "YlGnBu"
CELL_M = 4000.0    # approximate gridMET cell size in metres (1/24 degree)
REF_SPEED = 10.0   # a 10 m/s arrow spans one sampling interval
REF_SPEED_KEY = 10.0
EARTH_R = 6371000.0
rng = np.random.default_rng(732)
log = []


def fields(vs, th, i):
    s = vs.isel(day=i)
    u, v = wind_components(s.values, th.isel(day=i).values)
    return s, u, v


def background(ax, s, alpha=0.75):
    return mesh(ax, s, cmap=BG_CMAP, norm=SPEED_NORM, alpha=alpha)


def subsample(s, u, v, k, offset=None):
    o = k // 2 if offset is None else offset
    lon, lat = np.meshgrid(s.lon.values[o::k], s.lat.values[o::k])
    return lon, lat, u[o::k, o::k], v[o::k, o::k]


def block_mean(s, u, v, k):
    ny, nx = (u.shape[0] // k) * k, (u.shape[1] // k) * k
    ub = np.nanmean(u[:ny, :nx].reshape(ny // k, k, nx // k, k), axis=(1, 3))
    vb = np.nanmean(v[:ny, :nx].reshape(ny // k, k, nx // k, k), axis=(1, 3))
    lat = s.lat.values[:ny].reshape(ny // k, k).mean(1)
    lon = s.lon.values[:nx].reshape(nx // k, k).mean(1)
    frac = np.isfinite(u[:ny, :nx]).reshape(ny // k, k, nx // k, k).mean(axis=(1, 3))
    ub[frac < 0.5] = np.nan     # blocks that are mostly ocean are dropped
    vb[frac < 0.5] = np.nan
    lon2, lat2 = np.meshgrid(lon, lat)
    return lon2, lat2, ub, vb


def jittered(s, u, v, k):
    ny, nx = u.shape
    rows, cols = [], []
    for i0 in range(0, ny - k + 1, k):
        for j0 in range(0, nx - k + 1, k):
            rows.append(i0 + rng.integers(k))
            cols.append(j0 + rng.integers(k))
    rows, cols = np.array(rows), np.array(cols)
    return s.lon.values[cols], s.lat.values[rows], u[rows, cols], v[rows, cols]


def quiver(ax, lon, lat, u, v, spacing_cells, mode="proportional", color="black", width=0.0022, **kw):
    """Draw arrows in projected (metre) units so that arrow length is tied to the sampling spacing:
    a 10 m/s arrow spans one grid spacing (proportional), unit arrows span 0.75 spacing (uniform)."""
    m = np.isfinite(u) & np.isfinite(v)
    lon, lat, u, v = np.asarray(lon)[m], np.asarray(lat)[m], u[m], v[m]
    u, v = geo_vectors(u, v, lat)   # speed is unchanged, only the direction convention
    spd = np.hypot(u, v)
    spacing_m = spacing_cells * CELL_M
    common = dict(transform=PC, scale_units="xy", angles="xy", width=width, zorder=5, **kw)
    if mode == "proportional":
        q = ax.quiver(lon, lat, u, v, color=color, scale=REF_SPEED / spacing_m, **common)
    elif mode == "uniform":
        q = ax.quiver(lon, lat, u / spd, v / spd, spd, cmap="viridis", norm=SPEED_NORM, scale=1 / (0.75 * spacing_m),
                      **common)
    elif mode == "sqrt":
        f = np.sqrt(spd) / spd
        q = ax.quiver(lon, lat, u * f, v * f, color=color, scale=np.sqrt(REF_SPEED) / spacing_m, **common)
    else:
        raise ValueError(mode)
    return q, int(m.sum())


def quiver_key(ax, q, value=REF_SPEED_KEY, label=None):
    ax.quiverkey(q, 0.84, 0.045, value, label or f"{value:g} m/s", labelpos="E", coordinates="axes",
                 fontproperties={"size": 8}, color="black")


def angle_diff_deg(u1, v1, u2, v2):
    a = np.degrees(np.arctan2(v1, u1) - np.arctan2(v2, u2))
    return (a + 180) % 360 - 180


# --- regridding for streamlines --------------------------------------------------------------------------------
def proj_grid(s, u, v, nx=360, extent=None, stride=2):
    """Interpolate the (east, north) wind onto a regular grid in projected coordinates, rotating the vectors.
    Points far from any valid land cell (ocean, lakes, outside CONUS) are masked."""
    lon2, lat2 = np.meshgrid(s.lon.values[::stride], s.lat.values[::stride])
    uu, vv, ss = u[::stride, ::stride], v[::stride, ::stride], s.values[::stride, ::stride]
    m = np.isfinite(uu) & np.isfinite(vv)
    ext = extent if extent is not None else conus_extent()
    ny = int(nx * (ext[3] - ext[2]) / (ext[1] - ext[0]))
    ue, ve = geo_vectors(uu[m], vv[m], lat2[m])
    xg, yg, ug, vg, sg = vector_scalar_to_grid(PC, PROJ, (nx, ny), lon2[m], lat2[m], ue, ve, ss[m],
                                               target_extent=ext)
    pts = PROJ.transform_points(PC, lon2[m], lat2[m])[:, :2]
    dist, _ = cKDTree(pts).query(np.column_stack([xg.ravel(), yg.ravel()]))
    far = (dist > 9000 * stride).reshape(xg.shape)
    for a in (ug, vg, sg):
        a[far] = np.nan
    return xg, yg, ug, vg, sg


def streamlines(ax, grid, start_points=None, density=1.6, lw_scale=True, color_by_speed=True, cmap="plasma",
                max_length=4.0):
    xg, yg, ug, vg, sg = grid
    kw = dict(density=density, arrowsize=0.8, arrowstyle="->", zorder=5, integration_direction="both",
              broken_streamlines=True, maxlength=max_length)
    if lw_scale:
        kw["linewidth"] = 0.4 + 1.6 * np.clip(np.nan_to_num(sg) / 12, 0, 1)
    if color_by_speed:
        kw["color"] = np.nan_to_num(sg)
        kw["cmap"] = cmap
        kw["norm"] = SPEED_NORM
    else:
        kw["color"] = "#222222"
    if start_points is not None:
        kw["start_points"] = start_points
    return Axes.streamplot(ax, xg[0, :], yg[:, 0], ug, vg, **kw)


def coverage(sp, grid, cell=60000):
    """Share of land covered by streamlines: fraction of 60 km land cells containing at least one streamline vertex."""
    xg, yg, ug, _, _ = grid
    segs = sp.lines.get_segments()
    if not segs:
        return 0.0, 0
    pts = np.concatenate(segs)
    land = np.isfinite(ug)
    cx = ((xg[land] - xg.min()) // cell).astype(int)
    cy = ((yg[land] - yg.min()) // cell).astype(int)
    land_cells = set(zip(cx, cy))
    px = ((pts[:, 0] - xg.min()) // cell).astype(int)
    py = ((pts[:, 1] - yg.min()) // cell).astype(int)
    hit = set(zip(px, py)) & land_cells
    return len(hit) / len(land_cells), len(segs)


def seeds_to_proj(lon, lat):
    return PROJ.transform_points(PC, np.asarray(lon), np.asarray(lat))[:, :2]


# ---------------------------------------------------------------------------------------------------------------
def fig16_sampling(vs, th):
    i = 9
    s, u, v = fields(vs, th, i)
    fig = plt.figure(figsize=(17, 9.6))
    panels = []
    for k, stride in enumerate([10, 20, 40]):
        ax = map_axes(fig, 2, 3, k + 1)
        background(ax, s)
        lon, lat, uu, vv = subsample(s, u, v, stride)
        q, n = quiver(ax, lon, lat, uu, vv, stride, width=0.0016 if stride == 10 else 0.0022)
        quiver_key(ax, q)
        ax.set_title(f"({'abc'[k]}) Regular point sampling, every {stride}th cell\n"
                     f"(~{stride * 4} km spacing, {n:,} arrows)", fontsize=9.5)
        panels.append(q)
    ax = map_axes(fig, 2, 3, 4)
    background(ax, s)
    lon, lat, ub, vb = block_mean(s, u, v, 20)
    q, n = quiver(ax, lon, lat, ub, vb, 20)
    quiver_key(ax, q)
    ax.set_title(f"(d) Block-averaged vectors, 20 × 20 cells\n(mean u, v per ~80 km block, {n:,} arrows)", fontsize=9.5)
    ax = map_axes(fig, 2, 3, 5)
    background(ax, s)
    lon, lat, uj, vj = jittered(s, u, v, 20)
    q, n = quiver(ax, lon, lat, uj, vj, 20)
    quiver_key(ax, q)
    ax.set_title(f"(e) Jittered random sampling, one random cell per\n20 × 20 block ({n:,} arrows)", fontsize=9.5)

    # aliasing: regular point sample vs block mean at the same 20-cell spacing
    _, _, up, vp = subsample(s, u, v, 20, offset=10)
    ny, nx = ub.shape
    up, vp = up[:ny, :nx], vp[:ny, :nx]
    m = np.isfinite(up) & np.isfinite(ub)
    dang = angle_diff_deg(up[m], vp[m], ub[m], vb[m])
    ratio = np.hypot(ub[m], vb[m]) / np.hypot(up[m], vp[m])
    sm = s.values[:ny * 20, :nx * 20].reshape(ny, 20, nx, 20)
    steadiness = np.hypot(ub, vb) / np.nanmean(sm, axis=(1, 3))
    ax = fig.add_subplot(2, 3, 6)
    ax.hist(np.abs(dang), bins=np.arange(0, 61, 2.5), color="#41b6c4", edgecolor="white")
    ax.set_xlim(0, 60)
    ax.axvline(np.median(np.abs(dang)), color="#253494", ls="--")
    ax.set_xlabel("|direction difference|, point sample vs block mean (°)")
    ax.set_ylabel("Number of 80 km blocks")
    ax.set_title(f"(f) Aliasing of point sampling (stride 20):\nmedian {np.median(np.abs(dang)):.0f}°, "
                 f"{np.mean(np.abs(dang) > 45):.0%} of blocks differ by > 45°", fontsize=9.5)
    ax.text(0.97, 0.95, f"block-mean speed / point speed:\nmedian {np.median(ratio):.2f}\n"
                        f"vector-mean / scalar-mean speed:\nmedian {np.nanmedian(steadiness):.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(fc="white", ec="#cccccc"))
    ax.grid(lw=0.3, alpha=0.6)
    cax = fig.add_axes([0.06, 0.065, 0.55, 0.018])
    cb = fig.colorbar(plt.cm.ScalarMappable(SPEED_NORM, BG_CMAP), cax=cax, orientation="horizontal", extend="max")
    cb.set_label(SPEED_LABEL + " (background; magnitude-proportional arrows: a 10 m/s arrow spans one sampling interval)")
    fig.suptitle(f"Quiver grid-sampling experiments, t10 = {date_label(s.day.values)} (strong autumn trough over the West)",
                 y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.12, wspace=0.06, hspace=0.18)
    stamp(fig, SOURCE)
    save(fig, "Fig16_quiver_grid_sampling_experiments")
    log.append(f"Fig16 aliasing stride20: median |dangle| {np.median(np.abs(dang)):.1f}, >45deg {np.mean(np.abs(dang) > 45):.3f}, "
               f"median speed ratio {np.median(ratio):.2f}, median steadiness {np.nanmedian(steadiness):.2f}, n {m.sum()}")


def fig17_magnitude(vs, th):
    i = 9
    s, u, v = fields(vs, th, i)
    lon, lat, uu, vv = subsample(s, u, v, 20)
    fig = plt.figure(figsize=(17, 5.6))
    ax = map_axes(fig, 1, 3, 1)
    background(ax, s, alpha=0.6)
    q, n = quiver(ax, lon, lat, uu, vv, 20)
    quiver_key(ax, q)
    ax.set_title("(a) Magnitude-proportional arrows\n(length ∝ speed)", fontsize=10)
    ax = map_axes(fig, 1, 3, 2)
    q, _ = quiver(ax, lon, lat, uu, vv, 20, mode="uniform", width=0.0028)
    cb = fig.colorbar(q, ax=ax, orientation="horizontal", pad=0.03, fraction=0.05, aspect=35, extend="max")
    cb.set_label(SPEED_LABEL + " (arrow colour)")
    ax.set_title("(b) Same-size arrows (direction only),\nspeed encoded by colour", fontsize=10)
    ax = map_axes(fig, 1, 3, 3)
    background(ax, s, alpha=0.6)
    q, _ = quiver(ax, lon, lat, uu, vv, 20, mode="sqrt")
    quiver_key(ax, q, value=np.sqrt(10), label="10 m/s (√-scaled)")
    ax.set_title("(c) Square-root scaled arrows\n(length ∝ √speed, compresses the range)", fontsize=10)
    fig.suptitle(f"Magnitude-proportional vs same-sized vectors, t10 = {date_label(s.day.values)}, every 20th cell "
                 f"({n:,} arrows)", y=0.99)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.04, wspace=0.04)
    stamp(fig, SOURCE)
    save(fig, "Fig17_quiver_proportional_vs_uniform")


def fig18_seeding(vs, th):
    i = 9
    s, u, v = fields(vs, th, i)
    grid = proj_grid(s, u, v)
    xg, yg, ug, vg, sg = grid
    land = np.isfinite(ug)
    n_seed = 160
    # uniform grid seeds on land
    gx = np.linspace(xg.min(), xg.max(), 20)[1:-1]
    gy = np.linspace(yg.min(), yg.max(), 12)[1:-1]
    GX, GY = np.meshgrid(gx, gy)
    tree = cKDTree(np.column_stack([xg[land], yg[land]]))
    ok = tree.query(np.column_stack([GX.ravel(), GY.ravel()]))[0] < 20000
    uni = np.column_stack([GX.ravel()[ok], GY.ravel()[ok]])
    # random seeds on land (same count)
    idx = rng.choice(land.sum(), size=len(uni), replace=False)
    rnd = np.column_stack([xg[land][idx], yg[land][idx]])
    # feature-based seeds: cells in the top 5% of wind speed, thinned to the same count
    thr = np.nanpercentile(sg, 95)
    hot = np.column_stack([xg[sg >= thr], yg[sg >= thr]])
    feat = hot[rng.choice(len(hot), size=min(len(uni), len(hot)), replace=False)]
    specs = [(None, "(a) Density-based (evenly spaced, matplotlib default)"),
             (uni, f"(b) Uniform grid seeding ({len(uni)} seeds)"),
             (rnd, f"(c) Random seeding ({len(rnd)} seeds)"),
             (feat, f"(d) Feature-based: seeds in the top 5% speed (≥ {thr:.1f} m/s)")]
    fig = plt.figure(figsize=(16, 10.6))
    rows = []
    for k, (seeds, title) in enumerate(specs):
        ax = map_axes(fig, 2, 2, k + 1)
        sp = streamlines(ax, grid, start_points=seeds, density=2.0)   # same collision-mask resolution for all
        if seeds is not None:
            ax.scatter(seeds[:, 0], seeds[:, 1], s=7, c="#e7298a", zorder=6, edgecolors="none")
        cov, nlines = coverage(sp, grid)
        n_stream = len(sp.arrows.get_paths())
        ax.set_title(f"{title}\n{n_stream} streamlines, land coverage {cov:.0%} of 60 km cells", fontsize=10)
        rows.append(dict(strategy=title.split(") ")[1].split(" (")[0], seeds=0 if seeds is None else len(seeds),
                         streamlines=n_stream, coverage=round(cov, 3), line_segments=nlines))
    cax = fig.add_axes([0.3, 0.05, 0.4, 0.016])
    cb = fig.colorbar(plt.cm.ScalarMappable(SPEED_NORM, "plasma"), cax=cax, orientation="horizontal", extend="max")
    cb.set_label(SPEED_LABEL + " (streamline colour; width also ∝ speed)")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color="#e7298a", ms=4, label="seed point")],
               loc="lower right", bbox_to_anchor=(0.82, 0.035), frameon=False)
    fig.suptitle(f"Streamline seeding strategies, t10 = {date_label(s.day.values)}", y=0.96)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.08, wspace=0.04, hspace=0.12)
    stamp(fig, SOURCE)
    save(fig, "Fig18_streamline_seeding_strategies")
    tab = pd.DataFrame(rows)
    tab.to_csv(config.DATA_PROCESSED / "streamline_seeding_coverage.csv", index=False)
    log.append(tab.to_string())


def fig19_quiver_vs_stream(vs, th):
    zoom = [-114.5, -97.5, 35.0, 45.5]   # Central Rockies and High Plains
    zext = None
    fig = plt.figure(figsize=(16, 11.6))
    for r, (i, ext, stride, title) in enumerate([(9, zoom, 6, "Central Rockies / High Plains (zoom)"),
                                                 (2, None, 20, "CONUS")]):
        s, u, v = fields(vs, th, i)
        ax = map_axes(fig, 2, 2, 2 * r + 1, extent=ext)
        background(ax, s)
        lon, lat, uu, vv = subsample(s, u, v, stride)
        if ext is not None:   # only arrows inside the zoom window
            inside = (lon >= ext[0] - 1) & (lon <= ext[1] + 1) & (lat >= ext[2] - 1) & (lat <= ext[3] + 1)
            uu, vv = np.where(inside, uu, np.nan), np.where(inside, vv, np.nan)
        q, n = quiver(ax, lon, lat, uu, vv, stride, width=0.0022)
        quiver_key(ax, q)
        ax.set_title(f"({'ac'[r]}) Quiver, every {stride}th cell ({n:,} arrows) – {title}, {date_label(s.day.values)}",
                     fontsize=9.5)
        ax2 = map_axes(fig, 2, 2, 2 * r + 2, extent=ext)
        if ext is not None:
            zext = ax2.get_extent(crs=PROJ)
        grid = proj_grid(s, u, v, nx=300 if ext else 360, extent=zext if ext else None, stride=1 if ext else 2)
        background(ax2, s, alpha=0.35)
        streamlines(ax2, grid, density=1.6 if ext else 1.8, color_by_speed=False)
        ax2.set_title(f"({'bd'[r]}) Streamlines (density-based seeding) – {title}, {date_label(s.day.values)}",
                      fontsize=9.5)
    cax = fig.add_axes([0.3, 0.045, 0.4, 0.014])
    cb = fig.colorbar(plt.cm.ScalarMappable(SPEED_NORM, BG_CMAP), cax=cax, orientation="horizontal", extend="max")
    cb.set_label(SPEED_LABEL + " (background)")
    fig.suptitle("Quiver plots vs streamlines in strong (29 Sep, top) and weak (21 Jul, bottom) flow", y=0.95)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.07, wspace=0.04, hspace=0.12)
    stamp(fig, SOURCE)
    save(fig, "Fig19_quiver_vs_streamlines_comparison")


def fig20_21_time(vs, th):
    # quiver small multiples + GIF
    def draw_quiver(ax, i, stride=34):
        s, u, v = fields(vs, th, i)
        background(ax, s)
        lon, lat, uu, vv = subsample(s, u, v, stride)
        q, _ = quiver(ax, lon, lat, uu, vv, stride, width=0.004 if stride > 30 else 0.0026)
        return s, q

    fig = plt.figure(figsize=(17, 5.9))
    for i in range(vs.sizes["day"]):
        ax = map_axes(fig, 2, 5, i + 1)
        s, q = draw_quiver(ax, i)
        ax.set_title(f"t{i + 1} · {date_label(s.day.values)}  (mean {float(s.mean()):.1f} m/s)", fontsize=9)
    quiver_key(ax, q)
    cax = fig.add_axes([0.3, 0.065, 0.4, 0.028])
    cb = fig.colorbar(plt.cm.ScalarMappable(SPEED_NORM, BG_CMAP), cax=cax, orientation="horizontal", extend="max")
    cb.set_label(SPEED_LABEL)
    fig.suptitle("Daily mean 10 m wind: speed (colour) and magnitude-proportional quiver (every 34th cell), all 10 time steps",
                 y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.15, wspace=0.03, hspace=0.14)
    stamp(fig, SOURCE)
    save(fig, "Fig20_wind_quiver_small_multiples")
    frames = []
    for i in range(vs.sizes["day"]):
        fig = plt.figure(figsize=(8.4, 5.9))
        ax = map_axes(fig, 1, 1, 1)
        s, q = draw_quiver(ax, i, stride=18)
        quiver_key(ax, q)
        ax.set_title(f"Daily mean 10 m wind – quiver (every 18th cell) over wind speed\n"
                     f"t{i + 1} of 10 · {date_label(s.day.values)}", fontsize=10.5)
        cb = fig.colorbar(plt.cm.ScalarMappable(SPEED_NORM, BG_CMAP), ax=ax, orientation="horizontal", pad=0.04,
                          fraction=0.05, aspect=40, extend="max")
        cb.set_label(SPEED_LABEL)
        stamp(fig, SOURCE)
        frames.append(fig_to_image(fig, dpi=95))
    save_gif(frames, "Fig20_wind_quiver_animation")

    # streamline small multiples + GIF
    fig = plt.figure(figsize=(17, 5.9))
    for i in range(vs.sizes["day"]):
        s, u, v = fields(vs, th, i)
        ax = map_axes(fig, 2, 5, i + 1)
        streamlines(ax, proj_grid(s, u, v, nx=240, stride=3), density=1.0, max_length=3.0)
        ax.set_title(f"t{i + 1} · {date_label(s.day.values)}", fontsize=9.5)
    cax = fig.add_axes([0.3, 0.065, 0.4, 0.028])
    cb = fig.colorbar(plt.cm.ScalarMappable(SPEED_NORM, "plasma"), cax=cax, orientation="horizontal", extend="max")
    cb.set_label(SPEED_LABEL + " (streamline colour and width)")
    fig.suptitle("Streamlines of the daily mean 10 m wind (density-based seeding), all 10 time steps", y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.15, wspace=0.03, hspace=0.14)
    stamp(fig, SOURCE)
    save(fig, "Fig21_wind_streamlines_small_multiples")
    frames = []
    for i in range(vs.sizes["day"]):
        s, u, v = fields(vs, th, i)
        fig = plt.figure(figsize=(8.4, 5.9))
        ax = map_axes(fig, 1, 1, 1)
        streamlines(ax, proj_grid(s, u, v, nx=300, stride=2), density=1.6)
        ax.set_title(f"Streamlines of the daily mean 10 m wind (colour and width = speed)\n"
                     f"t{i + 1} of 10 · {date_label(s.day.values)}", fontsize=10.5)
        cb = fig.colorbar(plt.cm.ScalarMappable(SPEED_NORM, "plasma"), ax=ax, orientation="horizontal", pad=0.04,
                          fraction=0.05, aspect=40, extend="max")
        cb.set_label(SPEED_LABEL)
        stamp(fig, SOURCE)
        frames.append(fig_to_image(fig, dpi=95))
    save_gif(frames, "Fig21_wind_streamlines_animation")


STORMS = [  # (label, day, lon/lat window): storm days between the samples, from the 4 km storm files
    ("Hurricane Barry", "2019-07-13", [-96.5, -87.5, 28.8, 35.0], "landfall in Louisiana"),
    ("Hurricane Dorian", "2019-09-05", [-84.0, -74.8, 30.5, 37.8], "moving up the Carolinas coast"),
    ("Tropical Storm Imelda", "2019-09-18", [-98.5, -89.5, 27.5, 33.5], "slow-moving over south-east Texas"),
]
PR_NORM = mcolors.LogNorm(1, 700)


def load_storm(var):
    ds = xr.open_dataset(config.GRIDMET_RAW / f"{var}_2019_storms_4km.nc")
    return ds[list(ds.data_vars)[0]].load()


def kinematics(s, u, v, box=9):
    """Relative vorticity and divergence (1/s) of the wind smoothed with a mask-aware ~36 km box filter."""
    ok = np.isfinite(u)

    def smooth(a):
        num, den = uniform_filter(np.where(ok, a, 0.0), box), uniform_filter(ok.astype(float), box)
        return np.where(den > 0.6, num / np.maximum(den, 1e-9), np.nan)

    us, vsm = smooth(u), smooth(v)
    lon2, lat2 = np.meshgrid(s.lon.values, s.lat.values)
    dy = np.gradient(lat2, axis=0) * np.pi / 180 * EARTH_R
    dx = np.gradient(lon2, axis=1) * np.pi / 180 * EARTH_R * np.cos(np.deg2rad(lat2))
    zeta = np.gradient(vsm, axis=1) / dx - np.gradient(us, axis=0) / dy
    div = np.gradient(us, axis=1) / dx + np.gradient(vsm, axis=0) / dy
    return zeta, div


def window(da, ext):
    return da.sel(lon=slice(ext[0], ext[1]), lat=slice(ext[3], ext[2]))


def fig22_storms(vs, th):
    svs, sth, spr = load_storm("vs"), load_storm("th"), load_storm("pr")
    fig = plt.figure(figsize=(17, 11.4))
    top, bottom = [], []
    for k, (name, day, ext, note) in enumerate(STORMS):
        s, p = svs.sel(day=day), spr.sel(day=day)
        u, v = wind_components(s.values, sth.sel(day=day).values)
        sw, pw = window(s, ext), window(p, ext)
        # (top) wind speed + magnitude-proportional quiver, every 8th cell (~32 km)
        ax = map_axes(fig, 2, 3, k + 1, extent=ext)
        background(ax, s)
        lon, lat, uu, vv = subsample(s, u, v, 8, offset=4)
        inside = (lon >= ext[0]) & (lon <= ext[1]) & (lat >= ext[2]) & (lat <= ext[3])
        q, n = quiver(ax, lon, lat, np.where(inside, uu, np.nan), np.where(inside, vv, np.nan), 8, width=0.0032)
        quiver_key(ax, q)
        i, j = np.unravel_index(np.nanargmax(sw.values), sw.shape)
        ax.plot(sw.lon.values[j], sw.lat.values[i], marker="X", ms=9, color="#e7298a", mec="white", mew=0.8,
                transform=PC, zorder=7)
        ax.set_title(f"({'abc'[k]}) {name}, {date_label(day)} ({note})\nwind every 8th cell ({n:,} arrows); "
                     f"max {float(sw.max()):.1f} m/s (×)", fontsize=9.5)
        top.append(ax)
        # (bottom) precipitation (log) + streamlines
        ax2 = map_axes(fig, 2, 3, k + 4, extent=ext)
        mesh(ax2, p.where(p >= 1), cmap="YlGnBu", norm=PR_NORM)
        grid = proj_grid(s, u, v, nx=220, extent=ax2.get_extent(crs=PROJ), stride=1)
        streamlines(ax2, grid, density=1.3, color_by_speed=False)
        i, j = np.unravel_index(np.nanargmax(pw.values), pw.shape)
        ax2.plot(pw.lon.values[j], pw.lat.values[i], marker="^", ms=9, color="#e7298a", mec="white", mew=0.8,
                 transform=PC, zorder=7)
        ax2.set_title(f"({'def'[k]}) Precipitation and streamlines, {date_label(day)}\n"
                      f"max {float(pw.max()):.0f} mm at {float(pw.lat[i]):.2f}° N, {-float(pw.lon[j]):.2f}° W (▲)",
                      fontsize=9.5)
        bottom.append(ax2)
        # diagnostics for the report: rotation (vorticity) and convergence under the heavy rain
        zeta, div = kinematics(sw, *wind_components(sw.values, window(sth.sel(day=day), ext).values))
        ref = [np.nanmean(kinematics(window(vs.isel(day=t), ext),
                                     *wind_components(window(vs.isel(day=t), ext).values,
                                                      window(th.isel(day=t), ext).values))[0])
               for t in range(vs.sizes["day"])]
        heavy = (pw.values >= 100) & np.isfinite(div)
        dry = (pw.values < 10) & np.isfinite(div)
        log.append(f"Fig22 {name} {day}: speed max {float(sw.max()):.1f} mean {float(sw.mean()):.1f} m/s; "
                   f"pr max {float(pw.max()):.1f} mm at {float(pw.lat[i]):.2f}N {float(pw.lon[j]):.2f}E; "
                   f"area>=100mm {heavy.sum() * 16:,} km2 mean div {np.nanmean(div[heavy]) * 1e5:+.2f}e-5/s "
                   f"({np.mean(div[heavy] < 0):.0%} convergent) vs {np.nanmean(div[dry]) * 1e5:+.2f}e-5/s where <10mm; "
                   f"area-mean vorticity {np.nanmean(zeta) * 1e5:+.2f}e-5/s (sampled days "
                   f"{min(ref) * 1e5:+.2f}..{max(ref) * 1e5:+.2f}e-5/s)")
    fig.subplots_adjust(left=0.01, right=0.92, top=0.93, bottom=0.03, wspace=0.05, hspace=0.16)
    for axes, sm, label, ext in [(top, plt.cm.ScalarMappable(SPEED_NORM, BG_CMAP), SPEED_LABEL, "max"),
                                 (bottom, plt.cm.ScalarMappable(PR_NORM, "YlGnBu"),
                                  "Precipitation (mm/day, log scale; < 1 mm blank)", "max")]:
        y0 = min(a.get_position().y0 for a in axes)
        y1 = max(a.get_position().y1 for a in axes)
        cax = fig.add_axes([0.935, y0 + 0.03, 0.012, y1 - y0 - 0.06])
        cb = fig.colorbar(sm, cax=cax, extend=ext)
        cb.set_label(label)
    fig.suptitle("Storm close-ups on days between the samples (4 km): wind vectors (top), rain and streamlines (bottom)",
                 y=0.985)
    stamp(fig, SOURCE.replace("Jul–Sep 2019", "storm days of Jul–Sep 2019"))
    save(fig, "Fig22_storm_closeups_barry_dorian_imelda")


def check_vector_transform(vs, th):
    """Direction error of the projected arrows without and with geo_vectors(), against a finite-difference
    ground truth (a 100 m step east/north on the sphere, projected), on the t10 field."""
    s, u, v = fields(vs, th, 9)
    lon2, lat2 = np.meshgrid(s.lon.values[::10], s.lat.values[::10])
    uu, vv = u[::10, ::10], v[::10, ::10]
    m = np.isfinite(uu) & (np.hypot(uu, vv) > 0.5)
    lon, lat, uu, vv = lon2[m], lat2[m], uu[m], vv[m]
    step = 100.0 / np.hypot(uu, vv)
    lon1 = lon + np.degrees(uu * step / (EARTH_R * np.cos(np.deg2rad(lat))))
    lat1 = lat + np.degrees(vv * step / EARTH_R)
    p0, p1 = PROJ.transform_points(PC, lon, lat), PROJ.transform_points(PC, lon1, lat1)
    truth = np.arctan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])
    out = []
    for label, (a, b) in [("naive", (uu, vv)), ("corrected", geo_vectors(uu, vv, lat))]:
        ur, vr = PROJ.transform_vectors(PC, lon, lat, a, b)
        err = np.abs(angle_diff_deg(ur, vr, np.cos(truth), np.sin(truth)))
        out.append(f"{label}: median {np.median(err):.2f}, p95 {np.percentile(err, 95):.2f}, max {err.max():.2f} deg; "
                   f"speed kept {np.allclose(np.hypot(ur, vr), np.hypot(uu, vv))}")
    log.append(f"Vector transform check on t10 ({m.sum():,} arrows): " + " | ".join(out))


def describe(vs, th):
    for i in range(vs.sizes["day"]):
        s, u, v = fields(vs, th, i)
        um, vm = np.nanmean(u), np.nanmean(v)
        steady = np.hypot(um, vm) / np.nanmean(s.values)
        direction_from = (np.degrees(np.arctan2(-um, -vm)) + 360) % 360
        log.append(f"t{i + 1} {date_label(s.day.values)} mean speed {float(s.mean()):.2f} p95 "
                   f"{float(s.quantile(0.95)):.2f} max {float(s.max()):.1f} | domain vector-mean from {direction_from:.0f} deg, "
                   f"CONUS steadiness {steady:.2f}")


def main() -> None:
    vs, th = load("vs"), load("th")
    describe(vs, th)
    check_vector_transform(vs, th)
    fig16_sampling(vs, th)
    fig17_magnitude(vs, th)
    fig18_seeding(vs, th)
    fig19_quiver_vs_stream(vs, th)
    fig20_21_time(vs, th)
    fig22_storms(vs, th)
    (config.DATA_PROCESSED / "scivis_quiver_stats.txt").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
