"""SciVis Task 2 - Contour mapping (Kalpit, BT2024093).

Scalar field: gridMET daily mean vapour pressure deficit (vpd, kPa), 10 uniformly sampled days of Jul-Sep 2019.
Isolines are extracted with our own vectorised Marching Squares implementation (scivis/marching_squares.py);
filled contours (colour bands between isolines) use matplotlib's contourf (contour-fill algorithm).

Figures written to images/:
  Fig10  Marching Squares: 16-case lookup table, zoomed worked example, interpolation vs midpoint, validation
  Fig11  CONUS isolines of VPD (own Marching Squares) on 21 Jul 2019
  Fig12  line contours vs filled contours vs number of levels
  Fig13  effect of Gaussian pre-smoothing on isoline complexity (western US)
  Fig14  filled contours + 3 kPa isoline for all 10 time steps (+ GIF)
  Fig15  contour-derived statistics over time (area above thresholds, isoline length)

Usage:  python code/scivis/contour_mapping.py
"""
import sys
import time
from pathlib import Path

import contourpy
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scivis.common import (PC, PROJ, SOURCE, date_label, fig_to_image, load, map_axes, mesh, plt, save,  # noqa: E402
                           save_gif, stamp)
from scivis.marching_squares import CASE_SEGMENTS, SADDLE_SEGMENTS, marching_squares, segment_length_km  # noqa: E402

VPD_LABEL = "Vapour pressure deficit (kPa)"
LEVELS = [0.5, 1.0, 2.0, 3.0, 4.0]
HALF_STEP = 0.005          # gridMET VPD is stored in 0.01 kPa steps -> evaluate isolines between two storable values
FILL_LEVELS = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 8.5]
LEVEL_COLORS = plt.get_cmap("plasma")(np.linspace(0.05, 0.85, len(LEVELS)))
stats_lines = []


def iso_value(level):
    return level - HALF_STEP


def add_segments(ax, segs, **kw):
    """Project lon/lat segments to the map projection and draw them as one LineCollection."""
    if len(segs) == 0:
        return
    pts = PROJ.transform_points(PC, segs[..., 0].ravel(), segs[..., 1].ravel())[:, :2].reshape(segs.shape)
    ax.add_collection(LineCollection(pts, transform=ax.transData, **kw))


def nan_gaussian(F, sigma):
    """Normalised (mask-aware) Gaussian smoothing so the coastline is not pulled towards zero."""
    if sigma == 0:
        return F
    m = np.isfinite(F)
    num = gaussian_filter(np.where(m, F, 0.0), sigma)
    den = gaussian_filter(m.astype(float), sigma)
    out = num / np.where(den > 0, den, np.nan)
    out[~m] = np.nan
    return out


def cell_area_km2(lat, nx):
    dlat = dlon = 1 / 24
    return np.repeat((dlat * 110.57 * dlon * 111.32 * np.cos(np.deg2rad(lat)))[:, None], nx, axis=1)


# ---------------------------------------------------------------------------------------------------------------
def draw_case(ax, case):
    corners = {"a": (0, 1), "b": (1, 1), "c": (1, 0), "d": (0, 0)}
    bits = {"a": 8, "b": 4, "c": 2, "d": 1}
    mids = {0: (0.5, 1), 1: (1, 0.5), 2: (0.5, 0), 3: (0, 0.5)}
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=0.8, ec="#555555"))
    if case in (5, 10):
        for (e0, e1) in SADDLE_SEGMENTS[(case, True)]:
            ax.plot(*zip(mids[e0], mids[e1]), color="#d7301f", lw=2.2)
        for (e0, e1) in SADDLE_SEGMENTS[(case, False)]:
            ax.plot(*zip(mids[e0], mids[e1]), color="#2b8cbe", lw=1.6, ls="--")
    else:
        for (e0, e1) in CASE_SEGMENTS[case]:
            ax.plot(*zip(mids[e0], mids[e1]), color="#d7301f", lw=2.2)
    for k, (cx, cy) in corners.items():
        above = case & bits[k]
        ax.scatter(cx, cy, s=55, c="black" if above else "white", edgecolors="black", zorder=3, lw=0.9)
    ax.set_title(f"case {case}", fontsize=8, pad=2)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")


def fig10_algorithm(v):
    fig = plt.figure(figsize=(17, 8.2))
    gs = fig.add_gridspec(4, 9, wspace=0.25, hspace=0.35, width_ratios=[1, 1, 1, 1, 0.25, 1.4, 1.4, 1.4, 1.4])
    for case in range(16):
        draw_case(fig.add_subplot(gs[case // 4, case % 4]), case)
    fig.text(0.255, 0.955, "(a) Marching Squares lookup table (● value ≥ isovalue, ○ below)", ha="center", fontsize=10)
    fig.text(0.255, 0.035, "Saddles 5 and 10: red = cell centre ≥ isovalue (chosen), blue dashed = centre below",
             ha="center", fontsize=8.5, color="#333333")

    d = 2
    fr = v.isel(day=d)
    sub = fr.sel(lat=slice(41.0, 40.0), lon=slice(-112.3, -111.3))
    F, x, y = sub.values, sub.lon.values, sub.lat.values
    level = 3.0
    iso = iso_value(level)
    segs_i = marching_squares(F, x, y, iso, interpolate=True)
    segs_m = marching_squares(F, x, y, iso, interpolate=False)
    gen = contourpy.contour_generator(x, y, F, name="serial", line_type="Separate")
    lib = gen.lines(iso)
    ax = fig.add_subplot(gs[:, 5:])
    X, Y = np.meshgrid(x, y)
    norm = mcolors.TwoSlopeNorm(level, np.nanmin(F), np.nanmax(F))
    ax.pcolormesh(x, y, F, cmap="BrBG_r", norm=norm, shading="nearest", alpha=0.35)
    sc = ax.scatter(X, Y, c=F, cmap="BrBG_r", norm=norm, s=16, edgecolors="#333333", linewidths=0.3, zorder=3)
    ax.scatter(X[F >= iso], Y[F >= iso], s=4, c="black", zorder=4)
    ax.add_collection(LineCollection(segs_m, colors="#ff7f00", lw=1.4, linestyles=":", zorder=5))
    ax.add_collection(LineCollection(segs_i, colors="#d7301f", lw=2.4, zorder=6))
    for line in lib:
        ax.plot(line[:, 0], line[:, 1], color="white", lw=0.9, ls="--", zorder=7)
    ax.set_xlim(x.min() - 0.03, x.max() + 0.03)
    ax.set_ylim(y.min() - 0.03, y.max() + 0.03)
    ax.set_aspect(1 / np.cos(np.deg2rad(40.5)))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(VPD_LABEL)
    ax.legend(handles=[
        Line2D([], [], color="#d7301f", lw=2.4, label="own Marching Squares, linear interpolation"),
        Line2D([], [], color="#ff7f00", lw=1.4, ls=":", label="own, edge midpoints (no interpolation)"),
        Line2D([], [], color="white", lw=1.2, ls="--", label="contourpy reference (dashed, over red)"),
        Line2D([], [], color="black", marker="o", ls="", ms=3, label="grid point ≥ isovalue"),
    ], loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(f"(b) Worked example: {level:.0f} kPa isoline, Wasatch Front (Utah), {date_label(fr.day.values)}\n"
                 f"25 × 25 grid points (4 km spacing)", fontsize=10)

    # validation over the full CONUS field
    rows = []
    Ff, xf, yf = fr.values, fr.lon.values, fr.lat.values
    genf = contourpy.contour_generator(xf, yf, np.ma.masked_invalid(Ff), name="serial", corner_mask=False,
                                       line_type="Separate")
    for lev in LEVELS:
        t0 = time.perf_counter()
        s, counts = marching_squares(Ff, xf, yf, iso_value(lev), return_stats=True)
        t1 = time.perf_counter()
        lines = genf.lines(iso_value(lev))
        t2 = time.perf_counter()
        pl, po = np.concatenate(lines), s.reshape(-1, 2)
        h = max(cKDTree(pl).query(po)[0].max(), cKDTree(po).query(pl)[0].max())
        tie = marching_squares(Ff, xf, yf, lev)
        lt = genf.lines(lev)
        pl2, po2 = np.concatenate(lt), tie.reshape(-1, 2)
        h_tie = max(cKDTree(pl2).query(po2)[0].max(), cKDTree(po2).query(pl2)[0].max())
        rows.append(dict(level=lev, segments=len(s), length_km=segment_length_km(s), saddles=int(counts[5] + counts[10]),
                         own_ms=1000 * (t1 - t0), lib_ms=1000 * (t2 - t1), hausdorff_deg=h, hausdorff_tie_deg=h_tie,
                         cells_exactly_at_level=int(np.sum(Ff == lev))))
    tab = pd.DataFrame(rows)
    tab.to_csv(config.DATA_PROCESSED / "contour_validation_21Jul2019.csv", index=False)
    stats_lines.append(tab.to_string())
    txt = ("(c) Validation on the full CONUS field (21 Jul): own vs contourpy, isovalue = level − 0.005 kPa\n"
           f"max. Hausdorff distance ≤ {tab.hausdorff_deg.max():.1e}° for all levels; "
           f"own {tab.own_ms.mean():.0f} ms vs contourpy {tab.lib_ms.mean():.0f} ms per level; "
           f"with the isovalue exactly on a stored value (e.g. {int(tab.cells_exactly_at_level.max())} cells = 1.00 kPa) "
           f"the ≥ / > tie rules differ by up to {tab.hausdorff_tie_deg.max():.2f}°")
    fig.text(0.5, -0.02, txt, ha="center", fontsize=8.5, color="#222222")
    fig.suptitle("Contour mapping with Marching Squares: algorithm and verification", y=1.0)
    stamp(fig, SOURCE)
    save(fig, "Fig10_marching_squares_algorithm_validation")


def fig11_isolines(v):
    d = 2
    fr = v.isel(day=d)
    fig = plt.figure(figsize=(13, 8.4))
    ax = map_axes(fig, 1, 1, 1)
    mesh(ax, fr, cmap="Greys", norm=mcolors.Normalize(0, 8), alpha=0.55)
    handles = []
    for lev, col in zip(LEVELS, LEVEL_COLORS):
        s = marching_squares(fr.values, fr.lon.values, fr.lat.values, iso_value(lev))
        add_segments(ax, s, colors=[col], linewidths=1.0 if lev != 3 else 1.6, zorder=4)
        handles.append(Line2D([], [], color=col, lw=2, label=f"{lev:g} kPa  ({segment_length_km(s):,.0f} km of isoline)"))
    ax.legend(handles=handles, title="Isovalue (own Marching Squares)", loc="lower left", fontsize=8.5, title_fontsize=9,
              framealpha=0.92)
    ax.set_title(f"Isolines of daily mean vapour pressure deficit, t3 = {date_label(fr.day.values)} "
                 "(grey background: VPD, darker = drier air)", fontsize=11)
    stamp(fig, SOURCE)
    save(fig, "Fig11_vpd_isolines_marching_squares_21Jul2019")


def fig12_line_vs_fill(v):
    d = 2
    fr = v.isel(day=d)
    F, x, y = fr.values, fr.lon.values, fr.lat.values
    Fm = np.ma.masked_invalid(F)
    cmap = plt.get_cmap("YlOrBr")
    fig = plt.figure(figsize=(15, 10.6))
    # (a) isolines only
    ax = map_axes(fig, 2, 2, 1)
    for lev, col in zip(LEVELS, LEVEL_COLORS):
        add_segments(ax, marching_squares(F, x, y, iso_value(lev)), colors=[col], linewidths=0.9, zorder=4)
    ax.legend(handles=[Line2D([], [], color=c, lw=2, label=f"{lv:g} kPa") for lv, c in zip(LEVELS, LEVEL_COLORS)],
              loc="lower left", fontsize=7.5, ncol=1)
    ax.set_title("(a) Line contours only: 5 isolines (Marching Squares)")
    # (b) filled, same 5 levels
    ax = map_axes(fig, 2, 2, 2)
    lv5 = [0] + LEVELS + [8.5]
    norm5 = mcolors.BoundaryNorm(lv5, cmap.N)
    cf = ax.contourf(x, y, Fm, levels=lv5, cmap=cmap, norm=norm5, transform=PC)
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.03, fraction=0.05, aspect=40, spacing="uniform")
    cb.set_label(VPD_LABEL)
    ax.set_title("(b) Filled contours (contour fill): the same 5 levels as bands")
    # (c) filled, 10 levels + isolines
    ax = map_axes(fig, 2, 2, 3)
    normf = mcolors.BoundaryNorm(FILL_LEVELS, cmap.N)
    cf = ax.contourf(x, y, Fm, levels=FILL_LEVELS, cmap=cmap, norm=normf, transform=PC)
    for lev in LEVELS:
        add_segments(ax, marching_squares(F, x, y, iso_value(lev)), colors=["#3b0f2e"], linewidths=0.45, zorder=4)
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.03, fraction=0.05, aspect=40, spacing="uniform")
    cb.set_label(VPD_LABEL)
    ax.set_title(f"(c) Filled contours, {len(FILL_LEVELS) - 1} levels, with the 5 isolines overlaid")
    # (d) continuous reference
    ax = map_axes(fig, 2, 2, 4)
    im = mesh(ax, fr, cmap=cmap, norm=mcolors.Normalize(0, 8.5))
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, fraction=0.05, aspect=40)
    cb.set_label(VPD_LABEL)
    ax.set_title("(d) Reference: continuous colour mapping (no contouring)")
    fig.suptitle(f"Line contours vs filled contours of vapour pressure deficit, t3 = {date_label(fr.day.values)}", y=0.95)
    fig.subplots_adjust(hspace=0.08, wspace=0.04)
    stamp(fig, SOURCE)
    save(fig, "Fig12_line_vs_filled_contours_vpd_21Jul2019")


def fig13_smoothing(v):
    d = 2
    fr = v.isel(day=d)
    F, x, y = fr.values, fr.lon.values, fr.lat.values
    west = [-125, -101.5, 30.5, 49.5]
    sigmas = [0, 2, 5]
    fig = plt.figure(figsize=(17, 6.2))
    rows = []
    for k, sg in enumerate(sigmas):
        Fs = nan_gaussian(F, sg)
        ax = map_axes(fig, 1, 3, k + 1, extent=west)
        ax.pcolormesh(x, y, np.ma.masked_invalid(Fs), cmap="Greys", norm=mcolors.Normalize(0, 8), alpha=0.5,
                      transform=PC, shading="nearest", rasterized=True)
        tot_len, tot_seg = 0, 0
        for lev, col in zip(LEVELS[1:], LEVEL_COLORS[1:]):
            s, counts = marching_squares(Fs, x, y, iso_value(lev), return_stats=True)
            add_segments(ax, s, colors=[col], linewidths=0.9, zorder=4)
            L = segment_length_km(s)
            tot_len += L
            tot_seg += len(s)
            rows.append(dict(sigma_cells=sg, sigma_km=4 * sg, level=lev, segments=len(s), length_km=round(L),
                             saddles=int(counts[5] + counts[10])))
        ax.set_title(f"(σ = {sg} cells ≈ {4 * sg} km)" if sg else "(raw 4 km field, σ = 0)", fontsize=10)
        ax.text(0.02, 0.02, f"whole CONUS, 1–4 kPa isolines:\n{tot_len:,.0f} km, {tot_seg:,} segments",
                transform=ax.transAxes,
                fontsize=8.5, va="bottom", bbox=dict(fc="white", ec="none", alpha=0.85))
    fig.legend(handles=[Line2D([], [], color=c, lw=2, label=f"{lv:g} kPa") for lv, c in zip(LEVELS[1:], LEVEL_COLORS[1:])],
               loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Effect of Gaussian pre-smoothing on VPD isolines over the western US, {date_label(fr.day.values)}", y=0.98)
    fig.subplots_adjust(wspace=0.04, bottom=0.08, top=0.88)
    stamp(fig, SOURCE)
    save(fig, "Fig13_contour_smoothing_experiment_west")
    tab = pd.DataFrame(rows)
    tab.to_csv(config.DATA_PROCESSED / "contour_smoothing_experiment.csv", index=False)
    stats_lines.append(tab.to_string())


def fig14_time(v):
    cmap = plt.get_cmap("YlOrBr")
    normf = mcolors.BoundaryNorm(FILL_LEVELS, cmap.N)

    def draw(ax, fr):
        cf = ax.contourf(fr.lon.values, fr.lat.values, np.ma.masked_invalid(fr.values), levels=FILL_LEVELS, cmap=cmap,
                         norm=normf, transform=PC)
        add_segments(ax, marching_squares(fr.values, fr.lon.values, fr.lat.values, iso_value(3.0)),
                     colors=["#7a0177"], linewidths=1.0, zorder=4)
        add_segments(ax, marching_squares(fr.values, fr.lon.values, fr.lat.values, iso_value(1.0)),
                     colors=["#225ea8"], linewidths=0.6, zorder=4)
        return cf

    fig = plt.figure(figsize=(17, 5.9))
    for i, d in enumerate(v.day.values):
        ax = map_axes(fig, 2, 5, i + 1)
        cf = draw(ax, v.isel(day=i))
        ax.set_title(f"t{i + 1} · {date_label(d)}", fontsize=9.5)
    cax = fig.add_axes([0.2, 0.065, 0.42, 0.028])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", spacing="uniform")
    cb.set_label(VPD_LABEL)
    fig.legend(handles=[Line2D([], [], color="#7a0177", lw=2, label="3 kPa isoline (very dry air)"),
                        Line2D([], [], color="#225ea8", lw=2, label="1 kPa isoline (humid air)")],
               loc="lower right", bbox_to_anchor=(0.92, 0.03), fontsize=9, frameon=False)
    fig.suptitle("Filled contours of vapour pressure deficit with Marching-Squares isolines, all 10 time steps "
                 "(fixed levels for comparability)", y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.15, wspace=0.03, hspace=0.14)
    stamp(fig, SOURCE)
    save(fig, "Fig14_vpd_contours_small_multiples")

    frames = []
    for i, d in enumerate(v.day.values):
        fig = plt.figure(figsize=(8.4, 5.9))
        ax = map_axes(fig, 1, 1, 1)
        cf = draw(ax, v.isel(day=i))
        ax.set_title(f"Vapour pressure deficit: filled contours, 1 kPa (blue) and 3 kPa (purple) isolines\n"
                     f"t{i + 1} of 10 · {date_label(d)}", fontsize=10.5)
        cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04, fraction=0.05, aspect=40, spacing="uniform")
        cb.set_label(VPD_LABEL)
        stamp(fig, SOURCE)
        frames.append(fig_to_image(fig, dpi=95))
    save_gif(frames, "Fig14_vpd_contours_animation")


def fig15_stats(v):
    area = cell_area_km2(v.lat.values, v.sizes["lon"])
    rows = []
    for i, d in enumerate(v.day.values):
        F = v.isel(day=i).values
        valid = np.isfinite(F)
        row = dict(date=pd.Timestamp(d), t=f"t{i + 1}", land_km2=area[valid].sum())
        for lev in [1.0, 2.0, 3.0]:
            row[f"area_ge_{lev:g}_km2"] = area[valid & (F >= lev)].sum()
            row[f"len_{lev:g}_km"] = segment_length_km(marching_squares(F, v.lon.values, v.lat.values, iso_value(lev)))
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(config.DATA_PROCESSED / "contour_stats_by_timestep.csv", index=False)
    stats_lines.append(tab.to_string())

    daily = load("vpd", "daily_stride4")
    area4 = cell_area_km2(daily.lat.values, daily.sizes["lon"]) * 16
    dd = []
    for i, d in enumerate(daily.day.values):
        F = daily.isel(day=i).values
        valid = np.isfinite(F)
        dd.append(dict(date=pd.Timestamp(d), **{f"share_ge_{lev:g}": area4[valid & (F >= lev)].sum() / area4[valid].sum()
                                                for lev in [2.0, 3.0]}))
    dd = pd.DataFrame(dd)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    ax = axes[0]
    for lev, col in [(2.0, "#ec7014"), (3.0, "#7a0177")]:
        ax.plot(dd.date, 100 * dd[f"share_ge_{lev:g}"], color=col, lw=1, alpha=0.6,
                label=f"VPD ≥ {lev:g} kPa, all 92 days (16 km grid)")
        ax.plot(tab.date, 100 * tab[f"area_ge_{lev:g}_km2"] / tab.land_km2, "o", color=col, ms=6,
                label=f"VPD ≥ {lev:g} kPa, sampled days (4 km grid)")
    ax.set_ylabel("Share of CONUS land area (%)")
    ax.set_title("(a) Area enclosed by the 2 and 3 kPa isolines")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(lw=0.3, alpha=0.6)
    ax = axes[1]
    w = 2.6
    off = {1.0: -w, 2.0: 0, 3.0: w}
    for lev, col in [(1.0, "#225ea8"), (2.0, "#ec7014"), (3.0, "#7a0177")]:
        ax.bar(tab.date + pd.to_timedelta(off[lev], unit="D"), tab[f"len_{lev:g}_km"] / 1000, width=w, color=col,
               label=f"{lev:g} kPa isoline")
    ax.set_ylabel("Total isoline length (thousand km)")
    ax.set_title("(b) Isoline length per sampled time step (Marching Squares)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", lw=0.3, alpha=0.6)
    for a in axes:
        a.set_xticks(tab.date)
        a.set_xticklabels([f"{r.t}\n{r.date:%d %b}" for r in tab.itertuples()], fontsize=7.5)
    fig.suptitle("Contour-derived statistics of vapour pressure deficit across Jul–Sep 2019", y=1.0)
    fig.tight_layout()
    stamp(fig, SOURCE)
    save(fig, "Fig15_vpd_contour_statistics_over_time")


def main() -> None:
    v = load("vpd")
    fig10_algorithm(v)
    fig11_isolines(v)
    fig12_line_vs_fill(v)
    fig13_smoothing(v)
    fig14_time(v)
    fig15_stats(v)
    (config.DATA_PROCESSED / "scivis_contour_stats.txt").write_text("\n\n".join(stats_lines), encoding="utf-8")
    print("\n\n".join(stats_lines))


if __name__ == "__main__":
    main()
