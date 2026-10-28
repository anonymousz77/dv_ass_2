"""SciVis Task 1 - Colour mapping (Lakshya Jain, BT2024044).

Scalar fields: gridMET daily maximum temperature (tmmx, °C) and daily precipitation (pr, mm/day),
10 uniformly sampled days of Jul-Sep 2019 (every 10th day from 1 July).

Figures written to images/:
  Fig02  colormap palettes on one day (sequential / rainbow / diverging) + CIE L* lightness profiles
  Fig03  discrete colour mapping: equal-interval vs quantile classes vs continuous
  Fig04  global min/max normalisation across all time steps (+ GIF)
  Fig05  local (per time step) min/max normalisation (+ GIF)
  Fig06  per-time-step value ranges vs global and robust-global ranges
  Fig07  anomaly field with a diverging colormap centred at 0 (+ GIF)
  Fig08  precipitation: linear vs logarithmic vs discrete mapping, with the value distribution
  Fig09  precipitation with a global logarithmic mapping across time steps (+ GIF)

Usage:  python code/scivis/color_mapping.py
"""
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scivis.common import (PROJ, SOURCE, add_features, date_label, fig_to_image, load, map_axes, mesh,  # noqa: E402
                           plt, save, save_gif, stamp)

TMMX_LABEL = "Daily max. temperature (°C)"
PR_LABEL = "Precipitation (mm/day)"


def srgb_lightness(cmap, n=256):
    """CIE L* (D65) of a colormap, from the relative luminance of its linearised sRGB colours."""
    rgb = cmap(np.linspace(0, 1, n))[:, :3]
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    y = lin @ np.array([0.2126, 0.7152, 0.0722])
    f = np.where(y > (6 / 29) ** 3, np.cbrt(y), y / (3 * (6 / 29) ** 2) + 4 / 29)
    return 116 * f - 16


def small_multiples(da, title, cmap, norm_fn, cbar_label, name, shared_cbar=True, extend="neither"):
    """2 x 5 grid of the 10 time steps. norm_fn(i, frame) returns the Normalize for frame i."""
    fig = plt.figure(figsize=(17, 5.7 if shared_cbar else 6.6))
    ims = []
    for i, d in enumerate(da.day.values):
        ax = map_axes(fig, 2, 5, i + 1)
        fr = da.isel(day=i)
        im = mesh(ax, fr, cmap=cmap, norm=norm_fn(i, fr))
        ims.append(im)
        if shared_cbar:
            ax.set_title(f"t{i + 1} · {date_label(d)}", fontsize=9.5)
        else:
            n = im.norm
            ax.set_title(f"t{i + 1} · {date_label(d)}\nrange {n.vmin:.1f} to {n.vmax:.1f}", fontsize=9)
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, fraction=0.06, aspect=22, extend=extend)
            cb.ax.tick_params(labelsize=7)
            cb.locator = MaxNLocator(nbins=5)
            cb.update_ticks()
    if shared_cbar:
        cax = fig.add_axes([0.25, 0.06, 0.5, 0.028])
        cb = fig.colorbar(ims[0], cax=cax, orientation="horizontal", extend=extend)
        cb.set_label(cbar_label)
    else:
        fig.text(0.5, 0.005, cbar_label, ha="center", fontsize=9)
    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.14 if shared_cbar else 0.05, wspace=0.03, hspace=0.14)
    stamp(fig, SOURCE)
    return save(fig, name)


def gif(da, title, cmap, norm_fn, cbar_label, name, extend="neither"):
    frames = []
    for i, d in enumerate(da.day.values):
        fig = plt.figure(figsize=(8.4, 5.7))
        ax = map_axes(fig, 1, 1, 1)
        fr = da.isel(day=i)
        im = mesh(ax, fr, cmap=cmap, norm=norm_fn(i, fr))
        ax.set_title(f"{title}\nt{i + 1} of 10 · {date_label(d)}", fontsize=11)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.05, aspect=40, extend=extend)
        cb.set_label(cbar_label)
        stamp(fig, SOURCE)
        frames.append(fig_to_image(fig, dpi=95))
    save_gif(frames, name)


def fig02_palettes(t):
    d = 2  # t3 = 21 Jul 2019, peak of the central/eastern heat wave
    fr = t.isel(day=d)
    vmin, vmax = np.floor(float(fr.min())), np.ceil(float(fr.max()))
    specs = [
        ("viridis", mcolors.Normalize(vmin, vmax), "(a) viridis – sequential, perceptually uniform"),
        ("YlOrRd", mcolors.Normalize(vmin, vmax), "(b) YlOrRd – sequential, multi-hue (ColorBrewer)"),
        ("jet", mcolors.Normalize(vmin, vmax), "(c) jet – rainbow (non-uniform lightness)"),
        ("RdYlBu_r", mcolors.TwoSlopeNorm(35, vmin, vmax), "(d) RdYlBu_r – diverging, centred at 35 °C"),
    ]
    fig = plt.figure(figsize=(14, 11.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.62], hspace=0.22, wspace=0.05)
    for k, (cm, norm, lab) in enumerate(specs):
        ax = fig.add_subplot(gs[k // 2, k % 2], projection=PROJ)
        add_features(ax)
        im = mesh(ax, fr, cmap=cm, norm=norm)
        ax.set_title(lab, fontsize=10)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, fraction=0.05, aspect=35)
        cb.set_label(TMMX_LABEL)
    ax = fig.add_subplot(gs[2, :])
    x = np.linspace(0, 1, 256)
    for cm, _, lab in specs:
        L = srgb_lightness(plt.get_cmap(cm))
        ax.plot(x, L, lw=2.2, label=lab.split(" – ")[0][4:], color=plt.get_cmap(cm)(0.8 if cm != "RdYlBu_r" else 0.1))
        ax.scatter(x[::8], L[::8], c=plt.get_cmap(cm)(x[::8]), s=18, zorder=3, edgecolors="none")
    ax.set_xlabel("Normalised data value (0 = minimum, 1 = maximum of the colour range)")
    ax.set_ylabel("CIE L* lightness")
    ax.set_title("(e) Lightness profile of each colormap (dots show the colours themselves)", fontsize=10)
    ax.legend(ncol=4, frameon=False, loc="lower center")
    ax.grid(lw=0.3, alpha=0.6)
    ax.set_ylim(0, 100)
    fig.suptitle(f"Colormap palettes for daily maximum temperature, t3 = {date_label(t.day.values[d])} "
                 "(heat wave over the central and eastern US)", y=0.93)
    stamp(fig, SOURCE)
    save(fig, "Fig02_colormap_palettes_tmmx_21Jul2019")


def fig03_discrete(t):
    d = 9  # t10 = 29 Sep 2019, strongest north-south contrast of the quarter
    fr = t.isel(day=d)
    vals = fr.values[np.isfinite(fr.values)]
    lo, hi = np.floor(vals.min() / 5) * 5, np.ceil(vals.max() / 5) * 5
    eq_bounds = np.arange(lo, hi + 0.1, 5)
    q_bounds = np.unique(np.round(np.quantile(vals, np.linspace(0, 1, 9)), 1))
    cm = "RdYlBu_r"
    specs = [
        (plt.get_cmap(cm), mcolors.Normalize(lo, hi), "(a) Continuous"),
        (plt.get_cmap(cm, len(eq_bounds) - 1), mcolors.BoundaryNorm(eq_bounds, len(eq_bounds) - 1),
         f"(b) Discrete, equal interval: {len(eq_bounds) - 1} classes of 5 °C"),
        (plt.get_cmap(cm, len(q_bounds) - 1), mcolors.BoundaryNorm(q_bounds, len(q_bounds) - 1),
         f"(c) Discrete, quantile: {len(q_bounds) - 1} classes of equal area"),
    ]
    fig = plt.figure(figsize=(17, 5.0))
    for k, (cmap, norm, lab) in enumerate(specs):
        ax = map_axes(fig, 1, 3, k + 1)
        im = mesh(ax, fr, cmap=cmap, norm=norm)
        ax.set_title(lab, fontsize=10)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, fraction=0.06, aspect=30,
                          spacing="proportional")
        cb.set_label(TMMX_LABEL)
        if k == 2:
            cb.set_ticks(q_bounds)
            cb.ax.set_xticklabels([f"{b:.0f}" for b in q_bounds], fontsize=7)
    fig.suptitle(f"Continuous vs discrete colour mapping of daily maximum temperature, t10 = {date_label(t.day.values[d])}",
                 y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.06, wspace=0.04)
    stamp(fig, SOURCE)
    save(fig, "Fig03_discrete_vs_continuous_tmmx_29Sep2019")
    return eq_bounds, q_bounds


def fig06_ranges(t, gmin, gmax, rlo, rhi):
    fig, ax = plt.subplots(figsize=(12, 5))
    data = [t.isel(day=i).values[np.isfinite(t.isel(day=i).values)] for i in range(t.sizes["day"])]
    pos = np.arange(1, len(data) + 1)
    vp = ax.violinplot(data, pos, widths=0.8, showextrema=False)
    for b in vp["bodies"]:
        b.set_facecolor("#9ecae1")
        b.set_alpha(0.7)
    mins = [x.min() for x in data]
    maxs = [x.max() for x in data]
    ax.vlines(pos, mins, maxs, color="#08519c", lw=1.5, label="local range (min–max of the time step)")
    ax.scatter(pos, mins, color="#08519c", marker="v", zorder=3)
    ax.scatter(pos, maxs, color="#08519c", marker="^", zorder=3)
    ax.axhline(gmin, color="#cb181d", ls="--", lw=1.2, label=f"global min / max over all time steps ({gmin:.1f}, {gmax:.1f} °C)")
    ax.axhline(gmax, color="#cb181d", ls="--", lw=1.2)
    ax.axhspan(rlo, rhi, color="#fdae6b", alpha=0.18, label=f"robust global range, 1st–99th percentile ({rlo:.1f}, {rhi:.1f} °C)")
    for i, (a, b) in enumerate(zip(mins, maxs)):
        used = (b - a) / (gmax - gmin)
        ax.text(pos[i], gmax + 2.2, f"{used:.0%}", ha="center", fontsize=8, color="#cb181d")
    ax.text(0.5, gmax + 2.8, "share of the global\ncolour range used:", ha="right", va="center", fontsize=7.5,
            color="#cb181d")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"t{i + 1}\n{date_label(d)[:6]}" for i, d in enumerate(t.day.values)])
    ax.set_ylabel(TMMX_LABEL)
    ax.set_xlim(-0.2, len(data) + 0.6)
    ax.set_ylim(gmin - 3, gmax + 5.5)
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    ax.grid(axis="y", lw=0.3, alpha=0.6)
    ax.set_title("Distribution of daily maximum temperature over CONUS grid cells per time step "
                 "(violins) vs local and global normalisation ranges")
    stamp(fig, SOURCE)
    save(fig, "Fig06_tmmx_local_vs_global_ranges")


def fig08_precip(p):
    d = 8  # t9 = 19 Sep 2019, Tropical Storm Imelda over SE Texas / SW Louisiana
    fr = p.isel(day=d)
    vmax = float(fr.max())
    wet = fr.where(fr >= 0.1)
    bounds = [0.1, 1, 2.5, 5, 10, 25, 50, 100, 200, 400]
    fig = plt.figure(figsize=(14, 10.2))
    gs = fig.add_gridspec(2, 2, hspace=0.16, wspace=0.05)
    specs = [
        ("(a) Linear: 0 – max", fr, plt.get_cmap("YlGnBu"), mcolors.Normalize(0, vmax), "neither"),
        ("(b) Logarithmic: 0.1 – max (dry cells < 0.1 mm grey)", wet, plt.get_cmap("YlGnBu"),
         mcolors.LogNorm(0.1, vmax), "neither"),
        ("(c) Discrete, rainfall-intensity classes", wet, plt.get_cmap("YlGnBu", len(bounds) - 1),
         mcolors.BoundaryNorm(bounds, len(bounds) - 1), "neither"),
    ]
    for k, (lab, da, cmap, norm, ext) in enumerate(specs):
        ax = fig.add_subplot(gs[k // 2, k % 2], projection=PROJ)
        add_features(ax)
        im = mesh(ax, da, cmap=cmap, norm=norm)
        ax.set_title(lab, fontsize=10)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, fraction=0.05, aspect=35, extend=ext)
        cb.set_label(PR_LABEL)
        if k == 2:
            cb.set_ticks(bounds)
            cb.ax.set_xticklabels([str(b) for b in bounds], fontsize=7.5)
    ax = fig.add_subplot(gs[1, 1])
    v = fr.values[np.isfinite(fr.values)]
    dry = (v < 0.1).mean()
    bins = np.logspace(-1, np.log10(vmax), 45)
    ax.hist(v[v >= 0.1], bins=bins, color="#41b6c4", edgecolor="white", lw=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(PR_LABEL + " – log scale")
    ax.set_ylabel("Number of grid cells (log scale)")
    ax.set_title(f"(d) Distribution of wet cells: {dry:.0%} of land cells are dry (< 0.1 mm)", fontsize=10)
    for q in [0.5, 0.9, 0.99]:
        val = np.quantile(v[v >= 0.1], q)
        ax.axvline(val, color="#253494", ls="--", lw=1)
        ax.text(val, ax.get_ylim()[1] * 0.6, f" p{int(q * 100)}={val:.1f}", fontsize=7.5, color="#253494")
    ax.axvline(vmax, color="#c51b7d", lw=1.2)
    ax.text(vmax, 0.04, f"max {vmax:.0f} ", ha="right", fontsize=7.5, color="#c51b7d", transform=ax.get_xaxis_transform())
    ax.text(0.11, 0.5, "gaps below 0.3 mm:\nvalues are stored\nin 0.1 mm steps", fontsize=7, color="#555555",
            transform=ax.get_xaxis_transform())
    ax.grid(lw=0.3, alpha=0.5)
    fig.suptitle(f"Linear, logarithmic and discrete colour mapping of precipitation, t9 = {date_label(p.day.values[d])} "
                 "(Tropical Storm Imelda)", y=0.93)
    stamp(fig, SOURCE)
    save(fig, "Fig08_precip_linear_log_discrete_19Sep2019")
    return dry


def main() -> None:
    t = load("tmmx")
    p = load("pr")
    fig02_palettes(t)
    fig03_discrete(t)

    vals = t.values[np.isfinite(t.values)]
    gmin, gmax = float(vals.min()), float(vals.max())
    rlo, rhi = np.percentile(vals, [1, 99])
    print(f"tmmx global min/max {gmin:.2f} {gmax:.2f}; robust p1/p99 {rlo:.2f} {rhi:.2f}")
    gnorm = lambda i, fr: mcolors.Normalize(gmin, gmax)  # noqa: E731
    lnorm = lambda i, fr: mcolors.Normalize(float(fr.min()), float(fr.max()))  # noqa: E731
    small_multiples(t, f"Global normalisation: one colour scale for all 10 time steps (min {gmin:.1f} °C, max {gmax:.1f} °C)",
                    "inferno", gnorm, TMMX_LABEL, "Fig04_tmmx_global_minmax_small_multiples")
    gif(t, "Daily max. temperature – global min/max colour scale", "inferno", gnorm, TMMX_LABEL,
        "Fig04_tmmx_global_minmax_animation")
    small_multiples(t, "Local normalisation: each time step stretched to its own min/max (note the different colour bars)",
                    "inferno", lnorm, TMMX_LABEL, "Fig05_tmmx_local_minmax_small_multiples", shared_cbar=False)
    gif(t, "Daily max. temperature – local (per-day) min/max colour scale", "inferno", lnorm, TMMX_LABEL,
        "Fig05_tmmx_local_minmax_animation")
    fig06_ranges(t, gmin, gmax, rlo, rhi)

    anom = t - t.mean(dim="day")
    amax = float(np.nanpercentile(np.abs(anom.values), 99.5))
    amax = np.ceil(amax)
    anorm = lambda i, fr: mcolors.TwoSlopeNorm(0, -amax, amax)  # noqa: E731
    small_multiples(anom, f"Anomaly from the mean of the 10 sampled days, diverging RdBu_r centred at 0 (clipped at ±{amax:.0f} °C)",
                    "RdBu_r", anorm, "Max. temperature anomaly (°C)", "Fig07_tmmx_anomaly_diverging_small_multiples",
                    extend="both")
    gif(anom, "Max. temperature anomaly vs. Jul–Sep 2019 sampled mean", "RdBu_r", anorm, "Anomaly (°C)",
        "Fig07_tmmx_anomaly_diverging_animation", extend="both")

    dry = fig08_precip(p)
    pv = p.values[np.isfinite(p.values)]
    pmax = float(pv.max())
    pw = p.where(p >= 0.1)
    plog = lambda i, fr: mcolors.LogNorm(0.1, pmax)  # noqa: E731
    small_multiples(pw, f"Precipitation with a global logarithmic colour scale (0.1 – {pmax:.0f} mm/day; dry cells grey)",
                    "YlGnBu", plog, PR_LABEL, "Fig09_precip_global_log_small_multiples")
    gif(pw, "Daily precipitation – global logarithmic colour scale", "YlGnBu", plog, PR_LABEL,
        "Fig09_precip_global_log_animation")

    lines = [f"tmmx global min {gmin:.2f} max {gmax:.2f} robust p1 {rlo:.2f} p99 {rhi:.2f}",
             f"anomaly clip +-{amax:.0f}", f"pr global max {pmax:.1f}; dry share on t9 {dry:.3f}"]
    for i, d in enumerate(t.day.values):
        fr = t.isel(day=i)
        a = anom.isel(day=i)
        lines.append(f"t{i + 1} {date_label(d)} tmmx min {float(fr.min()):.1f} max {float(fr.max()):.1f} "
                     f"mean {float(fr.mean()):.1f} | anomaly min {float(a.min()):.1f} max {float(a.max()):.1f} | "
                     f"pr max {float(p.isel(day=i).max()):.1f} wet>=1mm {float((p.isel(day=i) >= 1).sum() / np.isfinite(p.isel(day=i)).sum()):.3f}")
    (config.DATA_PROCESSED / "scivis_colormapping_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
