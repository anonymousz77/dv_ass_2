"""InfoVis Task 3 - Horizon plots and ridgeline plots of the EM-DAT time series (Arnav, BT2024233).

The A1 dataset is yearly (no month field), so the required variations use different groups and different temporal
granularity (yearly vs 5-year / decade):
  Fig37  horizon: annual number of events per disaster type, 1950-2022 (yearly)
  Fig38  horizon: each type's share of the events, anomaly from its 1970-2022 share (removes the growth in reporting)
  Fig39  horizon: annual deaths per continent as a log-anomaly from the continent's median year, 1970-2022 (yearly)
  Fig40  horizon: events per disaster type at 5-year granularity (annual rate per 5-year bin), 1950-2022
  Fig41  ridgeline: distribution over time (year of occurrence) of the events of each disaster type, 1900-2022
  Fig42  ridgeline: distribution of deaths per event by decade (1950s-2010s)
  Fig43  ridgeline: distribution of people affected per event by continent (group-wise), 1970-2022
2023 is excluded everywhere (partial year in the 2023-04-06 snapshot).

Usage:  python code/infovis/horizon_ridgeline.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SOURCE = "Data: EM-DAT country profiles (A1 dataset, Kaggle snapshot 2023-04-06); 2023 excluded (partial year)"
POS = ["#c6dbef", "#6baed6", "#08519c"]
NEG = ["#fcbba1", "#fb6a4a", "#a50f15"]
CONT_COLORS = {"Africa": "#E69F00", "Americas": "#009E73", "Asia": "#D55E00", "Europe": "#0072B2", "Oceania": "#CC79A7"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
                     "figure.titlesize": 12, "figure.titleweight": "bold"})
log = []


def load():
    df = pd.read_csv(config.DATA_PROCESSED / "emdat_clean.csv")
    return df[df.year <= 2022].copy()


def horizon_row(ax, x, y, vmax, bands=3, step=True):
    bh = vmax / bands
    kw = dict(step="mid") if step else {}
    for b in range(bands):
        ax.fill_between(x, 0, np.clip(y - b * bh, 0, bh), color=POS[b], lw=0, **kw)
        ax.fill_between(x, 0, np.clip(-y - b * bh, 0, bh), color=NEG[b], lw=0, **kw)
    ax.set_ylim(0, bh)
    ax.set_yticks([])
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#dddddd")


def horizon_chart(series, title, fname, vmax_mode="row", vmax=None, label_fmt="{:.0f}", annotate=None, xlabel="Year",
                  note="", legend_pos=None, legend_neg=None, step=True, xticks=None, right_labels=None):
    n = series.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(14, 0.52 * n + 1.6), sharex=True)
    for ax, (name, s) in zip(axes, series.items()):
        vm = s.abs().max() if vmax_mode == "row" else vmax
        horizon_row(ax, s.index.values, s.values, vm, step=step)
        ax.text(-0.005, 0.5, name, transform=ax.transAxes, ha="right", va="center", fontsize=9)
        right = right_labels.get(name, "") if right_labels else ("max " + label_fmt.format(vm)) if vmax_mode == "row" else ""
        ax.text(1.005, 0.5, right, transform=ax.transAxes, ha="left", va="center", fontsize=7.5, color="#555555")
        if annotate is not None and name in annotate:
            for xx, txt in annotate[name]:
                edge = xx <= series.index.min() + 1
                ax.text(xx - 0.45 if edge else xx, 0.5 * ax.get_ylim()[1], txt, fontsize=7,
                        ha="left" if edge else "center", va="center", color="white", fontweight="bold")
    axes[-1].set_xlabel(xlabel)
    if xticks is not None:
        axes[-1].set_xticks(xticks)
    axes[0].set_xlim(series.index.min() - 0.5, series.index.max() + 0.5)
    handles = [Patch(color=c, label=l) for c, l in zip(POS, legend_pos)]
    if legend_neg:
        handles += [Patch(color=c, label=l) for c, l in zip(NEG, legend_neg)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02 - (0.02 if note else 0)))
    fig.suptitle(title, y=0.995)
    if note:
        fig.text(0.5, -0.045, note, ha="center", fontsize=8, color="#444444")
    fig.subplots_adjust(hspace=0.12, left=0.13, right=0.93, top=1 - 0.55 / (0.52 * n + 1.6), bottom=0.9 / (0.52 * n + 1.6))
    fig.text(0.995, -0.075 if note else -0.05, SOURCE, ha="right", fontsize=7, color="#555555")
    fig.savefig(config.IMAGES / f"{fname}.png", dpi=config.FIG_DPI)
    plt.close(fig)
    print("saved", fname)


def fig37_types_yearly(df):
    w = df[df.year >= 1950]
    tab = w.pivot_table(index="year", columns="type", values="events", aggfunc="sum", fill_value=0)
    tab = tab.reindex(range(1950, 2023), fill_value=0)
    order = tab.sum().sort_values(ascending=False).index[:10]
    tab = tab[order]
    peaks = {t: int(tab[t].idxmax()) for t in order}
    first_half = tab.loc[1950:1985].sum() / tab.sum()
    log.append("Fig37 peak years: " + ", ".join(f"{t} {y} ({tab[t].max()})" for t, y in peaks.items()))
    log.append("Fig37 share of 1950-2022 events before 1986: " + ", ".join(f"{t} {v:.1%}" for t, v in first_half.items()))
    horizon_chart(tab, "Horizon chart – annual number of recorded events per disaster type, 1950–2022 (yearly)",
                  "Fig37_horizon_events_by_type_yearly", vmax_mode="row",
                  legend_pos=["0–⅓ of row max", "⅓–⅔", "⅔–max"],
                  note="Each row is scaled to its own maximum (printed on the right) and folded into 3 bands; darker = more events.",
                  xticks=range(1950, 2023, 5))
    return tab


def fig38_type_share_anomaly(df):
    """Share of the recorded events per type relative to the type's 1970-2022 share. Total counts grow ~5x with
    better reporting (Fig37); shares cancel that growth and show which hazards became relatively more frequent."""
    w = df[df.year >= 1970]
    tab = w.pivot_table(index="year", columns="type", values="events", aggfunc="sum", fill_value=0)
    tab = tab.reindex(range(1970, 2023), fill_value=0)
    order = tab.sum().sort_values(ascending=False).index[:10]
    roll = tab.rolling(5, center=True, min_periods=3).sum()     # pooled counts in a centred 5-year window
    share = roll.div(roll.sum(axis=1), axis=0)
    base = tab.sum() / tab.values.sum()
    anom = 100 * (share[order] - base[order])
    dec = tab.groupby((tab.index // 10) * 10).sum()
    dshare = 100 * dec.div(dec.sum(axis=1), axis=0)
    log.append("Fig38 share of events 1970s -> 2010s (%): " + ", ".join(
        f"{t} {dshare.loc[1970, t]:.1f} -> {dshare.loc[2010, t]:.1f}" for t in order))
    log.append("Fig38 1970-2022 share (%): " + ", ".join(f"{t} {100 * base[t]:.1f}" for t in order))
    log.append("Fig38 anomaly range (pp): " + ", ".join(
        f"{t} {anom[t].min():+.1f} ({anom[t].idxmin()}) .. {anom[t].max():+.1f} ({anom[t].idxmax()})" for t in order))
    log.append("Fig38 events per year 1970s vs 2010s: " + f"{tab.loc[1970:1979].sum(axis=1).mean():.0f} vs "
               f"{tab.loc[2010:2019].sum(axis=1).mean():.0f}")
    horizon_chart(anom, "Horizon chart – each disaster type's share of the recorded events, anomaly from its 1970–2022 "
                        "share (5-year window)",
                  "Fig38_horizon_event_share_anomaly_by_type", vmax_mode="common", vmax=7.5,
                  legend_pos=["0 to +2.5 pp", "+2.5 to +5 pp", "> +5 pp"],
                  legend_neg=["0 to −2.5 pp", "−2.5 to −5 pp", "< −5 pp"],
                  right_labels={t: f"mean {100 * base[t]:.1f} %" for t in order},
                  note="Share = events of the type / all events in a centred 5-year window (pooled counts); value = share − "
                       "the type's 1970–2022 share (right), in percentage points. Blue: larger share than usual; red "
                       "(mirrored): smaller. Unlike Fig. 37, the growth in total reporting cancels out.",
                  xticks=range(1970, 2023, 4))


def fig39_continent_deaths(df):
    w = df[(df.year >= 1970)]
    tab = w.groupby(["year", "continent"]).deaths.sum(min_count=1).unstack().reindex(range(1970, 2023)).fillna(0)
    logd = np.log10(tab + 1)
    anom = logd - logd.median()
    annotate, notes = {}, []
    for c in anom.columns:
        top = anom[c].nlargest(2)
        annotate[c] = [(y, f"{int(y)}") for y in top.index]
        notes.append(f"{c}: " + ", ".join(f"{int(y)} ({tab.loc[y, c]:,.0f} deaths, {10 ** top[y]:.0f}x median)" for y in top.index))
    log.append("Fig39 peaks: " + " | ".join(notes))
    log.append("Fig39 median annual deaths: " + ", ".join(f"{c} {10 ** logd[c].median() - 1:,.0f}" for c in anom.columns))
    horizon_chart(anom, "Horizon chart – annual disaster deaths per continent relative to the continent's median year, "
                        "1970–2022 (log scale)",
                  "Fig39_horizon_deaths_by_continent_anomaly", vmax_mode="common", vmax=1.5,
                  legend_pos=["up to 3× median", "3–10×", "> 10× median"],
                  legend_neg=["down to ⅓ of median", "⅓–1/10", "< 1/10"], annotate=annotate,
                  note="Value = log10(deaths in year + 1) − median over 1970–2022; bands of 0.5 log units (≈ 3×), "
                       "blue above and red (mirrored) below the median. Labels mark each continent's two most extreme years.",
                  xticks=range(1970, 2023, 4))
    return tab


def fig40_types_5yr(df, yearly):
    bins = list(range(1950, 2021, 5))
    w = yearly.copy()
    w["bin"] = (w.index // 5) * 5
    agg = w.groupby("bin").sum()
    years_in_bin = w.groupby("bin").size()
    rate = agg.div(years_in_bin, axis=0)         # events per year in each 5-year bin (2020-22 has only 3 years)
    rate.index = rate.index + 2                  # plot at bin centre
    horizon_chart(rate, "Horizon chart – events per disaster type at 5-year granularity (mean events per year in each "
                        "5-year bin), 1950–2022",
                  "Fig40_horizon_events_by_type_5year", vmax_mode="row", label_fmt="{:.1f}/yr",
                  legend_pos=["0–⅓ of row max", "⅓–⅔", "⅔–max"],
                  note="Same data as Fig. 37 aggregated to 5-year bins (last bin 2020–22 averaged over 3 years); "
                       "the year-to-year noise disappears and the long-term shape remains.",
                  xticks=[b + 2 for b in bins], xlabel="5-year bin (centre year)")
    log.append("Fig40 bin with max rate: " + ", ".join(f"{t} {int(rate[t].idxmax()) - 2}-{int(rate[t].idxmax()) + 2}"
                                                       for t in rate.columns))


def ridgeline(groups, title, fname, xlabel, grid, colors, weights=None, xlim=None, overlap=1.6, log_x=False,
              note="", fmt_median="{:.0f}", xticks=None, xticklabels=None):
    n = len(groups)
    fig, ax = plt.subplots(figsize=(12, 0.62 * n + 1.8))
    for i, (name, vals) in enumerate(groups.items()):
        w = None if weights is None else weights[name]
        kde = gaussian_kde(vals, weights=w, bw_method="scott")
        dens = kde(grid)
        dens = dens / dens.max() * overlap
        base = (n - 1 - i)
        ax.fill_between(grid, base, base + dens, color=colors[i], alpha=0.82, lw=0, zorder=2 * (i + 1))
        ax.plot(grid, base + dens, color="white", lw=1.0, zorder=2 * (i + 1) + 1)
        ax.axhline(base, color="#bbbbbb", lw=0.5, zorder=0)
        med = weighted_median(vals, w)
        ax.plot([med, med], [base, base + kde(np.array([med]))[0] / kde(grid).max() * overlap], color="black", lw=1.1,
                zorder=2 * (i + 1) + 1)
        cnt = len(vals) if w is None else int(w.sum())
        ax.text(grid[0], base + 0.18, f"{name}", ha="left", va="bottom", fontsize=9, fontweight="bold", zorder=100)
        ax.text(grid[-1], base + 0.18, f"median {fmt_median.format(10 ** med if log_x else med)} · n = {cnt:,}",
                ha="right", va="bottom", fontsize=7.5, color="#333333", zorder=100)
    ax.set_yticks([])
    ax.set_ylim(-0.1, n - 1 + overlap + 0.2)
    ax.set_xlim(grid[0], grid[-1])
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_xlabel(xlabel)
    if xticks is not None:
        ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    if note:
        fig.text(0.5, -0.03, note, ha="center", fontsize=8, color="#444444")
    fig.text(0.995, -0.06 if note else -0.03, SOURCE, ha="right", fontsize=7, color="#555555")
    fig.savefig(config.IMAGES / f"{fname}.png", dpi=config.FIG_DPI)
    plt.close(fig)
    print("saved", fname)


def weighted_median(vals, w):
    if w is None:
        return float(np.median(vals))
    o = np.argsort(vals)
    cw = np.cumsum(w[o]) / w.sum()
    return float(vals[o][np.searchsorted(cw, 0.5)])


def fig41_years_by_type(df):
    tab = df.groupby("type").events.sum().sort_values(ascending=False)
    types = [t for t in tab.index if tab[t] >= 40]
    meds = {}
    groups, weights = {}, {}
    for t in types:
        s = df[df.type == t]
        groups[t] = s.year.values.astype(float)
        weights[t] = s.events.values.astype(float)
        meds[t] = weighted_median(groups[t], weights[t])
    order = sorted(types, key=lambda t: meds[t])
    groups = {t: groups[t] for t in order}
    weights = {t: weights[t] for t in order}
    cmap = plt.get_cmap("viridis")
    colors = [cmap(0.1 + 0.8 * i / (len(order) - 1)) for i in range(len(order))]
    log.append("Fig41 median event year: " + ", ".join(f"{t} {meds[t]:.0f}" for t in order))
    ridgeline(groups, "Ridgeline – when were the events of each disaster type recorded? (1900–2022)",
              "Fig41_ridgeline_event_years_by_type", "Year of occurrence", np.linspace(1900, 2022, 500), colors,
              weights=weights, overlap=1.35,
              note="Kernel density of the year of occurrence, weighted by the number of events; rows sorted by the "
                   "median year (black tick). Types with ≥ 40 events.")


def fig42_deaths_by_decade(df):
    w = df[(df.decade >= 1950) & (df.decade <= 2010) & df.deaths.notna() & (df.deaths > 0)].copy()
    w["dpe"] = np.log10(w.deaths / w.events)
    groups = {f"{d}s": w[w.decade == d].dpe.values for d in sorted(w.decade.unique())}
    weights = {f"{d}s": w[w.decade == d].events.values.astype(float) for d in sorted(w.decade.unique())}
    allrows = df[(df.decade >= 1950) & (df.decade <= 2010)]
    zero_share = allrows.groupby("decade").apply(lambda s: ((s.deaths == 0) | s.deaths.isna()).mean())
    meds = {k: 10 ** weighted_median(v, weights[k]) for k, v in groups.items()}
    log.append("Fig42 median deaths per event by decade: " + ", ".join(f"{k} {v:.1f}" for k, v in meds.items()))
    log.append("Fig42 share of records with 0 / unreported deaths: " + ", ".join(f"{d}s {v:.1%}" for d, v in zero_share.items()))
    cmap = plt.get_cmap("magma")
    colors = [cmap(0.15 + 0.7 * i / (len(groups) - 1)) for i in range(len(groups))]
    ridgeline(groups, "Ridgeline – distribution of deaths per event, by decade (1950s–2010s)",
              "Fig42_ridgeline_deaths_per_event_by_decade", "Deaths per event (log scale)", np.linspace(-0.5, 5.5, 500),
              colors, weights=weights, log_x=True, fmt_median="{:.0f}",
              xticks=np.arange(0, 6), xticklabels=["1", "10", "100", "1k", "10k", "100k"],
              note="One value per country × year × subtype record with deaths reported (deaths / events), weighted by "
                   "the number of events. Records with zero or unreported deaths are excluded.")


def fig43_affected_by_continent(df):
    w = df[(df.year >= 1970) & df.affected.notna() & (df.affected > 0)].copy()
    w["ape"] = np.log10(w.affected / w.events)
    conts = sorted(w.continent.unique(), key=lambda c: weighted_median(w[w.continent == c].ape.values,
                                                                       w[w.continent == c].events.values.astype(float)))
    groups = {c: w[w.continent == c].ape.values for c in conts}
    weights = {c: w[w.continent == c].events.values.astype(float) for c in conts}
    meds = {c: 10 ** weighted_median(groups[c], weights[c]) for c in conts}
    log.append("Fig43 median people affected per event: " + ", ".join(f"{c} {v:,.0f}" for c, v in meds.items()))
    ridgeline(groups, "Ridgeline – distribution of people affected per event, by continent (1970–2022)",
              "Fig43_ridgeline_affected_per_event_by_continent", "People affected per event (log scale)",
              np.linspace(0, 8.5, 500), [CONT_COLORS[c] for c in conts], weights=weights, log_x=True,
              fmt_median="{:,.0f}", xticks=np.arange(0, 9),
              xticklabels=["1", "10", "100", "1k", "10k", "100k", "1M", "10M", "100M"],
              note="One value per country × year × subtype record with people affected reported, weighted by events; "
                   "rows sorted by the median (black tick).")


def main() -> None:
    df = load()
    yearly = fig37_types_yearly(df)
    fig38_type_share_anomaly(df)
    fig39_continent_deaths(df)
    fig40_types_5yr(df, yearly)
    fig41_years_by_type(df)
    fig42_deaths_by_decade(df)
    fig43_affected_by_continent(df)
    (config.DATA_PROCESSED / "horizon_ridgeline_log.txt").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
