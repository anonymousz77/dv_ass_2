"""Build the data files for the browser visualisations of InfoVis Task 2 (Lakshya Jain, BT2024044).

  code/web/data/pcp_data.js      multivariate tables for the parallel-coordinates plot
                                  (a) one row per country, 1970-2022 (167 countries, 12 dimensions)
                                  (b) one row per country and decade, 1970s-2010s (country-decades with >= 3 events)
                                  plus k-means clusters (k chosen by silhouette) used for edge bundling
  code/web/data/treemap_data.js  dictionary-encoded rollup (decade x country x hazard subtype) of events, deaths,
                                  people affected and damage, 1900-2022, from which the browser builds several
                                  hierarchies (tabular data -> tree remodels)

Usage:  python code/infovis/build_web_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

rng = np.random.default_rng(2026)
log = []

DIMENSIONS = [  # key, label, scale, format
    ("events", "Events", "log", "int"),
    ("deaths", "Deaths", "log", "int"),
    ("affected", "People affected", "log", "si"),
    ("damage", "Damage (2022 US$)", "log", "usd"),
    ("deaths_per_event", "Deaths per event", "log", "1f"),
    ("share_hydrological", "Hydrological share", "linear", "pct"),
    ("share_meteorological", "Meteorological share", "linear", "pct"),
    ("share_geophysical", "Geophysical share", "linear", "pct"),
    ("share_climatological", "Climatological share", "linear", "pct"),
    ("n_types", "Hazard types (count)", "linear", "int"),
    ("growth", "Growth: events 1996–2022 / 1970–1995", "log", "2f"),
    ("damage_reported", "Records with damage reported", "linear", "pct"),
]


def kmeans(X, k, n_init=25, iters=200):
    best = None
    for _ in range(n_init):
        c = [X[rng.integers(len(X))]]
        for _ in range(1, k):                                  # k-means++ initialisation
            d2 = np.min(((X[:, None, :] - np.array(c)[None]) ** 2).sum(-1), axis=1)
            c.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        c = np.array(c)
        for _ in range(iters):
            lab = np.argmin(((X[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
            new = np.array([X[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = ((X - c[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab, c)
    return best[1], best[2]


def silhouette(X, lab):
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    s = []
    for i in range(len(X)):
        same = lab == lab[i]
        a = D[i, same & (np.arange(len(X)) != i)].mean() if same.sum() > 1 else 0
        b = min(D[i, lab == j].mean() for j in set(lab) if j != lab[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def view_dims(view):
    if view == "country":
        return DIMENSIONS
    return [d for d in DIMENSIONS if d[0] != "growth"] + [("decade", "Decade", "linear", "decade")]


def feature_matrix(df, dims):
    cols = []
    for key, _, scale, _ in dims:
        if key == "decade":      # time is shown as an axis but not used to form clusters
            continue
        v = df[key].astype(float).values
        cols.append(np.log10(1 + v) if scale == "log" else v)
    X = np.column_stack(cols)
    return (X - X.mean(0)) / X.std(0)


def describe_clusters(lab, X, dims):
    names = []
    used = [d for d in dims if d[0] != "decade"]
    for j in range(lab.max() + 1):
        z = X[lab == j].mean(0)
        order = np.argsort(-np.abs(z))[:2]
        parts = [f"{'high' if z[o] > 0 else 'low'} {used[o][1].lower()}" for o in order]
        names.append(", ".join(parts))
    return names


def country_table(df):
    prof = pd.read_csv(config.DATA_PROCESSED / "emdat_country_1970_2022.csv")
    t = pd.DataFrame({
        "id": prof.iso, "name": prof.country, "continent": prof.continent, "subregion": prof.subregion,
        "events": prof.events, "deaths": prof.deaths, "affected": prof.affected, "damage": prof.damage_usd_adj,
        "deaths_per_event": prof.deaths_per_event,
        "share_hydrological": prof.share_hydrological, "share_meteorological": prof.share_meteorological,
        "share_geophysical": prof.share_geophysical, "share_climatological": prof.share_climatological,
        "n_types": prof.n_types,
        "growth": (prof.events_1996_2022 + 1) / (prof.events_1970_1995 + 1),
        "damage_reported": prof.damage_reported_share,
    })
    return t


def decade_table(df):
    w = df[(df.year >= 1970) & (df.year <= 2019) & ~df.historical]
    keep = pd.read_csv(config.DATA_PROCESSED / "emdat_country_1970_2022.csv").iso
    w = w[w.iso.isin(keep)]
    g = w.groupby(["iso", "decade"])
    t = pd.DataFrame({
        "events": g.events.sum(),
        "deaths": g.deaths.sum(min_count=1).fillna(0),
        "affected": g.affected.sum(min_count=1).fillna(0),
        "damage": g.damage_usd_adj.sum(min_count=1).fillna(0),
        "n_types": g.type.nunique(),
        "damage_reported": g.damage_usd_adj.apply(lambda s: s.notna().mean()),
        "name": g.country_short.first(), "continent": g.continent.first(), "subregion": g.subregion.first(),
    })
    sub = w.pivot_table(index=["iso", "decade"], columns="subgroup", values="events", aggfunc="sum", fill_value=0)
    for s in ["Hydrological", "Meteorological", "Geophysical", "Climatological"]:
        t[f"share_{s.lower()}"] = sub[s] / t.events
    t["deaths_per_event"] = t.deaths / t.events
    t = t[t.events >= 3].reset_index()
    t["id"] = t.iso + "-" + t.decade.astype(str)
    t["name"] = t.name + " " + t.decade.astype(str) + "s"
    return t


def cluster(t, dims, kmin=3, kmax=7):
    X = feature_matrix(t, dims)
    scores = {}
    labs = {}
    for k in range(kmin, kmax + 1):
        lab, _ = kmeans(X, k)
        scores[k] = silhouette(X, lab)
        labs[k] = lab
    k_best = max(scores, key=scores.get)
    if scores[k_best] - scores.get(5, -1) < 0.02:   # prefer 5 clusters when within noise of the best
        k_best = 5
    lab = labs[k_best]
    order = np.argsort(-np.bincount(lab))          # cluster 0 = largest
    remap = {old: new for new, old in enumerate(order)}
    lab = np.array([remap[v] for v in lab])
    return lab, k_best, scores, X


def pcp_payload(df):
    out = {"views": {}}
    for view, t in [("country", country_table(df)), ("decade", decade_table(df))]:
        dims = view_dims(view)
        lab, k, scores, X = cluster(t, dims)
        names = describe_clusters(lab, X, dims)
        t["cluster"] = lab
        log.append(f"PCP {view}: {len(t)} rows, k-means silhouette {dict((kk, round(v, 3)) for kk, v in scores.items())} -> k={k}")
        for j, n in enumerate(names):
            members = t[t.cluster == j]
            log.append(f"   cluster {j + 1} ({len(members)}): {n}; e.g. {', '.join(members.sort_values('events', ascending=False).name.head(6))}")
        rows = []
        for r in t.itertuples():
            row = dict(id=r.id, name=r.name, continent=r.continent, subregion=r.subregion, cluster=int(r.cluster))
            for key, _, _, _ in dims:
                v = float(getattr(r, key))
                row[key] = round(v, 4) if abs(v) < 100 else round(v)
            rows.append(row)
        out["views"][view] = dict(dimensions=[dict(key=k, label=lb, scale=sc, format=f) for k, lb, sc, f in dims],
                                  rows=rows, clusters=[dict(id=j, label=f"Cluster {j + 1}: {n}") for j, n in enumerate(names)],
                                  k=int(k))
        t.to_csv(config.DATA_PROCESSED / f"pcp_{view}_table.csv", index=False)
    return out


def treemap_payload(df):
    w = df[df.year <= 2022]
    keys = ["decade", "continent", "subregion", "country_short", "subgroup", "type", "subtype"]
    g = w.groupby(keys, dropna=False)
    agg = pd.DataFrame({
        "events": g.events.sum(), "deaths": g.deaths.sum(min_count=1).fillna(0),
        "affected": g.affected.sum(min_count=1).fillna(0), "damage": g.damage_usd_adj.sum(min_count=1).fillna(0),
    }).reset_index()
    dicts, rows = {}, []
    for k in keys[1:]:
        dicts[k] = sorted(agg[k].unique().tolist())
    idx = {k: {v: i for i, v in enumerate(dicts[k])} for k in dicts}
    for r in agg.itertuples(index=False):
        rec = [int(r.decade)] + [idx[k][getattr(r, k)] for k in keys[1:]]
        rec += [int(r.events), int(r.deaths), int(r.affected), int(round(r.damage / 1e6))]   # damage in million US$
        rows.append(rec)
    log.append(f"treemap rollup: {len(rows)} rows (decade x country x subtype), 1900-2022")
    return dict(columns=keys + ["events", "deaths", "affected", "damage_musd"], dicts=dicts, rows=rows)


def main() -> None:
    df = pd.read_csv(config.DATA_PROCESSED / "emdat_clean.csv")
    config.WEB_DATA.mkdir(parents=True, exist_ok=True)
    pcp = pcp_payload(df)
    (config.WEB_DATA / "pcp_data.js").write_text("window.PCP_DATA = " + json.dumps(pcp, ensure_ascii=False) + ";\n",
                                                 encoding="utf-8")
    tm = treemap_payload(df)
    (config.WEB_DATA / "treemap_data.js").write_text("window.TREEMAP_DATA = " + json.dumps(tm, ensure_ascii=False) + ";\n",
                                                     encoding="utf-8")
    (config.DATA_PROCESSED / "web_data_log.txt").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
