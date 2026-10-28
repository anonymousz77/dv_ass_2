"""InfoVis Task 1 - Node-link diagram and graph-layout comparison with NetworkX (Kalpit, BT2024093).

Network derived from the A1 dataset (EM-DAT country profiles, 1970-2022):
  * node  = country with >= 10 recorded natural-disaster events (historical entities excluded)
  * node profile = share of the country's events in 15 hazard categories (unspecified flood / storm
    events re-assigned in proportion to the country's specified subtypes)
  * edge  = "similar disaster regime": each country is linked to its k = 4 most similar countries
            (Bhattacharyya coefficient between profiles, >= 0.70), plus the maximum spanning tree so the
            graph is connected. Edge weight = similarity, edge distance = 1 - similarity.
Layouts compared: Fruchterman-Reingold, Kamada-Kawai, ForceAtlas2, spectral, circular (grouped by community).

Outputs: images/Fig23-Fig26, data/processed/network_*.csv, data/processed/network_disaster_regimes.gexf (Gephi, with
ForceAtlas2 positions, community colours and sizes), code/web/data/network_data.js (interactive viewer)
Usage:  python code/infovis/network_layouts.py
"""
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

K_NEIGHBOURS = 4
MIN_SIM = 0.70
SEED = 42
SOURCE = "Data: EM-DAT country profiles (A1 dataset), 1970–2022, countries with ≥ 10 events"

FEATURES = ["Riverine flood", "Flash flood", "Coastal flood", "Tropical cyclone", "Extra-tropical storm",
            "Convective storm", "Heat wave", "Cold wave / winter", "Earthquake", "Volcanic activity", "Drought",
            "Landslide", "Avalanche", "Wildfire", "Insect infestation"]
CONTINENT_COLORS = {"Africa": "#E69F00", "Americas": "#009E73", "Asia": "#D55E00", "Europe": "#0072B2",
                    "Oceania": "#CC79A7"}
COMMUNITY_PALETTE = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666",
                     "#1f78b4", "#b15928"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
                     "figure.titlesize": 12, "figure.titleweight": "bold"})
log = []


def feature_of(row):
    t, s = row.type, row.subtype
    if t == "Flood":
        return {"Riverine flood": "Riverine flood", "Flash flood": "Flash flood",
                "Coastal flood": "Coastal flood"}.get(s, "Flood?")
    if t == "Storm":
        return {"Tropical cyclone": "Tropical cyclone", "Extra-tropical storm": "Extra-tropical storm",
                "Convective storm": "Convective storm"}.get(s, "Storm?")
    if t == "Extreme temperature":
        return "Heat wave" if s == "Heat wave" else "Cold wave / winter"
    if t in ("Landslide", "Mass movement (dry)"):
        return "Avalanche" if s == "Avalanche" else "Landslide"
    if t in ("Earthquake", "Volcanic activity", "Drought", "Wildfire", "Insect infestation"):
        return t
    return None   # glacial lake outburst, fog, animal accident: 4 events in total, dropped


def build_profiles():
    df = pd.read_csv(config.DATA_PROCESSED / "emdat_clean.csv")
    prof = pd.read_csv(config.DATA_PROCESSED / "emdat_country_1970_2022.csv").set_index("iso")
    w = df[(df.year >= 1970) & (df.year <= 2022) & ~df.historical & df.iso.isin(prof.index)].copy()
    w["feature"] = w.apply(feature_of, axis=1)
    dropped = int(w.feature.isna().sum())
    w = w.dropna(subset=["feature"])
    counts = w.pivot_table(index="iso", columns="feature", values="events", aggfunc="sum", fill_value=0).astype(float)
    for unspecified, parts in [("Flood?", ["Riverine flood", "Flash flood", "Coastal flood"]),
                               ("Storm?", ["Tropical cyclone", "Extra-tropical storm", "Convective storm"])]:
        glob = counts[parts].sum() / counts[parts].values.sum()
        has_own = counts[parts].sum(axis=1) > 0
        own = counts[parts].div(counts[parts].sum(axis=1).where(has_own, 1), axis=0)
        own.loc[~has_own, parts] = glob.values
        counts[parts] += own.mul(counts[unspecified], axis=0)
        log.append(f"re-assigned {counts[unspecified].sum():.0f} unspecified events of {unspecified[:-1]} "
                   f"(global split {dict(glob.round(3))})")
        counts = counts.drop(columns=unspecified)
    counts = counts.reindex(columns=FEATURES, fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    log.append(f"profiles: {len(shares)} countries x {len(FEATURES)} features; {dropped} rows of rare types dropped")
    return prof, counts, shares


def build_graph(prof, shares):
    P = np.sqrt(shares.values)
    S = np.clip(P @ P.T, 0, 1)                        # Bhattacharyya coefficient
    np.fill_diagonal(S, 0)
    isos = shares.index.tolist()
    G = nx.Graph()
    for iso in isos:
        r = prof.loc[iso]
        G.add_node(iso, name=r.country, continent=r.continent, subregion=r.subregion, events=int(r.events),
                   deaths=float(r.deaths), affected=float(r.affected), damage=float(r.damage_usd_adj))
    knn_edges = 0
    for i, iso in enumerate(isos):
        for j in np.argsort(-S[i])[:K_NEIGHBOURS]:
            if S[i, j] >= MIN_SIM and not G.has_edge(iso, isos[j]):
                G.add_edge(iso, isos[j], weight=float(S[i, j]), kind="knn")
                knn_edges += 1
    full = nx.Graph()
    for i in range(len(isos)):
        for j in range(i + 1, len(isos)):
            full.add_edge(isos[i], isos[j], weight=float(S[i, j]))
    mst_added = 0
    for a, b, d in nx.maximum_spanning_edges(full, weight="weight", data=True):
        if not G.has_edge(a, b):
            G.add_edge(a, b, weight=d["weight"], kind="mst")
            mst_added += 1
    for a, b, d in G.edges(data=True):
        d["dist"] = 1.0 - d["weight"] + 0.02
    log.append(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ({knn_edges} kNN, {mst_added} MST), "
               f"density {nx.density(G):.4f}, connected={nx.is_connected(G)}, "
               f"mean degree {np.mean([d for _, d in G.degree()]):.2f}, max degree {max(d for _, d in G.degree())}, "
               f"clustering {nx.average_clustering(G):.3f}, diameter {nx.diameter(G)}")
    return G, S


def communities(G, shares):
    comms = nx.community.louvain_communities(G, weight="weight", resolution=1.0, seed=SEED)
    comms = sorted(comms, key=len, reverse=True)
    labels = {}
    info = []
    for c, members in enumerate(comms):
        mean = shares.loc[sorted(members)].mean().sort_values(ascending=False)
        top = [f for f in mean.index[:2] if mean[f] > 0.12] or [mean.index[0]]
        lab = " + ".join(top)
        for m in members:
            labels[m] = c
        cont = pd.Series([G.nodes[m]["continent"] for m in members]).value_counts()
        info.append(dict(community=c, label=lab, size=len(members), top_share=round(mean.iloc[0], 3),
                         continents=", ".join(f"{k} {v}" for k, v in cont.items()),
                         members=", ".join(sorted(G.nodes[m]["name"] for m in members))))
    nx.set_node_attributes(G, labels, "community")
    q = nx.community.modularity(G, comms, weight="weight")
    log.append(f"Louvain: {len(comms)} communities, modularity {q:.3f}")
    return comms, pd.DataFrame(info), q


def compute_layouts(G, comms):
    lay, rt = {}, {}

    def timed(name, fn):
        t0 = time.perf_counter()
        pos = fn()
        rt[name] = time.perf_counter() - t0
        arr = np.array([pos[n] for n in G.nodes()], dtype=float)
        arr = (arr - arr.min(0)) / (arr.max(0) - arr.min(0))
        lay[name] = {n: arr[i] for i, n in enumerate(G.nodes())}

    timed("Fruchterman–Reingold", lambda: nx.spring_layout(G, weight="weight", seed=SEED, iterations=500))
    timed("Kamada–Kawai", lambda: nx.kamada_kawai_layout(G, weight="dist"))
    timed("ForceAtlas2", lambda: nx.forceatlas2_layout(G, weight="weight", seed=SEED, max_iter=600, scaling_ratio=2.0,
                                                       gravity=1.0, linlog=False, distributed_action=True,
                                                       strong_gravity=False))
    timed("Spectral", lambda: nx.spectral_layout(G, weight="weight"))
    order = []
    for members in comms:
        order += sorted(members, key=lambda m: (G.nodes[m]["continent"], G.nodes[m]["name"]))
    timed("Circular (by community)", lambda: nx.circular_layout(order))   # node order = community, then continent
    return lay, rt


def crossings(G, pos):
    E = list(G.edges())
    A = np.array([pos[a] for a, b in E])
    B = np.array([pos[b] for a, b in E])
    names = [(a, b) for a, b in E]
    n = len(E)
    i, j = np.triu_indices(n, 1)
    shared = np.array([len({names[x][0], names[x][1]} & {names[y][0], names[y][1]}) > 0 for x, y in zip(i, j)])

    def orient(p, q, r):
        return np.sign((q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0]))
    p1, p2, p3, p4 = A[i], B[i], A[j], B[j]
    cross = (orient(p1, p2, p3) * orient(p1, p2, p4) < 0) & (orient(p3, p4, p1) * orient(p3, p4, p2) < 0)
    return int(np.sum(cross & ~shared))


def layout_metrics(G, lay, rt, comms):
    nodes = list(G.nodes())
    idx = {n: k for k, n in enumerate(nodes)}
    D = np.full((len(nodes), len(nodes)), np.nan)
    for s, dd in nx.shortest_path_length(G):
        for t, d in dd.items():
            D[idx[s], idx[t]] = d
    comm_of = np.array([G.nodes[n]["community"] for n in nodes])
    rows = []
    for name, pos in lay.items():
        X = np.array([pos[n] for n in nodes])
        E = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        iu = np.triu_indices(len(nodes), 1)
        d, e = D[iu], E[iu]
        alpha = np.mean(e / d)   # optimal scale for stress weighted by 1/d^2
        stress = np.mean(((e - alpha * d) / d) ** 2)
        lengths = np.array([np.linalg.norm(pos[a] - pos[b]) for a, b in G.edges()])
        npres = []
        for n in nodes:
            nb = set(G.neighbors(n))
            k = len(nb)
            near = np.argsort(E[idx[n]])[1:k + 1]
            npres.append(len(nb & {nodes[x] for x in near}) / len(nb | {nodes[x] for x in near}))
        sil = []
        for k in range(len(nodes)):
            same = comm_of == comm_of[k]
            a = E[k, same & (np.arange(len(nodes)) != k)].mean() if same.sum() > 1 else 0
            b = min(E[k, comm_of == c].mean() for c in set(comm_of) if c != comm_of[k])
            sil.append((b - a) / max(a, b))
        rows.append(dict(layout=name, runtime_s=rt[name], edge_crossings=crossings(G, pos), normalised_stress=stress,
                         edge_length_cv=lengths.std() / lengths.mean(), neighbourhood_preservation=np.mean(npres),
                         community_silhouette=np.mean(sil)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- drawing
def draw(ax, G, pos, color_by="community", label_top=22, title=None, comm_labels=None, node_scale=1.0):
    ws = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    for (a, b, d), w in zip(G.edges(data=True), ws):
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="#9a9a9a", lw=0.3 + 1.2 * (w - 0.6) / 0.4,
                alpha=0.25 + 0.45 * (w - 0.6) / 0.4 if d["kind"] == "knn" else 0.35,
                ls="-" if d["kind"] == "knn" else (0, (2, 2)), zorder=1)
    nodes = list(G.nodes())
    xy = np.array([pos[n] for n in nodes])
    size = np.array([G.nodes[n]["events"] for n in nodes])
    if color_by == "community":
        cols = [COMMUNITY_PALETTE[G.nodes[n]["community"] % len(COMMUNITY_PALETTE)] for n in nodes]
    else:
        cols = [CONTINENT_COLORS[G.nodes[n]["continent"]] for n in nodes]
    area = node_scale * (8 + 2.2 * np.sqrt(size) ** 1.25)
    ax.scatter(xy[:, 0], xy[:, 1], s=area, c=cols, edgecolors="white", linewidths=0.6, zorder=3)
    radius = dict(zip(nodes, np.sqrt(area) / 2))   # marker radius in points
    top = sorted(nodes, key=lambda n: -G.nodes[n]["events"])[:label_top]
    # labels are placed by place_labels() once the figure layout is final
    ax.node_labels = [(pos[n], G.nodes[n]["name"].replace("United States of America", "USA")
                       .replace("United Kingdom of Great Britain and Northern Ireland", "UK"), radius[n]) for n in top]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.07)
    for s in ax.spines.values():
        s.set_color("#cccccc")
    if title:
        ax.set_title(title, fontsize=10)


def place_labels(fig, fontsize=6.5):
    """Greedy collision-free placement of the node labels of every axes in fig, most events first. Each label tries
    8 directions around its node on 3 rings of increasing distance (a leader line is drawn beyond the first ring);
    it must not overlap a placed label or another labelled node and must stay inside the axes, else it is dropped."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    style = dict(fontsize=fontsize, zorder=4, color="#111111",
                 bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.7))
    stats = []
    for ax in fig.axes:
        items = getattr(ax, "node_labels", None)
        if not items:
            continue
        frame = ax.get_window_extent(renderer)
        nodes = []
        for (x, y), _, r in items:
            cx, cy = ax.transData.transform((x, y))
            nodes.append(Bbox.from_extents(cx - r * px, cy - r * px, cx + r * px, cy + r * px))
        placed, moved, dropped = [], 0, []
        for k, ((x, y), text, r) in enumerate(items):
            spot = None
            for ring, f in enumerate((1.0, 2.3, 3.8)):
                for dx, dy in dirs:
                    d = (r + 2.0) * f + 2.0 * ring
                    off = (dx * d / (1.4 if dx and dy else 1), dy * d / (1.4 if dx and dy else 1))
                    ha = "center" if dx == 0 else "left" if dx > 0 else "right"
                    va = "center" if dy == 0 else "bottom" if dy > 0 else "top"
                    t = ax.annotate(text, (x, y), xytext=off, textcoords="offset points", ha=ha, va=va, **style)
                    bb = t.get_window_extent(renderer).expanded(1.06, 1.15)
                    t.remove()
                    inside = frame.x0 <= bb.x0 and bb.x1 <= frame.x1 and frame.y0 <= bb.y0 and bb.y1 <= frame.y1
                    if inside and not any(bb.overlaps(b) for b in placed) and                             not any(bb.overlaps(b) for j, b in enumerate(nodes) if j != k):
                        spot = (off, ha, va, ring, bb)
                        break
                if spot:
                    break
            if spot is None:
                dropped.append(text)
                continue
            off, ha, va, ring, bb = spot
            arrow = dict(arrowstyle="-", color="#666666", lw=0.5, shrinkA=0, shrinkB=r) if ring else None
            ax.annotate(text, (x, y), xytext=off, textcoords="offset points", ha=ha, va=va, arrowprops=arrow, **style)
            placed.append(bb)
            moved += ring > 0
        stats.append(f"{len(placed)} placed ({moved} with leader line), dropped: {', '.join(dropped) or 'none'}")
    log.append("node labels – " + " | ".join(stats))


def write_gexf(G, pos, path):
    """Export the network for Gephi: all node/edge attributes plus ForceAtlas2 positions, community colours, sizes."""
    H = nx.Graph()
    py = lambda v: v.item() if hasattr(v, "item") else ("" if v is None or (isinstance(v, float) and np.isnan(v)) else v)  # noqa: E731
    for n, a in G.nodes(data=True):
        rgb = matplotlib.colors.to_rgb(COMMUNITY_PALETTE[a["community"] % len(COMMUNITY_PALETTE)])
        H.add_node(n, **{k: py(v) for k, v in a.items()}, label=a["name"], viz={
            "position": {"x": float(pos[n][0]) * 1000, "y": float(pos[n][1]) * 1000, "z": 0.0},
            "size": float(3 + np.sqrt(a["events"])),
            "color": {"r": int(255 * rgb[0]), "g": int(255 * rgb[1]), "b": int(255 * rgb[2]), "a": 1.0}})
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, **{k: py(x) for k, x in d.items()})
    nx.write_gexf(H, path)


def legend_handles(color_by, comm_info=None):
    if color_by == "continent":
        return [Line2D([], [], marker="o", ls="", color=c, ms=7, label=k) for k, c in CONTINENT_COLORS.items()]
    return [Line2D([], [], marker="o", ls="", color=COMMUNITY_PALETTE[r.community % len(COMMUNITY_PALETTE)], ms=7,
                   label=f"C{r.community + 1}: {r.label} ({r.size})") for r in comm_info.itertuples()]


def size_legend():
    return [Line2D([], [], marker="o", ls="", color="#bbbbbb", ms=np.sqrt(8 + 2.2 * np.sqrt(ev) ** 1.25),
                   label=f"{ev} events") for ev in [10, 100, 500]]


def main() -> None:
    prof, counts, shares = build_profiles()
    G, S = build_graph(prof, shares)
    comms, comm_info, q = communities(G, shares)
    lay, rt = compute_layouts(G, comms)
    metrics = layout_metrics(G, lay, rt, comms)
    bc = nx.betweenness_centrality(G, weight="dist")
    nx.set_node_attributes(G, bc, "betweenness")

    # ---- Fig23: Fruchterman-Reingold, coloured by community and by continent
    pos = lay["Fruchterman–Reingold"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    draw(axes[0], G, pos, "community", title="(a) Colour = Louvain community (disaster regime)")
    draw(axes[1], G, pos, "continent", title="(b) Same layout, colour = continent")
    axes[0].legend(handles=legend_handles("community", comm_info), loc="upper center", bbox_to_anchor=(0.5, -0.01),
                   ncol=2, fontsize=8, frameon=False)
    axes[1].legend(handles=legend_handles("continent") + size_legend(), loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4,
                   fontsize=8, frameon=False)
    fig.suptitle(f"Country network of similar natural-disaster regimes – Fruchterman–Reingold layout "
                 f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges; node area ∝ events; dashed = spanning-tree edge)",
                 y=0.97)
    fig.text(0.995, 0.005, SOURCE, ha="right", fontsize=7, color="#555555")
    place_labels(fig)
    fig.savefig(config.IMAGES / "Fig23_network_fruchterman_reingold.png", dpi=config.FIG_DPI)
    plt.close(fig)

    # ---- Fig24: other layouts
    fig, axes = plt.subplots(2, 2, figsize=(16, 15.5))
    for ax, name, tag in zip(axes.ravel(), ["Kamada–Kawai", "ForceAtlas2", "Spectral", "Circular (by community)"], "abcd"):
        draw(ax, G, lay[name], "community", label_top=18, title=f"({tag}) {name} layout  ·  {rt[name] * 1000:.0f} ms",
             node_scale=0.8)
    fig.legend(handles=legend_handles("community", comm_info), loc="lower center", ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("The same network under four more layout algorithms (colour = Louvain community, area ∝ events)", y=0.93)
    fig.subplots_adjust(hspace=0.08, wspace=0.04, bottom=0.08)
    fig.text(0.995, 0.005, SOURCE, ha="right", fontsize=7, color="#555555")
    place_labels(fig)
    fig.savefig(config.IMAGES / "Fig24_network_layout_comparison.png", dpi=config.FIG_DPI)
    plt.close(fig)

    # ---- Fig25: layout quality metrics
    cols = [("edge_crossings", "Edge crossings (lower is better)"),
            ("normalised_stress", "Normalised stress vs hop distance (lower)"),
            ("neighbourhood_preservation", "Neighbourhood preservation (higher)"),
            ("community_silhouette", "Community separation – silhouette (higher)"),
            ("edge_length_cv", "Edge-length variation, CV (lower)"),
            ("runtime_s", "Run time in seconds (lower)")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.6))
    pal = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
    for ax, (c, lab) in zip(axes.ravel(), cols):
        vals = metrics[c].values
        bars = ax.barh(metrics.layout, vals, color=pal)
        ax.invert_yaxis()
        ax.set_title(lab, fontsize=10)
        for b, v in zip(bars, vals):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                    f" {v:,.0f}" if c == "edge_crossings" else f" {v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, vals.max() * 1.25)
        ax.grid(axis="x", lw=0.3, alpha=0.6)
    for ax in axes[:, 1:].ravel():
        ax.set_yticklabels([])
    fig.suptitle("Quantitative comparison of the five graph layouts (positions normalised to the unit square)", y=0.99)
    fig.tight_layout()
    fig.savefig(config.IMAGES / "Fig25_network_layout_metrics.png", dpi=config.FIG_DPI)
    plt.close(fig)

    # ---- Fig26: community profiles
    nodes_df = pd.DataFrame([dict(iso=n, **G.nodes[n], degree=G.degree(n)) for n in G.nodes()])
    prof_c = shares.join(nodes_df.set_index("iso")["community"]).groupby("community").mean()
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), gridspec_kw=dict(width_ratios=[2.3, 1]))
    ax = axes[0]
    im = ax.imshow(prof_c.values, cmap="Purples", aspect="auto", vmin=0, vmax=max(0.5, prof_c.values.max()))
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=40, ha="right")
    ax.set_yticks(range(len(prof_c)))
    ax.set_yticklabels([f"C{r.community + 1} ({r.size})" for r in comm_info.itertuples()])
    for i in range(prof_c.shape[0]):
        for j in range(prof_c.shape[1]):
            v = prof_c.values[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7.5, color="white" if v > 0.3 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("Mean share of the community's events")
    ax.set_title("(a) Mean hazard profile of each community", fontsize=10)
    ax = axes[1]
    comp = nodes_df.groupby(["community", "continent"]).size().unstack(fill_value=0)
    left = np.zeros(len(comp))
    for cont, col in CONTINENT_COLORS.items():
        if cont in comp:
            ax.barh(range(len(comp)), comp[cont], left=left, color=col, label=cont)
            left += comp[cont].values
    ax.set_yticks(range(len(comp)))
    ax.set_yticklabels([f"C{c + 1}" for c in comp.index])
    ax.invert_yaxis()
    ax.set_xlabel("Number of countries")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("(b) Continent composition of each community", fontsize=10)
    fig.suptitle(f"What the Louvain communities mean (modularity Q = {q:.2f})", y=1.0)
    fig.tight_layout()
    fig.text(0.995, -0.01, SOURCE, ha="right", fontsize=7, color="#555555")
    fig.savefig(config.IMAGES / "Fig26_network_community_profiles.png", dpi=config.FIG_DPI)
    plt.close(fig)

    # ---- data for the interactive viewer and the report
    keys = {"Fruchterman–Reingold": "fr", "Kamada–Kawai": "kk", "ForceAtlas2": "fa2", "Spectral": "spectral",
            "Circular (by community)": "circular"}
    nodes_out = []
    for n in G.nodes():
        a = G.nodes[n]
        top = shares.loc[n].sort_values(ascending=False)
        nodes_out.append(dict(id=n, name=a["name"], continent=a["continent"], subregion=a["subregion"],
                              community=int(a["community"]), events=a["events"], deaths=a["deaths"],
                              affected=a["affected"], damage=a["damage"], degree=G.degree(n),
                              betweenness=round(a["betweenness"], 4),
                              profile=[[f, round(float(v), 3)] for f, v in top.items() if v > 0][:5],
                              pos={keys[k]: [round(float(p[n][0]), 4), round(float(p[n][1]), 4)] for k, p in lay.items()}))
    edges_out = [dict(source=a, target=b, weight=round(d["weight"], 3), kind=d["kind"]) for a, b, d in G.edges(data=True)]
    payload = dict(nodes=nodes_out, edges=edges_out,
                   communities=[dict(id=int(r.community), label=r.label, size=int(r.size)) for r in comm_info.itertuples()],
                   layouts=[dict(key=keys[k], name=k) for k in lay], metrics=metrics.round(4).to_dict(orient="records"),
                   palette=COMMUNITY_PALETTE, continentColors=CONTINENT_COLORS)
    config.WEB_DATA.mkdir(parents=True, exist_ok=True)
    (config.WEB_DATA / "network_data.js").write_text("window.NETWORK_DATA = " + json.dumps(payload) + ";\n",
                                                     encoding="utf-8")
    nodes_df.drop(columns=["betweenness"], errors="ignore").assign(betweenness=nodes_df.iso.map(bc)).to_csv(
        config.DATA_PROCESSED / "network_nodes.csv", index=False)
    pd.DataFrame(edges_out).to_csv(config.DATA_PROCESSED / "network_edges.csv", index=False)
    comm_info.to_csv(config.DATA_PROCESSED / "network_communities.csv", index=False)
    metrics.to_csv(config.DATA_PROCESSED / "network_layout_metrics.csv", index=False)
    write_gexf(G, lay["ForceAtlas2"], config.DATA_PROCESSED / "network_disaster_regimes.gexf")
    top_bc = sorted(bc.items(), key=lambda kv: -kv[1])[:8]
    log.append("top betweenness: " + ", ".join(f"{G.nodes[k]['name']} {v:.3f}" for k, v in top_bc))
    log.append(comm_info.to_string())
    log.append(metrics.round(4).to_string())
    (config.DATA_PROCESSED / "network_log.txt").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
