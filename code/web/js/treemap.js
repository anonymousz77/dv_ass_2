// Zoomable treemap: the same tabular rollup is remodelled into several trees; two levels are drawn at a time.
(() => {
  const RAW = window.TREEMAP_DATA;
  const { param, duration, fmt, showTip, hideTip, continentColors } = Common;

  // ---- decode the dictionary-encoded rollup into records
  const C = RAW.columns;
  const records = RAW.rows.map((r) => {
    const o = {};
    C.forEach((c, i) => {
      if (c === "decade" || ["events", "deaths", "affected", "damage_musd"].includes(c)) o[c] = r[i];
      else o[c] = RAW.dicts[c][r[i]];
    });
    o.damage = o.damage_musd * 1e6;
    o.decadeLabel = o.decade === 2020 ? "2020–22" : `${o.decade}s`;
    return o;
  });

  const HIER = {
    geo: ["continent", "subregion", "country_short", "type"],
    hazard: ["subgroup", "type", "subtype", "continent"],
    hazgeo: ["subgroup", "type", "continent", "country_short"],
    time: ["decadeLabel", "subgroup", "type", "subtype"],
  };
  const LEVEL_NAMES = { continent: "continent", subregion: "sub-region", country_short: "country", type: "hazard type",
    subgroup: "hazard group", subtype: "subtype", decadeLabel: "decade" };
  const PLURAL = { continent: "continents", subregion: "sub-regions", country_short: "countries", type: "hazard types",
    subgroup: "hazard groups", subtype: "subtypes", decadeLabel: "decades" };
  const TILES = { squarify: d3.treemapSquarify, binary: d3.treemapBinary, sliceDice: d3.treemapSliceDice,
    slice: d3.treemapSlice, dice: d3.treemapDice, resquarify: d3.treemapResquarify };
  const subgroupColors = { Hydrological: "#2c7fb8", Meteorological: "#7b3294", Geophysical: "#a6611a",
    Climatological: "#d95f0e", Biological: "#4d9221" };
  const measureFmt = { events: fmt.int, deaths: fmt.int, affected: fmt.si, damage: fmt.usd };
  const measureLabel = { events: "events", deaths: "deaths", affected: "people affected", damage: "damage (2022 US$)" };

  const state = {
    hier: param("hier", "geo"), measure: param("measure", "events"), tile: param("tile", "squarify"),
    period: param("period", "1900"), zoomPath: (param("zoom", "") || "").split(">").filter(Boolean),
  };
  ["hier", "measure", "tile", "period"].forEach((id) => { document.getElementById(id).value = state[id]; });

  const svg = d3.select("#tm");
  let W, H, root, current;

  function size() {
    const box = document.querySelector(".chart").getBoundingClientRect();
    W = Math.max(860, box.width - 24);
    H = Math.max(520, window.innerHeight - 205);
    svg.attr("width", W).attr("height", H);
  }

  function topColor(node) {
    const top = node.ancestors().find((a) => a.depth === 1) || node;
    const key = HIER[state.hier][0];
    if (key === "continent") return continentColors[top.data.name] || "#888";
    if (key === "subgroup") return subgroupColors[top.data.name] || "#888";
    const decades = root.children.map((c) => c.data.name).sort();
    return d3.interpolateViridis(0.12 + 0.8 * decades.indexOf(top.data.name) / Math.max(1, decades.length - 1));
  }

  function build() {
    const recs = records.filter((r) => r.decade >= +state.period);
    const keys = HIER[state.hier];
    const nested = d3.rollup(recs, (v) => d3.sum(v, (r) => r[state.measure]), ...keys.map((k) => (r) => r[k]));
    const toTree = (name, m) => (m instanceof Map
      ? { name, children: Array.from(m, ([k, v]) => toTree(k, v)) }
      : { name, value: m });
    const tree = toTree("All disasters", nested);
    root = d3.hierarchy(tree).sum((d) => d.value || 0).sort((a, b) => b.value - a.value);
    root.each((n) => { if (n.children) n.children = n.children.filter((c) => c.value > 0); if (n.children && !n.children.length) delete n.children; });
    current = root;
    for (const name of state.zoomPath) {
      const next = (current.children || []).find((c) => c.data.name === name);
      if (!next) break;
      current = next;
    }
  }

  function layoutOf(node) {
    const copy = node.copy().sum((d) => d.value || 0).sort((a, b) => b.value - a.value);
    copy.each((n) => { if (n.children) n.children = n.children.filter((c) => c.value > 0); if (n.children && !n.children.length) delete n.children; });
    d3.treemap().tile(state.tile === "squarify" ? d3.treemapSquarify.ratio(1.618) : TILES[state.tile])
      .size([W, H]).paddingInner(2).paddingOuter(2)
      .paddingTop((n) => (n.depth === 1 ? 19 : n.depth === 0 ? 0 : 2)).round(true)(copy);
    return copy;
  }

  function pathOf(node) { return node.ancestors().reverse().slice(1).map((a) => a.data.name); }

  function render(fromRect) {
    size();
    const lay = layoutOf(current);
    const depth1 = lay.children || [];
    const depth2 = lay.descendants().filter((d) => d.depth === 2);
    svg.selectAll("g.view").transition().duration(duration(250)).style("opacity", 0).remove();
    const g = svg.append("g").attr("class", "view");
    const curPath = pathOf(current);
    const nodeFor = (d) => {          // map the layout copy back to the node in the full hierarchy
      let n = current;
      for (const nm of d.ancestors().reverse().slice(1).map((a) => a.data.name)) n = (n.children || []).find((c) => c.data.name === nm) || n;
      return n;
    };
    const total = root.value;
    const tip = (event, d) => {
      const n = nodeFor(d);
      const parentVal = n.parent ? n.parent.value : n.value;
      const lvl = HIER[state.hier][n.depth - 1];
      showTip(event, `<div class="t">${pathOf(n).join(" › ")}</div>
        <table><tr><td>${LEVEL_NAMES[lvl] || ""}</td><td class="v">${n.data.name}</td></tr>
        <tr><td>${measureLabel[state.measure]}</td><td class="v">${measureFmt[state.measure](n.value)}</td></tr>
        <tr><td>share of parent</td><td class="v">${d3.format(".1%")(n.value / parentVal)}</td></tr>
        <tr><td>share of total</td><td class="v">${d3.format(".2%")(n.value / total)}</td></tr></table>
        ${n.children ? '<div class="status">click to zoom in</div>' : ""}`);
    };

    const cell = g.selectAll("g.c1").data(depth1).join("g").attr("class", "c1");
    cell.append("rect").attr("x", (d) => d.x0).attr("y", (d) => d.y0)
      .attr("width", (d) => Math.max(0, d.x1 - d.x0)).attr("height", (d) => Math.max(0, d.y1 - d.y0))
      .attr("fill", (d) => d3.color(topColor(nodeFor(d))).darker(0.35)).attr("stroke", "#fff")
      .style("cursor", (d) => (nodeFor(d).children ? "zoom-in" : "default"))
      .on("mousemove", tip).on("mouseleave", hideTip).on("click", (event, d) => zoomTo(nodeFor(d), d));
    cell.append("text").attr("x", (d) => d.x0 + 5).attr("y", (d) => d.y0 + 14).attr("fill", "#fff")
      .style("font-size", "12px").style("font-weight", 650).style("pointer-events", "none")
      .text((d) => {
        const w = d.x1 - d.x0;
        const label = `${d.data.name} · ${measureFmt[state.measure](d.value)}`;
        if (w < 40 || d.y1 - d.y0 < 18) return "";
        return label.length * 6.6 < w ? label : d.data.name.length * 6.6 < w ? d.data.name : d.data.name.slice(0, Math.floor(w / 7)) + "…";
      });

    const leaf = g.selectAll("g.c2").data(depth2).join("g").attr("class", "c2");
    leaf.append("rect").attr("x", (d) => d.x0).attr("y", (d) => d.y0)
      .attr("width", (d) => Math.max(0, d.x1 - d.x0)).attr("height", (d) => Math.max(0, d.y1 - d.y0))
      .attr("fill", (d) => {
        const base = d3.hsl(topColor(nodeFor(d)));
        const sib = d.parent.children;
        const t = sib.length > 1 ? sib.indexOf(d) / (sib.length - 1) : 0;
        base.l = Math.min(0.88, base.l + 0.12 + 0.28 * t);
        return base;
      })
      .attr("stroke", "#fff").attr("stroke-width", 0.6)
      .style("cursor", (d) => (nodeFor(d).children ? "zoom-in" : "default"))
      .on("mousemove", tip).on("mouseleave", hideTip).on("click", (event, d) => zoomTo(nodeFor(d.parent), d.parent));
    leaf.append("text").attr("x", (d) => d.x0 + 4).attr("y", (d) => d.y0 + 13)
      .style("font-size", "11px").attr("fill", "#1d1d1f").style("pointer-events", "none")
      .each(function (d) {
        const w = d.x1 - d.x0, h = d.y1 - d.y0;
        if (w < 46 || h < 16) return;
        const t = d3.select(this);
        const name = d.data.name.length * 6.1 < w - 6 ? d.data.name : d.data.name.slice(0, Math.max(3, Math.floor((w - 10) / 6.3))) + "…";
        t.append("tspan").text(name);
        if (h > 30) t.append("tspan").attr("x", d.x0 + 4).attr("dy", 13).attr("fill", "#444")
          .text(`${measureFmt[state.measure](d.value)} · ${d3.format(".0%")(d.value / d.parent.value)}`);
      });

    if (fromRect) {
      const sx = (fromRect.x1 - fromRect.x0) / W, sy = (fromRect.y1 - fromRect.y0) / H;
      g.attr("transform", `translate(${fromRect.x0},${fromRect.y0}) scale(${sx},${sy})`).style("opacity", 0.3)
        .transition().duration(duration(550)).attr("transform", "translate(0,0) scale(1,1)").style("opacity", 1);
    }

    // breadcrumbs and side panel
    const crumbs = d3.select("#crumbs").html("");
    current.ancestors().reverse().forEach((a, i, arr) => {
      if (i) crumbs.append("span").attr("class", "sep").text("›");
      const s = crumbs.append("span").text(a.data.name);
      if (i < arr.length - 1) s.attr("class", "crumb").on("click", () => { current = a; state.zoomPath = pathOf(a); render(); });
      else s.style("font-weight", 650);
    });
    const lvl = HIER[state.hier][current.depth] || "";
    d3.select("#status").html(`<b>${measureFmt[state.measure](current.value)}</b> ${measureLabel[state.measure]}
      (${d3.format(".1%")(current.value / total)} of all)<br>${(current.children || []).length} ${(current.children || []).length === 1 ? LEVEL_NAMES[lvl] || "item" : PLURAL[lvl] || "items"} shown,
      each split by ${LEVEL_NAMES[HIER[state.hier][current.depth + 1]] || "–"}`);
    d3.select("#top").html((current.children || []).slice(0, 8).map((c) =>
      `${c.data.name}: <b>${measureFmt[state.measure](c.value)}</b> (${d3.format(".0%")(c.value / current.value)})`).join("<br>"));
    const tiles = depth2.length ? depth2 : depth1;
    const ar = tiles.filter((d) => d.x1 - d.x0 > 0.5 && d.y1 - d.y0 > 0.5)
      .map((d) => Math.max((d.x1 - d.x0) / (d.y1 - d.y0), (d.y1 - d.y0) / (d.x1 - d.x0)));
    const thin = ar.filter((a) => a > 10).length;
    d3.select("#aspect").html(`${state.tile} · ${ar.length} tiles<br>median aspect ratio <b>${d3.format(".2f")(d3.median(ar) || 0)}</b>,
      mean <b>${d3.format(".2f")(d3.mean(ar) || 0)}</b><br>${thin} sliver tiles (ratio &gt; 10)`);
    document.title = `Zoomable Treemap – ${pathOf(current).join(" › ") || "All disasters"}`;
  }

  function zoomTo(node, rect) {
    if (!node || !node.children) return;
    current = node;
    state.zoomPath = pathOf(node);
    hideTip();
    render(rect);
  }

  document.getElementById("up").addEventListener("click", () => {
    if (current.parent) { current = current.parent; state.zoomPath = pathOf(current); render(); }
  });
  ["hier", "measure", "tile", "period"].forEach((id) => document.getElementById(id).addEventListener("change", (e) => {
    state[id] = e.target.value;
    if (id === "hier") state.zoomPath = [];
    build(); render();
  }));
  window.addEventListener("resize", () => render());
  build();
  render();
})();
