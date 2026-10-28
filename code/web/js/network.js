// Interactive node-link viewer for the NetworkX layouts (layout switching, filtering, focus+context, zoom, drag).
(() => {
  const D = window.NETWORK_DATA;
  const { param, duration, fmt, showTip, hideTip, continentColors } = Common;
  const layouts = D.layouts.concat([{ key: "force", name: "Live force simulation (d3-force)" }]);

  const state = {
    layout: param("layout", "fr"), color: param("color", "community"), size: param("size", "events"),
    minWeight: +param("minWeight", 0.7), minEventsLog: Math.log10(Math.max(10, +param("minEvents", 10))),
    labels: param("labels", "top"), focus: param("focus", ""),
    continents: new Set((param("continents", "") || Object.keys(continentColors).join(",")).split(",")),
  };

  const svg = d3.select("#net");
  let W, H;
  function size() {
    const box = document.querySelector(".chart").getBoundingClientRect();
    W = Math.max(820, box.width - 24);
    H = Math.max(560, window.innerHeight - 200);
    svg.attr("width", W).attr("height", H);
  }
  size();

  const nodes = D.nodes.map((n) => ({ ...n }));
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const links = D.edges.map((e) => ({ ...e, source: byId.get(e.source), target: byId.get(e.target) }));
  const neighbours = new Map(nodes.map((n) => [n.id, new Set()]));
  links.forEach((l) => { neighbours.get(l.source.id).add(l.target.id); neighbours.get(l.target.id).add(l.source.id); });

  const pad = 40;
  const sx = () => d3.scaleLinear().domain([0, 1]).range([pad, W - pad]);
  const sy = () => d3.scaleLinear().domain([0, 1]).range([H - pad, pad]);
  function place(key) {
    const X = sx(), Y = sy();
    nodes.forEach((n) => { const p = n.pos[key] || n.pos.fr; n.x = X(p[0]); n.y = Y(p[1]); });
  }

  const color = (n) => (state.color === "community" ? D.palette[n.community % D.palette.length] : continentColors[n.continent]);
  const sizeScale = () => {
    const ext = d3.extent(nodes, (n) => n[state.size]);
    return d3.scaleSqrt().domain([0, ext[1]]).range([3, state.size === "betweenness" ? 22 : 20]);
  };

  const root = svg.append("g");
  const gLinks = root.append("g");
  const gNodes = root.append("g");
  const gLabels = root.append("g").style("pointer-events", "none");
  const zoom = d3.zoom().scaleExtent([0.4, 8]).on("zoom", (e) => root.attr("transform", e.transform));
  svg.call(zoom).on("dblclick.zoom", null);
  svg.on("click", (e) => { if (e.target === svg.node()) { state.focus = ""; update(); } });

  let sim = null;
  function visibleNode(n) {
    return n.events >= Math.pow(10, state.minEventsLog) - 1e-9 && state.continents.has(n.continent);
  }

  function update(animate = false) {
    const vis = new Set(nodes.filter(visibleNode).map((n) => n.id));
    const visLinks = links.filter((l) => l.weight >= state.minWeight && vis.has(l.source.id) && vis.has(l.target.id));
    const r = sizeScale();
    const focus = state.focus && byId.get(state.focus) ? state.focus : "";
    const nb = focus ? neighbours.get(focus) : null;
    const isOn = (n) => !focus || n.id === focus || nb.has(n.id);
    const t = svg.transition().duration(animate ? duration(800) : 0);

    gLinks.selectAll("line").data(visLinks, (l) => l.source.id + "-" + l.target.id).join(
      (enter) => enter.append("line").attr("x1", (l) => l.source.x).attr("y1", (l) => l.source.y)
        .attr("x2", (l) => l.target.x).attr("y2", (l) => l.target.y),
      (u) => u, (exit) => exit.remove())
      .attr("stroke", (l) => (focus && (l.source.id === focus || l.target.id === focus) ? "#333" : "#999"))
      .attr("stroke-width", (l) => 0.4 + 2.2 * (l.weight - 0.7) / 0.3)
      .attr("stroke-dasharray", (l) => (l.kind === "mst" ? "4,3" : null))
      .attr("stroke-opacity", (l) => (focus ? (l.source.id === focus || l.target.id === focus ? 0.9 : 0.05) : 0.35))
      .transition(t).attr("x1", (l) => l.source.x).attr("y1", (l) => l.source.y)
      .attr("x2", (l) => l.target.x).attr("y2", (l) => l.target.y);

    gNodes.selectAll("circle").data(nodes.filter((n) => vis.has(n.id)), (n) => n.id).join(
      (enter) => enter.append("circle").attr("cx", (n) => n.x).attr("cy", (n) => n.y)
        .attr("stroke", "#fff").attr("stroke-width", 0.8).style("cursor", "pointer")
        .on("mousemove", (e, n) => showTip(e, tip(n))).on("mouseleave", hideTip)
        .on("click", (e, n) => { e.stopPropagation(); state.focus = state.focus === n.id ? "" : n.id; update(); })
        .call(drag),
      (u) => u, (exit) => exit.remove())
      .attr("fill", color).attr("r", (n) => r(n[state.size]))
      .attr("opacity", (n) => (isOn(n) ? 1 : 0.12))
      .attr("stroke", (n) => (n.id === focus ? "#111" : "#fff")).attr("stroke-width", (n) => (n.id === focus ? 2.4 : 0.8))
      .transition(t).attr("cx", (n) => n.x).attr("cy", (n) => n.y);

    const top = new Set(nodes.filter((n) => vis.has(n.id)).sort((a, b) => b.events - a.events).slice(0, 25).map((n) => n.id));
    const labelled = nodes.filter((n) => vis.has(n.id) && (focus ? isOn(n) : state.labels === "all" || (state.labels === "top" && top.has(n.id))));
    placeLabels(labelled, (n) => r(n[state.size]), focus);
    gLabels.selectAll("text").data(labelled, (n) => n.id).join("text")
      .attr("text-anchor", "middle").attr("dominant-baseline", "central")
      .style("font-size", (n) => (n.id === focus ? "13px" : "10.5px"))
      .style("font-weight", (n) => (n.id === focus ? 700 : 500)).attr("fill", "#111")
      .attr("paint-order", "stroke").attr("stroke", "#fff").attr("stroke-width", 3).attr("stroke-opacity", 0.85)
      .text((n) => n.name)
      .transition(t).attr("x", (n) => n.lx).attr("y", (n) => n.ly);

    d3.select("#status").html(`<b>${vis.size}</b> of ${nodes.length} countries, <b>${visLinks.length}</b> of ${links.length} edges<br>
      layout: ${layouts.find((l) => l.key === state.layout).name}`);
    renderInfo(focus);
    renderLegend();
    renderMetrics();
  }

  // Label placement: most events first (focused node before all), each label tries 8 positions around its node on two
  // rings and takes the first that overlaps neither an earlier label nor another labelled node; if none is free, the
  // position with the smallest overlap is used.
  const measure = document.createElement("canvas").getContext("2d");
  const SPOTS = [[0, -1], [0, 1], [1, 0], [-1, 0], [1, -1], [-1, -1], [1, 1], [-1, 1]];
  function placeLabels(list, radius, focus) {
    const family = getComputedStyle(svg.node()).fontFamily;
    const boxes = [];
    const nodeHits = (b, self) => list.filter((o) => {
      if (o === self) return false;
      const cx = Math.max(b[0], Math.min(o.x, b[2])), cy = Math.max(b[1], Math.min(o.y, b[3]));
      return (cx - o.x) ** 2 + (cy - o.y) ** 2 < radius(o) ** 2;
    }).length;
    const overlap = (b) => boxes.reduce((s, o) => s + Math.max(0, Math.min(b[2], o[2]) - Math.max(b[0], o[0]))
      * Math.max(0, Math.min(b[3], o[3]) - Math.max(b[1], o[1])), 0);
    list.slice().sort((a, b) => (b.id === focus) - (a.id === focus) || b.events - a.events).forEach((n) => {
      const fs = n.id === focus ? 13 : 10.5;
      measure.font = `${n.id === focus ? 700 : 500} ${fs}px ${family}`;
      const w = measure.measureText(n.name).width + 10, h = fs + 4, rr = radius(n) + 3;   // + white halo
      let spot = null, best = Infinity;
      for (const ring of [0, h]) {
        for (const [dx, dy] of SPOTS) {
          const diag = dx && dy ? 0.7 : 1, d = rr * diag + ring;
          const cx = n.x + dx * (d + w / 2), cy = n.y + dy * (d + h / 2);
          const b = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
          const cost = overlap(b) + 60 * nodeHits(b, n) + (ring ? 20 : 0);   // small preference for the inner ring
          if (cost < best) { best = cost; spot = b; }
        }
        if (best === 0) break;
      }
      boxes.push(spot);
      n.lx = (spot[0] + spot[2]) / 2;
      n.ly = (spot[1] + spot[3]) / 2;
    });
  }

  function tip(n) {
    const prof = n.profile.map(([f, v]) => `<tr><td>${f}</td><td class="v">${d3.format(".0%")(v)}</td></tr>`).join("");
    return `<div class="t">${n.name}</div><div class="status">${n.subregion} · community C${n.community + 1}</div>
      <table><tr><td>events</td><td class="v">${fmt.int(n.events)}</td></tr><tr><td>deaths</td><td class="v">${fmt.int(n.deaths)}</td></tr>
      <tr><td>degree</td><td class="v">${n.degree}</td></tr><tr><td>betweenness</td><td class="v">${n.betweenness.toFixed(3)}</td></tr></table>
      <div class="status" style="margin-top:4px">hazard profile</div><table>${prof}</table>`;
  }

  function renderInfo(focus) {
    if (!focus) { d3.select("#info").html("<i>Click a node to focus on it and its neighbours.</i>"); return; }
    const n = byId.get(focus);
    const nb = Array.from(neighbours.get(focus)).map((id) => byId.get(id));
    const w = (o) => links.find((l) => (l.source.id === focus && l.target.id === o.id) || (l.target.id === focus && l.source.id === o.id)).weight;
    nb.sort((a, b) => w(b) - w(a));
    const bars = n.profile.map(([f, v]) => `<div style="display:flex;align-items:center;gap:6px"><span style="width:118px">${f}</span>
      <span style="display:inline-block;height:9px;width:${Math.round(v * 120)}px;background:${color(n)}"></span><span>${d3.format(".0%")(v)}</span></div>`).join("");
    d3.select("#info").html(`<b>${n.name}</b> (${n.subregion})<br>C${n.community + 1}: ${D.communities[n.community].label}<br>
      ${fmt.int(n.events)} events · ${fmt.int(n.deaths)} deaths<div style="margin:6px 0">${bars}</div>
      Most similar: ${nb.map((o) => `${o.name} (${w(o).toFixed(2)})`).join(", ")}`);
  }

  function renderLegend() {
    const items = state.color === "community"
      ? D.communities.map((c) => ({ color: D.palette[c.id % D.palette.length], label: `C${c.id + 1}: ${c.label} (${c.size})` }))
      : Object.entries(continentColors).map(([k, v]) => ({ color: v, label: k }));
    d3.select("#legend").selectAll("div").data(items).join("div").attr("class", "legend-item")
      .html((d) => `<span class="swatch" style="background:${d.color}"></span><span>${d.label}</span>`);
    d3.select("#contFilter").selectAll("div").data(Object.keys(continentColors)).join("div")
      .attr("class", (c) => "legend-item" + (state.continents.has(c) ? "" : " off"))
      .html((c) => `<span class="swatch" style="background:${continentColors[c]}"></span><span>${c} (${nodes.filter((n) => n.continent === c).length})</span>`)
      .on("click", (e, c) => { if (state.continents.has(c)) state.continents.delete(c); else state.continents.add(c); update(); });
  }

  function renderMetrics() {
    const rows = D.metrics.map((m) => `<tr${D.layouts.find((l) => l.name === m.layout)?.key === state.layout ? ' style="font-weight:700"' : ""}>
      <td>${m.layout.replace(" (by community)", "")}</td><td class="v">${m.edge_crossings}</td><td class="v">${m.neighbourhood_preservation.toFixed(2)}</td>
      <td class="v">${m.community_silhouette.toFixed(2)}</td></tr>`).join("");
    d3.select("#metrics").html(`<table><tr><td></td><td class="v">cross.</td><td class="v">NP</td><td class="v">sil.</td></tr>${rows}</table>`);
  }

  const drag = d3.drag()
    .on("start", (e, n) => { if (sim) { sim.alphaTarget(0.3).restart(); n.fx = n.x; n.fy = n.y; } })
    .on("drag", (e, n) => {
      n.x = e.x; n.y = e.y;
      if (sim) { n.fx = e.x; n.fy = e.y; } else update();
    })
    .on("end", (e, n) => { if (sim) { sim.alphaTarget(0); n.fx = null; n.fy = null; } });

  function setLayout(key, animate = true) {
    state.layout = key;
    if (sim) { sim.stop(); sim = null; }
    if (key === "force") {
      sim = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links.filter((l) => l.weight >= state.minWeight)).distance((l) => 20 + 140 * (1 - l.weight)).strength(0.6))
        .force("charge", d3.forceManyBody().strength(-40))
        .force("center", d3.forceCenter(W / 2, H / 2))
        .force("collide", d3.forceCollide(6))
        .on("tick", () => update(false));
      if (Common.staticMode) { sim.stop(); for (let i = 0; i < 400; i++) sim.tick(); update(false); }
      return;
    }
    place(key);
    update(animate);
  }

  // ---- controls
  const layoutSel = d3.select("#layout");
  layoutSel.selectAll("option").data(layouts).join("option").attr("value", (l) => l.key).text((l) => l.name);
  layoutSel.property("value", state.layout).on("change", function () { setLayout(this.value); });
  d3.select("#color").property("value", state.color).on("change", function () { state.color = this.value; update(); });
  d3.select("#size").property("value", state.size).on("change", function () { state.size = this.value; update(); });
  d3.select("#labels").property("value", state.labels).on("change", function () { state.labels = this.value; update(); });
  const mw = d3.select("#minWeight").property("value", state.minWeight);
  const showMw = () => d3.select("#minWeightVal").text(state.minWeight.toFixed(2));
  mw.on("input", function () { state.minWeight = +this.value; showMw(); update(); });
  const me = d3.select("#minEvents").property("value", state.minEventsLog);
  const showMe = () => d3.select("#minEventsVal").text(Math.round(Math.pow(10, state.minEventsLog)));
  me.on("input", function () { state.minEventsLog = Math.max(1, +this.value); showMe(); update(); });
  showMw(); showMe();
  d3.select("#search").property("value", "").on("input", function () {
    const q = this.value.trim().toLowerCase();
    const hit = q ? nodes.find((n) => n.name.toLowerCase().startsWith(q)) || nodes.find((n) => n.name.toLowerCase().includes(q)) : null;
    state.focus = hit ? hit.id : "";
    update();
  });
  d3.select("#reset").on("click", () => {
    state.minWeight = 0.7; state.minEventsLog = 1; state.focus = ""; state.continents = new Set(Object.keys(continentColors));
    mw.property("value", 0.7); me.property("value", 1); showMw(); showMe();
    svg.call(zoom.transform, d3.zoomIdentity);
    setLayout(state.layout, true);
  });
  window.addEventListener("resize", () => { size(); setLayout(state.layout, false); });

  if (!byId.has(state.focus)) {
    const hit = nodes.find((n) => n.name.toLowerCase() === state.focus.toLowerCase());
    state.focus = hit ? hit.id : "";
  }
  setLayout(state.layout, false);
})();
