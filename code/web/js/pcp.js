// Parallel coordinates with cluster-based edge bundling, brushing, axis reordering and inversion.
// Bundling follows the centroid approach of illustrative/bundled parallel coordinates: between two adjacent axes each
// polyline passes through a point pulled towards the centroid of its group (strength beta) and is drawn as a cubic
// Bezier spline whose control points depend on a smoothness parameter.
(() => {
  const DATA = window.PCP_DATA;
  const { param, duration, fmt, showTip, hideTip, continentColors } = Common;
  const clusterColors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"];

  const state = {
    view: param("view", "country"),
    color: param("color", "cluster"),
    group: param("group", "cluster"),
    beta: +param("bundle", 0),
    smooth: +param("smooth", 0),
    order: null,
    inverted: new Set((param("invert", "") || "").split(",").filter(Boolean)),
    brushes: {},            // key -> [lo, hi] in data units
    hidden: new Set(),      // legend categories switched off
    highlight: param("highlight", ""),
  };

  const svg = d3.select("#pcp");
  const margin = { top: 64, right: 40, bottom: 26, left: 40 };
  let width, height, dims, rows, x, y = {}, dragging = {};

  function sizeSvg() {
    const box = document.querySelector(".chart").getBoundingClientRect();
    width = Math.max(900, box.width - 24);
    height = Math.max(520, window.innerHeight - 190);
    svg.attr("width", width).attr("height", height);
  }

  function groupKey(r, by) { return by === "continent" ? r.continent : r.cluster; }
  function colorOf(r) {
    return state.color === "continent" ? continentColors[r.continent] : clusterColors[r.cluster % clusterColors.length];
  }

  function makeScale(d) {
    const vals = rows.map((r) => r[d.key]);
    let [lo, hi] = d3.extent(vals);
    const range = state.inverted.has(d.key) ? [margin.top, height - margin.bottom] : [height - margin.bottom, margin.top];
    if (d.scale === "log") {
      const c = d.key === "growth" || d.key === "deaths_per_event" ? 0.1 : 1;
      return d3.scaleSymlog().constant(c).domain([lo, hi]).range(range);
    }
    if (d.format === "pct") return d3.scaleLinear().domain([0, 1]).range(range);
    return d3.scaleLinear().domain([lo, hi]).range(range).nice();
  }

  function logTicks(scale) {
    const [lo, hi] = scale.domain();
    const cand = [];
    if (lo <= 0) cand.push(0);
    for (let p = -2; p <= 13; p++) {
      for (const m of [1, 3]) {
        const v = m * Math.pow(10, p);
        if (v >= lo && v <= hi) cand.push(v);
      }
    }
    const kept = [];
    cand.forEach((v) => {               // keep ticks at least 18 px apart, preferring powers of ten
      const isPow = v === 0 || Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9;
      if (!kept.length || Math.abs(scale(v) - scale(kept[kept.length - 1])) >= (isPow ? 18 : 34)) kept.push(v);
    });
    return kept;
  }
  const logFormat = (v) => (v === 0 ? "0" : v < 1 ? d3.format("~g")(v) : d3.format("~s")(v));

  function setup() {
    const view = DATA.views[state.view];
    rows = view.rows;
    dims = view.dimensions;
    const ord = param("order", "");
    if (!state.order) state.order = ord ? ord.split(",").filter((k) => dims.some((d) => d.key === k)) : null;
    if (!state.order || state.order.length !== dims.length) state.order = dims.map((d) => d.key);
    const brushParam = param("brush", "");
    if (brushParam && !Object.keys(state.brushes).length) {
      brushParam.split(";").forEach((b) => {
        const [k, lo, hi] = b.split(":");
        if (dims.some((d) => d.key === k)) state.brushes[k] = [+lo, +hi];
      });
    }
  }

  const dimByKey = (k) => dims.find((d) => d.key === k);
  const position = (k) => (dragging[k] != null ? dragging[k] : x(k));

  function selected(r) {
    for (const [k, [lo, hi]] of Object.entries(state.brushes)) {
      const v = r[k];
      if (v < Math.min(lo, hi) || v > Math.max(lo, hi)) return false;
    }
    const cat = state.color === "continent" ? r.continent : `c${r.cluster}`;
    return !state.hidden.has(cat);
  }

  // centroid of each group on each axis, in screen coordinates
  function centroids() {
    const by = state.group;
    const out = new Map();
    d3.group(rows, (r) => groupKey(r, by)).forEach((members, g) => {
      const m = {};
      state.order.forEach((k) => { m[k] = d3.mean(members, (r) => y[k](r[k])); });
      out.set(g, m);
    });
    return out;
  }

  function pathFor(r, cents) {
    const keys = state.order;
    const pts = [];
    const cg = cents.get(groupKey(r, state.group));
    keys.forEach((k, i) => {
      const px = position(k), py = y[k](r[k]);
      pts.push([px, py]);
      if (i < keys.length - 1) {
        const k2 = keys[i + 1];
        const qx = position(k2), qy = y[k2](r[k2]);
        const mid = (py + qy) / 2;
        const cmid = (cg[k] + cg[k2]) / 2;
        pts.push([px + 0.5 * (qx - px), mid + state.beta * (cmid - mid)]);
      }
    });
    const a = state.smooth;
    if (a === 0 && state.beta === 0) return "M" + pts.filter((_, i) => i % 2 === 0).map((p) => p.join(",")).join("L");
    const n = pts.length;
    const cps = [pts[0], [pts[0][0] + a * 2 * (pts[1][0] - pts[0][0]), pts[0][1]]];
    for (let i = 1; i < n - 1; i++) {
      const mid = pts[i], l = pts[i - 1], rr = pts[i + 1];
      const diff = [l[0] - rr[0], l[1] - rr[1]];
      cps.push([mid[0] + a * diff[0], mid[1] + a * diff[1]], mid, [mid[0] - a * diff[0], mid[1] - a * diff[1]]);
    }
    cps.push([pts[n - 1][0] + a * 2 * (pts[n - 2][0] - pts[n - 1][0]), pts[n - 1][1]], pts[n - 1]);
    let d = `M${cps[0]}`;
    for (let i = 1; i < cps.length; i += 3) d += `C${cps[i]} ${cps[i + 1]} ${cps[i + 2]}`;
    return d;
  }

  function tipHtml(r) {
    const lines = state.order.map((k) => {
      const dm = dimByKey(k);
      return `<tr><td>${dm.label}</td><td class="v">${fmt[dm.format](r[k])}</td></tr>`;
    }).join("");
    const lab = DATA.views[state.view].clusters[r.cluster].label;
    return `<div class="t">${r.name}</div><div class="status">${r.continent} · ${r.subregion}<br>${lab}</div><table>${lines}</table>`;
  }

  let gBack, gFore, gAxes;
  function render() {
    sizeSvg();
    svg.selectAll("*").remove();
    x = d3.scalePoint().domain(state.order).range([margin.left + 30, width - margin.right - 30]);
    dims.forEach((d) => { y[d.key] = makeScale(d); });
    gBack = svg.append("g").attr("class", "back");
    gFore = svg.append("g").attr("class", "fore");
    gAxes = svg.append("g").attr("class", "axes");
    drawLines();
    drawAxes();
    updateLegend();
  }

  function drawLines() {
    const cents = centroids();
    const sel = rows.filter(selected);
    const selSet = new Set(sel.map((r) => r.id));
    const hl = state.highlight.toLowerCase();
    gBack.selectAll("path").data(rows.filter((r) => !selSet.has(r.id)), (r) => r.id).join("path")
      .attr("d", (r) => pathFor(r, cents)).attr("fill", "none").attr("stroke", "#c9c9c9")
      .attr("stroke-opacity", 0.22).attr("stroke-width", 0.8);
    const op = state.view === "decade" ? 0.38 : 0.55;
    gFore.selectAll("path").data(sel, (r) => r.id).join("path")
      .attr("d", (r) => pathFor(r, cents)).attr("fill", "none")
      .attr("stroke", (r) => colorOf(r))
      .attr("stroke-opacity", (r) => (hl ? (r.name.toLowerCase().includes(hl) ? 1 : 0.12) : op))
      .attr("stroke-width", (r) => (hl && r.name.toLowerCase().includes(hl) ? 3.2 : 1.2))
      .on("mouseenter", function (event, r) {
        d3.select(this).raise().attr("stroke-width", 3.2).attr("stroke-opacity", 1);
        showTip(event, tipHtml(r));
      })
      .on("mousemove", (event, r) => showTip(event, tipHtml(r)))
      .on("mouseleave", function () {
        d3.select(this).attr("stroke-width", 1.2).attr("stroke-opacity", hl ? 0.12 : op);
        hideTip();
      });
    if (hl) gFore.selectAll("path").filter((r) => r.name.toLowerCase().includes(hl)).raise();
    d3.select("#status").html(`<b>${sel.length}</b> of ${rows.length} rows selected` +
      (Object.keys(state.brushes).length ? ` · brushed axes: ${Object.keys(state.brushes).map((k) => dimByKey(k).label).join(", ")}` : ""));
    const list = sel.slice().sort((a, b) => b.events - a.events);
    d3.select("#list").html(sel.length && sel.length < rows.length
      ? list.slice(0, 40).map((r) => `<span style="color:${colorOf(r)}">●</span> ${r.name}`).join("<br>") +
        (sel.length > 40 ? `<br>… and ${sel.length - 40} more` : "")
      : "<i>no brush active</i>");
  }

  function drawAxes() {
    const axes = gAxes.selectAll("g.dim").data(state.order, (k) => k).join("g")
      .attr("class", "dim").attr("transform", (k) => `translate(${position(k)},0)`);
    axes.each(function (k) {
      const dm = dimByKey(k);
      const g = d3.select(this);
      const ax = d3.axisLeft(y[k]).tickSizeOuter(0);
      if (dm.scale === "log") ax.tickValues(logTicks(y[k])).tickFormat((v) => (dm.format === "usd" && v ? fmt.usd(v) : logFormat(v)));
      else if (dm.format === "pct") ax.ticks(5).tickFormat(d3.format(".0%"));
      else if (dm.format === "decade") ax.ticks(5).tickFormat((v) => `${v}s`);
      else ax.ticks(6);
      g.append("g").attr("class", "axis").call(ax);
      const words = dm.label.split(" ");
      const lines = [];
      words.forEach((w) => {
        if (lines.length && (lines[lines.length - 1] + " " + w).length <= 16) lines[lines.length - 1] += " " + w;
        else lines.push(w);
      });
      const title = g.append("text").attr("class", "axis-title" + (state.inverted.has(k) ? " inverted" : ""))
        .attr("text-anchor", "middle").attr("y", margin.top - 14 - 13 * (lines.length - 1));
      lines.forEach((ln, i) => title.append("tspan").attr("x", 0).attr("dy", i ? 13 : 0).text(ln));
      title.append("title").text("Drag to reorder · double-click to invert");
      title.on("dblclick", () => {
        if (state.inverted.has(k)) state.inverted.delete(k); else state.inverted.add(k);
        render();
      });
      title.call(d3.drag()
        .on("start", () => { dragging[k] = x(k); })
        .on("drag", (event) => {
          dragging[k] = Math.max(margin.left, Math.min(width - margin.right, event.x));
          state.order.sort((a, b) => position(a) - position(b));
          x.domain(state.order);
          gAxes.selectAll("g.dim").attr("transform", (kk) => `translate(${position(kk)},0)`);
          drawLines();
        })
        .on("end", () => {
          delete dragging[k];
          x.domain(state.order);
          gAxes.selectAll("g.dim").transition().duration(duration(250)).attr("transform", (kk) => `translate(${x(kk)},0)`);
          drawLines();
        }));
      const brush = d3.brushY().extent([[-11, margin.top], [11, height - margin.bottom]])
        .on("brush end", (event) => {
          if (!event.sourceEvent) return;
          if (!event.selection) delete state.brushes[k];
          else state.brushes[k] = event.selection.map((v) => y[k].invert(v));
          drawLines();
        });
      const gb = g.append("g").attr("class", "brush").call(brush);
      if (state.brushes[k]) gb.call(brush.move, state.brushes[k].map((v) => y[k](v)).sort((a, b) => a - b));
    });
  }

  function updateLegend() {
    const cats = state.color === "continent"
      ? Object.keys(continentColors).map((c) => ({ id: c, label: c, color: continentColors[c] }))
      : DATA.views[state.view].clusters.map((c) => ({ id: `c${c.id}`, label: c.label, color: clusterColors[c.id] }));
    const counts = state.color === "continent" ? d3.rollup(rows, (v) => v.length, (r) => r.continent)
      : d3.rollup(rows, (v) => v.length, (r) => `c${r.cluster}`);
    d3.select("#legend").selectAll("div").data(cats, (c) => c.id).join("div")
      .attr("class", (c) => "legend-item" + (state.hidden.has(c.id) ? " off" : ""))
      .html((c) => `<span class="swatch" style="background:${c.color}"></span><span>${c.label} (${counts.get(c.id) || 0})</span>`)
      .on("click", (event, c) => {
        if (state.hidden.has(c.id)) state.hidden.delete(c.id); else state.hidden.add(c.id);
        updateLegend();
        drawLines();
      });
  }

  // ---- controls
  const bind = (id, key, conv = (v) => v, after = render) => {
    const el = document.getElementById(id);
    el.value = state[key];
    el.addEventListener("input", () => { state[key] = conv(el.value); after(); });
  };
  bind("color", "color", String, () => { state.hidden.clear(); updateLegend(); drawLines(); });
  bind("group", "group", String, drawLines);
  bind("bundle", "beta", Number, () => { d3.select("#bundleVal").text(state.beta.toFixed(2)); drawLines(); });
  bind("smooth", "smooth", Number, () => { d3.select("#smoothVal").text(state.smooth.toFixed(2)); drawLines(); });
  d3.select("#bundleVal").text(state.beta.toFixed(2));
  d3.select("#smoothVal").text(state.smooth.toFixed(2));
  const viewEl = document.getElementById("view");
  viewEl.value = state.view;
  viewEl.addEventListener("change", () => {
    state.view = viewEl.value; state.order = null; state.brushes = {}; state.hidden.clear(); state.inverted.clear();
    setup(); render();
  });
  const search = document.getElementById("search");
  search.value = state.highlight;
  search.addEventListener("input", () => { state.highlight = search.value.trim(); drawLines(); });
  document.getElementById("reset").addEventListener("click", () => {
    state.brushes = {}; state.hidden.clear(); state.inverted.clear(); state.order = dims.map((d) => d.key);
    state.beta = 0; state.smooth = 0; state.highlight = "";
    ["bundle", "smooth"].forEach((id) => { document.getElementById(id).value = 0; });
    d3.select("#bundleVal").text("0.00"); d3.select("#smoothVal").text("0.00"); search.value = "";
    render();
  });
  window.addEventListener("resize", () => render());

  setup();
  render();
})();
