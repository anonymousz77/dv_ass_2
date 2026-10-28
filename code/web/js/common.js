// Shared helpers for the A2 browser visualisations (Team Pythagoras).
const Common = (() => {
  const params = new URLSearchParams(window.location.search);
  const param = (key, fallback) => (params.has(key) ? params.get(key) : fallback);
  const staticMode = params.get("static") === "1";          // used for reproducible screenshots
  const duration = (ms) => (staticMode ? 0 : ms);

  const si = d3.format(".3~s");
  const fmt = {
    int: (v) => d3.format(",")(Math.round(v)),
    si: (v) => si(v).replace("G", "B"),
    usd: (v) => "$" + si(v).replace("G", "B"),
    pct: (v) => d3.format(".0%")(v),
    "1f": (v) => d3.format(",.1f")(v),
    "2f": (v) => d3.format(".2f")(v),
    decade: (v) => `${v}s`,
  };

  const tooltip = d3.select("body").append("div").attr("class", "tooltip");
  function showTip(event, html) {
    tooltip.html(html).style("opacity", 1);
    const w = tooltip.node().offsetWidth, h = tooltip.node().offsetHeight;
    let x = event.clientX + 14, y = event.clientY + 14;
    if (x + w > window.innerWidth - 8) x = event.clientX - w - 14;
    if (y + h > window.innerHeight - 8) y = event.clientY - h - 14;
    tooltip.style("left", `${x}px`).style("top", `${y}px`);
  }
  const hideTip = () => tooltip.style("opacity", 0);

  const continentColors = { Africa: "#E69F00", Americas: "#009E73", Asia: "#D55E00", Europe: "#0072B2", Oceania: "#CC79A7" };

  return { param, staticMode, duration, fmt, showTip, hideTip, continentColors };
})();
