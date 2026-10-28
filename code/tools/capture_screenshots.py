"""Capture high-resolution screenshots of the browser visualisations for the report (Figs. 27-36).

Each interaction state is encoded in the page URL (layout, filters, brushes, bundling, zoom path, ...), so the
screenshots are reproducible. Uses a locally installed Microsoft Edge or Google Chrome in headless mode.

Usage:  python code/tools/capture_screenshots.py            (all)
        python code/tools/capture_screenshots.py Fig29      (only names containing "Fig29")
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]
WEB = config.CODE_DIR / "web"

SHOTS = [  # (file name, page, query, (width, height))
    ("Fig27a_network_viewer_forceatlas2", "network.html", "layout=fa2", (1700, 1050)),
    ("Fig27b_network_viewer_focus_india", "network.html", "layout=fr&color=continent&focus=IND", (1700, 1050)),
    ("Fig27c_network_viewer_filtered", "network.html", "layout=kk&minWeight=0.9&minEvents=50&labels=all", (1700, 1050)),
    ("Fig28_pcp_countries_no_bundling", "parallel_coordinates.html", "", (1750, 1000)),
    ("Fig29_pcp_countries_bundled", "parallel_coordinates.html", "bundle=0.8&smooth=0.18", (1750, 1000)),
    ("Fig30a_pcp_brush_damage_reported", "parallel_coordinates.html",
     "brush=damage_reported:0.6:1&color=continent&bundle=0.3&smooth=0.1", (1750, 1000)),
    ("Fig30b_pcp_brush_deadly_events", "parallel_coordinates.html",
     "brush=deaths_per_event:100:100000&color=continent&bundle=0.3&smooth=0.1", (1750, 1000)),
    ("Fig31_pcp_decades_reordered_bundled", "parallel_coordinates.html",
     "view=decade&order=decade,events,share_hydrological,share_meteorological,share_geophysical,share_climatological,"
     "deaths,deaths_per_event,affected,damage,damage_reported,n_types&invert=damage_reported&color=cluster&group=cluster"
     "&bundle=0.7&smooth=0.15", (1750, 1000)),
    ("Fig32_treemap_geography_events", "treemap.html", "hier=geo&measure=events&period=1900", (1700, 1000)),
    ("Fig33_treemap_zoom_southern_asia_deaths", "treemap.html",
     "hier=geo&measure=deaths&period=1900&zoom=" + quote("Asia>Southern Asia"), (1700, 1000)),
    ("Fig34a_treemap_hazard_deaths", "treemap.html", "hier=hazard&measure=deaths&period=1970", (1700, 1000)),
    ("Fig34b_treemap_hazard_damage", "treemap.html", "hier=hazard&measure=damage&period=1970", (1700, 1000)),
    ("Fig35_treemap_time_events", "treemap.html", "hier=time&measure=events&period=1900", (1700, 1000)),
    ("Fig36a_treemap_tiling_squarified", "treemap.html", "hier=geo&measure=events&period=1970&tile=squarify&zoom=Asia", (1300, 900)),
    ("Fig36b_treemap_tiling_slicedice", "treemap.html", "hier=geo&measure=events&period=1970&tile=sliceDice&zoom=Asia", (1300, 900)),
    ("Fig36c_treemap_tiling_binary", "treemap.html", "hier=geo&measure=events&period=1970&tile=binary&zoom=Asia", (1300, 900)),
]


def browser() -> str:
    for b in BROWSERS:
        if Path(b).exists():
            return b
    raise SystemExit("No Edge/Chrome installation found; open the pages manually instead.")


def shoot(exe: str, url: str, out: Path, w: int, h: int, profile: str) -> None:
    if out.exists():
        out.unlink()
    cmd = [exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", "--disable-sync",
           "--disable-extensions", "--disable-background-networking", f"--user-data-dir={profile}",
           f"--window-size={w},{h}", "--force-device-scale-factor=2", "--virtual-time-budget=9000",
           f"--screenshot={out}", url]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 120:
        if proc.poll() is not None:
            break
        if out.exists() and out.stat().st_size > 0:   # written; the browser sometimes lingers afterwards
            time.sleep(1.5)
            break
        time.sleep(0.5)
    if proc.poll() is None:
        proc.kill()
    if not out.exists():
        raise RuntimeError(f"screenshot failed: {url}")


def main() -> None:
    exe = browser()
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    with tempfile.TemporaryDirectory() as profile:
        for name, page, query, (w, h) in SHOTS:
            if only and only not in name:
                continue
            url = (WEB / page).as_uri() + "?static=1" + (f"&{query}" if query else "")
            out = config.IMAGES / f"{name}.png"
            shoot(exe, url, out, w, h, profile)
            print("saved", out.name)


if __name__ == "__main__":
    main()
