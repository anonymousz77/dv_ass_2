"""Build the PDF report from report/source/report.html.

1. Assembles the printable report in a temporary folder: report/source/report.html plus a reduced-size JPEG copy of
   every figure in images/ (the full-resolution PNGs stay in images/, named by their figure index, for assessment).
2. Prints it to PDF with a locally installed Edge/Chrome (headless).
3. Writes the PDF document properties.

Usage:  python code/tools/build_report.py
"""
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image
from pypdf import PdfReader, PdfWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from tools.capture_screenshots import browser  # noqa: E402

SRC = config.ROOT / "report" / "source"
OUT = config.ROOT / "report" / "Pythagoras_DAS732_A2_Report.pdf"
MAX_W = 2000


def assemble(build: Path) -> Path:
    fig = build / "figures"
    fig.mkdir(parents=True)
    for png in sorted(config.IMAGES.glob("Fig*.png")):
        im = Image.open(png).convert("RGB")
        if im.width > MAX_W:
            im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
        im.save(fig / f"{png.stem}.jpg", quality=88, optimize=True)
    shutil.copy(SRC / "report.html", build / "report.html")
    print(f"{len(list(fig.glob('*.jpg')))} report figures assembled")
    return build / "report.html"


def pdf(html: Path) -> None:
    tmp = OUT.with_suffix(".tmp.pdf")
    if tmp.exists():
        tmp.unlink()
    with tempfile.TemporaryDirectory() as profile:
        cmd = [browser(), "--headless=new", "--disable-gpu", "--no-first-run", "--disable-sync", "--disable-extensions",
               f"--user-data-dir={profile}", "--no-pdf-header-footer", "--virtual-time-budget=15000",
               f"--print-to-pdf={tmp}", html.as_uri()]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.time()
        last = -1
        while time.time() - t0 < 300:
            if proc.poll() is not None:
                break
            if (tmp.exists() and tmp.stat().st_size == last and last > 0
                    and tmp.read_bytes()[-1024:].rstrip().endswith(b"%%EOF")):
                break
            last = tmp.stat().st_size if tmp.exists() else -1
            time.sleep(2)
        if proc.poll() is None:
            proc.kill()
    reader = PdfReader(tmp)
    writer = PdfWriter(clone_from=reader)
    writer.add_metadata({
        "/Title": "DAS732 Programming Assignment 2 – SciVis and InfoVis (Team Pythagoras)",
        "/Author": "Lakshya Jain (BT2024044), Kalpit (BT2024093), Arnav (BT2024233)",
        "/Subject": "gridMET Jul–Sep 2019 SciVis and EM-DAT InfoVis",
        "/Keywords": "colour mapping, marching squares, quiver, streamlines, node-link, parallel coordinates, treemap, "
                     "horizon chart, ridgeline",
        "/Creator": "Team Pythagoras",
        "/Producer": "Team Pythagoras",
    })
    with open(OUT, "wb") as f:
        writer.write(f)
    tmp.unlink()
    print(f"wrote {OUT} ({len(reader.pages)} pages, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as build:
        pdf(assemble(Path(build)))
