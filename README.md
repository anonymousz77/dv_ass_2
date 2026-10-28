# DAS732 Data Visualization – Programming Assignment 2 (A2)

**Team Pythagoras** – Lakshya Jain (BT2024044), Kalpit (BT2024093), Arnav (BT2024233)
International Institute of Information Technology Bangalore, Term 1 2026-27

* SciVis dataset: gridMET daily meteorology, allotted slice **July–September 2019**, 10 uniformly sampled days
  (1, 11, 21, 31 Jul; 10, 20, 30 Aug; 9, 19, 29 Sep).
* InfoVis dataset (A1 dataset): EM-DAT natural-disaster country profiles, Kaggle
  `mexwell/natural-disasters-emergency-events-database` (snapshot 2023-04-06).

## Folder structure

```
report/                       final report (PDF) and its HTML source (report/source/report.html)
images/                       all figures at full resolution, named by figure index (FigNN_*.png) + animation GIFs
code/
  config.py                   paths, sampling dates, gridMET variables
  requirements.txt
  data_prep/                  shared data cleaning and processing
    download_gridmet.py       downloads the Jul–Sep 2019 gridMET subsets (THREDDS NetCDF Subset Service)
    gridmet_overview.py       Fig 1 – daily statistics of the quarter and the 10 sampled days
    prepare_emdat.py          cleans the EM-DAT table, adds UN M49 regions, builds the country profile
    regions.py                ISO3 -> UN M49 sub-region / continent lookup (incl. historical entities)
  scivis/
    common.py                 loading, projection (Lambert conformal), vector conversion for cartopy, map/GIF helpers
    color_mapping.py          SciVis 1 – colour mapping (Figs 2–9)              [Lakshya Jain]
    marching_squares.py       own vectorised Marching Squares implementation    [Kalpit]
    contour_mapping.py        SciVis 2 – contour mapping (Figs 10–15)           [Kalpit]
    quiver_streamlines.py     SciVis 3 – quiver plots, streamlines, storm close-ups (Figs 16–22) [Arnav]
  infovis/
    network_layouts.py        InfoVis 1 – NetworkX node-link, 5 layouts, Gephi export (Figs 23–26) [Kalpit]
    build_web_data.py         InfoVis 2 – data for parallel coordinates and treemap [Lakshya Jain]
    horizon_ridgeline.py      InfoVis 3 – horizon and ridgeline plots (Figs 37–43) [Arnav]
  web/                        browser visualisations (JavaScript + D3 v7)
    index.html                landing page
    lib/d3.v7.9.0.min.js      bundled D3 (the pages work offline; a CDN copy is only a fallback)
    network.html, js/network.js                     interactive network viewer (Fig 27)   [Kalpit]
    parallel_coordinates.html, js/pcp.js            PCP with edge bundling (Figs 28–31)   [Lakshya Jain]
    treemap.html, js/treemap.js                     zoomable treemap (Figs 32–36)         [Lakshya Jain]
    css/style.css, js/common.js, data/*.js          shared style, helpers and embedded data
  tools/
    capture_screenshots.py    reproducible high-resolution screenshots of the web pages
    build_report.py           builds the PDF report from report/source/report.html (JPEG copies of the figures are
                              made in a temporary folder, so no duplicate images are stored)
data/
  raw/emdat/                  original Kaggle CSV
  raw/gridmet/                gridMET subsets at 4 km: the 10 sampled days (<var>_2019JAS_samples.nc) and the storm days
                              between the samples over the Gulf/south-east coast (<var>_2019_storms_4km.nc)
  processed/                  cleaned tables, derived network/multivariate/hierarchical data, statistics logs,
                              network_disaster_regimes.gexf (open in Gephi: ForceAtlas2 positions, colours, sizes)
```

## How to run

Python 3.11, from the repository root:

```bash
pip install -r code/requirements.txt

python code/data_prep/download_gridmet.py      # skips files that already exist (see note below)
python code/data_prep/prepare_emdat.py
python code/data_prep/gridmet_overview.py      # Fig 1

python code/scivis/color_mapping.py            # Figs 2–9 + GIFs
python code/scivis/contour_mapping.py          # Figs 10–15 + GIF
python code/scivis/quiver_streamlines.py       # Figs 16–22 + GIFs

python code/infovis/network_layouts.py         # Figs 23–26 + web/data/network_data.js + Gephi .gexf
python code/infovis/build_web_data.py          # web/data/pcp_data.js, treemap_data.js
python code/infovis/horizon_ridgeline.py       # Figs 37–43

python code/tools/capture_screenshots.py       # Figs 27–36 (needs Edge or Chrome)
python code/tools/build_report.py              # report/Pythagoras_DAS732_A2_Report.pdf
```

**Data note.** To keep the submission small, the two daily-context products are not included and are fetched by
`download_gridmet.py` (about 2 minutes): all 92 days of tmmx, vpd and vs at every 4th cell (~16 km;
`<var>_2019JAS_daily_stride4.nc`, used by Fig 1 and Fig 15) and all 92 days of precipitation at 4 km
(`pr_2019JAS_daily.nc`, used by Fig 1, stored compressed). Every other figure runs from the files in `data/raw/`.
The daily statistics behind Fig 1 are also in `data/processed/gridmet_daily_stats_JAS2019.csv`.

The browser visualisations open directly from `code/web/index.html` (no server and no internet connection needed:
data are embedded as JavaScript files and D3 v7.9.0 is bundled in `code/web/lib/`). Interaction states can be
set in the URL, e.g. `parallel_coordinates.html?bundle=0.8&smooth=0.18`,
`treemap.html?hier=geo&measure=deaths&zoom=Asia>Southern%20Asia`, `network.html?layout=kk&focus=IND`;
`?static=1` disables animations.

## Interactions implemented

* **Network** – layout switch (Fruchterman–Reingold, Kamada–Kawai, ForceAtlas2, spectral, circular, live d3-force)
  with animated transitions, colour (community / continent), node size, edge-similarity and node filters, continent
  toggles, search, click-to-focus neighbourhood, zoom/pan, node drag; labels are placed without overlaps.
* **Parallel coordinates** – multi-axis brushing (AND), axis reordering by drag, axis inversion (double-click),
  cluster/continent edge bundling with adjustable strength β and smoothness, legend filtering, search, hover details;
  two row granularities (countries / country-decades).
* **Treemap** – click-to-zoom with breadcrumbs, four hierarchies built from the same table, four size measures,
  six tiling algorithms with live aspect-ratio statistics, period filter, hover details.

## Contributions

| Member | SciVis task | InfoVis task |
|---|---|---|
| Lakshya Jain (BT2024044) | Colour mapping | Parallel coordinates with edge bundling + zoomable treemap (JavaScript) |
| Kalpit (BT2024093) | Contour mapping (Marching Squares / contour fill) | Node-link diagram and layout comparison (NetworkX) |
| Arnav (BT2024233) | Quiver plots and streamlines | Horizon and ridgeline plots |

Data download, cleaning and processing (`code/data_prep/`) and shared helpers were done jointly.
