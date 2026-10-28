"""Shared paths and constants for the DAS732 A2 codebase (Team Pythagoras)."""
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
IMAGES = ROOT / "images"
WEB_DATA = CODE_DIR / "web" / "data"

GRIDMET_RAW = DATA_RAW / "gridmet"
EMDAT_RAW = DATA_RAW / "emdat" / "emdat_country_profiles_2023_04_06.csv"

# gridMET slice allotted to the team: July - September 2019 (92 days).
PERIOD_START = "2019-07-01"
PERIOD_END = "2019-09-30"

# Uniform temporal sampling: every 10th day starting 1 July -> 10 time instances.
SAMPLE_STRIDE_DAYS = 10
SAMPLE_DATES = [
    "2019-07-01", "2019-07-11", "2019-07-21", "2019-07-31", "2019-08-10",
    "2019-08-20", "2019-08-30", "2019-09-09", "2019-09-19", "2019-09-29",
]

# gridMET variables used in A2: short name -> (THREDDS aggregation, variable name, units)
GRIDMET_VARS = {
    "tmmx": ("agg_met_tmmx_1979_CurrentYear_CONUS.nc", "daily_maximum_temperature", "K"),
    "pr": ("agg_met_pr_1979_CurrentYear_CONUS.nc", "precipitation_amount", "mm"),
    "vpd": ("agg_met_vpd_1979_CurrentYear_CONUS.nc", "daily_mean_vapor_pressure_deficit", "kPa"),
    "vs": ("agg_met_vs_1979_CurrentYear_CONUS.nc", "daily_mean_wind_speed", "m/s"),
    "th": ("agg_met_th_1979_CurrentYear_CONUS.nc", "daily_mean_wind_direction", "degrees"),
}

FIG_DPI = 200
