"""Clean the EM-DAT country-profile table (A1 dataset) and derive the tables used by the InfoVis tasks.

Input : data/raw/emdat/emdat_country_profiles_2023_04_06.csv  (Kaggle: mexwell/natural-disasters-emergency-events-database)
Output: data/processed/emdat_clean.csv               cleaned long table (one row per year x country x type x subtype)
        data/processed/emdat_country_1970_2022.csv   per-country multivariate profile (network + parallel coordinates)
        data/processed/emdat_cleaning_log.txt        numbers quoted in the report's data-handling section

Usage:  python code/data_prep/prepare_emdat.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from data_prep.regions import HISTORICAL, ISO_TO_SUBREGION, SUBREGION_TO_CONTINENT, TERRITORIES  # noqa: E402

WINDOW = (1970, 2022)       # cross-sectional window (reporting is sparse before 1970; 2023 is a partial year)
MIN_EVENTS = 10             # minimum events in WINDOW for a country to enter country-level analyses
CPI_BASE_YEAR = 2022        # CPI = 100 in 2022 in the source file

# Readable, unique display names (the source uses UN long forms, and two Koreas / two Congos collide when shortened).
SHORT_NAMES = {
    "KOR": "South Korea", "PRK": "North Korea", "COD": "DR Congo", "COG": "Congo", "TZA": "Tanzania",
    "GBR": "United Kingdom", "USA": "United States", "RUS": "Russia", "IRN": "Iran", "VEN": "Venezuela",
    "BOL": "Bolivia", "LAO": "Laos", "SYR": "Syria", "VNM": "Vietnam", "MDA": "Moldova", "MKD": "North Macedonia",
    "FSM": "Micronesia", "TWN": "Taiwan", "CIV": "Côte d'Ivoire", "PSE": "Palestine", "CZE": "Czechia",
    "DOM": "Dominican Republic", "CAF": "Central African Rep.", "NLD": "Netherlands", "PHL": "Philippines",
    "ARE": "United Arab Emirates", "BHS": "Bahamas", "GMB": "Gambia", "NER": "Niger", "SDN": "Sudan",
    "COM": "Comoros", "MHL": "Marshall Islands", "CYM": "Cayman Islands", "TCA": "Turks and Caicos",
    "VGB": "British Virgin Islands", "VIR": "US Virgin Islands", "MNP": "Northern Mariana Islands",
    "SXM": "Sint Maarten", "MAF": "Saint Martin", "SHN": "Saint Helena", "MAC": "Macao", "HKG": "Hong Kong",
    "SWZ": "Eswatini", "KNA": "Saint Kitts and Nevis", "VCT": "Saint Vincent and the Grenadines",
    "BIH": "Bosnia and Herzegovina", "TLS": "Timor-Leste",
}

log_lines = []


def log(msg: str) -> None:
    print(msg)
    log_lines.append(msg)


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(config.EMDAT_RAW, sep=";", decimal=",")
    log(f"raw rows={len(df)}, columns={len(df.columns)}, years={df.Year.min()}-{df.Year.max()}, "
        f"entities={df.ISO.nunique()}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "Disaster Group": "group", "Disaster Subroup": "subgroup", "Disaster Type": "type",
        "Disaster Subtype": "subtype", "Total Events": "events", "Total Affected": "affected",
        "Total Deaths": "deaths", "Total Damage (USD, original)": "damage_usd",
        "Total Damage (USD, adjusted)": "damage_usd_adj", "CPI": "cpi", "Year": "year",
        "Country": "country", "ISO": "iso",
    })
    for c in ["country", "group", "subgroup", "type", "subtype"]:
        df[c] = df[c].str.strip()          # e.g. "Extreme temperature " and "Pyroclastic flow "

    # Missing subtypes: 20% of rows (mostly floods and storms). Keep them as an explicit
    # "<type> (unspecified)" category instead of dropping events.
    n_missing_sub = int(df.subtype.isna().sum())
    single = {"Fog", "Animal accident", "Glacial lake outburst"}
    df.loc[df.subtype.isna() & df.type.isin(single), "subtype"] = df.type
    df["subtype"] = df.subtype.fillna(df.type + " (unspecified)")
    log(f"missing subtype filled: {n_missing_sub} rows ({n_missing_sub / len(df):.1%})")

    # Damage: the adjusted column equals original * 100 / CPI (CPI base year 2022). 2023 rows have no CPI;
    # we assume 2023 = base-year prices so the 4 rows with a reported damage keep a value.
    chk = df.dropna(subset=["damage_usd", "damage_usd_adj", "cpi"])
    rel_err = np.abs(chk.damage_usd * 100 / chk.cpi - chk.damage_usd_adj) / chk.damage_usd_adj
    log(f"adjusted damage = original*100/CPI verified, max relative error {rel_err.max():.2e}")
    fill = df.damage_usd.notna() & df.damage_usd_adj.isna()
    df.loc[fill, "damage_usd_adj"] = df.loc[fill, "damage_usd"]
    log(f"adjusted damage filled for {int(fill.sum())} rows of {sorted(df.loc[fill, 'year'].unique())} (CPI missing)")

    for c in ["affected", "deaths", "damage_usd_adj"]:
        log(f"missing {c}: {df[c].isna().mean():.1%} of rows (kept as NaN = not reported)")

    df["subregion"] = df.iso.map(ISO_TO_SUBREGION)
    missing = sorted(df.loc[df.subregion.isna(), "iso"].unique())
    assert not missing, f"no region for {missing}"
    df["continent"] = df.subregion.map(SUBREGION_TO_CONTINENT)
    df["historical"] = df.iso.isin(HISTORICAL)
    df["territory"] = df.iso.isin(TERRITORIES)
    df["decade"] = (df.year // 10 * 10).astype(int)
    df["partial_year"] = df.year == 2023
    df["country_short"] = df.country.str.replace(r"\s*\(the\)", "", regex=True).str.replace(
        r"\s*\(.*?\)", "", regex=True).str.strip()
    df["country_short"] = df.iso.map(SHORT_NAMES).fillna(df.country_short)
    assert df.groupby("country_short").iso.nunique().max() == 1, "short names must be unique"
    log(f"historical entities: {df[df.historical].iso.nunique()} ({df[df.historical].events.sum()} events); "
        f"territories: {df[df.territory].iso.nunique()}")

    ev_year = df.groupby("year").events.sum()
    log(f"events in 2023 (snapshot 2023-04-06): {ev_year.get(2023, 0)} vs mean 2018-2022: "
        f"{ev_year.loc[2018:2022].mean():.0f} -> 2023 treated as partial and excluded from temporal analyses")
    pre = df[df.year < WINDOW[0]].events.sum() / df.events.sum()
    log(f"share of all events recorded before {WINDOW[0]}: {pre:.1%}")
    return df


def country_profile(df: pd.DataFrame) -> pd.DataFrame:
    w = df[(df.year >= WINDOW[0]) & (df.year <= WINDOW[1]) & ~df.historical]
    g = w.groupby("iso")
    prof = pd.DataFrame({
        "country": g.country_short.first(),
        "continent": g.continent.first(),
        "subregion": g.subregion.first(),
        "events": g.events.sum(),
        "deaths": g.deaths.sum(min_count=1).fillna(0),
        "affected": g.affected.sum(min_count=1).fillna(0),
        "damage_usd_adj": g.damage_usd_adj.sum(min_count=1).fillna(0),
        "n_types": g.type.nunique(),
        "active_years": g.year.nunique(),
    })
    sub = w.pivot_table(index="iso", columns="subgroup", values="events", aggfunc="sum", fill_value=0)
    for s in ["Hydrological", "Meteorological", "Geophysical", "Climatological", "Biological"]:
        prof[f"share_{s.lower()}"] = sub.get(s, 0) / prof.events
    early = w[w.year < 1996].groupby("iso").events.sum()
    late = w[w.year >= 1996].groupby("iso").events.sum()
    prof["events_1970_1995"] = early.reindex(prof.index).fillna(0)
    prof["events_1996_2022"] = late.reindex(prof.index).fillna(0)
    prof["deaths_per_event"] = prof.deaths / prof.events
    # Reporting completeness: share of country-year-type records with a damage figure.
    prof["damage_reported_share"] = g.damage_usd_adj.apply(lambda s: s.notna().mean())
    n_before = len(prof)
    prof = prof[prof.events >= MIN_EVENTS].copy()
    log(f"country profile {WINDOW[0]}-{WINDOW[1]}: {n_before} current entities, {len(prof)} with >= {MIN_EVENTS} events")
    return prof.reset_index()


def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df = clean(load_raw())
    cols = ["year", "decade", "iso", "country", "country_short", "continent", "subregion", "historical",
            "territory", "partial_year", "group", "subgroup", "type", "subtype", "events", "affected",
            "deaths", "damage_usd", "damage_usd_adj", "cpi"]
    df[cols].to_csv(config.DATA_PROCESSED / "emdat_clean.csv", index=False)
    prof = country_profile(df)
    prof.to_csv(config.DATA_PROCESSED / "emdat_country_1970_2022.csv", index=False)
    (config.DATA_PROCESSED / "emdat_cleaning_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
