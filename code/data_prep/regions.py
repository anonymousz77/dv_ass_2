"""ISO-3166 alpha-3 -> UN M49 sub-region / continent lookup for the EM-DAT country profiles.

The Kaggle EM-DAT country-profile file has no region column, so the geography used for the
treemap hierarchy, network colouring and continent-wise time series is derived here from the
UN M49 standard. Historical entities that appear in EM-DAT (USSR, Yugoslavia, ...) and island
territories reported separately (Azores, Canary Islands) are assigned to the region of their
successor / administering state.
"""

SUBREGION_TO_CONTINENT = {
    "Northern Africa": "Africa", "Eastern Africa": "Africa", "Middle Africa": "Africa",
    "Southern Africa": "Africa", "Western Africa": "Africa",
    "Caribbean": "Americas", "Central America": "Americas", "South America": "Americas",
    "Northern America": "Americas",
    "Central Asia": "Asia", "Eastern Asia": "Asia", "South-eastern Asia": "Asia",
    "Southern Asia": "Asia", "Western Asia": "Asia",
    "Eastern Europe": "Europe", "Northern Europe": "Europe", "Southern Europe": "Europe",
    "Western Europe": "Europe",
    "Australia and New Zealand": "Oceania", "Melanesia": "Oceania", "Micronesia": "Oceania",
    "Polynesia": "Oceania",
}

_SUBREGION_MEMBERS = {
    "Northern Africa": "DZA EGY LBY MAR SDN TUN ESH",
    "Eastern Africa": "BDI COM DJI ERI ETH KEN MDG MWI MUS MYT MOZ REU RWA SYC SOM SSD UGA TZA ZMB ZWE",
    "Middle Africa": "AGO CMR CAF TCD COG COD GNQ GAB STP",
    "Southern Africa": "BWA SWZ LSO NAM ZAF",
    "Western Africa": "BEN BFA CPV CIV GMB GHA GIN GNB LBR MLI MRT NER NGA SHN SEN SLE TGO",
    "Caribbean": "AIA ATG ABW BHS BRB BES VGB CYM CUB CUW DMA DOM GRD GLP HTI JAM MTQ MSR PRI "
                 "BLM KNA LCA MAF VCT SXM TTO TCA VIR ANT",
    "Central America": "BLZ CRI SLV GTM HND MEX NIC PAN",
    "South America": "ARG BOL BRA CHL COL ECU FLK GUF GUY PRY PER SUR URY VEN",
    "Northern America": "BMU CAN GRL SPM USA",
    "Central Asia": "KAZ KGZ TJK TKM UZB",
    "Eastern Asia": "CHN HKG MAC PRK JPN MNG KOR TWN",
    "South-eastern Asia": "BRN KHM IDN LAO MYS MMR PHL SGP THA TLS VNM",
    "Southern Asia": "AFG BGD BTN IND IRN MDV NPL PAK LKA",
    "Western Asia": "ARM AZE BHR CYP GEO IRQ ISR JOR KWT LBN OMN QAT SAU PSE SYR TUR ARE YEM YMD YMN",
    "Eastern Europe": "BLR BGR CZE HUN POL MDA ROU RUS SVK UKR SUN CSK",
    "Northern Europe": "DNK EST FIN ISL IRL IMN LVA LTU NOR SWE GBR FRO",
    "Southern Europe": "ALB AND BIH HRV GIB GRC ITA MLT MNE MKD PRT SMR SRB SVN ESP VAT XKX YUG SCG AZO SPI",
    "Western Europe": "AUT BEL FRA DEU LIE LUX MCO NLD CHE DDR DFR",
    "Australia and New Zealand": "AUS NZL NFK",
    "Melanesia": "FJI NCL PNG SLB VUT",
    "Micronesia": "GUM KIR MHL FSM NRU MNP PLW",
    "Polynesia": "ASM COK PYF NIU PCN WSM TKL TON TUV WLF",
}

ISO_TO_SUBREGION = {iso: sub for sub, members in _SUBREGION_MEMBERS.items() for iso in members.split()}

# Entities that no longer exist; kept as recorded by EM-DAT but flagged.
HISTORICAL = {
    "SUN": "Soviet Union (to 1991)", "YUG": "Yugoslavia (to 1991)", "CSK": "Czechoslovakia (to 1992)",
    "DDR": "German Dem. Rep. (to 1990)", "DFR": "German Fed. Rep. (to 1990)",
    "YMD": "South Yemen (to 1990)", "YMN": "North Yemen (to 1990)",
    "SCG": "Serbia and Montenegro (1992-2006)", "ANT": "Netherlands Antilles (to 2010)",
}
# Territories reported separately from their administering state.
TERRITORIES = {"AZO": "Azores (Portugal)", "SPI": "Canary Islands (Spain)"}
