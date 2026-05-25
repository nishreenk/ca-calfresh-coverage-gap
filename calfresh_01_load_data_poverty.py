"""
CalFresh Coverage Gap :” Data Loading
=====================================
Reads all data files and extracts only the columns we need.

Sources:
  dfa256m.csv
    - Monthly CalFresh admin data, 2004-2017, county x month
    - Cell_18 = total persons enrolled
    - Cell_26 = total households enrolled
    - Cell_30 = total benefits issued ($)
    - -999 = suppressed (small county)

  MMG individual year files (MMG20XX_20XXData_ToShare.xlsx)
    - One file per year, sheet = 'County'
    - 'Overall Food Insecurity Rate' stored as decimal (0.157 = 15.7%)
    - 'Child Food Insecurity Rate' also decimal
    - '% FI <= SNAP Threshold' = share of food insecure who are SNAP-eligible

  2025-food-environment-atlas-data.xlsx
    - sheet 'ASSISTANCE': per capita SNAP benefits (county-varying)
    - sheet 'ACCESS': food access barriers by county (all county-varying)
    - Real header is in row index 1 (row 0 has sheet label)

  Census API: ACS B22003 (pulled automatically)
    - hh_poor_not_enrolled = poor HHs not on SNAP (coverage gap numerator)
    - hh_snap_enrolled = HHs enrolled in SNAP

Outputs: data/dfa256_annual.csv
         data/mmg_ca_panel.csv
         data/food_env_atlas_ca.csv
         data/b22003_ca_panel.csv
"""

import pandas as pd
import numpy as np
import os
import glob
import json
import urllib.request

os.makedirs("data", exist_ok=True)

# Set to your local folder
UPLOADS = "/your/path/to/data/" 

def section(title):
    print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")

def clean_num(series):
    """Strip commas/dollar signs, return float. Treat -999 as NaN."""
    num = pd.to_numeric(
        series.astype(str)
              .str.replace(",", "", regex=False)
              .str.replace("$", "", regex=False)
              .str.strip(),
        errors="coerce"
    )
    return num.where(num >= 0, np.nan)


# =============================================================================
# 1. DFA256 :” CalFresh administrative data
# =============================================================================
section("1. DFA256 :” CalFresh admin data")

dfa_raw = pd.read_csv(UPLOADS + "dfa256m.csv", dtype=str)

dfa_raw["persons_enrolled"]    = clean_num(dfa_raw["Cell_18"])
dfa_raw["households_enrolled"] = clean_num(dfa_raw["Cell_26"])
dfa_raw["benefits_dollars"]    = clean_num(dfa_raw["Cell_30"])
dfa_raw["date"]  = pd.to_datetime(dfa_raw["Date"], format="%m/%d/%Y", errors="coerce")
dfa_raw["year"]  = dfa_raw["date"].dt.year
dfa_raw["month"] = dfa_raw["date"].dt.month

dfa = dfa_raw[dfa_raw["County"] != "Statewide"].copy()

dfa_annual = (
    dfa.groupby(["County", "County_Code", "FFY", "year"])
    .agg(
        persons_enrolled_avg    = ("persons_enrolled",    "mean"),
        households_enrolled_avg = ("households_enrolled", "mean"),
        benefits_total          = ("benefits_dollars",    "sum"),
        months_reported         = ("month",               "count"),
    )
    .reset_index()
    .rename(columns={"County": "county_name", "County_Code": "county_code"})
)
dfa_annual["persons_enrolled_avg"]    = dfa_annual["persons_enrolled_avg"].round(0)
dfa_annual["households_enrolled_avg"] = dfa_annual["households_enrolled_avg"].round(0)

print(f"  Rows: {len(dfa_annual)}")
print(f"  FFY range: {dfa_annual['FFY'].min()} :“ {dfa_annual['FFY'].max()}")
print(f"  Counties: {dfa_annual['county_name'].nunique()}")
print(f"\n  Sample:")
print(dfa_annual[["county_name","FFY","year",
                   "persons_enrolled_avg","households_enrolled_avg"]]
      .head(6).to_string(index=False))

dfa_annual.to_csv("data/dfa256_annual.csv", index=False)
print("\n  Saved: data/dfa256_annual.csv")


# =============================================================================
# 2. FEEDING AMERICA MAP THE MEAL GAP
#
#    Reads MMG2025_2019-2023_Data_To_Share.xlsx (covers 2019-2023).
#    To extend the panel further back, upload additional year files to the
#    same folder and they will be picked up automatically.
#
#    Sheet name varies across releases -- auto-detected below.
#    Columns used:
#      FIPS, State, 'County, State', Year
#      'Overall Food Insecurity Rate'   -- decimal (0.157 = 15.7%)
#      'Child Food Insecurity Rate'     -- decimal
#      '% FI <= SNAP Threshold'         -- decimal
#      '# of Food Insecure Persons Overall'
# =============================================================================
section("2. Feeding America Map the Meal Gap (2019-2023)")

mmg_raw = pd.read_excel(
    UPLOADS + "MMG2025_2019-2023_Data_To_Share.xlsx",
    sheet_name="County",
    dtype={"FIPS": str}
)
mmg_raw["FIPS"] = mmg_raw["FIPS"].str.zfill(5)

mmg = mmg_raw[mmg_raw["State"] == "CA"].copy()

mmg = mmg[[
    "FIPS",
    "County, State",
    "Year",
    "Overall Food Insecurity Rate",
    "# of Food Insecure Persons Overall",
    "Child Food Insecurity Rate",
    "# of Food Insecure Children",
    "% FI \u2264 SNAP Threshold",
    "Cost Per Meal",
]].rename(columns={
    "FIPS":                               "county_fips",
    "County, State":                      "county_name",
    "Year":                               "year",
    "Overall Food Insecurity Rate":       "food_insecurity_rate",
    "# of Food Insecure Persons Overall": "food_insecure_count",
    "Child Food Insecurity Rate":         "child_food_insecurity_rate",
    "# of Food Insecure Children":        "food_insecure_children",
    "% FI \u2264 SNAP Threshold":         "pct_fi_snap_eligible",
    "Cost Per Meal":                      "cost_per_meal",
})

# Rates are stored as decimals (0.157 = 15.7%) -- convert to percent
for col in ["food_insecurity_rate", "child_food_insecurity_rate", "pct_fi_snap_eligible"]:
    if mmg[col].dropna().max() <= 1.0:
        mmg[col] = (mmg[col] * 100).round(2)

mmg = mmg.sort_values(["county_fips", "year"]).reset_index(drop=True)

print(f"  Rows: {len(mmg)}  ({mmg['county_fips'].nunique()} counties x {mmg['year'].nunique()} years)")
print(f"  Years: {sorted(mmg['year'].unique())}")
print(f"\n  Sample:")
print(mmg[["county_name","year","food_insecurity_rate",
            "child_food_insecurity_rate","pct_fi_snap_eligible"]]
      .head(8).to_string(index=False))

mmg.to_csv("data/mmg_ca_panel.csv", index=False)
print("\n  Saved: data/mmg_ca_panel.csv")


# =============================================================================
# 3. FOOD ENVIRONMENT ATLAS :” ASSISTANCE + ACCESS sheets
#
#    ASSISTANCE sheet: per capita SNAP benefits (county-varying)
#      PC_SNAPBEN17  per capita SNAP benefits 2017
#      PC_SNAPBEN22  per capita SNAP benefits 2022
#
#    ACCESS sheet: food access barriers (all county-varying for CA)
#      PCT_LACCESS_POP19      % population with low food access
#      PCT_LACCESS_LOWI19     % low-income population with low food access
#      PCT_LACCESS_SNAP19     % SNAP recipients with low food access
#      PCT_LACCESS_HHNV19     % households: no vehicle + low food access
#
#    Note: Most other SNAP columns in ASSISTANCE are state-level constants
#    repeated for every county -- confirmed by inspection, not useful.
# =============================================================================
section("3. Food Environment Atlas :” ASSISTANCE + ACCESS sheets")

ATLAS_PATH = UPLOADS + "2025-food-environment-atlas-data.xlsx"

# --- ASSISTANCE ---
assist_raw = pd.read_excel(ATLAS_PATH, sheet_name="ASSISTANCE",
                            header=1, dtype={"FIPS": str})
assist_raw["FIPS"] = assist_raw["FIPS"].astype(str).str.zfill(5)
assist_ca = assist_raw[assist_raw["FIPS"].str.startswith("06")].copy()

ASSIST_WANT = {
    "FIPS":         "county_fips",
    "State":        "state",
    "County":       "county_name",
    "PC_SNAPBEN17": "snap_benefit_per_cap_2017",
    "PC_SNAPBEN22": "snap_benefit_per_cap_2022",
}
assist_ca = assist_ca[list(ASSIST_WANT.keys())].rename(columns=ASSIST_WANT)

# --- ACCESS ---
access_raw = pd.read_excel(ATLAS_PATH, sheet_name="ACCESS",
                            header=1, dtype={"FIPS": str})
access_raw["FIPS"] = access_raw["FIPS"].astype(str).str.zfill(5)
access_ca = access_raw[access_raw["FIPS"].str.startswith("06")].copy()

ACCESS_WANT = {
    "FIPS":                "county_fips",
    "PCT_LACCESS_POP19":   "pct_low_access_pop",
    "PCT_LACCESS_LOWI19":  "pct_low_access_lowincome",
    "PCT_LACCESS_SNAP19":  "pct_low_access_snap",
    "PCT_LACCESS_HHNV19":  "pct_low_access_novehicle",
}
access_ca = access_ca[list(ACCESS_WANT.keys())].rename(columns=ACCESS_WANT)

# Merge on county_fips
atlas = assist_ca.merge(access_ca, on="county_fips", how="left")

# -9999 = no data
for col in atlas.columns:
    if col not in ("county_fips", "state", "county_name"):
        atlas[col] = pd.to_numeric(atlas[col], errors="coerce")
        atlas[col] = atlas[col].where(atlas[col] != -9999, np.nan)

print(f"  CA counties: {len(atlas)}")
print(f"  Columns: {list(atlas.columns)}")
print(f"\n  Sample:")
print(atlas.head(8).to_string(index=False))

atlas.to_csv("data/food_env_atlas_ca.csv", index=False)
print("\n  Saved: data/food_env_atlas_ca.csv")


# =============================================================================
# 4. ACS B22003 :” Census API, all CA counties, 5-year estimates 2013-2023
#
#    Variables:
#      B22003_001E  total households (denominator)
#      B22003_002E  households receiving SNAP
#      B22003_006E  households below poverty NOT receiving SNAP  <- gap numerator
#    Plus margin of error (_M) for each.
# =============================================================================
section("4. ACS B22003 :” Census API (5-year estimates, county level)")

def fetch_b22003(year):
    variables = ("NAME,B22003_001E,B22003_001M,"
                 "B22003_002E,B22003_002M,"
                 "B22003_006E,B22003_006M")
    url = (f"https://api.census.gov/data/{year}/acs/acs5"
           f"?get={variables}&for=county:*&in=state:06")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.loads(r.read())
        df = pd.DataFrame(data[1:], columns=data[0])
        df["year"] = year
        return df
    except Exception as e:
        print(f"    {year}: {e}")
        return None

frames = []
for yr in range(2013, 2024):
    print(f"    Fetching {yr}...", end=" ", flush=True)
    df = fetch_b22003(yr)
    if df is not None:
        frames.append(df)
        print(f"OK ({len(df)} counties)")
    else:
        print("skipped")

if frames:
    acs = pd.concat(frames, ignore_index=True)
    acs = acs.rename(columns={
        "NAME":         "county_name",
        "B22003_001E":  "hh_total",
        "B22003_001M":  "hh_total_moe",
        "B22003_002E":  "hh_snap_enrolled",
        "B22003_002M":  "hh_snap_enrolled_moe",
        "B22003_006E":  "hh_poor_not_enrolled",
        "B22003_006M":  "hh_poor_not_enrolled_moe",
        "state":        "state_fips",
        "county":       "county_fips_suffix",
    })

    for col in ["hh_total","hh_total_moe","hh_snap_enrolled",
                "hh_snap_enrolled_moe","hh_poor_not_enrolled",
                "hh_poor_not_enrolled_moe"]:
        acs[col] = pd.to_numeric(acs[col], errors="coerce")
        acs[col] = acs[col].where(acs[col] >= 0, np.nan)

    acs["county_fips"] = acs["state_fips"] + acs["county_fips_suffix"]

    # Coverage gap rate: poor HHs not enrolled / total HHs
    acs["snap_participation_rate"] = (
        acs["hh_snap_enrolled"] / acs["hh_total"] * 100
    ).round(2)
    acs["coverage_gap_rate"] = (
        acs["hh_poor_not_enrolled"] / acs["hh_total"] * 100
    ).round(2)
    # Flag cells where MOE > 30% of estimate (high sampling uncertainty)
    acs["moe_unreliable"] = (
        acs["hh_poor_not_enrolled_moe"] > acs["hh_poor_not_enrolled"] * 0.30
    )

    acs = acs.sort_values(["county_fips", "year"]).reset_index(drop=True)

    print(f"\n  ACS panel: {len(acs)} county-year rows")
    print(f"  Years: {sorted(acs['year'].unique())}")
    print(f"  Unreliable MOE cells: {acs['moe_unreliable'].sum()} "
          f"({acs['moe_unreliable'].mean():.1%})")
    print(f"\n  Sample:")
    print(acs[["county_name","year","hh_snap_enrolled","hh_poor_not_enrolled",
                "snap_participation_rate","coverage_gap_rate","moe_unreliable"]]
          .head(8).to_string(index=False))

    acs.to_csv("data/b22003_ca_panel.csv", index=False)
    print("\n  Saved: data/b22003_ca_panel.csv")
else:
    print("\n  Census API not reachable in this environment.")
    print("  Run this script locally -- Census API pull is automatic.")




# =============================================================================
# 5. ACS B17001 :” Overall poverty rate, all CA counties, 5-year estimates
#
#    WHY OVERALL POVERTY AND NOT CHILD POVERTY?
#    CalFresh eligibility is based on household income regardless of whether
#    children are present. The program serves all household types. Food
#    insecurity (the outcome) is measured for the total population.
#    Overall poverty rate is the correct confounding variable to control for
#    :” child poverty rate would introduce a population mismatch.
#
#    Variables:
#      B17001_001E  total population for whom poverty status determined
#      B17001_002E  population below poverty level
#    Overall poverty rate = B17001_002E / B17001_001E * 100
# =============================================================================
section("5. ACS B17001 :” Overall poverty rate (5-year estimates, county level)")

def fetch_b17001(year):
    variables = "NAME,B17001_001E,B17001_002E"
    url = (f"https://api.census.gov/data/{year}/acs/acs5"
           f"?get={variables}&for=county:*&in=state:06")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.loads(r.read())
        df = pd.DataFrame(data[1:], columns=data[0])
        df["year"] = year
        return df
    except Exception as e:
        print(f"    {year}: {e}")
        return None

pov_frames = []
for yr in range(2019, 2023):   # 2019-2022 covers MMG panel period
    print(f"    Fetching {yr}...", end=" ", flush=True)
    df = fetch_b17001(yr)
    if df is not None:
        pov_frames.append(df)
        print(f"OK ({len(df)} counties)")
    else:
        print("skipped")

if pov_frames:
    pov = pd.concat(pov_frames, ignore_index=True)
    pov = pov.rename(columns={
        "NAME":         "county_name",
        "B17001_001E":  "pop_poverty_denom",
        "B17001_002E":  "pop_below_poverty",
        "state":        "state_fips",
        "county":       "county_fips_suffix",
    })

    for col in ["pop_poverty_denom", "pop_below_poverty"]:
        pov[col] = pd.to_numeric(pov[col], errors="coerce")
        pov[col] = pov[col].where(pov[col] >= 0, np.nan)

    pov["county_fips"] = pov["state_fips"] + pov["county_fips_suffix"]

    # Overall poverty rate
    pov["overall_poverty_rate"] = (
        pov["pop_below_poverty"] / pov["pop_poverty_denom"] * 100
    ).round(2)

    # Average across 2019-2022 for use in cross-sectional model
    pov_avg = (pov.groupby("county_name")["overall_poverty_rate"]
               .mean().round(2).reset_index()
               .rename(columns={"overall_poverty_rate": "overall_poverty_rate_avg"}))

    print(f"\n  Poverty panel: {len(pov)} county-year rows")
    print(f"  Years: {sorted(pov['year'].unique())}")
    print(f"\n  County average 2019-2022:")
    print(f"  Mean: {pov_avg['overall_poverty_rate_avg'].mean():.1f}%")
    print(f"  Range: {pov_avg['overall_poverty_rate_avg'].min():.1f}% - "
          f"{pov_avg['overall_poverty_rate_avg'].max():.1f}%")
    print(f"\n  Sample:")
    print(pov_avg.sort_values("overall_poverty_rate_avg", ascending=False)
          .head(5).to_string(index=False))

    pov.to_csv("data/b17001_ca_panel.csv", index=False)
    pov_avg.to_csv("data/b17001_ca_avg.csv", index=False)
    print("\n  Saved: data/b17001_ca_panel.csv")
    print("  Saved: data/b17001_ca_avg.csv")
else:
    print("\n  Census API not reachable in this environment.")
    print("  Run this script locally -- Census API pull is automatic.")



# =============================================================================
# SUMMARY
# =============================================================================
section("Summary")

files = [
    ("data/dfa256_annual.csv",     "DFA256 CalFresh admin (2004-2017), county x FFY"),
    ("data/mmg_ca_panel.csv",      "Feeding America MMG, county x year"),
    ("data/food_env_atlas_ca.csv", "Food Environment Atlas, CA counties cross-section"),
    ("data/b22003_ca_panel.csv",   "ACS B22003 SNAP/poverty, county x year (2013-2023)"),
    ("data/b17001_ca_panel.csv",   "ACS B17001 overall poverty rate, county x year (2019-2022)"),
    ("data/b17001_ca_avg.csv",     "ACS B17001 overall poverty rate, county avg 2019-2022"),
]
for path, desc in files:
    status = "SAVED   " if os.path.exists(path) else "MISSING "
    print(f"  [{status}] {path}")
    print(f"             {desc}")

print("""
  Panel date coverage:
    DFA256 admin data:        2004-2017
    ACS B22003:               2013-2023
    Food Environment Atlas:   2017 / 2019 / 2022 (cross-sections)
    Feeding America MMG:      depends on which year files you upload

  Overlapping analysis window:
    2019-2023  all sources
    2013-2023  ACS + Atlas only (longest panel, no admin data)

  Next: run calfresh_02_analyze.py
""")
