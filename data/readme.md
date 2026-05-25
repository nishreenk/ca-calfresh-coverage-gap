# Data Download Instructions

This folder holds the raw data files used in the analysis.
Most files cannot be included in the repository due to licensing.
Follow the instructions below to download each one.

---

## File 1: Feeding America Map the Meal Gap

**URL:** https://www.feedingamerica.org/research/map-the-meal-gap/by-county

1. Scroll to "Download the Data"
2. Download the multi-year Excel file: `MMG2025_2019-2023_Data_To_Share.xlsx`
3. Save to this `data/` folder

**Expected:** One Excel file with sheets for each year. The loading
script reads the county-level sheet and extracts California rows.

---

## File 2: USDA Food Environment Atlas

**URL:** https://www.ers.usda.gov/data-products/food-environment-atlas/

1. Click "Download the data"
2. Download: `2025-food-environment-atlas-data.xlsx`
3. Save to this `data/` folder

**Expected:** One Excel workbook with multiple sheets.
Only `ASSISTANCE` and `ACCESS` sheets are used.

**Important note:** Most SNAP participation columns in the Atlas are
state-level constants — identical for all California counties.
The loading script identifies and excludes these automatically.

---

## File 3: ACS B17001 — Overall Poverty Rate

Pulled automatically via the Census Bureau API when you run
`calfresh_01_load_data.py`. No manual download needed.

You need a free Census API key:
1. Go to https://api.census.gov/data/key_signup.html
2. Fill in name and email — key arrives by email immediately
3. Add your key to `calfresh_01_load_data.py`:

```python
API_KEY = "your_key_here"
```

The script generates two files:
- `data/b17001_ca_panel.csv` — county x year panel (2019–2022)
- `data/b17001_ca_avg.csv` — county averages 2019–2022

---

## File 4: DFA256 — California Admin CalFresh Data

**URL:** https://data.ca.gov (search "DFA256")

Monthly county-level CalFresh enrollment, 2004–2017.
Download `dfa256m.csv` and save to this `data/` folder.

**Note:** This file covers 2004–2017 only. The MMG panel starts in 2019,
creating a two-year gap. DFA256 is loaded and inspected in
`calfresh_01_load_data.py` but not used in the final regression.
It is included for completeness and for future analysis when
updated data becomes available.

---

## Generated Files (do not download — created by scripts)

After running the scripts in order, the following files are created:

| File | Created by | Description |
|---|---|---|
| `county_avg_2019_2023.csv` | `calfresh_01_load_data.py` | County averages, all sources merged |
| `mmg_ca_panel.csv` | `calfresh_01_load_data.py` | MMG panel, 58 counties x 5 years |
| `food_env_atlas_ca.csv` | `calfresh_01_load_data.py` | Atlas variables, California counties |
| `b22003_ca_panel.csv` | `calfresh_01_load_data.py` | ACS SNAP enrollment panel |
| `b17001_ca_panel.csv` | `calfresh_01_load_data.py` | ACS poverty rate panel |
| `b17001_ca_avg.csv` | `calfresh_01_load_data.py` | ACS poverty rate, county averages |
| `dfa256_annual.csv` | `calfresh_01_load_data.py` | DFA256 aggregated annually |
| `county_features.csv` | `calfresh_03_feature_engineering.py` | Final feature matrix, 58 counties |

---

## File Naming

The loading script auto-detects files by keyword in the filename:

| Dataset | Keyword |
|---|---|
| MMG | `MMG` or `MapTheMealGap` |
| Food Environment Atlas | `food-environment-atlas` |
| DFA256 | `dfa256` |

Filenames do not need to match exactly — just contain the keyword.

---

## Data Redistribution

Feeding America and USDA data are provided for research use.
This repository does not include the raw downloaded files.
Please download directly from the sources linked above.
