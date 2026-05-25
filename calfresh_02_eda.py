"""
CalFresh Coverage Gap & Exploratory Data Analysis
==================================================
Unit of analysis: county averages 2019-2023 (58 CA counties)

Sources merged:
  mmg_ca_panel.csv       -- food insecurity by county x year (MMG)
  food_env_atlas_ca.csv  -- food access barriers (Atlas, cross-section)
  b22003_ca_panel.csv    -- SNAP enrollment + poor-not-enrolled (ACS)
                           if not yet available, MMG-only analysis runs

Steps:
  1. Build county-average dataset
  2. Univariate analysis -- distributions, skew, outliers
  3. Bivariate analysis -- correlations, scatterplots
  4. Notes for feature engineering
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs("/your/path/to/data/" , exist_ok=True)
os.makedirs("/your/path/to/data/" , exist_ok=True)

DATA = "/your/path/to/data/"      # folder containing downloaded data files
FIG  = "/your/path/to/data/"    # folder for output figures

# --------------------------------------------------------------------------
# Override DATA path to use outputs from this session
# --------------------------------------------------------------------------
output = "/your/path/to/outputs/"


def section(title):
    print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")


# =============================================================================
# 1. BUILD COUNTY-AVERAGE DATASET
# =============================================================================
section("1. Building county-average dataset (2019-2023)")

# --- MMG: average across 2019-2023 ---
mmg = pd.read_csv(DATA + "mmg_ca_panel.csv")
mmg["county_fips"] = mmg["county_fips"].astype(str).str.zfill(5)

mmg_avg = (
    mmg.groupby(["county_fips","county_name"])
    .agg(
        food_insecurity_rate       = ("food_insecurity_rate",       "mean"),
        child_food_insecurity_rate = ("child_food_insecurity_rate", "mean"),
        pct_fi_snap_eligible       = ("pct_fi_snap_eligible",       "mean"),
        food_insecure_count        = ("food_insecure_count",        "mean"),
        cost_per_meal              = ("cost_per_meal",              "mean"),
    )
    .round(2)
    .reset_index()
)

# Clean county name: strip ", California" suffix
mmg_avg["county_short"] = (
    mmg_avg["county_name"]
    .str.replace(" County, California", "", regex=False)
    .str.strip()
)

# --- Atlas: already a cross-section, one row per county ---
atlas = pd.read_csv(DATA + "food_env_atlas_ca.csv")
atlas["county_fips"] = atlas["county_fips"].astype(str).str.zfill(5)

# --- ACS B22003: average 2019-2023 if available ---
acs_path = DATA + "b22003_ca_panel.csv"
if os.path.exists(acs_path):
    acs = pd.read_csv(acs_path)
    acs["county_fips"] = acs["county_fips"].astype(str).str.zfill(5)
    acs_1923 = acs[acs["year"].between(2019, 2023)].copy()
    acs_avg = (
        acs_1923.groupby("county_fips")
        .agg(
            snap_participation_rate = ("snap_participation_rate", "mean"),
            coverage_gap_rate       = ("coverage_gap_rate",       "mean"),
            hh_snap_enrolled        = ("hh_snap_enrolled",        "mean"),
            hh_poor_not_enrolled    = ("hh_poor_not_enrolled",    "mean"),
            hh_total                = ("hh_total",                "mean"),
            moe_unreliable_pct      = ("moe_unreliable",          "mean"),
        )
        .round(2)
        .reset_index()
    )
    has_acs = True
    print("  ACS B22003 loaded.")
else:
    has_acs = False
    print("  ACS B22003 not found -- run fc_01_load_data.py locally to get it.")
    print("  Continuing with MMG + Atlas only.")

# --- Merge ---
df = mmg_avg.merge(atlas, on="county_fips", how="left")
if has_acs:
    df = df.merge(acs_avg, on="county_fips", how="left")

print(f"\n  Counties: {len(df)}")
print(f"  Columns:  {list(df.columns)}")
print(f"\n  First 5 rows:")
print(df[["county_short","food_insecurity_rate","pct_fi_snap_eligible",
           "pct_low_access_lowincome","snap_benefit_per_cap_2022"]].head())

df.to_csv(output + "county_avg_2019_2023.csv", index=False)
print(f"\n  Saved: {DATA}county_avg_2019_2023.csv")


# =============================================================================
# 2. UNIVARIATE ANALYSIS
# =============================================================================
section("2. Univariate analysis")

# Variables to examine
UNI_VARS = {
    "food_insecurity_rate":       "Food insecurity rate (%)",
    "child_food_insecurity_rate": "Child food insecurity rate (%)",
    "pct_fi_snap_eligible":       "% food insecure below SNAP threshold",
    "cost_per_meal":              "Cost per meal ($)",
    "snap_benefit_per_cap_2022":  "SNAP benefit per capita 2022 ($)",
    "pct_low_access_pop":         "% pop with low food access",
    "pct_low_access_lowincome":   "% low-income with low food access",
    "pct_low_access_snap":        "% SNAP recipients with low access",
    "pct_low_access_novehicle":   "% HH: low access + no vehicle",
}
if has_acs:
    UNI_VARS["snap_participation_rate"] = "SNAP participation rate (ACS %)"
    UNI_VARS["coverage_gap_rate"]       = "Coverage gap rate (ACS %)"

print(f"\n  {'Variable':<35} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7} "
      f"{'Skew':>7} {'Missing':>8}")
print("  " + "-" * 78)

skew_flags = []
for col, label in UNI_VARS.items():
    if col not in df.columns:
        continue
    s = df[col].dropna()
    skew = s.skew()
    miss = df[col].isna().sum()
    flag = " <-- skewed" if abs(skew) > 1.0 else ""
    if abs(skew) > 1.0:
        skew_flags.append(col)
    print(f"  {label:<35} {s.mean():>7.2f} {s.std():>7.2f} "
          f"{s.min():>7.2f} {s.max():>7.2f} {skew:>7.2f} {miss:>8}")

if skew_flags:
    print(f"\n  Skewed variables (|skew| > 1.0) -- consider log transform:")
    for v in skew_flags:
        print(f"    {v}")

# --- Univariate plots ---
n_vars = len([c for c in UNI_VARS if c in df.columns])
ncols  = 3
nrows  = int(np.ceil(n_vars / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
axes = axes.flatten()
fig.suptitle("Univariate distributions & CA county averages 2019-2023",
             fontsize=13, fontweight="bold")

plot_vars = [(c, l) for c, l in UNI_VARS.items() if c in df.columns]
for i, (col, label) in enumerate(plot_vars):
    ax = axes[i]
    data = df[col].dropna()
    ax.hist(data, bins=15, color="#457B9D", edgecolor="white", alpha=0.85)
    ax.axvline(data.mean(), color="#E63946", ls="--", lw=1.5,
               label=f"Mean={data.mean():.1f}")
    ax.axvline(data.median(), color="#2A9D8F", ls=":", lw=1.5,
               label=f"Median={data.median():.1f}")
    ax.set_title(label, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(FIGS + "01_univariate.png", dpi=150, bbox_inches="tight")
print(f"\n  Figure saved: {FIGS}01_univariate.png")


# =============================================================================
# 3. BIVARIATE ANALYSIS
# =============================================================================
section("3. Bivariate analysis")

# Outcome: food_insecurity_rate (and coverage_gap_rate if ACS available)
OUTCOMES = ["food_insecurity_rate"]
if has_acs and "coverage_gap_rate" in df.columns:
    OUTCOMES.append("coverage_gap_rate")

PREDICTORS = [
    ("pct_low_access_lowincome", "% low-income, low food access"),
    ("pct_low_access_snap",      "% SNAP recipients, low access"),
    ("pct_low_access_novehicle", "% HH: low access + no vehicle"),
    ("pct_low_access_pop",       "% total pop, low food access"),
    ("snap_benefit_per_cap_2022","SNAP benefit per capita 2022 ($)"),
    ("pct_fi_snap_eligible",     "% food insecure below SNAP threshold"),
    ("cost_per_meal",            "Cost per meal ($)"),
]
if has_acs:
    PREDICTORS.append(("snap_participation_rate", "SNAP participation rate (ACS %)"))

print(f"\n  Pearson correlations with food insecurity rate:")
print(f"  {'Predictor':<40} {'r':>7} {'p':>8} {'sig':>5}")
print("  " + "-" * 62)

corr_results = []
for col, label in PREDICTORS:
    if col not in df.columns:
        continue
    sub = df[[col, "food_insecurity_rate"]].dropna()
    r, p = stats.pearsonr(sub[col], sub["food_insecurity_rate"])
    sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {label:<40} {r:>7.3f} {p:>8.4f} {sig:>5}")
    corr_results.append((col, label, r, p))

# --- Scatterplots ---
n_pred = len([c for c, l in PREDICTORS if c in df.columns])
ncols  = 3
nrows  = int(np.ceil(n_pred / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4))
axes = axes.flatten()
fig.suptitle("Bivariate: predictors vs food insecurity rate\n"
             "CA county averages 2019-2023  (dot size = county population)",
             fontsize=12, fontweight="bold")

for i, (col, label) in enumerate([(c,l) for c,l in PREDICTORS if c in df.columns]):
    ax   = axes[i]
    sub  = df[["county_short", col, "food_insecurity_rate",
                "food_insecure_count"]].dropna()
    size = np.sqrt(sub["food_insecure_count"]) / 8

    ax.scatter(sub[col], sub["food_insecurity_rate"],
               s=size, alpha=0.6, color="#457B9D", edgecolors="white", lw=0.5)

    # Regression line
    m, b, r, p, _ = stats.linregress(sub[col], sub["food_insecurity_rate"])
    xs = np.linspace(sub[col].min(), sub[col].max(), 100)
    ax.plot(xs, m*xs + b, color="#E63946", lw=1.5,
            label=f"r={r:.2f}{'*' if p<0.05 else ''}")

    # Label top 5 outliers by residual
    sub = sub.copy()
    sub["fitted"]  = m * sub[col] + b
    sub["resid"]   = (sub["food_insecurity_rate"] - sub["fitted"]).abs()
    top5 = sub.nlargest(5, "resid")
    for _, row in top5.iterrows():
        ax.annotate(row["county_short"],
                    xy=(row[col], row["food_insecurity_rate"]),
                    fontsize=6, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(label, fontsize=8)
    ax.set_ylabel("Food insecurity rate (%)", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(FIGS + "02_bivariate.png", dpi=150, bbox_inches="tight")
print(f"\n  Figure saved: {FIGS}02_bivariate.png")


# =============================================================================
# 4. CORRELATION MATRIX
# =============================================================================
section("4. Correlation matrix")

MATRIX_VARS = [c for c, l in PREDICTORS if c in df.columns] + OUTCOMES
matrix_df   = df[MATRIX_VARS].dropna()
corr_matrix = matrix_df.corr().round(2)

print(f"\n  Correlation matrix ({len(matrix_df)} counties with complete data):")
print(corr_matrix.to_string())

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)

short_labels = [c.replace("pct_","").replace("_"," ").replace("low access","LA")
                for c in corr_matrix.columns]
ax.set_xticks(range(len(corr_matrix)))
ax.set_yticks(range(len(corr_matrix)))
ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(short_labels, fontsize=8)

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        val = corr_matrix.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7, color="black" if abs(val) < 0.7 else "white")

ax.set_title("Correlation matrix — CA county averages 2019-2023", fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS + "03_correlation_matrix.png", dpi=150, bbox_inches="tight")
print(f"\n  Figure saved: {FIGS}03_correlation_matrix.png")


# =============================================================================
# 5. OUTLIER CHECK & flag counties with extreme values
# =============================================================================
section("5. Outlier check")

flag_cols = ["food_insecurity_rate","pct_low_access_pop","pct_low_access_lowincome"]
for col in flag_cols:
    if col not in df.columns:
        continue
    q1  = df[col].quantile(0.25)
    q3  = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = df[
        (df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)
    ][["county_short", col]].sort_values(col, ascending=False)
    if len(outliers):
        print(f"\n  Outliers in {col}:")
        print(outliers.to_string(index=False))


# =============================================================================
# 6. FEATURE ENGINEERING NOTES
# =============================================================================
section("6. Feature engineering notes (to act on next)")

print("""
  Based on the univariate and bivariate analysis above:

  TRANSFORMATIONS TO CONSIDER:
    pct_low_access_pop        -- likely right-skewed; try log(x + 1)
    pct_low_access_lowincome  -- likely right-skewed; try log(x + 1)
    food_insecure_count       -- raw count, use as population weight not feature
    snap_benefit_per_cap      -- check direction: higher benefit = higher need
                                 OR better program reach? Interpret carefully.

  INTERACTIONS TO CONSIDER:
    pct_low_access_lowincome x pct_fi_snap_eligible
      -- counties where many poor people can't reach food AND can't get SNAP
         may have a compounding coverage gap

    pct_low_access_novehicle x cost_per_meal
      -- no transport + high food cost = most constrained households

  POTENTIAL DERIVED VARIABLES:
    snap_benefit_change = snap_benefit_per_cap_2022 - snap_benefit_per_cap_2017
      -- did SNAP benefit reach grow between 2017 and 2022?

    unmet_need = food_insecurity_rate x (1 - pct_fi_snap_eligible/100)
      -- food insecure people who likely can't get SNAP even if they tried

  COLLINEARITY TO WATCH:
    The four low-access variables will be correlated with each other.
    Use pct_low_access_lowincome as the primary; others as robustness checks.
    Check VIF before including multiple access variables in the same model.

  SMALL COUNTY FLAG:
    Alpine, Sierra, Mono -- tiny populations, high variance.
    Consider sensitivity analysis dropping counties with population < 10,000.
""")

print("=== EDA complete ===")
print(f"  Figures in: {FIGS}")
print(f"  Merged dataset: {DATA}county_avg_2019_2023.csv")
