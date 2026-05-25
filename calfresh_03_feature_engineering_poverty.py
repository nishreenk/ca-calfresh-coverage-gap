"""
CalFresh Coverage Gap : Feature Engineering
=============================================
Transforms the raw county-average dataset into an analysis-ready feature
matrix. Every decision here is grounded in the EDA findings.

Input:  county_avg_2019_2023.csv  (58 CA counties, averages 2019-2023)
Output: county_features.csv       (same 58 rows, engineered features)

Run after calfresh_02_eda.py.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("figures", exist_ok=True)
DATA = "/your/path/to/data/"      # folder containing downloaded data files
FIG  = "/your/path/to/figures/"   # folder for output figures

def section(title):
    print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")


# =============================================================================
# LOAD
# =============================================================================

df = pd.read_csv(DATA + "county_avg_2019_2023.csv")

# Clean up duplicate county name columns from the merge
df = df.rename(columns={"county_name_x": "county_name"})
df = df.drop(columns=["county_name_y", "state"], errors="ignore")
df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

print(f"Loaded: {len(df)} counties")


# =============================================================================
# STEP 1 : LOG TRANSFORMS FOR RIGHT-SKEWED ACCESS VARIABLES
# =============================================================================
section("Step 1 : Log transforms")

from scipy.stats import boxcox

def best_transform(df, col, prefer=None):
    """
    Test log(x+1), sqrt(x), and Box-Cox transforms for a right-skewed
    variable. Apply the one that brings skewness closest to zero, unless
    a specific transform is preferred (prefer='log' or prefer='sqrt').

    WHY COMPARE TRANSFORMS?
    Different variables respond differently to power transformations.
    A blanket log transform can overcorrect some variables -- flipping
    positive skew to negative skew -- while undercorrecting others.
    Testing all three and picking the best for each variable ensures the
    transformation is data-driven and documented, not assumed.

    TRANSFORMS TESTED:
    - log(x + 1): natural log with +1 offset to handle near-zero values.
      Strongest compression -- best for highly skewed distributions.
      Coefficient interpretation: a 1-unit increase in log(x) is
      associated with beta units of y (semi-log).
    - sqrt(x): square root. Lighter compression than log -- better when
      log overcorrects (flips skew from positive to negative).
      Works on zero values without an offset.
    - Box-Cox: finds the optimal power lambda that minimizes skewness.
      Requires all values > 0; uses scipy.stats.boxcox. More flexible
      than fixed-power transforms but harder to interpret directly.

    WHY sqrt FOR pct_low_access_pop?
    Log transform flipped skew from +2.20 to -1.31 -- worse, not better.
    sqrt brought it to +0.46 -- close to symmetric without overcorrecting.
    Box-Cox gave -0.34. sqrt was chosen as the most interpretable option
    that achieved near-symmetry.

    WHY log FOR pct_low_access_lowincome AND pct_low_access_novehicle?
    Log brought skewness from 1.57 to -0.26 and from 1.46 to 0.37
    respectively -- both acceptable. sqrt also worked but log was
    already applied and the results were satisfactory.
    """
    raw   = df[col].dropna()
    raw_skew = raw.skew()

    candidates = {}

    # Log(x+1)
    log_vals  = np.log(raw + 1)
    candidates['log'] = (log_vals, abs(log_vals.skew()), f'log_{col}')

    # Sqrt(x)
    sqrt_vals = np.sqrt(raw)
    candidates['sqrt'] = (sqrt_vals, abs(sqrt_vals.skew()), f'sqrt_{col}')

    # Box-Cox (requires all values > 0; add small offset if needed)
    try:
        bc_input = raw + 1e-6 if (raw <= 0).any() else raw
        bc_vals, lam = boxcox(bc_input)
        bc_series = pd.Series(bc_vals, index=raw.index)
        candidates['boxcox'] = (bc_series, abs(bc_series.skew()),
                                f'boxcox_{col}')
    except Exception:
        pass

    # Pick best (closest skew to zero) unless preference specified
    if prefer and prefer in candidates:
        chosen_name = prefer
    else:
        chosen_name = min(candidates, key=lambda k: candidates[k][1])

    chosen_vals, chosen_skew, new_col = candidates[chosen_name]
    df[new_col] = chosen_vals.reindex(df.index)

    print(f"  {col}")
    print(f"    raw skew: {raw_skew:.2f}")
    for name, (_, sk, _) in candidates.items():
        marker = " <-- CHOSEN" if name == chosen_name else ""
        print(f"    {name:8s} skew: {sk:+.2f}{marker}")
    print(f"    output column: {new_col}")
    return df, new_col

# Apply best transform to each skewed variable.
# pct_low_access_pop: log overcorrects (+2.20 -> -1.31), sqrt chosen (+0.46).
# pct_low_access_lowincome and pct_low_access_novehicle: log works well.
TRANSFORM_SPECS = [
    ("pct_low_access_pop",       "sqrt"),
    ("pct_low_access_lowincome", "log"),
    ("pct_low_access_novehicle", "log"),
]

transformed_cols = {}
for col, prefer in TRANSFORM_SPECS:
    df, new_col = best_transform(df, col, prefer=prefer)
    transformed_cols[col] = new_col


# =============================================================================
# STEP 2 : SNAP BENEFIT GROWTH
# =============================================================================
section("Step 2 : SNAP benefit growth (2017 to 2022)")

def snap_benefit_growth(df):
    """
    Derive percentage change in per-capita SNAP benefits from 2017 to 2022.

    WHY THIS VARIABLE?
    We have two cross-sectional benefit snapshots from the Food Environment
    Atlas: 2017 and 2022. The raw 2022 level is already in the dataset and
    correlates strongly (r=0.83) with food insecurity -- but as discussed in
    the EDA, that correlation reflects need, not program reach.

    The *growth* in benefits is a different signal: counties where per-capita
    benefits grew faster between 2017 and 2022 may have expanded program
    reach (more people enrolled) or increased benefit generosity (the 2021
    pandemic SNAP expansion). This variable captures whether the program
    responded to need over that period.

    We use percentage change rather than absolute dollar change so that
    growth is comparable across counties with very different baseline levels:
    a $5 increase in a county with $10/capita baseline (50% growth) means
    something very different from a $5 increase in a county with $50/capita.

    Interpretation in regression: a positive coefficient would mean counties
    where SNAP benefits grew faster tended to have higher food insecurity --
    consistent with the program expanding in response to rising need.
    A negative coefficient would suggest benefit growth was associated with
    *reducing* food insecurity over the period.
    """
    df["snap_benefit_growth_pct"] = (
        (df["snap_benefit_per_cap_2022"] - df["snap_benefit_per_cap_2017"])
        / df["snap_benefit_per_cap_2017"] * 100
    ).round(2)

    print(f"  Mean growth:   {df['snap_benefit_growth_pct'].mean():.1f}%")
    print(f"  Range:         {df['snap_benefit_growth_pct'].min():.1f}% "
          f"to {df['snap_benefit_growth_pct'].max():.1f}%")
    print(f"  Skew:          {df['snap_benefit_growth_pct'].skew():.2f}")

    top3 = df.nlargest(3, "snap_benefit_growth_pct")[
        ["county_short","snap_benefit_growth_pct","snap_benefit_per_cap_2017",
         "snap_benefit_per_cap_2022"]]
    print(f"\n  Highest growth counties:")
    print(top3.to_string(index=False))
    return df

df = snap_benefit_growth(df)


# =============================================================================
# STEP 3 : UNMET NEED INDEX
# =============================================================================
section("Step 3 : Unmet need index")

def unmet_need_index(df):
    """
    Construct a county-level estimate of food-insecure people who are
    unlikely to qualify for SNAP even if they tried to enroll.

    FORMULA:
      unmet_need_rate = food_insecurity_rate * (1 - pct_fi_snap_eligible / 100)

    WHAT IT MEASURES:
    pct_fi_snap_eligible is the share of food-insecure people whose income
    falls at or below the SNAP eligibility threshold (130% FPL for most
    households, higher in states with broad-based categorical eligibility).
    Its complement -- (1 - pct_fi_snap_eligible/100) -- is the share of
    food-insecure people who are ABOVE the income threshold and therefore
    cannot receive SNAP benefits regardless of outreach or enrollment effort.

    Multiplying by food_insecurity_rate gives the estimated share of the
    total population who are food insecure AND ineligible for SNAP. This
    is a structural gap -- it cannot be closed by better program administration
    or outreach. It requires either income support, benefit expansion, or
    eligibility rule changes (like AB 1045 in California, which extended
    CalFresh to SSI recipients).

    WHY THIS MATTERS FOR THE RESEARCH QUESTION:
    The coverage gap project asks why some counties have larger gaps between
    food need and SNAP enrollment. This variable helps separate two types of
    gap: (a) eligible people not enrolled (addressable through outreach) vs
    (b) ineligible people who need food but can't get SNAP (structural).
    Counties with high unmet_need_rate have a fundamentally different policy
    problem than counties where the gap is driven by non-enrollment.

    INTERPRETATION RANGE:
    If food_insecurity_rate = 15% and pct_fi_snap_eligible = 70%, then
    unmet_need_rate = 15 * (1 - 0.70) = 4.5% of the population are food
    insecure and structurally ineligible for SNAP.
    """
    df["unmet_need_rate"] = (
        df["food_insecurity_rate"] * (1 - df["pct_fi_snap_eligible"] / 100)
    ).round(3)

    print(f"  Mean unmet need rate:  {df['unmet_need_rate'].mean():.2f}%")
    print(f"  Range:                 {df['unmet_need_rate'].min():.2f}% "
          f"to {df['unmet_need_rate'].max():.2f}%")
    print(f"  Skew:                  {df['unmet_need_rate'].skew():.2f}")

    extremes = pd.concat([
        df.nlargest(3,  "unmet_need_rate")[["county_short","unmet_need_rate",
                                             "food_insecurity_rate","pct_fi_snap_eligible"]],
        df.nsmallest(3, "unmet_need_rate")[["county_short","unmet_need_rate",
                                             "food_insecurity_rate","pct_fi_snap_eligible"]],
    ])
    print(f"\n  Highest and lowest unmet need counties:")
    print(extremes.to_string(index=False))
    return df

df = unmet_need_index(df)


# =============================================================================
# STEP 4 : ACCESS x ELIGIBILITY INTERACTION
# =============================================================================
section("Step 4 : Access x eligibility interaction")

def access_eligibility_interaction(df):
    """
    Create an interaction term between food access barriers and SNAP eligibility.

    FORMULA:
      access_x_eligible = log_pct_low_access_lowincome * pct_fi_snap_eligible

    THEORETICAL MOTIVATION:
    The bivariate EDA showed that pct_low_access_lowincome (r=0.35) and
    pct_fi_snap_eligible (r=0.79) each correlate with food insecurity
    independently. But the combination may matter differently:

    - A county where many low-income people lack food access AND most food-
      insecure people are SNAP-eligible faces a compound problem: people
      qualify for benefits but can't reach food retailers to use them, AND
      the enrollment process itself may be harder to navigate without
      stable access to services.

    - A county where most food-insecure people are above the SNAP threshold
      (low eligibility) but also have poor food access faces a different
      challenge -- the food access problem can't be solved by SNAP at all.

    The interaction term lets the regression estimate whether the effect of
    access barriers on food insecurity is stronger or weaker depending on
    how SNAP-eligible the food-insecure population is.

    WHY THE LOG TRANSFORM OF THE ACCESS VARIABLE?
    We use the already-transformed log_pct_low_access_lowincome rather than
    the raw value to prevent the interaction term from being dominated by the
    outlier counties (Sierra, Alpine, Mono) whose extreme access values would
    otherwise create extreme interaction values that overwhelm the regression.

    COLLINEARITY NOTE:
    Interaction terms are by construction correlated with their component
    variables. This is expected and not a problem as long as we include both
    main effects (log_pct_low_access_lowincome and pct_fi_snap_eligible) in
    the model alongside the interaction. We will check VIF in the analysis
    script and may drop the interaction if it inflates variance too much.
    """
    df["access_x_eligible"] = (
        df["log_pct_low_access_lowincome"] * df["pct_fi_snap_eligible"]
    ).round(4)

    from scipy import stats
    r, p = stats.pearsonr(df["access_x_eligible"], df["food_insecurity_rate"])
    print(f"  Correlation with food insecurity rate: r={r:.3f}  p={p:.4f}")
    print(f"  Range: {df['access_x_eligible'].min():.2f} to {df['access_x_eligible'].max():.2f}")
    return df

df = access_eligibility_interaction(df)


# =============================================================================
# STEP 5 : SMALL COUNTY FLAG
# =============================================================================
section("Step 5 : Small county flag")

def small_county_flag(df, threshold=10_000):
    """
    Flag counties with fewer than 10,000 food-insecure persons on average.

    WHY FLAG RATHER THAN DROP?
    The outlier analysis identified Alpine (162 food-insecure persons avg),
    Sierra, Mono, and Modoc as counties whose access variable values are
    extreme. These counties are small enough that their MMG food insecurity
    estimates carry high uncertainty -- Feeding America's model is fitted
    at the state level and applied to county demographics, and with very
    small populations the county-level estimates are less stable.

    Dropping them would reduce our already-small sample (N=58) and could
    introduce selection bias (rural small counties have systematically
    different characteristics). Instead we flag them so we can:
      (a) run the main regression on all 58 counties
      (b) run a sensitivity analysis on the 54 non-flagged counties
      (c) compare results and note whether small counties drive findings

    THRESHOLD CHOICE:
    10,000 food-insecure persons is approximately the bottom decile of the
    distribution. It corresponds roughly to counties with total populations
    under ~80,000, which captures the four outlier counties identified in
    the EDA without flagging any county that contributes meaningfully to
    statewide food insecurity totals.
    """
    df["small_county"] = (df["food_insecure_count"] < threshold).astype(int)
    flagged = df[df["small_county"] == 1]["county_short"].tolist()
    print(f"  Threshold: < {threshold:,} avg food-insecure persons")
    print(f"  Flagged ({len(flagged)}): {', '.join(flagged)}")
    return df

df = small_county_flag(df, threshold=10_000)


# =============================================================================
# STEP 6 : STANDARDISE FEATURES FOR REGRESSION
# =============================================================================
section("Step 6 : Standardise features (z-scores)")

def standardise(df, cols):
    """
    Z-score standardise all features that will enter the regression.

    WHY STANDARDISE?
    Our predictors are on very different scales:
      - food_insecurity_rate:       7.7 to 19.5  (percent)
      - snap_benefit_per_cap_2022:  11 to 74     (dollars)
      - log_pct_low_access_lowincome: 0.1 to 3.2 (log percent)
      - cost_per_meal:              3.1 to 5.1   (dollars)

    Without standardisation, regression coefficients are not comparable to
    each other -- a coefficient of 0.5 on cost_per_meal (dollar scale) means
    something very different from 0.5 on log_pct_low_access_lowincome.

    Z-scoring (subtract mean, divide by std) puts every variable on a
    common scale: a one-unit change in any standardised variable equals one
    standard deviation change in the original. This lets us compare
    coefficient magnitudes directly as a measure of relative importance.

    NOTE: We standardise predictors but NOT the outcome (food_insecurity_rate).
    Keeping the outcome in its original percent scale makes the coefficients
    directly interpretable: 'a one-SD increase in X is associated with a
    beta percentage-point change in food insecurity rate.'

    We create _z suffix columns and keep the originals so the dataset can
    be used for both the standardised regression and descriptive reporting.
    """
    for col in cols:
        if col not in df.columns:
            print(f"  SKIP (not found): {col}")
            continue
        mean = df[col].mean()
        std  = df[col].std()
        df[f"{col}_z"] = ((df[col] - mean) / std).round(4)
        print(f"  {col:<40}  mean={mean:.3f}  std={std:.3f}")
    return df

FEATURES_TO_STANDARDISE = [
    "log_pct_low_access_lowincome",   # primary access variable (log-transformed)
    "log_pct_low_access_novehicle",   # transport barrier (log-transformed)
    "pct_low_access_snap",            # SNAP recipients with access barrier
    "snap_benefit_per_cap_2022",      # SNAP benefit level (need proxy)
    "snap_benefit_growth_pct",        # benefit growth 2017-2022
    "pct_fi_snap_eligible",           # share of FI pop below SNAP threshold
    "cost_per_meal",                  # local food cost
    "unmet_need_rate",                # structural ineligibility
    "access_x_eligible",              # interaction term
]

df = standardise(df, FEATURES_TO_STANDARDISE)


# =============================================================================
# STEP 7 : BEFORE / AFTER COMPARISON PLOT
# =============================================================================
section("Step 7 : Before / after distribution plots")

# Use actual output column names from best_transform
compare_pairs = [
    ("pct_low_access_pop",
     transformed_cols["pct_low_access_pop"],
     "% pop low food access (raw)",
     "sqrt(% pop low food access)  [sqrt chosen: log overcorrected]"),
    ("pct_low_access_lowincome",
     transformed_cols["pct_low_access_lowincome"],
     "% low-income low access (raw)",
     "log(% low-income low access + 1)"),
    ("pct_low_access_novehicle",
     transformed_cols["pct_low_access_novehicle"],
     "% HH low access + no vehicle (raw)",
     "log(% HH low access + no vehicle + 1)"),
]

fig, axes = plt.subplots(len(compare_pairs), 2, figsize=(12, 10))
fig.suptitle("Variable transformations: before and after\n"
             "Red dashed = mean, green dotted = median",
             fontsize=12, fontweight="bold")

for i, (raw_col, log_col, raw_label, log_label) in enumerate(compare_pairs):
    for j, (col, label) in enumerate([(raw_col, raw_label), (log_col, log_label)]):
        ax   = axes[i, j]
        data = df[col].dropna()
        ax.hist(data, bins=15, color="#457B9D" if j==0 else "#2A9D8F",
                edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(),   color="#E63946", ls="--", lw=1.5)
        ax.axvline(data.median(), color="#2A9D8F", ls=":",  lw=1.5)
        ax.set_title(f"{label}\nskew={data.skew():.2f}", fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG+"04_log_transforms.png", dpi=150, bbox_inches="tight")
print("  Saved: figures/04_log_transforms.png")


# =============================================================================
# STEP 8 : VARIANCE INFLATION FACTOR CHECK
# =============================================================================
section("Step 8 : VIF check (multicollinearity)")

def vif_check(df, feature_cols):
    """
    Compute Variance Inflation Factor (VIF) for each feature.

    WHAT IS VIF?
    VIF measures how much the variance of a regression coefficient is
    inflated due to collinearity with other predictors. It is calculated
    by regressing each feature on all other features and computing:
      VIF_j = 1 / (1 - R²_j)
    where R²_j is how well feature j is predicted by all the others.

    INTERPRETATION:
      VIF = 1.0   no collinearity (feature is orthogonal to all others)
      VIF < 5.0   acceptable
      VIF 5-10    moderate concern -- consider dropping or combining
      VIF > 10    severe -- coefficients are unstable, standard errors
                  are inflated, and results are unreliable

    WHY CHECK NOW (BEFORE REGRESSION)?
    The EDA showed pct_low_access_pop correlates 0.91 with
    pct_low_access_lowincome. If both enter the model, neither will
    have a stable coefficient. Checking VIF here tells us which
    features to drop before we run a single regression, so we don't
    end up fitting a model and then discovering the coefficients are
    meaningless.

    We check VIF on the standardised (_z) versions since that is what
    will enter the regression.
    """
    from numpy.linalg import lstsq

    z_cols = [f"{c}_z" for c in feature_cols if f"{c}_z" in df.columns]
    X = df[z_cols].dropna()

    vifs = {}
    for i, col in enumerate(z_cols):
        y_      = X[col].values
        X_other = X.drop(columns=[col]).values
        # Add constant
        X_other = np.column_stack([np.ones(len(X_other)), X_other])
        coef, _, _, _ = lstsq(X_other, y_, rcond=None)
        y_hat = X_other @ coef
        ss_res = np.sum((y_ - y_hat)**2)
        ss_tot = np.sum((y_ - y_.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        vifs[col] = round(vif, 2)

    print(f"\n  {'Feature':<45} {'VIF':>6}  {'Flag'}")
    print("  " + "-" * 60)
    for col, vif in sorted(vifs.items(), key=lambda x: -x[1]):
        flag = "  <-- DROP (severe)"    if vif > 10 else \
               "  <-- WATCH (moderate)" if vif > 5  else ""
        print(f"  {col:<45} {vif:>6.2f}{flag}")

    return vifs

VIF_FEATURES = [
    "log_pct_low_access_lowincome",
    "log_pct_low_access_novehicle",
    "pct_low_access_snap",
    "snap_benefit_per_cap_2022",
    "snap_benefit_growth_pct",
    "pct_fi_snap_eligible",
    "cost_per_meal",
    "access_x_eligible",
]
vifs = vif_check(df, VIF_FEATURES)


# =============================================================================
# MERGE OVERALL POVERTY RATE (ACS B17001)
# =============================================================================
section("Merging overall poverty rate")

# Overall poverty rate is the correct confounding variable for CalFresh
# analysis -- it covers all age groups, matching both the CalFresh eligible
# population and the MMG food insecurity outcome (total population).
# Child poverty rate would introduce a population mismatch.
#
# Source: ACS B17001, 2019-2022 county averages, from calfresh_01_load_data.py

pov_path = DATA + "b17001_ca_avg.csv"
if os.path.exists(pov_path):
    pov_avg = pd.read_csv(pov_path)
    # Harmonize county name -- strip ' County, California'
    pov_avg["county_short"] = (pov_avg["county_name"]
                               .str.replace(" County, California", "", regex=False)
                               .str.strip())
    # Drop county_name to avoid conflicts
    pov_avg = pov_avg.drop(columns=["county_name"])

    # Drop if already present from a previous run
    if "overall_poverty_rate_avg" in df.columns:
        df = df.drop(columns=["overall_poverty_rate_avg"])

    df = df.merge(pov_avg[["county_short","overall_poverty_rate_avg"]],
                  on="county_short", how="left")

    n_filled = df["overall_poverty_rate_avg"].notna().sum()
    print(f"  Merged overall poverty rate: {n_filled} of {len(df)} counties")
    print(f"  Mean: {df['overall_poverty_rate_avg'].mean():.1f}%  "
          f"Range: {df['overall_poverty_rate_avg'].min():.1f}% - "
          f"{df['overall_poverty_rate_avg'].max():.1f}%")

    # Standardize
    df["overall_poverty_rate_z"] = (
        (df["overall_poverty_rate_avg"] - df["overall_poverty_rate_avg"].mean())
        / df["overall_poverty_rate_avg"].std()
    ).round(4)

    # Correlation with food insecurity
    from scipy import stats as _stats
    r, p = _stats.pearsonr(df["food_insecurity_rate"].dropna(),
                            df.loc[df["food_insecurity_rate"].notna(),
                                   "overall_poverty_rate_avg"])
    print(f"  Correlation with food insecurity rate: r={r:.3f}, p={p:.4f}")
else:
    print(f"  WARNING: {pov_path} not found.")
    print("  Run calfresh_01_load_data.py locally to generate poverty data.")
    print("  overall_poverty_rate_avg will be absent from county_features.csv")


# =============================================================================
# SAVE FINAL FEATURE DATASET
# =============================================================================
section("Saving feature dataset")

out_path = DATA + "county_features.csv"
df.to_csv(out_path, index=False)
print(f"  Saved: {out_path}")
print(f"  Shape: {df.shape}")
feature_z_cols = [c for c in df.columns if c.endswith("_z")]
# Print final feature summary
for col in feature_z_cols:
    # Raw column may have _avg suffix (e.g. overall_poverty_rate_avg)
    base = col.replace("_z", "")
    if base not in df.columns:
        base = base + "_avg"
    if base in df.columns:
        hi = df.nlargest(1, base)["county_short"].values[0]
        lo = df.nsmallest(1, base)["county_short"].values[0]
        print(f"    {col:<45}  highest={hi}, lowest={lo}")
    else:
        print(f"    {col:<45}  (raw column not found for ranking)")
        
print("""
  Next step: calfresh_03_analyze.py
  Model recommendation based on VIF:
    Drop features with VIF > 10 from the full model.
    Run stepwise comparison: start with strongest bivariate predictors,
    add others one at a time, track R² and coefficient stability.
    Run sensitivity: all 58 counties vs 54 (excluding small_county==1).
""")



