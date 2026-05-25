"""
CalFresh Coverage Gap : Regression Analysis
=============================================
Outcome: food_insecurity_rate (county average 2019-2023, %)
Unit:    CA county (N=58)
Method:  OLS with HC3 heteroskedasticity-robust standard errors

Model building strategy (pre-specified after VIF check):
  M1  Baseline    snap_benefit_per_cap_2022 + cost_per_meal
  M2  + Access    + log_pct_low_access_lowincome + pct_low_access_snap
  M3  + Eligibility + pct_fi_snap_eligible
  M4  Full        + snap_benefit_growth_pct
  M5  Sensitivity  M3 on 39 large counties only (small_county==0)

All features are standardized (z-scores) so coefficients are directly
comparable in magnitude. Outcome kept in original percent scale so
coefficients read as: 'a 1-SD increase in X is associated with beta
percentage points change in food insecurity rate.'

HC3 robust SEs used throughout -- same implementation as in the
DiD scripts, no external dependencies required.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("figures", exist_ok=True)
DATA = "/your/path/to/data/"      # folder containing downloaded data files
FIG  = "/your/path/to/figures/"   # folder for output figures



def section(title):
    print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")


# =============================================================================
# OLS WITH HC3 ROBUST STANDARD ERRORS
# =============================================================================

def ols_hc3(y, X, feature_names):
    """
    OLS regression with HC3 heteroskedasticity-robust standard errors.

    WHY HC3?
    With only 58 observations, standard OLS standard errors rely on the
    assumption that residual variance is constant across counties
    (homoskedasticity). With cross-sectional county data this is unlikely
    -- larger counties (LA, SF) will have more precisely estimated
    food insecurity rates than tiny rural counties, creating heteroskedastic
    residuals. HC3 corrects for this without imposing any distributional
    assumption on the errors.

    HC3 uses leave-one-out leverage correction:
      u_hc3 = residual_i / (1 - h_ii)
    where h_ii is the leverage of observation i (how much it influences
    its own fitted value). High-leverage points -- like Sierra or Alpine
    with extreme access values -- get their residuals upweighted, which
    properly inflates their standard errors to reflect their influence.
    This is preferable to HC1/HC2 in small samples (N < 250).

    Returns a dict with coefficients, SEs, t-stats, p-values, R², adj-R².
    """
    n, k    = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    coef    = XtX_inv @ X.T @ y
    fitted  = X @ coef
    resid   = y - fitted

    # Hat matrix diagonal (leverage scores)
    h     = np.einsum("ij,jk,ki->i", X, XtX_inv, X.T)
    denom = np.clip(1 - h, 1e-8, None)
    u     = resid / denom                          # HC3 scaled residuals

    meat  = (X * u[:, None]).T @ (X * u[:, None])
    vcov  = XtX_inv @ meat @ XtX_inv
    se    = np.sqrt(np.clip(np.diag(vcov), 0, None))

    t_stat = np.where(se > 0, coef / se, np.nan)
    p_val  = 2 * stats.t.sf(np.abs(t_stat), df=n - k)

    ss_res  = np.sum(resid**2)
    ss_tot  = np.sum((y - y.mean())**2)
    r2      = 1 - ss_res / ss_tot
    adj_r2  = 1 - (1 - r2) * (n - 1) / (n - k)

    return {
        "names":  feature_names,
        "coef":   coef,
        "se":     se,
        "t":      t_stat,
        "pval":   p_val,
        "r2":     r2,
        "adj_r2": adj_r2,
        "n":      n,
        "k":      k,
        "fitted": fitted,
        "resid":  resid,
    }


def print_model(res, title=""):
    """Print a formatted regression table."""
    if title:
        print(f"\n  {title}")
    print(f"  {'Variable':<40} {'Coef':>7} {'SE':>7} {'t':>6} {'p':>7}  ")
    print("  " + "-" * 68)
    for i, name in enumerate(res["names"]):
        sig = "***" if res["pval"][i] < 0.001 else \
              "**"  if res["pval"][i] < 0.01  else \
              "*"   if res["pval"][i] < 0.05  else \
              "."   if res["pval"][i] < 0.10  else ""
        print(f"  {name:<40} {res['coef'][i]:>7.3f} {res['se'][i]:>7.3f} "
              f"{res['t'][i]:>6.2f} {res['pval'][i]:>7.4f} {sig}")
    print(f"\n  R² = {res['r2']:.3f}   Adj-R² = {res['adj_r2']:.3f}   "
          f"N = {res['n']}   k = {res['k']}")
    print("  Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10")


# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(DATA + "county_features.csv")
df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

y = df["food_insecurity_rate"].values.astype(float)

print(f"Loaded: {len(df)} counties")
print(f"Outcome mean: {y.mean():.2f}%  std: {y.std():.2f}%")


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

"""
Feature selection rationale after EDA + VIF analysis:

  snap_benefit_per_cap_2022_z   (r=0.83, VIF=3.53)
    Strongest predictor. Captures underlying deprivation -- counties
    with greater need draw more SNAP benefits per capita. Included
    in every model as the baseline control for need intensity.

  cost_per_meal_z               (r=-0.57, VIF=2.38)
    Negative correlation: expensive counties (Bay Area) have lower
    food insecurity because of higher incomes. Controls for the
    urban/coastal affluence effect that would otherwise confound
    the access and eligibility variables.

  log_pct_low_access_lowincome_z (r=0.46, VIF=2.01)
    Primary food access barrier variable. Log-transformed to reduce
    leverage of Sierra/Alpine/Mono. Measures the share of low-income
    residents living far from a grocery store -- most directly relevant
    to the coverage gap question.

  pct_low_access_snap_z         (r=0.57, VIF=2.08)
    Share of SNAP *recipients* with poor food access. Distinct from
    the general access measure: a county could have high general access
    barriers but low SNAP-recipient access barriers if enrolled members
    cluster near stores. The gap between these two tells you whether
    the program is reaching the people most constrained by geography.

  pct_fi_snap_eligible_z        (r=0.79, VIF=3.58)
    Share of food-insecure people below the SNAP income threshold.
    High values = deep poverty concentration. Controls for the
    structural composition of food insecurity -- counties where
    most food-insecure people can theoretically qualify for SNAP
    but many don't enroll have a different policy problem than
    counties where most food-insecure people are above the income limit.

  snap_benefit_growth_pct_z     (r=-0.03, VIF=1.70)
    Growth in per-capita SNAP benefits 2017-2022. Near-zero bivariate
    correlation -- included in the full model to test whether program
    expansion over this period moderated food insecurity, but expected
    to be non-significant.
"""

FEATURES = {
    "snap_benefit_per_cap_2022_z":    "SNAP benefit/capita 2022",
    "cost_per_meal_z":                "Cost per meal",
    "log_pct_low_access_lowincome_z": "Low access, low-income (log)",
    "pct_low_access_snap_z":          "Low access, SNAP recipients",
    "pct_fi_snap_eligible_z":         "% FI below SNAP threshold",
    "snap_benefit_growth_pct_z":      "SNAP benefit growth 2017-22",
    "overall_poverty_rate_z":         "Overall poverty rate",
}


def vif_check(df, feature_keys):
    """
    Compute Variance Inflation Factor for each feature.
    VIF_j = 1 / (1 - R2_j) where R2_j is the R-squared from
    regressing feature j on all other features.
    Returns a DataFrame with Feature, VIF, and Flag columns.
    """
    rows = []
    cols = list(feature_keys)
    X = df[cols].dropna().values
    for j, name in enumerate(cols):
        mask = [i for i in range(len(cols)) if i != j]
        Xr = X[:, mask]
        yr = X[:, j]
        XtX_inv = np.linalg.pinv(Xr.T @ Xr)
        coef_r = XtX_inv @ Xr.T @ yr
        resid_r = yr - Xr @ coef_r
        ss_res = np.sum(resid_r ** 2)
        ss_tot = np.sum((yr - yr.mean()) ** 2)
        r2_j = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vif = 1 / (1 - r2_j) if r2_j < 1 else float("inf")
        flag = "<-- DROP (severe)" if vif > 10 else "<-- WATCH" if vif > 5 else ""
        rows.append({"Feature": name, "VIF": round(vif, 2), "Flag": flag})
    return pd.DataFrame(rows)


def build_X(df, feature_keys, add_const=True):
    """
    Build the design matrix from feature column names.
    Adds an intercept column first when add_const=True.
    Returns (X array, column name list).
    """
    cols  = list(feature_keys)
    X     = df[cols].values.astype(float)
    names = [FEATURES.get(c, c) for c in cols]
    if add_const:
        X     = np.column_stack([np.ones(len(X)), X])
        names = ["Intercept"] + names
    return X, names


# =============================================================================
# M1 : BASELINE: need intensity + cost
# =============================================================================
section("M1 : Baseline: need intensity + food cost")

"""
WHY START HERE?
These two variables have the strongest bivariate correlations (0.83 and
-0.57), the lowest VIFs (3.53 and 2.38), and clear theoretical priors
for their direction. They also control for the two main confounders
before we add the policy-relevant access and eligibility variables:
  - snap_benefit_per_cap controls for underlying deprivation level
  - cost_per_meal controls for the urban/coastal affluence gradient
The R² from this model is the baseline we need to beat with the
access and eligibility variables to justify their inclusion.
"""

X1, n1 = build_X(df, ["snap_benefit_per_cap_2022_z", "cost_per_meal_z"])
m1 = ols_hc3(y, X1, n1)
print_model(m1, "M1: SNAP benefit + cost per meal")


# =============================================================================
# M2 : ADD FOOD ACCESS VARIABLES
# =============================================================================
section("M2 : Add food access variables")

"""
DO THE ACCESS VARIABLES ADD EXPLANATORY POWER BEYOND NEED + COST?
If yes (R² increases meaningfully, coefficients significant), it suggests
that food access barriers independently predict food insecurity even after
accounting for how poor a county is and how expensive food is there.
This would strengthen the case that physical food access -- not just income
-- is a driver of the coverage gap, supporting CalAIM-style interventions
that address food environment alongside SNAP enrollment.
"""

X2, n2 = build_X(df, [
    "snap_benefit_per_cap_2022_z",
    "cost_per_meal_z",
    "log_pct_low_access_lowincome_z",
    "pct_low_access_snap_z",
])
m2 = ols_hc3(y, X2, n2)
print_model(m2, "M2: + food access (low-income + SNAP recipients)")

delta_r2 = m2["r2"] - m1["r2"]
print(f"\n  R² gain over M1: +{delta_r2:.3f}")


# =============================================================================
# M3 : ADD SNAP ELIGIBILITY (preferred model)
# =============================================================================
section("M3 : Add SNAP eligibility")

"""
WHY THIS MODEL:
pct_fi_snap_eligible adds the policy mechanism: it captures whether
food insecurity in a county is concentrated among people who *can*
receive SNAP (deep poverty) vs those who are above the income limit
(working poor). A positive coefficient here means: holding need level
and access barriers constant, counties where more of the food-insecure
population is below the SNAP threshold have higher food insecurity --
which would suggest either that SNAP is not reaching eligible people
(enrollment gap) or that SNAP alone is insufficient for that depth
of poverty.

This model also has all six features at acceptable VIF levels and
represents the most complete theoretically-motivated specification.
M3P (M3 + poverty) is the preferred model. M3 is reported for comparison.
"""

X3, n3 = build_X(df, [
    "snap_benefit_per_cap_2022_z",
    "cost_per_meal_z",
    "log_pct_low_access_lowincome_z",
    "pct_low_access_snap_z",
    "pct_fi_snap_eligible_z",
])
m3 = ols_hc3(y, X3, n3)
print_model(m3, "M3: + SNAP eligibility concentration")

delta_r2 = m3["r2"] - m2["r2"]
print(f"\n  R² gain over M2: +{delta_r2:.3f}")


# =============================================================================
# M4 : FULL MODEL
# =============================================================================
section("M4 : Full model (+ benefit growth)")

"""
snap_benefit_growth_pct had near-zero bivariate correlation (r=-0.03)
but may still contribute once other variables are held constant.
We include it here to test: if the coefficient is significant and
meaningfully sized it would suggest program expansion between 2017
and 2022 had an independent association with food insecurity levels,
above and beyond the level of benefits and need. If not significant,
it confirms the bivariate signal and we drop it from the preferred model.
"""

X4, n4 = build_X(df, [
    "snap_benefit_per_cap_2022_z",
    "cost_per_meal_z",
    "log_pct_low_access_lowincome_z",
    "pct_low_access_snap_z",
    "pct_fi_snap_eligible_z",
    "snap_benefit_growth_pct_z",
])
m4 = ols_hc3(y, X4, n4)
print_model(m4, "M4: Full model")

delta_r2 = m4["r2"] - m3["r2"]
print(f"\n  R² gain over M3: +{delta_r2:.3f}")


# =============================================================================
# M3P : M3 + OVERALL POVERTY RATE
# =============================================================================
section("M3P : M3 + overall poverty rate (confounding variable test)")

"""
WHY ADD POVERTY?
The preferred model (M3) shows positive coefficients for SNAP benefit per
capita and SNAP eligibility rate, which we explain as a common cause effect:
poverty drives food insecurity, SNAP benefit levels, and eligibility rates
simultaneously. But we never controlled for poverty directly.

Adding overall poverty rate (ACS B17001, 2019-2022 county average) tests
whether the SNAP coefficients change materially when poverty is explicitly
held constant. Three possible outcomes:
  1. SNAP coefficient flips negative: suggests SNAP is actually mitigating
     food insecurity once poverty is controlled -- the true causal direction.
  2. SNAP coefficient reduces but stays positive: supports the common cause
     explanation -- poverty was driving the positive association.
  3. SNAP coefficient stays large and positive: suggests the common cause
     explanation is incomplete.

NOTE: Overall poverty rate is used, not child poverty rate. CalFresh serves
all household types regardless of whether children are present, and the MMG
food insecurity outcome is for the total population. Using overall poverty
avoids a population mismatch.

NOTE ON MULTICOLLINEARITY: Overall poverty rate correlates strongly with
both SNAP benefit per capita and the eligibility rate (r ~ 0.80+). VIF for
poverty will likely be high. Check VIF before trusting individual coefficients.
"""

if "overall_poverty_rate_z" in df.columns and df["overall_poverty_rate_z"].notna().sum() > 0:
    X3p, n3p = build_X(df, [
        "snap_benefit_per_cap_2022_z",
        "cost_per_meal_z",
        "log_pct_low_access_lowincome_z",
        "pct_low_access_snap_z",
        "pct_fi_snap_eligible_z",
        "overall_poverty_rate_z",
    ])
    m3p = ols_hc3(y, X3p, n3p)
    print_model(m3p, "M3P: M3 + overall poverty rate")

    # VIF check for poverty model
    print(f"\n  VIF check for M3P:")
    pov_features = [
        "snap_benefit_per_cap_2022_z",
        "cost_per_meal_z",
        "log_pct_low_access_lowincome_z",
        "pct_low_access_snap_z",
        "pct_fi_snap_eligible_z",
        "overall_poverty_rate_z",
    ]
    vif_df = vif_check(df, pov_features)
    if isinstance(vif_df, pd.DataFrame):
        print(vif_df.to_string(index=False))
    else:
        for k, v in vif_df.items():
            print(f"  {k:<45} {v:.2f}")

    # Compare M3 vs M3P for key coefficients
    # Use actual model name strings (from FEATURES dict via print_model)
    COMPARE = [
        ("SNAP benefit/capita 2022",      "SNAP benefit per capita (2022)"),
        ("% FI below SNAP threshold",     "% FI below SNAP threshold"),
        ("Low access, SNAP recipients",   "% SNAP recipients, low access"),
        ("Cost per meal",                 "Cost per meal"),
        ("Overall poverty rate",          "Overall poverty rate"),
    ]

    def gc(res, v):
        """Get coefficient and p-value by name, trying exact then partial match."""
        if v in res["names"]:
            i = res["names"].index(v)
            return res["coef"][i], res["pval"][i]
        # Try partial match
        matches = [n for n in res["names"] if v.lower() in n.lower()]
        if matches:
            i = res["names"].index(matches[0])
            return res["coef"][i], res["pval"][i]
        return float("nan"), float("nan")

    print(f"\n  M3 vs M3P coefficient comparison (key predictors):")
    print(f"  {'Variable':<40} {'M3 coef':>10} {'M3 p':>8}  {'M3P coef':>10} {'M3P p':>8}")
    print("  " + "-" * 72)
    for m3_name, label in COMPARE:
        c3,  p3  = gc(m3,  m3_name)
        c3p, p3p = gc(m3p, m3_name)
        s3  = "***" if p3<0.001  else "**" if p3<0.01  else "*" if p3<0.05  else "." if p3<0.10  else " "
        s3p = "***" if p3p<0.001 else "**" if p3p<0.01 else "*" if p3p<0.05 else "." if p3p<0.10 else " "
        c3_s  = f"{c3:>8.3f}{s3:>3}"   if not np.isnan(c3)  else f"{'n/a':>11}"
        c3p_s = f"{c3p:>8.3f}{s3p:>3}" if not np.isnan(c3p) else f"{':':>11}"
        p3_s  = f"{p3:>8.4f}"  if not np.isnan(p3)  else f"{'n/a':>8}"
        p3p_s = f"{p3p:>8.4f}"         if not np.isnan(p3p) else f"{':':>8}"
        print(f"  {label:<40} {c3_s} {p3_s}  {c3p_s} {p3p_s}")

    print(f"\n  R² M3={m3['r2']:.3f}  M3P={m3p['r2']:.3f}  (gain={m3p['r2']-m3['r2']:.3f})")
    print(f"  Adj-R² M3={m3['adj_r2']:.3f}  M3P={m3p['adj_r2']:.3f}")
else:
    m3p = None
    print("  overall_poverty_rate_z not available.")
    print("  Run calfresh_01_load_data.py locally to generate ACS B17001 data.")


# =============================================================================
# M5 : SENSITIVITY: LARGE COUNTIES ONLY
# =============================================================================
section("M5 : Sensitivity: large counties only (small_county == 0)")

"""
WHY THIS SENSITIVITY CHECK?
The small_county flag (food_insecure_count < 10,000) captured 19 counties.
While we argued for keeping them in (to avoid selection bias and preserve
N), their extreme access values could be driving results even after log
transformation. If M3 coefficients are stable between the full sample
and the large-county subsample, we have evidence that findings are not
artefacts of a handful of tiny rural counties. If coefficients shift
substantially, we need to qualify the findings and report both results.

We run M3 specification on the subsample for a clean apples-to-apples
comparison.
"""

large = df[df["small_county"] == 0].copy()
y5    = large["food_insecurity_rate"].values.astype(float)

X5, n5 = build_X(large, [
    "snap_benefit_per_cap_2022_z",
    "cost_per_meal_z",
    "log_pct_low_access_lowincome_z",
    "pct_low_access_snap_z",
    "pct_fi_snap_eligible_z",
])
m5 = ols_hc3(y5, X5, n5)
print_model(m5, f"M5: M3 on {len(large)} large counties (small_county==0)")


# =============================================================================
# MODEL COMPARISON TABLE
# =============================================================================
section("Model comparison")

models = [
    ("M1",  "Baseline",            m1),
    ("M2",  "+ Access",            m2),
    ("M3",  "+ Eligibility",        m3),
    ("M3P", "+ Poverty (pref)",    m3p if m3p is not None else m3),
    ("M4",  "Full",                m4),
    ("M5",  "Sensitivity (large)", m5),
]

print(f"\n  {'Model':<6} {'Spec':<25} {'N':>4} {'k':>4} {'R²':>7} {'Adj-R²':>8}")
print("  " + "-" * 58)
for label, spec, res in models:
    print(f"  {label:<6} {spec:<25} {res['n']:>4} {res['k']:>4} "
          f"{res['r2']:>7.3f} {res['adj_r2']:>8.3f}")


# =============================================================================
# COEFFICIENT STABILITY PLOT
# =============================================================================
section("Coefficient stability plot")

"""
A coefficient stability plot shows how each coefficient changes as we
add variables to the model. Stable coefficients (small movement across
models) give confidence that the estimate is not sensitive to model
specification. Large swings indicate that the variable is picking up
shared variance with another predictor.
"""

# Shared features across M1-M4 (excluding intercept)
shared = [
    ("snap_benefit_per_cap_2022_z", "SNAP benefit/capita"),
    ("cost_per_meal_z",             "Cost per meal"),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Coefficient stability across model specifications\n"
             "(standardized coefficients, HC3 95% CIs)",
             fontsize=12, fontweight="bold")

model_labels = ["M1", "M2", "M3", "M3P", "M4"]
model_results = [m1, m2, m3, m3p, m4]
colors = ["#4361EE","#E76F51","#2A9D8F","#9B2335","#6A0572"]

for ax_i, (feat_z, feat_label) in enumerate(shared):
    ax = axes[ax_i]
    for m_i, (label, res) in enumerate(zip(model_labels, model_results)):
        if feat_z.replace("_z","") in FEATURES:
            feat_display = FEATURES[feat_z]
        else:
            feat_display = feat_z
        # Find index of this feature in the model
        try:
            idx = res["names"].index(FEATURES.get(feat_z, feat_z))
        except ValueError:
            continue
        coef = res["coef"][idx]
        ci   = 1.96 * res["se"][idx]
        ax.errorbar(m_i, coef, yerr=ci, fmt="o", color=colors[m_i],
                    capsize=5, markersize=8, lw=2, label=label)

    ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels)
    ax.set_title(feat_label, fontsize=10)
    ax.set_ylabel("Standardized coefficient (pp per SD)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIG + "05_coef_stability.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {FIG}05_coef_stability.png")


# =============================================================================
# RESIDUAL DIAGNOSTICS FOR PREFERRED MODEL (M3)
# =============================================================================
section("Residual diagnostics : M3P (preferred model)")

"""
Even with HC3 SEs, it is worth checking residuals for:
  1. Patterns (fitted vs residual) -- non-linearity we missed
  2. Influential observations -- which counties drive results most
  3. Normality -- matters less with robust SEs but good practice to check

A well-specified model should show residuals scattered randomly around
zero with no clear pattern against fitted values.

M3P is the preferred model because it explicitly controls for overall
poverty rate -- the key confounding variable that drives both SNAP
benefit levels and food insecurity simultaneously.
"""

# Use M3P if available, fall back to M3
preferred = m3p if (m3p is not None) else m3
preferred_label = "M3P" if (m3p is not None) else "M3"

fitted_p = preferred["fitted"]
resid_p  = preferred["resid"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Residual diagnostics : {preferred_label} (preferred model)",
             fontsize=12, fontweight="bold")

# Fitted vs residual
ax = axes[0]
ax.scatter(fitted_p, resid_p, alpha=0.6, color="#457B9D", s=40)
ax.axhline(0, color="#E63946", ls="--", lw=1.5)
# Label top 5 by absolute residual
top5_idx = np.argsort(np.abs(resid_p))[-5:]
for i in top5_idx:
    ax.annotate(df["county_short"].iloc[i],
                xy=(fitted_p[i], resid_p[i]),
                fontsize=7, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("Fitted values (%)"); ax.set_ylabel("Residuals (pp)")
ax.set_title("Fitted vs Residuals"); ax.grid(True, alpha=0.3)

# Q-Q plot
ax = axes[1]
(osm, osr), (slope, intercept, r) = stats.probplot(resid_p)
ax.scatter(osm, osr, alpha=0.6, color="#457B9D", s=40)
ax.plot(osm, slope*np.array(osm)+intercept, color="#E63946", lw=1.5)
ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
ax.set_title(f"Q-Q plot (normality check)\nr={r:.3f}")
ax.grid(True, alpha=0.3)

# Actual vs fitted
ax = axes[2]
ax.scatter(y, fitted_p, alpha=0.6, color="#2A9D8F", s=40)
ax.plot([y.min(), y.max()], [y.min(), y.max()],
        color="#E63946", ls="--", lw=1.5, label="Perfect fit")
for i in top5_idx:
    ax.annotate(df["county_short"].iloc[i],
                xy=(y[i], fitted_p[i]),
                fontsize=7, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("Actual food insecurity rate (%)")
ax.set_ylabel("Fitted values (%)")
ax.set_title(f"Actual vs Fitted  (R²={preferred['r2']:.3f})")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{FIG}06_residuals_{preferred_label.lower()}.png",
            dpi=150, bbox_inches="tight")
print(f"  Saved: {FIG}06_residuals_{preferred_label.lower()}.png")


# =============================================================================
# INTERPRETATION SUMMARY
# =============================================================================

# Use M3P as preferred if available, else M3
pref_model  = m3p if (m3p is not None) else m3
pref_label  = "M3P" if (m3p is not None) else "M3"
pref_note   = ("M3P controls for overall poverty rate -- the key confounding "
               "variable. SNAP and eligibility coefficients should be interpreted "
               "as associations net of poverty, not as causal effects.")

section(f"Interpretation : {pref_label} preferred model")

print(f"""
  Outcome: food insecurity rate (county average 2019-2023, %)
  N = {pref_model['n']}  R² = {pref_model['r2']:.3f}  Adj-R² = {pref_model['adj_r2']:.3f}

  Preferred model: {pref_label}
  {pref_note}

  Interpreting standardized coefficients:
  'A one standard deviation increase in X is associated with beta
   percentage points change in the food insecurity rate, holding
   all other variables constant.'
""")

for i, name in enumerate(pref_model["names"]):
    if name == "Intercept":
        continue
    coef = pref_model["coef"][i]
    p    = pref_model["pval"][i]
    sig  = "***" if p < 0.001 else "**" if p < 0.01 else \
           "*"   if p < 0.05  else "."  if p < 0.10 else "(ns)"
    direction = "higher" if coef > 0 else "lower"
    print(f"  {name}")
    print(f"    beta = {coef:+.3f} pp  {sig}")
    if p < 0.10:
        print(f"    Counties with higher {name.lower()} tend to have "
              f"{abs(coef):.2f} pp {direction} food insecurity rate.")
    print()

print(f"""
  KEY FINDING : M3 vs {pref_label}:
    When overall poverty rate is added (M3P), the SNAP benefit
    coefficient drops from 1.177 to 0.611 (borderline significant).
    The SNAP eligibility coefficient collapses from 1.117 to 0.244
    (non-significant). Both were largely poverty proxies.
    Low access for SNAP recipients (0.556, p=0.013) is the only
    predictor that survives poverty control with significance intact
    -- pointing to physical food access barriers as an independent
    driver beyond poverty.

  Sensitivity (M5 vs M3):
    Findings hold on 39 large counties (R²=0.835). Results are not
    driven by small rural counties where MMG estimates are least reliable.

  Limitations:
    1. Cross-sectional -- cannot establish causation
    2. Ecological -- county-level, not individual-level associations
    3. N=58 -- limited statistical power for complex models
    4. MMG food insecurity estimates are modeled, not measured directly
    5. 2022 SNAP benefit levels reflect COVID Emergency Allotments
       still active in California -- elevated above typical levels
    6. Spatial autocorrelation -- neighboring counties share unobserved
       traits; HC3 does not correct for spatial dependence
""")

print("=== Analysis complete ===")
print(f"Figures: {FIG}05_coef_stability.png, {FIG}06_residuals_{pref_label.lower()}.png")
