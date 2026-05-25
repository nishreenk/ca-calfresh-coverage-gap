"""
CalFresh — Poverty rate vs food insecurity rate scatter plot
============================================================
Requires:
  data/county_features.csv      -- from calfresh_03_feature_engineering.py
  data/b17001_ca_avg.csv        -- from calfresh_01_load_data.py (ACS B17001)

Output:
  figures/calfresh_poverty_vs_fi.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import os

DATA = "/your/path/to/data/"      # folder containing downloaded data files
FIG  = "/your/path/to/figures/"   # folder for output figures
os.makedirs(FIG, exist_ok=True)

# Load features
features = pd.read_csv(DATA + "county_features.csv")
features["county_short"] = (features["county_name"]
                             .str.replace(" County, California", "", regex=False)
                             .str.strip())

# Load overall poverty rate
pov = pd.read_csv(DATA + "b17001_ca_avg.csv")
pov["county_short"] = (pov["county_name"]
                        .str.replace(" County, California", "", regex=False)
                        .str.strip())
pov = pov[["county_short", "overall_poverty_rate_avg"]]


#debug
#df = features.merge(pov, on="county_short", how="left")
#print("Columns after merge:", [c for c in df.columns if 'poverty' in c])
#print("Features sample:", features['county_short'].head(3).tolist())
#print("Pov sample:", pov['county_short'].head(3).tolist())

# Merge
#df = features.merge(pov, on="county_short", how="left")
#df = df.dropna(subset=["overall_poverty_rate_avg", "food_insecurity_rate"])


# overall_poverty_rate_avg already in county_features.csv
# no merge needed
df = features.copy()
df = df.dropna(subset=["overall_poverty_rate_avg", "food_insecurity_rate"])

print(f"N counties: {len(df)}")

# Correlation
r, p = stats.pearsonr(df["overall_poverty_rate_avg"], df["food_insecurity_rate"])
print(f"r={r:.3f}, p={p:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(df["overall_poverty_rate_avg"], df["food_insecurity_rate"],
           color="#457B9D", alpha=0.6, s=55, edgecolors="white")

# Label notable counties
highlight = (df.nlargest(8, "food_insecurity_rate")["county_short"].tolist()
           + df.nlargest(5, "overall_poverty_rate_avg")["county_short"].tolist())
highlight = list(set(highlight))

for _, row in df[df["county_short"].isin(highlight)].iterrows():
    ax.annotate(row["county_short"],
                (row["overall_poverty_rate_avg"], row["food_insecurity_rate"]),
                fontsize=8, xytext=(5, 4), textcoords="offset points",
                color="#333333")

# Regression line
m, b, *_ = stats.linregress(df["overall_poverty_rate_avg"],
                              df["food_insecurity_rate"])
xs = np.linspace(df["overall_poverty_rate_avg"].min(),
                  df["overall_poverty_rate_avg"].max(), 100)
ax.plot(xs, m*xs + b, color="#E63946", lw=1.5,
        label=f"r={r:.2f}, p<0.001")

ax.set_xlabel("Overall poverty rate (%, 2019-2022 avg)", fontsize=11)
ax.set_ylabel("Food insecurity rate (%, 2019-2023 avg)", fontsize=11)
ax.set_title("Poverty rate vs food insecurity rate across California counties\n"
             "Each dot is one county  |  Source: ACS B17001 + Feeding America MMG",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = FIG + "calfresh_poverty_vs_fi.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
