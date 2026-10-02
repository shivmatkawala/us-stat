# imports:

import os
import numpy as np
import pandas as pd
from scipy import stats

# Load Data:
CSV= "ab_test_data.csv"

if not os.path.exists(CSV):
    rng = np.random.default_rng(42)
    n = 5000
    group = np.array(["A_Green"] * n + ["B_Orange"] * n)

    rng.shuffle(group)

    p = np.where(group =="A_Green", 0.05, 0.07)
    df = pd.DataFrame({
        "visitor_id": np.arange(1, 2*n+1),
        "group": group,
        "device": rng.choice(["Mobile", "Desktop"],2+n, p=[0.6, 0.4]),
        "clicked": (rng.random(2*n)<p).astype(int)
    })

    df.to_csv(CSV, index=False)
df = pd.read_csv(CSV)
print(df.head(), "\n")


# Check Data Quality:
assert df["clicked"].isin([0, 1]).all, "cliecked must be 0 or 1"

assert df["visitor_id"].is_unique, "duplicate visitors found"
print(df.isna().sum(), "\n")


# Check Split is Balanced:

counts = df["group"].value_counts()
srm_p = stats.chisquare(counts.values).pvalue

print(f"Group Size: {counts}")
print(f"SRM check p-value = {srm_p:.3f}"
      f"({'OK, Split looks fair' if srm_p > 0.01 else 'WARNING: Split is Unbalanced'})\n")


# Summarize Each Group:

summary = df.groupby("group")["clicked"].agg(visitors="count", clicks="sum", rate="mean")
print(summary, "\n")

a =df[df["group"] == "A_Green"]["clicked"]
b =df[df["group"] == "B_Orange"]["clicked"]
n_a, n_b = len(a), len(b)
x_a, x_b = a.sum(), b.sum()
p_a, p_b = x_a/n_a, x_b/n_b

# CALCULATE LIFT
abs_lift = p_b - p_a
rel_lift = abs_lift /p_a
print(f"Absolute Lif: {abs_lift*100:.2f}%")
print(f"Relative Lift: {rel_lift*100:.1f}%")


# State the Hypothesis:

ALPHA = 0.05  # tolerance for false alarm

print(f"H0 = No Difference, H1 = There is a Difference, alpha = {ALPHA}\n")

# Two- Proportion Z-Test:

p_pool = (x_a +x_b) / (n_a+n_b)
se_pool = np.sqrt(p_pool * (1- p_pool) * (1/n_a + 1/n_b))
z = (p_b - p_a) / se_pool
p_value = 2 * (1-stats.norm.cdf(abs(z)))

print(f"z-score: {z:.3f}")
print(f"p_value: {p_value:.6f}")

if p_value < ALPHA:
    print(f"p < {ALPHA}: diff is statistically significant -> reject H0\n")
else:
    print(f"p > {ALPHA}: not enough evidences of a difference -> keep H0 \n")
    

