import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Safe printing (Windows console friendly)
def to_ascii(obj):
    if isinstance(obj, str):
        return obj.encode("ascii", "replace").decode("ascii")
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_ascii(x) for x in obj)
    if isinstance(obj, dict):
        return {to_ascii(k): to_ascii(v) for k, v in obj.items()}
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return to_ascii(obj.to_string())
    return obj
def p(label, obj):
    print(to_ascii(label))
    print(to_ascii(obj))

# Load fact and customer dimension
fact = pd.read_csv("output/fact_orders.csv")
dim_customer = pd.read_csv("output/dim_customer.csv")

p("=== Loaded tables ===", {"fact": fact.shape, "dim_customer": dim_customer.shape})

# Merge gender to fact
fact_l = fact.merge(dim_customer[["customer_key","gender"]], on="customer_key", how="left")
p("=== Null counts (gender after merge) ===", fact_l["gender"].isna().sum())

# Compute revenue and aggregate by gender
fact_l["revenue"] = fact_l["line_total"].fillna(0.0)
fact_l["gender"] = fact_l["gender"].fillna("Unknown")
gender_rev = fact_l.groupby("gender", dropna=False)["revenue"].sum().reset_index().sort_values("revenue", ascending=False)

p("=== Gender-wise Revenue (INR) ===", gender_rev)
p("Row count check:", {"rows": len(fact_l), "groups": len(gender_rev)})

# Donut chart
plt.figure(figsize=(6,6))
colors = sns.color_palette("pastel", n_colors=max(3, len(gender_rev)))
wedges, texts, autotexts = plt.pie(
    gender_rev["revenue"],
    labels=gender_rev["gender"],
    autopct=lambda pct: f"{pct:.1f}%",
    startangle=90,
    colors=colors,
    wedgeprops=dict(width=0.4, edgecolor="white")
)
plt.title("Gender-wise Revenue (Donut Chart)")
# Add center text with total
total_rev = gender_rev["revenue"].sum()
plt.text(0, 0, f"Total\nINR {total_rev:,.0f}", ha="center", va="center", fontsize=10)
plt.tight_layout()
out_path = "output/gender_revenue_donut.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()

p("=== Saved chart ===", out_path)