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

# Load fact and dimensions
fact = pd.read_csv("output/fact_orders.csv")
dim_product = pd.read_csv("output/dim_product.csv")
dim_payment = pd.read_csv("output/dim_payment_method.csv")
dim_status = pd.read_csv("output/dim_order_status.csv")
dim_date = pd.read_csv("output/dim_date.csv", parse_dates=["date"])
dim_geo = pd.read_csv("output/dim_geography.csv")

p("=== Loaded tables ===", {
    "fact": fact.shape,
    "dim_product": dim_product.shape,
    "dim_payment": dim_payment.shape,
    "dim_status": dim_status.shape,
    "dim_date": dim_date.shape,
    "dim_geo": dim_geo.shape
})

# Merge lookups for analytics labels
fact_l = fact.merge(dim_product[["product_key","product_id","product_name","category"]], on="product_key", how="left") \
             .merge(dim_payment, on="payment_method_key", how="left") \
             .merge(dim_status, on="order_status_key", how="left") \
             .merge(dim_date.rename(columns={"date_key":"order_date_key"}), on="order_date_key", how="left") \
             .merge(dim_geo, on="geography_key", how="left")

# Evidence: head and nulls after merge
p("\n=== Merged fact head(5) ===", fact_l.head(5))
p("\n=== Null counts (selected label cols) ===", fact_l[["category","product_name","payment_method","order_status","date","city","state","country"]].isna().sum())

# KPIs
fact_l["revenue"] = fact_l["line_total"].fillna(0.0)
fact_l["net_amount"] = fact_l["item_net_amount"].fillna(0.0)
fact_l["month"] = fact_l["date"].dt.to_period("M").astype(str)

kpis = {
    "rows": int(len(fact_l)),
    "orders_distinct": int(fact_l["order_id"].nunique()),
    "products_distinct": int(fact_l["product_id"].nunique()),
    "customers_distinct": int(fact_l["customer_key"].nunique()),
    "total_quantity": float(fact_l["quantity"].fillna(0).sum()),
    "total_revenue": float(fact_l["revenue"].sum()),
    "avg_order_value": float(fact_l.groupby("order_id")["revenue"].sum().mean()),
}
p("\n=== KPI Summary ===", kpis)

# 1) Revenue by Category
cat_rev = fact_l.groupby("category", dropna=False)["revenue"].sum().sort_values(ascending=False).reset_index()
p("\n=== Revenue by Category ===", cat_rev.head(10))

plt.figure(figsize=(10,6))
sns.barplot(data=cat_rev, x="revenue", y="category", color="#4C78A8")
plt.title("Revenue by Category")
plt.xlabel("Revenue (INR)")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig("output/revenue_by_category.png", dpi=150, bbox_inches="tight")
plt.close()

# 2) Monthly Revenue trend (by order_date)
month_rev = fact_l.dropna(subset=["month"]).groupby("month")["revenue"].sum().reset_index()
month_rev = month_rev.sort_values("month")
p("\n=== Monthly Revenue ===", month_rev)

plt.figure(figsize=(10,5))
sns.lineplot(data=month_rev, x="month", y="revenue", marker="o", color="#F58518")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month (YYYY-MM)")
plt.ylabel("Revenue (INR)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/monthly_revenue.png", dpi=150, bbox_inches="tight")
plt.close()

# 3) Top 10 Products by Revenue
prod_rev = fact_l.groupby(["product_id","product_name"], dropna=False)["revenue"].sum().reset_index()
prod_rev = prod_rev.sort_values("revenue", ascending=False).head(10)
p("\n=== Top 10 Products by Revenue ===", prod_rev)

plt.figure(figsize=(10,6))
sns.barplot(data=prod_rev, x="revenue", y="product_name", color="#54A24B")
plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue (INR)")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig("output/top_products_revenue.png", dpi=150, bbox_inches="tight")
plt.close()

# 4) Payment Method Mix (counts and revenue)
pm_mix_cnt = fact_l.groupby("payment_method")["order_id"].nunique().sort_values(ascending=False).reset_index().rename(columns={"order_id":"orders"})
pm_mix_rev = fact_l.groupby("payment_method")["revenue"].sum().sort_values(ascending=False).reset_index()
p("\n=== Payment Method - Orders by Method ===", pm_mix_cnt)
p("\n=== Payment Method - Revenue by Method ===", pm_mix_rev.head(10))

plt.figure(figsize=(10,5))
sns.barplot(data=pm_mix_rev, x="payment_method", y="revenue", color="#E45756")
plt.title("Revenue by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Revenue (INR)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/payment_method_revenue.png", dpi=150, bbox_inches="tight")
plt.close()

# 5) Order Status Distribution
status_cnt = fact_l["order_status"].value_counts(dropna=False).reset_index()
status_cnt.columns = ["order_status","count"]
p("\n=== Order Status Counts ===", status_cnt)

plt.figure(figsize=(8,5))
sns.barplot(data=status_cnt, x="order_status", y="count", color="#72B7B2")
plt.title("Order Status Distribution")
plt.xlabel("Order Status")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/order_status_counts.png", dpi=150, bbox_inches="tight")
plt.close()

# 6) Top 10 Cities by Revenue
city_rev = fact_l.groupby("city", dropna=False)["revenue"].sum().sort_values(ascending=False).reset_index().head(10)
p("\n=== Top Cities by Revenue ===", city_rev)

plt.figure(figsize=(10,5))
sns.barplot(data=city_rev, x="revenue", y="city", color="#B279A2")
plt.title("Top 10 Cities by Revenue")
plt.xlabel("Revenue (INR)")
plt.ylabel("City")
plt.tight_layout()
plt.savefig("output/top_cities_revenue.png", dpi=150, bbox_inches="tight")
plt.close()

# Final evidence: files created
created_files = [
    "output/revenue_by_category.png",
    "output/monthly_revenue.png",
    "output/top_products_revenue.png",
    "output/payment_method_revenue.png",
    "output/order_status_counts.png",
    "output/top_cities_revenue.png"
]
p("\n=== Charts saved ===", created_files)