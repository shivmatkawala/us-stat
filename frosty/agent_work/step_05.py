import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Safe printing for Windows consoles
def to_ascii(obj):
    if isinstance(obj, str):
        return obj.encode("ascii", "replace").decode("ascii")
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_ascii(x) for x in obj.items())
    if isinstance(obj, dict):
        return {to_ascii(k): to_ascii(v) for k, v in obj.items()}
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return to_ascii(obj.to_string())
    return obj
def p(label, obj):
    print(to_ascii(label))
    print(to_ascii(obj))

# Load cleaned data and dimensions
clean_path = os.path.join("output", "ecommerce_orders_clean.csv")
df = pd.read_csv(clean_path, parse_dates=["order_date","delivery_date"])
p("=== Loaded cleaned dataset ===", f"path={clean_path} shape={df.shape}")

dim_date = pd.read_csv("output/dim_date.csv", parse_dates=["date"])
dim_customer = pd.read_csv("output/dim_customer.csv")
dim_product = pd.read_csv("output/dim_product.csv")
dim_geography = pd.read_csv("output/dim_geography.csv")
dim_payment_method = pd.read_csv("output/dim_payment_method.csv")
dim_order_status = pd.read_csv("output/dim_order_status.csv")
dim_acq = pd.read_csv("output/dim_acquisition_channel.csv")

# Build lookup maps
date_key_map = dict(zip(dim_date["date"].dt.strftime("%Y-%m-%d"), dim_date["date_key"]))
cust_key_map = dict(zip(dim_customer["customer_id"], dim_customer["customer_key"]))
prod_key_map = dict(zip(dim_product["product_id"], dim_product["product_key"]))
geo_key_map = { (r["city"], r["state"], r["country"]): r["geography_key"] for _, r in dim_geography.iterrows() }
pm_key_map = dict(zip(dim_payment_method["payment_method"], dim_payment_method["payment_method_key"]))
status_key_map = dict(zip(dim_order_status["order_status"], dim_order_status["order_status_key"]))
acq_key_map = dict(zip(dim_acq["acquisition_channel"], dim_acq["acquisition_channel_key"]))

# 1) Ensure grain: (order_id, product_id)
before = len(df)
df = df.drop_duplicates(subset=["order_id","product_id"], keep="first")
after = len(df)
p("\n=== Grain enforcement (order_id, product_id) ===", f"rows before={before} after={after} duplicates_removed={before-after}")

# 2) Create foreign keys
# Dates: format to YYYY-MM-DD string then map
df["order_date_key"] = df["order_date"].dt.strftime("%Y-%m-%d").map(date_key_map)
df["delivery_date_key"] = df["delivery_date"].dt.strftime("%Y-%m-%d").map(date_key_map)

df["customer_key"] = df["customer_id"].map(cust_key_map)
df["product_key"] = df["product_id"].map(prod_key_map)
df["geography_key"] = df.apply(lambda r: geo_key_map.get((r["city"], r["state"], r["country"])), axis=1)
df["payment_method_key"] = df["payment_method"].map(pm_key_map)
df["order_status_key"] = df["order_status"].map(status_key_map)
df["acquisition_channel_key"] = df["acquisition_channel"].map(acq_key_map)

# 3) Measures
# Treat missing quantity or unit_price as 0 for subtotal computation; discount_pct NaN -> 0
qty = df["quantity"].fillna(0)
price = df["unit_price"].fillna(0.0)
disc_pct = df["discount_pct"].fillna(0.0)
subtot = qty * price
disc_amt = subtot * (disc_pct / 100.0)
net_amt = subtot - disc_amt
line_total = net_amt + df["shipping_cost"].fillna(0)

df["item_subtotal"] = subtot.round(2)
df["discount_amount"] = disc_amt.round(2)
df["item_net_amount"] = net_amt.round(2)
df["line_total"] = line_total.round(2)

# 4) Fact table selection
fact_cols = [
    "order_id",
    "product_id",
    "order_date_key",
    "delivery_date_key",
    "customer_key",
    "product_key",
    "geography_key",
    "payment_method_key",
    "order_status_key",
    "acquisition_channel_key",
    "quantity",
    "unit_price",
    "discount_pct",
    "shipping_cost",
    "item_subtotal",
    "discount_amount",
    "item_net_amount",
    "line_total",
    "rating",
]
fact = df[fact_cols].copy()

# 5) Integrity checks
p("\n=== Fact head() ===", fact.head())

# Key nulls
key_nulls = {
    "order_date_key_nulls": int(fact["order_date_key"].isna().sum()),
    "delivery_date_key_nulls": int(fact["delivery_date_key"].isna().sum()),  # expected due to missing delivery_date
    "customer_key_nulls": int(fact["customer_key"].isna().sum()),
    "product_key_nulls": int(fact["product_key"].isna().sum()),
    "geography_key_nulls": int(fact["geography_key"].isna().sum()),
    "payment_method_key_nulls": int(fact["payment_method_key"].isna().sum()),
    "order_status_key_nulls": int(fact["order_status_key"].isna().sum()),
    "acquisition_channel_key_nulls": int(fact["acquisition_channel_key"].isna().sum()),
}
p("=== Key null counts ===", key_nulls)

# Row counts and uniqueness of grain
p("=== Fact shape and grain uniqueness ===", {"rows": len(fact), "unique (order_id, product_id)": fact[["order_id","product_id"]].drop_duplicates().shape[0]})

# Summary measures
summary = {
    "sum_quantity": float(fact["quantity"].fillna(0).sum()),
    "sum_shipping_cost": float(fact["shipping_cost"].fillna(0).sum()),
    "sum_item_subtotal": float(fact["item_subtotal"].sum()),
    "sum_discount_amount": float(fact["discount_amount"].sum()),
    "sum_item_net_amount": float(fact["item_net_amount"].sum()),
    "sum_line_total": float(fact["line_total"].sum()),
    "avg_rating_non_null": float(fact["rating"].dropna().mean()) if fact["rating"].dropna().size > 0 else np.nan,
}
p("=== Measure summaries ===", summary)

# 6) Save fact
fact_path = os.path.join("output", "fact_orders.csv")
fact.to_csv(fact_path, index=False)
p("=== Saved fact table ===", fact_path)