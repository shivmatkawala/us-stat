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
        return type(obj)(to_ascii(x) for x in obj)
    if isinstance(obj, dict):
        return {to_ascii(k): to_ascii(v) for k, v in obj.items()}
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return to_ascii(obj.to_string())
    return obj
def p(label, obj):
    print(to_ascii(label))
    print(to_ascii(obj))

# Load cleaned data
clean_path = os.path.join("output", "ecommerce_orders_clean.csv")
df = pd.read_csv(clean_path, parse_dates=["order_date","delivery_date"])
p("=== Loaded cleaned dataset ===", f"path={clean_path} shape={df.shape}")

# Helper: create surrogate keys by stable enumeration
def make_surrogate_key(series_unique_sorted, start=1):
    # Returns dict {value: surrogate_key}
    return {val: i for i, val in enumerate(series_unique_sorted, start=start)}

# --------------------
# dim_date (union of order_date and delivery_date)
# --------------------
date_vals = pd.concat([df["order_date"], df["delivery_date"]], ignore_index=True).dropna().drop_duplicates().sort_values()
dim_date = pd.DataFrame({"date": date_vals})
dim_date["date_key"] = dim_date["date"].dt.strftime("%Y%m%d").astype(int)
dim_date["year"] = dim_date["date"].dt.year
dim_date["month"] = dim_date["date"].dt.month
dim_date["day"] = dim_date["date"].dt.day
dim_date["day_name"] = dim_date["date"].dt.day_name()

# Evidence
p("\n=== dim_date ===", f"rows={len(dim_date)}")
p("head():", dim_date.head())
p("null counts:", dim_date.isna().sum())

# Save
dim_date_out = dim_date[["date_key","date","year","month","day","day_name"]].copy()
dim_date_out.to_csv("output/dim_date.csv", index=False)

# --------------------
# dim_customer
# --------------------
cust_cols = ["customer_id","customer_name","email","phone_normalized","gender","customer_age","city","state","country","acquisition_channel"]
dim_customer_raw = df[cust_cols].copy()
dim_customer_raw = dim_customer_raw.drop_duplicates().sort_values(["customer_id"])
# Surrogate key: customer_key by enumeration of unique customer_id
cust_ids_sorted = dim_customer_raw["customer_id"].dropna().drop_duplicates().sort_values()
cust_key_map = make_surrogate_key(cust_ids_sorted)
dim_customer_raw["customer_key"] = dim_customer_raw["customer_id"].map(cust_key_map)

# Evidence
p("\n=== dim_customer raw ===", f"rows={len(dim_customer_raw)} unique customer_id={cust_ids_sorted.size}")
p("head():", dim_customer_raw.head())
p("null counts:", dim_customer_raw.isna().sum())

# Save
dim_customer = dim_customer_raw[["customer_key","customer_id","customer_name","email","phone_normalized","gender","customer_age","city","state","country","acquisition_channel"]].copy()
dim_customer.to_csv("output/dim_customer.csv", index=False)

# --------------------
# dim_product
# --------------------
prod_cols = ["product_id","product_name","category"]
dim_product_raw = df[prod_cols].drop_duplicates().sort_values(["product_id"])
prod_ids_sorted = dim_product_raw["product_id"].dropna().drop_duplicates().sort_values()
prod_key_map = make_surrogate_key(prod_ids_sorted)
dim_product_raw["product_key"] = dim_product_raw["product_id"].map(prod_key_map)

# Evidence
p("\n=== dim_product raw ===", f"rows={len(dim_product_raw)} unique product_id={prod_ids_sorted.size}")
p("head():", dim_product_raw.head())
p("null counts:", dim_product_raw.isna().sum())

# Save
dim_product = dim_product_raw[["product_key","product_id","product_name","category"]].copy()
dim_product.to_csv("output/dim_product.csv", index=False)

# --------------------
# dim_geography (unique combinations of city/state/country)
# --------------------
geo_cols = ["city","state","country"]
dim_geo_raw = df[geo_cols].drop_duplicates().sort_values(geo_cols)
# Surrogate key via enumeration of tuple
geo_tuples = dim_geo_raw.apply(lambda r: (r["city"], r["state"], r["country"]), axis=1)
geo_key_map = make_surrogate_key(geo_tuples)
dim_geo_raw["geography_key"] = geo_tuples.map(geo_key_map)

# Evidence
p("\n=== dim_geography raw ===", f"rows={len(dim_geo_raw)}")
p("head():", dim_geo_raw.head())
p("null counts:", dim_geo_raw.isna().sum())

# Save
dim_geography = dim_geo_raw[["geography_key","city","state","country"]].copy()
dim_geography.to_csv("output/dim_geography.csv", index=False)

# --------------------
# dim_payment_method
# --------------------
pm_vals = df["payment_method"].dropna().drop_duplicates().sort_values()
pm_key_map = make_surrogate_key(pm_vals)
dim_payment_method = pd.DataFrame({
    "payment_method_key": [pm_key_map[v] for v in pm_vals],
    "payment_method": pm_vals
})
# Evidence
p("\n=== dim_payment_method ===", f"rows={len(dim_payment_method)} unique payment_method={len(pm_vals)}")
p("head():", dim_payment_method.head())

dim_payment_method.to_csv("output/dim_payment_method.csv", index=False)

# --------------------
# dim_order_status
# --------------------
status_vals = df["order_status"].dropna().drop_duplicates().sort_values()
status_key_map = make_surrogate_key(status_vals)
dim_order_status = pd.DataFrame({
    "order_status_key": [status_key_map[v] for v in status_vals],
    "order_status": status_vals
})
p("\n=== dim_order_status ===", f"rows={len(dim_order_status)} unique statuses={len(status_vals)}")
p("head():", dim_order_status.head())
dim_order_status.to_csv("output/dim_order_status.csv", index=False)

# --------------------
# dim_acquisition_channel
# --------------------
acq_vals = df["acquisition_channel"].dropna().drop_duplicates().sort_values()
acq_key_map = make_surrogate_key(acq_vals)
dim_acquisition_channel = pd.DataFrame({
    "acquisition_channel_key": [acq_key_map[v] for v in acq_vals],
    "acquisition_channel": acq_vals
})
p("\n=== dim_acquisition_channel ===", f"rows={len(dim_acquisition_channel)} unique channels={len(acq_vals)}")
p("head():", dim_acquisition_channel.head())
dim_acquisition_channel.to_csv("output/dim_acquisition_channel.csv", index=False)

# --------------------
# Key integrity checks (mapping coverage)
# --------------------
checks = {
    "customer_key_mapped_rows": int(df["customer_id"].map(cust_key_map).notna().sum()),
    "product_key_mapped_rows": int(df["product_id"].map(prod_key_map).notna().sum()),
    "payment_method_key_mapped_rows": int(df["payment_method"].map(pm_key_map).notna().sum()),
    "order_status_key_mapped_rows": int(df["order_status"].map(status_key_map).notna().sum()),
    "acquisition_channel_key_mapped_rows": int(df["acquisition_channel"].map(acq_key_map).notna().sum()),
}
p("\n=== Dimension key mapping coverage (row-level) ===", checks)

# Geography coverage: build tuple from df then map
df_geo_tuples = df.apply(lambda r: (r["city"], r["state"], r["country"]), axis=1)
geo_cov = int(df_geo_tuples.map(geo_key_map).notna().sum())
p("geography_key_mapped_rows:", geo_cov)

# Date coverage
order_date_cov = int(df["order_date"].notna().sum())
delivery_date_cov = int(df["delivery_date"].notna().sum())
p("date coverage (non-null in df):", {"order_date_non_null": order_date_cov, "delivery_date_non_null": delivery_date_cov})

# Unique key checks for dims
p("\n=== Unique key checks ===", {
    "dim_date date_key unique": dim_date_out["date_key"].is_unique,
    "dim_customer customer_key unique": dim_customer["customer_key"].is_unique,
    "dim_product product_key unique": dim_product["product_key"].is_unique,
    "dim_geography geography_key unique": dim_geography["geography_key"].is_unique,
    "dim_payment_method payment_method_key unique": dim_payment_method["payment_method_key"].is_unique,
    "dim_order_status order_status_key unique": dim_order_status["order_status_key"].is_unique,
    "dim_acquisition_channel acquisition_channel_key unique": dim_acquisition_channel["acquisition_channel_key"].is_unique,
})