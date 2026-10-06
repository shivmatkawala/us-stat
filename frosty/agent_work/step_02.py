import os
import re
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no interactive display per instruction

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Helper: make any Python object safe for cp1252/ASCII console printing
def to_ascii(obj):
    if isinstance(obj, str):
        return obj.encode("ascii", "replace").decode("ascii")
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_ascii(x) for x in obj)
    if isinstance(obj, dict):
        return {to_ascii(k): to_ascii(v) for k, v in obj.items()}
    # For pandas objects, convert to string then sanitize
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return to_ascii(obj.to_string())
    return obj

def p(label, obj):
    print(to_ascii(label))
    print(to_ascii(obj))

# 1) Load raw data (strings to avoid premature type coercion)
df = pd.read_csv("ecommerce_orders_raw.csv", dtype=str, keep_default_na=True)

p("=== SHAPE ===", df.shape)

p("\n=== COLUMN DTYPES (as loaded) ===", df.dtypes)

p("\n=== HEAD(5) ===", df.head(5))

# 2) Null / missing profiling
null_counts = df.isna().sum()
null_pct = (df.isna().mean() * 100).round(2)
p("\n=== NULL COUNTS & PCT ===", pd.DataFrame({"null_count": null_counts, "null_pct": null_pct}).sort_values("null_count", ascending=False))

# 3) Basic cardinality checks for key-ish columns
for col in ["order_id", "customer_id", "product_id"]:
    uniq = df[col].nunique(dropna=True)
    total = len(df)
    p(f"\n=== Cardinality: {col} ===", f"unique values: {uniq} | total rows: {total}")

# 4) Grain/duplicates
dup_pair = df.duplicated(subset=["order_id", "product_id"]).sum()
p("\n=== Duplicate check for (order_id, product_id) pair ===", f"duplicate rows count: {dup_pair}")

order_id_repeats = df.duplicated(subset=["order_id"]).sum()
p("\n=== Duplicate check for order_id alone ===", f"order_id duplicate rows count: {order_id_repeats}")

# 5) Numeric-like fields stored as strings: format diagnostics using full-series masks (avoid index misalignment)
unit_price_series = df["unit_price"].astype(str)
unit_price_non_numeric_mask = unit_price_series.apply(lambda x: bool(re.search(r"[^0-9.\-]", x)))
unit_price_bad_count = int(unit_price_non_numeric_mask.sum())
unit_price_bad_examples = unit_price_series[unit_price_non_numeric_mask].head(10).tolist()
p("\n=== unit_price formatting issues ===", f"rows with non-numeric characters: {unit_price_bad_count}")
p("examples:", unit_price_bad_examples)

discount_series = df["discount_pct"].astype(str)
discount_non_numeric_mask = discount_series.apply(lambda x: bool(re.search(r"[^0-9.\-]", x)))
discount_bad_count = int(discount_non_numeric_mask.sum())
discount_bad_examples = discount_series[discount_non_numeric_mask].head(10).tolist()
p("\n=== discount_pct formatting issues ===", f"rows with non-numeric characters: {discount_bad_count}")
p("examples:", discount_bad_examples)

qty_series = df["quantity"].astype(str)
qty_non_int_mask = qty_series.apply(lambda x: not re.fullmatch(r"\d+", x))
p("\n=== quantity formatting issues ===", f"rows not pure integers: {int(qty_non_int_mask.sum())}")
p("examples:", qty_series[qty_non_int_mask].head(10).tolist())

ship_series = df["shipping_cost"].astype(str)
ship_non_int_mask = ship_series.apply(lambda x: not re.fullmatch(r"\d+", x))
p("\n=== shipping_cost formatting issues ===", f"rows not pure integers: {int(ship_non_int_mask.sum())}")
p("examples:", ship_series[ship_non_int_mask].head(10).tolist())

rating_series = df["rating"].astype(str)
rating_non_float_mask = rating_series.dropna().apply(lambda x: not re.fullmatch(r"\d+(\.\d+)?", x))
# Align mask to original index for examples; fill False where NaN
rating_mask_aligned = rating_series.index.to_series().isin(rating_series.dropna().index[rating_non_float_mask])
p("\n=== rating formatting issues ===", f"rows not float-like: {int(rating_non_float_mask.sum())}")
p("examples:", rating_series[rating_mask_aligned].head(10).tolist())

age_series = df["customer_age"].astype(str)
age_non_float_mask = age_series.dropna().apply(lambda x: not re.fullmatch(r"\d+(\.\d+)?", x))
age_mask_aligned = age_series.index.to_series().isin(age_series.dropna().index[age_non_float_mask])
p("\n=== customer_age formatting issues ===", f"rows not float-like: {int(age_non_float_mask.sum())}")
p("examples:", age_series[age_mask_aligned].head(10).tolist())

# 6) Date fields: check format variability
def classify_date(x):
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", x):
        return "ISO_YYYY-MM-DD"
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", x):
        return "SLASH_DD/MM/YYYY_or_MM/DD/YYYY"
    if re.fullmatch(r"[A-Za-z]{3,9} \d{1,2}, \d{4}", x):
        return "MonthName DD, YYYY"
    return "Other"

def date_format_report(series, name):
    s = series.dropna().astype(str)
    cats = s.apply(classify_date).value_counts()
    p(f"\n=== {name} format categories ===", cats)
    examples = {}
    for cat in cats.index[:3]:
        examples[cat] = s[s.apply(classify_date) == cat].head(10).tolist()
    p(f"Sample examples for {name}:", examples)

date_format_report(df["order_date"], "order_date")
date_format_report(df["delivery_date"], "delivery_date")

# 7) Phone/email sanity checks
phone_clean = df["phone"].dropna().astype(str)
phone_char_issues = phone_clean.apply(lambda x: bool(re.search(r"[^\d+ ]", x)))
p("\n=== phone formatting issues ===", f"rows with unexpected characters: {int(phone_char_issues.sum())}")
p("examples:", phone_clean[phone_char_issues].head(10).tolist())

email_clean = df["email"].dropna().astype(str)
email_basic_valid = email_clean.apply(lambda x: bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", x)))
p("\n=== email basic validity ===", f"valid emails: {int(email_basic_valid.sum())} / {len(email_clean)}")
p("invalid examples:", email_clean[~email_basic_valid].head(10).tolist())

# 8) Gender distribution and anomalies
gender_clean = df["gender"].fillna("NULL").astype(str).str.strip()
gender_counts = gender_clean.value_counts()
unexpected_gender = gender_clean[~gender_clean.str.lower().isin(["male", "female", "null"])].unique().tolist()
p("\n=== gender distribution ===", gender_counts)
p("unexpected gender values:", unexpected_gender)

# 9) Order status distribution
if "order_status" in df.columns:
    p("\n=== order_status distribution ===", df["order_status"].fillna("NULL").value_counts())

# 10) Category/product consistency quick check
p("\n=== Top categories and products ===", "Top 10 categories:")
p("", df["category"].fillna("NULL").value_counts().head(10))
p("Top 10 products:", df["product_name"].fillna("NULL").value_counts().head(10))

# 11) Geography distribution (top)
p("\n=== Geography distribution (top) ===", "Top countries:")
p("", df["country"].fillna("NULL").value_counts().head(10).to_dict())
p("Top states:", df["state"].fillna("NULL").value_counts().head(10).to_dict())
p("Top cities:", df["city"].fillna("NULL").value_counts().head(10).to_dict())

# 12) Payment method distribution
p("\n=== payment_method distribution ===", df["payment_method"].fillna("NULL").value_counts().head(20))

# 13) Acquisition channel distribution
p("\n=== acquisition_channel distribution ===", df["acquisition_channel"].fillna("NULL").value_counts().head(20))