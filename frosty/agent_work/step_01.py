import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # per instruction; no interactive display

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# 1) Load raw data (read all columns as strings to avoid premature type coercion during profiling)
df = pd.read_csv("ecommerce_orders_raw.csv", dtype=str, keep_default_na=True)

print("=== SHAPE ===")
print(df.shape)

print("\n=== COLUMN DTYPES (as loaded) ===")
print(df.dtypes)

print("\n=== HEAD(5) ===")
print(df.head(5))

# 2) Null / missing profiling
null_counts = df.isna().sum()
null_pct = (df.isna().mean() * 100).round(2)
print("\n=== NULL COUNTS & PCT ===")
print(pd.DataFrame({"null_count": null_counts, "null_pct": null_pct}).sort_values("null_count", ascending=False))

# 3) Basic cardinality checks for key-ish columns
key_cols = ["order_id", "customer_id", "product_id"]
for col in key_cols:
    uniq = df[col].nunique(dropna=True)
    total = len(df)
    print(f"\n=== Cardinality: {col} ===")
    print(f"unique values: {uniq} | total rows: {total}")

# 4) Check potential grain/duplicates:
#    - Line-level grain is often (order_id, product_id). We'll inspect duplicates for that pair.
dup_pair = df.duplicated(subset=["order_id", "product_id"]).sum()
print("\n=== Duplicate check for (order_id, product_id) pair ===")
print(f"duplicate rows count: {dup_pair}")

# Also check if order_id alone repeats (suggesting multi-line orders)
order_id_repeats = df.duplicated(subset=["order_id"]).sum()
print("\n=== Duplicate check for order_id alone ===")
print(f"order_id duplicate rows count: {order_id_repeats}")

# 5) Numeric-like fields stored as strings: unit_price, discount_pct, quantity, shipping_cost, rating, customer_age
def sample_non_numeric(col, pat):
    s = df[col].dropna()
    mask = ~s.str.fullmatch(pat)
    examples = s[mask].head(10).tolist()
    return mask.sum(), examples

# Patterns:
# - unit_price may have currency symbols and commas; numeric after stripping these should be digits with optional decimal
#   We'll first inspect raw format.
unit_price_non_digits = df["unit_price"].dropna().apply(lambda x: bool(re.search(r"[^0-9.\-]", x)))
unit_price_bad_count = int(unit_price_non_digits.sum())
unit_price_bad_examples = df.loc[df["unit_price"].dropna().index[unit_price_non_digits.head(10).index], "unit_price"].head(10).tolist()
print("\n=== unit_price formatting issues ===")
print(f"rows with non-numeric characters: {unit_price_bad_count}")
print(f"examples: {unit_price_bad_examples}")

# discount_pct may include % sign; check for non-digit characters aside from decimal and minus
discount_non_digits = df["discount_pct"].dropna().apply(lambda x: bool(re.search(r"[^0-9.\-]", x)))
discount_bad_count = int(discount_non_digits.sum())
discount_bad_examples = df.loc[df["discount_pct"].dropna().index[discount_non_digits.head(10).index], "discount_pct"].head(10).tolist()
print("\n=== discount_pct formatting issues ===")
print(f"rows with non-numeric characters: {discount_bad_count}")
print(f"examples: {discount_bad_examples}")

# quantity should be integer-like
qty_non_int = df["quantity"].dropna().apply(lambda x: not re.fullmatch(r"\d+", x))
print("\n=== quantity formatting issues ===")
print(f"rows not pure integers: {int(qty_non_int.sum())}")
print(f"examples: {df.loc[df['quantity'].dropna().index[qty_non_int.head(10).index], 'quantity'].head(10).tolist()}")

# shipping_cost should be integer-like
ship_non_int = df["shipping_cost"].dropna().apply(lambda x: not re.fullmatch(r"\d+", x))
print("\n=== shipping_cost formatting issues ===")
print(f"rows not pure integers: {int(ship_non_int.sum())}")
print(f"examples: {df.loc[df['shipping_cost'].dropna().index[ship_non_int.head(10).index], 'shipping_cost'].head(10).tolist()}")

# rating should be float-like
rating_non_float = df["rating"].dropna().apply(lambda x: not re.fullmatch(r"\d+(\.\d+)?", x))
print("\n=== rating formatting issues ===")
print(f"rows not float-like: {int(rating_non_float.sum())}")
print(f"examples: {df.loc[df['rating'].dropna().index[rating_non_float.head(10).index], 'rating'].head(10).tolist()}")

# customer_age should be float-like
age_non_float = df["customer_age"].dropna().apply(lambda x: not re.fullmatch(r"\d+(\.\d+)?", x))
print("\n=== customer_age formatting issues ===")
print(f"rows not float-like: {int(age_non_float.sum())}")
print(f"examples: {df.loc[df['customer_age'].dropna().index[age_non_float.head(10).index], 'customer_age'].head(10).tolist()}")

# 6) Date fields: check format variability for order_date and delivery_date
def date_format_samples(series, name, n=10):
    s = series.dropna().astype(str)
    # Heuristic classify: ISO (YYYY-MM-DD), D/M/YYYY, M/D/YYYY, Month DD, YYYY, etc.
    def classify(x):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", x):
            return "ISO_YYYY-MM-DD"
        if re.fullmatch(r"\d{2}/\d{2}/\d{4}", x):
            # Ambiguous: could be DD/MM/YYYY or MM/DD/YYYY
            return "SLASH_DD/MM/YYYY_or_MM/DD/YYYY"
        if re.fullmatch(r"[A-Za-z]{3,9} \d{1,2}, \d{4}", x):
            return "MonthName DD, YYYY"
        return "Other"
    cats = s.apply(classify).value_counts()
    print(f"\n=== {name} format categories ===")
    print(cats)
    examples = {}
    for cat in cats.index[:3]:
        examples[cat] = s[s.apply(classify) == cat].head(n).tolist()
    print(f"Sample examples for {name}:")
    for k, v in examples.items():
        print(f"{k}: {v}")

date_format_samples(df["order_date"], "order_date")
date_format_samples(df["delivery_date"], "delivery_date")

# 7) Phone/email sanity checks
phone_clean = df["phone"].dropna().astype(str)
phone_char_issues = phone_clean.apply(lambda x: bool(re.search(r"[^\d+ ]", x)))
print("\n=== phone formatting issues ===")
print(f"rows with unexpected characters: {int(phone_char_issues.sum())}")
print(f"examples: {df.loc[phone_clean.index[phone_char_issues.head(10).index], 'phone'].head(10).tolist()}")

email_clean = df["email"].dropna().astype(str)
email_basic_valid = email_clean.apply(lambda x: bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", x)))
print("\n=== email basic validity ===")
print(f"valid emails: {int(email_basic_valid.sum())} / {len(email_clean)}")
invalid_examples = email_clean[~email_basic_valid].head(10).tolist()
print(f"invalid examples: {invalid_examples}")

# 8) Gender distribution and anomalies (e.g., 'NaN', unexpected labels)
gender_clean = df["gender"].fillna("NULL").astype(str).str.strip()
gender_counts = gender_clean.value_counts()
unexpected_gender = gender_clean[~gender_clean.str.lower().isin(["male", "female", "null"])].unique().tolist()
print("\n=== gender distribution ===")
print(gender_counts)
print(f"unexpected gender values: {unexpected_gender}")

# 9) Order status distribution
if "order_status" in df.columns:
    print("\n=== order_status distribution ===")
    print(df["order_status"].fillna("NULL").value_counts())

# 10) Category/product consistency quick check
print("\n=== Top categories and products ===")
print("Top 10 categories:")
print(df["category"].fillna("NULL").value_counts().head(10))
print("Top 10 products:")
print(df["product_name"].fillna("NULL").value_counts().head(10))

# 11) Country/state/city distribution (top)
print("\n=== Geography distribution (top) ===")
print("Top countries:", df["country"].fillna("NULL").value_counts().head(10).to_dict())
print("Top states:", df["state"].fillna("NULL").value_counts().head(10).to_dict())
print("Top cities:", df["city"].fillna("NULL").value_counts().head(10).to_dict())

# 12) Payment method distribution
print("\n=== payment_method distribution ===")
print(df["payment_method"].fillna("NULL").value_counts().head(20))

# 13) Acquisition channel distribution
print("\n=== acquisition_channel distribution ===")
print(df["acquisition_channel"].fillna("NULL").value_counts().head(20))