import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no interactive display per instruction

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Safe printing for Windows consoles (avoid UnicodeEncodeError)
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

# Load raw data as strings
raw = pd.read_csv("ecommerce_orders_raw.csv", dtype=str, keep_default_na=True)
p("=== Raw shape ===", raw.shape)

# Strip whitespace from all string cells
df = raw.copy()
for col in df.columns:
    df[col] = df[col].astype(str).str.strip()

# Initialize quality flags
flags = pd.DataFrame(index=df.index)
flags["quantity_negative"] = False
flags["shipping_cost_negative"] = False
flags["age_invalid"] = False
flags["discount_out_of_range"] = False
flags["unit_price_nonpositive"] = False
flags["email_invalid"] = False

# Helper: parse numeric strings
def parse_currency_inr(s):
    if pd.isna(s):
        return np.nan
    s = str(s)
    # Remove INR symbols and textual prefixes; keep digits, dot, minus
    s = s.replace("Rs.", "").replace("Rs", "").replace("INR", "")
    s = s.replace("₹", "")
    # Remove commas and spaces
    s = s.replace(",", "").strip()
    # If now empty, return NaN
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

def parse_int(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    return np.nan

def parse_float_pct(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip().replace("%", "")
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan

# 1) Numeric conversions
df["unit_price"] = df["unit_price"].apply(parse_currency_inr)
df["discount_pct"] = df["discount_pct"].apply(parse_float_pct)
df["quantity"] = df["quantity"].apply(parse_int)
df["shipping_cost"] = df["shipping_cost"].apply(parse_int)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["customer_age"] = pd.to_numeric(df["customer_age"], errors="coerce")

# 2) Validate/adjust numeric values, set flags
# Quantity: negative -> flag and set to NaN (await business rule), zero allowed
neg_qty_mask = df["quantity"].notna() & (df["quantity"] < 0)
flags.loc[neg_qty_mask, "quantity_negative"] = True
df.loc[neg_qty_mask, "quantity"] = np.nan

# Shipping cost: negative -> flag and set to 0
neg_ship_mask = df["shipping_cost"].notna() & (df["shipping_cost"] < 0)
flags.loc[neg_ship_mask, "shipping_cost_negative"] = True
df.loc[neg_ship_mask, "shipping_cost"] = 0

# Unit price: <=0 -> flag and set to NaN
nonpos_price_mask = df["unit_price"].notna() & (df["unit_price"] <= 0)
flags.loc[nonpos_price_mask, "unit_price_nonpositive"] = True
df.loc[nonpos_price_mask, "unit_price"] = np.nan

# Discount: <0 or >100 -> flag and set to NaN
bad_disc_mask = df["discount_pct"].notna() & ((df["discount_pct"] < 0) | (df["discount_pct"] > 100))
flags.loc[bad_disc_mask, "discount_out_of_range"] = True
df.loc[bad_disc_mask, "discount_pct"] = np.nan

# Age: negative or >120 -> flag and set to NaN
bad_age_mask = df["customer_age"].notna() & ((df["customer_age"] < 0) | (df["customer_age"] > 120))
flags.loc[bad_age_mask, "age_invalid"] = True
df.loc[bad_age_mask, "customer_age"] = np.nan

# 3) Dates: standardize to ISO YYYY-MM-DD
def parse_date_any(s):
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    # ISO YYYY-MM-DD
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        try:
            return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
        except Exception:
            return pd.NaT
    # Slash format: assume DD/MM/YYYY (India convention)
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", s):
        try:
            return pd.to_datetime(s, dayfirst=True, errors="coerce")
        except Exception:
            return pd.NaT
    # MonthName DD, YYYY
    if re.fullmatch(r"[A-Za-z]{3,9} \d{1,2}, \d{4}", s):
        try:
            return pd.to_datetime(s, errors="coerce")
        except Exception:
            return pd.NaT
    # Fallback: try general parse
    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=True)
    except Exception:
        return pd.NaT

df["order_date"] = df["order_date"].apply(parse_date_any)
df["delivery_date"] = df["delivery_date"].apply(parse_date_any)

# 4) Email/phone validation and normalization
email_series = df["email"].replace({np.nan: None})
email_valid = email_series.apply(lambda x: bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", x)) if x is not None else False)
flags["email_invalid"] = ~email_valid

# Normalize phone: keep only digits and leading + if present; remove dashes/spaces
def normalize_phone(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    # Keep '+' if leading, then digits
    if s.startswith("+"):
        return "+" + re.sub(r"\D", "", s)
    return re.sub(r"\D", "", s)

df["phone_normalized"] = df["phone"].apply(normalize_phone)

# 5) Categorical normalization
# Gender
def norm_gender(x):
    if pd.isna(x):
        return np.nan
    v = str(x).strip().lower()
    if v in ["male", "m"]:
        return "Male"
    if v in ["female", "f"]:
        return "Female"
    return np.nan

df["gender"] = df["gender"].apply(norm_gender)

# Order status
status_map = {
    "delivered": "Delivered",
    "shipped": "Shipped",
    "pending": "Pending",
    "cancelled": "Cancelled",
    "returned": "Returned",
}
def norm_status(x):
    if pd.isna(x):
        return np.nan
    v = str(x).strip().lower()
    return status_map.get(v, v.title())

df["order_status"] = df["order_status"].apply(norm_status)

# Payment method
pm_map = {
    "cash on delivery": "Cash on Delivery",
    "cod": "Cash on Delivery",
    "credit card": "Credit Card",
    "debit card": "Debit Card",
    "net banking": "Net Banking",
    "upi": "UPI",
    "wallet": "Wallet",
}
def norm_payment(x):
    if pd.isna(x):
        return np.nan
    v = str(x).strip().lower()
    return pm_map.get(v, v.title())

df["payment_method"] = df["payment_method"].apply(norm_payment)

# Country normalization
country_map = {"india": "India", "bharat": "India", "in": "India", "india ": "India", "india": "India"}
def norm_country(x):
    if pd.isna(x):
        return np.nan
    v = str(x).strip().lower()
    return country_map.get(v, "India" if v.upper() in ["INDIA"] else v.title())

df["country"] = df["country"].apply(norm_country)

# State normalization (minimal known abbreviations)
state_map = {"mh": "Maharashtra", "dl": "Delhi", "gj": "Gujarat", "tn": "Tamil Nadu", "wb": "West Bengal", "ka": "Karnataka", "tg": "Telangana", "kl": "Kerala"}
def norm_state(x):
    if pd.isna(x):
        return np.nan
    v = str(x).strip()
    low = v.lower()
    if low in state_map:
        return state_map[low]
    # Title case common pattern
    return v.title()

df["state"] = df["state"].apply(norm_state)

# City normalization: title-case; special handling for known uppercase variants
df["city"] = df["city"].apply(lambda x: np.nan if pd.isna(x) else str(x).strip().title())

# Category normalization
cat_map = {
    "electronics": "Electronics",
    "home & kitchen": "Home & Kitchen",
    "beauty": "Beauty",
    "fashion": "Fashion",
    "books": "Books",
    "sports": "Sports",
}
def norm_category(x):
    if pd.isna(x):
        return np.nan
    v = str(x).strip()
    key = v.lower().replace("  ", " ")
    return cat_map.get(key, v.title())

df["category"] = df["category"].apply(norm_category)

# Product name: strip and title-case (keep as-is if mixed case significant)
df["product_name"] = df["product_name"].apply(lambda x: np.nan if pd.isna(x) else str(x).strip())

# 6) Deduplicate exact rows (keep first)
before_rows = len(df)
df = df.drop_duplicates(keep="first")
after_rows = len(df)

# 7) Evidence prints
p("\n=== Rows before/after exact-duplicate drop ===", f"{before_rows} -> {after_rows}")

p("\n=== Cleaned dtypes ===", df.dtypes)

p("\n=== Head(5) cleaned ===", df.head(5))

# Flag summary counts
flag_counts = flags.loc[df.index].sum().astype(int)
p("\n=== Data quality flags (counts) ===", flag_counts)

# Null counts after cleaning
null_counts_after = df.isna().sum()
p("\n=== Null counts after cleaning ===", null_counts_after.sort_values(ascending=False).head(15))

# Date parsing success rates
p("\n=== Date parse success ===", {
    "order_date_non_null": int(df["order_date"].notna().sum()),
    "delivery_date_non_null": int(df["delivery_date"].notna().sum())
})

# 8) Save cleaned CSV
out_path = os.path.join("output", "ecommerce_orders_clean.csv")
df_out = df.copy()

# Ensure dates are saved as ISO strings
for dcol in ["order_date", "delivery_date"]:
    df_out[dcol] = df_out[dcol].dt.strftime("%Y-%m-%d")

# Append flags to the output for traceability
for col in flags.columns:
    df_out[col] = flags.loc[df_out.index, col].astype(int)

df_out.to_csv(out_path, index=False)
p("\n=== Saved cleaned file ===", out_path)