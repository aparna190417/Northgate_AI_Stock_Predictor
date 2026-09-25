from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

FILES = {
    "TRAIN": DATA_DIR / "AAPL_5day_normalized_train.parquet",
    "VALIDATION": DATA_DIR / "AAPL_5day_normalized_validation.parquet",
    "TEST": DATA_DIR / "AAPL_5day_normalized_test.parquet",
}

print("=" * 80)
print("NORTHGATE AI — AAPL 5-DAY DATASET AUDIT")
print("=" * 80)

datasets = {}

# ============================================================
# LOAD
# ============================================================

for name, path in FILES.items():

    print(f"\n{'=' * 80}")
    print(f"LOADING {name}")
    print(f"{'=' * 80}")
    print(f"File: {path}")

    if not path.exists():
        print("ERROR: File not found!")
        continue

    df = pd.read_parquet(path)
    datasets[name] = df

    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")

# ============================================================
# BASIC INFORMATION
# ============================================================

for name, df in datasets.items():

    print(f"\n{'=' * 80}")
    print(f"{name} — BASIC INFORMATION")
    print(f"{'=' * 80}")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nFirst 10 columns:")
    for col in df.columns[:10]:
        print(f"  {col}")

    print("\nLast 10 columns:")
    for col in df.columns[-10:]:
        print(f"  {col}")

# ============================================================
# DATE ANALYSIS
# ============================================================

for name, df in datasets.items():

    print(f"\n{'=' * 80}")
    print(f"{name} — DATE ANALYSIS")
    print(f"{'=' * 80}")

    date_columns = [
        c for c in df.columns
        if c.lower() in ["date", "datetime", "timestamp"]
    ]

    if not date_columns:
        print("No explicit date column found.")

        if isinstance(df.index, pd.DatetimeIndex):
            print("DatetimeIndex detected.")
            print(f"Start: {df.index.min()}")
            print(f"End  : {df.index.max()}")
        else:
            print("No DatetimeIndex detected.")

    else:
        for col in date_columns:
            dates = pd.to_datetime(df[col], errors="coerce")

            print(f"\nDate column: {col}")
            print(f"Invalid dates: {dates.isna().sum()}")
            print(f"Start: {dates.min()}")
            print(f"End  : {dates.max()}")
            print(f"Unique dates: {dates.nunique()}")
            print(f"Sorted: {dates.is_monotonic_increasing}")
            print(f"Duplicates: {dates.duplicated().sum()}")

# ============================================================
# TARGET ANALYSIS
# ============================================================

for name, df in datasets.items():

    print(f"\n{'=' * 80}")
    print(f"{name} — TARGET ANALYSIS")
    print(f"{'=' * 80}")

    if "target" not in df.columns:
        print("ERROR: target column missing!")
        continue

    target = df["target"]

    print("\nTarget dtype:")
    print(target.dtype)

    print("\nTarget values:")
    print(target.value_counts(dropna=False).sort_index())

    print("\nTarget proportions:")
    print(target.value_counts(normalize=True, dropna=False).sort_index())

    print(f"\nMissing target: {target.isna().sum()}")

    print(f"Unique target values: {target.nunique(dropna=False)}")

# ============================================================
# DATA QUALITY
# ============================================================

for name, df in datasets.items():

    print(f"\n{'=' * 80}")
    print(f"{name} — DATA QUALITY")
    print(f"{'=' * 80}")

    numeric_df = df.select_dtypes(include=[np.number])

    missing_total = df.isna().sum().sum()
    infinite_total = np.isinf(numeric_df).sum().sum()
    duplicate_rows = df.duplicated().sum()

    print(f"Total missing values : {missing_total}")
    print(f"Total infinite values: {infinite_total}")
    print(f"Duplicate rows       : {duplicate_rows}")

    constant_columns = [
        col for col in numeric_df.columns
        if numeric_df[col].nunique(dropna=False) <= 1
    ]

    print(f"Constant columns     : {len(constant_columns)}")

    if constant_columns:
        print("\nConstant columns:")
        for col in constant_columns:
            print(f"  {col}")

# ============================================================
# SUSPICIOUS FUTURE / TARGET COLUMNS
# ============================================================

for name, df in datasets.items():

    print(f"\n{'=' * 80}")
    print(f"{name} — SUSPICIOUS COLUMN CHECK")
    print(f"{'=' * 80}")

    suspicious_words = [
        "future",
        "forward",
        "target",
        "label",
        "next",
        "lead",
        "5d",
        "5_day"
    ]

    suspicious = []

    for col in df.columns:

        col_lower = col.lower()

        if any(word in col_lower for word in suspicious_words):
            suspicious.append(col)

    if suspicious:
        print("Potentially suspicious columns:")

        for col in suspicious:
            print(f"  {col}")
    else:
        print("No suspicious column names found.")

# ============================================================
# AAPL-SPECIFIC FEATURES
# ============================================================

for name, df in datasets.items():

    print(f"\n{'=' * 80}")
    print(f"{name} — AAPL FEATURE CHECK")
    print(f"{'=' * 80}")

    aapl_columns = [
        col for col in df.columns
        if "AAPL" in col.upper()
    ]

    print(f"AAPL-related columns: {len(aapl_columns)}")

    for col in aapl_columns:
        print(f"  {col}")

# ============================================================
# TRAIN / VALIDATION / TEST OVERLAP
# ============================================================

print(f"\n{'=' * 80}")
print("TRAIN / VALIDATION / TEST OVERLAP CHECK")
print(f"{'=' * 80}")

def get_dates(df):

    for col in ["date", "Date", "datetime", "Datetime", "timestamp", "Timestamp"]:

        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce")

    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=df.index)

    return None


date_sets = {}

for name, df in datasets.items():

    dates = get_dates(df)

    if dates is not None:

        dates = pd.Series(dates).dropna()

        date_sets[name] = set(dates.dt.normalize())

        print(
            f"{name}: "
            f"{len(date_sets[name])} unique dates | "
            f"{min(date_sets[name])} -> {max(date_sets[name])}"
        )
    else:
        print(f"{name}: date information unavailable")

if len(date_sets) == 3:

    train_val = date_sets["TRAIN"] & date_sets["VALIDATION"]
    train_test = date_sets["TRAIN"] & date_sets["TEST"]
    val_test = date_sets["VALIDATION"] & date_sets["TEST"]

    print("\nDate overlap:")
    print(f"TRAIN ↔ VALIDATION: {len(train_val)}")
    print(f"TRAIN ↔ TEST      : {len(train_test)}")
    print(f"VALIDATION ↔ TEST : {len(val_test)}")

# ============================================================
# FEATURE DISTRIBUTION SHIFT
# ============================================================

print(f"\n{'=' * 80}")
print("FEATURE DISTRIBUTION SHIFT")
print(f"{'=' * 80}")

if all(x in datasets for x in ["TRAIN", "VALIDATION", "TEST"]):

    train = datasets["TRAIN"]
    validation = datasets["VALIDATION"]
    test = datasets["TEST"]

    numeric_columns = train.select_dtypes(include=[np.number]).columns

    numeric_columns = [
        c for c in numeric_columns
        if c != "target"
        and c in validation.columns
        and c in test.columns
    ]

    shifts = []

    for col in numeric_columns:

        train_mean = train[col].mean()
        val_mean = validation[col].mean()
        test_mean = test[col].mean()

        train_std = train[col].std()

        if pd.isna(train_std) or train_std == 0:
            continue

        val_shift = abs(val_mean - train_mean) / train_std
        test_shift = abs(test_mean - train_mean) / train_std

        shifts.append({
            "feature": col,
            "validation_shift": val_shift,
            "test_shift": test_shift
        })

    shift_df = pd.DataFrame(shifts)

    if not shift_df.empty:

        print("\nLargest TEST distribution shifts:")

        print(
            shift_df
            .sort_values("test_shift", ascending=False)
            .head(20)
            .to_string(index=False)
        )

# ============================================================
# CORRELATION WITH TARGET
# ============================================================

print(f"\n{'=' * 80}")
print("TRAIN FEATURE / TARGET CORRELATION")
print(f"{'=' * 80}")

if "TRAIN" in datasets:

    train = datasets["TRAIN"]

    numeric_train = train.select_dtypes(include=[np.number])

    if "target" in numeric_train.columns:

        correlations = (
            numeric_train
            .corr()["target"]
            .drop("target")
            .abs()
            .sort_values(ascending=False)
        )

        print("\nTop 20 absolute correlations:")

        print(correlations.head(20).to_string())

# ============================================================
# FINAL SUMMARY
# ============================================================

print(f"\n{'=' * 80}")
print("AUDIT COMPLETE")
print(f"{'=' * 80}")

print("""
IMPORTANT:
This script does NOT modify the datasets.
It only audits them.

Next decision will be based on:
1. Date alignment
2. Target structure
3. Suspicious columns
4. Train/validation/test overlap
5. Distribution shift
6. Missing/infinite/duplicate data
7. Feature-target relationships
""")
