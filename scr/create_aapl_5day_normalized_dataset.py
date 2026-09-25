from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

HORIZON = 5

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING FEATURE DATASET")
print("=" * 70)

df = pd.read_parquet(FEATURES_PATH)

print(f"Original shape: {df.shape}")
print(f"Original index type: {type(df.index)}")


# ============================================================
# FLATTEN MULTI-INDEX COLUMNS
# ============================================================

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [
        "_".join(
            str(level)
            for level in column
            if str(level) not in ["", "None"]
        ).strip("_")
        for column in df.columns
    ]

print("\nFlattened columns:")
for column in df.columns:
    print(column)


# ============================================================
# PREPARE DATE COLUMN
# ============================================================

if isinstance(df.index, pd.DatetimeIndex):
    df = df.reset_index()

    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})

elif "Date" in df.columns:
    df = df.rename(columns={"Date": "date"})

elif "date" not in df.columns:
    raise ValueError(
        "Date information not found in index or columns."
    )


df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# CHECK REQUIRED CLOSE COLUMN
# ============================================================

if "Close_AAPL" not in df.columns:
    raise ValueError(
        "Close_AAPL column not found. "
        "Please check the columns in features.parquet."
    )


# ============================================================
# CREATE FUTURE RETURN AND TARGET
# ============================================================

print("\nCreating 5-day future target...")

df["future_close_5d"] = df["Close_AAPL"].shift(-HORIZON)

df["AAPL_future_return_5d"] = (
    df["future_close_5d"] / df["Close_AAPL"]
) - 1

# UP = 1
# DOWN = 0
df["target"] = (
    df["AAPL_future_return_5d"] > 0
).astype(int)


# ============================================================
# SELECT NORMALIZED FEATURES
# ============================================================

normalized_features = [
    # Stock returns
    "Features_AAPL_return_1d",
    "Features_AAPL_return_5d",
    "Features_AAPL_return_20d",
    "Features_AAPL_return_60d",
    "Features_AAPL_log_return",

    # Price ratios
    "Features_AAPL_price_to_ma20",
    "Features_AAPL_price_to_ma50",
    "Features_AAPL_price_to_ma200",

    # Volatility
    "Features_AAPL_volatility_5d",
    "Features_AAPL_volatility_20d",

    # Intraday and volume features
    "Features_AAPL_intraday_range",
    "Features_AAPL_volume_change",
    "Features_AAPL_volume_ma20",

    # Market features
    "Features_market_return_1d",
    "Features_market_return_5d",
    "Features_market_return_20d",
    "Features_market_volatility_20d",

    # Macro features
    "Features_CPIAUCSL",
    "Features_UNRATE",
    "Features_cpi_change",
    "Features_unemployment_change",

    # Calendar features
    "Features_day_of_week",
    "Features_month",
    "Features_quarter",
]


# ============================================================
# CHECK MISSING FEATURES
# ============================================================

missing_features = [
    feature
    for feature in normalized_features
    if feature not in df.columns
]

if missing_features:
    print("\nMissing features:")
    for feature in missing_features:
        print(f" - {feature}")

    raise ValueError(
        "Some normalized features are missing from the dataset."
    )


# ============================================================
# BUILD FINAL MODEL DATASET
# ============================================================

final_columns = [
    "date",
    *normalized_features,
    "target",
]

model_df = df[final_columns].copy()


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

before_drop = len(model_df)

model_df = model_df.dropna().reset_index(drop=True)

after_drop = len(model_df)

print("\nRows removed because of missing values:")
print(before_drop - after_drop)

print(f"Final dataset shape: {model_df.shape}")


# ============================================================
# VERIFY TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")
print(model_df["target"].value_counts())

print("\nTarget percentage:")
print(
    model_df["target"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# CHRONOLOGICAL TRAIN/VALIDATION/TEST SPLIT
# ============================================================

total_rows = len(model_df)

train_end = int(total_rows * TRAIN_RATIO)

validation_end = int(
    total_rows * (TRAIN_RATIO + VALIDATION_RATIO)
)

train_df = model_df.iloc[:train_end].copy()

validation_df = model_df.iloc[
    train_end:validation_end
].copy()

test_df = model_df.iloc[
    validation_end:
].copy()


# ============================================================
# SAVE DATASETS
# ============================================================

full_output_path = (
    OUTPUT_DIR / "AAPL_5day_normalized_model_dataset.parquet"
)

train_output_path = (
    OUTPUT_DIR / "AAPL_5day_normalized_train.parquet"
)

validation_output_path = (
    OUTPUT_DIR / "AAPL_5day_normalized_validation.parquet"
)

test_output_path = (
    OUTPUT_DIR / "AAPL_5day_normalized_test.parquet"
)


model_df.to_parquet(full_output_path, index=False)

train_df.to_parquet(train_output_path, index=False)

validation_df.to_parquet(
    validation_output_path,
    index=False
)

test_df.to_parquet(
    test_output_path,
    index=False
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZED DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print(f"Full dataset:       {model_df.shape}")
print(f"Training dataset:    {train_df.shape}")
print(f"Validation dataset:  {validation_df.shape}")
print(f"Test dataset:        {test_df.shape}")

print("\nDate ranges:")

print(
    f"Train:       {train_df['date'].min().date()} "
    f"to {train_df['date'].max().date()}"
)

print(
    f"Validation:   {validation_df['date'].min().date()} "
    f"to {validation_df['date'].max().date()}"
)

print(
    f"Test:         {test_df['date'].min().date()} "
    f"to {test_df['date'].max().date()}"
)

print("\nSaved files:")

print(full_output_path)
print(train_output_path)
print(validation_output_path)
print(test_output_path)

print("\nRaw moving averages are excluded:")
print(
    [
        column
        for column in model_df.columns
        if "ma_" in column.lower()
    ]
)

print("\nFuture return feature is excluded:")
print("AAPL_future_return_5d" not in model_df.columns)

print("\nFinal columns:")
for column in model_df.columns:
    print(f" - {column}")