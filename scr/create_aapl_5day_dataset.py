from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = DATA_DIR / "features.parquet"

OUTPUT_FULL = DATA_DIR / "AAPL_5day_model_dataset.parquet"
OUTPUT_TRAIN = DATA_DIR / "AAPL_5day_train.parquet"
OUTPUT_VALIDATION = DATA_DIR / "AAPL_5day_validation.parquet"
OUTPUT_TEST = DATA_DIR / "AAPL_5day_test.parquet"


print("=" * 60)
print("CREATING AAPL 5-DAY TARGET DATASET")
print("=" * 60)

df = pd.read_parquet(INPUT_FILE)

print("Original shape:", df.shape)

# Convert MultiIndex columns into simple column names
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [
        f"{level_0}_{level_1}"
        for level_0, level_1 in df.columns
    ]

# Convert Date index into a normal column
if isinstance(df.index, pd.DatetimeIndex):
    df = df.reset_index()

else:
    raise ValueError(
        "Expected a DatetimeIndex in features.parquet"
    )

# Rename Date column consistently
if "Date" in df.columns:
    df = df.rename(columns={"Date": "date"})

elif "date" not in df.columns:
    raise ValueError(
        "Date column not found after resetting index"
    )

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


# Check required AAPL close column
if "Close_AAPL" not in df.columns:
    raise ValueError(
        "Close_AAPL column not found in features.parquet"
    )


# Calculate future 5-trading-day return
df["AAPL_future_return_5d"] = (
    df["Close_AAPL"].shift(-5) / df["Close_AAPL"]
) - 1


# 1 = positive future return
# 0 = zero or negative future return
df["target"] = (
    df["AAPL_future_return_5d"] > 0
).astype(int)


# Remove rows without future price
df = df.dropna(
    subset=["AAPL_future_return_5d"]
).copy()


selected_features = [
    "date",

    # AAPL features
    "Features_AAPL_return_1d",
    "Features_AAPL_return_5d",
    "Features_AAPL_return_20d",
    "Features_AAPL_return_60d",
    "Features_AAPL_log_return",

    "Features_AAPL_ma_5",
    "Features_AAPL_ma_20",
    "Features_AAPL_ma_50",
    "Features_AAPL_ma_200",

    "Features_AAPL_price_to_ma20",
    "Features_AAPL_price_to_ma50",
    "Features_AAPL_price_to_ma200",

    "Features_AAPL_volatility_5d",
    "Features_AAPL_volatility_20d",

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

    # New target
    "AAPL_future_return_5d",
    "target",
]


missing_columns = [
    column
    for column in selected_features
    if column not in df.columns
]

if missing_columns:
    print("\nMissing columns:")
    for column in missing_columns:
        print(" -", column)

    raise ValueError(
        "Some selected features are missing from features.parquet"
    )


model_df = df[selected_features].copy()

model_df = model_df.replace(
    [np.inf, -np.inf],
    np.nan
)


feature_columns = [
    column
    for column in selected_features
    if column not in [
        "date",
        "AAPL_future_return_5d",
        "target",
    ]
]

model_df = model_df.dropna(
    subset=feature_columns
).reset_index(drop=True)


# Chronological split
n = len(model_df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train_df = model_df.iloc[:train_end].copy()

validation_df = model_df.iloc[
    train_end:validation_end
].copy()

test_df = model_df.iloc[
    validation_end:
].copy()


print("\nFinal dataset shape:", model_df.shape)

print("\nDate ranges:")

print(
    "Train:",
    train_df["date"].min().date(),
    "to",
    train_df["date"].max().date()
)

print(
    "Validation:",
    validation_df["date"].min().date(),
    "to",
    validation_df["date"].max().date()
)

print(
    "Test:",
    test_df["date"].min().date(),
    "to",
    test_df["date"].max().date()
)

print("\nTarget distribution:")
print(
    model_df["target"]
    .value_counts()
    .sort_index()
)

print("\nTarget proportion:")
print(
    model_df["target"]
    .value_counts(normalize=True)
    .sort_index()
    .round(4)
)


print("\nSaving datasets...")

model_df.to_parquet(
    OUTPUT_FULL,
    index=False
)

train_df.to_parquet(
    OUTPUT_TRAIN,
    index=False
)

validation_df.to_parquet(
    OUTPUT_VALIDATION,
    index=False
)

test_df.to_parquet(
    OUTPUT_TEST,
    index=False
)


print("\nSaved files:")
print(OUTPUT_FULL)
print(OUTPUT_TRAIN)
print(OUTPUT_VALIDATION)
print(OUTPUT_TEST)

print("\n5-day dataset creation complete.")