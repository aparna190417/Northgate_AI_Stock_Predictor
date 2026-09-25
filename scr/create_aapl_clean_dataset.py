import pandas as pd
import numpy as np

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

STOCK = "AAPL"

INPUT_FILE = Path(
    "data/processed/features.parquet"
)

OUTPUT_FILE = Path(
    f"data/processed/{STOCK}_clean_model_dataset.parquet"
)

TRAIN_FILE = Path(
    f"data/processed/{STOCK}_clean_train.parquet"
)

VALIDATION_FILE = Path(
    f"data/processed/{STOCK}_clean_validation.parquet"
)

TEST_FILE = Path(
    f"data/processed/{STOCK}_clean_test.parquet"
)


# ============================================================
# LOAD FEATURE DATA
# ============================================================

print("=" * 60)
print("LOADING FEATURE DATA")
print("=" * 60)

df = pd.read_parquet(INPUT_FILE)

print(f"Full dataset shape: {df.shape}")


# ============================================================
# EXTRACT FEATURES AND TARGET
# ============================================================

features = df["Features"].copy()
targets = df["Target"].copy()

target_column = f"{STOCK}_next_direction"

if target_column not in targets.columns:
    raise ValueError(
        f"Target column not found: {target_column}"
    )

target = targets[target_column].rename("target")


# ============================================================
# SELECT CONTROLLED FEATURE GROUPS
# ============================================================

selected_columns = []

for column in features.columns:

    column_name = str(column)

    # AAPL-specific features
    if column_name.startswith(f"{STOCK}_"):
        selected_columns.append(column)

    # Market features
    elif column_name.startswith("market_"):
        selected_columns.append(column)

    # VIX features
    elif column_name.startswith("VIX_"):
        selected_columns.append(column)

    # Macro features
    elif column_name in [
        "CPIAUCSL",
        "UNRATE",
        "cpi_change",
        "unemployment_change"
    ]:
        selected_columns.append(column)

    # Calendar features
    elif column_name in [
        "day_of_week",
        "month",
        "quarter",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end"
    ]:
        selected_columns.append(column)


if len(selected_columns) == 0:
    raise ValueError(
        "No feature columns were selected. "
        "Please check the feature column names."
    )

clean_features = features[selected_columns].copy()


# ============================================================
# COMBINE FEATURES AND TARGET
# ============================================================

model_df = pd.concat(
    [
        clean_features,
        target
    ],
    axis=1
)

print(f"\nSelected feature count: {len(selected_columns)}")
print(f"Dataset shape before cleaning: {model_df.shape}")


# ============================================================
# CLEAN INF VALUES
# ============================================================

model_df = model_df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

rows_before = len(model_df)

model_df = model_df.dropna()

rows_after = len(model_df)

print(f"Rows removed: {rows_before - rows_after}")
print(f"Dataset shape after cleaning: {model_df.shape}")


# ============================================================
# SORT BY DATE
# ============================================================

model_df = model_df.sort_index()


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

total_rows = len(model_df)

train_end = int(total_rows * 0.70)
validation_end = int(total_rows * 0.85)

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

model_df.to_parquet(OUTPUT_FILE)

train_df.to_parquet(TRAIN_FILE)

validation_df.to_parquet(
    VALIDATION_FILE
)

test_df.to_parquet(TEST_FILE)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("AAPL CLEAN DATASET CREATED")
print("=" * 60)

print(f"\nTotal dataset: {model_df.shape}")

print("\nDataset shapes:")
print(f"Train      : {train_df.shape}")
print(f"Validation : {validation_df.shape}")
print(f"Test       : {test_df.shape}")

print("\nDate ranges:")

print(
    f"Train      : "
    f"{train_df.index.min().date()} → "
    f"{train_df.index.max().date()}"
)

print(
    f"Validation : "
    f"{validation_df.index.min().date()} → "
    f"{validation_df.index.max().date()}"
)

print(
    f"Test       : "
    f"{test_df.index.min().date()} → "
    f"{test_df.index.max().date()}"
)

print("\nTarget distribution:")

print(
    model_df["target"]
    .value_counts(normalize=True)
    .sort_index()
)

print("\nSelected features:")

for index, column in enumerate(selected_columns, start=1):
    print(f"{index}. {column}")

print("\nSaved files:")
print(OUTPUT_FILE)
print(TRAIN_FILE)
print(VALIDATION_FILE)
print(TEST_FILE)

print("\nAAPL clean dataset preparation complete 🚀")