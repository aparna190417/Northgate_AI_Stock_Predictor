import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

STOCK = "AAPL"

INPUT_FILE = Path("data/processed/features.parquet")

TRAIN_FILE = Path(f"data/processed/{STOCK}_train.parquet")
VAL_FILE = Path(f"data/processed/{STOCK}_validation.parquet")
TEST_FILE = Path(f"data/processed/{STOCK}_test.parquet")

# ============================================================
# LOAD DATA
# ============================================================

print("Loading feature dataset...")

df = pd.read_parquet(INPUT_FILE)

print(f"Full dataset shape: {df.shape}")

# ============================================================
# SELECT FEATURES AND TARGET
# ============================================================

feature_df = df["Features"].copy()
target_df = df["Target"].copy()

target_column = f"{STOCK}_next_direction"

if target_column not in target_df.columns:
    raise ValueError(f"Target column not found: {target_column}")

target = target_df[target_column].rename("target")

model_df = pd.concat(
    [feature_df, target],
    axis=1
)

# ============================================================
# CLEAN INFINITE VALUES
# ============================================================

model_df = model_df.replace(
    [np.inf, -np.inf],
    np.nan
)

# ============================================================
# REMOVE MISSING VALUES
# ============================================================

before = len(model_df)

model_df = model_df.dropna()

after = len(model_df)

print(f"Rows before cleaning: {before}")
print(f"Rows removed: {before - after}")
print(f"Rows after cleaning: {after}")

# ============================================================
# SORT BY DATE
# ============================================================

model_df = model_df.sort_index()

# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(model_df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = model_df.iloc[:train_end].copy()
val_df = model_df.iloc[train_end:val_end].copy()
test_df = model_df.iloc[val_end:].copy()

# ============================================================
# SAVE DATASETS
# ============================================================

train_df.to_parquet(TRAIN_FILE)
val_df.to_parquet(VAL_FILE)
test_df.to_parquet(TEST_FILE)

# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 60)
print("MODEL DATA PREPARATION COMPLETE")
print("=" * 60)

print(f"\nStock: {STOCK}")
print(f"Total samples: {len(model_df)}")

print("\nDataset shapes:")
print(f"Train      : {train_df.shape}")
print(f"Validation : {val_df.shape}")
print(f"Test       : {test_df.shape}")

print("\nDate ranges:")

print(
    f"Train      : "
    f"{train_df.index.min().date()} → {train_df.index.max().date()}"
)

print(
    f"Validation : "
    f"{val_df.index.min().date()} → {val_df.index.max().date()}"
)

print(
    f"Test       : "
    f"{test_df.index.min().date()} → {test_df.index.max().date()}"
)

print("\nTarget distribution:")
print(
    model_df["target"]
    .value_counts(normalize=True)
    .sort_index()
)

print("\nSaved files:")
print(TRAIN_FILE)
print(VAL_FILE)
print(TEST_FILE)

print("\nReady for model training 🚀")