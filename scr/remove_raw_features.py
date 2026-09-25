import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/aapl_5day_clean.parquet")
OUTPUT = Path("data/processed/aapl_5day_relative.parquet")

df = pd.read_parquet(INPUT)

# Raw absolute price/volume columns
raw_prefixes = [
    "Adj Close",
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
]

raw_cols = [
    c for c in df.columns
    if any(str(c).startswith(prefix + " ") for prefix in raw_prefixes)
]

print("=" * 70)
print("REMOVE RAW PRICE/VOLUME FEATURES")
print("=" * 70)

print(f"Original shape: {df.shape}")
print(f"Raw columns to remove: {len(raw_cols)}")

print("\nRemoving:")
for c in raw_cols:
    print(" -", c)

# Remove raw columns
df_relative = df.drop(columns=raw_cols)

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print("New shape:", df_relative.shape)
print("Columns:", len(df_relative.columns))

print("\nTarget present:", "target" in df_relative.columns)
print("Date present:", "date" in df_relative.columns)

feature_cols = [
    c for c in df_relative.columns
    if c not in ["date", "target"]
]

print("Model features:", len(feature_cols))

print("\nTarget distribution:")
print(df_relative["target"].value_counts().sort_index())

print("\nMissing values:")
missing = df_relative[feature_cols].isna().sum()
print("Features with NaN:", int((missing > 0).sum()))
print("Total NaN:", int(missing.sum()))

df_relative.to_parquet(OUTPUT, index=False)

print("\nSaved:")
print(OUTPUT.resolve())