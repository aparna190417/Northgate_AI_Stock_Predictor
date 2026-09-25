from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "features.parquet"

df = pd.read_parquet(DATA_FILE)

print("=" * 60)
print("FEATURES DATASET DETAILS")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nIndex name:")
print(df.index.name)

print("\nIndex type:")
print(type(df.index))

print("\nFirst 20 columns:")
for column in df.columns[:20]:
    print(column)

print("\nLast 20 columns:")
for column in df.columns[-20:]:
    print(column)

print("\nFirst 5 rows:")
print(df.head())

print("\nFeatures columns check complete.")