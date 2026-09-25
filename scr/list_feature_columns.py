from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "features.parquet"

df = pd.read_parquet(DATA_FILE)

print("=" * 60)
print("ALL FEATURE COLUMN NAMES")
print("=" * 60)

if isinstance(df.columns, pd.MultiIndex):
    columns = [
        f"{level_0}_{level_1}"
        for level_0, level_1 in df.columns
    ]
else:
    columns = list(df.columns)

for index, column in enumerate(columns, start=1):
    print(f"{index:03d}. {column}")

print("\nTotal columns:", len(columns))