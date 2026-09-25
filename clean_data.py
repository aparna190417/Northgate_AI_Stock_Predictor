import pandas as pd
from pathlib import Path

# PATHS

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# LOAD DATA

prices = pd.read_parquet(RAW_DIR / "prices.parquet")
macro = pd.read_parquet(RAW_DIR / "macro.parquet")

print("=" * 60)
print("DATA CLEANING")
print("=" * 60)

# 1. REMOVE DUPLICATE DATES

prices = prices[~prices.index.duplicated(keep="first")]
macro = macro[~macro.index.duplicated(keep="first")]

print("Duplicate market dates removed.")
print("Duplicate macro dates removed.")

# 2. REMOVE COMMON NON-TRADING DAY

holiday_date = pd.Timestamp("2026-05-25")

if holiday_date in prices.index:
    prices = prices.drop(index=holiday_date)
    print(f"Removed non-trading day: {holiday_date.date()}")

# 3. SORT DATES

prices = prices.sort_index()
macro = macro.sort_index()

# 4. ALIGN MACRO DATA TO MARKET DATES

macro_aligned = macro.reindex(prices.index)

macro_aligned["DGS10"] = macro_aligned["DGS10"].ffill()
macro_aligned["DGS3MO"] = macro_aligned["DGS3MO"].ffill()
macro_aligned["CPIAUCSL"] = macro_aligned["CPIAUCSL"].ffill()
macro_aligned["UNRATE"] = macro_aligned["UNRATE"].ffill()

# 5. COMBINE MARKET AND MACRO DATA

macro_aligned.columns = pd.MultiIndex.from_tuples(
    [("Macro", col) for col in macro_aligned.columns],
    names=["Price", "Ticker"])

combined = pd.concat(
    [prices, macro_aligned],
    axis=1)

# 6. FINAL MISSING VALUE CHECK

print("\nFinal market missing values:")
print(prices.isna().sum().sum())

print("\nFinal macro missing values:")
print(macro_aligned.isna().sum())

print("\nCombined data shape:")
print(combined.shape)

# 7. SAVE PROCESSED DATA

prices.to_parquet(PROCESSED_DIR / "prices_clean.parquet")
macro_aligned.to_parquet(PROCESSED_DIR / "macro_aligned.parquet")
combined.to_parquet(PROCESSED_DIR / "combined_clean.parquet")

print("\nProcessed files saved successfully.")

print("\nSaved files:")
print(PROCESSED_DIR / "prices_clean.parquet")
print(PROCESSED_DIR / "macro_aligned.parquet")
print(PROCESSED_DIR / "combined_clean.parquet")