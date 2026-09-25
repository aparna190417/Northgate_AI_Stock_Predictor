import pandas as pd
from pathlib import Path

stocks = [
    "AAPL", "MSFT", "JPM", "XOM", "JNJ",
    "PG", "NVDA", "KO", "CAT", "HD"]

benchmark = "^GSPC"
vix = "^VIX"

# 1. Load raw data
RAW_DIR = Path("data/raw")

prices = pd.read_parquet(RAW_DIR / "prices.parquet")
macro = pd.read_parquet(RAW_DIR / "macro.parquet")

# 2. Basic information
print("=" * 60)
print("MARKET DATA INSPECTION")
print("=" * 60)

print("Shape:", prices.shape)

print("\nColumns:")
print(prices.columns)

print("\nDate range:")
print(prices.index.min(), "to", prices.index.max())

print("\nFirst 5 rows:")
print(prices.head())

# 3. Missing values
print("\nMissing values:")
print(prices.isna().sum())

# 4. Macro inspection
print("\n" + "=" * 60)
print("MACRO DATA INSPECTION")
print("=" * 60)

print("Shape:", macro.shape)

print("\nColumns:")
print(macro.columns)

print("\nDate range:")
print(macro.index.min(), "to", macro.index.max())

print("\nFirst 5 rows:")
print(macro.head())

print("\nMissing values:")
print(macro.isna().sum())

# 5. Detailed data quality check
print("\n" + "=" * 60)
print("XOM DETAILED CHECK")
print("=" * 60)

print("\nXOM missing values:")
print(prices.xs("XOM", level="Ticker", axis=1).isna().sum())

print("\nXOM first 5 rows:")
print(prices.xs("XOM", level="Ticker", axis=1).head())

print("\nXOM last 5 rows:")
print(prices.xs("XOM", level="Ticker", axis=1).tail())


print("\n" + "=" * 60)
print("MACRO DETAILED CHECK")
print("=" * 60)

print("\nMacro valid values:")
print(macro.notna().sum())

for col in macro.columns:
    valid = macro[col].dropna()

    print(f"\n{col}")
    print("First valid date:", valid.index.min())
    print("Last valid date:", valid.index.max())
    print("Valid observations:", valid.shape[0])

# 6. Exact missing dates

print("\n" + "=" * 60)
print("EXACT MISSING DATES")
print("=" * 60)

xom_missing = prices.xs("XOM", level="Ticker", axis=1).isna().any(axis=1)

print("\nXOM missing dates:")
print(prices.index[xom_missing])

print("\nMissing rows per ticker:")

for ticker in stocks + [benchmark, vix]:
    ticker_data = prices.xs(ticker, level="Ticker", axis=1)
    missing_rows = ticker_data.isna().any(axis=1).sum()
    print(f"{ticker}: {missing_rows}")

# 7. Duplicate dates

print("\n" + "=" * 60)
print("DUPLICATE DATE CHECK")
print("=" * 60)

print("Duplicate market dates:", prices.index.duplicated().sum())
print("Duplicate macro dates:", macro.index.duplicated().sum())

# 8. Macro missing dates

print("\n" + "=" * 60)
print("MACRO MISSING DATE SUMMARY")
print("=" * 60)

for col in macro.columns:
    missing = macro[col].isna().sum()
    print(f"{col}: {missing} missing")    