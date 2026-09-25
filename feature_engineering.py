import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "combined_clean.parquet"
OUTPUT_FILE = PROCESSED_DIR / "features.parquet"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_parquet(INPUT_FILE)

print(f"Original shape: {df.shape}")


# ============================================================
# PREPARE DATE INDEX
# ============================================================

if not isinstance(df.index, pd.DatetimeIndex):
    try:
        df.index = pd.to_datetime(df.index)
    except Exception as error:
        raise ValueError(
            "The dataset index could not be converted to datetime."
        ) from error

df = df.sort_index()

print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")


# ============================================================
# CHECK MULTI-INDEX COLUMNS
# ============================================================

if not isinstance(df.columns, pd.MultiIndex):
    raise ValueError(
        "Expected MultiIndex columns in combined_clean.parquet."
    )


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_price(ticker, column="Adj Close"):
    """
    Safely return a numeric price/volume series
    from the MultiIndex dataset.
    """

    required_column = (column, ticker)

    if required_column not in df.columns:
        raise KeyError(
            f"Required column not found: {required_column}"
        )

    return pd.to_numeric(
        df[required_column],
        errors="coerce"
    )


# ============================================================
# STOCKS AND MARKET VARIABLES
# ============================================================

stocks = [
    "AAPL",
    "MSFT",
    "JPM",
    "XOM",
    "JNJ",
    "PG",
    "NVDA",
    "KO",
    "CAT",
    "HD",
]

benchmark = "^GSPC"
vix = "^VIX"


# ============================================================
# 1. STOCK RETURNS
# ============================================================

print("\nCreating stock return features...")

for ticker in stocks:

    close = get_price(ticker, "Adj Close")

    df[("Features", f"{ticker}_return_1d")] = (
        close.pct_change()
    )

    df[("Features", f"{ticker}_return_5d")] = (
        close.pct_change(5)
    )

    df[("Features", f"{ticker}_return_20d")] = (
        close.pct_change(20)
    )

    df[("Features", f"{ticker}_return_60d")] = (
        close.pct_change(60)
    )


# ============================================================
# 2. LOG RETURNS
# ============================================================

print("Creating log-return features...")

for ticker in stocks:

    close = get_price(ticker, "Adj Close")

    df[("Features", f"{ticker}_log_return")] = (
        np.log(close / close.shift(1))
    )


# ============================================================
# 3. MOVING AVERAGES
# ============================================================

print("Creating moving-average features...")

for ticker in stocks:

    close = get_price(ticker, "Adj Close")

    df[("Features", f"{ticker}_ma_5")] = (
        close.rolling(window=5, min_periods=5).mean()
    )

    df[("Features", f"{ticker}_ma_20")] = (
        close.rolling(window=20, min_periods=20).mean()
    )

    df[("Features", f"{ticker}_ma_50")] = (
        close.rolling(window=50, min_periods=50).mean()
    )

    df[("Features", f"{ticker}_ma_200")] = (
        close.rolling(window=200, min_periods=200).mean()
    )


# ============================================================
# 4. PRICE VS MOVING AVERAGE
# ============================================================

print("Creating price-to-moving-average features...")

for ticker in stocks:

    close = get_price(ticker, "Adj Close")

    ma_20 = close.rolling(
        window=20,
        min_periods=20
    ).mean()

    ma_50 = close.rolling(
        window=50,
        min_periods=50
    ).mean()

    ma_200 = close.rolling(
        window=200,
        min_periods=200
    ).mean()

    df[("Features", f"{ticker}_price_to_ma20")] = (
        close / ma_20
    )

    df[("Features", f"{ticker}_price_to_ma50")] = (
        close / ma_50
    )

    df[("Features", f"{ticker}_price_to_ma200")] = (
        close / ma_200
    )


# ============================================================
# 5. VOLATILITY
# ============================================================

print("Creating volatility features...")

for ticker in stocks:

    returns = df[("Features", f"{ticker}_log_return")]

    df[("Features", f"{ticker}_volatility_5d")] = (
        returns.rolling(
            window=5,
            min_periods=5
        ).std() * np.sqrt(252)
    )

    df[("Features", f"{ticker}_volatility_20d")] = (
        returns.rolling(
            window=20,
            min_periods=20
        ).std() * np.sqrt(252)
    )

    df[("Features", f"{ticker}_volatility_60d")] = (
        returns.rolling(
            window=60,
            min_periods=60
        ).std() * np.sqrt(252)
    )


# ============================================================
# 6. HIGH-LOW INTRADAY RANGE
# ============================================================

print("Creating intraday-range features...")

for ticker in stocks:

    high = get_price(ticker, "High")
    low = get_price(ticker, "Low")
    close = get_price(ticker, "Adj Close")

    df[("Features", f"{ticker}_intraday_range")] = (
        (high - low) / close
    )


# ============================================================
# 7. VOLUME FEATURES
# ============================================================

print("Creating volume features...")

for ticker in stocks:

    volume = get_price(ticker, "Volume")

    df[("Features", f"{ticker}_volume_change")] = (
        volume.pct_change()
    )

    df[("Features", f"{ticker}_volume_ma20")] = (
        volume.rolling(
            window=20,
            min_periods=20
        ).mean()
    )

    df[("Features", f"{ticker}_volume_ratio")] = (
        volume / df[("Features", f"{ticker}_volume_ma20")]
    )


# ============================================================
# 8. MARKET FEATURES
# ============================================================

print("Creating market features...")

benchmark_close = get_price(
    benchmark,
    "Adj Close"
)

df[("Features", "market_return_1d")] = (
    benchmark_close.pct_change()
)

df[("Features", "market_return_5d")] = (
    benchmark_close.pct_change(5)
)

df[("Features", "market_return_20d")] = (
    benchmark_close.pct_change(20)
)

benchmark_log_return = np.log(
    benchmark_close / benchmark_close.shift(1)
)

df[("Features", "market_volatility_20d")] = (
    benchmark_log_return.rolling(
        window=20,
        min_periods=20
    ).std() * np.sqrt(252)
)


# ============================================================
# 9. VIX FEATURES
# ============================================================

print("Creating VIX features...")

vix_close = get_price(
    vix,
    "Close"
)

df[("Features", "vix_level")] = (
    vix_close
)

df[("Features", "vix_change_1d")] = (
    vix_close.pct_change()
)

df[("Features", "vix_change_5d")] = (
    vix_close.pct_change(5)
)


# ============================================================
# 10. MACRO FEATURES
# ============================================================

print("Creating macroeconomic features...")

macro_columns = [
    "DGS10",
    "DGS3MO",
    "CPIAUCSL",
    "UNRATE",
]

for column in macro_columns:

    required_column = ("Macro", column)

    if required_column not in df.columns:
        raise KeyError(
            f"Required macro column not found: {required_column}"
        )

    df[("Features", column)] = pd.to_numeric(
        df[required_column],
        errors="coerce"
    )


df[("Features", "yield_spread")] = (
    df[("Features", "DGS10")]
    - df[("Features", "DGS3MO")]
)

df[("Features", "cpi_change")] = (
    df[("Features", "CPIAUCSL")].pct_change(12)
)

df[("Features", "unemployment_change")] = (
    df[("Features", "UNRATE")].diff(12)
)


# ============================================================
# 11. CALENDAR FEATURES
# ============================================================

print("Creating calendar features...")

df[("Features", "day_of_week")] = (
    df.index.dayofweek
)

df[("Features", "month")] = (
    df.index.month
)

df[("Features", "quarter")] = (
    df.index.quarter
)

df[("Features", "day_of_month")] = (
    df.index.day
)


# ============================================================
# 12. CYCLICAL CALENDAR ENCODING
# ============================================================

print("Creating cyclical calendar features...")

df[("Features", "month_sin")] = (
    np.sin(
        2 * np.pi
        * df[("Features", "month")]
        / 12
    )
)

df[("Features", "month_cos")] = (
    np.cos(
        2 * np.pi
        * df[("Features", "month")]
        / 12
    )
)

df[("Features", "day_of_week_sin")] = (
    np.sin(
        2 * np.pi
        * df[("Features", "day_of_week")]
        / 5
    )
)

df[("Features", "day_of_week_cos")] = (
    np.cos(
        2 * np.pi
        * df[("Features", "day_of_week")]
        / 5
    )
)


# ============================================================
# 13. REMOVE INVALID VALUES
# ============================================================

print("Cleaning invalid values...")

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 14. CHECK FEATURE COLUMNS
# ============================================================

if "Features" not in df.columns.get_level_values(0):
    raise ValueError(
        "Features section was not created correctly."
    )

feature_columns = df["Features"].columns.tolist()

print(f"\nTotal feature count: {len(feature_columns)}")

print("\nFeature columns:")
for column in feature_columns:
    print(f" - {column}")


# ============================================================
# 15. CHECK MISSING VALUES
# ============================================================

missing_feature_values = (
    df["Features"]
    .isna()
    .sum()
    .sum()
)

print("\nTotal missing values in feature section:")
print(missing_feature_values)


# ============================================================
# 16. SAVE FEATURES
# ============================================================

df.to_parquet(
    OUTPUT_FILE,
    index=True
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Final dataset shape: {df.shape}")
print(f"Total features created: {len(feature_columns)}")
print(f"Features saved to: {OUTPUT_FILE}")

print("\nImportant note:")
print(
    "No target variable was created in this file. "
    "The AAPL 5-day future target will be created separately "
    "in create_aapl_5day_normalized_dataset.py."
)

print("\nFeature engineering completed.")