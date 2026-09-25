import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
from pathlib import Path

# 1. Project folders
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 2. Stock universe
stocks = [
    "AAPL", "MSFT", "JPM", "XOM", "JNJ",
    "PG", "NVDA", "KO", "CAT", "HD"]

benchmark = "^GSPC"
vix = "^VIX"

start = "2015-01-01"
end = pd.Timestamp.today().strftime("%Y-%m-%d")

# 3. Download market data
tickers = stocks + [benchmark, vix]

prices = yf.download(
    tickers,
    start=start,
    end=end,
    auto_adjust=False,
    group_by="column",
    progress=False)

print("Market data shape:", prices.shape)
print("Market data date range:")
print(prices.index.min(), "to", prices.index.max())

# 3A. Download XOM separately
print("\nDownloading XOM separately...")

xom = yf.download(
    "XOM",
    start=start,
    end=end,
    auto_adjust=False,
    progress=False)

xom.columns = xom.columns.droplevel("Ticker")

for col in xom.columns:
    prices[(col, "XOM")] = xom[col]

print("XOM merged successfully!")

# 4. Download macro data
macro = web.DataReader(
    ["DGS10", "DGS3MO", "CPIAUCSL", "UNRATE"],
    "fred",
    start,
    end)

print("\nMacro data shape:", macro.shape)
print("Macro data date range:")
print(macro.index.min(), "to", macro.index.max())

# 5. Save raw data
prices.to_parquet(RAW_DIR / "prices.parquet")
macro.to_parquet(RAW_DIR / "macro.parquet")

print("\n Dataset downloaded successfully!")
print("Saved inside:", RAW_DIR)