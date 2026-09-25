import yfinance as yf
import pandas as pd

print("Downloading XOM separately...")

xom = yf.download(
    "XOM",
    start="2015-01-01",
    end="2026-09-05",
    auto_adjust=False,
    progress=False
)

print("\nXOM shape:", xom.shape)
print("\nXOM columns:")
print(xom.columns)

print("\nFirst 5 rows:")
print(xom.head())

print("\nLast 5 rows:")
print(xom.tail())

print("\nMissing values:")
print(xom.isna().sum())