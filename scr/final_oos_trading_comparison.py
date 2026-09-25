"""
NORTHGATE AI — FINAL OOS TOP-5 VS 24-FEATURE TRADING COMPARISON

Final OOS trading analysis.

IMPORTANT:
- No model retraining.
- No threshold optimization.
- Threshold locked at 0.65.
- Holding period locked at 5 trading days.
- Transaction cost locked at 0.10% per side.
- Final test is used only for final OOS reporting.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PREDICTION_FILE = (
    BASE_DIR
    / "results"
    / "final_oos_top5_vs_24"
    / "final_test_row_level.csv"
)

PRICE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "prices_clean.parquet"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "final_oos_trading_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


THRESHOLD = 0.65
HOLDING_DAYS = 5
TRANSACTION_COST_PER_SIDE = 0.001
INITIAL_CAPITAL = 100000.0


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("NORTHGATE AI — FINAL OOS TOP-5 VS 24-FEATURE TRADING COMPARISON")
print("=" * 80)

print("""
Purpose:

1. Load FINAL OOS predictions.
2. Load actual FINAL TEST prices.
3. Align predictions and prices by date.
4. Apply identical locked trading rules.
5. Compare TOP-5 vs 24-feature performance.
6. Produce overall and yearly trading statistics.

IMPORTANT:
- No threshold optimization.
- No production threshold selection.
- No model retraining.
- FINAL TEST is used only for this final OOS report.
""")

print("=" * 80)
print("LOCKED TRADING RULES")
print("=" * 80)

print(f"Threshold        : {THRESHOLD}")
print(f"Holding period   : {HOLDING_DAYS} trading days")
print(
    f"Transaction cost : "
    f"{TRANSACTION_COST_PER_SIDE * 100:.2f}% per side"
)
print(f"Initial capital  : ₹{INITIAL_CAPITAL:,.2f}")


# ============================================================
# 1. CHECK FILES
# ============================================================

print("\n" + "=" * 80)
print("1. CHECKING INPUT FILES")
print("=" * 80)

print(f"Prediction file:\n{PREDICTION_FILE}")
print(f"Price file:\n{PRICE_FILE}")

if not PREDICTION_FILE.exists():
    raise FileNotFoundError(
        f"Prediction file not found:\n{PREDICTION_FILE}"
    )

if not PRICE_FILE.exists():
    raise FileNotFoundError(
        f"Price file not found:\n{PRICE_FILE}"
    )

print("\nBoth input files found.")


# ============================================================
# 2. LOAD FINAL OOS PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("2. LOADING FINAL OOS PREDICTIONS")
print("=" * 80)

pred = pd.read_csv(PREDICTION_FILE)

print(f"Prediction shape: {pred.shape}")
print(
    f"Prediction columns:\n"
    f"{pred.columns.tolist()}"
)

required_prediction_columns = [
    "date",
    "actual_target",
    "top5_probability",
    "model24_probability",
]

missing = [
    c for c in required_prediction_columns
    if c not in pred.columns
]

if missing:
    raise ValueError(
        "Missing prediction columns:\n"
        + "\n".join(f" - {c}" for c in missing)
    )

pred["date"] = pd.to_datetime(
    pred["date"],
    errors="coerce"
)

if pred["date"].isna().any():
    raise ValueError(
        "Prediction file contains invalid dates."
    )

pred = pred.sort_values("date").reset_index(drop=True)


# ============================================================
# 3. LOAD PRICE DATA
# ============================================================

print("\n" + "=" * 80)
print("3. LOADING ACTUAL PRICE DATA")
print("=" * 80)

prices_raw = pd.read_parquet(PRICE_FILE)

print(f"Price shape: {prices_raw.shape}")

print(
    f"Price columns:\n"
    f"{prices_raw.columns.tolist()}"
)


# ============================================================
# 4. HANDLE MULTIINDEX PRICE DATA
# ============================================================

print("\n" + "=" * 80)
print("4. IDENTIFYING DATE AND AAPL PRICE COLUMNS")
print("=" * 80)


# ------------------------------------------------------------
# DATE
# ------------------------------------------------------------

if isinstance(prices_raw.index, pd.DatetimeIndex):

    prices_raw = prices_raw.reset_index()

    date_column = prices_raw.columns[0]

else:

    date_column = None

    for candidate in [
        "Date",
        "date",
        "datetime",
        "Datetime",
    ]:
        if candidate in prices_raw.columns:
            date_column = candidate
            break

    if date_column is None:
        raise ValueError(
            "Could not identify date column."
        )


print(f"Date column: {date_column}")


# ------------------------------------------------------------
# AAPL ADJUSTED CLOSE
# ------------------------------------------------------------

price_series = None


# Case 1: MultiIndex columns
if isinstance(prices_raw.columns, pd.MultiIndex):

    print("\nMultiIndex detected.")

    print(
        "\nSearching for ('Adj Close', 'AAPL')..."
    )

    if ("Adj Close", "AAPL") in prices_raw.columns:

        price_series = prices_raw[
            ("Adj Close", "AAPL")
        ]

        print(
            "Found AAPL adjusted close."
        )

    elif ("Close", "AAPL") in prices_raw.columns:

        price_series = prices_raw[
            ("Close", "AAPL")
        ]

        print(
            "WARNING: Adjusted Close not found."
        )

        print(
            "Using AAPL Close instead."
        )

    else:

        raise ValueError(
            "\nCould not find AAPL price in MultiIndex data.\n\n"
            "Available AAPL-related columns:\n"
            + "\n".join(
                f" - {c}"
                for c in prices_raw.columns
                if "AAPL" in str(c)
            )
        )


# Case 2: normal columns
else:

    possible_columns = [
        "AAPL_Adj Close",
        "AAPL_adj_close",
        "AAPL_Close",
        "AAPL_close",
        "Adj Close",
        "adj_close",
        "Close",
        "close",
    ]

    for candidate in possible_columns:

        if candidate in prices_raw.columns:

            price_series = prices_raw[
                candidate
            ]

            print(
                f"Found price column: {candidate}"
            )

            break


if price_series is None:

    raise ValueError(
        "\nCould not identify AAPL price column.\n\n"
        "Available columns:\n"
        + "\n".join(
            f" - {c}"
            for c in prices_raw.columns
        )
    )


# ============================================================
# 5. CREATE CLEAN PRICE TABLE
# ============================================================

prices = pd.DataFrame(
    {
        "date": prices_raw[date_column],
        "price": price_series,
    }
)

prices["date"] = pd.to_datetime(
    prices["date"],
    errors="coerce"
)

prices["price"] = pd.to_numeric(
    prices["price"],
    errors="coerce"
)

prices = prices.dropna(
    subset=["date", "price"]
)

prices = prices.sort_values(
    "date"
)

prices = prices.drop_duplicates(
    subset=["date"],
    keep="last"
)

print(
    f"\nClean price rows: {len(prices)}"
)

print(
    f"Price period: "
    f"{prices['date'].min().date()} → "
    f"{prices['date'].max().date()}"
)

print(
    f"AAPL first price: "
    f"{prices['price'].iloc[0]:.2f}"
)

print(
    f"AAPL last price: "
    f"{prices['price'].iloc[-1]:.2f}"
)


# ============================================================
# 6. ALIGN PREDICTIONS WITH PRICES
# ============================================================

print("\n" + "=" * 80)
print("5. ALIGNING FINAL OOS PREDICTIONS WITH PRICES")
print("=" * 80)

merged = pd.merge(
    pred,
    prices,
    on="date",
    how="left"
)

missing_prices = (
    merged["price"].isna().sum()
)

print(
    f"Prediction rows : {len(pred)}"
)

print(
    f"Rows with prices: "
    f"{merged['price'].notna().sum()}"
)

print(
    f"Missing prices  : "
    f"{missing_prices}"
)

if missing_prices > 0:

    print(
        "\nMissing dates:"
    )

    print(
        merged.loc[
            merged["price"].isna(),
            "date"
        ].head(20).to_string(
            index=False
        )
    )

    raise ValueError(
        f"\n{missing_prices} prediction rows "
        "have no matching price."
    )


merged = (
    merged
    .sort_values("date")
    .reset_index(drop=True)
)

print(
    f"\nAligned period: "
    f"{merged['date'].min().date()} → "
    f"{merged['date'].max().date()}"
)


# ============================================================
# 7. LOCKED SIGNALS
# ============================================================

print("\n" + "=" * 80)
print("6. APPLYING LOCKED SIGNAL RULE")
print("=" * 80)

merged["top5_signal"] = (
    merged["top5_probability"]
    >= THRESHOLD
)

merged["model24_signal"] = (
    merged["model24_probability"]
    >= THRESHOLD
)

print(
    f"TOP-5 signals     : "
    f"{merged['top5_signal'].sum()}"
)

print(
    f"24-feature signals: "
    f"{merged['model24_signal'].sum()}"
)


# ============================================================
# 8. TRADE GENERATOR
# ============================================================

def generate_trades(
    data,
    probability_column,
    model_name,
):

    trades = []

    n = len(data)

    i = 0

    while i < n:

        probability = float(
            data.iloc[i][probability_column]
        )

        if probability < THRESHOLD:

            i += 1
            continue

        exit_index = i + HOLDING_DAYS

        if exit_index >= n:
            break

        entry_date = (
            data.iloc[i]["date"]
        )

        exit_date = (
            data.iloc[exit_index]["date"]
        )

        entry_price = float(
            data.iloc[i]["price"]
        )

        exit_price = float(
            data.iloc[exit_index]["price"]
        )

        gross_return = (
            exit_price / entry_price
        ) - 1.0

        # Entry + exit transaction costs
        net_return = (
            (1.0 + gross_return)
            * (1.0 - TRANSACTION_COST_PER_SIDE)
            * (1.0 - TRANSACTION_COST_PER_SIDE)
        ) - 1.0

        trades.append(
            {
                "model": model_name,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "probability": probability,
                "gross_return": gross_return,
                "net_return": net_return,
                "year": entry_date.year,
            }
        )

        # Non-overlapping trades
        i = exit_index + 1

    return pd.DataFrame(trades)


# ============================================================
# 9. GENERATE TRADES
# ============================================================

print("\n" + "=" * 80)
print("7. GENERATING TRADES")
print("=" * 80)

trades_top5 = generate_trades(
    merged,
    "top5_probability",
    "TOP-5"
)

trades_24 = generate_trades(
    merged,
    "model24_probability",
    "24-feature"
)

print(
    f"TOP-5 completed trades      : "
    f"{len(trades_top5)}"
)

print(
    f"24-feature completed trades : "
    f"{len(trades_24)}"
)


# ============================================================
# 10. PERFORMANCE
# ============================================================

def calculate_performance(
    trades,
):

    if trades.empty:

        return {
            "trades": 0,
            "final_equity": INITIAL_CAPITAL,
            "total_return": 0.0,
            "cagr": np.nan,
            "max_drawdown": 0.0,
            "sharpe": np.nan,
            "win_rate": np.nan,
            "average_trade_return": np.nan,
            "median_trade_return": np.nan,
            "best_trade": np.nan,
            "worst_trade": np.nan,
        }

    returns = (
        trades["net_return"]
        .astype(float)
    )

    equity = (
        INITIAL_CAPITAL
        * (1.0 + returns).cumprod()
    )

    final_equity = float(
        equity.iloc[-1]
    )

    total_return = (
        final_equity
        / INITIAL_CAPITAL
    ) - 1.0

    start_date = pd.to_datetime(
        trades["entry_date"].iloc[0]
    )

    end_date = pd.to_datetime(
        trades["exit_date"].iloc[-1]
    )

    days = (
        end_date - start_date
    ).days

    years = days / 365.25

    if years > 0:

        cagr = (
            final_equity
            / INITIAL_CAPITAL
        ) ** (1.0 / years) - 1.0

    else:

        cagr = np.nan

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
    ) - 1.0

    max_drawdown = float(
        drawdown.min()
    )

    if returns.std(ddof=1) > 0:

        sharpe = (
            returns.mean()
            / returns.std(ddof=1)
        ) * np.sqrt(len(returns))

    else:

        sharpe = np.nan

    return {
        "trades": len(trades),
        "final_equity": final_equity,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "win_rate": (
            returns > 0
        ).mean(),
        "average_trade_return": returns.mean(),
        "median_trade_return": returns.median(),
        "best_trade": returns.max(),
        "worst_trade": returns.min(),
    }


# ============================================================
# 11. OVERALL PERFORMANCE
# ============================================================

print("\n" + "=" * 80)
print("8. OVERALL FINAL OOS TRADING PERFORMANCE")
print("=" * 80)

perf_top5 = calculate_performance(
    trades_top5
)

perf_24 = calculate_performance(
    trades_24
)

overall = pd.DataFrame(
    [
        {
            "model": "TOP-5",
            **perf_top5,
        },
        {
            "model": "24-feature",
            **perf_24,
        },
    ]
)

print(
    overall.to_string(
        index=False
    )
)


# ============================================================
# 12. YEARLY PERFORMANCE
# ============================================================

print("\n" + "=" * 80)
print("9. YEARLY FINAL OOS PERFORMANCE")
print("=" * 80)


def yearly_results(trades):

    if trades.empty:
        return pd.DataFrame()

    rows = []

    for year, group in trades.groupby(
        "year"
    ):

        returns = group[
            "net_return"
        ]

        rows.append(
            {
                "model": group[
                    "model"
                ].iloc[0],

                "year": year,

                "trades": len(group),

                "win_rate": (
                    returns > 0
                ).mean(),

                "mean_trade_return":
                    returns.mean(),

                "median_trade_return":
                    returns.median(),

                "best_trade":
                    returns.max(),

                "worst_trade":
                    returns.min(),
            }
        )

    return pd.DataFrame(rows)


yearly_top5 = yearly_results(
    trades_top5
)

yearly_24 = yearly_results(
    trades_24
)

yearly = pd.concat(
    [
        yearly_24,
        yearly_top5,
    ],
    ignore_index=True
)

print(
    yearly.to_string(
        index=False
    )
)


# ============================================================
# 13. SIGNAL OVERLAP
# ============================================================

print("\n" + "=" * 80)
print("10. SIGNAL OVERLAP")
print("=" * 80)

both = (
    merged["top5_signal"]
    & merged["model24_signal"]
).sum()

only_top5 = (
    merged["top5_signal"]
    & ~merged["model24_signal"]
).sum()

only_24 = (
    merged["model24_signal"]
    & ~merged["top5_signal"]
).sum()

neither = (
    ~merged["top5_signal"]
    & ~merged["model24_signal"]
).sum()

overlap = pd.DataFrame(
    {
        "category": [
            "both_signal",
            "only_top5",
            "only_24_feature",
            "neither",
        ],
        "rows": [
            both,
            only_top5,
            only_24,
            neither,
        ],
    }
)

print(
    overlap.to_string(
        index=False
    )
)


# ============================================================
# 14. PERFORMANCE DIFFERENCE
# ============================================================

print("\n" + "=" * 80)
print("11. PERFORMANCE DIFFERENCE")
print("=" * 80)

metrics = [
    "final_equity",
    "total_return",
    "cagr",
    "max_drawdown",
    "sharpe",
    "win_rate",
    "average_trade_return",
    "median_trade_return",
]

comparison_rows = []

for metric in metrics:

    v24 = perf_24[metric]
    v5 = perf_top5[metric]

    if (
        pd.notna(v24)
        and pd.notna(v5)
    ):
        difference = v5 - v24
    else:
        difference = np.nan

    comparison_rows.append(
        {
            "metric": metric,
            "24-feature": v24,
            "TOP-5": v5,
            "TOP5_minus_24": difference,
        }
    )

comparison = pd.DataFrame(
    comparison_rows
)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 15. SAVE
# ============================================================

print("\n" + "=" * 80)
print("12. SAVING RESULTS")
print("=" * 80)

files = {
    "overall":
        OUTPUT_DIR
        / "final_oos_overall_trading_comparison.csv",

    "yearly":
        OUTPUT_DIR
        / "final_oos_yearly_trading_comparison.csv",

    "difference":
        OUTPUT_DIR
        / "final_oos_performance_difference.csv",

    "overlap":
        OUTPUT_DIR
        / "final_oos_signal_overlap.csv",

    "top5_trades":
        OUTPUT_DIR
        / "final_oos_trades_top5.csv",

    "24_trades":
        OUTPUT_DIR
        / "final_oos_trades_24_feature.csv",

    "merged":
        OUTPUT_DIR
        / "final_oos_predictions_with_prices.csv",
}

overall.to_csv(
    files["overall"],
    index=False
)

yearly.to_csv(
    files["yearly"],
    index=False
)

comparison.to_csv(
    files["difference"],
    index=False
)

overlap.to_csv(
    files["overlap"],
    index=False
)

trades_top5.to_csv(
    files["top5_trades"],
    index=False
)

trades_24.to_csv(
    files["24_trades"],
    index=False
)

merged.to_csv(
    files["merged"],
    index=False
)


# ============================================================
# 16. FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("13. FINAL OOS TRADING RESULT")
print("=" * 80)

print("""
FINAL TEST
----------
This analysis uses the existing FINAL OOS predictions
and actual AAPL price data.

LOCKED RULES
------------
Threshold        : 0.65
Holding period   : 5 trading days
Transaction cost : 0.10% per side
Initial capital  : ₹100,000
""")


print("TOP-5")
print("-----")

print(
    f"Trades             : "
    f"{perf_top5['trades']}"
)

print(
    f"Final equity       : "
    f"₹{perf_top5['final_equity']:,.2f}"
)

print(
    f"Total return       : "
    f"{perf_top5['total_return'] * 100:.4f}%"
)

print(
    f"CAGR               : "
    f"{perf_top5['cagr'] * 100:.4f}%"
)

print(
    f"Max drawdown       : "
    f"{perf_top5['max_drawdown'] * 100:.4f}%"
)

print(
    f"Sharpe             : "
    f"{perf_top5['sharpe']:.4f}"
)

print(
    f"Win rate           : "
    f"{perf_top5['win_rate'] * 100:.4f}%"
)


print("\n24-FEATURE")
print("----------")

print(
    f"Trades             : "
    f"{perf_24['trades']}"
)

print(
    f"Final equity       : "
    f"₹{perf_24['final_equity']:,.2f}"
)

print(
    f"Total return       : "
    f"{perf_24['total_return'] * 100:.4f}%"
)

print(
    f"CAGR               : "
    f"{perf_24['cagr'] * 100:.4f}%"
)

print(
    f"Max drawdown       : "
    f"{perf_24['max_drawdown'] * 100:.4f}%"
)

print(
    f"Sharpe             : "
    f"{perf_24['sharpe']:.4f}"
)

print(
    f"Win rate           : "
    f"{perf_24['win_rate'] * 100:.4f}%"
)


print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

for path in files.values():
    print(path)


print("\n" + "=" * 80)
print("NORTHGATE AI — FINAL OOS TRADING COMPARISON COMPLETED")
print("=" * 80)