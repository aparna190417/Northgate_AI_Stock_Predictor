import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# NORTHGATE AI — TOP-5 VS 24-FEATURE TRADING COMPARISON
# ============================================================

print("=" * 80)
print("NORTHGATE AI — TOP-5 VS 24-FEATURE TRADING COMPARISON")
print("=" * 80)

print("""
Purpose:

Compare TOP-5 and existing 24-feature predictions
under the SAME locked trading rules.

Locked threshold : 0.65
Holding period   : 5 trading days
Transaction cost : 0.10%
Initial capital  : ₹100,000

IMPORTANT:
- No threshold optimization.
- No production threshold selection.
- Existing 24-feature predictions are reused.
- TOP-5 predictions from the previous statistical test are reused.
- This is diagnostic only.
""")

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

EXISTING_24_FILE = os.path.join(
    BASE_DIR,
    "results",
    "walk_forward_final",
    "walk_forward_predictions.csv"
)

TOP5_FILE = os.path.join(
    BASE_DIR,
    "results",
    "feature_set_statistical_significance_fast",
    "paired_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "top5_vs_24_trading_comparison"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# CONFIGURATION
# ============================================================

THRESHOLD = 0.65
HOLDING_DAYS = 5
TRANSACTION_COST = 0.001
INITIAL_CAPITAL = 100000.0

# ============================================================
# 1. LOAD FILES
# ============================================================

print("\n" + "=" * 80)
print("1. LOADING EXISTING PREDICTIONS")
print("=" * 80)

if not os.path.exists(EXISTING_24_FILE):
    raise FileNotFoundError(
        f"24-feature prediction file not found:\n{EXISTING_24_FILE}"
    )

if not os.path.exists(TOP5_FILE):
    raise FileNotFoundError(
        f"TOP-5 paired prediction file not found:\n{TOP5_FILE}"
    )

df24 = pd.read_csv(EXISTING_24_FILE)

top5 = pd.read_csv(TOP5_FILE)

df24["date"] = pd.to_datetime(df24["date"])
top5["date"] = pd.to_datetime(top5["date"])

df24 = df24.sort_values("date").reset_index(drop=True)
top5 = top5.sort_values("date").reset_index(drop=True)

print(f"24-feature rows : {len(df24):,}")
print(f"TOP-5 rows      : {len(top5):,}")

# ============================================================
# 2. MERGE
# ============================================================

print("\n" + "=" * 80)
print("2. MERGING PREDICTIONS")
print("=" * 80)

columns_24 = [
    "date",
    "target",
    "prediction_probability",
    "aapl_adj_close",
]

for c in columns_24:
    if c not in df24.columns:
        raise ValueError(
            f"Missing column in 24-feature predictions: {c}"
        )

required_top5 = [
    "date",
    "target",
    "prob_5",
]

for c in required_top5:
    if c not in top5.columns:
        raise ValueError(
            f"Missing column in TOP-5 predictions: {c}"
        )

merged = df24[
    columns_24
].merge(
    top5[
        ["date", "target", "prob_5"]
    ],
    on=["date", "target"],
    how="inner"
)

merged = merged.sort_values(
    "date"
).reset_index(drop=True)

print(f"Paired rows: {len(merged):,}")

if len(merged) == 0:
    raise ValueError(
        "No matching rows between TOP-5 and 24-feature predictions."
    )

# ============================================================
# 3. SIGNALS
# ============================================================

print("\n" + "=" * 80)
print("3. APPLYING LOCKED SIGNAL RULE")
print("=" * 80)

merged["signal_24"] = (
    merged["prediction_probability"] >= THRESHOLD
).astype(int)

merged["signal_5"] = (
    merged["prob_5"] >= THRESHOLD
).astype(int)

print(f"Locked threshold: {THRESHOLD}")

print(
    f"24-feature signals: "
    f"{merged['signal_24'].sum():,}"
)

print(
    f"TOP-5 signals     : "
    f"{merged['signal_5'].sum():,}"
)

# ============================================================
# 4. BUILD TRADES
# ============================================================

def build_trades(
    data,
    signal_column,
    probability_column,
    model_name
):

    rows = []

    dates = data["date"].tolist()
    prices = data["aapl_adj_close"].tolist()
    signals = data[signal_column].tolist()
    probabilities = data[probability_column].tolist()

    for i in range(len(data)):

        if signals[i] != 1:
            continue

        exit_index = i + HOLDING_DAYS

        if exit_index >= len(data):
            continue

        entry_price = prices[i]
        exit_price = prices[exit_index]

        if pd.isna(entry_price) or pd.isna(exit_price):
            continue

        gross_return = (
            exit_price / entry_price
        ) - 1.0

        net_return = (
            gross_return - TRANSACTION_COST
        )

        rows.append(
            {
                "model": model_name,
                "entry_date": dates[i],
                "exit_date": dates[exit_index],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "prediction_probability":
                    probabilities[i],
                "gross_return": gross_return,
                "transaction_cost":
                    TRANSACTION_COST,
                "net_return": net_return,
            }
        )

    trades = pd.DataFrame(rows)

    if len(trades) == 0:
        return trades

    trades["equity_multiplier"] = (
        1.0 + trades["net_return"]
    )

    equity = INITIAL_CAPITAL

    equity_values = []

    for multiplier in trades["equity_multiplier"]:
        equity *= multiplier
        equity_values.append(equity)

    trades["equity"] = equity_values

    return trades


# ============================================================
# 5. GENERATE TRADES
# ============================================================

print("\n" + "=" * 80)
print("4. GENERATING TRADES")
print("=" * 80)

trades_24 = build_trades(
    merged,
    "signal_24",
    "prediction_probability",
    "24-feature"
)

trades_5 = build_trades(
    merged,
    "signal_5",
    "prob_5",
    "TOP-5"
)

print(
    f"24-feature completed trades: "
    f"{len(trades_24):,}"
)

print(
    f"TOP-5 completed trades      : "
    f"{len(trades_5):,}"
)

# ============================================================
# 6. PERFORMANCE FUNCTION
# ============================================================

def calculate_metrics(
    trades,
    model_name
):

    if len(trades) == 0:

        return {
            "model": model_name,
            "signals": 0,
            "completed_trades": 0,
            "final_equity": INITIAL_CAPITAL,
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe": np.nan,
            "win_rate": np.nan,
            "average_trade_return": np.nan,
            "median_trade_return": np.nan,
            "best_trade": np.nan,
            "worst_trade": np.nan,
        }

    returns = trades["net_return"].astype(float)

    final_equity = trades["equity"].iloc[-1]

    total_return = (
        final_equity / INITIAL_CAPITAL
    ) - 1.0

    start_date = pd.to_datetime(
        trades["entry_date"]
    ).min()

    end_date = pd.to_datetime(
        trades["exit_date"]
    ).max()

    days = (
        end_date - start_date
    ).days

    years = days / 365.25

    if years > 0:
        cagr = (
            final_equity /
            INITIAL_CAPITAL
        ) ** (
            1 / years
        ) - 1
    else:
        cagr = np.nan

    equity_curve = trades["equity"]

    running_max = equity_curve.cummax()

    drawdown = (
        equity_curve /
        running_max
    ) - 1.0

    max_drawdown = drawdown.min()

    if len(returns) > 1 and returns.std(ddof=1) > 0:

        sharpe = (
            returns.mean() /
            returns.std(ddof=1)
        ) * np.sqrt(
            252 / HOLDING_DAYS
        )

    else:
        sharpe = np.nan

    win_rate = (
        returns > 0
    ).mean()

    return {
        "model": model_name,
        "signals": int(
            len(merged[
                merged[
                    "signal_24"
                    if model_name == "24-feature"
                    else "signal_5"
                ] == 1
            ])
        ),
        "completed_trades": len(trades),
        "final_equity": final_equity,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "average_trade_return":
            returns.mean(),
        "median_trade_return":
            returns.median(),
        "best_trade":
            returns.max(),
        "worst_trade":
            returns.min(),
    }


# ============================================================
# 7. CALCULATE OVERALL METRICS
# ============================================================

print("\n" + "=" * 80)
print("5. OVERALL TRADING PERFORMANCE")
print("=" * 80)

metrics_24 = calculate_metrics(
    trades_24,
    "24-feature"
)

metrics_5 = calculate_metrics(
    trades_5,
    "TOP-5"
)

summary = pd.DataFrame(
    [
        metrics_24,
        metrics_5,
    ]
)

display_summary = summary.copy()

percentage_columns = [
    "total_return",
    "cagr",
    "max_drawdown",
    "win_rate",
    "average_trade_return",
    "median_trade_return",
    "best_trade",
    "worst_trade",
]

for col in percentage_columns:
    display_summary[col] = (
        display_summary[col] * 100
    )

print(
    display_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ============================================================
# 8. YEARLY PERFORMANCE
# ============================================================

print("\n" + "=" * 80)
print("6. YEARLY PERFORMANCE")
print("=" * 80)


def yearly_metrics(trades):

    if len(trades) == 0:
        return pd.DataFrame()

    temp = trades.copy()

    temp["entry_date"] = pd.to_datetime(
        temp["entry_date"]
    )

    temp["year"] = (
        temp["entry_date"].dt.year
    )

    results = []

    for year, group in temp.groupby("year"):

        returns = group["net_return"]

        compounded = (
            (1 + returns).prod()
        ) - 1

        results.append(
            {
                "year": year,
                "trades": len(group),
                "win_rate":
                    (returns > 0).mean(),
                "mean_trade_return":
                    returns.mean(),
                "median_trade_return":
                    returns.median(),
                "strategy_return":
                    compounded,
            }
        )

    return pd.DataFrame(results)


yearly_24 = yearly_metrics(
    trades_24
)

yearly_5 = yearly_metrics(
    trades_5
)

if len(yearly_24):
    yearly_24["model"] = "24-feature"

if len(yearly_5):
    yearly_5["model"] = "TOP-5"

yearly = pd.concat(
    [
        yearly_24,
        yearly_5,
    ],
    ignore_index=True
)

yearly = yearly[
    [
        "model",
        "year",
        "trades",
        "win_rate",
        "mean_trade_return",
        "median_trade_return",
        "strategy_return",
    ]
]

print(
    yearly.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# ============================================================
# 9. SIGNAL OVERLAP
# ============================================================

print("\n" + "=" * 80)
print("7. SIGNAL OVERLAP")
print("=" * 80)

both_signal = (
    (merged["signal_24"] == 1) &
    (merged["signal_5"] == 1)
).sum()

only_24 = (
    (merged["signal_24"] == 1) &
    (merged["signal_5"] == 0)
).sum()

only_5 = (
    (merged["signal_24"] == 0) &
    (merged["signal_5"] == 1)
).sum()

neither = (
    (merged["signal_24"] == 0) &
    (merged["signal_5"] == 0)
).sum()

overlap = pd.DataFrame(
    [
        {
            "both_signal": both_signal,
            "only_24_feature": only_24,
            "only_top5": only_5,
            "neither": neither,
        }
    ]
)

print(
    overlap.to_string(
        index=False
    )
)

# ============================================================
# 10. DIFFERENCE METRICS
# ============================================================

print("\n" + "=" * 80)
print("8. PERFORMANCE DIFFERENCE")
print("=" * 80)

m24 = metrics_24
m5 = metrics_5

difference = pd.DataFrame(
    [
        {
            "metric": "Final Equity",
            "24-feature": m24["final_equity"],
            "TOP-5": m5["final_equity"],
            "TOP5_minus_24":
                m5["final_equity"]
                - m24["final_equity"],
        },
        {
            "metric": "Total Return",
            "24-feature": m24["total_return"],
            "TOP-5": m5["total_return"],
            "TOP5_minus_24":
                m5["total_return"]
                - m24["total_return"],
        },
        {
            "metric": "CAGR",
            "24-feature": m24["cagr"],
            "TOP-5": m5["cagr"],
            "TOP5_minus_24":
                m5["cagr"]
                - m24["cagr"],
        },
        {
            "metric": "Max Drawdown",
            "24-feature": m24["max_drawdown"],
            "TOP-5": m5["max_drawdown"],
            "TOP5_minus_24":
                m5["max_drawdown"]
                - m24["max_drawdown"],
        },
        {
            "metric": "Sharpe",
            "24-feature": m24["sharpe"],
            "TOP-5": m5["sharpe"],
            "TOP5_minus_24":
                m5["sharpe"]
                - m24["sharpe"],
        },
        {
            "metric": "Win Rate",
            "24-feature": m24["win_rate"],
            "TOP-5": m5["win_rate"],
            "TOP5_minus_24":
                m5["win_rate"]
                - m24["win_rate"],
        },
        {
            "metric": "Average Trade Return",
            "24-feature":
                m24["average_trade_return"],
            "TOP-5":
                m5["average_trade_return"],
            "TOP5_minus_24":
                m5["average_trade_return"]
                - m24["average_trade_return"],
        },
    ]
)

print(
    difference.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# ============================================================
# 11. SAVE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("9. SAVING RESULTS")
print("=" * 80)

summary_file = os.path.join(
    OUTPUT_DIR,
    "overall_trading_comparison.csv"
)

yearly_file = os.path.join(
    OUTPUT_DIR,
    "yearly_trading_comparison.csv"
)

difference_file = os.path.join(
    OUTPUT_DIR,
    "performance_difference.csv"
)

overlap_file = os.path.join(
    OUTPUT_DIR,
    "signal_overlap.csv"
)

trades_24_file = os.path.join(
    OUTPUT_DIR,
    "trades_24_feature.csv"
)

trades_5_file = os.path.join(
    OUTPUT_DIR,
    "trades_top5.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

yearly.to_csv(
    yearly_file,
    index=False
)

difference.to_csv(
    difference_file,
    index=False
)

overlap.to_csv(
    overlap_file,
    index=False
)

trades_24.to_csv(
    trades_24_file,
    index=False
)

trades_5.to_csv(
    trades_5_file,
    index=False
)

# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("10. FINAL DIAGNOSTIC SUMMARY")
print("=" * 80)

print(f"""
LOCKED RULES
-------------
Threshold       : {THRESHOLD}
Holding period   : {HOLDING_DAYS} trading days
Transaction cost : {TRANSACTION_COST * 100:.2f}%
Initial capital  : ₹{INITIAL_CAPITAL:,.2f}

24-FEATURE MODEL
----------------
Completed trades : {m24["completed_trades"]}
Final equity     : ₹{m24["final_equity"]:,.2f}
Total return     : {m24["total_return"] * 100:.4f}%
CAGR             : {m24["cagr"] * 100:.4f}%
Max drawdown     : {m24["max_drawdown"] * 100:.4f}%
Sharpe           : {m24["sharpe"]:.4f}
Win rate         : {m24["win_rate"] * 100:.4f}%

TOP-5 MODEL
-----------
Completed trades : {m5["completed_trades"]}
Final equity     : ₹{m5["final_equity"]:,.2f}
Total return     : {m5["total_return"] * 100:.4f}%
CAGR             : {m5["cagr"] * 100:.4f}%
Max drawdown     : {m5["max_drawdown"] * 100:.4f}%
Sharpe           : {m5["sharpe"]:.4f}
Win rate         : {m5["win_rate"] * 100:.4f}%

No threshold was optimized.
No production model was changed.
""")

print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(summary_file)
print(yearly_file)
print(difference_file)
print(overlap_file)
print(trades_24_file)
print(trades_5_file)

print("\n" + "=" * 80)
print("NORTHGATE AI — TRADING COMPARISON COMPLETED")
print("=" * 80)