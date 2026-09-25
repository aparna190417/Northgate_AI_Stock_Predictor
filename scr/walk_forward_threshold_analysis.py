from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# NORTHGATE AI
# WALK-FORWARD THRESHOLD STABILITY ANALYSIS
#
# IMPORTANT:
# - No model retraining
# - No threshold selection
# - Uses EXISTING walk-forward predictions
# - Does NOT touch final untouched test
# - Diagnostic analysis only
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    ROOT
    / "results"
    / "walk_forward_final"
    / "walk_forward_predictions.csv"
)

RESULTS_DIR = (
    ROOT
    / "results"
    / "walk_forward_threshold_analysis"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

INITIAL_CAPITAL = 100000

THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]

TRANSACTION_COST = 0.001

HOLDING_DAYS = 5


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("NORTHGATE AI — WALK-FORWARD THRESHOLD STABILITY ANALYSIS")
print("=" * 80)

print("""
Purpose:
This analysis uses the EXISTING walk-forward predictions.

It does NOT:
- retrain the model
- change the locked threshold
- use the untouched final test
- select a new production threshold

It compares threshold behaviour only.
""")

print("Configuration:")
print(f"Initial capital       : ₹{INITIAL_CAPITAL:,.2f}")
print(f"Thresholds            : {THRESHOLDS}")
print(f"Transaction cost      : {TRANSACTION_COST:.2%}")
print(f"Holding period        : {HOLDING_DAYS} trading days")
print("Overlapping trades    : NO")


# ============================================================
# LOAD EXISTING PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("1. LOADING EXISTING WALK-FORWARD PREDICTIONS")
print("=" * 80)

if not PREDICTIONS_PATH.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n{PREDICTIONS_PATH}\n\n"
        "Run walk_forward_final.py first."
    )


df = pd.read_csv(
    PREDICTIONS_PATH
)

df["date"] = pd.to_datetime(
    df["date"]
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


print(f"Rows loaded: {len(df)}")

print(
    f"Period: "
    f"{df['date'].min().date()} "
    f"→ "
    f"{df['date'].max().date()}"
)


# ============================================================
# COLUMN CHECK
# ============================================================

required_columns = [
    "date",
    "prediction_probability",
    "aapl_adj_close"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


print("\nRequired columns found:")
for column in required_columns:
    print(f" - {column}")


# ============================================================
# YEAR
# ============================================================

df["year"] = (
    df["date"]
    .dt
    .year
)


# ============================================================
# BACKTEST FUNCTION
# ============================================================

def run_threshold_backtest(
    prediction_df,
    threshold,
    transaction_cost
):

    data = (
        prediction_df
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    data["signal"] = (
        data["prediction_probability"]
        >= threshold
    ).astype(int)

    trades = []

    next_available_index = 0

    for idx in range(len(data)):

        if idx < next_available_index:
            continue

        row = data.iloc[idx]

        if row["signal"] != 1:
            continue

        exit_idx = (
            idx
            + HOLDING_DAYS
        )

        if exit_idx >= len(data):
            continue

        exit_row = data.iloc[
            exit_idx
        ]

        entry_price = (
            row["aapl_adj_close"]
        )

        exit_price = (
            exit_row["aapl_adj_close"]
        )

        actual_return = (
            exit_price
            / entry_price
        ) - 1

        net_return = (
            actual_return
            - transaction_cost
        )

        trades.append(
            {
                "entry_date":
                    row["date"],

                "exit_date":
                    exit_row["date"],

                "entry_year":
                    row["year"],

                "entry_price":
                    entry_price,

                "exit_price":
                    exit_price,

                "prediction_probability":
                    row[
                        "prediction_probability"
                    ],

                "actual_return":
                    actual_return,

                "net_return":
                    net_return
            }
        )

        # No overlapping trades
        next_available_index = (
            exit_idx
        )


    if not trades:

        return {
            "trades": 0,
            "final_equity":
                INITIAL_CAPITAL,
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "average_trade_return": 0.0,
            "trade_log":
                pd.DataFrame()
        }


    trades_df = pd.DataFrame(
        trades
    )


    # ========================================================
    # EQUITY
    # ========================================================

    trades_df["equity"] = (
        INITIAL_CAPITAL
        *
        (
            1
            + trades_df[
                "net_return"
            ]
        )
        .cumprod()
    )


    equity = trades_df[
        "equity"
    ]


    final_equity = (
        equity.iloc[-1]
    )


    total_return = (
        final_equity
        / INITIAL_CAPITAL
    ) - 1


    # ========================================================
    # DRAWDOWN
    # ========================================================

    running_max = (
        equity
        .cummax()
    )

    drawdown = (
        equity
        / running_max
    ) - 1


    max_drawdown = (
        drawdown.min()
    )


    # ========================================================
    # CAGR
    # ========================================================

    start_date = (
        trades_df[
            "entry_date"
        ].iloc[0]
    )

    end_date = (
        trades_df[
            "exit_date"
        ].iloc[-1]
    )

    years = (
        end_date - start_date
    ).days / 365.25


    if years > 0:

        cagr = (
            final_equity
            / INITIAL_CAPITAL
        ) ** (
            1 / years
        ) - 1

    else:

        cagr = 0.0


    # ========================================================
    # SHARPE
    # ========================================================

    returns = trades_df[
        "net_return"
    ]


    if (
        len(returns) > 1
        and returns.std() != 0
    ):

        sharpe = (
            returns.mean()
            / returns.std()
        ) * np.sqrt(
            252 / HOLDING_DAYS
        )

    else:

        sharpe = 0.0


    # ========================================================
    # WIN RATE
    # ========================================================

    win_rate = (
        trades_df[
            "actual_return"
        ] > 0
    ).mean()


    average_trade_return = (
        trades_df[
            "net_return"
        ].mean()
    )


    return {
        "trades":
            len(trades_df),

        "final_equity":
            final_equity,

        "total_return":
            total_return,

        "cagr":
            cagr,

        "max_drawdown":
            max_drawdown,

        "sharpe":
            sharpe,

        "win_rate":
            win_rate,

        "average_trade_return":
            average_trade_return,

        "trade_log":
            trades_df
    }


# ============================================================
# OVERALL THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("2. OVERALL THRESHOLD COMPARISON")
print("=" * 80)


overall_results = []


for threshold in THRESHOLDS:

    result = run_threshold_backtest(
        df,
        threshold,
        TRANSACTION_COST
    )


    overall_results.append(
        {
            "threshold":
                threshold,

            "trades":
                result["trades"],

            "final_equity":
                result["final_equity"],

            "total_return":
                result["total_return"],

            "cagr":
                result["cagr"],

            "max_drawdown":
                result["max_drawdown"],

            "sharpe":
                result["sharpe"],

            "win_rate":
                result["win_rate"],

            "average_trade_return":
                result[
                    "average_trade_return"
                ]
        }
    )


    print(
        f"\nThreshold={threshold:.2f} | "
        f"Trades={result['trades']} | "
        f"Return={result['total_return']:.2%} | "
        f"CAGR={result['cagr']:.2%} | "
        f"MaxDD={result['max_drawdown']:.2%} | "
        f"Sharpe={result['sharpe']:.4f} | "
        f"WinRate={result['win_rate']:.2%}"
    )


overall_df = pd.DataFrame(
    overall_results
)


# ============================================================
# YEARLY ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("3. YEARLY THRESHOLD ANALYSIS")
print("=" * 80)


yearly_results = []


years = sorted(
    df["year"].unique()
)


for year in years:

    year_df = (
        df[
            df["year"] == year
        ]
        .copy()
        .reset_index(drop=True)
    )


    print(
        f"\nYEAR {year}"
    )


    for threshold in THRESHOLDS:

        result = run_threshold_backtest(
            year_df,
            threshold,
            TRANSACTION_COST
        )


        yearly_results.append(
            {
                "year":
                    year,

                "threshold":
                    threshold,

                "prediction_rows":
                    len(year_df),

                "trades":
                    result["trades"],

                "final_equity":
                    result["final_equity"],

                "total_return":
                    result["total_return"],

                "cagr":
                    result["cagr"],

                "max_drawdown":
                    result["max_drawdown"],

                "sharpe":
                    result["sharpe"],

                "win_rate":
                    result["win_rate"],

                "average_trade_return":
                    result[
                        "average_trade_return"
                    ]
            }
        )


        print(
            f"  T={threshold:.2f} | "
            f"Trades={result['trades']:>3} | "
            f"Return={result['total_return']:>8.2%} | "
            f"Sharpe={result['sharpe']:>7.4f}"
        )


yearly_df = pd.DataFrame(
    yearly_results
)


# ============================================================
# POSITIVE YEARS
# ============================================================

print("\n" + "=" * 80)
print("4. POSITIVE-YEAR STABILITY")
print("=" * 80)


stability_results = []


for threshold in THRESHOLDS:

    threshold_years = (
        yearly_df[
            yearly_df["threshold"]
            == threshold
        ]
    )


    positive_years = (
        threshold_years[
            threshold_years[
                "total_return"
            ] > 0
        ]
    )


    negative_years = (
        threshold_years[
            threshold_years[
                "total_return"
            ] < 0
        ]
    )


    zero_trade_years = (
        threshold_years[
            threshold_years[
                "trades"
            ] == 0
        ]
    )


    stability_results.append(
        {
            "threshold":
                threshold,

            "total_years":
                len(threshold_years),

            "positive_years":
                len(positive_years),

            "negative_years":
                len(negative_years),

            "zero_trade_years":
                len(zero_trade_years),

            "positive_year_rate":
                (
                    len(positive_years)
                    /
                    len(threshold_years)
                )
        }
    )


    print(
        f"Threshold={threshold:.2f} | "
        f"Positive years="
        f"{len(positive_years)}/"
        f"{len(threshold_years)} | "
        f"Negative years="
        f"{len(negative_years)} | "
        f"Zero-trade years="
        f"{len(zero_trade_years)}"
    )


stability_df = pd.DataFrame(
    stability_results
)


# ============================================================
# SIGNAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("5. SIGNAL FREQUENCY")
print("=" * 80)


signal_results = []


for threshold in THRESHOLDS:

    signals = (
        df[
            "prediction_probability"
        ]
        >= threshold
    )


    count = (
        signals.sum()
    )


    percentage = (
        count
        / len(df)
    )


    signal_results.append(
        {
            "threshold":
                threshold,

            "buy_signals":
                int(count),

            "signal_percentage":
                percentage
        }
    )


    print(
        f"Threshold={threshold:.2f} | "
        f"BUY signals={count} | "
        f"Signal rate={percentage:.2%}"
    )


signal_df = pd.DataFrame(
    signal_results
)


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("6. PROBABILITY DISTRIBUTION")
print("=" * 80)


print(
    df[
        "prediction_probability"
    ].describe()
)


# ============================================================
# SAVE OVERALL RESULTS
# ============================================================

overall_path = (
    RESULTS_DIR
    / "threshold_overall_results.csv"
)

overall_df.to_csv(
    overall_path,
    index=False
)


# ============================================================
# SAVE YEARLY RESULTS
# ============================================================

yearly_path = (
    RESULTS_DIR
    / "threshold_yearly_results.csv"
)

yearly_df.to_csv(
    yearly_path,
    index=False
)


# ============================================================
# SAVE STABILITY RESULTS
# ============================================================

stability_path = (
    RESULTS_DIR
    / "threshold_stability_results.csv"
)

stability_df.to_csv(
    stability_path,
    index=False
)


# ============================================================
# SAVE SIGNAL RESULTS
# ============================================================

signal_path = (
    RESULTS_DIR
    / "threshold_signal_frequency.csv"
)

signal_df.to_csv(
    signal_path,
    index=False
)


# ============================================================
# SAVE ALL TRADE LOGS
# ============================================================

all_trade_logs = []


for threshold in THRESHOLDS:

    result = run_threshold_backtest(
        df,
        threshold,
        TRANSACTION_COST
    )


    trade_log = result[
        "trade_log"
    ]


    if not trade_log.empty:

        trade_log = trade_log.copy()

        trade_log[
            "threshold"
        ] = threshold

        all_trade_logs.append(
            trade_log
        )


if all_trade_logs:

    trade_log_df = pd.concat(
        all_trade_logs,
        ignore_index=True
    )

else:

    trade_log_df = pd.DataFrame()


trade_path = (
    RESULTS_DIR
    / "threshold_trade_logs.csv"
)

trade_log_df.to_csv(
    trade_path,
    index=False
)


# ============================================================
# FINAL CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("7. ANALYSIS SUMMARY")
print("=" * 80)

print("""
This is a diagnostic comparison only.

The previously locked threshold remains:
0.65

This script does NOT replace that threshold.

Interpretation should focus on:
- number of trades
- positive/negative years
- drawdown
- Sharpe
- signal frequency
- stability across thresholds

Do NOT select a new production threshold from this
analysis alone.
""")


print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"Overall results:\n{overall_path}"
)

print(
    f"\nYearly results:\n{yearly_path}"
)

print(
    f"\nStability results:\n{stability_path}"
)

print(
    f"\nSignal frequency:\n{signal_path}"
)

print(
    f"\nTrade logs:\n{trade_path}"
)


print("\n" + "=" * 80)
print(
    "NORTHGATE AI — WALK-FORWARD THRESHOLD "
    "ANALYSIS COMPLETED"
)
print("=" * 80)