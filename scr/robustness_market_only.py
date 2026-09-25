from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    ROOT
    / "data"
    / "processed"
    / "AAPL_5day_normalized_train.parquet"
)

VAL_PATH = (
    ROOT
    / "data"
    / "processed"
    / "AAPL_5day_normalized_validation.parquet"
)

TEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "AAPL_5day_normalized_test.parquet"
)

FEATURES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "features.parquet"
)

RESULTS_DIR = (
    ROOT
    / "results"
    / "robustness_market_only"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

INITIAL_CAPITAL = 100000

HOLDING_PERIOD = 5

MINIMUM_TRAIN_YEARS = 5

THRESHOLDS = [
    0.55,
    0.60,
    0.65,
    0.70
]

TRANSACTION_COSTS = [
    0.000,
    0.001,
    0.002,
    0.005
]

MARKET_FEATURES = [
    "Features_market_return_1d",
    "Features_market_return_5d",
    "Features_market_return_20d",
    "Features_market_volatility_20d"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("NORTHGATE AI — MARKET-ONLY ROBUSTNESS ANALYSIS")
print("=" * 80)

print()
print("Configuration:")
print(f"Initial capital       : ₹{INITIAL_CAPITAL:,.2f}")
print(f"Holding period        : {HOLDING_PERIOD} trading days")
print(f"Walk-forward training : YES")
print("Overlapping trades    : NO")

print()
print("Thresholds:")
for threshold in THRESHOLDS:
    print(f" - {threshold:.2f}")

print()
print("Transaction costs:")
for cost in TRANSACTION_COSTS:
    print(f" - {cost:.2%}")


# ============================================================
# LOAD NORMALIZED DATA
# ============================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

train_part = pd.read_parquet(TRAIN_PATH)
val_part = pd.read_parquet(VAL_PATH)
test_part = pd.read_parquet(TEST_PATH)

print(f"Train shape : {train_part.shape}")
print(f"Validation  : {val_part.shape}")
print(f"Test shape  : {test_part.shape}")

model_df = pd.concat(
    [
        train_part,
        val_part,
        test_part
    ],
    ignore_index=True
)

model_df["date"] = pd.to_datetime(
    model_df["date"]
)

model_df = (
    model_df
    .sort_values("date")
    .reset_index(drop=True)
)

print(f"Combined    : {model_df.shape}")


# ============================================================
# FEATURE CHECK
# ============================================================

print("\n" + "=" * 80)
print("FEATURE CHECK")
print("=" * 80)

missing_features = [
    feature
    for feature in MARKET_FEATURES
    if feature not in model_df.columns
]

if missing_features:
    raise ValueError(
        "Missing market features: "
        + str(missing_features)
    )

print(f"Selected features: {len(MARKET_FEATURES)}")

for feature in MARKET_FEATURES:
    print(f" - {feature}")


# ============================================================
# LOAD ACTUAL PRICE DATA
# ============================================================

print("\n" + "=" * 80)
print("LOADING ACTUAL AAPL PRICE")
print("=" * 80)

price_df = pd.read_parquet(
    FEATURES_PATH
)

price_df = price_df.copy()


# ------------------------------------------------------------
# Flatten MultiIndex columns
# ------------------------------------------------------------

if isinstance(
    price_df.columns,
    pd.MultiIndex
):

    price_df.columns = [
        "_".join(
            [
                str(level)
                for level in column
                if str(level) != ""
                and str(level) != "nan"
            ]
        )
        for column in price_df.columns
    ]


price_df = price_df.reset_index()


# ------------------------------------------------------------
# Find date column
# ------------------------------------------------------------

date_candidates = [
    column
    for column in price_df.columns
    if str(column).lower()
    in [
        "date",
        "datetime",
        "timestamp"
    ]
]

if date_candidates:

    price_date_column = date_candidates[0]

else:

    datetime_columns = [
        column
        for column in price_df.columns
        if pd.api.types.is_datetime64_any_dtype(
            price_df[column]
        )
    ]

    if not datetime_columns:
        raise ValueError(
            "Could not find date column in features.parquet."
        )

    price_date_column = datetime_columns[0]


price_df["date"] = pd.to_datetime(
    price_df[price_date_column]
)


# ------------------------------------------------------------
# Find AAPL adjusted close
# ------------------------------------------------------------

price_candidates = [
    column
    for column in price_df.columns
    if str(column).lower()
    in [
        "adj close_aapl",
        "adj_close_aapl",
        "adjclose_aapl"
    ]
]


if not price_candidates:

    price_candidates = [
        column
        for column in price_df.columns
        if (
            "adj close" in str(column).lower()
            and "aapl" in str(column).lower()
        )
    ]


if not price_candidates:
    raise ValueError(
        "AAPL adjusted close price column was not found."
    )


price_column = price_candidates[0]

print(f"Actual price column: {price_column}")


price_df = price_df[
    [
        "date",
        price_column
    ]
].copy()

price_df = price_df.rename(
    columns={
        price_column: "aapl_price"
    }
)

price_df["aapl_price"] = pd.to_numeric(
    price_df["aapl_price"],
    errors="coerce"
)

price_df = price_df.dropna(
    subset=[
        "date",
        "aapl_price"
    ]
)

price_df = (
    price_df
    .sort_values("date")
    .drop_duplicates("date")
    .reset_index(drop=True)
)


# ============================================================
# MERGE MODEL DATA WITH PRICE
# ============================================================

df = model_df.merge(
    price_df,
    on="date",
    how="left"
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

df = df.dropna(
    subset=["aapl_price"]
).reset_index(drop=True)


# ============================================================
# ACTUAL FUTURE 5-DAY RETURN
# ============================================================

df["future_price_5d"] = (
    df["aapl_price"]
    .shift(-HOLDING_PERIOD)
)

df["actual_5day_return"] = (
    df["future_price_5d"]
    / df["aapl_price"]
) - 1


# Remove rows where future price is unavailable

df = df.dropna(
    subset=["actual_5day_return"]
).reset_index(drop=True)


print(f"Backtest rows: {len(df)}")


# ============================================================
# WALK-FORWARD PROBABILITY GENERATION
# ============================================================

print("\n" + "=" * 80)
print("WALK-FORWARD MODEL PREDICTIONS")
print("=" * 80)


years = sorted(
    df["date"].dt.year.unique()
)

prediction_parts = []


for test_year in years:

    train_years = [
        year
        for year in years
        if year < test_year
    ]

    if len(train_years) < MINIMUM_TRAIN_YEARS:
        continue


    train_df = df[
        df["date"].dt.year < test_year
    ].copy()

    test_df = df[
        df["date"].dt.year == test_year
    ].copy()


    if train_df.empty or test_df.empty:
        continue


    X_train = train_df[
        MARKET_FEATURES
    ]

    y_train = train_df[
        "target"
    ]

    X_test = test_df[
        MARKET_FEATURES
    ]


    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


    model.fit(
        X_train,
        y_train
    )


    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    year_result = test_df[
        [
            "date",
            "aapl_price",
            "actual_5day_return",
            "target"
        ]
    ].copy()


    year_result[
        "prediction_probability"
    ] = probabilities


    prediction_parts.append(
        year_result
    )


    print(
        f"Year {test_year}"
        f" | rows={len(test_df)}"
    )


if not prediction_parts:

    raise ValueError(
        "No walk-forward predictions generated."
    )


predictions_df = pd.concat(
    prediction_parts,
    ignore_index=True
)

predictions_df = (
    predictions_df
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# BACKTEST FUNCTION
# ============================================================

def run_backtest(
    predictions,
    threshold,
    transaction_cost
):

    data = predictions.copy()

    data["signal"] = (
        data["prediction_probability"]
        >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # Non-overlapping trade simulation
    # --------------------------------------------------------

    capital = float(
        INITIAL_CAPITAL
    )

    equity_records = []

    trade_records = []

    next_available_index = 0


    for index in range(len(data)):

        if index < next_available_index:
            continue


        row = data.iloc[index]


        if row["signal"] != 1:
            continue


        entry_price = float(
            row["aapl_price"]
        )

        actual_return = float(
            row["actual_5day_return"]
        )


        net_return = (
            actual_return
            - transaction_cost
        )


        capital_before = capital

        capital = (
            capital
            * (1 + net_return)
        )


        exit_index = min(
            index + HOLDING_PERIOD,
            len(data) - 1
        )


        exit_row = data.iloc[
            exit_index
        ]


        trade_records.append(
            {
                "entry_date":
                    row["date"],

                "exit_date":
                    exit_row["date"],

                "entry_price":
                    entry_price,

                "exit_price":
                    float(
                        exit_row["aapl_price"]
                    ),

                "prediction_probability":
                    float(
                        row[
                            "prediction_probability"
                        ]
                    ),

                "actual_return":
                    actual_return,

                "transaction_cost":
                    transaction_cost,

                "net_return":
                    net_return,

                "capital_before":
                    capital_before,

                "capital_after":
                    capital,

                "threshold":
                    threshold,

                "transaction_cost_rate":
                    transaction_cost
            }
        )


        next_available_index = (
            index
            + HOLDING_PERIOD
        )


    trades_df = pd.DataFrame(
        trade_records
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    if trades_df.empty:

        return {
            "threshold":
                threshold,

            "transaction_cost":
                transaction_cost,

            "final_equity":
                INITIAL_CAPITAL,

            "total_return":
                0.0,

            "cagr":
                0.0,

            "max_drawdown":
                0.0,

            "sharpe":
                0.0,

            "trades":
                0,

            "win_rate":
                0.0,

            "average_trade_return":
                0.0,

            "median_trade_return":
                0.0
        }, trades_df


    trade_returns = trades_df[
        "net_return"
    ].astype(float)


    equity_curve = (
        INITIAL_CAPITAL
        * (1 + trade_returns)
        .cumprod()
    )


    running_max = (
        equity_curve
        .cummax()
    )


    drawdowns = (
        equity_curve
        / running_max
    ) - 1


    final_equity = float(
        equity_curve.iloc[-1]
    )


    total_return = (
        final_equity
        / INITIAL_CAPITAL
    ) - 1


    start_date = pd.to_datetime(
        trades_df[
            "entry_date"
        ].iloc[0]
    )

    end_date = pd.to_datetime(
        trades_df[
            "exit_date"
        ].iloc[-1]
    )


    years_elapsed = (
        end_date - start_date
    ).days / 365.25


    if years_elapsed > 0:

        cagr = (
            final_equity
            / INITIAL_CAPITAL
        ) ** (
            1 / years_elapsed
        ) - 1

    else:

        cagr = 0.0


    return_std = (
        trade_returns.std()
    )


    if (
        return_std != 0
        and not np.isnan(return_std)
    ):

        sharpe = (
            trade_returns.mean()
            / return_std
        ) * np.sqrt(
            252 / HOLDING_PERIOD
        )

    else:

        sharpe = 0.0


    win_rate = (
        trade_returns > 0
    ).mean()


    average_trade_return = (
        trade_returns.mean()
    )


    median_trade_return = (
        trade_returns.median()
    )


    metrics = {

        "threshold":
            threshold,

        "transaction_cost":
            transaction_cost,

        "final_equity":
            final_equity,

        "total_return":
            total_return,

        "cagr":
            cagr,

        "max_drawdown":
            float(
                drawdowns.min()
            ),

        "sharpe":
            float(sharpe),

        "trades":
            len(trades_df),

        "win_rate":
            float(win_rate),

        "average_trade_return":
            float(
                average_trade_return
            ),

        "median_trade_return":
            float(
                median_trade_return
            )
    }


    return metrics, trades_df


# ============================================================
# RUN ALL ROBUSTNESS EXPERIMENTS
# ============================================================

print("\n" + "=" * 80)
print("RUNNING ROBUSTNESS EXPERIMENTS")
print("=" * 80)


all_results = []

all_trade_logs = []


for threshold in THRESHOLDS:

    for transaction_cost in TRANSACTION_COSTS:

        print(
            f"\nThreshold={threshold:.2f}"
            f" | Cost={transaction_cost:.2%}"
        )


        metrics, trades = run_backtest(
            predictions_df,
            threshold,
            transaction_cost
        )


        all_results.append(
            metrics
        )


        if not trades.empty:

            all_trade_logs.append(
                trades
            )


        print(
            f"Trades={metrics['trades']}"
            f" | Return="
            f"{metrics['total_return']:.2%}"
            f" | CAGR="
            f"{metrics['cagr']:.2%}"
            f" | MaxDD="
            f"{metrics['max_drawdown']:.2%}"
            f" | Sharpe="
            f"{metrics['sharpe']:.4f}"
        )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


results_df = results_df.sort_values(
    [
        "transaction_cost",
        "threshold"
    ]
).reset_index(
    drop=True
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    RESULTS_DIR
    / "market_only_robustness_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# SAVE ALL TRADE LOGS
# ============================================================

if all_trade_logs:

    trade_log_df = pd.concat(
        all_trade_logs,
        ignore_index=True
    )

else:

    trade_log_df = pd.DataFrame()


trade_log_path = (
    RESULTS_DIR
    / "market_only_robustness_trade_log.csv"
)

trade_log_df.to_csv(
    trade_log_path,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ROBUSTNESS SUMMARY")
print("=" * 80)


display_columns = [
    "threshold",
    "transaction_cost",
    "trades",
    "final_equity",
    "total_return",
    "cagr",
    "max_drawdown",
    "sharpe",
    "win_rate",
    "average_trade_return"
]


summary_display = results_df[
    display_columns
].copy()


print(
    summary_display.to_string(
        index=False
    )
)


# ============================================================
# BEST RESULT BY EACH TRANSACTION COST
# ============================================================

print("\n" + "=" * 80)
print("BEST THRESHOLD AT EACH TRANSACTION COST")
print("=" * 80)


for cost in TRANSACTION_COSTS:

    subset = results_df[
        results_df[
            "transaction_cost"
        ] == cost
    ].copy()


    if subset.empty:
        continue


    best = subset.loc[
        subset["total_return"].idxmax()
    ]


    print(
        f"Cost {cost:.2%}"
        f" | Best threshold="
        f"{best['threshold']:.2f}"
        f" | Return="
        f"{best['total_return']:.2%}"
        f" | CAGR="
        f"{best['cagr']:.2%}"
        f" | MaxDD="
        f"{best['max_drawdown']:.2%}"
        f" | Sharpe="
        f"{best['sharpe']:.4f}"
        f" | Trades="
        f"{int(best['trades'])}"
    )


# ============================================================
# BEST RESULT OVERALL
# ============================================================

print("\n" + "=" * 80)
print("BEST OVERALL CONFIGURATION")
print("=" * 80)


best_overall = results_df.loc[
    results_df[
        "total_return"
    ].idxmax()
]


print(
    f"Threshold:              "
    f"{best_overall['threshold']:.2f}"
)

print(
    f"Transaction cost:       "
    f"{best_overall['transaction_cost']:.2%}"
)

print(
    f"Final equity:            "
    f"₹{best_overall['final_equity']:,.2f}"
)

print(
    f"Total return:            "
    f"{best_overall['total_return']:.2%}"
)

print(
    f"CAGR:                    "
    f"{best_overall['cagr']:.2%}"
)

print(
    f"Maximum drawdown:        "
    f"{best_overall['max_drawdown']:.2%}"
)

print(
    f"Sharpe ratio:            "
    f"{best_overall['sharpe']:.4f}"
)

print(
    f"Trades:                  "
    f"{int(best_overall['trades'])}"
)

print(
    f"Win rate:                "
    f"{best_overall['win_rate']:.2%}"
)


# ============================================================
# ROBUSTNESS CHECK
# ============================================================

print("\n" + "=" * 80)
print("ROBUSTNESS CHECK")
print("=" * 80)


# Check whether each threshold remains profitable
# across all transaction-cost assumptions.

for threshold in THRESHOLDS:

    subset = results_df[
        results_df[
            "threshold"
        ] == threshold
    ]


    profitable_count = (
        subset["total_return"] > 0
    ).sum()


    total_count = len(subset)


    print(
        f"Threshold {threshold:.2f}"
        f" | Profitable in "
        f"{profitable_count}/{total_count}"
        f" cost scenarios"
    )


# Check whether each cost scenario has
# at least one profitable configuration.

print()

for cost in TRANSACTION_COSTS:

    subset = results_df[
        results_df[
            "transaction_cost"
        ] == cost
    ]


    profitable = (
        subset["total_return"] > 0
    ).sum()


    print(
        f"Cost {cost:.2%}"
        f" | Profitable configurations: "
        f"{profitable}/{len(subset)}"
    )


# ============================================================
# FILES SAVED
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"Robustness results:"
)

print(
    results_path
)

print()

print(
    f"Trade log:"
)

print(
    trade_log_path
)

print("\n" + "=" * 80)
print("MARKET-ONLY ROBUSTNESS ANALYSIS COMPLETED")
print("=" * 80)