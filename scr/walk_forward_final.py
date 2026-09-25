from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# NORTHGATE AI
# FINAL WALK-FORWARD OUT-OF-SAMPLE EVALUATION
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

IMPORTANCE_PATH = (
    ROOT
    / "results"
    / "feature_analysis"
    / "combined_feature_importance.csv"
)

RESULTS_DIR = (
    ROOT
    / "results"
    / "walk_forward_final"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

INITIAL_CAPITAL = 100000

LOCKED_THRESHOLD = 0.65

TRANSACTION_COST = 0.001

HOLDING_DAYS = 5

MINIMUM_TRAIN_ROWS = 1000

TEST_START_YEAR = 2020


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("NORTHGATE AI — FINAL WALK-FORWARD OUT-OF-SAMPLE EVALUATION")
print("=" * 80)

print("\nConfiguration:")
print(f"Initial capital       : ₹{INITIAL_CAPITAL:,.2f}")
print(f"Locked threshold      : {LOCKED_THRESHOLD:.2f}")
print(f"Transaction cost      : {TRANSACTION_COST:.2%}")
print(f"Holding period        : {HOLDING_DAYS} trading days")
print("Overlapping trades    : NO")
print("Walk-forward training : YES")
print("Threshold selection   : NOT performed here")
print("Threshold source      : Validation-only selection")
print()


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("1. LOADING DATA")
print("=" * 80)

train_df = pd.read_parquet(TRAIN_PATH)
val_df = pd.read_parquet(VAL_PATH)
test_df = pd.read_parquet(TEST_PATH)


def prepare_dataset(df):

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = (
        df
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    return df


train_df = prepare_dataset(train_df)
val_df = prepare_dataset(val_df)
test_df = prepare_dataset(test_df)


print(
    f"TRAIN      : {len(train_df)} rows | "
    f"{train_df['date'].min().date()} → "
    f"{train_df['date'].max().date()}"
)

print(
    f"VALIDATION : {len(val_df)} rows | "
    f"{val_df['date'].min().date()} → "
    f"{val_df['date'].max().date()}"
)

print(
    f"TEST       : {len(test_df)} rows | "
    f"{test_df['date'].min().date()} → "
    f"{test_df['date'].max().date()}"
)


# ============================================================
# FEATURE SELECTION
# ============================================================

print("\n" + "=" * 80)
print("2. FEATURE SELECTION")
print("=" * 80)

importance_df = pd.read_csv(
    IMPORTANCE_PATH
)

all_features = [
    column
    for column in train_df.columns
    if column not in [
        "date",
        "target"
    ]
]

ranked_features = [
    feature
    for feature in importance_df["feature"].tolist()
    if feature in all_features
]

selected_features = ranked_features[:10]


if not selected_features:

    raise ValueError(
        "No model features found."
    )


print(
    f"Selected features: "
    f"{len(selected_features)}"
)

for feature in selected_features:

    print(
        f" - {feature}"
    )


# ============================================================
# LOAD ACTUAL AAPL PRICE
# ============================================================

print("\n" + "=" * 80)
print("3. LOADING ACTUAL AAPL PRICE")
print("=" * 80)

price_df = pd.read_parquet(
    FEATURES_PATH
).copy()


# Flatten MultiIndex

if isinstance(
    price_df.columns,
    pd.MultiIndex
):

    price_df.columns = [

        "_".join(
            [
                str(level)
                for level in column
                if str(level)
                not in [
                    "",
                    "nan"
                ]
            ]
        )

        for column in price_df.columns
    ]


price_df = price_df.reset_index()


# Detect date

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

    price_date_column = (
        date_candidates[0]
    )

else:

    datetime_candidates = [
        column
        for column in price_df.columns
        if pd.api.types.is_datetime64_any_dtype(
            price_df[column]
        )
    ]

    if not datetime_candidates:

        raise ValueError(
            "Could not find date column."
        )

    price_date_column = (
        datetime_candidates[0]
    )


price_df["date"] = pd.to_datetime(
    price_df[price_date_column]
)


# Detect AAPL adjusted close

price_candidates = [

    column
    for column in price_df.columns

    if (
        "adj close"
        in str(column).lower()
        and
        "aapl"
        in str(column).lower()
    )
]


if not price_candidates:

    raise ValueError(
        "AAPL adjusted close column not found."
    )


price_column = price_candidates[0]


print(
    f"Actual price column: "
    f"{price_column}"
)


price_df = price_df[
    [
        "date",
        price_column
    ]
].copy()


price_df = price_df.rename(
    columns={
        price_column:
        "aapl_adj_close"
    }
)


price_df[
    "aapl_adj_close"
] = pd.to_numeric(
    price_df[
        "aapl_adj_close"
    ],
    errors="coerce"
)


price_df = (
    price_df
    .dropna(
        subset=[
            "date",
            "aapl_adj_close"
        ]
    )
    .sort_values("date")
    .drop_duplicates("date")
    .reset_index(drop=True)
)


# ============================================================
# COMBINE MODEL DATA
# ============================================================

print("\n" + "=" * 80)
print("4. PREPARING WALK-FORWARD DATA")
print("=" * 80)


combined = pd.concat(
    [
        train_df,
        val_df,
        test_df
    ],
    ignore_index=True
)


combined = (
    combined
    .sort_values("date")
    .drop_duplicates("date")
    .reset_index(drop=True)
)


combined = combined.merge(
    price_df,
    on="date",
    how="inner"
)


print(
    f"Combined rows: "
    f"{len(combined)}"
)

print(
    f"Combined period: "
    f"{combined['date'].min().date()} "
    f"→ "
    f"{combined['date'].max().date()}"
)


# ============================================================
# WALK-FORWARD PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("5. WALK-FORWARD PREDICTIONS")
print("=" * 80)

print(
    "Each year's model is trained only on "
    "data available before that year."
)


years = sorted(
    combined[
        combined["date"].dt.year
        >= TEST_START_YEAR
    ]["date"]
    .dt.year
    .unique()
)


all_predictions = []


for year in years:

    year_start = pd.Timestamp(
        f"{year}-01-01"
    )

    year_end = pd.Timestamp(
        f"{year}-12-31"
    )

    train_before_year = combined[
        combined["date"]
        < year_start
    ].copy()


    current_year = combined[
        (
            combined["date"]
            >= year_start
        )
        &
        (
            combined["date"]
            <= year_end
        )
    ].copy()


    if len(train_before_year) < MINIMUM_TRAIN_ROWS:

        print(
            f"Year {year} | "
            f"SKIPPED | "
            f"training rows="
            f"{len(train_before_year)}"
        )

        continue


    if current_year.empty:

        continue


    X_train = train_before_year[
        selected_features
    ]

    y_train = train_before_year[
        "target"
    ]

    X_current = current_year[
        selected_features
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


    probabilities = (
        model
        .predict_proba(
            X_current
        )[:, 1]
    )


    current_year[
        "prediction_probability"
    ] = probabilities


    all_predictions.append(
        current_year[
            [
                "date",
                "target",
                "prediction_probability",
                "aapl_adj_close"
            ]
        ]
    )


    buy_count = int(
        (
            probabilities
            >= LOCKED_THRESHOLD
        ).sum()
    )


    print(
        f"Year {year} | "
        f"train_rows={len(train_before_year)} | "
        f"test_rows={len(current_year)} | "
        f"BUY signals={buy_count}"
    )


if not all_predictions:

    raise ValueError(
        "No walk-forward predictions generated."
    )


predictions = pd.concat(
    all_predictions,
    ignore_index=True
)


predictions = (
    predictions
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# SIGNAL
# ============================================================

predictions[
    "signal"
] = (
    predictions[
        "prediction_probability"
    ]
    >= LOCKED_THRESHOLD
).astype(int)


print(
    f"\nTotal prediction rows: "
    f"{len(predictions)}"
)

print(
    f"Total BUY signals: "
    f"{predictions['signal'].sum()}"
)


# ============================================================
# TRADE SIMULATION
# ============================================================

print("\n" + "=" * 80)
print("6. NON-OVERLAPPING TRADE SIMULATION")
print("=" * 80)


trades = []


next_available_index = 0


for idx in range(
    len(predictions)
):

    if idx < next_available_index:

        continue


    row = predictions.iloc[idx]


    if row["signal"] != 1:

        continue


    exit_idx = (
        idx + HOLDING_DAYS
    )


    if exit_idx >= len(predictions):

        continue


    exit_row = (
        predictions
        .iloc[exit_idx]
    )


    entry_price = (
        row["aapl_adj_close"]
    )

    exit_price = (
        exit_row["aapl_adj_close"]
    )


    actual_return = (
        exit_price
        /
        entry_price
    ) - 1


    net_return = (
        actual_return
        - TRANSACTION_COST
    )


    trades.append(
        {
            "entry_date":
                row["date"],

            "exit_date":
                exit_row["date"],

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


    next_available_index = (
        exit_idx
    )


trades_df = pd.DataFrame(
    trades
)


print(
    f"Completed trades: "
    f"{len(trades_df)}"
)


# ============================================================
# STRATEGY METRICS
# ============================================================

print("\n" + "=" * 80)
print("7. STRATEGY METRICS")
print("=" * 80)


if trades_df.empty:

    print(
        "No completed trades."
    )

    strategy_final_equity = (
        INITIAL_CAPITAL
    )

    strategy_return = 0.0

    strategy_cagr = 0.0

    strategy_max_drawdown = 0.0

    strategy_sharpe = 0.0

    win_rate = 0.0

    average_trade_return = 0.0

    equity_curve = pd.Series(
        dtype=float
    )

else:

    equity_curve = (
        INITIAL_CAPITAL
        *
        (
            1
            +
            trades_df[
                "net_return"
            ]
        ).cumprod()
    )


    trades_df[
        "equity"
    ] = equity_curve


    strategy_final_equity = (
        equity_curve.iloc[-1]
    )


    strategy_return = (
        strategy_final_equity
        /
        INITIAL_CAPITAL
    ) - 1


    running_max = (
        equity_curve
        .cummax()
    )


    drawdown = (
        equity_curve
        /
        running_max
    ) - 1


    strategy_max_drawdown = (
        drawdown.min()
    )


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


    years_elapsed = (
        end_date
        -
        start_date
    ).days / 365.25


    if years_elapsed > 0:

        strategy_cagr = (
            strategy_final_equity
            /
            INITIAL_CAPITAL
        ) ** (
            1 / years_elapsed
        ) - 1

    else:

        strategy_cagr = 0.0


    returns = trades_df[
        "net_return"
    ]


    if (
        len(returns) > 1
        and returns.std() != 0
    ):

        strategy_sharpe = (
            returns.mean()
            /
            returns.std()
        ) * np.sqrt(
            252 / HOLDING_DAYS
        )

    else:

        strategy_sharpe = 0.0


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


print(
    f"Final strategy equity : "
    f"₹{strategy_final_equity:,.2f}"
)

print(
    f"Strategy return        : "
    f"{strategy_return:.2%}"
)

print(
    f"Strategy CAGR          : "
    f"{strategy_cagr:.2%}"
)

print(
    f"Strategy max drawdown  : "
    f"{strategy_max_drawdown:.2%}"
)

print(
    f"Strategy Sharpe        : "
    f"{strategy_sharpe:.4f}"
)

print(
    f"Trade win rate         : "
    f"{win_rate:.2%}"
)

print(
    f"Average trade return   : "
    f"{average_trade_return:.4%}"
)


# ============================================================
# BUY & HOLD
# ============================================================

print("\n" + "=" * 80)
print("8. BUY & HOLD COMPARISON")
print("=" * 80)


if len(predictions) > 1:

    first_price = (
        predictions[
            "aapl_adj_close"
        ].iloc[0]
    )

    last_price = (
        predictions[
            "aapl_adj_close"
        ].iloc[-1]
    )


    buy_hold_return = (
        last_price
        /
        first_price
    ) - 1


    buy_hold_final_equity = (
        INITIAL_CAPITAL
        *
        (1 + buy_hold_return)
    )


    total_years = (
        predictions[
            "date"
        ].iloc[-1]
        -
        predictions[
            "date"
        ].iloc[0]
    ).days / 365.25


    if total_years > 0:

        buy_hold_cagr = (
            buy_hold_final_equity
            /
            INITIAL_CAPITAL
        ) ** (
            1 / total_years
        ) - 1

    else:

        buy_hold_cagr = 0.0

else:

    buy_hold_return = 0.0

    buy_hold_final_equity = (
        INITIAL_CAPITAL
    )

    buy_hold_cagr = 0.0


print(
    f"Buy & hold equity     : "
    f"₹{buy_hold_final_equity:,.2f}"
)

print(
    f"Buy & hold return     : "
    f"{buy_hold_return:.2%}"
)

print(
    f"Buy & hold CAGR       : "
    f"{buy_hold_cagr:.2%}"
)


# ============================================================
# YEARLY ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("9. YEARLY WALK-FORWARD RESULTS")
print("=" * 80)


predictions[
    "year"
] = predictions[
    "date"
].dt.year


yearly_rows = []


for year, group in predictions.groupby(
    "year"
):

    year_buy_signals = int(
        group[
            "signal"
        ].sum()
    )


    year_trades = (
        trades_df[
            trades_df[
                "entry_date"
            ].dt.year
            == year
        ]
        if not trades_df.empty
        else pd.DataFrame()
    )


    if not year_trades.empty:

        year_return = (
            (
                1
                +
                year_trades[
                    "net_return"
                ]
            ).prod()
            - 1
        )


        year_win_rate = (
            year_trades[
                "actual_return"
            ] > 0
        ).mean()


    else:

        year_return = 0.0

        year_win_rate = 0.0


    yearly_rows.append(
        {
            "year":
                year,

            "prediction_rows":
                len(group),

            "buy_signals":
                year_buy_signals,

            "completed_trades":
                len(year_trades),

            "strategy_return":
                year_return,

            "win_rate":
                year_win_rate
        }
    )


yearly_df = pd.DataFrame(
    yearly_rows
)


print(
    yearly_df.to_string(
        index=False
    )
)


# ============================================================
# SIGNAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("10. PREDICTION DISTRIBUTION")
print("=" * 80)


print(
    f"Probability minimum : "
    f"{predictions['prediction_probability'].min():.6f}"
)

print(
    f"Probability maximum : "
    f"{predictions['prediction_probability'].max():.6f}"
)

print(
    f"Probability mean    : "
    f"{predictions['prediction_probability'].mean():.6f}"
)

print(
    f"Probability median  : "
    f"{predictions['prediction_probability'].median():.6f}"
)

print(
    f"Probability >= "
    f"{LOCKED_THRESHOLD:.2f}: "
    f"{predictions['signal'].sum()}"
)


# ============================================================
# ROBUSTNESS INTERPRETATION
# ============================================================

print("\n" + "=" * 80)
print("11. WALK-FORWARD INTERPRETATION")
print("=" * 80)


if len(trades_df) < 30:

    print(
        "WARNING: Fewer than 30 completed trades."
    )

    print(
        "Performance statistics have limited "
        "statistical reliability."
    )

else:

    print(
        "Completed trade count is "
        f"{len(trades_df)}."
    )

    print(
        "This provides substantially more "
        "observations than the final "
        "single-trade test."
    )


if strategy_return > 0:

    print(
        "Strategy produced a positive "
        "cumulative return over the "
        "walk-forward period."
    )

else:

    print(
        "Strategy produced a negative "
        "cumulative return over the "
        "walk-forward period."
    )


print(
    "\nIMPORTANT:"
)

print(
    "This evaluation does NOT prove "
    "future profitability."
)

print(
    "It is an out-of-sample historical "
    "robustness check."
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_path = (
    RESULTS_DIR
    / "walk_forward_predictions.csv"
)


predictions.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# SAVE TRADE LOG
# ============================================================

trade_path = (
    RESULTS_DIR
    / "walk_forward_trade_log.csv"
)


trades_df.to_csv(
    trade_path,
    index=False
)


# ============================================================
# SAVE YEARLY RESULTS
# ============================================================

yearly_path = (
    RESULTS_DIR
    / "walk_forward_yearly_results.csv"
)


yearly_df.to_csv(
    yearly_path,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "locked_threshold":
                LOCKED_THRESHOLD,

            "transaction_cost":
                TRANSACTION_COST,

            "holding_days":
                HOLDING_DAYS,

            "start_date":
                predictions[
                    "date"
                ].min(),

            "end_date":
                predictions[
                    "date"
                ].max(),

            "prediction_rows":
                len(predictions),

            "buy_signals":
                int(
                    predictions[
                        "signal"
                    ].sum()
                ),

            "completed_trades":
                len(trades_df),

            "final_strategy_equity":
                strategy_final_equity,

            "strategy_total_return":
                strategy_return,

            "strategy_cagr":
                strategy_cagr,

            "strategy_max_drawdown":
                strategy_max_drawdown,

            "strategy_sharpe":
                strategy_sharpe,

            "strategy_win_rate":
                win_rate,

            "average_trade_return":
                average_trade_return,

            "buy_hold_final_equity":
                buy_hold_final_equity,

            "buy_hold_return":
                buy_hold_return,

            "buy_hold_cagr":
                buy_hold_cagr
        }
    ]
)


summary_path = (
    RESULTS_DIR
    / "walk_forward_summary.csv"
)


summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("12. FILES SAVED")
print("=" * 80)

print(
    f"Predictions:\n"
    f"{predictions_path}"
)

print(
    f"\nTrade log:\n"
    f"{trade_path}"
)

print(
    f"\nYearly results:\n"
    f"{yearly_path}"
)

print(
    f"\nSummary:\n"
    f"{summary_path}"
)


print("\n" + "=" * 80)
print(
    "NORTHGATE AI — FINAL WALK-FORWARD "
    "EVALUATION COMPLETED"
)
print("=" * 80)