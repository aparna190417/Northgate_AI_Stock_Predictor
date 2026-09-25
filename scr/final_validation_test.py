from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# NORTHGATE AI
# FINAL VALIDATION + UNTOUCHED TEST
#
# IMPORTANT:
# - Threshold selected ONLY on validation
# - Test remains completely untouched for model/threshold choice
# - Real AAPL prices used
# - 5 trading-day non-overlapping trades
# - CAGR calculated over FULL TEST PERIOD
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
    / "final_validation_test"
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
print("NORTHGATE AI — FINAL VALIDATION + UNTOUCHED TEST")
print("=" * 80)

print("\nConfiguration:")
print(
    f"Initial capital       : ₹{INITIAL_CAPITAL:,.2f}"
)

print(
    f"Threshold candidates  : {THRESHOLDS}"
)

print(
    f"Transaction cost      : {TRANSACTION_COST:.2%}"
)

print(
    f"Holding period        : {HOLDING_DAYS} trading days"
)

print(
    "Overlapping trades    : NO"
)

print(
    "Threshold selection  : VALIDATION ONLY"
)

print(
    "Final test            : UNTOUCHED"
)


# ============================================================
# LOAD MODEL DATA
# ============================================================

train_df = pd.read_parquet(
    TRAIN_PATH
)

val_df = pd.read_parquet(
    VAL_PATH
)

test_df = pd.read_parquet(
    TEST_PATH
)


for data in [
    train_df,
    val_df,
    test_df
]:

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data.sort_values(
        "date",
        inplace=True
    )

    data.reset_index(
        drop=True,
        inplace=True
    )


print("\n" + "=" * 80)
print("DATA")
print("=" * 80)

print(
    f"Train shape      : {train_df.shape}"
)

print(
    f"Validation shape : {val_df.shape}"
)

print(
    f"Test shape       : {test_df.shape}"
)


# ============================================================
# FEATURE SELECTION
# ============================================================

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
        "No selected features found."
    )


print("\n" + "=" * 80)
print("FEATURE CHECK")
print("=" * 80)

print(
    f"Selected features: "
    f"{len(selected_features)}"
)

for feature in selected_features:

    print(
        f" - {feature}"
    )


# ============================================================
# LOAD ACTUAL AAPL PRICES
# ============================================================

print("\n" + "=" * 80)
print("LOADING ACTUAL AAPL PRICE")
print("=" * 80)


price_df = pd.read_parquet(
    FEATURES_PATH
).copy()


# ------------------------------------------------------------
# Flatten MultiIndex
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


# ------------------------------------------------------------
# Detect date
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
    price_df[
        price_date_column
    ]
)


# ------------------------------------------------------------
# Detect AAPL adjusted close
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
            "adj close"
            in str(column).lower()
        )
        and
        (
            "aapl"
            in str(column).lower()
        )
    ]


if not price_candidates:

    raise ValueError(
        "AAPL adjusted close column not found."
    )


price_column = (
    price_candidates[0]
)


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


price_df = price_df.dropna(
    subset=[
        "date",
        "aapl_adj_close"
    ]
)


price_df = (
    price_df
    .sort_values("date")
    .drop_duplicates("date")
    .reset_index(drop=True)
)


# ============================================================
# BACKTEST FUNCTION
# ============================================================

def run_backtest(
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
        data[
            "prediction_probability"
        ]
        >= threshold
    ).astype(int)


    trades = []


    next_available_index = 0


    # --------------------------------------------------------
    # Generate non-overlapping trades
    # --------------------------------------------------------

    for idx in range(
        len(data)
    ):

        if idx < next_available_index:

            continue


        row = data.iloc[idx]


        if row["signal"] != 1:

            continue


        exit_idx = (
            idx + HOLDING_DAYS
        )


        if exit_idx >= len(data):

            continue


        exit_row = (
            data.iloc[exit_idx]
        )


        entry_price = (
            row[
                "aapl_adj_close"
            ]
        )


        exit_price = (
            exit_row[
                "aapl_adj_close"
            ]
        )


        actual_return = (
            exit_price
            /
            entry_price
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


        # Next trade can begin AFTER exit
        next_available_index = (
            exit_idx + 1
        )


    trades_df = pd.DataFrame(
        trades
    )


    if trades_df.empty:

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
                trades_df
        }


    # --------------------------------------------------------
    # Trade equity
    # --------------------------------------------------------

    trades_df[
        "equity"
    ] = (
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


    final_equity = (
        trades_df[
            "equity"
        ].iloc[-1]
    )


    total_return = (
        final_equity
        /
        INITIAL_CAPITAL
    ) - 1


    # --------------------------------------------------------
    # FULL PERIOD CAGR
    #
    # IMPORTANT:
    # CAGR uses the complete prediction period,
    # NOT only the first/last trade.
    # --------------------------------------------------------

    full_start_date = (
        data["date"].min()
    )

    full_end_date = (
        data["date"].max()
    )


    full_years = (
        full_end_date
        -
        full_start_date
    ).days / 365.25


    if full_years > 0:

        cagr = (
            final_equity
            /
            INITIAL_CAPITAL
        ) ** (
            1 / full_years
        ) - 1

    else:

        cagr = 0.0


    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    running_max = (
        trades_df[
            "equity"
        ].cummax()
    )


    drawdown = (
        trades_df[
            "equity"
        ]
        /
        running_max
    ) - 1


    max_drawdown = (
        drawdown.min()
    )


    # --------------------------------------------------------
    # Sharpe
    # --------------------------------------------------------

    returns = (
        trades_df[
            "net_return"
        ]
    )


    if (
        len(returns) > 1
        and returns.std() > 0
    ):

        sharpe = (

            returns.mean()
            /
            returns.std()

        ) * np.sqrt(
            252 / HOLDING_DAYS
        )

    else:

        sharpe = 0.0


    # --------------------------------------------------------
    # Win rate
    # --------------------------------------------------------

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
# VALIDATION MODEL
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION MODEL")
print("=" * 80)

print(
    "Training ONLY on training data."
)

print(
    "Validation is used ONLY for threshold selection."
)


X_train = train_df[
    selected_features
]

y_train = train_df[
    "target"
]

X_val = val_df[
    selected_features
]


validation_model = RandomForestClassifier(

    n_estimators=500,

    max_depth=8,

    min_samples_leaf=10,

    max_features="sqrt",

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


validation_model.fit(
    X_train,
    y_train
)


validation_probabilities = (
    validation_model
    .predict_proba(
        X_val
    )[:, 1]
)


validation_predictions = (
    val_df[
        [
            "date",
            "target"
        ]
    ].copy()
)


validation_predictions[
    "prediction_probability"
] = validation_probabilities


validation_predictions = (
    validation_predictions
    .merge(

        price_df[
            [
                "date",
                "aapl_adj_close"
            ]
        ],

        on="date",

        how="inner"
    )
)


# ============================================================
# VALIDATION THRESHOLD TEST
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION THRESHOLD EXPERIMENT")
print("=" * 80)


validation_results = []


for threshold in THRESHOLDS:

    result = run_backtest(

        validation_predictions,

        threshold,

        TRANSACTION_COST
    )


    validation_results.append(

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

        f"Threshold={threshold:.2f} | "

        f"Trades={result['trades']} | "

        f"Return={result['total_return']:.2%} | "

        f"CAGR={result['cagr']:.2%} | "

        f"MaxDD={result['max_drawdown']:.2%} | "

        f"Sharpe={result['sharpe']:.4f} | "

        f"WinRate={result['win_rate']:.2%}"
    )


validation_results_df = pd.DataFrame(
    validation_results
)


# ============================================================
# LOCK THRESHOLD
# ============================================================

validation_results_df = (
    validation_results_df
    .sort_values(
        [
            "sharpe",
            "cagr",
            "total_return"
        ],
        ascending=False
    )
    .reset_index(drop=True)
)


selected_threshold = float(
    validation_results_df.iloc[0][
        "threshold"
    ]
)


print("\n" + "=" * 80)
print("THRESHOLD SELECTION")
print("=" * 80)

print(
    "Selection criterion:"
)

print(
    "Highest VALIDATION Sharpe ratio."
)

print(
    f"\nLOCKED THRESHOLD: "
    f"{selected_threshold:.2f}"
)

print(
    "\nThis threshold is now LOCKED."
)

print(
    "The final test will NOT be used "
    "to change it."
)


# ============================================================
# SAVE VALIDATION
# ============================================================

validation_path = (
    RESULTS_DIR
    / "validation_threshold_results.csv"
)


validation_results_df.to_csv(
    validation_path,
    index=False
)


# ============================================================
# FINAL MODEL
# ============================================================

print("\n" + "=" * 80)
print("FINAL MODEL TRAINING")
print("=" * 80)

print(
    "Training on TRAIN + VALIDATION."
)


combined_train = pd.concat(

    [
        train_df,
        val_df
    ],

    ignore_index=True
)


X_final_train = (
    combined_train[
        selected_features
    ]
)

y_final_train = (
    combined_train[
        "target"
    ]
)

X_test = (
    test_df[
        selected_features
    ]
)


final_model = RandomForestClassifier(

    n_estimators=500,

    max_depth=8,

    min_samples_leaf=10,

    max_features="sqrt",

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


final_model.fit(
    X_final_train,
    y_final_train
)


# ============================================================
# FINAL UNTOUCHED TEST
# ============================================================

print("\n" + "=" * 80)
print("FINAL UNTOUCHED TEST")
print("=" * 80)


test_probabilities = (
    final_model
    .predict_proba(
        X_test
    )[:, 1]
)


final_test_predictions = (
    test_df[
        [
            "date",
            "target"
        ]
    ].copy()
)


final_test_predictions[
    "prediction_probability"
] = test_probabilities


final_test_predictions = (
    final_test_predictions
    .merge(

        price_df[
            [
                "date",
                "aapl_adj_close"
            ]
        ],

        on="date",

        how="inner"
    )
)


print(
    f"Test period: "
    f"{final_test_predictions['date'].min().date()} "
    f"→ "
    f"{final_test_predictions['date'].max().date()}"
)


print(
    f"LOCKED threshold: "
    f"{selected_threshold:.2f}"
)


# ============================================================
# FINAL TEST BACKTEST
# ============================================================

final_result = run_backtest(

    final_test_predictions,

    selected_threshold,

    TRANSACTION_COST
)


trade_log = (
    final_result[
        "trade_log"
    ]
)


# ============================================================
# BUY & HOLD
# ============================================================

test_prices = (
    final_test_predictions
    .sort_values("date")
    .reset_index(drop=True)
)


buy_hold_return = (

    test_prices[
        "aapl_adj_close"
    ].iloc[-1]

    /

    test_prices[
        "aapl_adj_close"
    ].iloc[0]

) - 1


buy_hold_final_equity = (

    INITIAL_CAPITAL
    *
    (
        1
        +
        buy_hold_return
    )
)


test_start = (
    test_prices[
        "date"
    ].iloc[0]
)


test_end = (
    test_prices[
        "date"
    ].iloc[-1]
)


test_years = (
    test_end
    -
    test_start
).days / 365.25


if test_years > 0:

    buy_hold_cagr = (

        buy_hold_final_equity
        /
        INITIAL_CAPITAL

    ) ** (
        1 / test_years
    ) - 1

else:

    buy_hold_cagr = 0.0


# ============================================================
# FINAL TEST SIGNALS
# ============================================================

final_test_predictions[
    "signal"
] = (

    final_test_predictions[
        "prediction_probability"
    ]

    >= selected_threshold

).astype(int)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)


print(
    f"Locked threshold:       "
    f"{selected_threshold:.2f}"
)


print(
    f"Transaction cost:       "
    f"{TRANSACTION_COST:.2%}"
)


print(
    f"Number of trades:       "
    f"{final_result['trades']}"
)


print(
    f"Final strategy equity:  "
    f"₹{final_result['final_equity']:,.2f}"
)


print(
    f"Buy & hold equity:      "
    f"₹{buy_hold_final_equity:,.2f}"
)


print(
    f"Strategy total return:  "
    f"{final_result['total_return']:.2%}"
)


print(
    f"Buy & hold return:      "
    f"{buy_hold_return:.2%}"
)


print(
    f"Strategy CAGR:          "
    f"{final_result['cagr']:.2%}"
)


print(
    f"Buy & hold CAGR:        "
    f"{buy_hold_cagr:.2%}"
)


print(
    f"Strategy max drawdown:  "
    f"{final_result['max_drawdown']:.2%}"
)


print(
    f"Strategy Sharpe:        "
    f"{final_result['sharpe']:.4f}"
)


print(
    f"Trade win rate:         "
    f"{final_result['win_rate']:.2%}"
)


print(
    f"Average trade return:   "
    f"{final_result['average_trade_return']:.4%}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_path = (
    RESULTS_DIR
    / "final_test_predictions.csv"
)


final_test_predictions.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# SAVE TRADE LOG
# ============================================================

trade_path = (
    RESULTS_DIR
    / "final_test_trade_log.csv"
)


trade_log.to_csv(
    trade_path,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(

    [
        {

            "locked_threshold":
                selected_threshold,

            "transaction_cost":
                TRANSACTION_COST,

            "holding_days":
                HOLDING_DAYS,

            "test_start":
                test_start,

            "test_end":
                test_end,

            "test_years":
                test_years,

            "trades":
                final_result[
                    "trades"
                ],

            "final_strategy_equity":
                final_result[
                    "final_equity"
                ],

            "buy_hold_final_equity":
                buy_hold_final_equity,

            "strategy_total_return":
                final_result[
                    "total_return"
                ],

            "buy_hold_return":
                buy_hold_return,

            "strategy_cagr":
                final_result[
                    "cagr"
                ],

            "buy_hold_cagr":
                buy_hold_cagr,

            "strategy_max_drawdown":
                final_result[
                    "max_drawdown"
                ],

            "strategy_sharpe":
                final_result[
                    "sharpe"
                ],

            "win_rate":
                final_result[
                    "win_rate"
                ],

            "average_trade_return":
                final_result[
                    "average_trade_return"
                ]
        }
    ]
)


summary_path = (
    RESULTS_DIR
    / "final_test_summary.csv"
)


summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"Validation results:"
    f"\n{validation_path}"
)

print(
    f"\nFinal predictions:"
    f"\n{predictions_path}"
)

print(
    f"\nTrade log:"
    f"\n{trade_path}"
)

print(
    f"\nFinal summary:"
    f"\n{summary_path}"
)

print("\n" + "=" * 80)
print(
    "FINAL VALIDATION + UNTOUCHED TEST COMPLETED"
)
print("=" * 80)