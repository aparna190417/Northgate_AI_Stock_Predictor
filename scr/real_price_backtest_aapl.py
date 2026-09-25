from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# Paths
# --------------------------------------------------

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

RESULTS_DIR = ROOT / "results" / "real_price_backtest"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Settings
# --------------------------------------------------

INITIAL_CAPITAL = 100000

BUY_THRESHOLD = 0.50

TRANSACTION_COST = 0.001

MINIMUM_TRAIN_YEARS = 5

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------
# Load normalized data
# --------------------------------------------------

print("=" * 80)
print("REAL-PRICE AAPL TRADING BACKTEST")
print("=" * 80)

train_part = pd.read_parquet(TRAIN_PATH)
val_part = pd.read_parquet(VAL_PATH)
test_part = pd.read_parquet(TEST_PATH)

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


# --------------------------------------------------
# Load original features and actual AAPL price
# --------------------------------------------------

price_df = pd.read_parquet(
    FEATURES_PATH
)

price_df = price_df.copy()

if isinstance(price_df.columns, pd.MultiIndex):

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

# Detect date column
date_candidates = [
    column
    for column in price_df.columns
    if str(column).lower() in ["date", "datetime", "timestamp"]
]

if date_candidates:

    price_date_column = date_candidates[0]

else:

    possible_datetime_columns = [
        column
        for column in price_df.columns
        if pd.api.types.is_datetime64_any_dtype(
            price_df[column]
        )
    ]

    if possible_datetime_columns:
        price_date_column = possible_datetime_columns[0]
    else:
        raise ValueError(
            "Date column could not be found in features.parquet."
        )

price_df["date"] = pd.to_datetime(
    price_df[price_date_column]
)

# Detect flattened AAPL adjusted-close column
price_column_candidates = [
    column
    for column in price_df.columns
    if str(column).lower()
    in [
        "adj close_aapl",
        "adj_close_aapl",
        "adjclose_aapl"
    ]
]

if not price_column_candidates:

    price_column_candidates = [
        column
        for column in price_df.columns
        if "adj close" in str(column).lower()
        and "aapl" in str(column).lower()
    ]

if not price_column_candidates:
    raise ValueError(
        "AAPL adjusted close price column was not found."
    )

price_column = price_column_candidates[0]

price_df = price_df[
    [
        "date",
        price_column
    ]
].copy()

price_df = price_df.rename(
    columns={
        price_column: "aapl_adj_close"
    }
)

price_df["aapl_adj_close"] = pd.to_numeric(
    price_df["aapl_adj_close"],
    errors="coerce"
)

price_df = price_df.dropna(
    subset=["date", "aapl_adj_close"]
)

price_df = (
    price_df
    .sort_values("date")
    .drop_duplicates("date")
    .reset_index(drop=True)
)


# --------------------------------------------------
# Merge model data with actual price
# --------------------------------------------------

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
    subset=["aapl_adj_close"]
).reset_index(drop=True)


# --------------------------------------------------
# Load selected features
# --------------------------------------------------

importance_df = pd.read_csv(
    IMPORTANCE_PATH
)

all_features = [
    column
    for column in model_df.columns
    if column not in ["date", "target"]
]

ranked_features = [
    feature
    for feature in importance_df["feature"].tolist()
    if feature in all_features
]

selected_features = ranked_features[:10]

if not selected_features:
    raise ValueError(
        "No selected features were found."
    )


# --------------------------------------------------
# Calculate actual future 5-day return
# --------------------------------------------------

df["future_5day_price"] = (
    df["aapl_adj_close"]
    .shift(-5)
)

df["actual_5day_return"] = (
    df["future_5day_price"]
    / df["aapl_adj_close"]
) - 1

df = df.dropna(
    subset=["actual_5day_return"]
).reset_index(drop=True)


# --------------------------------------------------
# Output columns
# --------------------------------------------------

df["prediction_probability"] = np.nan
df["signal"] = 0


# --------------------------------------------------
# Walk-forward prediction
# --------------------------------------------------

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

    X_train = train_df[selected_features]
    y_train = train_df["target"]

    X_test = test_df[selected_features]

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

    signals = (
        probabilities >= BUY_THRESHOLD
    ).astype(int)

    year_result = test_df[
        [
            "date",
            "aapl_adj_close",
            "actual_5day_return",
            "target"
        ]
    ].copy()

    year_result[
        "prediction_probability"
    ] = probabilities

    year_result["signal"] = signals

    prediction_parts.append(
        year_result
    )

    print(
        f"Processed year: {test_year}"
        f" | Rows: {len(test_df)}"
        f" | BUY signals: {signals.sum()}"
    )


# --------------------------------------------------
# Combine predictions
# --------------------------------------------------

if not prediction_parts:
    raise ValueError(
        "No walk-forward predictions were generated."
    )

results_df = pd.concat(
    prediction_parts,
    ignore_index=True
)

results_df = (
    results_df
    .sort_values("date")
    .reset_index(drop=True)
)


# --------------------------------------------------
# Calculate actual strategy returns
# --------------------------------------------------

results_df["gross_strategy_return"] = np.where(
    results_df["signal"] == 1,
    results_df["actual_5day_return"],
    0.0
)

results_df["transaction_cost"] = np.where(
    results_df["signal"] == 1,
    TRANSACTION_COST,
    0.0
)

results_df["net_strategy_return"] = (
    results_df["gross_strategy_return"]
    - results_df["transaction_cost"]
)

results_df["buy_hold_return"] = (
    results_df["actual_5day_return"]
)


# --------------------------------------------------
# Calculate equity curves
# --------------------------------------------------

results_df["strategy_equity"] = (
    1 + results_df["net_strategy_return"]
).cumprod() * INITIAL_CAPITAL

results_df["buy_hold_equity"] = (
    1 + results_df["buy_hold_return"]
).cumprod() * INITIAL_CAPITAL


# --------------------------------------------------
# Calculate drawdowns
# --------------------------------------------------

strategy_peak = (
    results_df["strategy_equity"]
    .cummax()
)

buy_hold_peak = (
    results_df["buy_hold_equity"]
    .cummax()
)

results_df["strategy_drawdown"] = (
    results_df["strategy_equity"]
    / strategy_peak
) - 1

results_df["buy_hold_drawdown"] = (
    results_df["buy_hold_equity"]
    / buy_hold_peak
) - 1


# --------------------------------------------------
# Save predictions
# --------------------------------------------------

predictions_path = (
    RESULTS_DIR
    / "aapl_real_price_backtest_predictions.csv"
)

results_df.to_csv(
    predictions_path,
    index=False
)


# --------------------------------------------------
# Summary metrics
# --------------------------------------------------

strategy_final_equity = (
    results_df["strategy_equity"].iloc[-1]
)

buy_hold_final_equity = (
    results_df["buy_hold_equity"].iloc[-1]
)

strategy_total_return = (
    strategy_final_equity
    / INITIAL_CAPITAL
) - 1

buy_hold_total_return = (
    buy_hold_final_equity
    / INITIAL_CAPITAL
) - 1

strategy_max_drawdown = (
    results_df["strategy_drawdown"].min()
)

buy_hold_max_drawdown = (
    results_df["buy_hold_drawdown"].min()
)

buy_signals = (
    results_df["signal"] == 1
)

active_days = buy_signals.sum()

total_days = len(results_df)

if active_days > 0:

    signal_returns = results_df.loc[
        buy_signals,
        "actual_5day_return"
    ]

    win_rate = (
        signal_returns > 0
    ).mean()

    average_signal_return = (
        signal_returns.mean()
    )

else:

    win_rate = 0.0

    average_signal_return = 0.0


strategy_returns = (
    results_df["net_strategy_return"]
)

strategy_return_std = (
    strategy_returns.std()
)

if strategy_return_std != 0:

    annualized_sharpe = (
        strategy_returns.mean()
        / strategy_return_std
    ) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )

else:

    annualized_sharpe = 0.0


# --------------------------------------------------
# Print summary
# --------------------------------------------------

print("\n" + "=" * 80)
print("REAL-PRICE BACKTEST SUMMARY")
print("=" * 80)

print(
    f"Initial capital:              "
    f"₹{INITIAL_CAPITAL:,.2f}"
)

print(
    f"Final strategy equity:         "
    f"₹{strategy_final_equity:,.2f}"
)

print(
    f"Final buy-and-hold equity:     "
    f"₹{buy_hold_final_equity:,.2f}"
)

print(
    f"Strategy total return:         "
    f"{strategy_total_return:.2%}"
)

print(
    f"Buy-and-hold total return:     "
    f"{buy_hold_total_return:.2%}"
)

print(
    f"Strategy maximum drawdown:     "
    f"{strategy_max_drawdown:.2%}"
)

print(
    f"Buy-and-hold maximum drawdown: "
    f"{buy_hold_max_drawdown:.2%}"
)

print(
    f"Annualized Sharpe ratio:       "
    f"{annualized_sharpe:.4f}"
)

print(
    f"Active BUY signals:            "
    f"{active_days} / {total_days}"
)

print(
    f"Signal win rate:               "
    f"{win_rate:.2%}"
)

print(
    f"Average actual 5-day return:   "
    f"{average_signal_return:.4%}"
)

print(
    f"BUY threshold:                 "
    f"{BUY_THRESHOLD:.2f}"
)

print(
    f"Transaction cost:              "
    f"{TRANSACTION_COST:.2%}"
)

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"Saved predictions to: "
    f"{predictions_path}"
)

print("\nREAL-PRICE BACKTEST COMPLETED")