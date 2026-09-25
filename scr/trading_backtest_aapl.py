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

IMPORTANCE_PATH = (
    ROOT
    / "results"
    / "feature_analysis"
    / "combined_feature_importance.csv"
)

RESULTS_DIR = ROOT / "results" / "trading_backtest"
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


# --------------------------------------------------
# Load data
# --------------------------------------------------

print("=" * 80)
print("AAPL TRADING BACKTEST")
print("=" * 80)

train_part = pd.read_parquet(TRAIN_PATH)
val_part = pd.read_parquet(VAL_PATH)
test_part = pd.read_parquet(TEST_PATH)

df = pd.concat(
    [
        train_part,
        val_part,
        test_part
    ],
    ignore_index=True
)

df["date"] = pd.to_datetime(
    df["date"]
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

importance_df = pd.read_csv(
    IMPORTANCE_PATH
)

all_features = [
    column
    for column in train_part.columns
    if column not in ["date", "target"]
]

ranked_features = [
    feature
    for feature in importance_df["feature"].tolist()
    if feature in all_features
]

selected_features = ranked_features[:10]

print(f"Dataset shape: {df.shape}")
print(f"Selected features: {len(selected_features)}")

for feature in selected_features:
    print(f" - {feature}")


# --------------------------------------------------
# Prepare output columns
# --------------------------------------------------

df["prediction_probability"] = np.nan
df["signal"] = 0
df["strategy_return"] = 0.0
df["buy_hold_return"] = 0.0


# --------------------------------------------------
# Walk-forward prediction
# --------------------------------------------------

years = sorted(
    df["date"].dt.year.unique()
)

prediction_results = []

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

    year_predictions = test_df[
        [
            "date",
            "target"
        ]
    ].copy()

    year_predictions[
        "prediction_probability"
    ] = probabilities

    year_predictions["signal"] = signals

    prediction_results.append(
        year_predictions
    )

    print(
        f"Processed test year: {test_year} "
        f"| Rows: {len(test_df)} "
        f"| Buy signals: {signals.sum()}"
    )


# --------------------------------------------------
# Combine predictions
# --------------------------------------------------

if not prediction_results:
    raise ValueError(
        "No walk-forward predictions were generated."
    )

predictions_df = pd.concat(
    prediction_results,
    ignore_index=True
)

predictions_df = (
    predictions_df
    .sort_values("date")
    .reset_index(drop=True)
)


# --------------------------------------------------
# Calculate returns
# --------------------------------------------------

# The target represents the future 5-day direction.
# This backtest uses target as a simplified direction-return proxy:
# target 1 = +1 unit return
# target 0 = -1 unit return.
#
# This is not a real price-based backtest.
# It is a directional strategy evaluation.

predictions_df["market_return_proxy"] = np.where(
    predictions_df["target"] == 1,
    1.0,
    -1.0
)

predictions_df["strategy_return"] = np.where(
    predictions_df["signal"] == 1,
    predictions_df["market_return_proxy"],
    0.0
)

predictions_df["buy_hold_return"] = (
    predictions_df["market_return_proxy"]
)

predictions_df["strategy_return"] = (
    predictions_df["strategy_return"] / 100
)

predictions_df["buy_hold_return"] = (
    predictions_df["buy_hold_return"] / 100
)

predictions_df["strategy_return_after_cost"] = np.where(
    predictions_df["signal"] == 1,
    predictions_df["strategy_return"] - TRANSACTION_COST,
    0.0
)

predictions_df["strategy_equity"] = (
    1 + predictions_df[
        "strategy_return_after_cost"
    ]
).cumprod() * INITIAL_CAPITAL

predictions_df["buy_hold_equity"] = (
    1 + predictions_df[
        "buy_hold_return"
    ]
).cumprod() * INITIAL_CAPITAL


# --------------------------------------------------
# Drawdown calculation
# --------------------------------------------------

strategy_running_max = (
    predictions_df["strategy_equity"]
    .cummax()
)

buy_hold_running_max = (
    predictions_df["buy_hold_equity"]
    .cummax()
)

predictions_df["strategy_drawdown"] = (
    predictions_df["strategy_equity"]
    / strategy_running_max
) - 1

predictions_df["buy_hold_drawdown"] = (
    predictions_df["buy_hold_equity"]
    / buy_hold_running_max
) - 1


# --------------------------------------------------
# Save daily predictions
# --------------------------------------------------

predictions_path = (
    RESULTS_DIR
    / "aapl_backtest_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# --------------------------------------------------
# Summary metrics
# --------------------------------------------------

strategy_final_equity = (
    predictions_df["strategy_equity"].iloc[-1]
)

buy_hold_final_equity = (
    predictions_df["buy_hold_equity"].iloc[-1]
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
    predictions_df["strategy_drawdown"].min()
)

buy_hold_max_drawdown = (
    predictions_df["buy_hold_drawdown"].min()
)

strategy_active_days = (
    predictions_df["signal"] == 1
).sum()

total_days = len(predictions_df)

strategy_win_rate = (
    predictions_df.loc[
        predictions_df["signal"] == 1,
        "market_return_proxy"
    ] > 0
).mean()

strategy_average_return = (
    predictions_df.loc[
        predictions_df["signal"] == 1,
        "strategy_return_after_cost"
    ].mean()
)


# --------------------------------------------------
# Print summary
# --------------------------------------------------

print("\n" + "=" * 80)
print("BACKTEST SUMMARY")
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
    f"Active trading days:           "
    f"{strategy_active_days} / {total_days}"
)

print(
    f"Strategy win rate:             "
    f"{strategy_win_rate:.2%}"
)

print(
    f"Average return per BUY signal: "
    f"{strategy_average_return:.4%}"
)

print(
    f"Buy threshold:                 "
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
    f"Predictions saved to: "
    f"{predictions_path}"
)

print("\nBACKTEST COMPLETED")