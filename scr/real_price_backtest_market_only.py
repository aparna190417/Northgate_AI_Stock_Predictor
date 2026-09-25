from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"

TRAIN_PATH = DATA_DIR / "AAPL_5day_normalized_train.parquet"
VAL_PATH = DATA_DIR / "AAPL_5day_normalized_validation.parquet"
TEST_PATH = DATA_DIR / "AAPL_5day_normalized_test.parquet"
FEATURES_PATH = DATA_DIR / "features.parquet"

RESULTS_DIR = ROOT / "results" / "real_price_market_only"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

INITIAL_CAPITAL = 100000

BUY_THRESHOLD = 0.60

TRANSACTION_COST = 0.001

MINIMUM_TRAIN_YEARS = 5

HOLDING_DAYS = 5

TRADING_DAYS_PER_YEAR = 252

FEATURES = [
    "Features_market_return_1d",
    "Features_market_return_5d",
    "Features_market_return_20d",
    "Features_market_volatility_20d",
]


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("NORTHGATE AI — AAPL REAL-PRICE MARKET-ONLY BACKTEST")
print("=" * 80)

print("\nConfiguration:")
print(f"Initial capital       : ₹{INITIAL_CAPITAL:,.2f}")
print(f"BUY threshold         : {BUY_THRESHOLD:.2f}")
print(f"Transaction cost      : {TRANSACTION_COST:.2%}")
print(f"Holding period        : {HOLDING_DAYS} trading days")
print("Overlapping trades    : NO")
print("Walk-forward training : YES")


# ============================================================
# LOAD MODEL DATA
# ============================================================

train_part = pd.read_parquet(TRAIN_PATH)
val_part = pd.read_parquet(VAL_PATH)
test_part = pd.read_parquet(TEST_PATH)

model_df = pd.concat(
    [
        train_part,
        val_part,
        test_part,
    ],
    ignore_index=True,
)

model_df["date"] = pd.to_datetime(model_df["date"])

model_df = (
    model_df
    .sort_values("date")
    .reset_index(drop=True)
)

print("\n" + "=" * 80)
print("DATA")
print("=" * 80)

print(f"Train shape : {train_part.shape}")
print(f"Validation  : {val_part.shape}")
print(f"Test shape  : {test_part.shape}")
print(f"Combined    : {model_df.shape}")


# ============================================================
# FEATURE CHECK
# ============================================================

print("\n" + "=" * 80)
print("FEATURE CHECK")
print("=" * 80)

missing_features = [
    feature
    for feature in FEATURES
    if feature not in model_df.columns
]

if missing_features:
    print("\nMissing features:")

    for feature in missing_features:
        print(" -", feature)

    raise ValueError(
        "Required market features are missing."
    )

print(f"Selected features: {len(FEATURES)}")

for feature in FEATURES:
    print(" -", feature)


# ============================================================
# LOAD ACTUAL AAPL PRICE
# ============================================================

price_df = pd.read_parquet(FEATURES_PATH)

price_df = price_df.copy()

if isinstance(price_df.columns, pd.MultiIndex):

    price_df.columns = [
        "_".join(
            [
                str(level)
                for level in column
                if str(level) not in ["", "nan"]
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
    in ["date", "datetime", "timestamp"]
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


# ============================================================
# DETECT AAPL ADJUSTED CLOSE
# ============================================================

price_candidates = [
    column
    for column in price_df.columns
    if str(column).lower()
    in [
        "adj close_aapl",
        "adj_close_aapl",
        "adjclose_aapl",
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
        "AAPL adjusted close column not found."
    )

PRICE_COLUMN = price_candidates[0]

print("\nActual price column:")
print(" -", PRICE_COLUMN)


price_df = price_df[
    [
        "date",
        PRICE_COLUMN,
    ]
].copy()

price_df = price_df.rename(
    columns={
        PRICE_COLUMN: "aapl_adj_close"
    }
)

price_df["aapl_adj_close"] = pd.to_numeric(
    price_df["aapl_adj_close"],
    errors="coerce",
)

price_df = price_df.dropna(
    subset=[
        "date",
        "aapl_adj_close",
    ]
)

price_df = (
    price_df
    .sort_values("date")
    .drop_duplicates("date")
    .reset_index(drop=True)
)


# ============================================================
# MERGE
# ============================================================

df = model_df.merge(
    price_df,
    on="date",
    how="left",
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

df = df.dropna(
    subset=["aapl_adj_close"]
).reset_index(drop=True)


# ============================================================
# FUTURE 5-DAY PRICE
# ============================================================

df["future_5day_price"] = (
    df["aapl_adj_close"]
    .shift(-HOLDING_DAYS)
)

df["actual_5day_return"] = (
    df["future_5day_price"]
    / df["aapl_adj_close"]
) - 1


# ============================================================
# REMOVE FINAL ROWS WITHOUT FUTURE PRICE
# ============================================================

df = df.dropna(
    subset=[
        "actual_5day_return"
    ]
).reset_index(drop=True)


# ============================================================
# WALK-FORWARD PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("WALK-FORWARD PREDICTION")
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

    X_train = train_df[FEATURES]
    y_train = train_df["target"]

    X_test = test_df[FEATURES]

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    year_result = test_df[
        [
            "date",
            "aapl_adj_close",
            "future_5day_price",
            "actual_5day_return",
            "target",
        ]
    ].copy()

    year_result[
        "prediction_probability"
    ] = probabilities

    year_result["raw_signal"] = (
        probabilities >= BUY_THRESHOLD
    ).astype(int)

    prediction_parts.append(
        year_result
    )

    print(
        f"Year {test_year}"
        f" | rows={len(test_df)}"
        f" | raw BUY={year_result['raw_signal'].sum()}"
    )


if not prediction_parts:

    raise ValueError(
        "No walk-forward predictions generated."
    )


results = pd.concat(
    prediction_parts,
    ignore_index=True,
)

results = (
    results
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# NON-OVERLAPPING TRADING LOGIC
# ============================================================

print("\n" + "=" * 80)
print("NON-OVERLAPPING TRADE SIMULATION")
print("=" * 80)

results["signal"] = 0
results["trade_return"] = 0.0
results["transaction_cost"] = 0.0

next_available_index = 0

trade_records = []

for i in range(len(results)):

    # Skip dates inside an existing 5-day trade
    if i < next_available_index:
        continue

    probability = (
        results.loc[
            i,
            "prediction_probability"
        ]
    )

    if probability < BUY_THRESHOLD:
        continue

    entry_date = results.loc[i, "date"]

    entry_price = results.loc[
        i,
        "aapl_adj_close"
    ]

    exit_index = i + HOLDING_DAYS

    if exit_index >= len(results):
        break

    exit_date = results.loc[
        exit_index,
        "date"
    ]

    exit_price = results.loc[
        exit_index,
        "aapl_adj_close"
    ]

    gross_return = (
        exit_price / entry_price
    ) - 1

    cost = TRANSACTION_COST

    net_return = (
        gross_return - cost
    )

    results.loc[i, "signal"] = 1
    results.loc[i, "trade_return"] = net_return
    results.loc[i, "transaction_cost"] = cost

    trade_records.append(
        {
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "probability": probability,
            "gross_return": gross_return,
            "transaction_cost": cost,
            "net_return": net_return,
        }
    )

    # No overlapping trades
    next_available_index = exit_index + 1


trades_df = pd.DataFrame(
    trade_records
)


# ============================================================
# DAILY PORTFOLIO RETURNS
# ============================================================

results["strategy_daily_return"] = 0.0

for trade in trade_records:

    entry_date = trade["entry_date"]
    exit_date = trade["exit_date"]

    entry_index = results.index[
        results["date"] == entry_date
    ][0]

    exit_index = results.index[
        results["date"] == exit_date
    ][0]

    if exit_index > entry_index:

        entry_price = results.loc[
            entry_index,
            "aapl_adj_close"
        ]

        for j in range(
            entry_index,
            exit_index
        ):

            current_price = results.loc[
                j,
                "aapl_adj_close"
            ]

            next_price = results.loc[
                j + 1,
                "aapl_adj_close"
            ]

            daily_return = (
                next_price / current_price
            ) - 1

            results.loc[
                j,
                "strategy_daily_return"
            ] = daily_return

        # Apply transaction cost at entry
        results.loc[
            entry_index,
            "strategy_daily_return"
        ] -= TRANSACTION_COST


# ============================================================
# BUY & HOLD DAILY RETURNS
# ============================================================

results["buy_hold_daily_return"] = (
    results["aapl_adj_close"]
    .pct_change()
    .fillna(0)
)


# ============================================================
# EQUITY CURVES
# ============================================================

results["strategy_equity"] = (
    1
    + results["strategy_daily_return"]
).cumprod() * INITIAL_CAPITAL

results["buy_hold_equity"] = (
    1
    + results["buy_hold_daily_return"]
).cumprod() * INITIAL_CAPITAL


# ============================================================
# DRAWDOWN
# ============================================================

results["strategy_peak"] = (
    results["strategy_equity"]
    .cummax()
)

results["buy_hold_peak"] = (
    results["buy_hold_equity"]
    .cummax()
)

results["strategy_drawdown"] = (
    results["strategy_equity"]
    / results["strategy_peak"]
) - 1

results["buy_hold_drawdown"] = (
    results["buy_hold_equity"]
    / results["buy_hold_peak"]
) - 1


# ============================================================
# METRICS
# ============================================================

strategy_final = (
    results["strategy_equity"].iloc[-1]
)

buy_hold_final = (
    results["buy_hold_equity"].iloc[-1]
)

strategy_return = (
    strategy_final / INITIAL_CAPITAL
) - 1

buy_hold_return = (
    buy_hold_final / INITIAL_CAPITAL
) - 1

strategy_max_dd = (
    results["strategy_drawdown"].min()
)

buy_hold_max_dd = (
    results["buy_hold_drawdown"].min()
)

number_of_trades = len(trades_df)

if number_of_trades > 0:

    trade_win_rate = (
        trades_df["net_return"] > 0
    ).mean()

    average_trade_return = (
        trades_df["net_return"].mean()
    )

else:

    trade_win_rate = 0.0

    average_trade_return = 0.0


daily_returns = (
    results["strategy_daily_return"]
)

daily_std = daily_returns.std()

if daily_std > 0:

    sharpe = (
        daily_returns.mean()
        / daily_std
    ) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )

else:

    sharpe = 0.0


# ============================================================
# CAGR
# ============================================================

start_date = results["date"].min()
end_date = results["date"].max()

years_elapsed = (
    end_date - start_date
).days / 365.25

if years_elapsed > 0:

    strategy_cagr = (
        strategy_final / INITIAL_CAPITAL
    ) ** (
        1 / years_elapsed
    ) - 1

    buy_hold_cagr = (
        buy_hold_final / INITIAL_CAPITAL
    ) ** (
        1 / years_elapsed
    ) - 1

else:

    strategy_cagr = 0.0
    buy_hold_cagr = 0.0


# ============================================================
# YEARLY RESULTS
# ============================================================

yearly_results = []

for year, year_df in results.groupby(
    results["date"].dt.year
):

    year_trades = trades_df[
        trades_df["entry_date"].dt.year == year
    ] if not trades_df.empty else pd.DataFrame()

    yearly_results.append(
        {
            "year": year,
            "rows": len(year_df),
            "trades": len(year_trades),
            "strategy_final_equity": (
                year_df["strategy_equity"].iloc[-1]
            ),
            "buy_hold_final_equity": (
                year_df["buy_hold_equity"].iloc[-1]
            ),
            "strategy_max_drawdown": (
                year_df["strategy_drawdown"].min()
            ),
            "buy_hold_max_drawdown": (
                year_df["buy_hold_drawdown"].min()
            ),
        }
    )

yearly_df = pd.DataFrame(
    yearly_results
)


# ============================================================
# SAVE FILES
# ============================================================

predictions_path = (
    RESULTS_DIR
    / "market_only_real_price_predictions.csv"
)

trades_path = (
    RESULTS_DIR
    / "market_only_trade_log.csv"
)

yearly_path = (
    RESULTS_DIR
    / "market_only_yearly_results.csv"
)

summary_path = (
    RESULTS_DIR
    / "market_only_backtest_summary.csv"
)

results.to_csv(
    predictions_path,
    index=False,
)

trades_df.to_csv(
    trades_path,
    index=False,
)

yearly_df.to_csv(
    yearly_path,
    index=False,
)


summary_df = pd.DataFrame(
    [
        {
            "initial_capital": INITIAL_CAPITAL,
            "final_strategy_equity": strategy_final,
            "final_buy_hold_equity": buy_hold_final,
            "strategy_total_return": strategy_return,
            "buy_hold_total_return": buy_hold_return,
            "strategy_cagr": strategy_cagr,
            "buy_hold_cagr": buy_hold_cagr,
            "strategy_max_drawdown": strategy_max_dd,
            "buy_hold_max_drawdown": buy_hold_max_dd,
            "sharpe": sharpe,
            "number_of_trades": number_of_trades,
            "trade_win_rate": trade_win_rate,
            "average_trade_return": average_trade_return,
            "buy_threshold": BUY_THRESHOLD,
            "transaction_cost": TRANSACTION_COST,
            "holding_days": HOLDING_DAYS,
        }
    ]
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("REAL-PRICE BACKTEST SUMMARY")
print("=" * 80)

print(
    f"Initial capital:          ₹{INITIAL_CAPITAL:,.2f}"
)

print(
    f"Final strategy equity:    ₹{strategy_final:,.2f}"
)

print(
    f"Final buy & hold equity:  ₹{buy_hold_final:,.2f}"
)

print(
    f"Strategy total return:    {strategy_return:.2%}"
)

print(
    f"Buy & hold total return:  {buy_hold_return:.2%}"
)

print(
    f"Strategy CAGR:             {strategy_cagr:.2%}"
)

print(
    f"Buy & hold CAGR:           {buy_hold_cagr:.2%}"
)

print(
    f"Strategy max drawdown:    {strategy_max_dd:.2%}"
)

print(
    f"Buy & hold max drawdown:  {buy_hold_max_dd:.2%}"
)

print(
    f"Sharpe ratio:              {sharpe:.4f}"
)

print(
    f"Number of trades:          {number_of_trades}"
)

print(
    f"Trade win rate:            {trade_win_rate:.2%}"
)

print(
    f"Average trade return:      {average_trade_return:.4%}"
)

print(
    f"BUY threshold:             {BUY_THRESHOLD:.2f}"
)

print(
    f"Transaction cost:          {TRANSACTION_COST:.2%}"
)

print(
    f"Holding period:            {HOLDING_DAYS} days"
)


# ============================================================
# YEARLY SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("YEARLY RESULTS")
print("=" * 80)

if not yearly_df.empty:
    print(
        yearly_df.to_string(
            index=False
        )
    )


# ============================================================
# FILES
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    "Predictions:"
    f"\n{predictions_path}"
)

print(
    "\nTrade log:"
    f"\n{trades_path}"
)

print(
    "\nYearly results:"
    f"\n{yearly_path}"
)

print(
    "\nSummary:"
    f"\n{summary_path}"
)

print("\n" + "=" * 80)
print("REAL-PRICE MARKET-ONLY BACKTEST COMPLETED")
print("=" * 80)