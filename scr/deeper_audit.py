from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# NORTHGATE AI
# DEEPER DATA / TARGET / LEAKAGE / BACKTEST AUDIT
#
# IMPORTANT:
# - Does NOT retrain the model
# - Does NOT change threshold
# - Does NOT modify existing results
# - Audit only
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

FINAL_PREDICTIONS_PATH = (
    ROOT
    / "results"
    / "final_validation_test"
    / "final_test_predictions.csv"
)

FINAL_TRADES_PATH = (
    ROOT
    / "results"
    / "final_validation_test"
    / "final_test_trade_log.csv"
)

FINAL_SUMMARY_PATH = (
    ROOT
    / "results"
    / "final_validation_test"
    / "final_test_summary.csv"
)


# ============================================================
# SETTINGS
# ============================================================

HOLDING_DAYS = 5
LOCKED_THRESHOLD = 0.65


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("NORTHGATE AI — DEEPER TECHNICAL AUDIT")
print("=" * 80)

print("""
Purpose:
This audit does NOT retrain the model.
This audit does NOT change the threshold.
This audit does NOT modify existing results.

It checks:
1. Target construction
2. Feature/date alignment
3. Potential future leakage
4. Price alignment
5. Final prediction behaviour
6. Why very few BUY signals occurred
7. Final trade consistency
""")


# ============================================================
# HELPER
# ============================================================

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def pass_fail(condition, pass_text, fail_text):
    if condition:
        print(f"PASS: {pass_text}")
    else:
        print(f"WARNING: {fail_text}")


# ============================================================
# 1. LOAD DATASETS
# ============================================================

section("1. LOADING DATASETS")

train = pd.read_parquet(TRAIN_PATH).copy()
validation = pd.read_parquet(VAL_PATH).copy()
test = pd.read_parquet(TEST_PATH).copy()

for df in [train, validation, test]:
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

print(
    f"TRAIN      : {len(train)} rows | "
    f"{train['date'].min().date()} → {train['date'].max().date()}"
)

print(
    f"VALIDATION : {len(validation)} rows | "
    f"{validation['date'].min().date()} → {validation['date'].max().date()}"
)

print(
    f"TEST       : {len(test)} rows | "
    f"{test['date'].min().date()} → {test['date'].max().date()}"
)


# ============================================================
# 2. BASIC DATA INTEGRITY
# ============================================================

section("2. BASIC DATA INTEGRITY")

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:

    duplicate_dates = df["date"].duplicated().sum()
    missing = df.isna().sum().sum()

    print(
        f"{name:<11}: "
        f"duplicate_dates={duplicate_dates} | "
        f"missing_values={missing}"
    )

    pass_fail(
        duplicate_dates == 0,
        f"{name} has no duplicate dates.",
        f"{name} contains duplicate dates."
    )

    pass_fail(
        missing == 0,
        f"{name} has no missing values.",
        f"{name} contains missing values."
    )


# ============================================================
# 3. DATE SEPARATION
# ============================================================

section("3. DATE SEPARATION")

print(f"Train end        : {train['date'].max().date()}")
print(f"Validation start : {validation['date'].min().date()}")
print(f"Validation end   : {validation['date'].max().date()}")
print(f"Test start       : {test['date'].min().date()}")

pass_fail(
    train["date"].max() < validation["date"].min(),
    "Train ends before validation.",
    "Train overlaps validation."
)

pass_fail(
    validation["date"].max() < test["date"].min(),
    "Validation ends before test.",
    "Validation overlaps test."
)


# ============================================================
# 4. TARGET DISTRIBUTION
# ============================================================

section("4. TARGET DISTRIBUTION")

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:

    print(f"\n{name}")

    print(
        df["target"]
        .value_counts()
        .sort_index()
    )

    print(
        f"Target mean: "
        f"{df['target'].mean():.6f}"
    )


# ============================================================
# 5. FEATURE LIST
# ============================================================

section("5. MODEL FEATURE CHECK")

feature_columns = [
    column
    for column in train.columns
    if column not in ["date", "target"]
]

print(
    f"Model feature count: "
    f"{len(feature_columns)}"
)

for feature in feature_columns:
    print(f" - {feature}")


# ============================================================
# 6. OBVIOUS FUTURE-LOOKING NAMES
# ============================================================

section("6. FUTURE-LOOKING FEATURE NAME CHECK")

future_keywords = [
    "future",
    "forward",
    "next",
    "target",
    "label",
    "lead",
    "lookahead",
    "future_return",
    "future_price"
]

suspicious_features = []

for feature in feature_columns:

    lower_name = str(feature).lower()

    for keyword in future_keywords:

        if keyword in lower_name:

            suspicious_features.append(
                feature
            )

            break


if suspicious_features:

    print("WARNING: Suspicious feature names found:")

    for feature in suspicious_features:
        print(f" - {feature}")

else:

    print(
        "PASS: No obvious future-looking "
        "feature names detected."
    )


# ============================================================
# 7. TARGET vs MARKET PRICE
# ============================================================

section("7. TARGET CONSTRUCTION INVESTIGATION")

print("""
The processed datasets contain:
- date
- target
- engineered features

This audit will compare target values against actual
future 5-trading-day AAPL adjusted-close direction.

NOTE:
A matching pattern strongly supports that the target is
based on future 5-day direction, but it cannot prove the
original target-generation code without that source script.
""")


# Load original features
raw = pd.read_parquet(FEATURES_PATH).copy()

print(
    f"\nOriginal features shape: "
    f"{raw.shape}"
)

# Flatten MultiIndex
if isinstance(raw.columns, pd.MultiIndex):

    raw.columns = [
        "_".join(
            [
                str(level)
                for level in column
                if str(level) not in ["", "nan"]
            ]
        )
        for column in raw.columns
    ]


# Handle Date index
raw = raw.reset_index()

date_candidates = [
    column
    for column in raw.columns
    if str(column).lower()
    in ["date", "datetime", "timestamp"]
]

if date_candidates:

    raw_date_column = date_candidates[0]

else:

    datetime_columns = [
        column
        for column in raw.columns
        if pd.api.types.is_datetime64_any_dtype(
            raw[column]
        )
    ]

    if not datetime_columns:

        raise ValueError(
            "Could not detect date column in features.parquet."
        )

    raw_date_column = datetime_columns[0]


raw["date"] = pd.to_datetime(
    raw[raw_date_column]
)


# Find AAPL adjusted close
price_candidates = [
    column
    for column in raw.columns
    if (
        "adj close" in str(column).lower()
        and
        "aapl" in str(column).lower()
    )
]

if not price_candidates:

    raise ValueError(
        "Could not find AAPL adjusted close in features.parquet."
    )

price_column = price_candidates[0]

print(
    f"AAPL price column: "
    f"{price_column}"
)

price_data = raw[
    [
        "date",
        price_column
    ]
].copy()

price_data = price_data.rename(
    columns={
        price_column:
        "aapl_adj_close"
    }
)

price_data["aapl_adj_close"] = pd.to_numeric(
    price_data["aapl_adj_close"],
    errors="coerce"
)

price_data = (
    price_data
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


# Future 5-day price
price_data["future_5day_price"] = (
    price_data["aapl_adj_close"]
    .shift(-HOLDING_DAYS)
)

price_data["actual_5day_return"] = (
    price_data["future_5day_price"]
    /
    price_data["aapl_adj_close"]
) - 1

price_data["actual_future_direction"] = (
    price_data["actual_5day_return"] > 0
).astype(int)


# ============================================================
# COMPARE TARGET TO FUTURE DIRECTION
# ============================================================

combined_target_check = pd.concat(
    [
        train[
            ["date", "target"]
        ],
        validation[
            ["date", "target"]
        ],
        test[
            ["date", "target"]
        ]
    ],
    ignore_index=True
)

combined_target_check = (
    combined_target_check
    .drop_duplicates("date")
    .merge(
        price_data[
            [
                "date",
                "actual_5day_return",
                "actual_future_direction"
            ]
        ],
        on="date",
        how="inner"
    )
)

# Remove dates without future price
combined_target_check = (
    combined_target_check
    .dropna(
        subset=[
            "actual_5day_return"
        ]
    )
)

target_match = (
    combined_target_check["target"]
    ==
    combined_target_check[
        "actual_future_direction"
    ]
)

match_rate = target_match.mean()

print(
    f"\nTarget vs actual future 5-day direction:"
)

print(
    f"Rows compared: "
    f"{len(combined_target_check)}"
)

print(
    f"Exact direction match: "
    f"{match_rate:.2%}"
)

print(
    f"Mismatches: "
    f"{(~target_match).sum()}"
)

if match_rate >= 0.99:

    print(
        "PASS: Target strongly matches actual "
        "future 5-day price direction."
    )

elif match_rate >= 0.95:

    print(
        "PASS/WARNING: Target mostly matches "
        "future 5-day direction, but some differences exist."
    )

else:

    print(
        "WARNING: Target does not closely match "
        "actual future 5-day direction."
    )


# ============================================================
# 8. FEATURE DATE ALIGNMENT
# ============================================================

section("8. FEATURE DATE ALIGNMENT")

processed_dates = set(
    pd.concat(
        [
            train["date"],
            validation["date"],
            test["date"]
        ]
    )
)

raw_dates = set(
    price_data["date"]
)

intersection = (
    processed_dates
    .intersection(raw_dates)
)

print(
    f"Processed unique dates : "
    f"{len(processed_dates)}"
)

print(
    f"Raw price unique dates : "
    f"{len(raw_dates)}"
)

print(
    f"Common dates            : "
    f"{len(intersection)}"
)

coverage = (
    len(intersection)
    /
    len(processed_dates)
)

print(
    f"Processed-date coverage "
    f"in raw prices: {coverage:.2%}"
)


# ============================================================
# 9. FEATURE VALUE SANITY CHECK
# ============================================================

section("9. FEATURE VALUE SANITY CHECK")

for feature in feature_columns:

    series = pd.concat(
        [
            train[feature],
            validation[feature],
            test[feature]
        ],
        ignore_index=True
    )

    finite = np.isfinite(
        series.to_numpy(
            dtype=float
        )
    ).all()

    minimum = series.min()
    maximum = series.max()
    mean = series.mean()

    print(
        f"{feature:<45} "
        f"min={minimum:.6f} "
        f"max={maximum:.6f} "
        f"mean={mean:.6f} "
        f"finite={finite}"
    )


# ============================================================
# 10. FINAL PREDICTIONS
# ============================================================

section("10. FINAL PREDICTION BEHAVIOUR")

if not FINAL_PREDICTIONS_PATH.exists():

    print(
        "WARNING: Final prediction file not found."
    )

else:

    predictions = pd.read_csv(
        FINAL_PREDICTIONS_PATH
    )

    predictions["date"] = pd.to_datetime(
        predictions["date"]
    )

    probabilities = (
        predictions[
            "prediction_probability"
        ]
    )

    print(
        f"Prediction rows: "
        f"{len(predictions)}"
    )

    print(
        f"Probability minimum: "
        f"{probabilities.min():.6f}"
    )

    print(
        f"Probability maximum: "
        f"{probabilities.max():.6f}"
    )

    print(
        f"Probability mean: "
        f"{probabilities.mean():.6f}"
    )

    print(
        f"Probability median: "
        f"{probabilities.median():.6f}"
    )

    print(
        f"Probability std: "
        f"{probabilities.std():.6f}"
    )

    print(
        f"BUY threshold: "
        f"{LOCKED_THRESHOLD:.2f}"
    )

    print(
        f"Predictions >= threshold: "
        f"{(probabilities >= LOCKED_THRESHOLD).sum()}"
    )

    print(
        f"BUY signal percentage: "
        f"{(probabilities >= LOCKED_THRESHOLD).mean():.2%}"
    )


# ============================================================
# 11. PROBABILITY DISTRIBUTION
# ============================================================

section("11. PROBABILITY DISTRIBUTION")

if FINAL_PREDICTIONS_PATH.exists():

    bins = [
        0.00,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        1.00
    ]

    distribution = pd.cut(
        probabilities,
        bins=bins,
        include_lowest=True
    ).value_counts(
        sort=False
    )

    print(distribution)


# ============================================================
# 12. FINAL TRADE LOG
# ============================================================

section("12. FINAL TRADE LOG CONSISTENCY")

if not FINAL_TRADES_PATH.exists():

    print(
        "WARNING: Final trade log not found."
    )

else:

    trades = pd.read_csv(
        FINAL_TRADES_PATH
    )

    if len(trades) == 0:

        print(
            "WARNING: Trade log contains zero trades."
        )

    else:

        trades["entry_date"] = pd.to_datetime(
            trades["entry_date"]
        )

        trades["exit_date"] = pd.to_datetime(
            trades["exit_date"]
        )

        print(
            f"Number of trades: "
            f"{len(trades)}"
        )

        for _, trade in trades.iterrows():

            holding_days = (
                trade["exit_date"]
                - trade["entry_date"]
            ).days

            print(
                f"\nEntry: {trade['entry_date'].date()}"
            )

            print(
                f"Exit : {trade['exit_date'].date()}"
            )

            print(
                f"Price: "
                f"{trade['entry_price']:.4f}"
                f" → "
                f"{trade['exit_price']:.4f}"
            )

            print(
                f"Probability: "
                f"{trade['prediction_probability']:.6f}"
            )

            print(
                f"Actual return: "
                f"{trade['actual_return']:.4%}"
            )

            print(
                f"Net return: "
                f"{trade['net_return']:.4%}"
            )

            print(
                f"Calendar holding days: "
                f"{holding_days}"
            )

            if holding_days <= 9:

                print(
                    "PASS: Exit appears consistent "
                    "with approximately 5 trading days."
                )

            else:

                print(
                    "WARNING: Exit interval appears unusually long."
                )


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

section("13. FINAL SUMMARY CONSISTENCY")

if FINAL_SUMMARY_PATH.exists():

    summary = pd.read_csv(
        FINAL_SUMMARY_PATH
    )

    print(
        summary.to_string(
            index=False
        )
    )

else:

    print(
        "WARNING: Final summary file not found."
    )


# ============================================================
# 14. CRITICAL INTERPRETATION
# ============================================================

section("14. AUDIT INTERPRETATION")

print("""
IMPORTANT:

A clean structural audit does NOT prove that a trading
strategy will make money in the future.

The most important issue currently observed is:

- The locked threshold is 0.65.
- The final test contains very few BUY signals.
- Only completed trades are counted in the trade log.
- Therefore performance statistics such as Sharpe,
  win rate, and drawdown have very low statistical reliability.

This audit therefore distinguishes:

PASS:
Structural/data checks that can be verified directly.

WARNING:
Issues requiring further investigation.

NOT PROVEN:
Claims that cannot be established from the available
processed datasets alone.
""")


# ============================================================
# FINAL STATUS
# ============================================================

section("NORTHGATE AI — DEEPER AUDIT COMPLETED")

print("""
No model was retrained.
No threshold was changed.
No existing result file was modified.

Next step should be decided from the audit output.
""")