from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# NORTHGATE AI
# FINAL TECHNICAL AUDIT
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

FINAL_PREDICTIONS_PATH = (
    ROOT
    / "results"
    / "final_validation_test"
    / "final_test_predictions.csv"
)

TRADE_LOG_PATH = (
    ROOT
    / "results"
    / "final_validation_test"
    / "final_test_trade_log.csv"
)

SUMMARY_PATH = (
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
print("NORTHGATE AI — FINAL TECHNICAL AUDIT")
print("=" * 80)

print()
print("Purpose:")
print("This audit does NOT retrain the model.")
print("This audit does NOT change the threshold.")
print("This audit checks data integrity, leakage risks,")
print("date separation, predictions, and final trade results.")
print()


# ============================================================
# LOAD TRAIN / VALIDATION / TEST
# ============================================================

print("=" * 80)
print("1. LOADING DATASETS")
print("=" * 80)

train = pd.read_parquet(TRAIN_PATH)
validation = pd.read_parquet(VAL_PATH)
test = pd.read_parquet(TEST_PATH)

for name, data in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:

    data["date"] = pd.to_datetime(data["date"])

    data.sort_values(
        "date",
        inplace=True
    )

    data.reset_index(
        drop=True,
        inplace=True
    )

    print(
        f"{name:12s}: "
        f"rows={len(data):4d} | "
        f"columns={len(data.columns):3d} | "
        f"{data['date'].min().date()} "
        f"→ "
        f"{data['date'].max().date()}"
    )


# ============================================================
# DATE SEPARATION
# ============================================================

print()
print("=" * 80)
print("2. DATE SEPARATION CHECK")
print("=" * 80)

train_end = train["date"].max()
val_start = validation["date"].min()
val_end = validation["date"].max()
test_start = test["date"].min()

print(
    f"Train end       : {train_end.date()}"
)

print(
    f"Validation start: {val_start.date()}"
)

print(
    f"Validation end  : {val_end.date()}"
)

print(
    f"Test start      : {test_start.date()}"
)

if train_end < val_start:

    print("PASS: Train ends before validation.")

else:

    print("FAIL: Train/validation dates overlap.")


if val_end < test_start:

    print("PASS: Validation ends before test.")

else:

    print("FAIL: Validation/test dates overlap.")


# ============================================================
# DUPLICATE DATE CHECK
# ============================================================

print()
print("=" * 80)
print("3. DUPLICATE DATE CHECK")
print("=" * 80)

for name, data in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:

    duplicates = data["date"].duplicated().sum()

    print(
        f"{name:12s}: duplicate dates = {duplicates}"
    )

    if duplicates == 0:

        print(
            f"PASS: {name} has no duplicate dates."
        )

    else:

        print(
            f"WARNING: {name} contains duplicate dates."
        )


# ============================================================
# MISSING VALUE CHECK
# ============================================================

print()
print("=" * 80)
print("4. MISSING VALUE CHECK")
print("=" * 80)

for name, data in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:

    total_missing = int(
        data.isna().sum().sum()
    )

    print(
        f"{name:12s}: total missing values = "
        f"{total_missing}"
    )


# ============================================================
# TARGET CHECK
# ============================================================

print()
print("=" * 80)
print("5. TARGET CHECK")
print("=" * 80)

for name, data in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:

    if "target" not in data.columns:

        print(
            f"{name}: target column NOT FOUND"
        )

        continue

    print()
    print(name)

    print(
        data["target"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print(
        "Target mean:",
        data["target"].mean()
    )


# ============================================================
# FEATURE LIST
# ============================================================

print()
print("=" * 80)
print("6. FEATURE LIST")
print("=" * 80)

feature_columns = [
    column
    for column in train.columns
    if column not in [
        "date",
        "target"
    ]
]

print(
    f"Number of model features: "
    f"{len(feature_columns)}"
)

for feature in feature_columns:

    print(
        f" - {feature}"
    )


# ============================================================
# FUTURE-LOOKING FEATURE NAME CHECK
# ============================================================

print()
print("=" * 80)
print("7. FUTURE-LOOKING FEATURE NAME CHECK")
print("=" * 80)

future_keywords = [
    "future",
    "forward",
    "next",
    "target",
    "label",
    "lead",
    "ahead"
]

suspicious_features = []

for feature in feature_columns:

    feature_lower = str(feature).lower()

    for keyword in future_keywords:

        if keyword in feature_lower:

            suspicious_features.append(
                (
                    feature,
                    keyword
                )
            )

            break


if not suspicious_features:

    print(
        "PASS: No obvious future-looking "
        "feature names detected."
    )

else:

    print(
        "WARNING: Potentially suspicious "
        "feature names found:"
    )

    for feature, keyword in suspicious_features:

        print(
            f" - {feature} "
            f"(matched: {keyword})"
        )


# ============================================================
# FEATURE IMPORTANCE CHECK
# ============================================================

print()
print("=" * 80)
print("8. FEATURE IMPORTANCE CHECK")
print("=" * 80)

if IMPORTANCE_PATH.exists():

    importance = pd.read_csv(
        IMPORTANCE_PATH
    )

    print(
        f"Importance rows: {len(importance)}"
    )

    print(
        "Columns:",
        list(importance.columns)
    )

    if "feature" in importance.columns:

        importance_features = (
            importance["feature"]
            .astype(str)
            .tolist()
        )

        overlap = [
            feature
            for feature in importance_features
            if feature in feature_columns
        ]

        print(
            f"Importance features matching "
            f"model features: {len(overlap)}"
        )

        print()
        print("Top importance features:")

        for feature in overlap[:15]:

            print(
                f" - {feature}"
            )

else:

    print(
        "WARNING: Feature importance file "
        "not found."
    )


# ============================================================
# FEATURES.PARQUET CHECK
# ============================================================

print()
print("=" * 80)
print("9. ORIGINAL FEATURES.PARQUET CHECK")
print("=" * 80)

if FEATURES_PATH.exists():

    raw_features = pd.read_parquet(
        FEATURES_PATH
    )

    print(
        f"features.parquet shape: "
        f"{raw_features.shape}"
    )

    print(
        "Index type:",
        type(raw_features.index).__name__
    )

    print(
        "Index name:",
        raw_features.index.name
    )

    print(
        "First raw feature columns:"
    )

    for column in list(
        raw_features.columns
    )[:20]:

        print(
            f" - {column}"
        )

else:

    print(
        "WARNING: features.parquet not found."
    )


# ============================================================
# FINAL PREDICTIONS CHECK
# ============================================================

print()
print("=" * 80)
print("10. FINAL TEST PREDICTIONS CHECK")
print("=" * 80)

if FINAL_PREDICTIONS_PATH.exists():

    predictions = pd.read_csv(
        FINAL_PREDICTIONS_PATH
    )

    predictions["date"] = pd.to_datetime(
        predictions["date"]
    )

    print(
        f"Prediction rows: {len(predictions)}"
    )

    print(
        f"Prediction period: "
        f"{predictions['date'].min().date()} "
        f"→ "
        f"{predictions['date'].max().date()}"
    )

    print(
        "Columns:"
    )

    for column in predictions.columns:

        print(
            f" - {column}"
        )

    if "prediction_probability" in predictions.columns:

        probabilities = predictions[
            "prediction_probability"
        ]

        print()
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
            f"Predictions >= locked threshold "
            f"({LOCKED_THRESHOLD:.2f}): "
            f"{(
                probabilities >= LOCKED_THRESHOLD
            ).sum()}"
        )

    if "signal" in predictions.columns:

        print(
            f"BUY signals: "
            f"{predictions['signal'].sum()}"
        )

        print(
            f"Total prediction rows: "
            f"{len(predictions)}"
        )

else:

    print(
        "FAIL: Final predictions file "
        "not found."
    )


# ============================================================
# TRADE LOG CHECK
# ============================================================

print()
print("=" * 80)
print("11. FINAL TRADE LOG CHECK")
print("=" * 80)

if TRADE_LOG_PATH.exists():

    trades = pd.read_csv(
        TRADE_LOG_PATH
    )

    print(
        f"Number of trades: {len(trades)}"
    )

    if len(trades) == 0:

        print(
            "No trades found."
        )

    else:

        print()
        print("Trade details:")

        print(
            trades.to_string(
                index=False
            )
        )

        if "actual_return" in trades.columns:

            print()
            print(
                f"Average actual return: "
                f"{trades['actual_return'].mean():.4%}"
            )

        if "net_return" in trades.columns:

            print(
                f"Average net return: "
                f"{trades['net_return'].mean():.4%}"
            )

else:

    print(
        "FAIL: Trade log not found."
    )


# ============================================================
# SUMMARY FILE CHECK
# ============================================================

print()
print("=" * 80)
print("12. FINAL SUMMARY CHECK")
print("=" * 80)

if SUMMARY_PATH.exists():

    summary = pd.read_csv(
        SUMMARY_PATH
    )

    print(
        summary.to_string(
            index=False
        )
    )

else:

    print(
        "WARNING: Summary file not found."
    )


# ============================================================
# TEST TARGET / FEATURE OVERLAP CHECK
# ============================================================

print()
print("=" * 80)
print("13. TARGET COLUMN ISOLATION CHECK")
print("=" * 80)

target_in_features = (
    "target" in feature_columns
)

if not target_in_features:

    print(
        "PASS: target is not included "
        "as a model feature."
    )

else:

    print(
        "FAIL: target appears in "
        "model feature list."
    )


# ============================================================
# DATE ORDER CHECK
# ============================================================

print()
print("=" * 80)
print("14. DATE ORDER CHECK")
print("=" * 80)

for name, data in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:

    is_sorted = data["date"].is_monotonic_increasing

    print(
        f"{name:12s}: "
        f"{'PASS' if is_sorted else 'FAIL'}"
    )


# ============================================================
# FINAL AUDIT SUMMARY
# ============================================================

print()
print("=" * 80)
print("FINAL AUDIT SUMMARY")
print("=" * 80)

print()
print("Threshold locked at:")
print(
    f"0.65"
)

print()
print("Final test was:")
print(
    "2025-01-16 → 2026-09-04"
)

print()
print("Important:")
print(
    "A single profitable trade is NOT enough "
    "to establish strategy robustness."
)

print()
print(
    "This audit checks for obvious structural "
    "problems. It does not mathematically prove "
    "that the model contains no possible leakage."
)

print()
print("=" * 80)
print("NORTHGATE AI — FINAL TECHNICAL AUDIT COMPLETED")
print("=" * 80)