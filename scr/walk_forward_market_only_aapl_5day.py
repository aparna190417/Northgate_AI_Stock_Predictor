from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "walk_forward_market_only"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_PATH = DATA_DIR / "AAPL_5day_normalized_train.parquet"
VALIDATION_PATH = DATA_DIR / "AAPL_5day_normalized_validation.parquet"
TEST_PATH = DATA_DIR / "AAPL_5day_normalized_test.parquet"


# ============================================================================
# CONFIGURATION
# ============================================================================

MARKET_FEATURES = [
    "Features_market_return_1d",
    "Features_market_return_5d",
    "Features_market_return_20d",
    "Features_market_volatility_20d",
]

TARGET_COLUMN = "target"
DATE_COLUMN = "date"

# Same general model family used in the previous walk-forward experiment.
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": None,
}


# ============================================================================
# HEADER
# ============================================================================

print("=" * 80)
print("NORTHGATE AI — AAPL 5-DAY MARKET-ONLY WALK-FORWARD VALIDATION")
print("=" * 80)


# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

train_df = pd.read_parquet(TRAIN_PATH)
validation_df = pd.read_parquet(VALIDATION_PATH)
test_df = pd.read_parquet(TEST_PATH)

print(f"Train part shape:      {train_df.shape}")
print(f"Validation part shape: {validation_df.shape}")
print(f"Test part shape:       {test_df.shape}")

for df in [train_df, validation_df, test_df]:
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    df.sort_values(DATE_COLUMN, inplace=True)
    df.reset_index(drop=True, inplace=True)


# ============================================================================
# COMBINE DATA
# ============================================================================

df = pd.concat(
    [train_df, validation_df, test_df],
    ignore_index=True,
)

df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

print(f"Combined dataset:      {df.shape}")


# ============================================================================
# FEATURE CHECK
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE CHECK")
print("=" * 80)

missing_features = [
    feature
    for feature in MARKET_FEATURES
    if feature not in df.columns
]

if missing_features:
    print("\nMissing market features:")

    for feature in missing_features:
        print(" -", feature)

    raise ValueError(
        "Required market features are missing."
    )

print(f"\nNumber of selected features: {len(MARKET_FEATURES)}")

print("\nSelected features:")

for feature in MARKET_FEATURES:
    print(" -", feature)


# ============================================================================
# DATA QUALITY
# ============================================================================

print("\n" + "=" * 80)
print("DATA QUALITY CHECK")
print("=" * 80)

required_columns = MARKET_FEATURES + [
    TARGET_COLUMN,
    DATE_COLUMN,
]

working_df = df[required_columns].copy()

working_df = working_df.replace(
    [np.inf, -np.inf],
    np.nan,
)

missing_count = working_df.isna().sum().sum()
duplicate_count = working_df.duplicated().sum()

print(f"Total missing values:   {missing_count}")
print(f"Total duplicate rows:   {duplicate_count}")

if missing_count > 0:
    print("\nMissing values by column:")
    print(working_df.isna().sum())

    raise ValueError(
        "Missing values found in walk-forward dataset."
    )

if duplicate_count > 0:
    raise ValueError(
        "Duplicate rows found in walk-forward dataset."
    )


# ============================================================================
# TARGET CHECK
# ============================================================================

print("\n" + "=" * 80)
print("TARGET CHECK")
print("=" * 80)

print("\nTarget distribution:")

print(
    working_df[TARGET_COLUMN]
    .value_counts()
    .sort_index()
)

print("\nTarget proportions:")

print(
    working_df[TARGET_COLUMN]
    .value_counts(normalize=True)
    .sort_index()
    .round(4)
)


# ============================================================================
# WALK-FORWARD CONFIGURATION
# ============================================================================

working_df["year"] = working_df[DATE_COLUMN].dt.year

years = sorted(working_df["year"].unique())

# We need at least one complete previous year for training.
test_years = [
    year
    for year in years
    if year >= 2020
]


print("\n" + "=" * 80)
print("WALK-FORWARD CONFIGURATION")
print("=" * 80)

print("\nTest years:")

for year in test_years:
    print(" -", year)


# ============================================================================
# WALK-FORWARD LOOP
# ============================================================================

results = []

print("\n" + "=" * 80)
print("YEAR-BY-YEAR WALK-FORWARD RESULTS")
print("=" * 80)


for test_year in test_years:

    train_mask = working_df["year"] < test_year
    test_mask = working_df["year"] == test_year

    train_part = working_df.loc[
        train_mask
    ].copy()

    test_part = working_df.loc[
        test_mask
    ].copy()

    if len(train_part) == 0 or len(test_part) == 0:
        continue

    X_train = train_part[MARKET_FEATURES]
    y_train = train_part[TARGET_COLUMN].astype(int)

    X_test = test_part[MARKET_FEATURES]
    y_test = test_part[TARGET_COLUMN].astype(int)

    model = RandomForestClassifier(
        **MODEL_PARAMS
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    positive_rate = np.mean(
        predictions == 1
    )

    result = {
        "test_year": test_year,

        "train_start": train_part[DATE_COLUMN].min(),
        "train_end": train_part[DATE_COLUMN].max(),

        "test_start": test_part[DATE_COLUMN].min(),
        "test_end": test_part[DATE_COLUMN].max(),

        "train_rows": len(train_part),
        "test_rows": len(test_part),

        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,

        "positive_prediction_rate": positive_rate,

        "mean_probability": probabilities.mean(),
        "min_probability": probabilities.min(),
        "max_probability": probabilities.max(),
    }

    results.append(result)

    print(
        f"\nTest year: {test_year}"
        f" | Train rows: {len(train_part)}"
        f" | Test rows: {len(test_part)}"
        f" | BA: {balanced_accuracy:.4f}"
        f" | F1: {f1:.4f}"
        f" | Precision: {precision:.4f}"
        f" | Recall: {recall:.4f}"
    )


# ============================================================================
# RESULTS DATAFRAME
# ============================================================================

results_df = pd.DataFrame(results)


if results_df.empty:
    raise RuntimeError(
        "No walk-forward results were generated."
    )


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("WALK-FORWARD SUMMARY")
print("=" * 80)

print("\nYear-wise results:")

display_columns = [
    "test_year",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "train_rows",
    "test_rows",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "positive_prediction_rate",
]

print(
    results_df[
        display_columns
    ].to_string(index=False)
)


# ============================================================================
# OVERALL METRICS
# ============================================================================

average_ba = results_df[
    "balanced_accuracy"
].mean()

median_ba = results_df[
    "balanced_accuracy"
].median()

average_f1 = results_df[
    "f1"
].mean()

average_precision = results_df[
    "precision"
].mean()

average_recall = results_df[
    "recall"
].mean()

baseline = 0.50

above_baseline = (
    results_df["balanced_accuracy"] > baseline
).sum()

baseline_difference = (
    average_ba - baseline
)


print("\n" + "-" * 80)
print("OVERALL METRICS")
print("-" * 80)

print(
    f"Average Balanced Accuracy: {average_ba:.4f}"
)

print(
    f"Median Balanced Accuracy:  {median_ba:.4f}"
)

print(
    f"Average F1 Score:           {average_f1:.4f}"
)

print(
    f"Average Precision:          {average_precision:.4f}"
)

print(
    f"Average Recall:             {average_recall:.4f}"
)

print(
    f"\nYears beating 0.5000 baseline: "
    f"{above_baseline} out of {len(results_df)}"
)

print(
    f"Baseline difference: {baseline_difference:+.4f}"
)


# ============================================================================
# BEST / WORST YEAR
# ============================================================================

best_year = results_df.loc[
    results_df["balanced_accuracy"].idxmax()
]

worst_year = results_df.loc[
    results_df["balanced_accuracy"].idxmin()
]


print("\n" + "-" * 80)
print("BEST WALK-FORWARD YEAR")
print("-" * 80)

print(
    best_year.to_string()
)


print("\n" + "-" * 80)
print("WORST WALK-FORWARD YEAR")
print("-" * 80)

print(
    worst_year.to_string()
)


# ============================================================================
# BASELINE BREAKDOWN
# ============================================================================

print("\n" + "-" * 80)
print("YEARS ABOVE 0.5000 BALANCED ACCURACY")
print("-" * 80)

above_df = results_df[
    results_df["balanced_accuracy"] > baseline
]

if len(above_df) == 0:
    print("None")
else:
    print(
        above_df[
            [
                "test_year",
                "balanced_accuracy",
                "f1",
                "precision",
                "recall",
            ]
        ].to_string(index=False)
    )


print("\n" + "-" * 80)
print("YEARS BELOW OR EQUAL TO 0.5000")
print("-" * 80)

below_df = results_df[
    results_df["balanced_accuracy"] <= baseline
]

if len(below_df) == 0:
    print("None")
else:
    print(
        below_df[
            [
                "test_year",
                "balanced_accuracy",
                "f1",
                "precision",
                "recall",
            ]
        ].to_string(index=False)
    )


# ============================================================================
# SAVE RESULTS
# ============================================================================

output_path = (
    RESULTS_DIR
    / "aapl_market_only_walk_forward_results.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)

print("\n" + "=" * 80)
print("WALK-FORWARD VALIDATION COMPLETED")
print("=" * 80)

print(
    f"\nSaved results to: {output_path}"
)

print("\nMarket-only walk-forward experiment complete.")