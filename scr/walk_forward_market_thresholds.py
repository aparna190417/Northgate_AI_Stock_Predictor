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

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "walk_forward_market_thresholds"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

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

THRESHOLDS = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]

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
print("NORTHGATE AI — MARKET-ONLY WALK-FORWARD THRESHOLD ANALYSIS")
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

print(f"Train shape:      {train_df.shape}")
print(f"Validation shape: {validation_df.shape}")
print(f"Test shape:       {test_df.shape}")


for df in [train_df, validation_df, test_df]:

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN]
    )

    df.sort_values(
        DATE_COLUMN,
        inplace=True,
    )

    df.reset_index(
        drop=True,
        inplace=True,
    )


# ============================================================================
# COMBINE
# ============================================================================

df = pd.concat(
    [
        train_df,
        validation_df,
        test_df,
    ],
    ignore_index=True,
)

df = df.sort_values(
    DATE_COLUMN
).reset_index(drop=True)


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

    print("\nMissing features:")

    for feature in missing_features:
        print(" -", feature)

    raise ValueError(
        "Required market features are missing."
    )

print(
    f"\nSelected features: "
    f"{len(MARKET_FEATURES)}"
)

for feature in MARKET_FEATURES:
    print(" -", feature)


# ============================================================================
# DATA QUALITY
# ============================================================================

working_df = df[
    MARKET_FEATURES
    + [
        TARGET_COLUMN,
        DATE_COLUMN,
    ]
].copy()

working_df = working_df.replace(
    [np.inf, -np.inf],
    np.nan,
)

missing_count = (
    working_df.isna().sum().sum()
)

duplicate_count = (
    working_df.duplicated().sum()
)

print("\n" + "=" * 80)
print("DATA QUALITY")
print("=" * 80)

print(
    f"Missing values: {missing_count}"
)

print(
    f"Duplicate rows: {duplicate_count}"
)

if missing_count > 0:
    raise ValueError(
        "Missing values detected."
    )

if duplicate_count > 0:
    raise ValueError(
        "Duplicate rows detected."
    )


# ============================================================================
# YEAR
# ============================================================================

working_df["year"] = (
    working_df[DATE_COLUMN].dt.year
)

years = sorted(
    working_df["year"].unique()
)

test_years = [
    year
    for year in years
    if year >= 2020
]


# ============================================================================
# WALK-FORWARD THRESHOLD EXPERIMENT
# ============================================================================

print("\n" + "=" * 80)
print("WALK-FORWARD THRESHOLD EXPERIMENT")
print("=" * 80)

print(
    "\nThresholds:",
    THRESHOLDS,
)

all_results = []


for test_year in test_years:

    train_part = working_df[
        working_df["year"] < test_year
    ].copy()

    test_part = working_df[
        working_df["year"] == test_year
    ].copy()

    X_train = train_part[
        MARKET_FEATURES
    ]

    y_train = train_part[
        TARGET_COLUMN
    ].astype(int)

    X_test = test_part[
        MARKET_FEATURES
    ]

    y_test = test_part[
        TARGET_COLUMN
    ].astype(int)

    # ------------------------------------------------------------------------
    # TRAIN MODEL
    # ------------------------------------------------------------------------

    model = RandomForestClassifier(
        **MODEL_PARAMS
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # ------------------------------------------------------------------------
    # EVALUATE ALL THRESHOLDS
    # ------------------------------------------------------------------------

    print(
        f"\n{'-' * 80}"
    )

    print(
        f"TEST YEAR: {test_year}"
    )

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        ba = balanced_accuracy_score(
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

        all_results.append(
            {
                "test_year": test_year,
                "threshold": threshold,
                "balanced_accuracy": ba,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "positive_prediction_rate": positive_rate,
            }
        )

        print(
            f"Threshold {threshold:.2f}"
            f" | BA: {ba:.4f}"
            f" | F1: {f1:.4f}"
            f" | Precision: {precision:.4f}"
            f" | Recall: {recall:.4f}"
            f" | UP rate: {positive_rate:.4f}"
        )


# ============================================================================
# RESULTS DATAFRAME
# ============================================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================================
# AVERAGE BY THRESHOLD
# ============================================================================

threshold_summary = (
    results_df
    .groupby("threshold")
    .agg(
        average_balanced_accuracy=(
            "balanced_accuracy",
            "mean",
        ),
        median_balanced_accuracy=(
            "balanced_accuracy",
            "median",
        ),
        average_f1=(
            "f1",
            "mean",
        ),
        average_precision=(
            "precision",
            "mean",
        ),
        average_recall=(
            "recall",
            "mean",
        ),
        average_positive_rate=(
            "positive_prediction_rate",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================================
# PRINT SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("THRESHOLD SUMMARY")
print("=" * 80)

print(
    threshold_summary.to_string(
        index=False
    )
)


# ============================================================================
# BEST THRESHOLD
# ============================================================================

best_index = (
    threshold_summary[
        "average_balanced_accuracy"
    ].idxmax()
)

best_row = threshold_summary.loc[
    best_index
]

best_threshold = (
    best_row["threshold"]
)

best_ba = (
    best_row[
        "average_balanced_accuracy"
    ]
)


print("\n" + "=" * 80)
print("BEST WALK-FORWARD THRESHOLD")
print("=" * 80)

print(
    f"\nBest threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Average Balanced Accuracy: "
    f"{best_ba:.4f}"
)

print(
    f"Average F1: "
    f"{best_row['average_f1']:.4f}"
)

print(
    f"Average Precision: "
    f"{best_row['average_precision']:.4f}"
)

print(
    f"Average Recall: "
    f"{best_row['average_recall']:.4f}"
)

print(
    f"Average UP prediction rate: "
    f"{best_row['average_positive_rate']:.4f}"
)


# ============================================================================
# YEAR-BY-YEAR PERFORMANCE AT BEST THRESHOLD
# ============================================================================

best_threshold_results = results_df[
    results_df["threshold"]
    == best_threshold
].copy()


print("\n" + "=" * 80)
print(
    "BEST THRESHOLD — YEAR-BY-YEAR"
)
print("=" * 80)

print(
    best_threshold_results[
        [
            "test_year",
            "threshold",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
            "positive_prediction_rate",
        ]
    ].to_string(index=False)
)


# ============================================================================
# BASELINE CHECK
# ============================================================================

baseline = 0.50

years_above_baseline = (
    best_threshold_results[
        "balanced_accuracy"
    ] > baseline
).sum()

print("\n" + "-" * 80)
print("BASELINE CHECK")
print("-" * 80)

print(
    f"Years above 0.5000 BA: "
    f"{years_above_baseline} "
    f"/ {len(best_threshold_results)}"
)

print(
    f"Average BA difference from baseline: "
    f"{best_ba - baseline:+.4f}"
)


# ============================================================================
# SAVE RESULTS
# ============================================================================

all_results_path = (
    RESULTS_DIR
    / "market_only_walk_forward_threshold_results.csv"
)

summary_path = (
    RESULTS_DIR
    / "market_only_threshold_summary.csv"
)

results_df.to_csv(
    all_results_path,
    index=False,
)

threshold_summary.to_csv(
    summary_path,
    index=False,
)


# ============================================================================
# COMPLETE
# ============================================================================

print("\n" + "=" * 80)
print("MARKET-ONLY WALK-FORWARD THRESHOLD ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"\nDetailed results saved to:"
    f"\n{all_results_path}"
)

print(
    f"\nThreshold summary saved to:"
    f"\n{summary_path}"
)