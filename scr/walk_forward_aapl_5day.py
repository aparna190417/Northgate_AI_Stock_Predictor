from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# --------------------------------------------------
# Project paths
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

RESULTS_DIR = ROOT / "results" / "walk_forward"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load normalized train, validation and test data
# --------------------------------------------------

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

train_part = pd.read_parquet(TRAIN_PATH)
val_part = pd.read_parquet(VAL_PATH)
test_part = pd.read_parquet(TEST_PATH)

df = pd.concat(
    [train_part, val_part, test_part],
    ignore_index=True
)

df["date"] = pd.to_datetime(df["date"])

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

print(f"Train part shape:      {train_part.shape}")
print(f"Validation part shape: {val_part.shape}")
print(f"Test part shape:       {test_part.shape}")
print(f"Combined dataset:      {df.shape}")


# --------------------------------------------------
# Load feature importance
# --------------------------------------------------

importance_df = pd.read_csv(IMPORTANCE_PATH)

all_features = [
    col
    for col in df.columns
    if col not in ["date", "target"]
]

ranked_features = [
    feature
    for feature in importance_df["feature"].tolist()
    if feature in all_features
]

selected_features = ranked_features[:10]


# --------------------------------------------------
# Basic checks
# --------------------------------------------------

if len(selected_features) == 0:
    raise ValueError(
        "No selected features found. "
        "Please check combined_feature_importance.csv"
    )

required_columns = ["date", "target"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' is missing from dataset."
        )

missing_features = [
    feature
    for feature in selected_features
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"These selected features are missing: {missing_features}"
    )

df = df.dropna(
    subset=selected_features + ["target"]
).reset_index(drop=True)


# --------------------------------------------------
# Project information
# --------------------------------------------------

print("\n" + "=" * 80)
print("AAPL 5-DAY WALK-FORWARD VALIDATION")
print("=" * 80)

print(f"Final dataset shape: {df.shape}")
print(f"Selected features: {len(selected_features)}")

print("\nSelected features:")

for feature in selected_features:
    print(f" - {feature}")


# --------------------------------------------------
# Year-based walk-forward validation
# --------------------------------------------------

years = sorted(
    df["date"]
    .dt
    .year
    .unique()
)

results = []

minimum_train_years = 5

print("\n" + "=" * 80)
print("YEAR-BY-YEAR WALK-FORWARD RESULTS")
print("=" * 80)

for test_year in years:

    train_years = [
        year
        for year in years
        if year < test_year
    ]

    if len(train_years) < minimum_train_years:
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
    y_test = test_df["target"]

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

    predictions = model.predict(X_test)

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    metrics = {
        "test_year": test_year,
        "train_start": train_df["date"].min(),
        "train_end": train_df["date"].max(),
        "test_start": test_df["date"].min(),
        "test_end": test_df["date"].max(),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

    results.append(metrics)

    print(
        f"\nTest year: {test_year}"
        f" | Train rows: {len(train_df)}"
        f" | Test rows: {len(test_df)}"
        f" | BA: {balanced_accuracy:.4f}"
        f" | F1: {f1:.4f}"
        f" | Precision: {precision:.4f}"
        f" | Recall: {recall:.4f}"
    )


# --------------------------------------------------
# Save year-wise results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_path = (
    RESULTS_DIR
    / "aapl_walk_forward_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# --------------------------------------------------
# Walk-forward summary
# --------------------------------------------------

print("\n" + "=" * 80)
print("WALK-FORWARD SUMMARY")
print("=" * 80)

if not results_df.empty:

    print("\nYear-wise results:\n")
    print(
        results_df.to_string(
            index=False
        )
    )

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

    beating_baseline = (
        results_df["balanced_accuracy"] > 0.5000
    ).sum()

    total_years = len(results_df)

    print("\n" + "-" * 80)
    print("OVERALL METRICS")
    print("-" * 80)

    print(
        f"Average Balanced Accuracy: "
        f"{average_ba:.4f}"
    )

    print(
        f"Median Balanced Accuracy:  "
        f"{median_ba:.4f}"
    )

    print(
        f"Average F1 Score:           "
        f"{average_f1:.4f}"
    )

    print(
        f"Average Precision:          "
        f"{average_precision:.4f}"
    )

    print(
        f"Average Recall:             "
        f"{average_recall:.4f}"
    )

    print(
        "\nYears beating 0.5000 baseline: "
        f"{beating_baseline} out of {total_years}"
    )

    print(
        f"Baseline difference: "
        f"{(average_ba - 0.5000):+.4f}"
    )

    print("\nBest walk-forward year:")

    best_row = results_df.loc[
        results_df["balanced_accuracy"].idxmax()
    ]

    print(
        best_row.to_string()
    )

    print("\nWorst walk-forward year:")

    worst_row = results_df.loc[
        results_df["balanced_accuracy"].idxmin()
    ]

    print(
        worst_row.to_string()
    )

    print("\nYears with Balanced Accuracy above baseline:")

    above_baseline_df = results_df[
        results_df["balanced_accuracy"] > 0.5000
    ]

    if not above_baseline_df.empty:
        print(
            above_baseline_df[
                [
                    "test_year",
                    "balanced_accuracy",
                    "f1",
                    "precision",
                    "recall"
                ]
            ].to_string(index=False)
        )
    else:
        print("No year beat the 0.5000 baseline.")

    print("\nYears below or equal to baseline:")

    below_baseline_df = results_df[
        results_df["balanced_accuracy"] <= 0.5000
    ]

    if not below_baseline_df.empty:
        print(
            below_baseline_df[
                [
                    "test_year",
                    "balanced_accuracy",
                    "f1",
                    "precision",
                    "recall"
                ]
            ].to_string(index=False)
        )
    else:
        print("All evaluated years beat the baseline.")

else:

    print(
        "No walk-forward results were generated."
    )

    print(
        "Check whether the dataset contains "
        "at least 5 complete training years."
    )


# --------------------------------------------------
# Final message
# --------------------------------------------------

print("\n" + "=" * 80)
print("WALK-FORWARD VALIDATION COMPLETED")
print("=" * 80)

print(
    f"Saved results to: {results_path}"
)