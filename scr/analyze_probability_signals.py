from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)


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

RESULTS_DIR = ROOT / "results" / "probability_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------

train_df = pd.read_parquet(TRAIN_PATH)
val_df = pd.read_parquet(VAL_PATH)
test_df = pd.read_parquet(TEST_PATH)

importance_df = pd.read_csv(IMPORTANCE_PATH)

all_features = [
    column
    for column in train_df.columns
    if column not in ["date", "target"]
]

ranked_features = [
    feature
    for feature in importance_df["feature"].tolist()
    if feature in all_features
]

selected_features = ranked_features[:10]

print("=" * 80)
print("PROBABILITY-BASED SIGNAL ANALYSIS")
print("=" * 80)

print(f"Selected features: {len(selected_features)}")

for feature in selected_features:
    print(f" - {feature}")


# --------------------------------------------------
# Prepare data
# --------------------------------------------------

X_train = train_df[selected_features]
y_train = train_df["target"]

X_val = val_df[selected_features]
y_val = val_df["target"]

X_test = test_df[selected_features]
y_test = test_df["target"]


# --------------------------------------------------
# Train model
# --------------------------------------------------

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
    y_train
)


# --------------------------------------------------
# Get probability predictions
# --------------------------------------------------

val_probability = model.predict_proba(X_val)[:, 1]
test_probability = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# Test different probability thresholds
# --------------------------------------------------

thresholds = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]

results = []

for threshold in thresholds:

    val_prediction = (
        val_probability >= threshold
    ).astype(int)

    test_prediction = (
        test_probability >= threshold
    ).astype(int)

    val_ba = balanced_accuracy_score(
        y_val,
        val_prediction
    )

    test_ba = balanced_accuracy_score(
        y_test,
        test_prediction
    )

    val_positive_rate = val_prediction.mean()
    test_positive_rate = test_prediction.mean()

    results.append(
        {
            "threshold": threshold,
            "validation_balanced_accuracy": val_ba,
            "test_balanced_accuracy": test_ba,
            "validation_positive_rate": val_positive_rate,
            "test_positive_rate": test_positive_rate,
        }
    )

    print("\n" + "-" * 80)
    print(f"Threshold: {threshold:.2f}")
    print(
        f"Validation BA: {val_ba:.4f}"
    )
    print(
        f"Test BA:       {test_ba:.4f}"
    )
    print(
        f"Validation positive rate: "
        f"{val_positive_rate:.4f}"
    )
    print(
        f"Test positive rate:       "
        f"{test_positive_rate:.4f}"
    )


# --------------------------------------------------
# Save threshold results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_path = (
    RESULTS_DIR
    / "probability_threshold_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# --------------------------------------------------
# Select best threshold using validation score
# --------------------------------------------------

best_row = results_df.loc[
    results_df[
        "validation_balanced_accuracy"
    ].idxmax()
]

best_threshold = best_row["threshold"]

final_test_prediction = (
    test_probability >= best_threshold
).astype(int)

final_test_ba = balanced_accuracy_score(
    y_test,
    final_test_prediction
)

print("\n" + "=" * 80)
print("BEST THRESHOLD RESULT")
print("=" * 80)

print(
    f"Best threshold from validation: "
    f"{best_threshold:.2f}"
)

print(
    f"Final test balanced accuracy: "
    f"{final_test_ba:.4f}"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        final_test_prediction
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        final_test_prediction,
        zero_division=0
    )
)

print(
    f"\nSaved results to: {results_path}"
)

print("\nPROBABILITY ANALYSIS COMPLETED")