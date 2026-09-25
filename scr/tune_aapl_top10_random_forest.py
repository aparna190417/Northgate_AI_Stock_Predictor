from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "AAPL_5day_normalized_train.parquet"
VAL_PATH = ROOT / "data" / "processed" / "AAPL_5day_normalized_validation.parquet"
TEST_PATH = ROOT / "data" / "processed" / "AAPL_5day_normalized_test.parquet"

IMPORTANCE_PATH = (
    ROOT
    / "results"
    / "feature_analysis"
    / "combined_feature_importance.csv"
)

RESULTS_DIR = ROOT / "results" / "hyperparameter_tuning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------

train_df = pd.read_parquet(TRAIN_PATH)
val_df = pd.read_parquet(VAL_PATH)
test_df = pd.read_parquet(TEST_PATH)

importance_df = pd.read_csv(IMPORTANCE_PATH)

all_features = [
    col for col in train_df.columns
    if col not in ["date", "target"]
]

ranked_features = [
    feature
    for feature in importance_df["feature"].tolist()
    if feature in all_features
]

selected_features = ranked_features[:10]

print("=" * 80)
print("TOP-10 RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 80)

print("\nSelected features:")
for feature in selected_features:
    print(f" - {feature}")

X_train = train_df[selected_features]
y_train = train_df["target"]

X_val = val_df[selected_features]
y_val = val_df["target"]

X_test = test_df[selected_features]
y_test = test_df["target"]


# --------------------------------------------------
# Hyperparameter configurations
# --------------------------------------------------

configs = [
    {
        "name": "rf_1",
        "n_estimators": 300,
        "max_depth": 3,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "name": "rf_2",
        "n_estimators": 400,
        "max_depth": 4,
        "min_samples_leaf": 15,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "name": "rf_3",
        "n_estimators": 500,
        "max_depth": 5,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "name": "rf_4",
        "n_estimators": 500,
        "max_depth": 6,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "name": "rf_5",
        "n_estimators": 500,
        "max_depth": 8,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "name": "rf_6",
        "n_estimators": 500,
        "max_depth": 5,
        "min_samples_leaf": 20,
        "max_features": 0.7,
        "class_weight": "balanced",
    },
    {
        "name": "rf_7",
        "n_estimators": 500,
        "max_depth": 6,
        "min_samples_leaf": 20,
        "max_features": 0.7,
        "class_weight": "balanced",
    },
    {
        "name": "rf_8",
        "n_estimators": 700,
        "max_depth": 4,
        "min_samples_leaf": 25,
        "max_features": 0.7,
        "class_weight": "balanced",
    },
    {
        "name": "rf_9",
        "n_estimators": 700,
        "max_depth": 6,
        "min_samples_leaf": 15,
        "max_features": "log2",
        "class_weight": "balanced",
    },
    {
        "name": "rf_10",
        "n_estimators": 700,
        "max_depth": 8,
        "min_samples_leaf": 20,
        "max_features": "log2",
        "class_weight": "balanced",
    },
]


def calculate_metrics(y_true, predictions):
    return {
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
    }


results = []

for config in configs:
    config_name = config.pop("name")

    model = RandomForestClassifier(
        **config,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    train_metrics = calculate_metrics(y_train, train_pred)
    val_metrics = calculate_metrics(y_val, val_pred)
    test_metrics = calculate_metrics(y_test, test_pred)

    result = {
        "model": config_name,
        **{
            f"train_{key}": value
            for key, value in train_metrics.items()
        },
        **{
            f"validation_{key}": value
            for key, value in val_metrics.items()
        },
        **{
            f"test_{key}": value
            for key, value in test_metrics.items()
        },
        **config,
    }

    results.append(result)

    print(
        f"\n{config_name} | "
        f"Train BA: {train_metrics['balanced_accuracy']:.4f} | "
        f"Val BA: {val_metrics['balanced_accuracy']:.4f} | "
        f"Test BA: {test_metrics['balanced_accuracy']:.4f} | "
        f"Test F1: {test_metrics['f1']:.4f}"
    )


# --------------------------------------------------
# Save and display results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "validation_balanced_accuracy",
    ascending=False
)

results_path = RESULTS_DIR / "top10_random_forest_tuning.csv"
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 80)
print("RESULTS SORTED BY VALIDATION BALANCED ACCURACY")
print("=" * 80)

print(results_df.to_string(index=False))

print(f"\nSaved results to: {results_path}")
print("\nHYPERPARAMETER TUNING COMPLETED")
