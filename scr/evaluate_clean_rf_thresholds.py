from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

print("=" * 60)
print("THRESHOLD EVALUATION")
print("=" * 60)

validation_df = pd.read_parquet(
    DATA_DIR / "AAPL_clean_validation.parquet"
)

test_df = pd.read_parquet(
    DATA_DIR / "AAPL_clean_test.parquet"
)

model = joblib.load(
    DATA_DIR / "AAPL_clean_random_forest_model.pkl"
)

target_column = "target"

feature_columns = [
    column for column in validation_df.columns
    if column != target_column
]

X_validation = validation_df[feature_columns].copy()
y_validation = validation_df[target_column]

X_test = test_df[feature_columns].copy()
y_test = test_df[target_column]

# Use validation training-independent medians
train_df = pd.read_parquet(
    DATA_DIR / "AAPL_clean_train.parquet"
)

train_medians = train_df[feature_columns].median()

X_validation = (
    X_validation
    .replace([np.inf, -np.inf], np.nan)
    .fillna(train_medians)
)

X_test = (
    X_test
    .replace([np.inf, -np.inf], np.nan)
    .fillna(train_medians)
)

validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]

test_probabilities = model.predict_proba(
    X_test
)[:, 1]


def evaluate_threshold(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "predicted_up": int(predictions.sum()),
        "predicted_down": int((predictions == 0).sum()),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


thresholds = [
    0.30,
    0.35,
    0.38,
    0.40,
    0.42,
    0.44,
    0.45,
    0.46,
    0.48,
    0.50,
]

validation_results = []

for threshold in thresholds:
    result = evaluate_threshold(
        y_validation,
        validation_probabilities,
        threshold
    )
    validation_results.append(result)

validation_results_df = pd.DataFrame(validation_results)

print("\nVALIDATION THRESHOLD RESULTS")
print("=" * 60)
print(
    validation_results_df[
        [
            "threshold",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1_score",
            "predicted_up",
            "predicted_down",
        ]
    ].round(4).to_string(index=False)
)

best_threshold_row = validation_results_df.loc[
    validation_results_df["balanced_accuracy"].idxmax()
]

best_threshold = float(
    best_threshold_row["threshold"]
)

print("\nBest threshold selected using validation balanced accuracy:")
print(f"{best_threshold:.2f}")

test_result = evaluate_threshold(
    y_test,
    test_probabilities,
    best_threshold
)

print("\nTEST RESULT USING VALIDATION-SELECTED THRESHOLD")
print("=" * 60)

for key, value in test_result.items():
    if isinstance(value, float):
        print(f"{key:20s}: {value:.4f}")
    else:
        print(f"{key:20s}: {value}")

validation_results_df.to_csv(
    DATA_DIR / "AAPL_clean_rf_threshold_results.csv",
    index=False
)

pd.DataFrame([test_result]).to_csv(
    DATA_DIR / "AAPL_clean_rf_threshold_test_result.csv",
    index=False
)

print("\nResults saved successfully.")