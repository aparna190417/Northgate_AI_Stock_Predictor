from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

TRAIN_PATH = DATA_DIR / "aapl_5day_train.parquet"
VALIDATION_PATH = DATA_DIR / "aapl_5day_validation.parquet"
TEST_PATH = DATA_DIR / "aapl_5day_test.parquet"

MODEL_PATH = DATA_DIR / "AAPL_5day_random_forest_model.pkl"


print("=" * 70)
print("AAPL 5-DAY RANDOM FOREST THRESHOLD EVALUATION")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_parquet(TRAIN_PATH)
validation_df = pd.read_parquet(VALIDATION_PATH)
test_df = pd.read_parquet(TEST_PATH)

model = joblib.load(MODEL_PATH)

print(f"\nTrain shape      : {train_df.shape}")
print(f"Validation shape : {validation_df.shape}")
print(f"Test shape       : {test_df.shape}")


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

        df["date_year"] = df["date"].dt.year
        df["date_month"] = df["date"].dt.month
        df["date_day"] = df["date"].dt.day
        df["date_day_of_week"] = df["date"].dt.dayofweek

        df = df.drop(columns=["date"])

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

        df["date_year"] = df["Date"].dt.year
        df["date_month"] = df["Date"].dt.month
        df["date_day"] = df["Date"].dt.day
        df["date_day_of_week"] = df["Date"].dt.dayofweek

        df = df.drop(columns=["Date"])

    return df


train_df = prepare_features(train_df)
validation_df = prepare_features(validation_df)
test_df = prepare_features(test_df)


# ============================================================
# X / y
# ============================================================

X_train = train_df.drop(columns=["target"]).copy()
y_train = train_df["target"].copy()

X_validation = validation_df.drop(columns=["target"]).copy()
y_validation = validation_df["target"].copy()

X_test = test_df.drop(columns=["target"]).copy()
y_test = test_df["target"].copy()


# ============================================================
# SAME FEATURE ORDER
# ============================================================

feature_columns = X_train.columns

X_validation = X_validation[feature_columns]
X_test = X_test[feature_columns]


# ============================================================
# HANDLE INF / MISSING VALUES
# ============================================================

train_medians = X_train.median()

X_train = (
    X_train
    .replace([np.inf, -np.inf], np.nan)
    .fillna(train_medians)
)

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


# ============================================================
# PROBABILITIES
# ============================================================

validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]

test_probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

thresholds = np.arange(
    0.30,
    0.701,
    0.025
)

results = []

print("\n" + "=" * 70)
print("VALIDATION THRESHOLD RESULTS")
print("=" * 70)

for threshold in thresholds:

    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_validation,
        predictions
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0
    )

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predicted_up": predictions.sum(),
        "predicted_down": (predictions == 0).sum(),
    })

    print(
        f"Threshold {threshold:.3f} | "
        f"Accuracy {accuracy:.4f} | "
        f"Balanced Acc {balanced_accuracy:.4f} | "
        f"Precision {precision:.4f} | "
        f"Recall {recall:.4f} | "
        f"F1 {f1:.4f} | "
        f"UP {predictions.sum():3d} | "
        f"DOWN {(predictions == 0).sum():3d}"
    )


# ============================================================
# SELECT THRESHOLD USING VALIDATION ONLY
# ============================================================

results_df = pd.DataFrame(results)

best_row = results_df.loc[
    results_df["balanced_accuracy"].idxmax()
]

best_threshold = float(
    best_row["threshold"]
)


print("\n" + "=" * 70)
print("SELECTED THRESHOLD")
print("=" * 70)

print(
    f"Best validation threshold: "
    f"{best_threshold:.3f}"
)

print(
    f"Validation balanced accuracy: "
    f"{best_row['balanced_accuracy']:.4f}"
)

print(
    "\nIMPORTANT: "
    "Threshold selected using VALIDATION data only."
)


# ============================================================
# APPLY FROZEN THRESHOLD TO TEST
# ============================================================

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_cm = confusion_matrix(
    y_test,
    test_predictions
)


# ============================================================
# TEST RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TEST RESULTS USING FROZEN VALIDATION THRESHOLD")
print("=" * 70)

print(f"Threshold          : {best_threshold:.3f}")
print(f"Accuracy            : {test_accuracy:.4f}")
print(f"Balanced Accuracy   : {test_balanced_accuracy:.4f}")
print(f"Precision           : {test_precision:.4f}")
print(f"Recall              : {test_recall:.4f}")
print(f"F1 Score            : {test_f1:.4f}")

print("\nPrediction distribution:")
print(
    pd.Series(test_predictions)
    .value_counts()
    .sort_index()
)

print("\nConfusion Matrix:")
print(test_cm)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_PATH = (
    DATA_DIR /
    "AAPL_5day_rf_threshold_results.csv"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(OUTPUT_PATH)

print("\nThreshold evaluation complete.")
