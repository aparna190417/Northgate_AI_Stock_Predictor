import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

STOCK = "AAPL"

TRAIN_FILE = Path(
    f"data/processed/{STOCK}_train.parquet"
)

VALIDATION_FILE = Path(
    f"data/processed/{STOCK}_validation.parquet"
)

TEST_FILE = Path(
    f"data/processed/{STOCK}_test.parquet"
)

MODEL_FILE = Path(
    f"data/processed/{STOCK}_baseline_model.pkl"
)

SCALER_FILE = Path(
    f"data/processed/{STOCK}_scaler.pkl"
)

METRICS_FILE = Path(
    f"data/processed/{STOCK}_baseline_metrics.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

train_df = pd.read_parquet(TRAIN_FILE)
validation_df = pd.read_parquet(VALIDATION_FILE)
test_df = pd.read_parquet(TEST_FILE)

print(f"Train shape      : {train_df.shape}")
print(f"Validation shape : {validation_df.shape}")
print(f"Test shape       : {test_df.shape}")


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X_train = train_df.drop(columns=["target"]).copy()
y_train = train_df["target"].copy()

X_validation = validation_df.drop(columns=["target"]).copy()
y_validation = validation_df["target"].copy()

X_test = test_df.drop(columns=["target"]).copy()
y_test = test_df["target"].copy()


# ============================================================
# ENSURE SAME COLUMN ORDER
# ============================================================

X_train.columns = X_train.columns.astype(str)
X_validation.columns = X_validation.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

X_validation = X_validation[X_train.columns]
X_test = X_test[X_train.columns]


# ============================================================
# HANDLE INF AND MISSING VALUES
# ============================================================

print("\nCleaning feature values...")

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_validation = X_validation.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

# Median values are calculated only from training data
train_medians = X_train.median()

X_train = X_train.fillna(train_medians)
X_validation = X_validation.fillna(train_medians)
X_test = X_test.fillna(train_medians)

print("Missing values handled successfully.")


# ============================================================
# SCALE FEATURES
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

# Fit scaler only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Only transform validation and test data
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling complete.")


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression baseline...")

model = LogisticRegression(
    max_iter=2000,
    class_weight=None,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)

print("Model training complete.")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X, y, dataset_name):

    predictions = model.predict(X)

    print("\n" + "=" * 60)
    print(f"{dataset_name.upper()} RESULTS")
    print("=" * 60)

    # Prediction distribution
    print("\nPrediction distribution:")
    print(
        pd.Series(predictions)
        .value_counts()
        .sort_index()
    )

    # Actual target distribution
    print("\nActual target distribution:")
    print(
        pd.Series(y)
        .value_counts()
        .sort_index()
    )

    # Metrics
    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    print("\nMetrics:")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y, predictions))

    return {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


# ============================================================
# EVALUATE ON TRAIN DATA
# ============================================================

train_metrics = evaluate_model(
    model=model,
    X=X_train_scaled,
    y=y_train,
    dataset_name="Train"
)


# ============================================================
# EVALUATE ON VALIDATION DATA
# ============================================================

validation_metrics = evaluate_model(
    model=model,
    X=X_validation_scaled,
    y=y_validation,
    dataset_name="Validation"
)


# ============================================================
# EVALUATE ON TEST DATA
# ============================================================

test_metrics = evaluate_model(
    model=model,
    X=X_test_scaled,
    y=y_test,
    dataset_name="Test"
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING MODEL FILES")
print("=" * 60)

joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)

print(f"Model saved  : {MODEL_FILE}")
print(f"Scaler saved : {SCALER_FILE}")


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    [
        train_metrics,
        validation_metrics,
        test_metrics
    ]
)

metrics_df.to_csv(
    METRICS_FILE,
    index=False
)

print(f"Metrics saved: {METRICS_FILE}")


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("BASELINE TRAINING COMPLETE")
print("=" * 60)

print("\nFiles created successfully:")
print(f"1. {MODEL_FILE}")
print(f"2. {SCALER_FILE}")
print(f"3. {METRICS_FILE}")

print("\nReady for the next model comparison 🚀")