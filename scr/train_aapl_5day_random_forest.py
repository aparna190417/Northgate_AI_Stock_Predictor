from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

TRAIN_PATH = DATA_DIR / "AAPL_5day_train.parquet"
VALIDATION_PATH = DATA_DIR / "AAPL_5day_validation.parquet"
TEST_PATH = DATA_DIR / "AAPL_5day_test.parquet"

MODEL_PATH = DATA_DIR / "AAPL_5day_random_forest_model.pkl"
IMPORTANCE_PATH = DATA_DIR / "AAPL_5day_random_forest_feature_importance.csv"
METRICS_PATH = DATA_DIR / "AAPL_5day_random_forest_metrics.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("TRAINING AAPL 5-DAY RANDOM FOREST")
print("=" * 60)

train_df = pd.read_parquet(TRAIN_PATH)
validation_df = pd.read_parquet(VALIDATION_PATH)
test_df = pd.read_parquet(TEST_PATH)

print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {validation_df.shape}")
print(f"Test shape: {test_df.shape}")


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):
    df = df.copy()

    # Lowercase date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

        df["date_year"] = df["date"].dt.year
        df["date_month"] = df["date"].dt.month
        df["date_day"] = df["date"].dt.day
        df["date_day_of_week"] = df["date"].dt.dayofweek

        df = df.drop(columns=["date"])

    # Uppercase Date column
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
# SPLIT FEATURES AND TARGET
# ============================================================

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]

X_validation = validation_df.drop(columns=["target"])
y_validation = validation_df["target"]

X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]


# ============================================================
# REMOVE TARGET LEAKAGE
# ============================================================

# This feature is calculated using future prices.
# Therefore, it must NOT be given to the model.

leakage_columns = [
    "AAPL_future_return_5d",
]

X_train = X_train.drop(
    columns=leakage_columns,
    errors="ignore",
)

X_validation = X_validation.drop(
    columns=leakage_columns,
    errors="ignore",
)

X_test = X_test.drop(
    columns=leakage_columns,
    errors="ignore",
)


# ============================================================
# FINAL DATA CHECK
# ============================================================

print("\nLeakage columns removed:", leakage_columns)
print("Final feature count:", X_train.shape[1])

print("\nRemaining datetime columns:")
print(
    X_train.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()
)

print("\nMissing values in training data:", X_train.isna().sum().sum())


# ============================================================
# TRAIN MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X, y, dataset_name):
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    balanced_accuracy = balanced_accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)

    print("\n" + "=" * 60)
    print(f"{dataset_name.upper()} RESULTS")
    print("=" * 60)

    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Balanced Accuracy:  {balanced_accuracy:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"F1 Score:           {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y, predictions))

    print("\nPrediction Distribution:")
    print(
        pd.Series(predictions)
        .value_counts()
        .sort_index()
    )

    return {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


# ============================================================
# EVALUATE MODEL
# ============================================================

train_metrics = evaluate_model(
    model,
    X_train,
    y_train,
    "Train",
)

validation_metrics = evaluate_model(
    model,
    X_validation,
    y_validation,
    "Validation",
)

test_metrics = evaluate_model(
    model,
    X_test,
    y_test,
    "Test",
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_,
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False,
)

print("\n" + "=" * 60)
print("TOP 15 IMPORTANT FEATURES")
print("=" * 60)

print(
    feature_importance
    .head(15)
    .to_string(index=False)
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

joblib.dump(model, MODEL_PATH)

feature_importance.to_csv(
    IMPORTANCE_PATH,
    index=False,
)

metrics_df = pd.DataFrame([
    train_metrics,
    validation_metrics,
    test_metrics,
])

metrics_df.to_csv(
    METRICS_PATH,
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(MODEL_PATH)
print(IMPORTANCE_PATH)
print(METRICS_PATH)

print("\n5-day Random Forest training complete.")