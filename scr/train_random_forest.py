import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix)

# CONFIGURATION

STOCK = "AAPL"

TRAIN_FILE = Path(
    f"data/processed/{STOCK}_train.parquet")

VALIDATION_FILE = Path(
    f"data/processed/{STOCK}_validation.parquet")

TEST_FILE = Path(
    f"data/processed/{STOCK}_test.parquet")

MODEL_FILE = Path(
    f"data/processed/{STOCK}_random_forest_model.pkl")

METRICS_FILE = Path(
    f"data/processed/{STOCK}_random_forest_metrics.csv")

# LOAD DATA

print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

train_df = pd.read_parquet(TRAIN_FILE)
validation_df = pd.read_parquet(VALIDATION_FILE)
test_df = pd.read_parquet(TEST_FILE)

print(f"Train shape      : {train_df.shape}")
print(f"Validation shape : {validation_df.shape}")
print(f"Test shape       : {test_df.shape}")

# SEPARATE FEATURES AND TARGET

X_train = train_df.drop(columns=["target"]).copy()
y_train = train_df["target"].copy()

X_validation = validation_df.drop(columns=["target"]).copy()
y_validation = validation_df["target"].copy()

X_test = test_df.drop(columns=["target"]).copy()
y_test = test_df["target"].copy()

# ENSURE SAME COLUMNS

X_train.columns = X_train.columns.astype(str)
X_validation.columns = X_validation.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

X_validation = X_validation[X_train.columns]
X_test = X_test[X_train.columns]

# HANDLE INF AND MISSING VALUES

print("\nCleaning feature values...")

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan)

X_validation = X_validation.replace(
    [np.inf, -np.inf],
    np.nan)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan)

train_medians = X_train.median()

X_train = X_train.fillna(train_medians)
X_validation = X_validation.fillna(train_medians)
X_test = X_test.fillna(train_medians)

print("Missing values handled successfully.")

# TRAIN RANDOM FOREST

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1)

model.fit(
    X_train,
    y_train)

print("Random Forest training complete.")

# EVALUATION FUNCTION

def evaluate_model(model, X, y, dataset_name):

    predictions = model.predict(X)

    print("\n" + "=" * 60)
    print(f"{dataset_name.upper()} RESULTS")
    print("=" * 60)

    print("\nPrediction distribution:")
    print(
        pd.Series(predictions)
        .value_counts()
        .sort_index())

    print("\nActual target distribution:")
    print(
        pd.Series(y)
        .value_counts()
        .sort_index())

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y,
        predictions,
        zero_division=0)

    recall = recall_score(
        y,
        predictions,
        zero_division=0)

    f1 = f1_score(
        y,
        predictions,
        zero_division=0)

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
            zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y, predictions))

    return {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1}

# EVALUATE MODEL

train_metrics = evaluate_model(
    model,
    X_train,
    y_train,
    "Train")

validation_metrics = evaluate_model(
    model,
    X_validation,
    y_validation,
    "Validation")

test_metrics = evaluate_model(
    model,
    X_test,
    y_test,
    "Test")

# SAVE MODEL

print("\n" + "=" * 60)
print("SAVING RANDOM FOREST MODEL")
print("=" * 60)

joblib.dump(model, MODEL_FILE)

print(f"Model saved: {MODEL_FILE}")

# FEATURE IMPORTANCE

importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False)

importance_file = Path(
    f"data/processed/{STOCK}_random_forest_feature_importance.csv")

importance_df.to_csv(
    importance_file,
    index=False)

print(f"Feature importance saved: {importance_file}")

print("\nTop 20 important features:")
print(importance_df.head(20).to_string(index=False))

# SAVE METRICS

metrics_df = pd.DataFrame([
    train_metrics,
    validation_metrics,
    test_metrics])

metrics_df.to_csv(
    METRICS_FILE,
    index=False)

print(f"\nMetrics saved: {METRICS_FILE}")

print("\nRandom Forest training complete 🚀")