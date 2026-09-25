import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path("data/processed/aapl_5day_relative.parquet")
MODEL_DIR = Path("results/relative_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(DATA_PATH)

# Ensure chronological order
df = df.sort_values("date").reset_index(drop=True)

# ============================================================
# SPLIT
# ============================================================

train = df[
    (df["date"] >= "2015-10-16") &
    (df["date"] <= "2023-05-25")
].copy()

validation = df[
    (df["date"] >= "2023-05-26") &
    (df["date"] <= "2025-01-15")
].copy()

test = df[
    (df["date"] >= "2025-01-16") &
    (df["date"] <= "2026-09-04")
].copy()

# ============================================================
# FEATURES / TARGET
# ============================================================

DROP_COLUMNS = ["date", "target"]

features = [
    c for c in df.columns
    if c not in DROP_COLUMNS
]

X_train = train[features]
y_train = train["target"].astype(int)

X_val = validation[features]
y_val = validation["target"].astype(int)

X_test = test[features]
y_test = test["target"].astype(int)

print("=" * 70)
print("RELATIVE FEATURE MODEL TRAINING")
print("=" * 70)

print(f"Features: {len(features)}")
print(f"Train: {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test: {X_test.shape}")

# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=42,
            ),
        ),
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]),

    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=3,
                random_state=42,
            ),
        ),
    ]),
}

# ============================================================
# TRAIN + EVALUATE
# ============================================================

results = []
fitted_models = {}

for name, model in models.items():

    print("\n" + "=" * 70)
    print(f"TRAINING: {name}")
    print("=" * 70)

    model.fit(X_train, y_train)

    fitted_models[name] = model

    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    # Metrics
    train_bal = balanced_accuracy_score(y_train, train_pred)
    val_bal = balanced_accuracy_score(y_val, val_pred)
    test_bal = balanced_accuracy_score(y_test, test_pred)

    train_f1 = f1_score(y_train, train_pred, zero_division=0)
    val_f1 = f1_score(y_val, val_pred, zero_division=0)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)

    print(
        f"TRAIN       | Balanced Accuracy: {train_bal:.4f} | "
        f"F1: {train_f1:.4f}"
    )

    print(
        f"VALIDATION  | Balanced Accuracy: {val_bal:.4f} | "
        f"F1: {val_f1:.4f}"
    )

    print(
        f"TEST        | Balanced Accuracy: {test_bal:.4f} | "
        f"F1: {test_f1:.4f}"
    )

    print("\nTest Confusion Matrix:")
    print(confusion_matrix(y_test, test_pred, labels=[0, 1]))

    print("\nTest Classification Report:")
    print(
        classification_report(
            y_test,
            test_pred,
            labels=[0, 1],
            target_names=["DOWN", "UP"],
            zero_division=0,
        )
    )

    results.append({
        "Model": name,
        "Features": len(features),

        "Train_Balanced_Accuracy": train_bal,
        "Train_F1": train_f1,

        "Validation_Balanced_Accuracy": val_bal,
        "Validation_F1": val_f1,

        "Test_Balanced_Accuracy": test_bal,
        "Test_F1": test_f1,
    })

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("FINAL RELATIVE FEATURE RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))

results_path = MODEL_DIR / "relative_feature_results.csv"
results_df.to_csv(results_path, index=False)

# Save every model
for name, model in fitted_models.items():

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + ".joblib"
    )

    path = MODEL_DIR / filename
    joblib.dump(model, path)

print("\nModels saved to:")
print(MODEL_DIR.resolve())

print("\nResults saved to:")
print(results_path.resolve())