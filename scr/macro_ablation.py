from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aapl_5day_clean.parquet"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "macro_ablation_classification"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "target"
DATE_COLUMN = "date"
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("MACRO FEATURE ABLATION - CLASSIFICATION")
print("=" * 75)

df = pd.read_parquet(DATA_PATH)

df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

print(f"Dataset: {DATA_PATH}")
print(f"Rows: {len(df)}")


# ============================================================
# DEFINE FEATURES
# ============================================================

excluded_columns = {
    DATE_COLUMN,
    TARGET,
    "target_from_return",
    "future_return_5d",
}

all_features = [
    column
    for column in df.columns
    if column not in excluded_columns
]


# ============================================================
# IDENTIFY MACRO FEATURES
# ============================================================

macro_keywords = [
    "macro",
    "dgs10",
    "dgs3mo",
    "cpiaucsl",
    "unrate",
    "yield_spread",
    "cpi_change",
    "unemployment_change",
]

macro_features = [
    column
    for column in all_features
    if any(
        keyword in str(column).lower()
        for keyword in macro_keywords
    )
]

non_macro_features = [
    column
    for column in all_features
    if column not in macro_features
]


print("\n" + "=" * 75)
print("FEATURE SUMMARY")
print("=" * 75)

print(f"Total features:      {len(all_features)}")
print(f"Macro features:      {len(macro_features)}")
print(f"Non-macro features:  {len(non_macro_features)}")

print("\nMacro features:")
for column in macro_features:
    print(f" - {column}")


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train_df = df.iloc[:1915].copy()
validation_df = df.iloc[1915:2326].copy()
test_df = df.iloc[2326:].copy()

print("\n" + "=" * 75)
print("DATA SPLIT")
print("=" * 75)

print(f"Train:       {len(train_df)}")
print(f"Validation:  {len(validation_df)}")
print(f"Test:        {len(test_df)}")

print(
    f"\nTrain dates: "
    f"{train_df[DATE_COLUMN].min()} -> "
    f"{train_df[DATE_COLUMN].max()}"
)

print(
    f"Validation dates: "
    f"{validation_df[DATE_COLUMN].min()} -> "
    f"{validation_df[DATE_COLUMN].max()}"
)

print(
    f"Test dates: "
    f"{test_df[DATE_COLUMN].min()} -> "
    f"{test_df[DATE_COLUMN].max()}"
)


# ============================================================
# TARGET
# ============================================================

y_train = train_df[TARGET].astype(int)
y_validation = validation_df[TARGET].astype(int)
y_test = test_df[TARGET].astype(int)

print("\nTarget distribution:")
print("Train:")
print(y_train.value_counts().sort_index())

print("\nValidation:")
print(y_validation.value_counts().sort_index())

print("\nTest:")
print(y_test.value_counts().sort_index())


# ============================================================
# MODELS
# Same configuration as compare_aapl_5day_models.py
# ============================================================

def create_models():

    return {
        "Logistic Regression": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=8,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "Gradient Boosting": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.03,
                        max_depth=2,
                        min_samples_leaf=8,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, X, y):

    predictions = model.predict(X)

    balanced_accuracy = balanced_accuracy_score(
        y,
        predictions,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    return balanced_accuracy, f1


# ============================================================
# RUN ABLATION
# ============================================================

results = []

experiments = [
    (
        "WITH_MACRO",
        all_features,
    ),
    (
        "WITHOUT_MACRO",
        non_macro_features,
    ),
]


for feature_set_name, selected_features in experiments:

    print("\n" + "=" * 75)
    print(f"FEATURE SET: {feature_set_name}")
    print("=" * 75)

    print(f"Number of features: {len(selected_features)}")

    X_train = train_df[selected_features]
    X_validation = validation_df[selected_features]
    X_test = test_df[selected_features]

    models = create_models()

    for model_name, model in models.items():

        print("\n" + "-" * 75)
        print(f"TRAINING: {model_name}")
        print("-" * 75)

        model.fit(
            X_train,
            y_train,
        )

        train_bal_acc, train_f1 = evaluate(
            model,
            X_train,
            y_train,
        )

        validation_bal_acc, validation_f1 = evaluate(
            model,
            X_validation,
            y_validation,
        )

        test_bal_acc, test_f1 = evaluate(
            model,
            X_test,
            y_test,
        )

        print(
            f"TRAIN       | "
            f"Balanced Accuracy: {train_bal_acc:.4f} | "
            f"F1: {train_f1:.4f}"
        )

        print(
            f"VALIDATION  | "
            f"Balanced Accuracy: {validation_bal_acc:.4f} | "
            f"F1: {validation_f1:.4f}"
        )

        print(
            f"TEST        | "
            f"Balanced Accuracy: {test_bal_acc:.4f} | "
            f"F1: {test_f1:.4f}"
        )

        results.append(
            {
                "Feature_Set": feature_set_name,
                "Model": model_name,
                "Features": len(selected_features),

                "Train_Balanced_Accuracy": train_bal_acc,
                "Train_F1": train_f1,

                "Validation_Balanced_Accuracy": validation_bal_acc,
                "Validation_F1": validation_f1,

                "Test_Balanced_Accuracy": test_bal_acc,
                "Test_F1": test_f1,
            }
        )


# ============================================================
# FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_path = (
    RESULTS_DIR
    / "macro_ablation_classification_results.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)

print("\n" + "=" * 75)
print("FINAL MACRO ABLATION RESULTS")
print("=" * 75)

print(
    results_df[
        [
            "Feature_Set",
            "Model",
            "Features",
            "Validation_Balanced_Accuracy",
            "Validation_F1",
            "Test_Balanced_Accuracy",
            "Test_F1",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(output_path)

print("\n" + "=" * 75)
print("ABLATION COMPLETE")
print("=" * 75)