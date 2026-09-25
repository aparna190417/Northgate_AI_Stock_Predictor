from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "model_comparison_24_features"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "AAPL_5day_normalized_train.parquet"
VAL_PATH = DATA_DIR / "AAPL_5day_normalized_validation.parquet"
TEST_PATH = DATA_DIR / "AAPL_5day_normalized_test.parquet"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("AAPL 5-DAY — 24 FEATURE MODEL COMPARISON")
print("=" * 75)

train_df = pd.read_parquet(TRAIN_PATH)
val_df = pd.read_parquet(VAL_PATH)
test_df = pd.read_parquet(TEST_PATH)

print(f"\nTrain shape      : {train_df.shape}")
print(f"Validation shape : {val_df.shape}")
print(f"Test shape       : {test_df.shape}")


# ============================================================
# PREPARE FEATURES
# ============================================================

TARGET = "target"

DROP_COLUMNS = [
    "date",
    TARGET,
]


def prepare_dataset(df):
    df = df.copy()

    # Remove unwanted columns
    df = df.drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )

    return df


X_train = prepare_dataset(train_df)
y_train = train_df[TARGET].copy()

X_val = prepare_dataset(val_df)
y_val = val_df[TARGET].copy()

X_test = prepare_dataset(test_df)
y_test = test_df[TARGET].copy()


# ============================================================
# ENSURE IDENTICAL FEATURE ORDER
# ============================================================

feature_columns = X_train.columns.tolist()

X_val = X_val[feature_columns]
X_test = X_test[feature_columns]


print("\n" + "=" * 75)
print("FEATURE CHECK")
print("=" * 75)

print(f"Number of features: {len(feature_columns)}")

print("\nFeatures:")
for i, feature in enumerate(feature_columns, start=1):
    print(f"{i:02d}. {feature}")


# ============================================================
# DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 75)
print("DATA QUALITY CHECK")
print("=" * 75)

for name, X in [
    ("TRAIN", X_train),
    ("VALIDATION", X_val),
    ("TEST", X_test),
]:

    print(f"\n{name}")

    print(
        f"Rows: {len(X)}"
    )

    print(
        f"Missing values: {X.isna().sum().sum()}"
    )

    print(
        f"Infinite values: "
        f"{np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}"
    )


# ============================================================
# HANDLE INF / MISSING VALUES
# ============================================================

print("\n" + "=" * 75)
print("CLEANING FEATURE VALUES")
print("=" * 75)

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_val = X_val.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

# IMPORTANT:
# Medians are calculated ONLY from training data.
train_medians = X_train.median()

X_train = X_train.fillna(train_medians)
X_val = X_val.fillna(train_medians)
X_test = X_test.fillna(train_medians)

print("Missing/infinite values handled.")


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("TARGET DISTRIBUTION")
print("=" * 75)

for name, y in [
    ("TRAIN", y_train),
    ("VALIDATION", y_val),
    ("TEST", y_test),
]:

    print(f"\n{name}")

    counts = y.value_counts().sort_index()

    print(counts)

    print(
        "Proportions:"
    )

    print(
        y.value_counts(
            normalize=True
        ).sort_index()
    )


# ============================================================
# SCALE ONLY FOR LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 75)
print("SCALING DATA FOR LOGISTIC REGRESSION")
print("=" * 75)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Scaler fitted on TRAIN only.")


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": {
        "model": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "scaled": True,
    },

    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "scaled": False,
    },

    "Gradient Boosting": {
        "model": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
        ),
        "scaled": False,
    },
}


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    dataset_name,
    model_name,
):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(
        y,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y,
        predictions
    )

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

    cm = confusion_matrix(
        y,
        predictions
    )

    print("\n" + "=" * 75)
    print(
        f"{model_name.upper()} — "
        f"{dataset_name.upper()}"
    )
    print("=" * 75)

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : {balanced_accuracy:.4f}"
    )

    print(
        f"Precision         : {precision:.4f}"
    )

    print(
        f"Recall            : {recall:.4f}"
    )

    print(
        f"F1 Score          : {f1:.4f}"
    )

    print(
        f"Mean Probability  : {probabilities.mean():.4f}"
    )

    print(
        f"Min Probability   : {probabilities.min():.4f}"
    )

    print(
        f"Max Probability   : {probabilities.max():.4f}"
    )

    print("\nPrediction distribution:")

    print(
        pd.Series(
            predictions
        ).value_counts().sort_index()
    )

    print("\nConfusion Matrix:")

    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            zero_division=0
        )
    )

    return {
        "model": model_name,
        "dataset": dataset_name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mean_probability": probabilities.mean(),
        "min_probability": probabilities.min(),
        "max_probability": probabilities.max(),
    }


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

all_results = []


for model_name, config in models.items():

    print("\n\n")
    print("#" * 75)
    print(f"TRAINING: {model_name}")
    print("#" * 75)

    model = config["model"]

    use_scaled = config["scaled"]

    if use_scaled:

        train_input = X_train_scaled
        val_input = X_val_scaled
        test_input = X_test_scaled

    else:

        train_input = X_train
        val_input = X_val
        test_input = X_test


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        train_input,
        y_train
    )

    print(
        f"\n{model_name} training complete."
    )


    # --------------------------------------------------------
    # TRAIN EVALUATION
    # --------------------------------------------------------

    train_result = evaluate_model(
        model,
        train_input,
        y_train,
        "Train",
        model_name,
    )

    all_results.append(
        train_result
    )


    # --------------------------------------------------------
    # VALIDATION EVALUATION
    # --------------------------------------------------------

    val_result = evaluate_model(
        model,
        val_input,
        y_val,
        "Validation",
        model_name,
    )

    all_results.append(
        val_result
    )


    # --------------------------------------------------------
    # TEST EVALUATION
    # --------------------------------------------------------

    test_result = evaluate_model(
        model,
        test_input,
        y_test,
        "Test",
        model_name,
    )

    all_results.append(
        test_result
    )


    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    model_path = (
        RESULTS_DIR /
        f"aapl_5day_24feature_{filename}.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"\nModel saved: {model_path}"
    )


# ============================================================
# SAVE SCALER
# ============================================================

scaler_path = (
    RESULTS_DIR /
    "aapl_5day_24feature_scaler.pkl"
)

joblib.dump(
    scaler,
    scaler_path
)

print(
    f"\nScaler saved: {scaler_path}"
)


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# SAVE ALL RESULTS
# ============================================================

results_path = (
    RESULTS_DIR /
    "aapl_5day_24feature_model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# VALIDATION COMPARISON
# ============================================================

validation_results = (
    results_df[
        results_df["dataset"] == "Validation"
    ]
    .sort_values(
        "balanced_accuracy",
        ascending=False
    )
)


print("\n" + "=" * 75)
print("VALIDATION MODEL COMPARISON")
print("=" * 75)

print(
    validation_results[
        [
            "model",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1_score",
        ]
    ].to_string(index=False)
)


# ============================================================
# TEST COMPARISON
# ============================================================

test_results = (
    results_df[
        results_df["dataset"] == "Test"
    ]
    .sort_values(
        "balanced_accuracy",
        ascending=False
    )
)


print("\n" + "=" * 75)
print("TEST MODEL RESULTS")
print("=" * 75)

print(
    test_results[
        [
            "model",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1_score",
        ]
    ].to_string(index=False)
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("24-FEATURE MODEL COMPARISON COMPLETE")
print("=" * 75)

print(
    f"\nResults saved to:"
    f"\n{results_path}"
)

print(
    "\nIMPORTANT:"
)

print(
    "Validation results are used for model comparison."
)

print(
    "Test results are reported separately and "
    "should NOT be used to tune the model."
)

print("\nDone.")