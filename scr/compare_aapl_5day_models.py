from pathlib import Path

import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "model_comparison"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_PATH = DATA_DIR / "AAPL_5day_normalized_train.parquet"

VALIDATION_PATH = (
    DATA_DIR / "AAPL_5day_normalized_validation.parquet"
)

TEST_PATH = DATA_DIR / "AAPL_5day_normalized_test.parquet"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "target"
DATE_COLUMN = "date"

RANDOM_STATE = 42

# Majority baseline predicts the most common class
# from the training dataset for every row.
MAJORITY_CLASS = None


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("LOADING NORMALIZED DATASETS")
print("=" * 75)

train_df = pd.read_parquet(TRAIN_PATH)
validation_df = pd.read_parquet(VALIDATION_PATH)
test_df = pd.read_parquet(TEST_PATH)

print(f"Train shape:       {train_df.shape}")
print(f"Validation shape:  {validation_df.shape}")
print(f"Test shape:        {test_df.shape}")


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

def prepare_xy(dataframe):
    """
    Separates model features and target.

    The date column is removed because raw dates should not be
    directly supplied to the machine-learning models.
    """

    X = dataframe.drop(
        columns=[TARGET_COLUMN, DATE_COLUMN],
        errors="ignore",
    )

    y = dataframe[TARGET_COLUMN].astype(int)

    return X, y


X_train, y_train = prepare_xy(train_df)
X_validation, y_validation = prepare_xy(validation_df)
X_test, y_test = prepare_xy(test_df)


# ============================================================
# FEATURE CHECKS
# ============================================================

print("\n" + "=" * 75)
print("FEATURE CHECKS")
print("=" * 75)

print(f"Number of features: {X_train.shape[1]}")

print("\nFeatures used by the models:")
for feature in X_train.columns:
    print(f" - {feature}")

# ------------------------------------------------------------
# Check for future-return leakage
# ------------------------------------------------------------

future_return_features = [
    column
    for column in X_train.columns
    if "future_return" in column.lower()
]

print("\nFuture-return features found:")
print(future_return_features)

if future_return_features:
    raise ValueError(
        "Future-return features are present in the model input. "
        "Remove them before training.")

# Final feature-check summary

print("\nFeature checks completed successfully.")
print("Historical price, volume, technical-indicator, and rolling features")
print("are allowed because they are calculated from information available")
print("on or before the prediction date.")

print("\n" + "=" * 75)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("TARGET DISTRIBUTION")
print("=" * 75)

print("Training target distribution:")
print(y_train.value_counts().sort_index())

print("\nValidation target distribution:")
print(y_validation.value_counts().sort_index())

print("\nTest target distribution:")
print(y_test.value_counts().sort_index())


# ============================================================
# MAJORITY-CLASS BASELINE
# ============================================================

majority_class = int(y_train.mode()[0])

baseline_predictions = np.full(
    shape=len(y_test),
    fill_value=majority_class,
    dtype=int,
)

baseline_balanced_accuracy = balanced_accuracy_score(
    y_test,
    baseline_predictions,
)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions,
    zero_division=0,
)

print("\n" + "=" * 75)
print("MAJORITY-CLASS BASELINE")
print("=" * 75)

print(f"Majority class from training data: {majority_class}")
print(
    "Baseline prediction distribution:",
    pd.Series(baseline_predictions).value_counts().sort_index().to_dict(),
)
print(f"Baseline balanced accuracy: {baseline_balanced_accuracy:.4f}")
print(f"Baseline F1-score:           {baseline_f1:.4f}")


# ============================================================
# DEFINE MODELS
# ============================================================

models = {
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
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X, y, split_name):
    """
    Evaluates a fitted model on one dataset split.
    """

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

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    )

    prediction_distribution = (
        pd.Series(predictions)
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .to_dict()
    )

    actual_distribution = (
        pd.Series(y)
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .to_dict()
    )

    print(f"\n{split_name} results:")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"F1-score:           {f1:.4f}")

    print("\nConfusion matrix:")
    print(matrix)

    print("\nActual distribution:")
    print(actual_distribution)

    print("\nPrediction distribution:")
    print(prediction_distribution)

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["DOWN", "UP"],
            zero_division=0,
        )
    )

    return {
        "balanced_accuracy": float(balanced_accuracy),
        "f1_score": float(f1),
        "confusion_matrix": matrix.tolist(),
        "actual_distribution": {
            str(key): int(value)
            for key, value in actual_distribution.items()
        },
        "prediction_distribution": {
            str(key): int(value)
            for key, value in prediction_distribution.items()
        },
        "predicts_both_classes": bool(
            len(np.unique(predictions)) == 2
        ),
    }


# ============================================================
# TRAIN AND EVALUATE ALL MODELS
# ============================================================

all_results = {}

fitted_models = {}

for model_name, model in models.items():

    print("\n" + "=" * 75)
    print(f"TRAINING: {model_name}")
    print("=" * 75)

    model.fit(X_train, y_train)

    fitted_models[model_name] = model

    train_results = evaluate_model(
        model,
        X_train,
        y_train,
        "TRAIN",
    )

    validation_results = evaluate_model(
        model,
        X_validation,
        y_validation,
        "VALIDATION",
    )

    test_results = evaluate_model(
        model,
        X_test,
        y_test,
        "TEST",
    )

    all_results[model_name] = {
        "train": train_results,
        "validation": validation_results,
        "test": test_results,
    }


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

comparison_rows = []

for model_name, results in all_results.items():

    train_result = results["train"]
    validation_result = results["validation"]
    test_result = results["test"]

    comparison_rows.append(
        {
            "model": model_name,

            "train_balanced_accuracy":
                train_result["balanced_accuracy"],

            "validation_balanced_accuracy":
                validation_result["balanced_accuracy"],

            "test_balanced_accuracy":
                test_result["balanced_accuracy"],

            "train_f1":
                train_result["f1_score"],

            "validation_f1":
                validation_result["f1_score"],

            "test_f1":
                test_result["f1_score"],

            "test_predicts_both_classes":
                test_result["predicts_both_classes"],
        }
    )


comparison_df = pd.DataFrame(comparison_rows)

comparison_df = comparison_df.sort_values(
    by=[
        "test_balanced_accuracy",
        "test_f1",
    ],
    ascending=False,
).reset_index(drop=True)


print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print(comparison_df.to_string(index=False))


# ============================================================
# BEST MODEL SELECTION
# ============================================================

# A model must beat the majority baseline by at least 2 percentage
# points and must predict both UP and DOWN classes.
minimum_required_improvement = 0.02

comparison_df["reliable_model"] = (
    (comparison_df["test_balanced_accuracy"]
     >= baseline_balanced_accuracy + minimum_required_improvement)
    & (comparison_df["test_predicts_both_classes"] == True)
)

eligible_models = comparison_df[
    comparison_df["reliable_model"]
].copy()

if eligible_models.empty:

    best_model_name = None

    print("\n" + "=" * 75)
    print("NO RELIABLE MODEL SELECTED")
    print("=" * 75)

    print(
        "No model passed the reliability conditions:"
    )
    print(
        f"1. Test balanced accuracy must be at least "
        f"{minimum_required_improvement:.2f} above the majority baseline."
    )
    print(
        "2. Model must predict both UP and DOWN classes."
    )

else:

    best_model_name = eligible_models.iloc[0]["model"]

    print("\n" + "=" * 75)
    print("RELIABLE MODEL SELECTED")
    print("=" * 75)

    print(f"Selected model: {best_model_name}")

    best_test_result = all_results[best_model_name]["test"]

    print(
        "Test balanced accuracy:",
        f"{best_test_result['balanced_accuracy']:.4f}",
    )

    print(
        "Test F1-score:",
        f"{best_test_result['f1_score']:.4f}",
    )

    print(
        "Majority baseline balanced accuracy:",
        f"{baseline_balanced_accuracy:.4f}",
    )


# ============================================================
# SAVE RESULTS
# ============================================================

comparison_path = RESULTS_DIR / "model_comparison.csv"
comparison_df.drop(columns=["reliable_model"], errors="ignore").to_csv(
    comparison_path,
    index=False,
)

results_json_path = RESULTS_DIR / "all_model_results.json"

json_results = {
    "majority_baseline": {
        "majority_class": majority_class,
        "balanced_accuracy": baseline_balanced_accuracy,
        "f1_score": baseline_f1,
        "prediction_distribution": {
            "0": int((baseline_predictions == 0).sum()),
            "1": int((baseline_predictions == 1).sum()),
        },
    },
    "models": all_results,
    "best_model": best_model_name,
}

with open(results_json_path, "w", encoding="utf-8") as file:
    json.dump(
        json_results,
        file,
        indent=4,
    )


# ============================================================
# SAVE BEST MODEL
# ============================================================

best_model_path = None

if best_model_name is not None:

    best_model_path = RESULTS_DIR / "best_aapl_5day_model.joblib"

    joblib.dump(
        fitted_models[best_model_name],
        best_model_path,
    )

    print("\nBest model saved at:")
    print(best_model_path)


# ============================================================
# SAVE CONFUSION MATRIX PLOTS
# ============================================================

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("dark_background")

obsidian_cyan_cmap = LinearSegmentedColormap.from_list(
    "obsidian_cyan", ["#161B22", "#094850", "#00B4B6", "#00F5D4"])

for model_name, results in all_results.items():
    safe_model_name = model_name.lower().replace(" ", "_")
    matrix = np.array(results["test"]["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=300)
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    image = ax.imshow(matrix, cmap=obsidian_cyan_cmap, aspect="auto")

    total_samples = matrix.sum()
    max_val = matrix.max()
    quadrant_labels = [["TN", "FP"], ["FN", "TP"]]

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            val = matrix[row, col]
            pct = (val / total_samples) * 100
            q_name = quadrant_labels[row][col]

            text_color = "#0D1117" if val > max_val * 0.6 else "#FFFFFF"
            sub_color = "#21262D" if val > max_val * 0.6 else "#8B949E"

            ax.text(
                col,
                row - 0.08,
                f"{val:,}",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color=text_color,)

            ax.text(
                col,
                row + 0.16,
                f"{q_name} ({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="medium",
                color=sub_color,)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["DOWN (0)", "UP (1)"], fontsize=10, fontweight="semibold", color="#E6EDF3")
    ax.set_yticklabels(["DOWN (0)", "UP (1)"], fontsize=10, fontweight="semibold", color="#E6EDF3")
    ax.set_xlabel("Predicted Trajectory", color="#8B949E", fontsize=10.5, labelpad=12)
    ax.set_ylabel("Ground Truth (Actual)", color="#8B949E", fontsize=10.5, labelpad=12)

    for spine in ax.spines.values():
        spine.set_color("#30363D")
        spine.set_linewidth(1.2)

    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color="#0D1117", linestyle="-", linewidth=2.5)

    accuracy = (matrix[0, 0] + matrix[1, 1]) / total_samples * 100
    fig.text(
        0.08,
        0.96,
        f"{model_name.upper()} — Test Confusion Matrix",
        fontsize=13.5,
        fontweight="bold",
        color="#FFFFFF",
        ha="left",)
    fig.text(
        0.08,
        0.915,
        f"Test Samples: {total_samples:,}   |   Overall Accuracy: {accuracy:.2f}%",
        fontsize=9.5,
        color="#00F5D4",
        ha="left",)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.05)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8.5, colors="#8B949E", length=0)
    cbar.set_label("Sample Intensity", color="#8B949E", fontsize=9, labelpad=8)

    plt.subplots_adjust(top=0.84, bottom=0.14, left=0.15, right=0.92)
    confusion_matrix_path = RESULTS_DIR / f"{safe_model_name}_test_confusion_matrix.png"
    plt.savefig(
        confusion_matrix_path,
        dpi=300,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("MODEL COMPARISON COMPLETED")
print("=" * 75)

print(f"Comparison CSV: {comparison_path}")
print(f"Results JSON:   {results_json_path}")

print("\nGenerated confusion matrices:")

for model_name in models:
    safe_model_name = (
        model_name.lower()
        .replace(" ", "_"))

    from pathlib import Path

confusion_matrix_path = (
    RESULTS_DIR / f"{safe_model_name}_test_confusion_matrix.png")

CYAN = "\033[96m"
GREEN = "\033[92m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"

file_size_kb = (
    confusion_matrix_path.stat().st_size / 1024
    if confusion_matrix_path.exists()
    else 0
)

print(f"{GRAY}┌─{RESET} {BOLD}{CYAN}ARTIFACT GENERATED{RESET} {GRAY}{'─' * 42}{RESET}")
print(f"{GRAY}│{RESET}  {GREEN}✔{RESET} Model Target : {BOLD}{model_name}{RESET}")
print(f"{GRAY}│{RESET}  {GRAY}📁 Destination  :{RESET} {confusion_matrix_path.parent}")
print(f"{GRAY}│{RESET}  {GRAY}🖼️  File Name    :{RESET} {CYAN}{confusion_matrix_path.name}{RESET}")
if file_size_kb > 0:
    print(f"{GRAY}│{RESET}  {GRAY}⚡ Resolution   :{RESET} 300 DPI ({file_size_kb:.1f} KB)")
print(f"{GRAY}└{'─' * 62}{RESET}\n")
print("\n" + "=" * 75)
print("FINAL MODEL DECISION")
print("=" * 75)

if best_model_name is not None:
    best_test_result = all_results[best_model_name]["test"]

    print(f"Selected model: {best_model_name}")
    print(
        f"Test balanced accuracy: "
        f"{best_test_result['balanced_accuracy']:.4f}"
    )
    print(
        f"Test F1-score: "
        f"{best_test_result['f1_score']:.4f}"
    )
    print(
        f"Majority baseline balanced accuracy: "
        f"{baseline_balanced_accuracy:.4f}"
    )

    print("\nModel passed the reliability checks.")

    if best_model_path is not None:
        print(f"\nBest model saved at:\n{best_model_path}")

else:
    best_observed_row = comparison_df.iloc[0]

    print("No reliable model selected.")
    print(
        "All evaluated models performed close to or below "
        "the majority-class baseline."
    )

    print(
        f"\nBest observed model: "
        f"{best_observed_row['model']}"
    )
    print(
        f"Best test balanced accuracy: "
        f"{best_observed_row['test_balanced_accuracy']:.4f}"
    )
    print(
        f"Best test F1-score: "
        f"{best_observed_row['test_f1']:.4f}"
    )
    print(
        f"Majority baseline balanced accuracy: "
        f"{baseline_balanced_accuracy:.4f}"
    )

    print(
        "\nRecommendation: Do not deploy the current models. "
        "Further feature validation, target validation, and "
        "time-series tuning are required."
    )

print("\n" + "=" * 75)

