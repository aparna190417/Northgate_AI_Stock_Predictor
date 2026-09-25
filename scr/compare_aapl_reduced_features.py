from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

RESULTS_DIR = ROOT / "results" / "reduced_feature_models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load datasets
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

feature_sets = {
    "top_10": ranked_features[:10],
    "top_15": ranked_features[:15],
    "top_20": ranked_features[:20],
    "all_24": all_features,
}


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    return {
        "train_balanced_accuracy": balanced_accuracy_score(
            y_train, train_pred
        ),
        "validation_balanced_accuracy": balanced_accuracy_score(
            y_val, val_pred
        ),
        "test_balanced_accuracy": balanced_accuracy_score(
            y_test, test_pred
        ),
        "train_f1": f1_score(y_train, train_pred, zero_division=0),
        "validation_f1": f1_score(y_val, val_pred, zero_division=0),
        "test_f1": f1_score(y_test, test_pred, zero_division=0),
    }


models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=0.1,
                random_state=42
            )
        )
    ]),
    "random_forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=10,
        random_state=42
    ),
}


results = []

print("=" * 80)
print("REDUCED FEATURE MODEL COMPARISON")
print("=" * 80)

for feature_set_name, selected_features in feature_sets.items():
    print(f"\nFeature set: {feature_set_name}")
    print(f"Number of features: {len(selected_features)}")
    print("Features:")
    print(", ".join(selected_features))

    X_train = train_df[selected_features]
    y_train = train_df["target"]

    X_val = val_df[selected_features]
    y_val = val_df["target"]

    X_test = test_df[selected_features]
    y_test = test_df["target"]

    for model_name, model in models.items():
        metrics = evaluate_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test
        )

        row = {
            "feature_set": feature_set_name,
            "feature_count": len(selected_features),
            "model": model_name,
            **metrics
        }

        results.append(row)

        print(
            f"{model_name:22s} | "
            f"Train BA: {metrics['train_balanced_accuracy']:.4f} | "
            f"Val BA: {metrics['validation_balanced_accuracy']:.4f} | "
            f"Test BA: {metrics['test_balanced_accuracy']:.4f} | "
            f"Test F1: {metrics['test_f1']:.4f}"
        )


results_df = pd.DataFrame(results)

results_path = RESULTS_DIR / "reduced_feature_comparison.csv"
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 80)
print("BEST RESULTS BY TEST BALANCED ACCURACY")
print("=" * 80)

best_results = results_df.sort_values(
    "test_balanced_accuracy",
    ascending=False
)

print(best_results.to_string(index=False))

print(f"\nSaved results to: {results_path}")
print("\nREDUCED FEATURE COMPARISON COMPLETED")