from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "feature_ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "AAPL_5day_normalized_train.parquet"
VAL_PATH = DATA_DIR / "AAPL_5day_normalized_validation.parquet"
TEST_PATH = DATA_DIR / "AAPL_5day_normalized_test.parquet"


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_parquet(TRAIN_PATH)
val_df = pd.read_parquet(VAL_PATH)
test_df = pd.read_parquet(TEST_PATH)

TARGET = "target"

DROP_COLS = ["date", TARGET]

X_train_all = train_df.drop(columns=DROP_COLS, errors="ignore")
y_train = train_df[TARGET]

X_val_all = val_df.drop(columns=DROP_COLS, errors="ignore")
y_val = val_df[TARGET]

X_test_all = test_df.drop(columns=DROP_COLS, errors="ignore")
y_test = test_df[TARGET]


print("=" * 80)
print("NORTHGATE AI — AAPL 5-DAY FEATURE ABLATION")
print("=" * 80)

print(f"Train shape:      {train_df.shape}")
print(f"Validation shape: {val_df.shape}")
print(f"Test shape:       {test_df.shape}")


# ============================================================
# FEATURE GROUPS
# ============================================================

groups = {

    "AAPL Technical": [
        "Features_AAPL_return_1d",
        "Features_AAPL_return_5d",
        "Features_AAPL_return_20d",
        "Features_AAPL_return_60d",
        "Features_AAPL_log_return",
        "Features_AAPL_price_to_ma20",
        "Features_AAPL_price_to_ma50",
        "Features_AAPL_price_to_ma200",
        "Features_AAPL_volatility_5d",
        "Features_AAPL_volatility_20d",
        "Features_AAPL_intraday_range",
        "Features_AAPL_volume_change",
        "Features_AAPL_volume_ma20",
    ],

    "Market": [
        "Features_market_return_1d",
        "Features_market_return_5d",
        "Features_market_return_20d",
        "Features_market_volatility_20d",
    ],

    "Macro": [
        "Features_CPIAUCSL",
        "Features_UNRATE",
        "Features_cpi_change",
        "Features_unemployment_change",
    ],

    "Calendar": [
        "Features_day_of_week",
        "Features_month",
        "Features_quarter",
    ],
}


# ============================================================
# VERIFY FEATURES
# ============================================================

print("\n" + "=" * 80)
print("FEATURE GROUP CHECK")
print("=" * 80)

for group_name, features in groups.items():

    missing = [
        feature
        for feature in features
        if feature not in X_train_all.columns
    ]

    print(f"\n{group_name}:")
    print(f"  Requested: {len(features)}")
    print(f"  Missing:   {len(missing)}")

    if missing:
        for feature in missing:
            print(f"    MISSING -> {feature}")


# ============================================================
# HELPER
# ============================================================

def clean_data(X_train, X_val, X_test):

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

    medians = X_train.median()

    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    return X_train, X_val, X_test


# ============================================================
# MODEL
# ============================================================

def evaluate_features(feature_list):

    feature_list = [
        feature
        for feature in feature_list
        if feature in X_train_all.columns
    ]

    X_train = X_train_all[feature_list].copy()
    X_val = X_val_all[feature_list].copy()
    X_test = X_test_all[feature_list].copy()

    X_train, X_val, X_test = clean_data(
        X_train,
        X_val,
        X_test
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_ba = balanced_accuracy_score(
        y_val,
        val_pred
    )

    test_ba = balanced_accuracy_score(
        y_test,
        test_pred
    )

    val_f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        zero_division=0
    )

    return {
        "features": len(feature_list),
        "validation_balanced_accuracy": val_ba,
        "test_balanced_accuracy": test_ba,
        "validation_f1": val_f1,
        "test_f1": test_f1,
        "validation_positive_rate": val_pred.mean(),
        "test_positive_rate": test_pred.mean(),
    }


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

experiments = {}


# Individual groups

for group_name, features in groups.items():

    experiments[group_name] = features


# Combinations

experiments["Technical + Market"] = (
    groups["AAPL Technical"]
    + groups["Market"]
)

experiments["Technical + Macro"] = (
    groups["AAPL Technical"]
    + groups["Macro"]
)

experiments["Technical + Calendar"] = (
    groups["AAPL Technical"]
    + groups["Calendar"]
)

experiments["Technical + Market + Macro"] = (
    groups["AAPL Technical"]
    + groups["Market"]
    + groups["Macro"]
)

experiments["Technical + Market + Calendar"] = (
    groups["AAPL Technical"]
    + groups["Market"]
    + groups["Calendar"]
)

experiments["Technical + Macro + Calendar"] = (
    groups["AAPL Technical"]
    + groups["Macro"]
    + groups["Calendar"]
)

experiments["ALL FEATURES"] = (
    groups["AAPL Technical"]
    + groups["Market"]
    + groups["Macro"]
    + groups["Calendar"]
)


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []

print("\n" + "=" * 80)
print("RUNNING FEATURE ABLATION EXPERIMENTS")
print("=" * 80)

for experiment_name, feature_list in experiments.items():

    print("\n" + "-" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Features:   {len(feature_list)}")

    metrics = evaluate_features(feature_list)

    row = {
        "experiment": experiment_name,
        **metrics,
    }

    results.append(row)

    print(
        f"Validation BA : "
        f"{metrics['validation_balanced_accuracy']:.4f}"
    )

    print(
        f"Test BA       : "
        f"{metrics['test_balanced_accuracy']:.4f}"
    )

    print(
        f"Validation F1 : "
        f"{metrics['validation_f1']:.4f}"
    )

    print(
        f"Test F1       : "
        f"{metrics['test_f1']:.4f}"
    )

    print(
        f"Test UP rate  : "
        f"{metrics['test_positive_rate']:.4f}"
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "validation_balanced_accuracy",
    ascending=False
)

output_path = (
    RESULTS_DIR
    / "aapl_5day_feature_ablation_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FEATURE ABLATION SUMMARY")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)

print("\n" + "=" * 80)
print("IMPORTANT")
print("=" * 80)

print(
    "Validation results are used for feature selection."
)

print(
    "Test results are reported separately and must NOT "
    "be used to tune the model."
)

print(
    "\nResults saved to:"
)

print(output_path)

print("\nFEATURE ABLATION COMPLETED")