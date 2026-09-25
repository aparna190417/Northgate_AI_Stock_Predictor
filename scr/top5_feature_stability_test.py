import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_FILE = os.path.join(
    BASE_DIR, "data", "processed", "AAPL_clean_train.parquet"
)

VALIDATION_FILE = os.path.join(
    BASE_DIR, "data", "processed", "AAPL_clean_validation.parquet"
)

TEST_FILE = os.path.join(
    BASE_DIR, "data", "processed", "AAPL_clean_test.parquet"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "results", "top5_feature_stability_test"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "target"

LOCKED_THRESHOLD = 0.65

RANDOM_SEEDS = [42, 123, 2024, 7, 99]

FEATURE_COUNTS = [5, 10, 15, 24]


# ============================================================
# TOP FEATURES FROM PREVIOUS TRAIN-ONLY FEATURE AUDIT
# ============================================================

TOP_FEATURES = [
    "AAPL_price_to_ma20",
    "AAPL_volatility_20d",
    "AAPL_volatility_5d",
    "AAPL_return_5d",
    "AAPL_return_20d",
]


# ============================================================
# ALL 24 FEATURES
# ============================================================

ALL_FEATURES = [
    "AAPL_return_1d",
    "AAPL_return_5d",
    "AAPL_return_20d",
    "AAPL_return_60d",
    "AAPL_log_return",
    "AAPL_price_to_ma20",
    "AAPL_price_to_ma50",
    "AAPL_price_to_ma200",
    "AAPL_volatility_5d",
    "AAPL_volatility_20d",
    "AAPL_intraday_range",
    "AAPL_volume_change",
    "AAPL_volume_ma20",
    "market_return_1d",
    "market_return_5d",
    "market_return_20d",
    "market_volatility_20d",
    "CPIAUCSL",
    "UNRATE",
    "cpi_change",
    "unemployment_change",
    "day_of_week",
    "month",
    "quarter",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("NORTHGATE AI — TOP-5 FEATURE STABILITY TEST")
print("=" * 80)

print("""
Purpose:

1. Validate the TOP-5 feature set from the previous audit.
2. Compare TOP-5 against the existing 24-feature model.
3. Evaluate performance year-by-year.
4. Check whether TOP-5 improvement is stable.
5. Check whether the improvement survives different random seeds.

IMPORTANT:
- No final test threshold is changed.
- No production threshold is selected.
- This is a diagnostic robustness test.
""")

print("\n" + "=" * 80)
print("1. LOADING DATA")
print("=" * 80)


def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found:\n{path}")

    df = pd.read_parquet(path)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        else:
            raise ValueError(
                f"{os.path.basename(path)} does not contain a Date index."
            )

    df = df.sort_index()

    return df


train = load_dataset(TRAIN_FILE)
validation = load_dataset(VALIDATION_FILE)
test = load_dataset(TEST_FILE)

print(
    f"TRAIN      : {len(train)} rows | "
    f"{train.index.min()} → {train.index.max()}"
)

print(
    f"VALIDATION : {len(validation)} rows | "
    f"{validation.index.min()} → {validation.index.max()}"
)

print(
    f"TEST       : {len(test)} rows | "
    f"{test.index.min()} → {test.index.max()}"
)


# ============================================================
# COLUMN VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("2. COLUMN VALIDATION")
print("=" * 80)

required = ALL_FEATURES + [TARGET]

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:

    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"\n{name} missing columns:")
        for c in missing:
            print(" -", c)

        raise ValueError(
            f"{name} is missing required columns."
        )

    print(f"{name}: all required columns found.")


# ============================================================
# DATA PREPARATION
# ============================================================

def clean_xy(df, features):

    x = df[features].copy()
    y = df[TARGET].copy()

    combined = pd.concat([x, y], axis=1)

    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna()

    x = combined[features]
    y = combined[TARGET]

    return x, y


# ============================================================
# WALK-FORWARD DATA
# ============================================================

print("\n" + "=" * 80)
print("3. PREPARING WALK-FORWARD DATA")
print("=" * 80)

combined = pd.concat(
    [train, validation, test],
    axis=0
)

combined = combined[~combined.index.duplicated(keep="first")]
combined = combined.sort_index()

print(f"Combined rows: {len(combined)}")
print(
    f"Period: {combined.index.min().date()} → "
    f"{combined.index.max().date()}"
)


# ============================================================
# FEATURE SETS
# ============================================================

feature_sets = {
    5: TOP_FEATURES,
    10: TOP_FEATURES + [
        "market_volatility_20d",
        "AAPL_volume_ma20",
        "AAPL_price_to_ma50",
        "market_return_20d",
        "AAPL_intraday_range",
    ],
    15: TOP_FEATURES + [
        "market_volatility_20d",
        "AAPL_volume_ma20",
        "AAPL_price_to_ma50",
        "market_return_20d",
        "AAPL_intraday_range",
        "month",
        "AAPL_price_to_ma200",
        "UNRATE",
        "CPIAUCSL",
        "market_return_5d",
    ],
    24: ALL_FEATURES,
}


# ============================================================
# WALK-FORWARD FUNCTION
# ============================================================

def run_walk_forward(features, seed):

    predictions = []

    years = sorted(
        combined.loc[
            combined.index >= pd.Timestamp("2020-01-01")
        ].index.year.unique()
    )

    for year in years:

        train_mask = combined.index.year < year
        test_mask = combined.index.year == year

        train_year = combined.loc[train_mask]
        test_year = combined.loc[test_mask]

        if len(train_year) == 0 or len(test_year) == 0:
            continue

        x_train, y_train = clean_xy(
            train_year,
            features
        )

        x_test, y_test = clean_xy(
            test_year,
            features
        )

        common_dates = x_test.index

        if len(x_train) < 100:
            continue

        if y_train.nunique() < 2:
            continue

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )

        model.fit(x_train, y_train)

        probability = model.predict_proba(
            x_test
        )[:, 1]

        pred_df = pd.DataFrame(
            {
                "date": common_dates,
                "prediction_probability": probability,
                "actual": y_test.loc[common_dates].values,
                "year": year,
            }
        )

        predictions.append(pred_df)

    if not predictions:
        return pd.DataFrame()

    return pd.concat(
        predictions,
        ignore_index=True
    )


# ============================================================
# FEATURE COUNT TEST
# ============================================================

print("\n" + "=" * 80)
print("4. FEATURE COUNT WALK-FORWARD TEST")
print("=" * 80)


all_results = []
yearly_results = []


for feature_count, features in feature_sets.items():

    print("\n" + "-" * 80)
    print(f"TESTING TOP {feature_count} FEATURES")
    print("-" * 80)

    for f in features:
        print(" -", f)

    # Use seed 42 for the main comparison
    pred = run_walk_forward(
        features,
        seed=42
    )

    if pred.empty:
        print("No predictions generated.")
        continue

    auc = roc_auc_score(
        pred["actual"],
        pred["prediction_probability"]
    )

    spearman = spearmanr(
        pred["prediction_probability"],
        pred["actual"]
    ).statistic

    print(f"\nRows              : {len(pred):,}")
    print(f"ROC-AUC           : {auc:.4f}")
    print(f"AUC vs random     : {auc - 0.50:.4f}")
    print(f"Spearman          : {spearman:.4f}")

    all_results.append(
        {
            "feature_count": feature_count,
            "seed": 42,
            "rows": len(pred),
            "roc_auc": auc,
            "auc_vs_random": auc - 0.50,
            "spearman_direction": spearman,
        }
    )

    for year, group in pred.groupby("year"):

        if group["actual"].nunique() < 2:
            year_auc = np.nan
        else:
            year_auc = roc_auc_score(
                group["actual"],
                group["prediction_probability"]
            )

        yearly_results.append(
            {
                "feature_count": feature_count,
                "seed": 42,
                "year": year,
                "rows": len(group),
                "roc_auc": year_auc,
                "spearman_direction": spearmanr(
                    group["prediction_probability"],
                    group["actual"]
                ).statistic,
                "actual_positive_rate": group["actual"].mean(),
            }
        )


# ============================================================
# RANDOM SEED STABILITY
# ============================================================

print("\n" + "=" * 80)
print("5. RANDOM-SEED STABILITY TEST")
print("=" * 80)

seed_results = []

for feature_count in [5, 24]:

    features = feature_sets[feature_count]

    print("\n" + "-" * 80)
    print(f"TOP {feature_count} FEATURES")
    print("-" * 80)

    for seed in RANDOM_SEEDS:

        pred = run_walk_forward(
            features,
            seed=seed
        )

        if pred.empty:
            continue

        auc = roc_auc_score(
            pred["actual"],
            pred["prediction_probability"]
        )

        spear = spearmanr(
            pred["prediction_probability"],
            pred["actual"]
        ).statistic

        print(
            f"Seed={seed:<5} | "
            f"AUC={auc:.4f} | "
            f"AUC-0.50={auc - 0.50:.4f} | "
            f"Spearman={spear:.4f}"
        )

        seed_results.append(
            {
                "feature_count": feature_count,
                "seed": seed,
                "rows": len(pred),
                "roc_auc": auc,
                "auc_vs_random": auc - 0.50,
                "spearman_direction": spear,
            }
        )


# ============================================================
# SEED SUMMARY
# ============================================================

seed_df = pd.DataFrame(seed_results)

seed_summary = (
    seed_df
    .groupby("feature_count")
    .agg(
        mean_auc=("roc_auc", "mean"),
        std_auc=("roc_auc", "std"),
        min_auc=("roc_auc", "min"),
        max_auc=("roc_auc", "max"),
        mean_spearman=("spearman_direction", "mean"),
        std_spearman=("spearman_direction", "std"),
    )
    .reset_index()
)

print("\n" + "=" * 80)
print("6. RANDOM-SEED SUMMARY")
print("=" * 80)

print(seed_summary.to_string(index=False))


# ============================================================
# YEARLY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("7. YEARLY FEATURE-SET RESULTS")
print("=" * 80)

yearly_df = pd.DataFrame(yearly_results)

if not yearly_df.empty:
    print(yearly_df.to_string(index=False))


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("8. SAVING RESULTS")
print("=" * 80)

ranking_path = os.path.join(
    OUTPUT_DIR,
    "feature_count_evaluation.csv"
)

yearly_path = os.path.join(
    OUTPUT_DIR,
    "yearly_feature_evaluation.csv"
)

seed_path = os.path.join(
    OUTPUT_DIR,
    "random_seed_stability.csv"
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "random_seed_summary.csv"
)

pd.DataFrame(all_results).to_csv(
    ranking_path,
    index=False
)

yearly_df.to_csv(
    yearly_path,
    index=False
)

seed_df.to_csv(
    seed_path,
    index=False
)

seed_summary.to_csv(
    summary_path,
    index=False
)

print("Feature count evaluation:")
print(ranking_path)

print("\nYearly evaluation:")
print(yearly_path)

print("\nRandom seed results:")
print(seed_path)

print("\nRandom seed summary:")
print(summary_path)


# ============================================================
# FINAL DIAGNOSTIC SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("9. DIAGNOSTIC SUMMARY")
print("=" * 80)

if not seed_summary.empty:

    top5_summary = seed_summary[
        seed_summary["feature_count"] == 5
    ]

    full_summary = seed_summary[
        seed_summary["feature_count"] == 24
    ]

    if not top5_summary.empty and not full_summary.empty:

        top5_mean = top5_summary["mean_auc"].iloc[0]
        full_mean = full_summary["mean_auc"].iloc[0]

        print(
            f"TOP-5 mean ROC-AUC across seeds : "
            f"{top5_mean:.4f}"
        )

        print(
            f"24-feature mean ROC-AUC         : "
            f"{full_mean:.4f}"
        )

        print(
            f"Mean AUC difference             : "
            f"{top5_mean - full_mean:+.4f}"
        )

print("""
IMPORTANT:

This is a diagnostic robustness test.

A higher ROC-AUC in one configuration does NOT
automatically establish that the feature set is
better for production.

Interpret:
- overall ROC-AUC
- AUC versus random
- yearly stability
- random-seed stability
- sample size

No production threshold has been changed.
No production threshold has been selected.
The untouched final test has NOT been used to
select the feature set.
""")


print("\n" + "=" * 80)
print("NORTHGATE AI — TOP-5 FEATURE STABILITY TEST COMPLETED")
print("=" * 80)