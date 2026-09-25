# ================================================================
# NORTHGATE AI — FEATURE SET STATISTICAL SIGNIFICANCE TEST
# ================================================================
#
# Purpose:
# 1. Compare TOP-5 vs 24-feature model statistically.
# 2. Measure ROC-AUC difference.
# 3. Bootstrap confidence interval for AUC difference.
# 4. Permutation test for AUC difference.
# 5. Compare performance year-by-year.
# 6. Check whether TOP-5 improvement is statistically meaningful.
#
# IMPORTANT:
# - No production threshold is changed.
# - No production threshold is selected.
# - Final untouched TEST is NOT used for feature selection.
# - Statistical comparison uses WALK-FORWARD predictions only.
# - This script is diagnostic only.
# ================================================================

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "feature_set_statistical_significance"
)

os.makedirs(RESULT_DIR, exist_ok=True)


TRAIN_FILE = os.path.join(
    DATA_DIR,
    "AAPL_clean_train.parquet"
)

VALID_FILE = os.path.join(
    DATA_DIR,
    "AAPL_clean_validation.parquet"
)

TEST_FILE = os.path.join(
    DATA_DIR,
    "AAPL_clean_test.parquet"
)


# ================================================================
# FEATURE SETS
# ================================================================

TOP5_FEATURES = [
    "AAPL_price_to_ma20",
    "AAPL_volatility_20d",
    "AAPL_volatility_5d",
    "AAPL_return_5d",
    "AAPL_return_20d",
]


FEATURES_24 = [
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


SEEDS = [
    42,
    123,
    2024,
    7,
    99,
]


N_ESTIMATORS = 300

BOOTSTRAP_ITERATIONS = 2000

PERMUTATION_ITERATIONS = 2000


# ================================================================
# HEADER
# ================================================================

print("=" * 80)
print("NORTHGATE AI — FEATURE SET STATISTICAL SIGNIFICANCE TEST")
print("=" * 80)

print(
    """
Purpose:

1. Compare TOP-5 vs 24-feature model.
2. Estimate ROC-AUC difference.
3. Calculate bootstrap confidence interval.
4. Perform permutation significance test.
5. Compare year-by-year performance.

IMPORTANT:
- No production threshold is changed.
- No production threshold is selected.
- Final TEST is not used for feature selection.
- This is a diagnostic statistical test only.
"""
)


# ================================================================
# LOAD DATA
# ================================================================

print("=" * 80)
print("1. LOADING DATA")
print("=" * 80)


def load_dataset(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nDataset not found:\n{path}"
        )

    df = pd.read_parquet(path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    df.index = pd.to_datetime(df.index)

    return df.sort_index()


train = load_dataset(TRAIN_FILE)
validation = load_dataset(VALID_FILE)
test = load_dataset(TEST_FILE)


print(
    f"TRAIN      : {len(train)} rows | "
    f"{train.index.min().date()} → {train.index.max().date()}"
)

print(
    f"VALIDATION : {len(validation)} rows | "
    f"{validation.index.min().date()} → {validation.index.max().date()}"
)

print(
    f"TEST       : {len(test)} rows | "
    f"{test.index.min().date()} → {test.index.max().date()}"
)


# ================================================================
# COLUMN VALIDATION
# ================================================================

print("\n" + "=" * 80)
print("2. COLUMN VALIDATION")
print("=" * 80)


required_columns = set(
    TOP5_FEATURES
    + FEATURES_24
    + ["target"]
)


for name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:

    missing = sorted(
        required_columns - set(df.columns)
    )

    if missing:

        print(f"\n{name} missing columns:")

        for col in missing:
            print(" -", col)

        raise ValueError(
            f"{name} is missing required columns."
        )

    print(
        f"{name}: all required columns found."
    )


# ================================================================
# PREPARE WALK-FORWARD DATA
# ================================================================

print("\n" + "=" * 80)
print("3. PREPARING WALK-FORWARD DATA")
print("=" * 80)


# IMPORTANT:
# Train/validation/test are kept in chronological order.
# The final test is NOT used to select the feature set.

walkforward = pd.concat(
    [
        train,
        validation,
        test
    ],
    axis=0
)

walkforward = walkforward[
    ~walkforward.index.duplicated(
        keep="first"
    )
].sort_index()


print(
    f"Combined rows: {len(walkforward)}"
)

print(
    f"Period: "
    f"{walkforward.index.min().date()} → "
    f"{walkforward.index.max().date()}"
)


# ================================================================
# WALK-FORWARD PREDICTION FUNCTION
# ================================================================

def generate_walkforward_predictions(
    data,
    features,
    seed
):

    predictions = []

    dates = []

    actuals = []

    years = []

    # ------------------------------------------------------------
    # Expanding-window walk-forward
    #
    # Initial training period = original TRAIN
    # Validation + TEST are evaluated chronologically.
    #
    # The model is retrained as new observations become available.
    # ------------------------------------------------------------

    initial_train_end = train.index.max()

    evaluation_data = data[
        data.index > initial_train_end
    ].copy()

    for current_date, row in evaluation_data.iterrows():

        historical = data[
            data.index < current_date
        ]

        if len(historical) < len(train):
            continue

        X_train = historical[features]
        y_train = historical["target"]

        X_current = row[
            features
        ].to_frame().T

        if (
            X_train.isnull().any().any()
            or X_current.isnull().any().any()
        ):
            continue

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
            max_features="sqrt",
        )

        model.fit(
            X_train,
            y_train
        )

        probability = model.predict_proba(
            X_current
        )[0, 1]

        predictions.append(
            probability
        )

        actuals.append(
            int(row["target"])
        )

        dates.append(
            current_date
        )

        years.append(
            current_date.year
        )

    result = pd.DataFrame(
        {
            "Date": dates,
            "probability": predictions,
            "target": actuals,
            "year": years,
        }
    )

    return result


# ================================================================
# NOTE
# ================================================================

print(
    """
Walk-forward evaluation will compare:

TOP-5:
AAPL_price_to_ma20
AAPL_volatility_20d
AAPL_volatility_5d
AAPL_return_5d
AAPL_return_20d

VERSUS:

24-feature model.

The final test is evaluated chronologically,
but it is NOT used to decide which feature set
should be selected.
"""
)


# ================================================================
# RUN MODELS
# ================================================================

all_predictions = []

seed_summary = []


print("\n" + "=" * 80)
print("4. GENERATING WALK-FORWARD PREDICTIONS")
print("=" * 80)


for seed in SEEDS:

    print(
        f"\nRunning seed {seed}..."
    )

    print(
        "  TOP-5 model..."
    )

    pred_top5 = generate_walkforward_predictions(
        walkforward,
        TOP5_FEATURES,
        seed
    )

    pred_top5["feature_count"] = 5
    pred_top5["seed"] = seed

    print(
        "  24-feature model..."
    )

    pred_24 = generate_walkforward_predictions(
        walkforward,
        FEATURES_24,
        seed
    )

    pred_24["feature_count"] = 24
    pred_24["seed"] = seed

    all_predictions.append(
        pred_top5
    )

    all_predictions.append(
        pred_24
    )


predictions = pd.concat(
    all_predictions,
    ignore_index=True
)


# ================================================================
# OVERALL AUC
# ================================================================

print("\n" + "=" * 80)
print("5. OVERALL ROC-AUC COMPARISON")
print("=" * 80)


overall_results = []


for seed in SEEDS:

    top5 = predictions[
        (predictions["seed"] == seed)
        &
        (predictions["feature_count"] == 5)
    ]

    full = predictions[
        (predictions["seed"] == seed)
        &
        (predictions["feature_count"] == 24)
    ]

    auc_top5 = roc_auc_score(
        top5["target"],
        top5["probability"]
    )

    auc_full = roc_auc_score(
        full["target"],
        full["probability"]
    )

    diff = auc_top5 - auc_full

    spear_top5 = spearmanr(
        top5["probability"],
        top5["target"]
    ).statistic

    spear_full = spearmanr(
        full["probability"],
        full["target"]
    ).statistic

    overall_results.append(
        {
            "seed": seed,
            "top5_auc": auc_top5,
            "full24_auc": auc_full,
            "auc_difference": diff,
            "top5_spearman": spear_top5,
            "full24_spearman": spear_full,
        }
    )

    print(
        f"Seed={seed:<5} | "
        f"TOP5 AUC={auc_top5:.4f} | "
        f"24 AUC={auc_full:.4f} | "
        f"Difference={diff:+.4f}"
    )


overall_df = pd.DataFrame(
    overall_results
)


# ================================================================
# SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("6. SEED SUMMARY")
print("=" * 80)


print(
    f"TOP-5 mean AUC : "
    f"{overall_df['top5_auc'].mean():.4f}"
)

print(
    f"24-feature mean AUC : "
    f"{overall_df['full24_auc'].mean():.4f}"
)

print(
    f"Mean AUC difference : "
    f"{overall_df['auc_difference'].mean():+.4f}"
)

print(
    f"TOP-5 AUC std : "
    f"{overall_df['top5_auc'].std(ddof=1):.4f}"
)

print(
    f"24-feature AUC std : "
    f"{overall_df['full24_auc'].std(ddof=1):.4f}"
)


# ================================================================
# BOOTSTRAP FUNCTION
# ================================================================

def bootstrap_auc_difference(
    y,
    pred_a,
    pred_b,
    iterations=2000,
    seed=42
):

    rng = np.random.default_rng(seed)

    n = len(y)

    differences = []

    for _ in range(iterations):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        y_sample = y[indices]

        # AUC requires both classes
        if len(np.unique(y_sample)) < 2:
            continue

        auc_a = roc_auc_score(
            y_sample,
            pred_a[indices]
        )

        auc_b = roc_auc_score(
            y_sample,
            pred_b[indices]
        )

        differences.append(
            auc_a - auc_b
        )

    differences = np.array(
        differences
    )

    lower = np.percentile(
        differences,
        2.5
    )

    upper = np.percentile(
        differences,
        97.5
    )

    return (
        differences.mean(),
        lower,
        upper,
        differences
    )


# ================================================================
# BOOTSTRAP TEST
# ================================================================

print("\n" + "=" * 80)
print("7. BOOTSTRAP CONFIDENCE INTERVAL")
print("=" * 80)


bootstrap_results = []


for seed in SEEDS:

    top5 = predictions[
        (predictions["seed"] == seed)
        &
        (predictions["feature_count"] == 5)
    ].sort_values("Date")

    full = predictions[
        (predictions["seed"] == seed)
        &
        (predictions["feature_count"] == 24)
    ].sort_values("Date")

    common = top5.merge(
        full,
        on=["Date", "target"],
        suffixes=("_top5", "_full")
    )

    y = common["target"].values

    pred_top5 = common[
        "probability_top5"
    ].values

    pred_full = common[
        "probability_full"
    ].values

    mean_diff, lower, upper, distribution = (
        bootstrap_auc_difference(
            y,
            pred_top5,
            pred_full,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=seed
        )
    )

    bootstrap_results.append(
        {
            "seed": seed,
            "bootstrap_mean_difference": mean_diff,
            "ci_lower_95": lower,
            "ci_upper_95": upper,
        }
    )

    print(
        f"Seed={seed:<5} | "
        f"Mean diff={mean_diff:+.4f} | "
        f"95% CI=[{lower:+.4f}, {upper:+.4f}]"
    )


bootstrap_df = pd.DataFrame(
    bootstrap_results
)


# ================================================================
# PERMUTATION TEST
# ================================================================

print("\n" + "=" * 80)
print("8. PERMUTATION TEST")
print("=" * 80)


def permutation_test(
    y,
    pred_a,
    pred_b,
    iterations=2000,
    seed=42
):

    rng = np.random.default_rng(seed)

    observed = (
        roc_auc_score(y, pred_a)
        -
        roc_auc_score(y, pred_b)
    )

    differences = []

    for _ in range(iterations):

        swap = rng.random(
            len(y)
        ) < 0.5

        shuffled_a = np.where(
            swap,
            pred_b,
            pred_a
        )

        shuffled_b = np.where(
            swap,
            pred_a,
            pred_b
        )

        difference = (
            roc_auc_score(
                y,
                shuffled_a
            )
            -
            roc_auc_score(
                y,
                shuffled_b
            )
        )

        differences.append(
            difference
        )

    differences = np.array(
        differences
    )

    p_value = (
        np.sum(
            np.abs(differences)
            >= abs(observed)
        )
        + 1
    ) / (
        len(differences)
        + 1
    )

    return (
        observed,
        p_value
    )


permutation_results = []


for seed in SEEDS:

    top5 = predictions[
        (predictions["seed"] == seed)
        &
        (predictions["feature_count"] == 5)
    ].sort_values("Date")

    full = predictions[
        (predictions["seed"] == seed)
        &
        (predictions["feature_count"] == 24)
    ].sort_values("Date")

    common = top5.merge(
        full,
        on=["Date", "target"],
        suffixes=("_top5", "_full")
    )

    y = common["target"].values

    pred_top5 = common[
        "probability_top5"
    ].values

    pred_full = common[
        "probability_full"
    ].values

    observed, p_value = permutation_test(
        y,
        pred_top5,
        pred_full,
        iterations=PERMUTATION_ITERATIONS,
        seed=seed
    )

    permutation_results.append(
        {
            "seed": seed,
            "observed_auc_difference": observed,
            "permutation_p_value": p_value,
        }
    )

    print(
        f"Seed={seed:<5} | "
        f"AUC difference={observed:+.4f} | "
        f"p-value={p_value:.4f}"
    )


permutation_df = pd.DataFrame(
    permutation_results
)


# ================================================================
# YEARLY COMPARISON
# ================================================================

print("\n" + "=" * 80)
print("9. YEARLY COMPARISON")
print("=" * 80)


yearly_results = []


for seed in SEEDS:

    top5 = predictions[
        (predictions["seed"] == seed)
        &
        (predictions["feature_count"] == 5)
    ]

    full = predictions[
        (predictions["seed"] == seed)
        &
        (predictions["feature_count"] == 24)
    ]

    common = top5.merge(
        full,
        on=["Date", "target", "year"],
        suffixes=("_top5", "_full")
    )

    for year in sorted(
        common["year"].unique()
    ):

        subset = common[
            common["year"] == year
        ]

        if len(subset) < 30:
            continue

        if subset["target"].nunique() < 2:
            continue

        auc_top5 = roc_auc_score(
            subset["target"],
            subset["probability_top5"]
        )

        auc_full = roc_auc_score(
            subset["target"],
            subset["probability_full"]
        )

        yearly_results.append(
            {
                "seed": seed,
                "year": year,
                "rows": len(subset),
                "top5_auc": auc_top5,
                "full24_auc": auc_full,
                "auc_difference": auc_top5 - auc_full,
            }
        )


yearly_df = pd.DataFrame(
    yearly_results
)


print(
    yearly_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ================================================================
# YEARLY AGGREGATE
# ================================================================

print("\n" + "=" * 80)
print("10. YEARLY AGGREGATE")
print("=" * 80)


yearly_aggregate = (
    yearly_df
    .groupby("year")
    .agg(
        rows=("rows", "mean"),
        mean_top5_auc=("top5_auc", "mean"),
        mean_full24_auc=("full24_auc", "mean"),
        mean_auc_difference=("auc_difference", "mean"),
    )
    .reset_index()
)


print(
    yearly_aggregate.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ================================================================
# COUNT YEARS WHERE TOP5 > 24
# ================================================================

yearly_wins = (
    yearly_aggregate[
        "mean_auc_difference"
    ] > 0
).sum()

total_years = len(
    yearly_aggregate
)


print(
    f"\nYears where TOP-5 mean AUC > 24-feature mean AUC: "
    f"{yearly_wins}/{total_years}"
)


# ================================================================
# BOOTSTRAP OVER SEEDS
# ================================================================

print("\n" + "=" * 80)
print("11. CROSS-SEED AUC DIFFERENCE")
print("=" * 80)


seed_differences = (
    overall_df[
        "auc_difference"
    ].values
)


print(
    "Seed differences:"
)

for seed, diff in zip(
    SEEDS,
    seed_differences
):

    print(
        f"Seed={seed:<5} "
        f"{diff:+.4f}"
    )


print(
    f"\nMean difference: "
    f"{np.mean(seed_differences):+.4f}"
)

print(
    f"Median difference: "
    f"{np.median(seed_differences):+.4f}"
)

print(
    f"Positive seed differences: "
    f"{np.sum(seed_differences > 0)}/{len(seed_differences)}"
)


# ================================================================
# SAVE RESULTS
# ================================================================

print("\n" + "=" * 80)
print("12. SAVING RESULTS")
print("=" * 80)


overall_path = os.path.join(
    RESULT_DIR,
    "overall_auc_comparison.csv"
)

bootstrap_path = os.path.join(
    RESULT_DIR,
    "bootstrap_auc_difference.csv"
)

permutation_path = os.path.join(
    RESULT_DIR,
    "permutation_test.csv"
)

yearly_path = os.path.join(
    RESULT_DIR,
    "yearly_auc_comparison.csv"
)

yearly_aggregate_path = os.path.join(
    RESULT_DIR,
    "yearly_auc_aggregate.csv"
)

predictions_path = os.path.join(
    RESULT_DIR,
    "walkforward_predictions.csv"
)


overall_df.to_csv(
    overall_path,
    index=False
)

bootstrap_df.to_csv(
    bootstrap_path,
    index=False
)

permutation_df.to_csv(
    permutation_path,
    index=False
)

yearly_df.to_csv(
    yearly_path,
    index=False
)

yearly_aggregate.to_csv(
    yearly_aggregate_path,
    index=False
)

predictions.to_csv(
    predictions_path,
    index=False
)


print(
    f"Overall comparison:\n{overall_path}"
)

print(
    f"Bootstrap results:\n{bootstrap_path}"
)

print(
    f"Permutation results:\n{permutation_path}"
)

print(
    f"Yearly comparison:\n{yearly_path}"
)

print(
    f"Yearly aggregate:\n{yearly_aggregate_path}"
)

print(
    f"Walk-forward predictions:\n{predictions_path}"
)


# ================================================================
# FINAL DIAGNOSTIC SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("13. DIAGNOSTIC SUMMARY")
print("=" * 80)


mean_auc_diff = (
    overall_df[
        "auc_difference"
    ].mean()
)

mean_p_value = (
    permutation_df[
        "permutation_p_value"
    ].mean()
)

ci_lower_mean = (
    bootstrap_df[
        "ci_lower_95"
    ].mean()
)

ci_upper_mean = (
    bootstrap_df[
        "ci_upper_95"
    ].mean()
)


print(
    f"""
TOP-5 mean ROC-AUC:
{overall_df['top5_auc'].mean():.4f}

24-feature mean ROC-AUC:
{overall_df['full24_auc'].mean():.4f}

Mean AUC difference:
{mean_auc_diff:+.4f}

Average bootstrap 95% CI:
[{ci_lower_mean:+.4f}, {ci_upper_mean:+.4f}]

Average permutation p-value:
{mean_p_value:.4f}

Positive seed differences:
{np.sum(seed_differences > 0)}/{len(seed_differences)}

Years with TOP-5 higher mean AUC:
{yearly_wins}/{total_years}
"""
)


print(
    """
INTERPRETATION:

A positive AUC difference means TOP-5 produced
higher AUC than the 24-feature model.

However:

- A small difference alone is not enough evidence.
- A bootstrap CI containing zero means the difference
  is compatible with no meaningful improvement.
- A large permutation p-value means the observed
  difference is not statistically unusual under the
  null comparison.
- Yearly consistency matters.
- Seed stability matters.
- Statistical significance does not automatically
  establish production usefulness.

This test does NOT:
- change the production threshold
- select a production threshold
- modify the existing production model
- use the final test to select features

NORTHGATE AI — STATISTICAL SIGNIFICANCE TEST COMPLETED
"""
)

print("=" * 80)