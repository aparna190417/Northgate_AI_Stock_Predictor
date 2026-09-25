import os
import random
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ============================================================
# NORTHGATE AI — FAST FEATURE SET STATISTICAL SIGNIFICANCE TEST
# ============================================================

print("=" * 80)
print("NORTHGATE AI — FEATURE SET STATISTICAL SIGNIFICANCE TEST")
print("=" * 80)

print("""
Purpose:

1. Reuse EXISTING 24-feature walk-forward predictions.
2. Generate TOP-5 walk-forward predictions.
3. Compare TOP-5 vs 24-feature model.
4. Bootstrap the AUC difference.
5. Perform paired permutation test.
6. Compare year-by-year performance.

This is a diagnostic statistical test only.

IMPORTANT:
- No production threshold is changed.
- No production threshold is selected.
- Existing 24-feature predictions are reused.
- TOP-5 is evaluated chronologically.
- n_jobs=1 is used for stability on Windows.
""")

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_FILE = os.path.join(
    BASE_DIR, "data", "processed", "AAPL_clean_train.parquet"
)

VAL_FILE = os.path.join(
    BASE_DIR, "data", "processed", "AAPL_clean_validation.parquet"
)

TEST_FILE = os.path.join(
    BASE_DIR, "data", "processed", "AAPL_clean_test.parquet"
)

EXISTING_PRED_FILE = os.path.join(
    BASE_DIR,
    "results",
    "walk_forward_final",
    "walk_forward_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "feature_set_statistical_significance_fast"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TOP5_FEATURES = [
    "AAPL_price_to_ma20",
    "AAPL_volatility_20d",
    "AAPL_volatility_5d",
    "AAPL_return_5d",
    "AAPL_return_20d",
]

ALL_24_FEATURES = [
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

# Walk-forward settings
MIN_TRAIN_SIZE = 1000
STEP = 1

# Random forest
N_ESTIMATORS = 150
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 2

# Bootstrap
BOOTSTRAP_ITERATIONS = 2000

# Permutation
PERMUTATION_ITERATIONS = 2000

# ============================================================
# HELPER
# ============================================================

def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found:\n{path}")

    df = pd.read_parquet(path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("1. LOADING DATA")
print("=" * 80)

train = load_dataset(TRAIN_FILE)
validation = load_dataset(VAL_FILE)
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

# ============================================================
# 2. VALIDATE COLUMNS
# ============================================================

print("\n" + "=" * 80)
print("2. COLUMN VALIDATION")
print("=" * 80)

required = TOP5_FEATURES + ["target"]

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} missing columns:\n" +
            "\n".join(missing)
        )

    print(f"{name}: all required TOP-5 columns found.")

# ============================================================
# 3. LOAD EXISTING 24-FEATURE PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("3. LOADING EXISTING 24-FEATURE PREDICTIONS")
print("=" * 80)

if not os.path.exists(EXISTING_PRED_FILE):
    raise FileNotFoundError(
        f"Existing prediction file not found:\n{EXISTING_PRED_FILE}"
    )

existing = pd.read_csv(EXISTING_PRED_FILE)

existing["date"] = pd.to_datetime(existing["date"])

existing = existing.sort_values("date").reset_index(drop=True)

print(f"Existing prediction rows: {len(existing):,}")
print(
    f"Period: {existing['date'].min().date()} → "
    f"{existing['date'].max().date()}"
)

# Existing model probability
existing["prob_24"] = existing["prediction_probability"]

# ============================================================
# 4. COMBINE DATA
# ============================================================

print("\n" + "=" * 80)
print("4. PREPARING WALK-FORWARD DATA")
print("=" * 80)

combined = pd.concat(
    [train, validation, test],
    axis=0
)

combined = combined[
    ~combined.index.duplicated(keep="first")
].sort_index()

print(f"Combined rows: {len(combined):,}")
print(
    f"Period: {combined.index.min().date()} → "
    f"{combined.index.max().date()}"
)

# ============================================================
# 5. GENERATE TOP-5 WALK-FORWARD PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("5. GENERATING TOP-5 WALK-FORWARD PREDICTIONS")
print("=" * 80)

print("\nTOP-5 FEATURES:")
for feature in TOP5_FEATURES:
    print(" -", feature)

dates = existing["date"].tolist()

top5_predictions = []

total = len(dates)

for i, current_date in enumerate(dates):

    if i % 100 == 0:
        print(
            f"Progress: {i}/{total} "
            f"({100 * i / total:.1f}%)"
        )

    current_date = pd.Timestamp(current_date)

    historical = combined[
        combined.index < current_date
    ]

    if len(historical) < MIN_TRAIN_SIZE:
        continue

    if current_date not in combined.index:
        continue

    row = combined.loc[[current_date]]

    if row["target"].isna().any():
        continue

    X_train = historical[TOP5_FEATURES].copy()
    y_train = historical["target"].copy()

    X_current = row[TOP5_FEATURES].copy()

    # Remove NaN
    valid_train = X_train.notna().all(axis=1)

    X_train = X_train.loc[valid_train]
    y_train = y_train.loc[valid_train]

    if X_current.isna().any(axis=None):
        continue

    if y_train.nunique() < 2:
        continue

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=SEED,
        n_jobs=1,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced_subsample",
    )

    model.fit(X_train, y_train)

    probability = model.predict_proba(X_current)[0, 1]

    top5_predictions.append(
        {
            "date": current_date,
            "target": int(row["target"].iloc[0]),
            "prob_5": probability,
        }
    )

top5 = pd.DataFrame(top5_predictions)

print(
    f"\nTOP-5 predictions generated: "
    f"{len(top5):,}"
)

# ============================================================
# 6. MERGE TOP-5 + EXISTING 24-FEATURE
# ============================================================

print("\n" + "=" * 80)
print("6. MERGING MODEL PREDICTIONS")
print("=" * 80)

merged = existing[
    ["date", "target", "year", "prob_24"]
].copy()

merged = merged.merge(
    top5,
    on=["date", "target"],
    how="inner"
)

merged = merged.sort_values("date").reset_index(drop=True)

print(f"Paired observations: {len(merged):,}")

if len(merged) < 500:
    raise ValueError(
        "Too few paired observations for statistical comparison."
    )

# ============================================================
# 7. AUC
# ============================================================

print("\n" + "=" * 80)
print("7. ROC-AUC COMPARISON")
print("=" * 80)

y = merged["target"].astype(int)

auc_5 = roc_auc_score(y, merged["prob_5"])
auc_24 = roc_auc_score(y, merged["prob_24"])

auc_difference = auc_5 - auc_24

print(f"TOP-5 ROC-AUC       : {auc_5:.6f}")
print(f"24-feature ROC-AUC  : {auc_24:.6f}")
print(f"AUC difference      : {auc_difference:+.6f}")

# ============================================================
# 8. SPEARMAN
# ============================================================

spearman_5 = spearmanr(
    merged["prob_5"],
    y
).statistic

spearman_24 = spearmanr(
    merged["prob_24"],
    y
).statistic

print("\nSpearman probability vs direction:")
print(f"TOP-5               : {spearman_5:+.6f}")
print(f"24-feature          : {spearman_24:+.6f}")
print(
    f"Difference           : "
    f"{spearman_5 - spearman_24:+.6f}"
)

# ============================================================
# 9. BOOTSTRAP AUC DIFFERENCE
# ============================================================

print("\n" + "=" * 80)
print("8. BOOTSTRAP CONFIDENCE INTERVAL")
print("=" * 80)

rng = np.random.default_rng(SEED)

n = len(merged)

bootstrap_differences = []

for _ in range(BOOTSTRAP_ITERATIONS):

    indices = rng.integers(
        0,
        n,
        size=n
    )

    y_boot = y.iloc[indices]

    # Need both classes
    if y_boot.nunique() < 2:
        continue

    auc5_boot = roc_auc_score(
        y_boot,
        merged["prob_5"].iloc[indices]
    )

    auc24_boot = roc_auc_score(
        y_boot,
        merged["prob_24"].iloc[indices]
    )

    bootstrap_differences.append(
        auc5_boot - auc24_boot
    )

bootstrap_differences = np.array(
    bootstrap_differences
)

ci_low = np.percentile(
    bootstrap_differences,
    2.5
)

ci_high = np.percentile(
    bootstrap_differences,
    97.5
)

print(
    f"Bootstrap iterations : "
    f"{len(bootstrap_differences):,}"
)

print(
    f"Observed AUC diff    : "
    f"{auc_difference:+.6f}"
)

print(
    f"95% bootstrap CI     : "
    f"[{ci_low:+.6f}, {ci_high:+.6f}]"
)

if ci_low > 0:
    print(
        "\nBootstrap CI is entirely above zero."
    )
elif ci_high < 0:
    print(
        "\nBootstrap CI is entirely below zero."
    )
else:
    print(
        "\nBootstrap CI includes zero."
    )

# ============================================================
# 10. PAIRED PERMUTATION TEST
# ============================================================

print("\n" + "=" * 80)
print("9. PAIRED PERMUTATION TEST")
print("=" * 80)

observed = auc_difference

permutation_differences = []

prob5 = merged["prob_5"].to_numpy()
prob24 = merged["prob_24"].to_numpy()
target_array = y.to_numpy()

for _ in range(PERMUTATION_ITERATIONS):

    swap = rng.integers(
        0,
        2,
        size=n
    ).astype(bool)

    perm5 = np.where(
        swap,
        prob24,
        prob5
    )

    perm24 = np.where(
        swap,
        prob5,
        prob24
    )

    auc5_perm = roc_auc_score(
        target_array,
        perm5
    )

    auc24_perm = roc_auc_score(
        target_array,
        perm24
    )

    permutation_differences.append(
        auc5_perm - auc24_perm
    )

permutation_differences = np.array(
    permutation_differences
)

p_value = (
    np.sum(
        np.abs(permutation_differences)
        >= abs(observed)
    ) + 1
) / (
    len(permutation_differences) + 1
)

print(
    f"Permutation iterations : "
    f"{len(permutation_differences):,}"
)

print(
    f"Observed AUC difference: "
    f"{observed:+.6f}"
)

print(
    f"Permutation p-value    : "
    f"{p_value:.6f}"
)

# ============================================================
# 11. YEARLY COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("10. YEAR-BY-YEAR COMPARISON")
print("=" * 80)

yearly_results = []

for year, group in merged.groupby(
    merged["date"].dt.year
):

    if group["target"].nunique() < 2:
        continue

    y_year = group["target"]

    auc5_year = roc_auc_score(
        y_year,
        group["prob_5"]
    )

    auc24_year = roc_auc_score(
        y_year,
        group["prob_24"]
    )

    yearly_results.append(
        {
            "year": year,
            "rows": len(group),
            "top5_auc": auc5_year,
            "model24_auc": auc24_year,
            "auc_difference": auc5_year - auc24_year,
            "top5_spearman": spearmanr(
                group["prob_5"],
                y_year
            ).statistic,
            "model24_spearman": spearmanr(
                group["prob_24"],
                y_year
            ).statistic,
            "actual_positive_rate":
                y_year.mean(),
        }
    )

yearly = pd.DataFrame(yearly_results)

print(
    yearly.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# ============================================================
# 12. POSITIVE YEAR COUNT
# ============================================================

positive_years = (
    yearly["auc_difference"] > 0
).sum()

total_years = len(yearly)

print("\nTOP-5 higher AUC years:")
print(f"{positive_years}/{total_years}")

# ============================================================
# 13. SAVE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("11. SAVING RESULTS")
print("=" * 80)

paired_file = os.path.join(
    OUTPUT_DIR,
    "paired_predictions.csv"
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "statistical_significance_summary.csv"
)

yearly_file = os.path.join(
    OUTPUT_DIR,
    "yearly_comparison.csv"
)

bootstrap_file = os.path.join(
    OUTPUT_DIR,
    "bootstrap_auc_differences.csv"
)

permutation_file = os.path.join(
    OUTPUT_DIR,
    "permutation_auc_differences.csv"
)

merged.to_csv(
    paired_file,
    index=False
)

summary = pd.DataFrame(
    [
        {
            "top5_auc": auc_5,
            "model24_auc": auc_24,
            "auc_difference": auc_difference,
            "top5_spearman": spearman_5,
            "model24_spearman": spearman_24,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "permutation_p_value": p_value,
            "paired_rows": len(merged),
            "positive_auc_difference_years": positive_years,
            "total_years": total_years,
        }
    ]
)

summary.to_csv(
    summary_file,
    index=False
)

yearly.to_csv(
    yearly_file,
    index=False
)

pd.DataFrame(
    {
        "bootstrap_auc_difference":
            bootstrap_differences
    }
).to_csv(
    bootstrap_file,
    index=False
)

pd.DataFrame(
    {
        "permutation_auc_difference":
            permutation_differences
    }
).to_csv(
    permutation_file,
    index=False
)

# ============================================================
# 14. FINAL DIAGNOSTIC SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("12. DIAGNOSTIC SUMMARY")
print("=" * 80)

print(f"""
Paired observations       : {len(merged):,}

TOP-5 ROC-AUC              : {auc_5:.6f}
24-feature ROC-AUC         : {auc_24:.6f}

AUC difference             : {auc_difference:+.6f}

95% Bootstrap CI           :
[{ci_low:+.6f}, {ci_high:+.6f}]

Permutation p-value        : {p_value:.6f}

Years TOP-5 > 24-feature  :
{positive_years}/{total_years}

IMPORTANT:
This test does NOT automatically establish
that TOP-5 is better for production.

AUC differences can be small and uncertain.
The confidence interval and permutation result
must be interpreted together with yearly stability.

No production threshold has been changed.
No production threshold has been selected.
""")

print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(paired_file)
print(summary_file)
print(yearly_file)
print(bootstrap_file)
print(permutation_file)

print("\n" + "=" * 80)
print("NORTHGATE AI — STATISTICAL SIGNIFICANCE TEST COMPLETED")
print("=" * 80)