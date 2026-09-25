import os
import numpy as np
import pandas as pd

# ============================================================
# NORTHGATE AI — TOP-5 VS 24-FEATURE TRADING STATISTICAL TEST
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "top5_vs_24_trading_comparison"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "top5_trading_statistical_test"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRADES_24 = os.path.join(INPUT_DIR, "trades_24_feature.csv")
TRADES_TOP5 = os.path.join(INPUT_DIR, "trades_top5.csv")


print("=" * 80)
print("NORTHGATE AI — TOP-5 VS 24-FEATURE TRADING STATISTICAL TEST")
print("=" * 80)

print("""
Purpose:

1. Compare TOP-5 vs 24-feature trade returns.
2. Calculate mean/median differences.
3. Bootstrap the difference in average trade return.
4. Perform a paired permutation test where possible.
5. Compare yearly trading performance.
6. Check whether TOP-5 improvement is statistically robust.

IMPORTANT:
- No threshold is changed.
- No production model is changed.
- Locked threshold remains 0.65.
- This is diagnostic only.
""")

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("1. LOADING TRADE DATA")
print("=" * 80)

df24 = pd.read_csv(TRADES_24)
df5 = pd.read_csv(TRADES_TOP5)

print(f"24-feature trades : {len(df24)}")
print(f"TOP-5 trades      : {len(df5)}")

# Normalize dates
for df in [df24, df5]:
    if "entry_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["entry_date"])

    if "exit_date" in df.columns:
        df["exit_date"] = pd.to_datetime(df["exit_date"])

# ============================================================
# 2. BASIC STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("2. BASIC TRADE RETURN STATISTICS")
print("=" * 80)

r24 = df24["net_return"].dropna().to_numpy()
r5 = df5["net_return"].dropna().to_numpy()

stats = pd.DataFrame({
    "metric": [
        "Number of trades",
        "Mean net return",
        "Median net return",
        "Std net return",
        "Win rate",
        "Best trade",
        "Worst trade"
    ],
    "24-feature": [
        len(r24),
        np.mean(r24),
        np.median(r24),
        np.std(r24, ddof=1),
        np.mean(r24 > 0),
        np.max(r24),
        np.min(r24)
    ],
    "TOP-5": [
        len(r5),
        np.mean(r5),
        np.median(r5),
        np.std(r5, ddof=1),
        np.mean(r5 > 0),
        np.max(r5),
        np.min(r5)
    ]
})

stats["TOP5_minus_24"] = (
    stats["TOP-5"] - stats["24-feature"]
)

print(stats.to_string(index=False))

# ============================================================
# 3. ALIGN COMMON ENTRY DATES
# ============================================================

print("\n" + "=" * 80)
print("3. ALIGNING COMMON TRADES")
print("=" * 80)

common = pd.merge(
    df24[
        ["entry_date", "exit_date", "net_return"]
    ].rename(columns={
        "net_return": "return_24"
    }),
    df5[
        ["entry_date", "exit_date", "net_return"]
    ].rename(columns={
        "net_return": "return_top5"
    }),
    on=["entry_date", "exit_date"],
    how="inner"
)

print(f"Common entry/exit pairs: {len(common)}")

if len(common) > 0:

    common["return_difference"] = (
        common["return_top5"] -
        common["return_24"]
    )

    print(
        f"Mean paired difference: "
        f"{common['return_difference'].mean():.6f}"
    )

    print(
        f"Median paired difference: "
        f"{common['return_difference'].median():.6f}"
    )

else:
    print("No common trades found.")

# ============================================================
# 4. BOOTSTRAP DIFFERENCE IN MEAN TRADE RETURN
# ============================================================

print("\n" + "=" * 80)
print("4. BOOTSTRAP TEST")
print("=" * 80)

BOOTSTRAP_ITERATIONS = 5000
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

observed_difference = (
    np.mean(r5) -
    np.mean(r24)
)

bootstrap_differences = np.empty(
    BOOTSTRAP_ITERATIONS
)

for i in range(BOOTSTRAP_ITERATIONS):

    sample24 = rng.choice(
        r24,
        size=len(r24),
        replace=True
    )

    sample5 = rng.choice(
        r5,
        size=len(r5),
        replace=True
    )

    bootstrap_differences[i] = (
        np.mean(sample5) -
        np.mean(sample24)
    )

ci_low = np.percentile(
    bootstrap_differences,
    2.5
)

ci_high = np.percentile(
    bootstrap_differences,
    97.5
)

print(f"Bootstrap iterations : {BOOTSTRAP_ITERATIONS}")
print(f"Observed difference  : {observed_difference:.6f}")
print(
    f"95% bootstrap CI     : "
    f"[{ci_low:.6f}, {ci_high:.6f}]"
)

if ci_low <= 0 <= ci_high:
    print("CI includes zero.")
else:
    print("CI excludes zero.")

# ============================================================
# 5. PAIRED PERMUTATION TEST
# ============================================================

print("\n" + "=" * 80)
print("5. PAIRED PERMUTATION TEST")
print("=" * 80)

PERMUTATIONS = 5000

if len(common) >= 20:

    differences = common[
        "return_difference"
    ].to_numpy()

    observed_paired = np.mean(differences)

    permutation_means = np.empty(
        PERMUTATIONS
    )

    for i in range(PERMUTATIONS):

        signs = rng.choice(
            [-1, 1],
            size=len(differences)
        )

        permutation_means[i] = np.mean(
            differences * signs
        )

    p_value = (
        np.sum(
            np.abs(permutation_means)
            >= abs(observed_paired)
        ) + 1
    ) / (PERMUTATIONS + 1)

    print(
        f"Common paired trades : {len(differences)}"
    )

    print(
        f"Observed mean diff   : "
        f"{observed_paired:.6f}"
    )

    print(
        f"Permutation p-value  : "
        f"{p_value:.6f}"
    )

else:

    observed_paired = np.nan
    p_value = np.nan

    print(
        "Not enough common trades "
        "for paired permutation test."
    )

# ============================================================
# 6. YEARLY COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("6. YEAR-BY-YEAR COMPARISON")
print("=" * 80)

df24["year"] = df24["entry_date"].dt.year
df5["year"] = df5["entry_date"].dt.year

years = sorted(
    set(df24["year"].dropna().astype(int))
    |
    set(df5["year"].dropna().astype(int))
)

yearly_rows = []

for year in years:

    y24 = df24[
        df24["year"] == year
    ]["net_return"].dropna()

    y5 = df5[
        df5["year"] == year
    ]["net_return"].dropna()

    mean24 = y24.mean() if len(y24) else np.nan
    mean5 = y5.mean() if len(y5) else np.nan

    median24 = y24.median() if len(y24) else np.nan
    median5 = y5.median() if len(y5) else np.nan

    win24 = (
        np.mean(y24 > 0)
        if len(y24)
        else np.nan
    )

    win5 = (
        np.mean(y5 > 0)
        if len(y5)
        else np.nan
    )

    yearly_rows.append({
        "year": year,
        "trades_24": len(y24),
        "trades_top5": len(y5),
        "mean_return_24": mean24,
        "mean_return_top5": mean5,
        "mean_difference": (
            mean5 - mean24
            if not np.isnan(mean24)
            and not np.isnan(mean5)
            else np.nan
        ),
        "median_return_24": median24,
        "median_return_top5": median5,
        "win_rate_24": win24,
        "win_rate_top5": win5
    })

yearly = pd.DataFrame(yearly_rows)

print(
    yearly.to_string(index=False)
)

# ============================================================
# 7. YEARLY CONSISTENCY
# ============================================================

valid_years = yearly.dropna(
    subset=["mean_difference"]
)

top5_better_years = (
    valid_years["mean_difference"] > 0
).sum()

total_years = len(valid_years)

print("\nTOP-5 higher mean trade return:")
print(
    f"{top5_better_years}/{total_years} years"
)

# ============================================================
# 8. SAVE BOOTSTRAP RESULTS
# ============================================================

bootstrap_df = pd.DataFrame({
    "bootstrap_mean_difference":
        bootstrap_differences
})

bootstrap_file = os.path.join(
    OUTPUT_DIR,
    "bootstrap_trade_return_differences.csv"
)

bootstrap_df.to_csv(
    bootstrap_file,
    index=False
)

# ============================================================
# 9. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame({
    "metric": [
        "24_feature_trade_count",
        "top5_trade_count",
        "24_feature_mean_return",
        "top5_mean_return",
        "observed_mean_difference",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "paired_common_trades",
        "paired_mean_difference",
        "paired_permutation_p_value",
        "top5_better_years",
        "total_years"
    ],
    "value": [
        len(r24),
        len(r5),
        np.mean(r24),
        np.mean(r5),
        observed_difference,
        ci_low,
        ci_high,
        len(common),
        observed_paired,
        p_value,
        top5_better_years,
        total_years
    ]
})

summary_file = os.path.join(
    OUTPUT_DIR,
    "trading_statistical_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

# ============================================================
# 10. SAVE YEARLY
# ============================================================

yearly_file = os.path.join(
    OUTPUT_DIR,
    "yearly_trading_statistical_comparison.csv"
)

yearly.to_csv(
    yearly_file,
    index=False
)

# ============================================================
# 11. SAVE PAIRED DATA
# ============================================================

paired_file = os.path.join(
    OUTPUT_DIR,
    "paired_trade_returns.csv"
)

common.to_csv(
    paired_file,
    index=False
)

# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("7. FINAL DIAGNOSTIC SUMMARY")
print("=" * 80)

print(f"""
24-feature mean trade return :
{np.mean(r24):.6f}

TOP-5 mean trade return :
{np.mean(r5):.6f}

Difference :
{observed_difference:.6f}

95% bootstrap CI :
[{ci_low:.6f}, {ci_high:.6f}]

Paired permutation p-value :
{p_value if not np.isnan(p_value) else "N/A"}

TOP-5 higher mean return years :
{top5_better_years}/{total_years}

Common paired trades :
{len(common)}
""")

print("""
INTERPRETATION:

This test does NOT select a production model.

The trading comparison should be interpreted together
with:

- ROC-AUC statistical test
- bootstrap CI
- permutation p-value
- yearly stability
- drawdown
- Sharpe
- trade count
- transaction costs

No threshold was optimized.
No production model was changed.
""")

print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(bootstrap_file)
print(summary_file)
print(yearly_file)
print(paired_file)

print("\n" + "=" * 80)
print("NORTHGATE AI — TRADING STATISTICAL TEST COMPLETED")
print("=" * 80)