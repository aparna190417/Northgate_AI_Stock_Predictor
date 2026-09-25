from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# NORTHGATE AI
# PREDICTION RANKING / DISCRIMINATION AUDIT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    ROOT
    / "results"
    / "walk_forward_final"
    / "walk_forward_predictions.csv"
)

RESULTS_DIR = (
    ROOT
    / "results"
    / "prediction_ranking_audit"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INITIAL_CAPITAL = 100000
TRANSACTION_COST = 0.001
HOLDING_DAYS = 5

THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("NORTHGATE AI — PREDICTION RANKING / DISCRIMINATION AUDIT")
print("=" * 80)

print("""
Purpose:
This audit examines whether higher model probabilities
actually correspond to better future outcomes.

It does NOT:
- retrain the model
- change the locked threshold
- select a production threshold
- modify existing results
- use the untouched final test

This is a diagnostic analysis only.
""")


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("1. LOADING WALK-FORWARD PREDICTIONS")
print("=" * 80)

df = pd.read_csv(
    PREDICTIONS_PATH
)

df["date"] = pd.to_datetime(
    df["date"]
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

print(f"Rows loaded: {len(df):,}")
print(
    f"Period: "
    f"{df['date'].min().date()} "
    f"→ "
    f"{df['date'].max().date()}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "date",
    "prediction_probability",
    "aapl_adj_close"
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# ACTUAL FUTURE 5-DAY RETURN
# ============================================================

print("\n" + "=" * 80)
print("2. CALCULATING FUTURE 5-DAY OUTCOME")
print("=" * 80)

df["future_price"] = (
    df["aapl_adj_close"]
    .shift(-HOLDING_DAYS)
)

df["future_return"] = (
    df["future_price"]
    /
    df["aapl_adj_close"]
) - 1

df["future_positive"] = (
    df["future_return"] > 0
).astype(int)

df = df.dropna(
    subset=["future_return"]
).reset_index(drop=True)

print(
    f"Rows with valid future return: {len(df):,}"
)


# ============================================================
# OVERALL RANKING CORRELATION
# ============================================================

print("\n" + "=" * 80)
print("3. PROBABILITY RANKING QUALITY")
print("=" * 80)

spearman_return = (
    df["prediction_probability"]
    .corr(
        df["future_return"],
        method="spearman"
    )
)

spearman_direction = (
    df["prediction_probability"]
    .corr(
        df["future_positive"],
        method="spearman"
    )
)

pearson_return = (
    df["prediction_probability"]
    .corr(
        df["future_return"],
        method="pearson"
    )
)

print(
    f"Spearman probability vs future return: "
    f"{spearman_return:.4f}"
)

print(
    f"Spearman probability vs direction: "
    f"{spearman_direction:.4f}"
)

print(
    f"Pearson probability vs future return: "
    f"{pearson_return:.4f}"
)


# ============================================================
# DECILE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("4. PROBABILITY DECILE ANALYSIS")
print("=" * 80)

df["probability_decile"] = pd.qcut(
    df["prediction_probability"],
    q=10,
    labels=False,
    duplicates="drop"
)

decile_results = []

for decile, group in df.groupby(
    "probability_decile"
):

    decile_results.append(
        {
            "decile": int(decile) + 1,
            "rows": len(group),
            "mean_probability":
                group[
                    "prediction_probability"
                ].mean(),
            "median_probability":
                group[
                    "prediction_probability"
                ].median(),
            "actual_positive_rate":
                group[
                    "future_positive"
                ].mean(),
            "mean_future_return":
                group[
                    "future_return"
                ].mean(),
            "median_future_return":
                group[
                    "future_return"
                ].median()
        }
    )


decile_df = pd.DataFrame(
    decile_results
)

print(
    decile_df.to_string(
        index=False
    )
)


# ============================================================
# TOP VS BOTTOM GROUP
# ============================================================

print("\n" + "=" * 80)
print("5. TOP VS BOTTOM PROBABILITY GROUPS")
print("=" * 80)

sorted_df = (
    df
    .sort_values(
        "prediction_probability"
    )
    .reset_index(drop=True)
)

n = len(sorted_df)

group_size = max(
    1,
    n // 5
)

bottom_group = (
    sorted_df.iloc[:group_size]
)

top_group = (
    sorted_df.iloc[-group_size:]
)

top_positive_rate = (
    top_group[
        "future_positive"
    ].mean()
)

bottom_positive_rate = (
    bottom_group[
        "future_positive"
    ].mean()
)

top_return = (
    top_group[
        "future_return"
    ].mean()
)

bottom_return = (
    bottom_group[
        "future_return"
    ].mean()
)

print(
    f"Bottom 20% mean probability : "
    f"{bottom_group['prediction_probability'].mean():.4f}"
)

print(
    f"Bottom 20% positive rate    : "
    f"{bottom_positive_rate:.2%}"
)

print(
    f"Bottom 20% mean 5D return   : "
    f"{bottom_return:.4%}"
)

print()

print(
    f"Top 20% mean probability    : "
    f"{top_group['prediction_probability'].mean():.4f}"
)

print(
    f"Top 20% positive rate       : "
    f"{top_positive_rate:.2%}"
)

print(
    f"Top 20% mean 5D return      : "
    f"{top_return:.4%}"
)

print()

print(
    f"Positive-rate difference    : "
    f"{top_positive_rate - bottom_positive_rate:.2%}"
)

print(
    f"Return difference           : "
    f"{top_return - bottom_return:.4%}"
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("6. THRESHOLD OUTCOME ANALYSIS")
print("=" * 80)

threshold_results = []

for threshold in THRESHOLDS:

    subset = df[
        df["prediction_probability"]
        >= threshold
    ]

    if len(subset) == 0:
        continue

    threshold_results.append(
        {
            "threshold": threshold,
            "rows": len(subset),
            "signal_rate":
                len(subset) / len(df),
            "mean_probability":
                subset[
                    "prediction_probability"
                ].mean(),
            "actual_positive_rate":
                subset[
                    "future_positive"
                ].mean(),
            "mean_future_return":
                subset[
                    "future_return"
                ].mean(),
            "median_future_return":
                subset[
                    "future_return"
                ].median()
        }
    )

threshold_df = pd.DataFrame(
    threshold_results
)

print(
    threshold_df.to_string(
        index=False
    )
)


# ============================================================
# YEARLY RANKING QUALITY
# ============================================================

print("\n" + "=" * 80)
print("7. YEARLY RANKING QUALITY")
print("=" * 80)

df["year"] = (
    df["date"]
    .dt.year
)

yearly_results = []

for year, group in df.groupby(
    "year"
):

    if len(group) < 10:
        continue

    corr = group[
        "prediction_probability"
    ].corr(
        group[
            "future_return"
        ],
        method="spearman"
    )

    direction_corr = group[
        "prediction_probability"
    ].corr(
        group[
            "future_positive"
        ],
        method="spearman"
    )

    yearly_results.append(
        {
            "year": year,
            "rows": len(group),
            "spearman_return":
                corr,
            "spearman_direction":
                direction_corr,
            "mean_probability":
                group[
                    "prediction_probability"
                ].mean(),
            "actual_positive_rate":
                group[
                    "future_positive"
                ].mean(),
            "mean_future_return":
                group[
                    "future_return"
                ].mean()
        }
    )

yearly_df = pd.DataFrame(
    yearly_results
)

print(
    yearly_df.to_string(
        index=False
    )
)


# ============================================================
# RANKING MONOTONICITY
# ============================================================

print("\n" + "=" * 80)
print("8. MONOTONICITY CHECK")
print("=" * 80)

positive_rates = (
    decile_df[
        "actual_positive_rate"
    ].values
)

future_returns = (
    decile_df[
        "mean_future_return"
    ].values
)

positive_rate_increases = (
    np.sum(
        np.diff(
            positive_rates
        ) > 0
    )
)

return_increases = (
    np.sum(
        np.diff(
            future_returns
        ) > 0
    )
)

possible_steps = (
    len(positive_rates) - 1
)

print(
    f"Positive-rate upward steps: "
    f"{positive_rate_increases}/"
    f"{possible_steps}"
)

print(
    f"Return upward steps: "
    f"{return_increases}/"
    f"{possible_steps}"
)


# ============================================================
# HIGH PROBABILITY QUALITY
# ============================================================

print("\n" + "=" * 80)
print("9. HIGH-PROBABILITY GROUPS")
print("=" * 80)

high_probability_ranges = [
    (0.60, 0.65),
    (0.65, 0.70),
    (0.70, 0.75),
    (0.75, 1.00)
]

high_results = []

for low, high in high_probability_ranges:

    subset = df[
        (
            df["prediction_probability"]
            >= low
        )
        &
        (
            df["prediction_probability"]
            < high
        )
    ]

    if len(subset) == 0:
        continue

    high_results.append(
        {
            "range":
                f"{low:.2f}-{high:.2f}",
            "rows":
                len(subset),
            "mean_probability":
                subset[
                    "prediction_probability"
                ].mean(),
            "actual_positive_rate":
                subset[
                    "future_positive"
                ].mean(),
            "mean_future_return":
                subset[
                    "future_return"
                ].mean(),
            "median_future_return":
                subset[
                    "future_return"
                ].median()
        }
    )

high_df = pd.DataFrame(
    high_results
)

print(
    high_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("10. SAVING RESULTS")
print("=" * 80)

decile_path = (
    RESULTS_DIR
    / "probability_deciles.csv"
)

threshold_path = (
    RESULTS_DIR
    / "threshold_ranking_quality.csv"
)

yearly_path = (
    RESULTS_DIR
    / "yearly_ranking_quality.csv"
)

high_path = (
    RESULTS_DIR
    / "high_probability_quality.csv"
)

row_path = (
    RESULTS_DIR
    / "row_level_ranking_audit.csv"
)

decile_df.to_csv(
    decile_path,
    index=False
)

threshold_df.to_csv(
    threshold_path,
    index=False
)

yearly_df.to_csv(
    yearly_path,
    index=False
)

high_df.to_csv(
    high_path,
    index=False
)

df[
    [
        "date",
        "prediction_probability",
        "aapl_adj_close",
        "future_price",
        "future_return",
        "future_positive",
        "year"
    ]
].to_csv(
    row_path,
    index=False
)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 80)
print("11. AUDIT INTERPRETATION")
print("=" * 80)

print("""
This audit is diagnostic only.

The main question is:

Do higher prediction probabilities correspond
to better future outcomes?

Important indicators:

1. Ranking correlation
   Higher positive correlation is desirable.

2. Decile behaviour
   Higher-probability deciles should ideally have
   better future outcomes.

3. Top vs bottom groups
   A useful model should separate these groups.

4. Threshold behaviour
   Higher thresholds should represent increasingly
   selective predictions.

5. Yearly consistency
   Ranking quality should not depend entirely
   on one historical period.

No production threshold is selected by this script.
""")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"Probability deciles:\n{decile_path}"
)

print(
    f"\nThreshold ranking quality:\n{threshold_path}"
)

print(
    f"\nYearly ranking quality:\n{yearly_path}"
)

print(
    f"\nHigh probability quality:\n{high_path}"
)

print(
    f"\nRow-level audit:\n{row_path}"
)

print("\n" + "=" * 80)
print(
    "NORTHGATE AI — PREDICTION RANKING AUDIT COMPLETED"
)
print("=" * 80)