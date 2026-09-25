from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from scipy.stats import spearmanr


# ============================================================
# NORTHGATE AI
# MODEL DISCRIMINATION TEST
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    ROOT
    / "results"
    / "walk_forward_final"
    / "walk_forward_predictions.csv"
)

FEATURES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "features.parquet"
)

RESULTS_DIR = (
    ROOT
    / "results"
    / "model_discrimination_test"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


HOLDING_DAYS = 5
THRESHOLD = 0.65


print("=" * 80)
print("NORTHGATE AI — MODEL DISCRIMINATION TEST")
print("=" * 80)

print("""
Purpose:
Test whether the model actually discriminates
between positive and negative future 5-day outcomes.

This script:
- does NOT retrain the model
- does NOT change the production threshold
- does NOT modify existing results
- does NOT use the untouched final test separately

It evaluates the EXISTING walk-forward predictions.
""")


# ============================================================
# 1. LOAD WALK-FORWARD PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
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
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)


required_columns = [
    "date",
    "prediction_probability",
    "aapl_adj_close"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# 2. CALCULATE ACTUAL FUTURE 5-DAY OUTCOME
# ============================================================

print("\n" + "=" * 80)
print("2. CALCULATING FUTURE OUTCOME")
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

df["actual_direction"] = (
    df["future_return"] > 0
).astype(int)


df = df.dropna(
    subset=[
        "prediction_probability",
        "future_return"
    ]
).reset_index(drop=True)


print(
    f"Valid observations: {len(df):,}"
)


# ============================================================
# 3. BASIC DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("3. BASIC OUTCOME DISTRIBUTION")
print("=" * 80)

print(
    f"Positive outcomes : "
    f"{df['actual_direction'].sum():,}"
)

print(
    f"Negative outcomes : "
    f"{(df['actual_direction'] == 0).sum():,}"
)

print(
    f"Positive rate     : "
    f"{df['actual_direction'].mean():.2%}"
)

print(
    f"Mean future return: "
    f"{df['future_return'].mean():.4%}"
)

print(
    f"Median future return: "
    f"{df['future_return'].median():.4%}"
)


# ============================================================
# 4. ROC-AUC
# ============================================================

print("\n" + "=" * 80)
print("4. CLASSIFICATION DISCRIMINATION")
print("=" * 80)

y_true = df["actual_direction"]

probabilities = df[
    "prediction_probability"
]


roc_auc = roc_auc_score(
    y_true,
    probabilities
)

average_precision = (
    average_precision_score(
        y_true,
        probabilities
    )
)

predicted_class = (
    probabilities >= 0.50
).astype(int)


accuracy = accuracy_score(
    y_true,
    predicted_class
)

precision = precision_score(
    y_true,
    predicted_class,
    zero_division=0
)

recall = recall_score(
    y_true,
    predicted_class,
    zero_division=0
)

f1 = f1_score(
    y_true,
    predicted_class,
    zero_division=0
)


print(
    f"ROC-AUC            : {roc_auc:.4f}"
)

print(
    f"Average Precision  : {average_precision:.4f}"
)

print(
    f"Accuracy @ 0.50    : {accuracy:.2%}"
)

print(
    f"Precision @ 0.50   : {precision:.2%}"
)

print(
    f"Recall @ 0.50      : {recall:.2%}"
)

print(
    f"F1 @ 0.50           : {f1:.4f}"
)


# ============================================================
# 5. SPEARMAN RANKING
# ============================================================

print("\n" + "=" * 80)
print("5. RANKING QUALITY")
print("=" * 80)

spearman_return = spearmanr(
    probabilities,
    df["future_return"]
)

spearman_direction = spearmanr(
    probabilities,
    df["actual_direction"]
)


print(
    f"Probability vs future return:"
    f" {spearman_return.statistic:.4f}"
)

print(
    f"Probability vs direction:"
    f" {spearman_direction.statistic:.4f}"
)


# ============================================================
# 6. RANDOM / BASELINE COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("6. BASELINE COMPARISON")
print("=" * 80)

baseline_probability = (
    y_true.mean()
)

baseline_auc = 0.50


print(
    f"Actual positive rate:"
    f" {baseline_probability:.2%}"
)

print(
    f"Random ROC-AUC:"
    f" {baseline_auc:.2f}"
)

print(
    f"Model ROC-AUC:"
    f" {roc_auc:.4f}"
)

print(
    f"AUC improvement over random:"
    f" {roc_auc - baseline_auc:.4f}"
)


# ============================================================
# 7. PROBABILITY QUINTILES
# ============================================================

print("\n" + "=" * 80)
print("7. PROBABILITY QUINTILE TEST")
print("=" * 80)

df["probability_quintile"] = pd.qcut(
    df["prediction_probability"],
    q=5,
    labels=[
        "Q1 Lowest",
        "Q2",
        "Q3",
        "Q4",
        "Q5 Highest"
    ],
    duplicates="drop"
)


quintile_results = (
    df
    .groupby(
        "probability_quintile",
        observed=True
    )
    .agg(
        rows=(
            "prediction_probability",
            "size"
        ),
        mean_probability=(
            "prediction_probability",
            "mean"
        ),
        positive_rate=(
            "actual_direction",
            "mean"
        ),
        mean_future_return=(
            "future_return",
            "mean"
        ),
        median_future_return=(
            "future_return",
            "median"
        )
    )
    .reset_index()
)


print(
    quintile_results.to_string(
        index=False
    )
)


# ============================================================
# 8. TOP VS BOTTOM
# ============================================================

print("\n" + "=" * 80)
print("8. TOP VS BOTTOM 20%")
print("=" * 80)

bottom = df[
    df["probability_quintile"]
    == "Q1 Lowest"
]

top = df[
    df["probability_quintile"]
    == "Q5 Highest"
]


bottom_positive = (
    bottom["actual_direction"].mean()
)

top_positive = (
    top["actual_direction"].mean()
)

bottom_return = (
    bottom["future_return"].mean()
)

top_return = (
    top["future_return"].mean()
)


print(
    f"Bottom 20% positive rate:"
    f" {bottom_positive:.2%}"
)

print(
    f"Top 20% positive rate:"
    f" {top_positive:.2%}"
)

print(
    f"Difference:"
    f" {(top_positive - bottom_positive):.2%}"
)

print()

print(
    f"Bottom 20% mean return:"
    f" {bottom_return:.4%}"
)

print(
    f"Top 20% mean return:"
    f" {top_return:.4%}"
)

print(
    f"Return difference:"
    f" {(top_return - bottom_return):.4%}"
)


# ============================================================
# 9. LOCKED THRESHOLD
# ============================================================

print("\n" + "=" * 80)
print("9. LOCKED THRESHOLD ANALYSIS")
print("=" * 80)

threshold_df = df[
    df["prediction_probability"]
    >= THRESHOLD
]


print(
    f"Locked threshold:"
    f" {THRESHOLD:.2f}"
)

print(
    f"Rows >= threshold:"
    f" {len(threshold_df)}"
)

if len(threshold_df) > 0:

    print(
        f"Positive rate:"
        f" {threshold_df['actual_direction'].mean():.2%}"
    )

    print(
        f"Mean future return:"
        f" {threshold_df['future_return'].mean():.4%}"
    )

    print(
        f"Median future return:"
        f" {threshold_df['future_return'].median():.4%}"
    )


# ============================================================
# 10. YEARLY DISCRIMINATION
# ============================================================

print("\n" + "=" * 80)
print("10. YEARLY DISCRIMINATION")
print("=" * 80)

df["year"] = (
    df["date"].dt.year
)


yearly_results = []


for year, group in df.groupby(
    "year"
):

    if (
        group["actual_direction"]
        .nunique()
        < 2
    ):

        auc = np.nan

    else:

        auc = roc_auc_score(
            group["actual_direction"],
            group[
                "prediction_probability"
            ]
        )


    correlation = spearmanr(
        group[
            "prediction_probability"
        ],
        group[
            "future_return"
        ]
    ).statistic


    yearly_results.append(
        {
            "year": year,
            "rows": len(group),
            "roc_auc": auc,
            "spearman_return":
                correlation,
            "mean_probability":
                group[
                    "prediction_probability"
                ].mean(),
            "actual_positive_rate":
                group[
                    "actual_direction"
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
# 11. SAVE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("11. SAVING RESULTS")
print("=" * 80)


summary = pd.DataFrame(
    [
        {
            "rows": len(df),
            "roc_auc": roc_auc,
            "average_precision":
                average_precision,
            "accuracy":
                accuracy,
            "precision":
                precision,
            "recall":
                recall,
            "f1":
                f1,
            "spearman_return":
                spearman_return.statistic,
            "spearman_direction":
                spearman_direction.statistic,
            "bottom20_positive_rate":
                bottom_positive,
            "top20_positive_rate":
                top_positive,
            "positive_rate_difference":
                top_positive - bottom_positive,
            "bottom20_mean_return":
                bottom_return,
            "top20_mean_return":
                top_return,
            "return_difference":
                top_return - bottom_return,
            "threshold":
                THRESHOLD,
            "threshold_rows":
                len(threshold_df)
        }
    ]
)


summary_path = (
    RESULTS_DIR
    / "model_discrimination_summary.csv"
)

quintile_path = (
    RESULTS_DIR
    / "probability_quintiles.csv"
)

yearly_path = (
    RESULTS_DIR
    / "yearly_discrimination.csv"
)

row_path = (
    RESULTS_DIR
    / "row_level_discrimination.csv"
)


summary.to_csv(
    summary_path,
    index=False
)

quintile_results.to_csv(
    quintile_path,
    index=False
)

yearly_df.to_csv(
    yearly_path,
    index=False
)

df.to_csv(
    row_path,
    index=False
)


print(
    f"Summary:\n{summary_path}"
)

print(
    f"\nQuintiles:\n{quintile_path}"
)

print(
    f"\nYearly:\n{yearly_path}"
)

print(
    f"\nRow-level:\n{row_path}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print(
    "NORTHGATE AI — MODEL DISCRIMINATION TEST COMPLETED"
)
print("=" * 80)