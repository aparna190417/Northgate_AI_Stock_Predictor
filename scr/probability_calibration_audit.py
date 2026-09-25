from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# NORTHGATE AI
# PROBABILITY CALIBRATION + PREDICTION QUALITY AUDIT
#
# IMPORTANT:
# - Uses EXISTING walk-forward predictions
# - Does NOT retrain the model
# - Does NOT change the locked threshold
# - Does NOT use the untouched final test
# - Diagnostic analysis only
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
    / "probability_calibration_audit"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

LOCKED_THRESHOLD = 0.65

HOLDING_DAYS = 5

TRANSACTION_COST = 0.001

INITIAL_CAPITAL = 100000


# Probability bins
PROBABILITY_BINS = [
    0.00,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    1.00
]


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("NORTHGATE AI — PROBABILITY CALIBRATION + PREDICTION QUALITY AUDIT")
print("=" * 80)

print("""
Purpose:
This audit examines whether model probabilities correspond
to actual future 5-trading-day outcomes.

It does NOT:
- retrain the model
- change the locked threshold
- select a new production threshold
- modify existing results
- use the untouched final test

This is a diagnostic analysis only.
""")


print("Configuration:")
print(f"Locked threshold      : {LOCKED_THRESHOLD:.2f}")
print(f"Holding period        : {HOLDING_DAYS} trading days")
print(f"Transaction cost      : {TRANSACTION_COST:.2%}")
print(f"Initial capital       : ₹{INITIAL_CAPITAL:,.2f}")


# ============================================================
# 1. LOAD PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("1. LOADING EXISTING WALK-FORWARD PREDICTIONS")
print("=" * 80)


if not PREDICTIONS_PATH.exists():

    raise FileNotFoundError(
        f"""
Could not find:

{PREDICTIONS_PATH}

Run:

python scr/walk_forward_final.py

first.
"""
    )


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


required_columns = [
    "date",
    "prediction_probability",
    "aapl_adj_close"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing columns: "
        + ", ".join(missing_columns)
    )


print(
    f"Prediction rows: {len(df)}"
)

print(
    f"Prediction period: "
    f"{df['date'].min().date()} "
    f"→ "
    f"{df['date'].max().date()}"
)


# ============================================================
# 2. CALCULATE ACTUAL FUTURE 5-DAY RETURN
# ============================================================

print("\n" + "=" * 80)
print("2. CALCULATING ACTUAL FUTURE 5-DAY RETURNS")
print("=" * 80)


df["future_price"] = (
    df[
        "aapl_adj_close"
    ]
    .shift(-HOLDING_DAYS)
)


df["actual_5day_return"] = (
    df["future_price"]
    /
    df["aapl_adj_close"]
) - 1


df["actual_direction"] = (
    df["actual_5day_return"] > 0
).astype(int)


valid_df = (
    df[
        df["future_price"].notna()
    ]
    .copy()
    .reset_index(drop=True)
)


print(
    f"Rows with valid future return: "
    f"{len(valid_df)}"
)


# ============================================================
# 3. BASIC PREDICTION QUALITY
# ============================================================

print("\n" + "=" * 80)
print("3. BASIC PREDICTION QUALITY")
print("=" * 80)


probabilities = (
    valid_df[
        "prediction_probability"
    ]
)


actual_direction = (
    valid_df[
        "actual_direction"
    ]
)


predicted_direction = (
    probabilities >= 0.50
).astype(int)


accuracy = (
    predicted_direction
    == actual_direction
).mean()


print(
    f"Prediction accuracy at 0.50: "
    f"{accuracy:.2%}"
)


print(
    f"Average actual 5-day return: "
    f"{valid_df['actual_5day_return'].mean():.4%}"
)


print(
    f"Median actual 5-day return: "
    f"{valid_df['actual_5day_return'].median():.4%}"
)


# ============================================================
# 4. PROBABILITY BIN ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("4. PROBABILITY BIN ANALYSIS")
print("=" * 80)


valid_df["probability_bin"] = pd.cut(
    valid_df[
        "prediction_probability"
    ],
    bins=PROBABILITY_BINS,
    include_lowest=True,
    right=True
)


bin_results = []


for bin_value, group in valid_df.groupby(
    "probability_bin",
    observed=False
):

    if len(group) == 0:

        continue


    mean_probability = (
        group[
            "prediction_probability"
        ].mean()
    )


    actual_positive_rate = (
        group[
            "actual_direction"
        ].mean()
    )


    mean_return = (
        group[
            "actual_5day_return"
        ].mean()
    )


    median_return = (
        group[
            "actual_5day_return"
        ].median()
    )


    positive_return_rate = (
        group[
            "actual_5day_return"
        ] > 0
    ).mean()


    bin_results.append(
        {
            "probability_bin":
                str(bin_value),

            "rows":
                len(group),

            "mean_predicted_probability":
                mean_probability,

            "actual_positive_rate":
                actual_positive_rate,

            "probability_error":
                (
                    actual_positive_rate
                    -
                    mean_probability
                ),

            "mean_actual_5day_return":
                mean_return,

            "median_actual_5day_return":
                median_return,

            "positive_return_rate":
                positive_return_rate
        }
    )


    print(
        f"{str(bin_value):<18} | "
        f"Rows={len(group):>4} | "
        f"PredProb={mean_probability:.3f} | "
        f"ActualPositive={actual_positive_rate:.2%} | "
        f"Mean5DReturn={mean_return:.2%}"
    )


bin_df = pd.DataFrame(
    bin_results
)


# ============================================================
# 5. THRESHOLD-LEVEL PREDICTION QUALITY
# ============================================================

print("\n" + "=" * 80)
print("5. THRESHOLD-LEVEL PREDICTION QUALITY")
print("=" * 80)


thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]


threshold_quality = []


for threshold in thresholds:

    selected = (
        valid_df[
            valid_df[
                "prediction_probability"
            ] >= threshold
        ]
        .copy()
    )


    if len(selected) == 0:

        continue


    actual_positive_rate = (
        selected[
            "actual_direction"
        ].mean()
    )


    mean_probability = (
        selected[
            "prediction_probability"
        ].mean()
    )


    mean_return = (
        selected[
            "actual_5day_return"
        ].mean()
    )


    median_return = (
        selected[
            "actual_5day_return"
        ].median()
    )


    positive_return_rate = (
        selected[
            "actual_5day_return"
        ] > 0
    ).mean()


    threshold_quality.append(
        {
            "threshold":
                threshold,

            "rows":
                len(selected),

            "signal_rate":
                len(selected)
                /
                len(valid_df),

            "mean_probability":
                mean_probability,

            "actual_positive_rate":
                actual_positive_rate,

            "probability_error":
                (
                    actual_positive_rate
                    -
                    mean_probability
                ),

            "mean_actual_5day_return":
                mean_return,

            "median_actual_5day_return":
                median_return,

            "positive_return_rate":
                positive_return_rate
        }
    )


    print(
        f"Threshold={threshold:.2f} | "
        f"Rows={len(selected)} | "
        f"SignalRate="
        f"{len(selected)/len(valid_df):.2%} | "
        f"MeanProb="
        f"{mean_probability:.3f} | "
        f"ActualPositive="
        f"{actual_positive_rate:.2%} | "
        f"Mean5DReturn="
        f"{mean_return:.2%}"
    )


threshold_quality_df = pd.DataFrame(
    threshold_quality
)


# ============================================================
# 6. HIGH-CONFIDENCE PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("6. HIGH-CONFIDENCE PREDICTION ANALYSIS")
print("=" * 80)


high_confidence_levels = [
    0.55,
    0.60,
    0.65,
    0.70
]


high_confidence_results = []


for threshold in high_confidence_levels:

    selected = (
        valid_df[
            valid_df[
                "prediction_probability"
            ] >= threshold
        ]
        .copy()
    )


    if len(selected) == 0:

        continue


    high_confidence_results.append(
        {
            "minimum_probability":
                threshold,

            "rows":
                len(selected),

            "actual_positive_rate":
                selected[
                    "actual_direction"
                ].mean(),

            "mean_actual_5day_return":
                selected[
                    "actual_5day_return"
                ].mean(),

            "median_actual_5day_return":
                selected[
                    "actual_5day_return"
                ].median(),

            "positive_return_rate":
                (
                    selected[
                        "actual_5day_return"
                    ] > 0
                ).mean()
        }
    )


high_confidence_df = pd.DataFrame(
    high_confidence_results
)


# ============================================================
# 7. QUARTILE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("7. PROBABILITY QUARTILE ANALYSIS")
print("=" * 80)


valid_df["probability_quartile"] = (
    pd.qcut(
        valid_df[
            "prediction_probability"
        ],
        q=4,
        duplicates="drop"
    )
)


quartile_results = []


for quartile, group in valid_df.groupby(
    "probability_quartile",
    observed=False
):

    if len(group) == 0:

        continue


    quartile_results.append(
        {
            "quartile":
                str(quartile),

            "rows":
                len(group),

            "mean_probability":
                group[
                    "prediction_probability"
                ].mean(),

            "actual_positive_rate":
                group[
                    "actual_direction"
                ].mean(),

            "mean_actual_5day_return":
                group[
                    "actual_5day_return"
                ].mean(),

            "median_actual_5day_return":
                group[
                    "actual_5day_return"
                ].median(),

            "positive_return_rate":
                (
                    group[
                        "actual_5day_return"
                    ] > 0
                ).mean()
        }
    )


quartile_df = pd.DataFrame(
    quartile_results
)


for _, row in quartile_df.iterrows():

    print(
        f"{row['quartile']} | "
        f"Rows={int(row['rows'])} | "
        f"MeanProb="
        f"{row['mean_probability']:.3f} | "
        f"ActualPositive="
        f"{row['actual_positive_rate']:.2%} | "
        f"Mean5DReturn="
        f"{row['mean_actual_5day_return']:.2%}"
    )


# ============================================================
# 8. CALIBRATION ERROR
# ============================================================

print("\n" + "=" * 80)
print("8. CALIBRATION ERROR")
print("=" * 80)


if len(bin_df) > 0:

    bin_df["absolute_probability_error"] = (
        bin_df[
            "probability_error"
        ].abs()
    )


    weighted_calibration_error = (
        (
            bin_df[
                "absolute_probability_error"
            ]
            *
            bin_df["rows"]
        ).sum()
        /
        bin_df["rows"].sum()
    )


else:

    weighted_calibration_error = np.nan


print(
    "Weighted calibration error: "
    f"{weighted_calibration_error:.4f}"
)


# ============================================================
# 9. RANK CORRELATION
# ============================================================

print("\n" + "=" * 80)
print("9. PROBABILITY vs ACTUAL RETURN RELATIONSHIP")
print("=" * 80)


spearman_probability_return = (
    valid_df[
        [
            "prediction_probability",
            "actual_5day_return"
        ]
    ]
    .corr(
        method="spearman"
    )
    .iloc[0, 1]
)


spearman_probability_direction = (
    valid_df[
        [
            "prediction_probability",
            "actual_direction"
        ]
    ]
    .corr(
        method="spearman"
    )
    .iloc[0, 1]
)


print(
    "Spearman correlation "
    "probability vs 5-day return: "
    f"{spearman_probability_return:.4f}"
)


print(
    "Spearman correlation "
    "probability vs direction: "
    f"{spearman_probability_direction:.4f}"
)


# ============================================================
# 10. LOCKED THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("10. LOCKED THRESHOLD ANALYSIS")
print("=" * 80)


locked = (
    valid_df[
        valid_df[
            "prediction_probability"
        ] >= LOCKED_THRESHOLD
    ]
    .copy()
)


print(
    f"Locked threshold: "
    f"{LOCKED_THRESHOLD:.2f}"
)


print(
    f"Rows >= threshold: "
    f"{len(locked)}"
)


if len(locked) > 0:

    print(
        f"Mean probability: "
        f"{locked['prediction_probability'].mean():.4f}"
    )

    print(
        f"Actual positive rate: "
        f"{locked['actual_direction'].mean():.2%}"
    )

    print(
        f"Mean actual 5-day return: "
        f"{locked['actual_5day_return'].mean():.2%}"
    )

    print(
        f"Median actual 5-day return: "
        f"{locked['actual_5day_return'].median():.2%}"
    )

    print(
        f"Positive return rate: "
        f"{(locked['actual_5day_return'] > 0).mean():.2%}"
    )

else:

    print(
        "No predictions reached the locked threshold."
    )


# ============================================================
# 11. YEARLY CALIBRATION CHECK
# ============================================================

print("\n" + "=" * 80)
print("11. YEARLY CALIBRATION CHECK")
print("=" * 80)


valid_df["year"] = (
    valid_df["date"].dt.year
)


yearly_results = []


for year, group in valid_df.groupby(
    "year"
):

    if len(group) == 0:

        continue


    yearly_results.append(
        {
            "year":
                year,

            "rows":
                len(group),

            "mean_probability":
                group[
                    "prediction_probability"
                ].mean(),

            "actual_positive_rate":
                group[
                    "actual_direction"
                ].mean(),

            "mean_actual_5day_return":
                group[
                    "actual_5day_return"
                ].mean(),

            "positive_return_rate":
                (
                    group[
                        "actual_5day_return"
                    ] > 0
                ).mean()
        }
    )


yearly_df = pd.DataFrame(
    yearly_results
)


for _, row in yearly_df.iterrows():

    print(
        f"Year={int(row['year'])} | "
        f"Rows={int(row['rows'])} | "
        f"MeanProb="
        f"{row['mean_probability']:.3f} | "
        f"ActualPositive="
        f"{row['actual_positive_rate']:.2%} | "
        f"Mean5DReturn="
        f"{row['mean_actual_5day_return']:.2%}"
    )


# ============================================================
# 12. SAVE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("12. SAVING RESULTS")
print("=" * 80)


bin_path = (
    RESULTS_DIR
    / "probability_bins.csv"
)

bin_df.to_csv(
    bin_path,
    index=False
)


threshold_path = (
    RESULTS_DIR
    / "threshold_prediction_quality.csv"
)

threshold_quality_df.to_csv(
    threshold_path,
    index=False
)


high_confidence_path = (
    RESULTS_DIR
    / "high_confidence_predictions.csv"
)

high_confidence_df.to_csv(
    high_confidence_path,
    index=False
)


quartile_path = (
    RESULTS_DIR
    / "probability_quartiles.csv"
)

quartile_df.to_csv(
    quartile_path,
    index=False
)


yearly_path = (
    RESULTS_DIR
    / "yearly_probability_quality.csv"
)

yearly_df.to_csv(
    yearly_path,
    index=False
)


# Save row-level diagnostic data

row_level_path = (
    RESULTS_DIR
    / "row_level_probability_audit.csv"
)

valid_df.to_csv(
    row_level_path,
    index=False
)


# ============================================================
# 13. FINAL INTERPRETATION GUIDE
# ============================================================

print("\n" + "=" * 80)
print("13. INTERPRETATION GUIDE")
print("=" * 80)

print("""
Look for the following:

1. CALIBRATION
   If predicted probability rises from one bin to the next,
   actual positive rate should generally rise too.

2. RETURN RELATIONSHIP
   Higher probability groups should ideally have better
   average future 5-day returns.

3. THRESHOLD 0.65
   Check whether predictions >= 0.65 actually produce a
   meaningfully higher positive rate/return.

4. MONOTONICITY
   If probability increases but actual outcomes do not
   improve consistently, the probability output may not
   be well calibrated.

5. SAMPLE SIZE
   Very small high-probability groups can produce unstable
   statistics. Do not treat a tiny group as conclusive.

6. YEARLY CONSISTENCY
   Check whether the probability relationship remains
   reasonably consistent across years.

IMPORTANT:
This audit does NOT choose a new production threshold.
It is only used to understand model behaviour.
""")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"Probability bins:\n{bin_path}"
)

print(
    f"\nThreshold quality:\n{threshold_path}"
)

print(
    f"\nHigh-confidence analysis:\n{high_confidence_path}"
)

print(
    f"\nProbability quartiles:\n{quartile_path}"
)

print(
    f"\nYearly analysis:\n{yearly_path}"
)

print(
    f"\nRow-level audit:\n{row_level_path}"
)


print("\n" + "=" * 80)
print(
    "NORTHGATE AI — PROBABILITY CALIBRATION "
    "AUDIT COMPLETED"
)
print("=" * 80)