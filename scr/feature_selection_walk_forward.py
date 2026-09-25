from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


# ============================================================
# NORTHGATE AI
# LEAKAGE-SAFE FEATURE SELECTION + WALK-FORWARD TEST
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    ROOT
    / "data"
    / "processed"
    / "AAPL_5day_normalized_train.parquet"
)

VAL_PATH = (
    ROOT
    / "data"
    / "processed"
    / "AAPL_5day_normalized_validation.parquet"
)

TEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "AAPL_5day_normalized_test.parquet"
)

FEATURE_IMPORTANCE_PATH = (
    ROOT
    / "results"
    / "feature_analysis"
    / "combined_feature_importance.csv"
)

RESULTS_DIR = (
    ROOT
    / "results"
    / "feature_selection_walk_forward"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

HOLDING_DAYS = 5

RANDOM_STATE = 42

N_ESTIMATORS = 300

MAX_DEPTH = 8

MIN_SAMPLES_LEAF = 10

MAX_FEATURES = "sqrt"

CLASS_WEIGHT = "balanced"

FEATURE_COUNTS = [
    5,
    10,
    15,
    20,
    24
]


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("NORTHGATE AI — FEATURE SELECTION + WALK-FORWARD TEST")
print("=" * 80)

print("""
Purpose:

1. Investigate which features contain useful information.
2. Rank features using TRAINING DATA ONLY.
3. Test reduced feature sets using walk-forward evaluation.
4. Compare feature counts objectively.

IMPORTANT:
- No future test information is used for feature selection.
- No production threshold is changed.
- Existing results are not modified.
- The untouched final test is NOT used to select features.
""")


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("1. LOADING DATA")
print("=" * 80)


train_df = pd.read_parquet(
    TRAIN_PATH
)

val_df = pd.read_parquet(
    VAL_PATH
)

test_df = pd.read_parquet(
    TEST_PATH
)


for df in [
    train_df,
    val_df,
    test_df
]:

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df.sort_values(
        "date",
        inplace=True
    )

    df.reset_index(
        drop=True,
        inplace=True
    )


print(
    f"TRAIN      : {len(train_df)} rows | "
    f"{train_df['date'].min().date()} → "
    f"{train_df['date'].max().date()}"
)

print(
    f"VALIDATION : {len(val_df)} rows | "
    f"{val_df['date'].min().date()} → "
    f"{val_df['date'].max().date()}"
)

print(
    f"TEST       : {len(test_df)} rows | "
    f"{test_df['date'].min().date()} → "
    f"{test_df['date'].max().date()}"
)


# ============================================================
# 2. IDENTIFY FEATURES
# ============================================================

print("\n" + "=" * 80)
print("2. IDENTIFYING FEATURES")
print("=" * 80)


excluded_columns = {
    "date",
    "target"
}


all_features = [
    column
    for column in train_df.columns
    if column not in excluded_columns
]


print(
    f"Total features: {len(all_features)}"
)


for feature in all_features:

    print(
        f" - {feature}"
    )


# ============================================================
# 3. TRAIN-ONLY FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("3. TRAIN-ONLY FEATURE IMPORTANCE")
print("=" * 80)

print(
    "Feature ranking uses TRAIN data only."
)


X_train = train_df[
    all_features
]

y_train = train_df[
    "target"
]


importance_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    max_features=MAX_FEATURES,
    class_weight=CLASS_WEIGHT,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


importance_model.fit(
    X_train,
    y_train
)


importance_results = pd.DataFrame(
    {
        "feature": all_features,
        "rf_importance":
            importance_model.feature_importances_
    }
)


importance_results = (
    importance_results
    .sort_values(
        "rf_importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    "\nTRAIN-ONLY RANDOM FOREST RANKING:"
)

print(
    importance_results.to_string(
        index=False
    )
)


# ============================================================
# 4. SIMPLE TRAIN CORRELATION CHECK
# ============================================================

print("\n" + "=" * 80)
print("4. TRAIN FEATURE RELATIONSHIP CHECK")
print("=" * 80)

correlation_results = []


for feature in all_features:

    try:

        direction_corr = spearmanr(
            train_df[feature],
            train_df["target"]
        ).statistic

    except Exception:

        direction_corr = np.nan


    correlation_results.append(
        {
            "feature": feature,
            "spearman_direction":
                direction_corr,
            "abs_spearman_direction":
                abs(direction_corr)
                if pd.notna(direction_corr)
                else np.nan
        }
    )


correlation_df = pd.DataFrame(
    correlation_results
)


# ============================================================
# 5. COMBINE FEATURE RANKINGS
# ============================================================

print("\n" + "=" * 80)
print("5. COMBINING FEATURE RANKINGS")
print("=" * 80)


ranking_df = (
    importance_results
    .merge(
        correlation_df,
        on="feature",
        how="left"
    )
)


ranking_df[
    "importance_rank"
] = (
    ranking_df[
        "rf_importance"
    ]
    .rank(
        ascending=False,
        method="min"
    )
)


ranking_df[
    "correlation_rank"
] = (
    ranking_df[
        "abs_spearman_direction"
    ]
    .rank(
        ascending=False,
        method="min"
    )
)


ranking_df[
    "combined_rank_score"
] = (
    ranking_df[
        "importance_rank"
    ]
    +
    ranking_df[
        "correlation_rank"
    ]
)


ranking_df = (
    ranking_df
    .sort_values(
        [
            "combined_rank_score",
            "importance_rank"
        ]
    )
    .reset_index(drop=True)
)


ranking_df[
    "final_rank"
] = (
    np.arange(
        len(ranking_df)
    )
    + 1
)


print(
    ranking_df[
        [
            "final_rank",
            "feature",
            "rf_importance",
            "abs_spearman_direction",
            "combined_rank_score"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 6. CREATE COMBINED DATA
# ============================================================

print("\n" + "=" * 80)
print("6. PREPARING WALK-FORWARD DATA")
print("=" * 80)


combined_df = pd.concat(
    [
        train_df,
        val_df,
        test_df
    ],
    ignore_index=True
)


combined_df = (
    combined_df
    .sort_values("date")
    .reset_index(drop=True)
)


print(
    f"Combined rows: {len(combined_df)}"
)

print(
    f"Period: "
    f"{combined_df['date'].min().date()} → "
    f"{combined_df['date'].max().date()}"
)


# ============================================================
# 7. WALK-FORWARD PREDICTION FUNCTION
# ============================================================

def run_walk_forward(
    feature_list
):

    predictions = []

    years = sorted(
        combined_df["date"]
        .dt.year
        .unique()
    )

    # Start from 2020 because the existing
    # walk-forward evaluation starts there.

    years = [
        year
        for year in years
        if year >= 2020
    ]


    for year in years:

        test_mask = (
            combined_df["date"].dt.year
            == year
        )

        train_mask = (
            combined_df["date"].dt.year
            < year
        )


        year_train = combined_df[
            train_mask
        ]

        year_test = combined_df[
            test_mask
        ]


        if (
            len(year_train) == 0
            or len(year_test) == 0
        ):
            continue


        X_train_year = (
            year_train[
                feature_list
            ]
        )

        y_train_year = (
            year_train[
                "target"
            ]
        )

        X_test_year = (
            year_test[
                feature_list
            ]
        )


        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )


        model.fit(
            X_train_year,
            y_train_year
        )


        probabilities = (
            model
            .predict_proba(
                X_test_year
            )[:, 1]
        )


        temp = year_test[
            [
                "date",
                "target"
            ]
        ].copy()


        temp[
            "prediction_probability"
        ] = probabilities


        temp[
            "year"
        ] = year


        predictions.append(
            temp
        )


    if not predictions:

        return pd.DataFrame()


    return (
        pd.concat(
            predictions,
            ignore_index=True
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


# ============================================================
# 8. EVALUATE EACH FEATURE COUNT
# ============================================================

print("\n" + "=" * 80)
print("7. FEATURE COUNT WALK-FORWARD TEST")
print("=" * 80)


evaluation_results = []

all_ranked_features = (
    ranking_df[
        "feature"
    ].tolist()
)


for feature_count in FEATURE_COUNTS:

    selected_features = (
        all_ranked_features[
            :feature_count
        ]
    )


    print(
        "\n"
        + "-" * 80
    )

    print(
        f"Testing TOP {feature_count} FEATURES"
    )

    print(
        "-" * 80
    )


    for feature in selected_features:

        print(
            f" - {feature}"
        )


    predictions = run_walk_forward(
        selected_features
    )


    if predictions.empty:

        print(
            "No predictions generated."
        )

        continue


    valid = predictions[
        "prediction_probability"
    ].notna()


    predictions = predictions[
        valid
    ].copy()


    y_true = predictions[
        "target"
    ]

    probabilities = predictions[
        "prediction_probability"
    ]


    if (
        y_true.nunique()
        >= 2
    ):

        auc = roc_auc_score(
            y_true,
            probabilities
        )

    else:

        auc = np.nan


    spearman_direction = (
        spearmanr(
            probabilities,
            y_true
        ).statistic
    )


    # Future return cannot be calculated
    # from processed prediction data alone.
    #
    # Therefore this test evaluates
    # classification discrimination only.


    result = {
        "feature_count":
            feature_count,

        "rows":
            len(predictions),

        "roc_auc":
            auc,

        "auc_vs_random":
            auc - 0.50
            if pd.notna(auc)
            else np.nan,

        "spearman_direction":
            spearman_direction,

        "mean_probability":
            probabilities.mean(),

        "actual_positive_rate":
            y_true.mean()
    }


    evaluation_results.append(
        result
    )


    print(
        f"\nRows              : "
        f"{len(predictions):,}"
    )

    print(
        f"ROC-AUC           : "
        f"{auc:.4f}"
    )

    print(
        f"AUC vs random     : "
        f"{auc - 0.50:.4f}"
    )

    print(
        f"Spearman          : "
        f"{spearman_direction:.4f}"
    )


evaluation_df = pd.DataFrame(
    evaluation_results
)


# ============================================================
# 9. YEARLY RESULTS FOR EACH FEATURE SET
# ============================================================

print("\n" + "=" * 80)
print("8. YEARLY FEATURE-SET RESULTS")
print("=" * 80)


yearly_results = []


for feature_count in FEATURE_COUNTS:

    selected_features = (
        all_ranked_features[
            :feature_count
        ]
    )


    predictions = run_walk_forward(
        selected_features
    )


    if predictions.empty:

        continue


    for year, group in (
        predictions.groupby("year")
    ):

        if (
            group["target"].nunique()
            < 2
        ):

            auc = np.nan

        else:

            auc = roc_auc_score(
                group["target"],
                group[
                    "prediction_probability"
                ]
            )


        yearly_results.append(
            {
                "feature_count":
                    feature_count,

                "year":
                    year,

                "rows":
                    len(group),

                "roc_auc":
                    auc,

                "spearman_direction":
                    spearmanr(
                        group[
                            "prediction_probability"
                        ],
                        group["target"]
                    ).statistic,

                "actual_positive_rate":
                    group[
                        "target"
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
# 10. SAVE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("9. SAVING RESULTS")
print("=" * 80)


ranking_path = (
    RESULTS_DIR
    / "feature_ranking.csv"
)

evaluation_path = (
    RESULTS_DIR
    / "feature_count_evaluation.csv"
)

yearly_path = (
    RESULTS_DIR
    / "yearly_feature_evaluation.csv"
)


ranking_df.to_csv(
    ranking_path,
    index=False
)

evaluation_df.to_csv(
    evaluation_path,
    index=False
)

yearly_df.to_csv(
    yearly_path,
    index=False
)


print(
    f"Feature ranking:\n"
    f"{ranking_path}"
)

print(
    f"\nFeature count evaluation:\n"
    f"{evaluation_path}"
)

print(
    f"\nYearly evaluation:\n"
    f"{yearly_path}"
)


# ============================================================
# 11. FINAL DIAGNOSTIC
# ============================================================

print("\n" + "=" * 80)
print("10. DIAGNOSTIC SUMMARY")
print("=" * 80)


if not evaluation_df.empty:

    best_row = (
        evaluation_df
        .sort_values(
            "roc_auc",
            ascending=False
        )
        .iloc[0]
    )


    print(
        f"Best feature count by "
        f"walk-forward ROC-AUC: "
        f"{int(best_row['feature_count'])}"
    )

    print(
        f"ROC-AUC: "
        f"{best_row['roc_auc']:.4f}"
    )

    print(
        f"AUC vs random: "
        f"{best_row['auc_vs_random']:.4f}"
    )


print("""
IMPORTANT:

This script is diagnostic.

A feature set is NOT automatically considered
better just because its ROC-AUC is slightly higher.

Look for:
- improvement over 0.50
- consistency across years
- enough observations
- stability rather than one unusually strong year

No production threshold has been changed.
""")


print("\n" + "=" * 80)
print(
    "NORTHGATE AI — FEATURE SELECTION WALK-FORWARD COMPLETED"
)
print("=" * 80)