from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "AAPL_5day_normalized_model_dataset.parquet"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("NORTHGATE AI — AAPL 5-DAY TARGET VALIDATION")
print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================

raw_df = pd.read_parquet(RAW_PATH)
model_df = pd.read_parquet(MODEL_PATH)

print("\nRaw feature dataset shape      :", raw_df.shape)
print("Normalized model dataset shape:", model_df.shape)


# ============================================================
# NORMALIZE RAW COLUMNS
# ============================================================

if isinstance(raw_df.columns, pd.MultiIndex):
    raw_df.columns = [
        "_".join(
            str(part)
            for part in column
            if str(part) != "nan"
        )
        for column in raw_df.columns
    ]


# ============================================================
# DATE
# ============================================================

if "date" not in raw_df.columns:
    raw_df = raw_df.reset_index()

if "Date" in raw_df.columns:
    raw_df = raw_df.rename(
        columns={"Date": "date"}
    )

if "date" not in raw_df.columns:
    raise ValueError("Could not find date column.")


raw_df["date"] = pd.to_datetime(raw_df["date"])
model_df["date"] = pd.to_datetime(model_df["date"])

raw_df = (
    raw_df
    .sort_values("date")
    .reset_index(drop=True)
)

model_df = (
    model_df
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

if "Close_AAPL" not in raw_df.columns:
    raise ValueError(
        "Close_AAPL was not found in features.parquet."
    )

if "target" not in model_df.columns:
    raise ValueError(
        "target was not found in model dataset."
    )


print("\nUsing EXACT target source:")
print("Close column: Close_AAPL")


# ============================================================
# RECREATE TARGET
# ============================================================

raw_df["future_close_5d"] = (
    raw_df["Close_AAPL"].shift(-5)
)

raw_df["expected_future_return"] = (
    raw_df["future_close_5d"]
    / raw_df["Close_AAPL"]
) - 1

raw_df["expected_target"] = (
    raw_df["expected_future_return"] > 0
).astype("Int64")


# ============================================================
# COMPARE WITH MODEL TARGET
# ============================================================

comparison = model_df[
    ["date", "target"]
].merge(
    raw_df[
        [
            "date",
            "Close_AAPL",
            "future_close_5d",
            "expected_future_return",
            "expected_target",
        ]
    ],
    on="date",
    how="inner"
)


comparison = comparison.dropna(
    subset=[
        "expected_future_return",
        "expected_target"
    ]
).copy()


comparison["target_match"] = (
    comparison["target"].astype(int)
    ==
    comparison["expected_target"].astype(int)
)


# ============================================================
# RESULTS
# ============================================================

total = len(comparison)
matches = int(comparison["target_match"].sum())
mismatches = total - matches

match_pct = (
    matches / total * 100
    if total > 0
    else 0
)


print("\n" + "=" * 80)
print("TARGET VALIDATION RESULTS")
print("=" * 80)

print(f"\nRows compared     : {total}")
print(f"Target matches    : {matches}")
print(f"Target mismatches : {mismatches}")
print(f"Match percentage  : {match_pct:.4f}%")


# ============================================================
# DISTRIBUTIONS
# ============================================================

print("\n" + "-" * 80)
print("MODEL TARGET DISTRIBUTION")
print("-" * 80)

print(
    model_df["target"]
    .value_counts()
    .sort_index()
)


print("\n" + "-" * 80)
print("EXPECTED TARGET DISTRIBUTION")
print("-" * 80)

print(
    comparison["expected_target"]
    .value_counts()
    .sort_index()
)


# ============================================================
# MISMATCH INSPECTION
# ============================================================

if mismatches > 0:

    print("\n" + "-" * 80)
    print("MISMATCH DETAILS")
    print("-" * 80)

    mismatch_df = comparison[
        ~comparison["target_match"]
    ]

    print(
        mismatch_df[
            [
                "date",
                "Close_AAPL",
                "future_close_5d",
                "expected_future_return",
                "target",
                "expected_target",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

else:

    print("\nNO TARGET MISMATCHES FOUND.")


# ============================================================
# FINAL VERDICT
# ============================================================

print("\n" + "=" * 80)
print("FINAL TARGET VALIDATION")
print("=" * 80)

if mismatches == 0:

    print("""
PASS

The model target exactly matches:

future_close_5d / current_close - 1 > 0

using Close_AAPL.

Target construction is internally consistent.
""")

elif match_pct >= 99.0:

    print("""
PASS WITH REVIEW

Target agreement is above 99%.

Inspect the mismatch rows before rebuilding
the dataset.
""")

else:

    print("""
FAIL

Target mismatch is too high.

The target-generation pipeline must be investigated
before continuing with model development.
""")


print("=" * 80)
print("TARGET VALIDATION COMPLETE")
print("=" * 80)