from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data" / "processed" / "features.parquet"
OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def column_to_text(column):
    """
    Converts normal columns and tuple/MultiIndex columns
    into a searchable string.
    """
    if isinstance(column, tuple):
        return " ".join(
            str(value)
            for value in column
            if value is not None
            and str(value).lower() != "nan"
        )

    return str(column)


def find_close_column(columns):
    """
    Finds the AAPL close-price column safely,
    even when columns are tuples.
    """

    column_details = []

    for column in columns:
        column_text = column_to_text(column)
        column_lower = column_text.lower()

        column_details.append(
            (column, column_text, column_lower)
        )

    # Prefer columns containing both AAPL and Close
    preferred_matches = [
        item
        for item in column_details
        if "aapl" in item[2]
        and "close" in item[2]
    ]

    if preferred_matches:
        return preferred_matches[0][0]

    # Fallback: any close column
    close_matches = [
        item
        for item in column_details
        if "close" in item[2]
    ]

    if close_matches:
        return close_matches[0][0]

    print("\nPossible columns:")
    for _, column_text, _ in column_details:
        print(f"- {column_text}")

    raise ValueError(
        "AAPL close-price column could not be found."
    )


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print_section("Loading raw features")

df = pd.read_parquet(INPUT_PATH)

df.index = pd.to_datetime(df.index)
df = df.sort_index()

print(f"Original shape: {df.shape}")
print(
    f"Date range: "
    f"{df.index.min()} -> {df.index.max()}"
)


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

print_section("Normalizing column names")

original_columns = list(df.columns)

normalized_column_names = [
    column_to_text(column)
    for column in original_columns
]

# Make duplicate names unique
seen_names = {}
unique_column_names = []

for column_name in normalized_column_names:
    if column_name not in seen_names:
        seen_names[column_name] = 0
        unique_column_names.append(column_name)
    else:
        seen_names[column_name] += 1
        unique_column_names.append(
            f"{column_name}_{seen_names[column_name]}"
        )

df.columns = unique_column_names

print(f"Total columns after normalization: {len(df.columns)}")


# ============================================================
# FIND AAPL CLOSE COLUMN
# ============================================================

print_section("Finding AAPL close-price column")

close_column = find_close_column(original_columns)

# Convert original tuple column to its normalized name
close_column_text = column_to_text(close_column)

matching_normalized_columns = [
    column
    for column in df.columns
    if column == close_column_text
]

if not matching_normalized_columns:
    # Fallback search
    matching_normalized_columns = [
        column
        for column in df.columns
        if "aapl" in column.lower()
        and "close" in column.lower()
    ]

if not matching_normalized_columns:
    raise ValueError(
        "The close column was found in original columns "
        "but could not be matched after normalization."
    )

close_column = matching_normalized_columns[0]

print(f"Using close column: {close_column}")


# ============================================================
# CREATE 5-DAY TARGET
# ============================================================

print_section("Creating 5-day target")

future_close = df[close_column].shift(-5)

future_return = (
    future_close / df[close_column]
) - 1

df["target"] = (
    future_return > 0
).astype("Int64")

print("5-day target created successfully.")


# ============================================================
# REMOVE LEAKAGE COLUMNS
# ============================================================

print_section("Removing leakage columns")

leakage_keywords = [
    "target",
    "future",
    "next",
    "forward",
    "label",
]

leakage_columns = []

for column in df.columns:
    column_lower = str(column).lower()

    if column == "target":
        continue

    if any(
        keyword in column_lower
        for keyword in leakage_keywords
    ):
        leakage_columns.append(column)

print("Columns removed:")

if leakage_columns:
    for column in sorted(leakage_columns):
        print(f"- {column}")
else:
    print("No leakage-related columns found.")


df = df.drop(
    columns=leakage_columns,
    errors="ignore"
)


# ============================================================
# SELECT SAFE NUMERIC FEATURES
# ============================================================

print_section("Selecting safe numeric features")

safe_feature_columns = [
    column
    for column in df.columns
    if column != "target"
]

# Remove date-like columns
safe_feature_columns = [
    column
    for column in safe_feature_columns
    if str(column).lower()
    not in [
        "date",
        "timestamp",
        "datetime",
    ]
]

numeric_feature_columns = df[
    safe_feature_columns
].select_dtypes(
    include=[np.number]
).columns.tolist()

print(
    f"Number of safe numeric features: "
    f"{len(numeric_feature_columns)}")

print("Selected features:")
for column in numeric_feature_columns:
    print(f"- {column}")

# Remove constant features
feature_std = df[numeric_feature_columns].std()

constant_features = feature_std[
    feature_std == 0
].index.tolist()

if constant_features:
    print("\nRemoving constant features:")
    for column in constant_features:
        print(f"- {column}")

    numeric_feature_columns = [
        column
        for column in numeric_feature_columns
        if column not in constant_features]


# ============================================================
# BUILD CLEAN DATASET
# ============================================================

print_section("Building clean dataset")

clean_df = df[
    numeric_feature_columns + ["target"]
].copy()

clean_df = clean_df.replace(
    [np.inf, -np.inf],
    np.nan
)

clean_df = clean_df.dropna()

clean_df["target"] = clean_df["target"].astype(int)

clean_df = clean_df.reset_index()

# The old index may be called index
if "index" in clean_df.columns:
    clean_df = clean_df.rename(
        columns={"index": "date"}
    )

# If reset_index retained another name, detect it
if "date" not in clean_df.columns:
    possible_date_columns = [
        column
        for column in clean_df.columns
        if "date" in str(column).lower()
        or "time" in str(column).lower()
    ]

    if possible_date_columns:
        clean_df = clean_df.rename(
            columns={
                possible_date_columns[0]: "date"
            }
        )

if "date" not in clean_df.columns:
    raise ValueError(
        "Date column could not be identified after reset_index."
    )

clean_df["date"] = pd.to_datetime(
    clean_df["date"]
)

clean_df = clean_df.sort_values(
    by="date"
).reset_index(drop=True)


# ============================================================
# FINAL LEAKAGE CHECK
# ============================================================

print_section("Final leakage check")

remaining_leakage_columns = []

for column in clean_df.columns:
    if column in ["date", "target"]:
        continue

    column_lower = str(column).lower()

    if any(
        keyword in column_lower
        for keyword in leakage_keywords
    ):
        remaining_leakage_columns.append(column)

if remaining_leakage_columns:
    print("WARNING: Leakage columns still remain:")

    for column in remaining_leakage_columns:
        print(f"- {column}")

    raise ValueError(
        "Leakage columns still remain in the clean dataset."
    )

print("No leakage-related feature columns remain: PASS")


# ============================================================
# DATASET SUMMARY
# ============================================================

print_section("Final dataset summary")

print(f"Final shape: {clean_df.shape}")
print(
    f"Date range: "
    f"{clean_df['date'].min()} -> "
    f"{clean_df['date'].max()}"
)

print("\nFinal columns:")
for column in clean_df.columns:
    print(f"- {column}")

print("\nTarget distribution:")
print(clean_df["target"].value_counts())

print("\nMissing values:")
print(clean_df.isna().sum().sum())


# ============================================================
# TIME-BASED SPLIT
# ============================================================

print_section("Creating train, validation and test splits")

total_rows = len(clean_df)

train_end = int(total_rows * 0.70)
validation_end = int(total_rows * 0.85)

train_df = clean_df.iloc[
    :train_end
].copy()

validation_df = clean_df.iloc[
    train_end:validation_end
].copy()

test_df = clean_df.iloc[
    validation_end:
].copy()

print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {validation_df.shape}")
print(f"Test shape: {test_df.shape}")

print(
    f"\nTrain dates: "
    f"{train_df['date'].min()} -> "
    f"{train_df['date'].max()}"
)

print(
    f"Validation dates: "
    f"{validation_df['date'].min()} -> "
    f"{validation_df['date'].max()}"
)

print(
    f"Test dates: "
    f"{test_df['date'].min()} -> "
    f"{test_df['date'].max()}"
)


# ============================================================
# SAVE CLEAN DATASETS
# ============================================================

print_section("Saving clean datasets")

combined_path = OUTPUT_DIR / "aapl_5day_clean.parquet"
train_path = OUTPUT_DIR / "aapl_5day_train.parquet"
validation_path = OUTPUT_DIR / "aapl_5day_validation.parquet"
test_path = OUTPUT_DIR / "aapl_5day_test.parquet"

clean_df.to_parquet(
    combined_path,
    index=False
)

train_df.to_parquet(
    train_path,
    index=False
)

validation_df.to_parquet(
    validation_path,
    index=False
)

test_df.to_parquet(
    test_path,
    index=False
)

print(f"Saved: {combined_path}")
print(f"Saved: {train_path}")
print(f"Saved: {validation_path}")
print(f"Saved: {test_path}")

print("\nClean dataset creation completed successfully.")