from pathlib import Path
import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

print("=" * 60)
print("CHECKING RANDOM FOREST PROBABILITIES")
print("=" * 60)

train_df = pd.read_parquet(DATA_DIR / "AAPL_clean_train.parquet")
validation_df = pd.read_parquet(DATA_DIR / "AAPL_clean_validation.parquet")
test_df = pd.read_parquet(DATA_DIR / "AAPL_clean_test.parquet")

model = joblib.load(
    DATA_DIR / "AAPL_clean_random_forest_model.pkl"
)

target_column = "target"

feature_columns = [
    column for column in train_df.columns
    if column != target_column
]

X_train = train_df[feature_columns].copy()
X_validation = validation_df[feature_columns].copy()
X_test = test_df[feature_columns].copy()

# Same missing-value treatment used during training
train_medians = X_train.median()

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(train_medians)
X_validation = X_validation.replace(
    [np.inf, -np.inf], np.nan
).fillna(train_medians)
X_test = X_test.replace(
    [np.inf, -np.inf], np.nan
).fillna(train_medians)

for name, X in [
    ("TRAIN", X_train),
    ("VALIDATION", X_validation),
    ("TEST", X_test),
]:
    probabilities = model.predict_proba(X)[:, 1]

    print("\n" + "=" * 60)
    print(f"{name} PROBABILITIES")
    print("=" * 60)

    print(f"Minimum probability : {probabilities.min():.4f}")
    print(f"Maximum probability : {probabilities.max():.4f}")
    print(f"Mean probability    : {probabilities.mean():.4f}")
    print(f"Median probability  : {np.median(probabilities):.4f}")

    print("\nProbability ranges:")
    print(
        pd.cut(
            probabilities,
            bins=[0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
            include_lowest=True
        ).value_counts().sort_index()
    )

    print("\nPredictions at different thresholds:")

    for threshold in [0.30, 0.35, 0.40, 0.45, 0.50]:
        predictions = (probabilities >= threshold).astype(int)

        print(
            f"Threshold {threshold:.2f} | "
            f"Predicted UP: {predictions.sum():3d} | "
            f"Predicted DOWN: {(predictions == 0).sum():3d}"
        )

print("\nProbability check complete.")