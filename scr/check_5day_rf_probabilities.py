from pathlib import Path
import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data' / 'processed'

print('=' * 70)
print('CHECKING AAPL 5-DAY RANDOM FOREST PROBABILITIES')
print('=' * 70)

train_df = pd.read_parquet(DATA_DIR / 'aapl_5day_train.parquet')
validation_df = pd.read_parquet(DATA_DIR / 'aapl_5day_validation.parquet')
test_df = pd.read_parquet(DATA_DIR / 'aapl_5day_test.parquet')

model = joblib.load(
    DATA_DIR / 'AAPL_5day_random_forest_model.pkl'
)

def prepare_features(df):
    df = df.copy()

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['date_year'] = df['date'].dt.year
        df['date_month'] = df['date'].dt.month
        df['date_day'] = df['date'].dt.day
        df['date_day_of_week'] = df['date'].dt.dayofweek
        df = df.drop(columns=['date'])

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['date_year'] = df['Date'].dt.year
        df['date_month'] = df['Date'].dt.month
        df['date_day'] = df['Date'].dt.day
        df['date_day_of_week'] = df['Date'].dt.dayofweek
        df = df.drop(columns=['Date'])

    return df

train_df = prepare_features(train_df)
validation_df = prepare_features(validation_df)
test_df = prepare_features(test_df)

X_train = train_df.drop(columns=['target'])
X_validation = validation_df.drop(columns=['target'])
X_test = test_df.drop(columns=['target'])

feature_columns = X_train.columns

X_validation = X_validation[feature_columns]
X_test = X_test[feature_columns]

train_medians = X_train.median()

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(train_medians)
X_validation = X_validation.replace([np.inf, -np.inf], np.nan).fillna(train_medians)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(train_medians)

for name, X in [
    ('TRAIN', X_train),
    ('VALIDATION', X_validation),
    ('TEST', X_test),
]:
    probabilities = model.predict_proba(X)[:, 1]

    print()
    print('=' * 70)
    print(f'{name} PROBABILITIES')
    print('=' * 70)

    print(f'Minimum probability : {probabilities.min():.4f}')
    print(f'Maximum probability : {probabilities.max():.4f}')
    print(f'Mean probability    : {probabilities.mean():.4f}')
    print(f'Median probability  : {np.median(probabilities):.4f}')

    print()
    print('Probability ranges:')
    print(
        pd.cut(
            probabilities,
            bins=[0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
            include_lowest=True
        ).value_counts().sort_index()
    )

    print()
    print('Predictions at different thresholds:')

    for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        predictions = (probabilities >= threshold).astype(int)

        print(
            f'Threshold {threshold:.2f} | '
            f'Predicted UP: {predictions.sum():3d} | '
            f'Predicted DOWN: {(predictions == 0).sum():3d}'
        )

print()
print('=' * 70)
print('Probability check complete.')
print('=' * 70)
