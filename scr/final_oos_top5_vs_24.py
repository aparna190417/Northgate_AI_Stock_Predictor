import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


# ============================================================
# NORTHGATE AI — FINAL OUT-OF-SAMPLE TOP-5 VS 24-FEATURE TEST
# ============================================================

print("=" * 80)
print("NORTHGATE AI — FINAL OUT-OF-SAMPLE TOP-5 VS 24-FEATURE TEST")
print("=" * 80)

print("""
Purpose:

1. Train TOP-5 and 24-feature models using TRAIN + VALIDATION data.
2. Evaluate both exactly once on the untouched FINAL TEST.
3. Compare ROC-AUC and ranking quality.
4. Compare classification performance.
5. Compare locked-threshold trading performance.
6. Do NOT use FINAL TEST to select features or thresholds.

IMPORTANT:
- Final test is strictly out-of-sample.
- Threshold remains locked at 0.65.
- Holding period remains 5 trading days.
- Transaction cost remains 0.10%.
- No threshold optimization is performed.
""")


# ============================================================
# CONFIG
# ============================================================

TRAIN_FILE = "data/processed/AAPL_clean_train.parquet"
VAL_FILE = "data/processed/AAPL_clean_validation.parquet"
TEST_FILE = "data/processed/AAPL_clean_test.parquet"

OUTPUT_DIR = "results/final_oos_top5_vs_24"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42

THRESHOLD = 0.65
HOLDING_DAYS = 5
TRANSACTION_COST = 0.001
INITIAL_CAPITAL = 100000.0


# ============================================================
# FEATURES
# ============================================================

TOP5 = [
    "AAPL_price_to_ma20",
    "AAPL_volatility_20d",
    "AAPL_volatility_5d",
    "AAPL_return_5d",
    "AAPL_return_20d",
]

FEATURES_24 = [
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


# ============================================================
# MODEL
# ============================================================

def make_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_SEED,
        n_jobs=1,
        class_weight=None,
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("1. LOADING DATA")
print("=" * 80)

train = pd.read_parquet(TRAIN_FILE)
val = pd.read_parquet(VAL_FILE)
test = pd.read_parquet(TEST_FILE)

for df in [train, val, test]:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

    df.index = pd.to_datetime(df.index)


print(
    f"TRAIN      : {len(train)} rows | "
    f"{train.index.min().date()} → {train.index.max().date()}"
)

print(
    f"VALIDATION : {len(val)} rows | "
    f"{val.index.min().date()} → {val.index.max().date()}"
)

print(
    f"TEST       : {len(test)} rows | "
    f"{test.index.min().date()} → {test.index.max().date()}"
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("2. COLUMN VALIDATION")
print("=" * 80)

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", val),
    ("TEST", test),
]:
    missing = [
        c for c in FEATURES_24 + ["target"]
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} missing columns:\n" +
            "\n".join(missing)
        )

    print(f"{name}: all required columns found.")


# ============================================================
# TRAIN + VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("3. TRAINING DATA")
print("=" * 80)

development = pd.concat([train, val]).sort_index()

print(
    f"Development rows: {len(development)}"
)

print(
    f"Development period: "
    f"{development.index.min().date()} → "
    f"{development.index.max().date()}"
)

print(
    f"Final test period: "
    f"{test.index.min().date()} → "
    f"{test.index.max().date()}"
)


# ============================================================
# PREPARE DATA
# ============================================================

X_dev_5 = development[TOP5].copy()
X_test_5 = test[TOP5].copy()

X_dev_24 = development[FEATURES_24].copy()
X_test_24 = test[FEATURES_24].copy()

y_dev = development["target"].astype(int)
y_test = test["target"].astype(int)


# Remove invalid rows from development only
valid_dev_5 = X_dev_5.notna().all(axis=1)
valid_dev_24 = X_dev_24.notna().all(axis=1)

valid_dev = valid_dev_5 & valid_dev_24 & y_dev.notna()

X_dev_5 = X_dev_5.loc[valid_dev]
X_dev_24 = X_dev_24.loc[valid_dev]
y_dev = y_dev.loc[valid_dev]

# Test must remain untouched except rows that cannot be evaluated
valid_test_5 = X_test_5.notna().all(axis=1)
valid_test_24 = X_test_24.notna().all(axis=1)

valid_test = (
    valid_test_5 &
    valid_test_24 &
    y_test.notna()
)

X_test_5 = X_test_5.loc[valid_test]
X_test_24 = X_test_24.loc[valid_test]
y_test = y_test.loc[valid_test]

print(f"Development observations: {len(y_dev)}")
print(f"Final test observations:  {len(y_test)}")


# ============================================================
# TRAIN TOP-5
# ============================================================

print("\n" + "=" * 80)
print("4. TRAINING TOP-5 MODEL")
print("=" * 80)

print("Features:")
for f in TOP5:
    print(" -", f)

model_5 = make_model()
model_5.fit(X_dev_5, y_dev)

prob_5 = model_5.predict_proba(X_test_5)[:, 1]


# ============================================================
# TRAIN 24-FEATURE
# ============================================================

print("\n" + "=" * 80)
print("5. TRAINING 24-FEATURE MODEL")
print("=" * 80)

print("Features:")
for f in FEATURES_24:
    print(" -", f)

model_24 = make_model()
model_24.fit(X_dev_24, y_dev)

prob_24 = model_24.predict_proba(X_test_24)[:, 1]


# ============================================================
# ROC-AUC
# ============================================================

print("\n" + "=" * 80)
print("6. FINAL TEST ROC-AUC")
print("=" * 80)

auc_5 = roc_auc_score(y_test, prob_5)
auc_24 = roc_auc_score(y_test, prob_24)

print(f"TOP-5 ROC-AUC       : {auc_5:.6f}")
print(f"24-feature ROC-AUC  : {auc_24:.6f}")
print(f"AUC difference      : {auc_5 - auc_24:+.6f}")


# ============================================================
# SPEARMAN
# ============================================================

rho_5, _ = spearmanr(prob_5, y_test)
rho_24, _ = spearmanr(prob_24, y_test)

print("\n" + "=" * 80)
print("7. RANKING QUALITY")
print("=" * 80)

print(f"TOP-5 Spearman      : {rho_5:+.6f}")
print(f"24-feature Spearman : {rho_24:+.6f}")
print(f"Difference           : {rho_5 - rho_24:+.6f}")


# ============================================================
# CLASSIFICATION
# ============================================================

def classification_stats(y, p):
    pred = (p >= 0.50).astype(int)

    tp = np.sum((pred == 1) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))

    accuracy = (tp + tn) / len(y)

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else np.nan
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else np.nan
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else np.nan
    )

    return accuracy, precision, recall, f1


acc5, prec5, rec5, f15 = classification_stats(y_test.values, prob_5)
acc24, prec24, rec24, f124 = classification_stats(y_test.values, prob_24)

print("\n" + "=" * 80)
print("8. CLASSIFICATION @ 0.50")
print("=" * 80)

print(
    f"TOP-5     : "
    f"Accuracy={acc5:.4f} | "
    f"Precision={prec5:.4f} | "
    f"Recall={rec5:.4f} | "
    f"F1={f15:.4f}"
)

print(
    f"24-feature: "
    f"Accuracy={acc24:.4f} | "
    f"Precision={prec24:.4f} | "
    f"Recall={rec24:.4f} | "
    f"F1={f124:.4f}"
)


# ============================================================
# SIGNAL ANALYSIS
# ============================================================

signal_5 = prob_5 >= THRESHOLD
signal_24 = prob_24 >= THRESHOLD

print("\n" + "=" * 80)
print("9. LOCKED THRESHOLD")
print("=" * 80)

print(f"Locked threshold : {THRESHOLD}")

print(f"TOP-5 signals    : {signal_5.sum()}")
print(f"24-feature signals: {signal_24.sum()}")


# ============================================================
# FUTURE 5-DAY RETURN
# ============================================================

# We calculate future return from the actual test-period close.
# This is evaluation only.

price_column_candidates = [
    "AAPL_adj_close",
    "adj_close",
    "Adj Close",
    "AAPL_close",
    "close",
    "Close",
]

price_col = None

for c in price_column_candidates:
    if c in test.columns:
        price_col = c
        break

if price_col is None:
    print("\nWARNING:")
    print("No price column found.")
    print("Trading analysis will be skipped.")
    future_return = None
else:
    prices = test.loc[y_test.index, price_col].astype(float)

    future_return = (
        prices.shift(-HOLDING_DAYS) / prices - 1
    )

    future_return = future_return.values


# ============================================================
# TRADING ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("10. LOCKED TRADING ANALYSIS")
print("=" * 80)

if future_return is None:

    trading_results = []

else:

    valid_trade = np.isfinite(future_return)

    def evaluate_trading(signal):

        returns = future_return[valid_trade & signal]

        if len(returns) == 0:
            return {
                "signals": 0,
                "mean_return": np.nan,
                "median_return": np.nan,
                "win_rate": np.nan,
                "final_equity": INITIAL_CAPITAL,
                "total_return": 0.0,
            }

        net_returns = returns - TRANSACTION_COST

        equity = INITIAL_CAPITAL

        for r in net_returns:
            equity *= (1 + r)

        return {
            "signals": len(returns),
            "mean_return": np.mean(net_returns),
            "median_return": np.median(net_returns),
            "win_rate": np.mean(net_returns > 0),
            "final_equity": equity,
            "total_return": equity / INITIAL_CAPITAL - 1,
        }

    result_5 = evaluate_trading(signal_5)
    result_24 = evaluate_trading(signal_24)

    trading_results = []

    for model_name, result in [
        ("TOP-5", result_5),
        ("24-feature", result_24),
    ]:
        row = {
            "model": model_name,
            **result
        }
        trading_results.append(row)

        print(f"\n{model_name}")
        print(f"Signals          : {result['signals']}")
        print(f"Mean net return  : {result['mean_return']:.6f}")
        print(f"Median net return: {result['median_return']:.6f}")
        print(f"Win rate         : {result['win_rate']:.6f}")
        print(f"Final equity     : ₹{result['final_equity']:,.2f}")
        print(f"Total return     : {result['total_return']:.6%}")


# ============================================================
# SAVE ROW LEVEL RESULTS
# ============================================================

print("\n" + "=" * 80)
print("11. SAVING RESULTS")
print("=" * 80)

comparison = pd.DataFrame({
    "date": y_test.index,
    "actual_target": y_test.values,
    "top5_probability": prob_5,
    "model24_probability": prob_24,
    "top5_signal_0_65": signal_5,
    "model24_signal_0_65": signal_24,
})

if future_return is not None:
    comparison["future_5d_return"] = future_return

comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_test_row_level.csv"
    ),
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "test_start": y_test.index.min(),
    "test_end": y_test.index.max(),
    "test_rows": len(y_test),

    "top5_auc": auc_5,
    "model24_auc": auc_24,
    "auc_difference": auc_5 - auc_24,

    "top5_spearman": rho_5,
    "model24_spearman": rho_24,
    "spearman_difference": rho_5 - rho_24,

    "top5_accuracy": acc5,
    "model24_accuracy": acc24,

    "top5_precision": prec5,
    "model24_precision": prec24,

    "top5_recall": rec5,
    "model24_recall": rec24,

    "top5_f1": f15,
    "model24_f1": f124,

    "top5_signals": int(signal_5.sum()),
    "model24_signals": int(signal_24.sum()),
}

if future_return is not None:
    summary.update({
        "top5_final_equity": result_5["final_equity"],
        "model24_final_equity": result_24["final_equity"],
        "top5_total_return": result_5["total_return"],
        "model24_total_return": result_24["total_return"],
        "top5_mean_net_return": result_5["mean_return"],
        "model24_mean_net_return": result_24["mean_return"],
        "top5_win_rate": result_5["win_rate"],
        "model24_win_rate": result_24["win_rate"],
    })

summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_test_summary.csv"
    ),
    index=False
)

if trading_results:
    pd.DataFrame(trading_results).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "final_test_trading_comparison.csv"
        ),
        index=False
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("12. FINAL OUT-OF-SAMPLE RESULT")
print("=" * 80)

print(f"""
FINAL TEST PERIOD
-----------------
{y_test.index.min().date()} → {y_test.index.max().date()}

TOP-5 ROC-AUC
-------------
{auc_5:.6f}

24-FEATURE ROC-AUC
------------------
{auc_24:.6f}

AUC DIFFERENCE
--------------
{auc_5 - auc_24:+.6f}

TOP-5 SPEARMAN
--------------
{rho_5:+.6f}

24-FEATURE SPEARMAN
-------------------
{rho_24:+.6f}

This is an OUT-OF-SAMPLE evaluation.

The final test was NOT used to:
- select features
- optimize threshold
- tune the model
- select production configuration
""")

print("\nSaved:")
print(
    os.path.join(
        OUTPUT_DIR,
        "final_test_summary.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "final_test_row_level.csv"
    )
)

print("\n" + "=" * 80)
print("NORTHGATE AI — FINAL OOS TEST COMPLETED")
print("=" * 80)