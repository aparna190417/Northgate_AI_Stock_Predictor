from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# NORTHGATE AI — FINAL PROJECT AUDIT
# ============================================================
#
# Purpose:
# 1. Collect the completed TOP-5 vs 24-feature experiments.
# 2. Report statistical and trading evidence together.
# 3. Do NOT retrain models.
# 4. Do NOT optimize thresholds.
# 5. Do NOT select a production model automatically.
#
# This is a reporting/audit script only.
# ============================================================


BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"


def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_csv(path, required=None):
    if not path.exists():
        print(f"WARNING: Missing file:\n{path}")
        return None

    df = pd.read_csv(path)

    if required:
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"WARNING: {path.name} missing columns: {missing}")

    return df


# ============================================================
# 1. PATHS
# ============================================================

oos_summary_path = (
    RESULTS
    / "final_oos_top5_vs_24"
    / "final_test_summary.csv"
)

oos_stat_path = (
    RESULTS
    / "final_oos_statistical_test"
    / "final_oos_statistical_summary.csv"
)

trading_path = (
    RESULTS
    / "final_oos_trading_comparison"
    / "final_oos_overall_trading_comparison.csv"
)

yearly_trading_path = (
    RESULTS
    / "final_oos_trading_comparison"
    / "final_oos_yearly_trading_comparison.csv"
)

feature_stat_path = (
    RESULTS
    / "feature_set_statistical_significance_fast"
    / "statistical_significance_summary.csv"
)

trading_stat_path = (
    RESULTS
    / "top5_trading_statistical_test"
    / "trading_statistical_summary.csv"
)


# ============================================================
# 2. LOAD RESULTS
# ============================================================

header("1. LOADING COMPLETED RESULTS")

files = {
    "Final OOS model summary": oos_summary_path,
    "Final OOS statistical test": oos_stat_path,
    "Final OOS trading comparison": trading_path,
    "Final OOS yearly trading": yearly_trading_path,
    "Walk-forward statistical test": feature_stat_path,
    "Trading statistical test": trading_stat_path,
}

for name, path in files.items():
    status = "FOUND" if path.exists() else "MISSING"
    print(f"{status:8} | {name}")
    print(f"         {path}")


# ============================================================
# 3. FINAL OOS MODEL RESULTS
# ============================================================

header("2. FINAL OOS MODEL PERFORMANCE")

oos = load_csv(oos_summary_path)

if oos is not None:
    print(oos.to_string(index=False))

    # Try to extract values flexibly.
    print("\nRaw columns:")
    print(oos.columns.tolist())


# ============================================================
# 4. FINAL OOS STATISTICAL RESULTS
# ============================================================

header("3. FINAL OOS STATISTICAL TEST")

stat = load_csv(oos_stat_path)

if stat is not None:
    print(stat.to_string(index=False))

    print("\nRaw columns:")
    print(stat.columns.tolist())


# ============================================================
# 5. FINAL OOS TRADING RESULTS
# ============================================================

header("4. FINAL OOS TRADING PERFORMANCE")

trading = load_csv(trading_path)

if trading is not None:
    print(trading.to_string(index=False))

    print("\nRaw columns:")
    print(trading.columns.tolist())


# ============================================================
# 6. YEARLY TRADING PERFORMANCE
# ============================================================

header("5. YEARLY FINAL OOS TRADING PERFORMANCE")

yearly = load_csv(yearly_trading_path)

if yearly is not None:
    print(yearly.to_string(index=False))


# ============================================================
# 7. PREVIOUS WALK-FORWARD STATISTICAL TEST
# ============================================================

header("6. PREVIOUS WALK-FORWARD STATISTICAL TEST")

feature_stat = load_csv(feature_stat_path)

if feature_stat is not None:
    print(feature_stat.to_string(index=False))


# ============================================================
# 8. PREVIOUS TRADING STATISTICAL TEST
# ============================================================

header("7. PREVIOUS TRADING STATISTICAL TEST")

trading_stat = load_csv(trading_stat_path)

if trading_stat is not None:
    print(trading_stat.to_string(index=False))


# ============================================================
# 9. FINAL AUDIT INTERPRETATION
# ============================================================

header("8. FINAL AUDIT INTERPRETATION")

print("""
MODEL EVIDENCE
--------------

The completed experiments show:

1. TOP-5 produced higher FINAL OOS ROC-AUC than the 24-feature model.
2. The FINAL OOS AUC confidence interval included zero.
3. The FINAL OOS paired permutation test did not establish
   a statistically significant AUC difference.
4. TOP-5 generated substantially more locked-threshold signals.
5. The FINAL OOS locked trading comparison produced fewer trades
   for the 24-feature model but higher realized trading metrics
   in this particular untouched test period.


IMPORTANT LIMITATIONS
---------------------

The final test contains only 411 observations.

The 24-feature model generated only 8 completed trades
under the locked threshold.

The TOP-5 model generated 30 completed trades.

Therefore trading statistics such as Sharpe ratio and win rate
should be interpreted together with trade count and the limited
sample size.


FEATURE-SET CONCLUSION
----------------------

The experiments do NOT establish that TOP-5 is universally superior
to the 24-feature model.

TOP-5 shows an AUC advantage in the final OOS sample, but the
statistical uncertainty is substantial.

The locked final OOS trading comparison favors the 24-feature model
for the observed test period.


PRODUCTION DECISION
-------------------

This audit does not automatically select either model.

A production decision should remain a documented engineering/
research decision rather than being inferred from a single metric.


THRESHOLD
---------

The threshold remained locked at 0.65.

No threshold optimization was performed by this audit.


DATA LEAKAGE / OOS DISCIPLINE
-----------------------------

The final OOS evaluation was performed after development using
TRAIN + VALIDATION data.

The FINAL TEST was not used by these scripts to optimize the
threshold.

No retraining is performed by this audit script.
""")


# ============================================================
# 10. CREATE FINAL AUDIT SUMMARY
# ============================================================

header("9. CREATING FINAL AUDIT SUMMARY")

summary_rows = []

summary_rows.append({
    "area": "Final OOS AUC",
    "TOP5": 0.528093,
    "Model24": 0.504243,
    "difference_TOP5_minus_24": 0.023851,
    "status": "TOP-5 higher observed AUC"
})

summary_rows.append({
    "area": "Final OOS Spearman",
    "TOP5": 0.048572,
    "Model24": 0.007336,
    "difference_TOP5_minus_24": 0.041236,
    "status": "TOP-5 higher observed ranking correlation"
})

summary_rows.append({
    "area": "Final OOS AUC bootstrap CI lower",
    "TOP5": np.nan,
    "Model24": np.nan,
    "difference_TOP5_minus_24": -0.041559,
    "status": "CI includes zero"
})

summary_rows.append({
    "area": "Final OOS AUC bootstrap CI upper",
    "TOP5": np.nan,
    "Model24": np.nan,
    "difference_TOP5_minus_24": 0.089549,
    "status": "CI includes zero"
})

summary_rows.append({
    "area": "Final OOS permutation p-value",
    "TOP5": np.nan,
    "Model24": np.nan,
    "difference_TOP5_minus_24": 0.474753,
    "status": "Not statistically conclusive"
})

summary_rows.append({
    "area": "Final OOS trading return",
    "TOP5": 0.199832,
    "Model24": 0.295234,
    "difference_TOP5_minus_24": -0.095402,
    "status": "24-feature higher observed return"
})

summary_rows.append({
    "area": "Final OOS CAGR",
    "TOP5": 0.119818,
    "Model24": 0.182030,
    "difference_TOP5_minus_24": -0.062212,
    "status": "24-feature higher observed CAGR"
})

summary_rows.append({
    "area": "Final OOS max drawdown",
    "TOP5": -0.118214,
    "Model24": -0.007410,
    "difference_TOP5_minus_24": -0.110804,
    "status": "24-feature had smaller observed drawdown"
})

summary_rows.append({
    "area": "Final OOS Sharpe",
    "TOP5": 0.786886,
    "Model24": 3.167596,
    "difference_TOP5_minus_24": -2.380710,
    "status": "24-feature higher observed Sharpe"
})

summary_rows.append({
    "area": "Final OOS win rate",
    "TOP5": 0.566667,
    "Model24": 0.875000,
    "difference_TOP5_minus_24": -0.308333,
    "status": "24-feature higher observed win rate"
})

summary_rows.append({
    "area": "Final OOS completed trades",
    "TOP5": 30,
    "Model24": 8,
    "difference_TOP5_minus_24": 22,
    "status": "TOP-5 generated more trades"
})


final_summary = pd.DataFrame(summary_rows)

output_dir = RESULTS / "final_project_audit"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "final_project_audit_summary.csv"

final_summary.to_csv(output_file, index=False)


# ============================================================
# 11. FINAL REPORT
# ============================================================

header("10. FINAL NORTHGATE AI AUDIT")

print("""
FINAL OOS
---------

TOP-5 AUC       : 0.528093
24-feature AUC  : 0.504243
Difference      : +0.023851

95% AUC CI:
[-0.041559, +0.089549]

Permutation p-value:
0.474753


FINAL OOS TRADING
-----------------

TOP-5:
Trades          : 30
Final equity    : ₹119,983.22
Return          : 19.9832%
CAGR            : 11.9818%
Max drawdown    : -11.8214%
Sharpe          : 0.7869
Win rate        : 56.6667%


24-FEATURE:
Trades          : 8
Final equity    : ₹129,523.38
Return          : 29.5234%
CAGR            : 18.2030%
Max drawdown    : -0.7410%
Sharpe          : 3.1676
Win rate        : 87.5000%


FINAL INTERPRETATION
--------------------

TOP-5 has higher observed predictive AUC in the final OOS sample.

However, the AUC difference is statistically uncertain because
the bootstrap confidence interval includes zero and the paired
permutation test is not statistically conclusive.

Under the locked 0.65 trading rule, the observed FINAL OOS trading
results are higher for the 24-feature model.

Therefore this audit does NOT justify automatically replacing the
24-feature model with TOP-5.

No threshold was optimized.
No model was retrained.
No production configuration was automatically selected.
""")


print("\nSaved final audit:")
print(output_file)

print("\n" + "=" * 80)
print("NORTHGATE AI — FINAL PROJECT AUDIT COMPLETED")
print("=" * 80)