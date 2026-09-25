import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

print("=" * 80)
print("NORTHGATE AI — FINAL OOS STATISTICAL TEST")
print("=" * 80)

INPUT = "results/final_oos_top5_vs_24/final_test_row_level.csv"
OUT = "results/final_oos_statistical_test"

os.makedirs(OUT, exist_ok=True)

N_BOOTSTRAP = 10000
N_PERMUTATION = 10000
SEED = 42

rng = np.random.default_rng(SEED)

print("\n1. LOADING FINAL OOS PREDICTIONS")

df = pd.read_csv(INPUT)

required = [
    "actual_target",
    "top5_probability",
    "model24_probability",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        "Missing columns:\n" + "\n".join(missing)
    )

df = df.dropna(subset=required).reset_index(drop=True)

y = df["actual_target"].astype(int).values
p5 = df["top5_probability"].astype(float).values
p24 = df["model24_probability"].astype(float).values

print(f"Final OOS observations: {len(df)}")

print("\n2. OBSERVED PERFORMANCE")

auc5 = roc_auc_score(y, p5)
auc24 = roc_auc_score(y, p24)

rho5, _ = spearmanr(p5, y)
rho24, _ = spearmanr(p24, y)

observed_auc_diff = auc5 - auc24
observed_rho_diff = rho5 - rho24

print(f"TOP-5 AUC       : {auc5:.6f}")
print(f"24-feature AUC  : {auc24:.6f}")
print(f"AUC difference  : {observed_auc_diff:+.6f}")

print(f"\nTOP-5 Spearman      : {rho5:+.6f}")
print(f"24-feature Spearman : {rho24:+.6f}")
print(f"Difference           : {observed_rho_diff:+.6f}")


# ============================================================
# BOOTSTRAP AUC DIFFERENCE
# ============================================================

print("\n" + "=" * 80)
print("3. BOOTSTRAP AUC DIFFERENCE")
print("=" * 80)

bootstrap_diffs = []

n = len(y)

for i in range(N_BOOTSTRAP):

    idx = rng.integers(0, n, size=n)

    y_b = y[idx]

    # Need both classes for AUC
    if len(np.unique(y_b)) < 2:
        continue

    auc5_b = roc_auc_score(y_b, p5[idx])
    auc24_b = roc_auc_score(y_b, p24[idx])

    bootstrap_diffs.append(
        auc5_b - auc24_b
    )

bootstrap_diffs = np.array(bootstrap_diffs)

ci_low = np.percentile(bootstrap_diffs, 2.5)
ci_high = np.percentile(bootstrap_diffs, 97.5)

print(f"Iterations used : {len(bootstrap_diffs)}")
print(f"Observed diff   : {observed_auc_diff:+.6f}")
print(f"95% CI          : [{ci_low:+.6f}, {ci_high:+.6f}]")

if ci_low <= 0 <= ci_high:
    print("CI includes zero.")
else:
    print("CI excludes zero.")


# ============================================================
# PAIRED PERMUTATION TEST
# ============================================================

print("\n" + "=" * 80)
print("4. PAIRED PERMUTATION TEST")
print("=" * 80)

observed = observed_auc_diff

extreme = 0
valid_perm = 0

for i in range(N_PERMUTATION):

    swap = rng.random(n) < 0.5

    perm5 = np.where(
        swap,
        p24,
        p5
    )

    perm24 = np.where(
        swap,
        p5,
        p24
    )

    auc_a = roc_auc_score(y, perm5)
    auc_b = roc_auc_score(y, perm24)

    diff = auc_a - auc_b

    if abs(diff) >= abs(observed):
        extreme += 1

    valid_perm += 1

p_value = (extreme + 1) / (valid_perm + 1)

print(f"Iterations : {valid_perm}")
print(f"Observed difference : {observed:+.6f}")
print(f"Permutation p-value : {p_value:.6f}")


# ============================================================
# BOOTSTRAP SPEARMAN DIFFERENCE
# ============================================================

print("\n" + "=" * 80)
print("5. SPEARMAN BOOTSTRAP")
print("=" * 80)

rho_boot = []

for i in range(N_BOOTSTRAP):

    idx = rng.integers(0, n, size=n)

    if len(np.unique(y[idx])) < 2:
        continue

    r5, _ = spearmanr(
        p5[idx],
        y[idx]
    )

    r24, _ = spearmanr(
        p24[idx],
        y[idx]
    )

    rho_boot.append(r5 - r24)

rho_boot = np.array(rho_boot)

rho_low = np.percentile(rho_boot, 2.5)
rho_high = np.percentile(rho_boot, 97.5)

print(
    f"Observed Spearman difference : "
    f"{observed_rho_diff:+.6f}"
)

print(
    f"95% CI : "
    f"[{rho_low:+.6f}, {rho_high:+.6f}]"
)


# ============================================================
# SIGNAL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("6. LOCKED THRESHOLD SIGNAL ANALYSIS")
print("=" * 80)

threshold = 0.65

signal5 = p5 >= threshold
signal24 = p24 >= threshold

print(f"Locked threshold : {threshold}")
print(f"TOP-5 signals    : {signal5.sum()}")
print(f"24-feature signals: {signal24.sum()}")

both = np.sum(signal5 & signal24)
only5 = np.sum(signal5 & ~signal24)
only24 = np.sum(~signal5 & signal24)
neither = np.sum(~signal5 & ~signal24)

print("\nSignal overlap:")
print(f"Both       : {both}")
print(f"TOP-5 only : {only5}")
print(f"24 only    : {only24}")
print(f"Neither    : {neither}")


# ============================================================
# SAVE BOOTSTRAP
# ============================================================

pd.DataFrame({
    "auc_difference": bootstrap_diffs
}).to_csv(
    os.path.join(
        OUT,
        "bootstrap_auc_difference.csv"
    ),
    index=False
)

pd.DataFrame({
    "spearman_difference": rho_boot
}).to_csv(
    os.path.join(
        OUT,
        "bootstrap_spearman_difference.csv"
    ),
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame([{
    "test_rows": n,

    "top5_auc": auc5,
    "model24_auc": auc24,
    "auc_difference": observed_auc_diff,

    "auc_bootstrap_ci_low": ci_low,
    "auc_bootstrap_ci_high": ci_high,

    "permutation_p_value": p_value,

    "top5_spearman": rho5,
    "model24_spearman": rho24,
    "spearman_difference": observed_rho_diff,

    "spearman_bootstrap_ci_low": rho_low,
    "spearman_bootstrap_ci_high": rho_high,

    "threshold": threshold,
    "top5_signals": int(signal5.sum()),
    "model24_signals": int(signal24.sum()),

    "both_signals": int(both),
    "top5_only": int(only5),
    "model24_only": int(only24),
    "neither": int(neither),
}])

summary.to_csv(
    os.path.join(
        OUT,
        "final_oos_statistical_summary.csv"
    ),
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("7. FINAL STATISTICAL REPORT")
print("=" * 80)

print(f"""
FINAL OOS SAMPLE
----------------
Rows: {n}

ROC-AUC
-------
TOP-5       : {auc5:.6f}
24-feature  : {auc24:.6f}

Difference  : {observed_auc_diff:+.6f}

95% Bootstrap CI
----------------
[{ci_low:+.6f}, {ci_high:+.6f}]

Permutation p-value
-------------------
{p_value:.6f}

SPEARMAN
--------
TOP-5       : {rho5:+.6f}
24-feature  : {rho24:+.6f}

Difference  : {observed_rho_diff:+.6f}

Spearman 95% CI
---------------
[{rho_low:+.6f}, {rho_high:+.6f}]

LOCKED THRESHOLD
----------------
Threshold : {threshold}

TOP-5 signals       : {signal5.sum()}
24-feature signals  : {signal24.sum()}
""")

print("\nNo feature selection was performed using this test.")
print("No threshold was optimized.")
print("No production model was changed.")

print("\nOutput:")
print(
    os.path.join(
        OUT,
        "final_oos_statistical_summary.csv"
    )
)

print("=" * 80)
print("NORTHGATE AI — FINAL OOS STATISTICAL TEST COMPLETED")
print("=" * 80)