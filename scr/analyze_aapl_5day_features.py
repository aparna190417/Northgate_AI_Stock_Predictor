from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "AAPL_5day_normalized_train.parquet"
VAL_PATH = ROOT / "data" / "processed" / "AAPL_5day_normalized_validation.parquet"
TEST_PATH = ROOT / "data" / "processed" / "AAPL_5day_normalized_test.parquet"

RESULTS_DIR = ROOT / "results" / "feature_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------

train_df = pd.read_parquet(TRAIN_PATH)
val_df = pd.read_parquet(VAL_PATH)
test_df = pd.read_parquet(TEST_PATH)

target_col = "target"
drop_cols = ["date", target_col]

X_train = train_df.drop(columns=drop_cols, errors="ignore")
y_train = train_df[target_col]

X_val = val_df.drop(columns=drop_cols, errors="ignore")
y_val = val_df[target_col]

X_test = test_df.drop(columns=drop_cols, errors="ignore")
y_test = test_df[target_col]

print("=" * 75)
print("AAPL 5-DAY FEATURE IMPORTANCE ANALYSIS")
print("=" * 75)

print(f"Training shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Test shape: {X_test.shape}")


# --------------------------------------------------
# Random Forest
# --------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=8,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

rf_importance.to_csv(
    RESULTS_DIR / "random_forest_feature_importance.csv",
    index=False
)


# --------------------------------------------------
# Gradient Boosting
# --------------------------------------------------

gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=2,
    min_samples_leaf=10,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": gb_model.feature_importances_
}).sort_values("importance", ascending=False)

gb_importance.to_csv(
    RESULTS_DIR / "gradient_boosting_feature_importance.csv",
    index=False
)


# --------------------------------------------------
# Combined importance
# --------------------------------------------------

importance_df = rf_importance.rename(
    columns={"importance": "rf_importance"}
).merge(
    gb_importance.rename(
        columns={"importance": "gb_importance"}
    ),
    on="feature",
    how="outer"
)

importance_df["rf_importance"] = importance_df["rf_importance"].fillna(0)
importance_df["gb_importance"] = importance_df["gb_importance"].fillna(0)

importance_df["average_importance"] = (
    importance_df["rf_importance"] +
    importance_df["gb_importance"]
) / 2

importance_df = importance_df.sort_values(
    "average_importance",
    ascending=False
)

importance_df.to_csv(
    RESULTS_DIR / "combined_feature_importance.csv",
    index=False
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\nTop 20 features according to Random Forest:")
print(rf_importance.head(20).to_string(index=False))

print("\nTop 20 features according to Gradient Boosting:")
print(gb_importance.head(20).to_string(index=False))

print("\nTop 20 combined features:")
print(importance_df.head(20).to_string(index=False))


# --------------------------------------------------
# Plot top 20 features
# --------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(13, 8.5), dpi=300)
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#0D1117")

top_features = (
    importance_df.head(20)
    .sort_values("average_importance", ascending=True)
    .copy())

clean_names = [
    str(f)
    .replace("Features_", "")
    .replace("Adj Close_", "")
    .replace("_", " ")
    .title()
    for f in top_features["feature"]]

importances = top_features["average_importance"].values
n_bars = len(importances)

colors = plt.cm.winter(np.linspace(0.2, 0.95, n_bars))

bars = ax.barh(
    clean_names,
    importances,
    height=0.55,
    color=colors,
    edgecolor="#FFFFFF",
    linewidth=0.5,
    alpha=0.92,
    zorder=3,)

max_val = importances.max()
for bar, val in zip(bars, importances):
    width = bar.get_width()
    ax.annotate(
        f"{val:.4f}",
        xy=(width, bar.get_y() + bar.get_height() / 2),
        xytext=(8, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#00F5D4" if val == max_val else "#C9D1D9",
        zorder=4,)

mean_imp = importances.mean()
ax.axvline(
    mean_imp,
    color="#8B949E",
    linestyle="--",
    linewidth=1.1,
    alpha=0.65,
    label=f"Top 20 Mean ({mean_imp:.4f})",
    zorder=2,)

fig.text(
    0.06,
    0.96,
    "Feature Attribution: Predictive Signal Ranking",
    fontsize=16,
    fontweight="bold",
    color="#FFFFFF",
    ha="left",)
fig.text(
    0.06,
    0.925,
    "Top 20 quantitative drivers for AAPL 5-Day forward direction | Normalised importance weight",
    fontsize=10,
    color="#8B949E",
    ha="left",)

ax.grid(
    True,
    which="major",
    axis="x",
    color="#30363D",
    linestyle="-",
    linewidth=0.7,
    alpha=0.5,)
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

ax.tick_params(
    axis="both", which="both", colors="#8B949E", labelsize=9.5, length=0)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax.set_xlabel(
    "Average Relative Feature Importance",
    color="#8B949E",
    fontsize=10.5,
    labelpad=12,)

ax.set_xlim(0, max_val * 1.16)

legend = ax.legend(
    loc="lower right",
    frameon=True,
    facecolor="#161B22",
    edgecolor="#30363D",
    fontsize=9.5,
    labelcolor="#8B949E",
    borderpad=0.7,)
legend.get_frame().set_linewidth(0.8)

plt.subplots_adjust(top=0.88, bottom=0.10, left=0.22, right=0.95)

plot_path = RESULTS_DIR / "top_20_feature_importance.png"
plt.savefig(
    plot_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()

print(f"\nSaved feature importance files to: {RESULTS_DIR}")
print(f"Saved plot to: {plot_path}")
print("\nFEATURE ANALYSIS COMPLETED")