from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "real_price_backtest"
    / "aapl_real_price_backtest_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "real_price_backtest"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load predictions
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print("Loaded rows:", len(df))
print("Columns:")
print(df.columns.tolist())


# --------------------------------------------------
# Detect required columns
# --------------------------------------------------
price_col = "aapl_adj_close"
actual_return_col = "actual_5day_return"

if price_col not in df.columns:
    raise ValueError(f"Missing column: {price_col}")

if actual_return_col not in df.columns:
    raise ValueError(f"Missing column: {actual_return_col}")


# --------------------------------------------------
# Build strategy returns
# --------------------------------------------------
if "signal" in df.columns:
    signal_col = "signal"
elif "prediction" in df.columns:
    signal_col = "prediction"
else:
    raise ValueError("Neither 'signal' nor 'prediction' column found.")

df["position"] = df[signal_col].astype(int)

# Backtest script ke calculated returns already available hain
if "net_strategy_return" in df.columns:
    df["net_strategy_return"] = pd.to_numeric(
        df["net_strategy_return"],
        errors="coerce"
    )

if "buy_hold_return" in df.columns:
    df["buy_hold_return"] = pd.to_numeric(
        df["buy_hold_return"],
        errors="coerce"
    )
else:
    df["buy_hold_return"] = df[actual_return_col]


# --------------------------------------------------
# Equity curves
# --------------------------------------------------
initial_capital = 100000

if "strategy_equity" in df.columns:
    df["strategy_equity"] = pd.to_numeric(
        df["strategy_equity"],
        errors="coerce"
    )
else:
    df["strategy_equity"] = (
        initial_capital
        * (1 + df["net_strategy_return"]).cumprod()
    )

if "buy_hold_equity" in df.columns:
    df["buy_hold_equity"] = pd.to_numeric(
        df["buy_hold_equity"],
        errors="coerce"
    )
else:
    df["buy_hold_equity"] = (
        initial_capital
        * (1 + df["buy_hold_return"]).cumprod()
    )

# --------------------------------------------------
# Drawdown
# --------------------------------------------------
if "strategy_drawdown" not in df.columns:
    strategy_peak = df["strategy_equity"].cummax()
    df["strategy_drawdown"] = (
        df["strategy_equity"] / strategy_peak - 1
    )

if "buy_hold_drawdown" not in df.columns:
    buy_hold_peak = df["buy_hold_equity"].cummax()
    df["buy_hold_drawdown"] = (
        df["buy_hold_equity"] / buy_hold_peak - 1
    )

# --------------------------------------------------
# Plot 1: Equity curve

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 1. Dark Terminal Style Configuration
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300)

fig.patch.set_facecolor("#0D1117")  
ax.set_facecolor("#161B22")  

COLOR_STRATEGY = "#00F2FE" 
COLOR_BENCHMARK = "#8B949E"  

(line_strat,) = ax.plot(
    df["date"],
    df["strategy_equity"],
    label="Model Strategy",
    color=COLOR_STRATEGY,
    linewidth=2.2,
    zorder=4,)

(line_bh,) = ax.plot(
    df["date"],
    df["buy_hold_equity"],
    label="Buy & Hold (AAPL)",
    color=COLOR_BENCHMARK,
    linewidth=1.8,
    linestyle="--",
    alpha=0.85,
    zorder=3,)
ax.fill_between(
    df["date"],
    df["strategy_equity"],
    df["strategy_equity"].min(),
    color=COLOR_STRATEGY,
    alpha=0.08,
    zorder=2,)

ax.set_yscale("log")
ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda y, _: f"₹{y:,.0f}" if y >= 1 else f"₹{y:.2f}"))

ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

ax.grid(True, which="major", color="#30363D", linestyle="--", linewidth=0.7, alpha=0.6)
ax.grid(True, which="minor", color="#21262D", linestyle=":", linewidth=0.5, alpha=0.4)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#30363D")
    ax.spines[spine].set_linewidth(1.2)

ax.set_title(
    "AAPL Real-Price Backtest: Equity Curve",
    fontsize=16,
    fontweight="bold",
    color="#F0F6FC",
    pad=20,
    loc="left",)
ax.set_xlabel("Date", fontsize=11, color="#8B949E", labelpad=12, fontweight="medium")
ax.set_ylabel(
    "Portfolio Value (Log Scale)",
    fontsize=11,
    color="#8B949E",
    labelpad=12,
    fontweight="medium",)

ax.tick_params(axis="both", colors="#8B949E", labelsize=10, length=4)

legend = ax.legend(
    loc="upper left",
    frameon=True,
    facecolor="#0D1117",
    edgecolor="#30363D",
    fontsize=10.5,
    labelcolor="#F0F6FC",
    borderpad=0.8,)
legend.get_frame().set_linewidth(1.0)

plt.tight_layout()

equity_path = OUTPUT_DIR / "equity_curve.png"
plt.savefig(
    equity_path,
    dpi=300,
    facecolor=fig.get_facecolor(),
    edgecolor="none",
    bbox_inches="tight",)
plt.close()

# --------------------------------------------------
#Plot 2: Drawdown curve
# --------------------------------------------------
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(15, 6.5), dpi=300)
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#0D1117")

strategy_color = "#00F5D4"
benchmark_color = "#FF5376"

strat_dd = df["strategy_drawdown"] * 100
bench_dd = df["buy_hold_drawdown"] * 100

ax.plot(
    df["date"],
    bench_dd,
    label="Buy & Hold Drawdown",
    color=benchmark_color,
    linewidth=1.4,
    linestyle="--",
    alpha=0.75,
    zorder=2,)

ax.fill_between(
    df["date"],
    bench_dd,
    0,
    color=benchmark_color,
    alpha=0.06,
    zorder=1,)

ax.plot(
    df["date"],
    strat_dd,
    label="Model Strategy Drawdown",
    color=strategy_color,
    linewidth=2.2,
    zorder=4,)

ax.fill_between(
    df["date"],
    strat_dd,
    0,
    color=strategy_color,
    alpha=0.15,
    zorder=3,)

ax.axhline(0, color="#484F58", linewidth=1.2, linestyle="-", zorder=5)

fig.text(
    0.08,
    0.95,
    "AAPL Real-Price Backtest: Underwater Drawdown Profile",
    fontsize=16,
    fontweight="bold",
    color="#FFFFFF",
    ha="left",)

fig.text(
    0.08,
    0.91,
    "Peak-to-trough capital decline (%) | Capital preservation efficiency",
    fontsize=10.5,
    color="#8B949E",
    ha="left",)

ax.set_ylim(min(bench_dd.min(), strat_dd.min()) * 1.15, 2)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

ax.grid(True, which="major", color="#30363D", linestyle="-", linewidth=0.7, alpha=0.5)
ax.grid(True, which="minor", color="#21262D", linestyle=":", linewidth=0.5, alpha=0.3)

for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

ax.tick_params(axis="both", which="both", colors="#8B949E", labelsize=10, length=0)
ax.set_ylabel("Drawdown from Peak", color="#8B949E", fontsize=11, labelpad=12)

legend = ax.legend(
    loc="lower left",
    frameon=True,
    facecolor="#161B22",
    edgecolor="#30363D",
    fontsize=10,
    labelcolor="#E6EDF3",
    borderpad=0.8,)
legend.get_frame().set_linewidth(0.8)

plt.subplots_adjust(top=0.86, bottom=0.12, left=0.08, right=0.95)

drawdown_path = OUTPUT_DIR / "drawdown_curve.png"
plt.savefig(drawdown_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()

# --------------------------------------------------
#Plot 3: Model signals and price
# --------------------------------------------------
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(15, 7), dpi=300)
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#0D1117")

price_color = "#58A6FF"       
signal_color = "#2EA043"      
signal_glow = "#3FB950"      

ax.plot(
    df["date"],
    df[price_col],
    label="AAPL Close Price",
    color=price_color,
    linewidth=1.8,
    alpha=0.9,
    zorder=2,)

ax.fill_between(
    df["date"],
    df[price_col],
    df[price_col].min() * 0.95,
    color=price_color,
    alpha=0.05,
    zorder=1,)

buy_mask = df["position"] == 1
buy_dates = df.loc[buy_mask, "date"]
buy_prices = df.loc[buy_mask, price_col]

ax.scatter(
    buy_dates,
    buy_prices,
    marker="^",
    s=70,
    color=signal_glow,
    alpha=0.35,
    edgecolors="none",
    zorder=4,)

ax.scatter(
    buy_dates,
    buy_prices,
    marker="^",
    s=35,
    color=signal_color,
    edgecolors="#FFFFFF",
    linewidths=0.6,
    label=f"BUY Signal (n={buy_mask.sum()})",
    zorder=5,)

fig.text(
    0.08,
    0.95,
    "AAPL Price Action: Quantitative Entry Signals",
    fontsize=16,
    fontweight="bold",
    color="#FFFFFF",
    ha="left",)

fig.text(
    0.08,
    0.91,
    f"Algorithmic execution trace | Total signals generated: {buy_mask.sum()}",
    fontsize=10.5,
    color="#8B949E",
    ha="left",)

ax.grid(True, which="major", color="#30363D", linestyle="-", linewidth=0.7, alpha=0.5)
ax.grid(True, which="minor", color="#21262D", linestyle=":", linewidth=0.5, alpha=0.3)

for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

ax.tick_params(axis="both", which="both", colors="#8B949E", labelsize=10, length=0)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("₹{x:,.0f}"))
ax.set_ylabel("Price Level", color="#8B949E", fontsize=11, labelpad=12)

price_min, price_max = df[price_col].min(), df[price_col].max()
ax.set_ylim(price_min * 0.96, price_max * 1.04)

legend = ax.legend(
    loc="upper left",
    frameon=True,
    facecolor="#161B22",
    edgecolor="#30363D",
    fontsize=10,
    labelcolor="#E6EDF3",
    borderpad=0.8,
)
legend.get_frame().set_linewidth(0.8)

plt.subplots_adjust(top=0.86, bottom=0.12, left=0.08, right=0.95)

signals_path = OUTPUT_DIR / "buy_signals.png"
plt.savefig(signals_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Year-wise performance
# --------------------------------------------------
df["year"] = df["date"].dt.year

yearly = (
    df.groupby("year")
    .agg(
        strategy_return=("net_strategy_return", lambda x: (1 + x).prod() - 1),
        buy_hold_return=("buy_hold_return", lambda x: (1 + x).prod() - 1),
        buy_signals=("position", "sum"),
        observations=("position", "count"),
    )
    .reset_index()
)

yearly["strategy_return_pct"] = yearly["strategy_return"] * 100
yearly["buy_hold_return_pct"] = yearly["buy_hold_return"] * 100

yearly_path = OUTPUT_DIR / "yearly_performance.csv"
yearly.to_csv(yearly_path, index=False)

print("\nYEAR-WISE PERFORMANCE")
print(yearly.to_string(index=False))

print("\nFILES SAVED")
print(equity_path)
print(drawdown_path)
print(signals_path)
print(yearly_path)