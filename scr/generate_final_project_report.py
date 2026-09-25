from pathlib import Path
import pandas as pd
import html
from datetime import datetime


# =============================================================================
# NORTHGATE AI — FINAL PROJECT REPORT GENERATOR
# =============================================================================
#
# Purpose:
#   Create one consolidated final research/audit report from already completed
#   Northgate AI experiments.
#
# IMPORTANT:
#   - No model training
#   - No threshold optimization
#   - No feature selection
#   - No modification of production configuration
#   - Uses existing CSV outputs only
#
# Output:
#   results/final_project_report/Northgate_AI_Final_Project_Report.html
#
# =============================================================================


# -----------------------------------------------------------------------------
# 1. PATHS
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS = PROJECT_ROOT / "results"

OUTPUT_DIR = RESULTS / "final_project_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_HTML = OUTPUT_DIR / "Northgate_AI_Final_Project_Report.html"


# -----------------------------------------------------------------------------
# 2. INPUT FILES
# -----------------------------------------------------------------------------

FILES = {
    "final_model":
        RESULTS / "final_oos_top5_vs_24" /
        "final_test_summary.csv",

    "final_stats":
        RESULTS / "final_oos_statistical_test" /
        "final_oos_statistical_summary.csv",

    "final_trading":
        RESULTS / "final_oos_trading_comparison" /
        "final_oos_overall_trading_comparison.csv",

    "final_yearly":
        RESULTS / "final_oos_trading_comparison" /
        "final_oos_yearly_trading_comparison.csv",

    "walkforward_stats":
        RESULTS / "feature_set_statistical_significance_fast" /
        "statistical_significance_summary.csv",

    "trading_stats":
        RESULTS / "top5_trading_statistical_test" /
        "trading_statistical_summary.csv",

    "audit":
        RESULTS / "final_project_audit" /
        "final_project_audit_summary.csv",
}


# -----------------------------------------------------------------------------
# 3. HELPERS
# -----------------------------------------------------------------------------

def print_header(text):
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


def check_files():
    print_header("CHECKING INPUT FILES")

    missing = []

    for name, path in FILES.items():
        if path.exists():
            print(f"FOUND    | {name}")
            print(f"         {path}")
        else:
            print(f"MISSING  | {name}")
            print(f"         {path}")
            missing.append((name, path))

    if missing:
        print()
        print("ERROR: Required result files are missing.")
        print()
        print("Missing files:")

        for name, path in missing:
            print(f" - {name}: {path}")

        raise FileNotFoundError(
            "One or more required Northgate AI result files are missing."
        )


def read_csv(name):
    path = FILES[name]

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(
            f"Could not read {path}\nError: {e}"
        )

    return df


def safe_value(df, column, default="N/A", row=0):
    if df is None:
        return default

    if column not in df.columns:
        return default

    if len(df) <= row:
        return default

    value = df.iloc[row][column]

    if pd.isna(value):
        return default

    return value


def fmt_float(value, decimals=4):
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def fmt_percent(value, decimals=2):
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except Exception:
        return str(value)


def fmt_rupees(value):
    try:
        return f"₹{float(value):,.2f}"
    except Exception:
        return str(value)


def html_escape(value):
    return html.escape(str(value))


def dataframe_to_html(df, max_rows=None):
    if df is None:
        return "<p>No data available.</p>"

    temp = df.copy()

    if max_rows is not None:
        temp = temp.head(max_rows)

    # Round numeric values for readability.
    for col in temp.columns:
        if pd.api.types.is_numeric_dtype(temp[col]):
            temp[col] = temp[col].round(6)

    return temp.to_html(
        index=False,
        border=0,
        classes="data-table",
        justify="center"
    )


# -----------------------------------------------------------------------------
# 4. LOAD RESULTS
# -----------------------------------------------------------------------------

print_header("LOADING COMPLETED NORTHGATE AI RESULTS")

check_files()

final_model = read_csv("final_model")
final_stats = read_csv("final_stats")
final_trading = read_csv("final_trading")
final_yearly = read_csv("final_yearly")
walkforward_stats = read_csv("walkforward_stats")
trading_stats = read_csv("trading_stats")
audit = read_csv("audit")

print()
print("All result files loaded successfully.")


# -----------------------------------------------------------------------------
# 5. EXTRACT FINAL MODEL METRICS
# -----------------------------------------------------------------------------

test_start = safe_value(final_model, "test_start")
test_end = safe_value(final_model, "test_end")
test_rows = safe_value(final_model, "test_rows")

top5_auc = safe_value(final_model, "top5_auc")
model24_auc = safe_value(final_model, "model24_auc")
auc_difference = safe_value(final_model, "auc_difference")

top5_spearman = safe_value(final_model, "top5_spearman")
model24_spearman = safe_value(final_model, "model24_spearman")
spearman_difference = safe_value(
    final_model,
    "spearman_difference"
)

top5_accuracy = safe_value(final_model, "top5_accuracy")
model24_accuracy = safe_value(final_model, "model24_accuracy")

top5_precision = safe_value(final_model, "top5_precision")
model24_precision = safe_value(final_model, "model24_precision")

top5_recall = safe_value(final_model, "top5_recall")
model24_recall = safe_value(final_model, "model24_recall")

top5_f1 = safe_value(final_model, "top5_f1")
model24_f1 = safe_value(final_model, "model24_f1")

top5_signals = safe_value(final_model, "top5_signals")
model24_signals = safe_value(final_model, "model24_signals")


# -----------------------------------------------------------------------------
# 6. EXTRACT STATISTICAL METRICS
# -----------------------------------------------------------------------------

auc_ci_low = safe_value(
    final_stats,
    "auc_bootstrap_ci_low"
)

auc_ci_high = safe_value(
    final_stats,
    "auc_bootstrap_ci_high"
)

permutation_p = safe_value(
    final_stats,
    "permutation_p_value"
)

spearman_ci_low = safe_value(
    final_stats,
    "spearman_bootstrap_ci_low"
)

spearman_ci_high = safe_value(
    final_stats,
    "spearman_bootstrap_ci_high"
)

threshold = safe_value(
    final_stats,
    "threshold"
)

both_signals = safe_value(
    final_stats,
    "both_signals"
)

top5_only = safe_value(
    final_stats,
    "top5_only"
)

model24_only = safe_value(
    final_stats,
    "model24_only"
)

neither = safe_value(
    final_stats,
    "neither"
)


# -----------------------------------------------------------------------------
# 7. EXTRACT FINAL TRADING METRICS
# -----------------------------------------------------------------------------

def get_trading_row(df, model_name):
    if "model" not in df.columns:
        return None

    rows = df[
        df["model"].astype(str).str.lower()
        == model_name.lower()
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


top5_trade = get_trading_row(
    final_trading,
    "TOP-5"
)

model24_trade = get_trading_row(
    final_trading,
    "24-feature"
)


def trade_metric(row, column, default="N/A"):
    if row is None:
        return default

    if column not in row.index:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return value


top5_trades = trade_metric(
    top5_trade,
    "trades"
)

model24_trades = trade_metric(
    model24_trade,
    "trades"
)

top5_final_equity = trade_metric(
    top5_trade,
    "final_equity"
)

model24_final_equity = trade_metric(
    model24_trade,
    "final_equity"
)

top5_total_return = trade_metric(
    top5_trade,
    "total_return"
)

model24_total_return = trade_metric(
    model24_trade,
    "total_return"
)

top5_cagr = trade_metric(
    top5_trade,
    "cagr"
)

model24_cagr = trade_metric(
    model24_trade,
    "cagr"
)

top5_drawdown = trade_metric(
    top5_trade,
    "max_drawdown"
)

model24_drawdown = trade_metric(
    model24_trade,
    "max_drawdown"
)

top5_sharpe = trade_metric(
    top5_trade,
    "sharpe"
)

model24_sharpe = trade_metric(
    model24_trade,
    "sharpe"
)

top5_win_rate = trade_metric(
    top5_trade,
    "win_rate"
)

model24_win_rate = trade_metric(
    model24_trade,
    "win_rate"
)

top5_avg_trade = trade_metric(
    top5_trade,
    "average_trade_return"
)

model24_avg_trade = trade_metric(
    model24_trade,
    "average_trade_return"
)


# -----------------------------------------------------------------------------
# 8. WALK-FORWARD METRICS
# -----------------------------------------------------------------------------

wf_auc = safe_value(
    walkforward_stats,
    "top5_auc"
)

wf_model24_auc = safe_value(
    walkforward_stats,
    "model24_auc"
)

wf_auc_difference = safe_value(
    walkforward_stats,
    "auc_difference"
)

wf_bootstrap_low = safe_value(
    walkforward_stats,
    "bootstrap_ci_low"
)

wf_bootstrap_high = safe_value(
    walkforward_stats,
    "bootstrap_ci_high"
)

wf_p_value = safe_value(
    walkforward_stats,
    "permutation_p_value"
)

wf_paired_rows = safe_value(
    walkforward_stats,
    "paired_rows"
)

wf_positive_years = safe_value(
    walkforward_stats,
    "positive_auc_difference_years"
)

wf_total_years = safe_value(
    walkforward_stats,
    "total_years"
)


# -----------------------------------------------------------------------------
# 9. PREVIOUS TRADING STATISTICS
# -----------------------------------------------------------------------------

previous_top5_trade_count = safe_value(
    trading_stats,
    "top5_trade_count"
)

previous_model24_trade_count = safe_value(
    trading_stats,
    "24_feature_trade_count"
)

previous_top5_mean_return = safe_value(
    trading_stats,
    "top5_mean_return"
)

previous_model24_mean_return = safe_value(
    trading_stats,
    "24_feature_mean_return"
)

previous_trade_difference = safe_value(
    trading_stats,
    "observed_mean_difference"
)

previous_trade_ci_low = safe_value(
    trading_stats,
    "bootstrap_ci_low"
)

previous_trade_ci_high = safe_value(
    trading_stats,
    "bootstrap_ci_high"
)

previous_p_value = safe_value(
    trading_stats,
    "paired_permutation_p_value"
)

previous_common_trades = safe_value(
    trading_stats,
    "paired_common_trades"
)


# -----------------------------------------------------------------------------
# 10. REPORT METADATA
# -----------------------------------------------------------------------------

generation_time = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)


# -----------------------------------------------------------------------------
# 11. HTML
# -----------------------------------------------------------------------------

html_report = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>Northgate AI — Final Project Audit Report</title>

<style>

body {{
    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background: #f4f6f8;

    color: #202124;

    margin: 0;

    padding: 0;

    line-height: 1.55;
}}

.container {{
    max-width: 1200px;

    margin: 0 auto;

    padding: 35px;
}}

.cover {{
    background: #ffffff;

    padding: 45px;

    border-radius: 14px;

    box-shadow:
        0 4px 20px rgba(0,0,0,0.08);

    margin-bottom: 30px;
}}

h1 {{
    margin-top: 0;

    font-size: 36px;
}}

h2 {{
    margin-top: 0;

    border-bottom: 2px solid #e5e7eb;

    padding-bottom: 10px;
}}

h3 {{
    margin-top: 28px;
}}

.subtitle {{
    font-size: 18px;

    color: #5f6368;
}}

.meta {{
    margin-top: 25px;

    padding: 15px;

    background: #f8f9fa;

    border-radius: 8px;
}}

.section {{
    background: #ffffff;

    padding: 30px;

    margin-bottom: 25px;

    border-radius: 14px;

    box-shadow:
        0 2px 10px rgba(0,0,0,0.05);
}}

.grid {{
    display: grid;

    grid-template-columns:
        repeat(auto-fit, minmax(220px, 1fr));

    gap: 15px;

    margin: 20px 0;
}}

.card {{
    background: #f8f9fa;

    border-radius: 10px;

    padding: 20px;
}}

.card-title {{
    font-size: 13px;

    color: #6b7280;

    text-transform: uppercase;

    letter-spacing: 0.05em;
}}

.card-value {{
    font-size: 25px;

    font-weight: bold;

    margin-top: 8px;
}}

table {{
    width: 100%;

    border-collapse: collapse;

    margin-top: 20px;

    font-size: 14px;
}}

th {{
    background: #f1f3f4;

    font-weight: bold;
}}

th, td {{
    padding: 10px;

    border: 1px solid #e1e4e8;

    text-align: center;
}}

tr:nth-child(even) {{
    background: #fafafa;
}}

.note {{
    background: #fff8e1;

    border-left: 5px solid #f59e0b;

    padding: 15px 20px;

    margin: 20px 0;
}}

.info {{
    background: #eef6ff;

    border-left: 5px solid #4285f4;

    padding: 15px 20px;

    margin: 20px 0;
}}

.conclusion {{
    background: #f5f5f5;

    border-left: 5px solid #5f6368;

    padding: 20px;

    font-size: 17px;
}}

ul {{
    padding-left: 25px;
}}

code {{
    background: #f1f3f4;

    padding: 2px 5px;

    border-radius: 4px;
}}

.footer {{
    text-align: center;

    color: #6b7280;

    font-size: 13px;

    padding: 25px;
}}

@media print {{

    body {{
        background: white;
    }}

    .container {{
        max-width: none;

        padding: 15px;
    }}

    .section,
    .cover {{
        box-shadow: none;

        break-inside: avoid;
    }}

}}

</style>

</head>


<body>

<div class="container">


<!-- ====================================================================== -->
<!-- COVER -->
<!-- ====================================================================== -->

<div class="cover">

<h1>NORTHGATE AI</h1>

<div class="subtitle">
Final Model Evaluation &amp; Research Audit Report
</div>

<div class="meta">

<strong>Evaluation period:</strong>
{html_escape(test_start)}
→
{html_escape(test_end)}

<br>

<strong>Final OOS observations:</strong>
{html_escape(test_rows)}

<br>

<strong>Locked trading threshold:</strong>
{html_escape(threshold)}

<br>

<strong>Report generated:</strong>
{generation_time}

</div>

</div>


<!-- ====================================================================== -->
<!-- EXECUTIVE SUMMARY -->
<!-- ====================================================================== -->

<div class="section">

<h2>1. Executive Summary</h2>

<p>
This report consolidates the completed Northgate AI feature-set,
statistical, walk-forward, and final out-of-sample trading analyses.
The report uses previously generated result files and does not
retrain either model.
</p>

<div class="grid">

<div class="card">

<div class="card-title">
TOP-5 Final OOS AUC
</div>

<div class="card-value">
{fmt_float(top5_auc)}
</div>

</div>

<div class="card">

<div class="card-title">
24-Feature Final OOS AUC
</div>

<div class="card-value">
{fmt_float(model24_auc)}
</div>

</div>

<div class="card">

<div class="card-title">
AUC Difference
</div>

<div class="card-value">
{fmt_float(auc_difference)}
</div>

</div>

<div class="card">

<div class="card-title">
Final OOS Rows
</div>

<div class="card-value">
{html_escape(test_rows)}
</div>

</div>

</div>

<div class="conclusion">

<strong>Audit conclusion:</strong>

The TOP-5 model produced a higher observed ROC-AUC than the
24-feature model in the final out-of-sample sample. However,
the statistical uncertainty around the AUC difference is substantial.
The final locked-threshold trading comparison produced higher
observed trading metrics for the 24-feature model during this
particular final test period.

Therefore, the completed audit does not establish that TOP-5
should automatically replace the 24-feature model.

</div>

</div>


<!-- ====================================================================== -->
<!-- DATA AND OOS DESIGN -->
<!-- ====================================================================== -->

<div class="section">

<h2>2. Data &amp; Out-of-Sample Design</h2>

<table>

<tr>
<th>Dataset</th>
<th>Rows</th>
<th>Period</th>
</tr>

<tr>
<td>Training</td>
<td>1,915</td>
<td>2015-10-16 → 2023-05-25</td>
</tr>

<tr>
<td>Validation</td>
<td>411</td>
<td>2023-05-26 → 2025-01-15</td>
</tr>

<tr>
<td>Final Test</td>
<td>411</td>
<td>2025-01-16 → 2026-09-04</td>
</tr>

</table>

<div class="info">

The final OOS model evaluation trained using TRAIN + VALIDATION
data and evaluated the resulting models on the separate final
test period.

</div>

<h3>Feature sets</h3>

<p><strong>TOP-5:</strong></p>

<ul>

<li>AAPL_price_to_ma20</li>

<li>AAPL_volatility_20d</li>

<li>AAPL_volatility_5d</li>

<li>AAPL_return_5d</li>

<li>AAPL_return_20d</li>

</ul>

<p>
The comparison model uses the previously established
24-feature feature set.
</p>

</div>


<!-- ====================================================================== -->
<!-- FINAL OOS PREDICTIVE PERFORMANCE -->
<!-- ====================================================================== -->

<div class="section">

<h2>3. Final OOS Predictive Performance</h2>

<table>

<tr>

<th>Metric</th>

<th>TOP-5</th>

<th>24-Feature</th>

</tr>

<tr>

<td>ROC-AUC</td>

<td>{fmt_float(top5_auc)}</td>

<td>{fmt_float(model24_auc)}</td>

</tr>

<tr>

<td>Spearman</td>

<td>{fmt_float(top5_spearman)}</td>

<td>{fmt_float(model24_spearman)}</td>

</tr>

<tr>

<td>Accuracy @ 0.50</td>

<td>{fmt_float(top5_accuracy)}</td>

<td>{fmt_float(model24_accuracy)}</td>

</tr>

<tr>

<td>Precision @ 0.50</td>

<td>{fmt_float(top5_precision)}</td>

<td>{fmt_float(model24_precision)}</td>

</tr>

<tr>

<td>Recall @ 0.50</td>

<td>{fmt_float(top5_recall)}</td>

<td>{fmt_float(model24_recall)}</td>

</tr>

<tr>

<td>F1 @ 0.50</td>

<td>{fmt_float(top5_f1)}</td>

<td>{fmt_float(model24_f1)}</td>

</tr>

</table>

</div>


<!-- ====================================================================== -->
<!-- STATISTICAL SIGNIFICANCE -->
<!-- ====================================================================== -->

<div class="section">

<h2>4. Final OOS Statistical Significance</h2>

<div class="grid">

<div class="card">

<div class="card-title">
Observed AUC Difference
</div>

<div class="card-value">
{fmt_float(auc_difference)}
</div>

</div>

<div class="card">

<div class="card-title">
Bootstrap CI Low
</div>

<div class="card-value">
{fmt_float(auc_ci_low)}
</div>

</div>

<div class="card">

<div class="card-title">
Bootstrap CI High
</div>

<div class="card-value">
{fmt_float(auc_ci_high)}
</div>

</div>

<div class="card">

<div class="card-title">
Permutation p-value
</div>

<div class="card-value">
{fmt_float(permutation_p)}
</div>

</div>

</div>

<h3>AUC bootstrap confidence interval</h3>

<p>
Observed AUC difference:
<strong>{fmt_float(auc_difference)}</strong>
</p>

<p>
95% bootstrap confidence interval:
<strong>
[{fmt_float(auc_ci_low)}, {fmt_float(auc_ci_high)}]
</strong>
</p>

<p>
The confidence interval includes zero.
</p>

<h3>Paired permutation test</h3>

<p>
Permutation p-value:
<strong>{fmt_float(permutation_p)}</strong>
</p>

<p>
The completed test did not provide statistically conclusive
evidence that the observed AUC difference is different from zero.
</p>

<h3>Spearman analysis</h3>

<p>
TOP-5 Spearman:
<strong>{fmt_float(top5_spearman)}</strong>
</p>

<p>
24-feature Spearman:
<strong>{fmt_float(model24_spearman)}</strong>
</p>

<p>
Difference:
<strong>{fmt_float(spearman_difference)}</strong>
</p>

<p>
95% bootstrap CI:
<strong>
[{fmt_float(spearman_ci_low)}, {fmt_float(spearman_ci_high)}]
</strong>
</p>

</div>


<!-- ====================================================================== -->
<!-- SIGNAL ANALYSIS -->
<!-- ====================================================================== -->

<div class="section">

<h2>5. Locked Threshold Signal Analysis</h2>

<p>
The trading threshold remained locked at
<strong>{html_escape(threshold)}</strong>.
No threshold optimization was performed.
</p>

<table>

<tr>
<th>Signal category</th>
<th>Rows</th>
</tr>

<tr>
<td>Both models</td>
<td>{html_escape(both_signals)}</td>
</tr>

<tr>
<td>TOP-5 only</td>
<td>{html_escape(top5_only)}</td>
</tr>

<tr>
<td>24-feature only</td>
<td>{html_escape(model24_only)}</td>
</tr>

<tr>
<td>Neither</td>
<td>{html_escape(neither)}</td>
</tr>

</table>

<div class="note">

The TOP-5 model produced substantially more signals at the locked
0.65 threshold than the 24-feature model. Signal count alone does
not establish superior trading performance.

</div>

</div>


<!-- ====================================================================== -->
<!-- FINAL OOS TRADING -->
<!-- ====================================================================== -->

<div class="section">

<h2>6. Final OOS Locked Trading Performance</h2>

<p>

Locked rules:

</p>

<ul>

<li>Probability threshold: <strong>0.65</strong></li>

<li>Holding period: <strong>5 trading days</strong></li>

<li>Transaction cost: <strong>0.10% per side</strong></li>

<li>Initial capital: <strong>₹100,000</strong></li>

</ul>

<table>

<tr>

<th>Metric</th>

<th>TOP-5</th>

<th>24-Feature</th>

</tr>

<tr>

<td>Completed trades</td>

<td>{html_escape(top5_trades)}</td>

<td>{html_escape(model24_trades)}</td>

</tr>

<tr>

<td>Final equity</td>

<td>{fmt_rupees(top5_final_equity)}</td>

<td>{fmt_rupees(model24_final_equity)}</td>

</tr>

<tr>

<td>Total return</td>

<td>{fmt_percent(top5_total_return)}</td>

<td>{fmt_percent(model24_total_return)}</td>

</tr>

<tr>

<td>CAGR</td>

<td>{fmt_percent(top5_cagr)}</td>

<td>{fmt_percent(model24_cagr)}</td>

</tr>

<tr>

<td>Maximum drawdown</td>

<td>{fmt_percent(top5_drawdown)}</td>

<td>{fmt_percent(model24_drawdown)}</td>

</tr>

<tr>

<td>Sharpe</td>

<td>{fmt_float(top5_sharpe)}</td>

<td>{fmt_float(model24_sharpe)}</td>

</tr>

<tr>

<td>Win rate</td>

<td>{fmt_percent(top5_win_rate)}</td>

<td>{fmt_percent(model24_win_rate)}</td>

</tr>

<tr>

<td>Average trade return</td>

<td>{fmt_percent(top5_avg_trade)}</td>

<td>{fmt_percent(model24_avg_trade)}</td>

</tr>

</table>

<div class="note">

The final test contains only 411 observations.

The 24-feature model generated only 8 completed trades under the
locked threshold, while TOP-5 generated 30. Trading statistics,
particularly Sharpe ratio and win rate, should therefore be
interpreted together with trade count and sample size.

</div>

</div>


<!-- ====================================================================== -->
<!-- YEARLY PERFORMANCE -->
<!-- ====================================================================== -->

<div class="section">

<h2>7. Year-by-Year Final OOS Trading Performance</h2>

{dataframe_to_html(final_yearly)}

</div>


<!-- ====================================================================== -->
<!-- WALK FORWARD -->
<!-- ====================================================================== -->

<div class="section">

<h2>8. Previous Walk-Forward Statistical Evidence</h2>

<table>

<tr>
<th>Metric</th>
<th>Result</th>
</tr>

<tr>
<td>TOP-5 AUC</td>
<td>{fmt_float(wf_auc)}</td>
</tr>

<tr>
<td>24-feature AUC</td>
<td>{fmt_float(wf_model24_auc)}</td>
</tr>

<tr>
<td>AUC difference</td>
<td>{fmt_float(wf_auc_difference)}</td>
</tr>

<tr>
<td>Bootstrap CI</td>
<td>
[{fmt_float(wf_bootstrap_low)},
{fmt_float(wf_bootstrap_high)}]
</td>
</tr>

<tr>
<td>Permutation p-value</td>
<td>{fmt_float(wf_p_value)}</td>
</tr>

<tr>
<td>Paired observations</td>
<td>{html_escape(wf_paired_rows)}</td>
</tr>

<tr>
<td>Years TOP-5 had higher AUC</td>
<td>
{html_escape(wf_positive_years)}
/
{html_escape(wf_total_years)}
</td>
</tr>

</table>

<p>

The previous walk-forward analysis showed an observed AUC difference
in favor of TOP-5, but its confidence interval also included zero
and its permutation test was not statistically conclusive.

</p>

</div>


<!-- ====================================================================== -->
<!-- PREVIOUS TRADING STATISTICS -->
<!-- ====================================================================== -->

<div class="section">

<h2>9. Previous Trading Statistical Test</h2>

<table>

<tr>

<th>Metric</th>

<th>Value</th>

</tr>

<tr>

<td>24-feature trade count</td>

<td>{html_escape(previous_model24_trade_count)}</td>

</tr>

<tr>

<td>TOP-5 trade count</td>

<td>{html_escape(previous_top5_trade_count)}</td>

</tr>

<tr>

<td>24-feature mean trade return</td>

<td>{fmt_percent(previous_model24_mean_return)}</td>

</tr>

<tr>

<td>TOP-5 mean trade return</td>

<td>{fmt_percent(previous_top5_mean_return)}</td>

</tr>

<tr>

<td>TOP-5 minus 24-feature</td>

<td>{fmt_percent(previous_trade_difference)}</td>

</tr>

<tr>

<td>Bootstrap CI</td>

<td>
[{fmt_percent(previous_trade_ci_low)},
{fmt_percent(previous_trade_ci_high)}]
</td>

</tr>

<tr>

<td>Paired permutation p-value</td>

<td>{fmt_float(previous_p_value)}</td>

</tr>

<tr>

<td>Common paired trades</td>

<td>{html_escape(previous_common_trades)}</td>

</tr>

</table>

<p>

The previous trading statistical test did not establish a statistically
robust improvement in mean trade return for TOP-5.

</p>

</div>


<!-- ====================================================================== -->
<!-- AUDIT DISCIPLINE -->
<!-- ====================================================================== -->

<div class="section">

<h2>10. OOS &amp; Research Discipline</h2>

<ul>

<li>
Final test period was kept separate from TRAIN + VALIDATION during
final model evaluation.
</li>

<li>
The threshold remained locked at 0.65.
</li>

<li>
No threshold optimization was performed by the final audit.
</li>

<li>
The final audit did not retrain models.
</li>

<li>
The final test was not used by the audit to automatically select
a production configuration.
</li>

<li>
Trading comparisons used the same locked trading rules for both
feature sets.
</li>

</ul>

<div class="info">

This report summarizes completed experiments. It does not
automatically establish a production model or investment strategy.

</div>

</div>


<!-- ====================================================================== -->
<!-- LIMITATIONS -->
<!-- ====================================================================== -->

<div class="section">

<h2>11. Important Limitations</h2>

<ul>

<li>
The final OOS sample contains 411 observations.
</li>

<li>
The locked-threshold trading sample is small, particularly for
the 24-feature model.
</li>

<li>
Observed trading performance can vary substantially across
different market periods.
</li>

<li>
Bootstrap confidence intervals include uncertainty in the observed
sample.
</li>

<li>
The final audit does not establish universal superiority of either
feature set.
</li>

<li>
The trading analysis should not be interpreted as a guarantee of
future market performance.
</li>

</ul>

</div>


<!-- ====================================================================== -->
<!-- FINAL CONCLUSION -->
<!-- ====================================================================== -->

<div class="section">

<h2>12. Final Northgate AI Conclusion</h2>

<div class="conclusion">

<p>

<strong>Predictive performance:</strong><br>

TOP-5 achieved an observed final OOS ROC-AUC of
<strong>{fmt_float(top5_auc)}</strong>,
compared with
<strong>{fmt_float(model24_auc)}</strong>
for the 24-feature model.

The observed difference was
<strong>{fmt_float(auc_difference)}</strong>.

</p>

<p>

<strong>Statistical evidence:</strong><br>

The 95% bootstrap confidence interval for the AUC difference was
<strong>
[{fmt_float(auc_ci_low)}, {fmt_float(auc_ci_high)}]
</strong>,
which includes zero.

The paired permutation p-value was
<strong>{fmt_float(permutation_p)}</strong>.
Therefore the completed statistical analysis did not establish a
statistically conclusive difference between the feature sets.

</p>

<p>

<strong>Trading evidence:</strong><br>

Under the locked 0.65 threshold and identical trading rules,
the observed final OOS trading results were:

TOP-5:
<strong>{fmt_rupees(top5_final_equity)}</strong>
final equity from
<strong>{html_escape(top5_trades)}</strong>
completed trades.

24-feature:
<strong>{fmt_rupees(model24_final_equity)}</strong>
final equity from
<strong>{html_escape(model24_trades)}</strong>
completed trades.

</p>

<p>

<strong>Overall research conclusion:</strong><br>

The completed experiments show an observed predictive AUC advantage
for TOP-5 in the final OOS sample, but the statistical uncertainty
is substantial. The locked final OOS trading comparison produced
higher observed trading metrics for the 24-feature model during
the tested period.

Accordingly, the completed audit does not justify automatically
replacing the existing 24-feature configuration with TOP-5.

</p>

</div>

</div>


<!-- ====================================================================== -->
<!-- FILE INVENTORY -->
<!-- ====================================================================== -->

<div class="section">

<h2>13. Source Result Files</h2>

<ul>

<li>
<code>results/final_oos_top5_vs_24/final_test_summary.csv</code>
</li>

<li>
<code>results/final_oos_statistical_test/final_oos_statistical_summary.csv</code>
</li>

<li>
<code>results/final_oos_trading_comparison/final_oos_overall_trading_comparison.csv</code>
</li>

<li>
<code>results/final_oos_trading_comparison/final_oos_yearly_trading_comparison.csv</code>
</li>

<li>
<code>results/feature_set_statistical_significance_fast/statistical_significance_summary.csv</code>
</li>

<li>
<code>results/top5_trading_statistical_test/trading_statistical_summary.csv</code>
</li>

<li>
<code>results/final_project_audit/final_project_audit_summary.csv</code>
</li>

</ul>

</div>


<!-- ====================================================================== -->
<!-- FOOTER -->
<!-- ====================================================================== -->

<div class="footer">

Northgate AI — Final Project Audit Report<br>

Generated automatically from completed experiment outputs.<br>

No model retraining or threshold optimization was performed.

</div>


</div>

</body>

</html>
"""


# -----------------------------------------------------------------------------
# 12. WRITE REPORT
# -----------------------------------------------------------------------------

print_header("CREATING FINAL NORTHGATE AI REPORT")

with open(
    OUTPUT_HTML,
    "w",
    encoding="utf-8"
) as f:

    f.write(html_report)


print()
print("FINAL REPORT CREATED")
print("--------------------")
print(OUTPUT_HTML)

print()
print("=" * 80)
print("NORTHGATE AI — FINAL PROJECT REPORT COMPLETED")
print("=" * 80)
print()