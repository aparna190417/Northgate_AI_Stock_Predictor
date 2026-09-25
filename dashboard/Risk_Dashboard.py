# NORTHGATE AI — RISK DASHBOARD TERMINAL 

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================
# 1. PAGE CONFIG (SAFEGUARDED FOR MULTIPAGE)
# ============================================================

try:
    st.set_page_config(
        page_title="Risk Dashboard | Northgate AI",
        page_icon="🛡️",
        layout="wide",
    )
except Exception:
    pass

# ============================================================
# 2. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MODEL_DIR = RESULTS_DIR / "model_comparison"

# ============================================================
# 3. DARK QUANT RESEARCH THEME (CSS)
# ============================================================

st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #050b14 !important;
        color: #e8eef7 !important;
    }

    .hero {
        background: linear-gradient(135deg, rgba(18,42,72,.95), rgba(8,20,35,.98));
        border: 1px solid rgba(71,145,219,.22);
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.2rem;
    }

    .eyebrow {
        color: #36bdf2;
        font-size: 0.70rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    .hero h1 {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0;
        color: #f4f8fc;
    }

    .hero p {
        color: #91a3b8;
        margin: 0.35rem 0 0 0;
        font-size: 0.84rem;
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-size: 0.68rem;
        font-weight: 800;
        margin-right: 0.35rem;
        margin-top: 0.8rem;
        border: 1px solid rgba(255,255,255,.10);
    }

    .badge-green { color: #00e58a; background: rgba(0,229,138,.08); border-color: rgba(0,229,138,0.3); }
    .badge-blue { color: #39bdf5; background: rgba(57,189,245,.08); border-color: rgba(57,189,245,0.3); }
    .badge-purple { color: #b78cff; background: rgba(183,140,255,.08); border-color: rgba(183,140,255,0.3); }

    .section-title {
        background: #071321;
        border: 1px solid rgba(255,255,255,.055);
        border-radius: 10px;
        padding: 0.8rem 1.1rem;
        margin: 1.2rem 0 0.8rem 0;
    }

    .section-title h3 {
        margin: 0;
        color: #eaf1f8;
        font-size: 0.95rem;
        font-weight: 750;
    }

    .section-title p {
        margin: 0.2rem 0 0 0;
        color: #708399;
        font-size: 0.72rem;
    }

    .metric-card {
        background: #071321;
        border: 1px solid rgba(255,255,255,.07);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        min-height: 98px;
    }

    .metric-label {
        color: #77899d;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 800;
    }

    .metric-value {
        color: #f2f7fb;
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }

    .metric-sub {
        color: #63768b;
        font-size: 0.68rem;
        margin-top: 0.2rem;
    }

    .info-box {
        background: #071321;
        border: 1px solid rgba(255,255,255,.06);
        border-radius: 10px;
        padding: 0.95rem 1.1rem;
        color: #9aabbd;
        font-size: 0.74rem;
        line-height: 1.6;
        height: 100%;
    }
    </style>""",
    unsafe_allow_html=True,
)

# ============================================================
# 4. PLOTLY THEME
# ============================================================

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#8da2be", size=11),
    margin=dict(l=20, r=25, t=30, b=20),
)

# ============================================================
# 5. SMART HELPERS WITH INTUITIVE SCALING
# ============================================================

def clean_numeric(df):
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            converted = pd.to_numeric(
                out[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False),
                errors="coerce",
            )
            if converted.notna().sum() >= max(3, int(len(out) * 0.5)):
                out[col] = converted
    return out

def find_artifact(candidates):
    folders = [MODEL_DIR, RESULTS_DIR]
    for folder in folders:
        if not folder.exists():
            continue
        for filename in candidates:
            path = folder / filename
            if path.exists():
                return path
    for folder in folders:
        if not folder.exists():
            continue
        for filename in candidates:
            matches = list(folder.rglob(filename))
            if matches:
                return matches[0]
    return None

def read_csv_safe(path):
    if path is None or not path.exists():
        return None
    try:
        return clean_numeric(pd.read_csv(path))
    except Exception:
        return None

def find_column(df, candidates):
    """Smart hierarchical matcher: exact -> normalized -> substring."""
    if df is None or df.empty:
        return None
    
    # 1. Exact lower match
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        cand_l = cand.lower().strip()
        if cand_l in lower_map:
            return lower_map[cand_l]

    # 2. Normalized alphanumeric match
    normalized = {re.sub(r"[^a-z0-9]", "", str(col).lower()): col for col in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in normalized:
            return normalized[key]

    # 3. Fuzzy substring containment
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        for norm_col, original_col in normalized.items():
            if key in norm_col or norm_col in key:
                return original_col
    return None

def first_value(df, candidates, default=np.nan):
    col = find_column(df, candidates)
    if col is None or df.empty:
        return default
    val = pd.to_numeric(df[col].iloc[0], errors="coerce")
    if pd.isna(val):
        return default
    return float(val)

def to_pct_scalar(val):
    """Converts decimal (0.2888) to percentage (28.88) safely."""
    if pd.isna(val):
        return np.nan
    v = float(val)
    if abs(v) <= 2.5 and v != 0:
        return v * 100.0
    return v

def fmt_pct(value, decimals=2):
    if pd.isna(value):
        return "—"
    v = to_pct_scalar(value)
    return f"{v:.{decimals}f}%"

def fmt_num(value, decimals=2):
    if pd.isna(value):
        return "—"
    return f"{float(value):.{decimals}f}"

# ============================================================
# 6. LOAD RISK ARTIFACT
# ============================================================

@st.cache_data
def load_risk_data():
    path = find_artifact([
        "portfolio_risk_snapshot.csv",
        "risk_snapshot.csv",
        "portfolio_risk.csv",
        "portfolio_research_signals.csv",
    ])
    df = read_csv_safe(path)
    return df, path

risk_df, risk_path = load_risk_data()

# ============================================================
# 7. HERO SECTION
# ============================================================

st.markdown(
    """<div class="hero">
        <div class="eyebrow">Quantitative Risk Terminal</div>
        <h1>Risk Dashboard</h1>
        <p>Asset-level volatility, drawdown, beta and risk-state diagnostics across the Northgate AI research universe.</p>
        <div>
            <span class="badge badge-green">● REAL RISK ARTIFACT</span>
            <span class="badge badge-blue">VOLATILITY / DRAWDOWN</span>
            <span class="badge badge-purple">BETA & RISK STATE</span>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# ============================================================
# 8. ARTIFACT STATUS
# ============================================================

if risk_df is not None:
    st.markdown(
        f"""<div class="info-box">
            <b style="color:#39bdf5;">Risk Snapshot:</b>
            Latest verified asset-level risk artifact detected:
            <b style="color:#dce8f4;">{risk_path.name}</b>.
            Risk metrics are displayed directly from the project's research output without synthetic fallbacks.
        </div>""",
        unsafe_allow_html=True,
    )
else:
    st.warning(
        "The verified portfolio risk snapshot artifact was not found. "
        "No synthetic risk values are being generated."
    )

# ============================================================
# 9. SELECT ASSET
# ============================================================

ticker_col = find_column(risk_df, ["ticker", "symbol", "asset"])

if ticker_col:
    tickers = sorted(risk_df[ticker_col].dropna().astype(str).unique().tolist())
else:
    tickers = []

if tickers:
    selected_ticker = st.selectbox(
        "Asset Coverage",
        tickers,
        index=tickers.index("AAPL") if "AAPL" in tickers else 0,
    )
    selected_df = risk_df[risk_df[ticker_col].astype(str) == selected_ticker].copy()
else:
    selected_ticker = None
    selected_df = pd.DataFrame()

# ============================================================
# 10. SELECTED ASSET METRICS (SCALED PROPERLY)
# ============================================================

st.markdown(
    """<div class="section-title">
        <h3>Selected Asset Risk Profile</h3>
        <p>Current risk characteristics from the verified research snapshot.</p>
    </div>""",
    unsafe_allow_html=True,
)

if not selected_df.empty:
    current_vol = first_value(
        selected_df,
        ["annual_volatility", "annualized_volatility", "volatility", "annual_vol", "vol"],
    )
    vol_20d = first_value(
        selected_df,
        ["20d_volatility", "vol_20d", "volatility_20d", "rolling_20d_volatility", "vol_20", "20d_vol"],
    )
    vol_60d = first_value(
        selected_df,
        ["60d_volatility", "vol_60d", "volatility_60d", "rolling_60d_volatility", "vol_60", "60d_vol"],
    )
    drawdown = first_value(
        selected_df,
        ["current_drawdown", "drawdown", "current_dd", "dd"],
    )
    max_drawdown = first_value(
        selected_df,
        ["max_drawdown", "maximum_drawdown", "max_dd", "mdd"],
    )
    beta = first_value(
        selected_df,
        ["beta_vs_sp500", "beta", "market_beta", "sp500_beta", "beta_sp500"],
    )
    risk_state_col = find_column(
        selected_df,
        ["risk_state", "risk_level", "risk_regime", "risk"],
    )
    risk_state = (
        str(selected_df[risk_state_col].iloc[0])
        if risk_state_col
        else "—"
    )

    cols = st.columns(6)
    cards = [
        ("ANNUAL VOLATILITY", fmt_pct(current_vol), "Annualized risk"),
        ("20D VOLATILITY", fmt_pct(vol_20d), "Short-term regime"),
        ("60D VOLATILITY", fmt_pct(vol_60d), "Medium-term regime"),
        ("CURRENT DRAWDOWN", fmt_pct(drawdown), "From recent peak"),
        ("MAX DRAWDOWN", fmt_pct(max_drawdown), "Historical snapshot"),
        ("BETA", fmt_num(beta), "vs S&P 500"),  # Beta is purely scalar, NOT percentage
    ]

    for col, (label, value, sub) in zip(cols, cards):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-sub">{sub}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    state_lower = risk_state.lower()
    if "high" in state_lower:
        state_color = "#ff4b4b"
    elif "moderate" in state_lower:
        state_color = "#f0b34b"
    elif "low" in state_lower:
        state_color = "#00df86"
    else:
        state_color = "#39bdf5"

    st.markdown(
        f"""<div style="margin-top:12px;background:#071321;border:1px solid rgba(255,255,255,.06);border-radius:9px;padding:12px 15px;">
            <span style="color:#71869b;font-size:9px;font-weight:800;letter-spacing:.7px;">CURRENT RISK STATE</span>
            <span style="color:{state_color};font-size:14px;font-weight:800;margin-left:12px;">● {risk_state.upper()}</span>
        </div>""",
        unsafe_allow_html=True,
    )

# ============================================================
# 11. RISK UNIVERSE TABLE (SCALED CONSISTENTLY)
# ============================================================

st.markdown(
    """<div class="section-title">
        <h3>Risk Monitor — Research Universe</h3>
        <p>Cross-sectional comparison of volatility, drawdown, beta and project-defined risk state.</p>
    </div>""",
    unsafe_allow_html=True,
)

if risk_df is not None and ticker_col:
    display = risk_df.copy()
    
    mapping_rules = {
        "Ticker": [ticker_col],
        "Annual Volatility": ["annual_volatility", "annualized_volatility", "volatility", "annual_vol"],
        "20D Volatility": ["20d_volatility", "vol_20d", "volatility_20d", "vol_20"],
        "60D Volatility": ["60d_volatility", "vol_60d", "volatility_60d", "vol_60"],
        "Current Drawdown": ["current_drawdown", "drawdown", "current_dd"],
        "Max Drawdown": ["max_drawdown", "maximum_drawdown", "max_dd"],
        "Beta": ["beta_vs_sp500", "beta", "market_beta"],
        "Risk State": ["risk_state", "risk_level", "risk"],
    }

    table_data = pd.DataFrame(index=display.index)
    for target_name, cand_list in mapping_rules.items():
        found = find_column(display, cand_list)
        if found:
            table_data[target_name] = display[found]

    percent_cols = [
        "Annual Volatility", "20D Volatility", "60D Volatility",
        "Current Drawdown", "Max Drawdown"
    ]

    for col in percent_cols:
        if col in table_data.columns:
            table_data[col] = pd.to_numeric(table_data[col], errors="coerce").map(
                lambda x: fmt_pct(x) if pd.notna(x) else "—"
            )

    if "Beta" in table_data.columns:
        table_data["Beta"] = pd.to_numeric(table_data["Beta"], errors="coerce").map(
            lambda x: fmt_num(x) if pd.notna(x) else "—"
        )

    st.dataframe(table_data, hide_index=True, width="stretch")

# ============================================================
# 12. ANNUAL VOLATILITY RANKING (PRO SPECTRUM BARS)
# ============================================================

st.markdown(
    """<div class="section-title">
        <h3>Annualized Volatility Ranking</h3>
        <p>Dispersion analysis with adaptive risk-spectrum coloring (Low / Moderate / High).</p>
    </div>""",
    unsafe_allow_html=True,
)

annual_vol_col = find_column(
    risk_df,
    ["annual_volatility", "annualized_volatility", "volatility", "annual_vol"],
)

if risk_df is not None and ticker_col and annual_vol_col:
    vol_plot = risk_df[[ticker_col, annual_vol_col]].copy()
    vol_plot[annual_vol_col] = pd.to_numeric(vol_plot[annual_vol_col], errors="coerce").map(to_pct_scalar)
    vol_plot = vol_plot.dropna()
    vol_plot.columns = ["Ticker", "Volatility"]
    vol_plot = vol_plot.sort_values("Volatility", ascending=True)

    colors = []
    for v in vol_plot["Volatility"]:
        if v < 22.0:
            colors.append("#00e58a")  # Low Vol
        elif v < 30.0:
            colors.append("#36bdf2")  # Moderate Vol
        elif v < 40.0:
            colors.append("#f0b34b")  # Elevated Vol
        else:
            colors.append("#ff4b4b")  # High Vol

    fig_vol = go.Figure()
    fig_vol.add_trace(
        go.Bar(
            x=vol_plot["Volatility"],
            y=vol_plot["Ticker"],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.12)", width=1),
            ),
            text=[f"<b>{v:.2f}%</b>" for v in vol_plot["Volatility"]],
            textposition="outside",
            textfont=dict(color="#cbd5e1", size=10),
            hovertemplate="<b>%{y}</b><br>Annual Volatility: %{x:.2f}%<extra></extra>",
        )
    )

    fig_vol.update_layout(
        **PLOTLY_THEME,
        height=380,
        showlegend=False,
        xaxis=dict(
            title="Annualized Volatility (%)",
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_vol, width="stretch")

# ============================================================
# 13. DUAL-TONE UNDERWATER DRAWDOWN SEVERITY
# ============================================================

st.markdown(
    """<div class="section-title">
        <h3>Peak-to-Trough Drawdown Severity</h3>
        <p>Comparative profile of current drawdown vs historical maximum observed contraction.</p>
    </div>""",
    unsafe_allow_html=True,
)

dd_col = find_column(risk_df, ["current_drawdown", "drawdown", "current_dd"])
maxdd_col = find_column(risk_df, ["max_drawdown", "maximum_drawdown", "max_dd"])

if risk_df is not None and ticker_col and dd_col and maxdd_col:
    dd_plot = risk_df[[ticker_col, dd_col, maxdd_col]].copy()
    dd_plot[dd_col] = pd.to_numeric(dd_plot[dd_col], errors="coerce").map(to_pct_scalar)
    dd_plot[maxdd_col] = pd.to_numeric(dd_plot[maxdd_col], errors="coerce").map(to_pct_scalar)
    dd_plot = dd_plot.dropna(subset=[dd_col, maxdd_col])
    dd_plot.columns = ["Ticker", "Current Drawdown", "Max Drawdown"]
    dd_plot = dd_plot.sort_values("Max Drawdown", ascending=True)

    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Bar(
            name="Current Drawdown",
            x=dd_plot["Ticker"],
            y=dd_plot["Current Drawdown"],
            marker=dict(color="rgba(54, 189, 242, 0.75)", line=dict(color="#36bdf2", width=1)),
            hovertemplate="Current: %{y:.2f}%<extra></extra>",
        )
    )
    fig_dd.add_trace(
        go.Bar(
            name="Max Historical Drawdown",
            x=dd_plot["Ticker"],
            y=dd_plot["Max Drawdown"],
            marker=dict(color="rgba(255, 75, 75, 0.8)", line=dict(color="#ff4b4b", width=1.2)),
            hovertemplate="Max Drawdown: %{y:.2f}%<extra></extra>",
        )
    )

    fig_dd.update_layout(
        **PLOTLY_THEME,
        height=380,
        barmode="group",
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(
            title="Contraction Depth (%)",
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.2)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0,
            font=dict(size=10, color="#cbd5e1"),
        ),
    )
    st.plotly_chart(fig_dd, width="stretch")

# ============================================================
# 14. 4-QUADRANT RISK MAP & REGIME BREAKDOWN
# ============================================================

col_map, col_pie = st.columns([1.5, 1], gap="medium")

beta_col = find_column(risk_df, ["beta_vs_sp500", "beta", "market_beta"])

with col_map:
    st.markdown(
        """<div class="section-title">
            <h3>Institutional Risk-Return Quadrant</h3>
            <p>Market Beta sensitivity vs. Annualized Return Dispersion.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    if risk_df is not None and ticker_col and beta_col and annual_vol_col:
        beta_plot = risk_df[[ticker_col, beta_col, annual_vol_col]].copy()
        beta_plot[beta_col] = pd.to_numeric(beta_plot[beta_col], errors="coerce")
        beta_plot[annual_vol_col] = pd.to_numeric(beta_plot[annual_vol_col], errors="coerce").map(to_pct_scalar)
        beta_plot = beta_plot.dropna()
        beta_plot.columns = ["Ticker", "Beta", "Volatility"]

        mean_vol = beta_plot["Volatility"].mean()

        fig_scat = go.Figure()
        fig_scat.add_trace(
            go.Scatter(
                x=beta_plot["Beta"],
                y=beta_plot["Volatility"],
                mode="markers+text",
                text=beta_plot["Ticker"],
                textposition="top center",
                textfont=dict(size=10, color="#f8fafc", family="Plus Jakarta Sans"),
                marker=dict(
                    size=14,
                    color=beta_plot["Volatility"],
                    colorscale="Tealgrn",
                    showscale=False,
                    line=dict(width=1.5, color="#ffffff"),
                    opacity=0.9,
                ),
                hovertemplate="<b>%{text}</b><br>Beta: %{x:.2f}<br>Volatility: %{y:.2f}%<extra></extra>",
            )
        )

        fig_scat.add_vline(x=1.0, line_dash="dash", line_color="rgba(255, 75, 75, 0.6)", annotation_text="Market Beta (1.0)", annotation_position="bottom right", annotation_font_size=9)
        fig_scat.add_hline(y=mean_vol, line_dash="dot", line_color="rgba(54, 189, 242, 0.5)", annotation_text="Universe Mean Vol", annotation_position="top left", annotation_font_size=9)

        fig_scat.add_annotation(x=beta_plot["Beta"].min(), y=beta_plot["Volatility"].min(), text="<b>DEFENSIVE STABLE</b>", showarrow=False, font=dict(size=8, color="#50657d"))
        fig_scat.add_annotation(x=beta_plot["Beta"].max()*0.9, y=beta_plot["Volatility"].max(), text="<b>HIGH-BETA EXPANSION</b>", showarrow=False, font=dict(size=8, color="#50657d"))

        fig_scat.update_layout(
            **PLOTLY_THEME,
            height=370,
            xaxis=dict(title="Market Beta (Sensitivity to SPX)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Annualized Volatility (%)", gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_scat, width="stretch")

with col_pie:
    st.markdown(
        """<div class="section-title">
            <h3>Risk State Breakdown</h3>
            <p>Distribution of assets across quant risk bands.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    risk_state_col = find_column(risk_df, ["risk_state", "risk_level", "risk"])

    if risk_df is not None and risk_state_col:
        state_counts = (
            risk_df[risk_state_col]
            .astype(str)
            .str.upper()
            .value_counts()
            .reset_index()
        )
        state_counts.columns = ["Risk State", "Assets"]

        pie_colors = {
            "LOW": "#00e58a",
            "MODERATE": "#f0b34b",
            "HIGH": "#ff4b4b",
            "CRITICAL": "#b78cff",
        }
        color_seq = [pie_colors.get(s, "#36bdf2") for s in state_counts["Risk State"]]

        fig_pie = go.Figure(
            go.Pie(
                labels=state_counts["Risk State"],
                values=state_counts["Assets"],
                hole=0.68,
                marker=dict(colors=color_seq, line=dict(color="#071321", width=2)),
                textinfo="none",
                hoverinfo="label+value+percent",
            )
        )

        fig_pie.add_annotation(
            text=f"<b>{len(risk_df)}</b><br><span style='font-size:10px;color:#71869c;'>Assets</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=18, color="#ffffff")
        )

        fig_pie.update_layout(
            **PLOTLY_THEME,
            height=370,
            showlegend=True,
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center", font=dict(size=10)),
        )
        st.plotly_chart(fig_pie, width="stretch")

# ============================================================
# 15. METHODOLOGY & PROVENANCE
# ============================================================

st.markdown(
    """<div class="section-title">
        <h3>Risk Methodology & Protocol</h3>
        <p>How Northgate AI computes and interprets multi-asset risk metrics.</p>
    </div>""",
    unsafe_allow_html=True,
)

method_cols = st.columns(4)
methodology = [
    ("01 — VOLATILITY", "Annualized standard deviation of returns measuring price dispersion."),
    ("02 — DRAWDOWN", "Maximum observed peak-to-trough decline over the evaluated window."),
    ("03 — BETA", "Relative sensitivity against the S&P 500 benchmark (Beta = 1.0 baseline)."),
    ("04 — RISK STATE", "Empirical classification into LOW, MODERATE, or HIGH risk regimes."),
]

for col, (title, body) in zip(method_cols, methodology):
    with col:
        st.markdown(
            f"""<div class="info-box">
                <div style="color:#39bdf5;font-size:0.68rem;font-weight:800;letter-spacing:.05em;margin-bottom:0.35rem;">{title}</div>
                {body}
            </div>""",
            unsafe_allow_html=True,
        )

st.write("")
filename = risk_path.name if risk_path is not None else "Not detected"
status = "● Available" if risk_df is not None else "● Not mounted"
status_color = "#00df86" if risk_df is not None else "#f0b34b"

st.markdown(
    f"""<div class="info-box">
        <div style="color:#71869c;font-size:0.65rem;font-weight:800;letter-spacing:.06em;">PORTFOLIO RISK SNAPSHOT ARTIFACT</div>
        <div style="color:#dbe5ef;font-size:0.85rem;font-weight:700;margin-top:0.2rem;">{filename}</div>
        <div style="color:{status_color};font-size:0.7rem;margin-top:0.2rem;">{status}</div>
    </div>""",
    unsafe_allow_html=True,
)

st.markdown(
    """<div style="margin-top:1.2rem;background:rgba(90,65,20,.13);border:1px solid rgba(230,173,65,.22);border-radius:8px;padding:0.75rem 1rem;color:#9d927b;font-size:0.72rem;line-height:1.5;">
        ⚠ <b style="color:#c5a766;">Quantitative Research Disclaimer:</b>
        Risk statistics shown here are empirical outputs generated by the Northgate AI research pipeline. 
        They describe observed risk regimes and do not guarantee future performance. Research and educational use only. Not financial advice.
    </div>""",
    unsafe_allow_html=True,)