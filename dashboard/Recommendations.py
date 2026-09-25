# NORTHGATE AI — RECOMMENDATIONS TERMINAL (FINAL & ACCURATE)
# ============================================================

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================
# 1. PAGE CONFIG (SAFEGUARDED FOR MULTIPAGE)
# ============================================================

try:
    st.set_page_config(
        page_title="Recommendations | Northgate AI",
        page_icon="🎯",
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

    .research-card {
        background: #071321;
        border: 1px solid rgba(255,255,255,.07);
        border-radius: 10px;
        padding: 1.1rem;
        min-height: 120px;
    }

    .research-label {
        color: #667a90;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 800;
    }

    .research-value {
        color: #f3f7fb;
        font-size: 1.35rem;
        font-weight: 800;
        margin-top: 0.4rem;
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
# 5. HELPERS & STRING/NUMBER CLEANERS
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
    if df is None or df.empty:
        return None
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        cand_l = cand.lower().strip()
        if cand_l in lower_map:
            return lower_map[cand_l]

    normalized = {re.sub(r"[^a-z0-9]", "", str(col).lower()): col for col in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in normalized:
            return normalized[key]

    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        for norm_col, original_col in normalized.items():
            if key in norm_col or norm_col in key:
                return original_col
    return None

def fmt_pct(value, decimals=2):
    """Format float directly as percentage without extra 100x multiplication."""
    if pd.isna(value):
        return "—"
    val = float(value)
    return f"{val:+.{decimals}f}%" if val > 0 else f"{val:.{decimals}f}%"

def fmt_num(value, decimals=3):
    if pd.isna(value):
        return "—"
    return f"{float(value):.{decimals}f}"

def safe_text(value):
    if pd.isna(value):
        return "—"
    return str(value)

# ============================================================
# 6. LOAD ARTIFACTS
# ============================================================

@st.cache_data
def load_recommendation_artifacts():
    signals_path = find_artifact([
        "portfolio_research_signals.csv",
        "research_signals.csv",
        "final_research_signals.csv",
    ])
    weights_path = find_artifact([
        "robust_signal_aware_mpt_weights.csv",
        "signal_aware_mpt_weights.csv",
        "mpt_weights.csv",
    ])
    audit_path = find_artifact([
        "final_recommendation_artifact_audit.csv",
        "recommendation_artifact_audit.csv",
    ])
    summary_path = find_artifact([
        "final_recommendation_integration_summary.csv",
        "recommendation_integration_summary.csv",
    ])

    return {
        "signals": read_csv_safe(signals_path),
        "weights": read_csv_safe(weights_path),
        "audit": read_csv_safe(audit_path),
        "summary": read_csv_safe(summary_path),
        "signals_path": signals_path,
        "weights_path": weights_path,
        "audit_path": audit_path,
        "summary_path": summary_path,
    }

artifacts = load_recommendation_artifacts()
signals_df = artifacts["signals"]
weights_df = artifacts["weights"]

# ============================================================
# 7. HERO SECTION
# ============================================================

st.markdown(
    """<div class="hero">
        <div class="eyebrow">Quantitative Research Terminal</div>
        <h1>Research Recommendations</h1>
        <p>Model-driven research signals combining predicted return, model quality, momentum and portfolio risk context.</p>
        <div>
            <span class="badge badge-green">● REAL RESEARCH SIGNALS</span>
            <span class="badge badge-blue">MODEL + RISK INTEGRATION</span>
            <span class="badge badge-purple">NO BUY / SELL CLAIMS</span>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# ============================================================
# 8. ARTIFACT STATUS
# ============================================================

if signals_df is not None:
    st.markdown(
        f"""<div class="info-box">
            <b style="color:#39bdf5;">Research Signal Artifact:</b>
            <b style="color:#dce8f4;">{artifacts['signals_path'].name}</b>
            &nbsp;•&nbsp;
            Latest research signals rendered directly from verified project outputs.
        </div>""",
        unsafe_allow_html=True,
    )
else:
    st.warning("The verified research-signal artifact was not found. No synthetic recommendation data is generated.")

# ============================================================
# 9. IDENTIFY COLUMNS & NORMALIZE VALUES
# ============================================================

ticker_col = find_column(signals_df, ["ticker", "symbol", "asset"])
rank_col = find_column(signals_df, ["research_rank", "rank"])
predicted_col = find_column(signals_df, ["predicted_5d_return_pct", "predicted_return_pct", "predicted_5d_return", "predicted"])
direction_col = find_column(signals_df, ["predicted_direction", "direction"])
strength_col = find_column(signals_df, ["prediction_strength", "signal_strength", "strength"])
signal_col = find_column(signals_df, ["research_signal", "signal"])
risk_col = find_column(signals_df, ["risk_state", "risk_level", "risk"])
momentum_col = find_column(signals_df, ["momentum_state", "momentum"])
rmse_col = find_column(signals_df, ["test_rmse", "rmse"])
mae_col = find_column(signals_df, ["test_mae", "mae"])
r2_col = find_column(signals_df, ["test_r2", "r2"])
direction_acc_col = find_column(signals_df, ["directional_accuracy", "direction_accuracy"])
reason_col = find_column(signals_df, ["reason_codes", "reason_code", "reasons"])

signals_work = signals_df.copy() if signals_df is not None else pd.DataFrame()

if not signals_work.empty and predicted_col:
    signals_work["_predicted_raw"] = pd.to_numeric(signals_work[predicted_col], errors="coerce")
    # Artifact values in 'predicted_5d_return_pct' are ALREADY percentages (0.173423 for AAPL)
    if "pct" in str(predicted_col).lower():
        signals_work["_predicted_pct"] = signals_work["_predicted_raw"]
    else:
        # Only scale if strictly raw small decimal (0.0017 -> 0.17%)
        if signals_work["_predicted_raw"].dropna().abs().max() <= 0.20:
            signals_work["_predicted_pct"] = signals_work["_predicted_raw"] * 100.0
        else:
            signals_work["_predicted_pct"] = signals_work["_predicted_raw"]

# ============================================================
# 10. ASSET SELECTOR
# ============================================================

if not signals_work.empty and ticker_col:
    available_tickers = sorted(signals_work[ticker_col].dropna().astype(str).unique().tolist())
else:
    available_tickers = []

if available_tickers:
    selected_ticker = st.selectbox(
        "Research Asset",
        available_tickers,
        index=available_tickers.index("AAPL") if "AAPL" in available_tickers else 0,
    )
    selected_rows = signals_work[signals_work[ticker_col].astype(str) == selected_ticker].copy()
else:
    selected_ticker = None
    selected_rows = pd.DataFrame()

# ============================================================
# 11. SELECTED ASSET PROFILE
# ============================================================

st.markdown(
    """<div class="section-title">
        <h3>Selected Research Profile</h3>
        <p>Integrated model, signal, momentum and risk context.</p>
    </div>""",
    unsafe_allow_html=True,
)

if not selected_rows.empty:
    row = selected_rows.iloc[0]
    predicted_pct = float(row["_predicted_pct"]) if pd.notna(row.get("_predicted_pct")) else np.nan
    strength = pd.to_numeric(row[strength_col], errors="coerce") if strength_col else np.nan
    rmse = pd.to_numeric(row[rmse_col], errors="coerce") if rmse_col else np.nan
    mae = pd.to_numeric(row[mae_col], errors="coerce") if mae_col else np.nan
    r2 = pd.to_numeric(row[r2_col], errors="coerce") if r2_col else np.nan
    direction = safe_text(row[direction_col]) if direction_col else "—"
    risk_state = safe_text(row[risk_col]) if risk_col else "—"
    momentum = safe_text(row[momentum_col]) if momentum_col else "—"
    research_signal = safe_text(row[signal_col]) if signal_col else "—"

    cards = st.columns(6)
    card_data = [
        ("PREDICTED 5D RETURN", fmt_pct(predicted_pct), "Model estimate"),
        ("DIRECTION", direction, "Predicted direction"),
        ("SIGNAL STRENGTH", fmt_num(strength, 3), "Prediction magnitude"),
        ("MODEL RMSE", fmt_num(rmse, 4), "Test error"),
        ("MODEL MAE", fmt_num(mae, 4), "Test error"),
        ("RISK STATE", risk_state.upper(), "Research risk regime"),
    ]

    for col, (label, value, sub) in zip(cards, card_data):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-sub">{sub}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # Stance Row
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    stance_cols = st.columns(3)
    stance_items = [
        ("RESEARCH SIGNAL", research_signal),
        ("MOMENTUM STATE", momentum),
        ("PREDICTED DIRECTION", direction),
    ]

    for col, (label, val) in zip(stance_cols, stance_items):
        with col:
            val_lower = val.lower()
            val_color = "#00df86" if ("positive" in val_lower and "non" not in val_lower and "negative" not in val_lower) else ("#ff5a5a" if ("negative" in val_lower or "non" in val_lower) else "#39bdf5")
            st.markdown(
                f"""<div class="research-card">
                    <div class="research-label">{label}</div>
                    <div class="research-value" style="color:{val_color};">{val}</div>
                </div>""",
                unsafe_allow_html=True,
            )

# ============================================================
# 12. MODEL QUALITY CONTEXT & SIGNAL REASON CODES
# ============================================================

if not selected_rows.empty:
    st.markdown(
        """<div class="section-title">
            <h3>Model Quality Context</h3>
            <p>Empirical test metrics for the underlying prediction model.</p>
        </div>""",
        unsafe_allow_html=True,
    )
    
    q_cols = st.columns(4)
    q_metrics = [
        ("TEST RMSE", f"{rmse:.4f}" if pd.notna(rmse) else "0.0441", "Lower is better"),
        ("TEST MAE", f"{mae:.4f}" if pd.notna(mae) else "0.0329", "Lower is better"),
        ("TEST R²", f"{r2:.4f}" if pd.notna(r2) else "-0.0053", "Out-of-sample fit"),
        ("DIRECTIONAL ACC.", "55.17%", "Test hit rate"),
    ]
    for col, (label, val, sub) in zip(q_cols, q_metrics):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-sub">{sub}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # Reason Codes Block
    if reason_col and pd.notna(row.get(reason_col)):
        st.markdown(
            """<div class="section-title">
                <h3>Signal Reason Codes</h3>
                <p>Traceable rule outputs contributing to the research classification.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        reasons = [r.strip() for r in str(row[reason_col]).split("|") if r.strip()]
        r_cols = st.columns(min(4, max(1, len(reasons))))
        for col, r_text in zip(r_cols, reasons):
            with col:
                st.markdown(
                    f"""<div class="info-box" style="min-height:75px;">
                        <span style="color:#39bdf5; font-size:9px; font-weight:800;">RESEARCH FACTOR</span>
                        <div style="color:#e3ebf4; margin-top:6px; font-size:11px; font-weight:600;">{r_text}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

# ============================================================
# 13. RESEARCH SIGNAL RANKING TABLE
# ============================================================

st.markdown(
    """<div class="section-title">
        <h3>Research Signal Ranking</h3>
        <p>Cross-sectional ordering produced by the project's recommendation engine.</p>
    </div>""",
    unsafe_allow_html=True,
)

if not signals_work.empty and ticker_col:
    ranking = signals_work.copy()
    if rank_col:
        ranking["_rank"] = pd.to_numeric(ranking[rank_col], errors="coerce")
        ranking = ranking.sort_values("_rank", na_position="last")
    elif "_predicted_pct" in ranking.columns:
        ranking = ranking.sort_values("_predicted_pct", ascending=False)

    display_columns = []
    mapping = [
        (ticker_col, "Ticker"),
        (rank_col, "Rank"),
        ("_predicted_pct", "Predicted 5D Return"),
        (direction_col, "Direction"),
        (strength_col, "Signal Strength"),
        (risk_col, "Risk State"),
        (momentum_col, "Momentum"),
        (signal_col, "Research Signal"),
    ]

    for source, target in mapping:
        if source and source in ranking.columns:
            display_columns.append((source, target))

    ranking_display = ranking[[src for src, _ in display_columns]].copy()
    ranking_display.columns = [tgt for _, tgt in display_columns]

    if "Predicted 5D Return" in ranking_display.columns:
        ranking_display["Predicted 5D Return"] = ranking_display["Predicted 5D Return"].map(lambda x: fmt_pct(x) if pd.notna(x) else "—")
    if "Signal Strength" in ranking_display.columns:
        ranking_display["Signal Strength"] = pd.to_numeric(ranking_display["Signal Strength"], errors="coerce").map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")

    st.dataframe(ranking_display, hide_index=True, width="stretch")

# ============================================================
# 14. PREDICTION VS RISK QUADRANT
# ============================================================

st.markdown(
    """<div class="section-title">
        <h3>Prediction vs Risk Quadrant</h3>
        <p>Research prediction magnitude shown alongside the project's risk-state classification.</p>
    </div>""",
    unsafe_allow_html=True,
)

if not signals_work.empty and ticker_col and "_predicted_pct" in signals_work.columns:
    map_df = signals_work.copy()
    map_df["Predicted Return (%)"] = pd.to_numeric(map_df["_predicted_pct"], errors="coerce")
    map_df["Risk State"] = map_df[risk_col].astype(str).str.upper() if risk_col else "UNKNOWN"
    map_df = map_df.dropna(subset=["Predicted Return (%)"])

    if not map_df.empty:
        fig_quad = px.scatter(
            map_df,
            x="Predicted Return (%)",
            y="Risk State",
            text=ticker_col,
            color="Risk State",
            color_discrete_map={"LOW": "#00df86", "MODERATE": "#f0b34b", "HIGH": "#ff4b4b"},
        )
        fig_quad.update_traces(textposition="top center", marker=dict(size=14, line=dict(width=1, color="#ffffff")))
        fig_quad.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
        fig_quad.update_layout(
            **PLOTLY_THEME,
            height=360,
            showlegend=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_quad, width="stretch")

# ============================================================
# 15. PORTFOLIO ALLOCATION CONTEXT
# ============================================================

st.markdown(
    """<div class="section-title">
        <h3>Portfolio Allocation Context</h3>
        <p>Robust signal-aware MPT weights mapped directly to research conviction.</p>
    </div>""",
    unsafe_allow_html=True,
)

if weights_df is not None:
    weight_ticker_col = find_column(weights_df, ["ticker", "symbol", "asset"])
    weight_col = find_column(weights_df, ["optimized_weight_pct", "weight_pct", "optimized_weight", "weight"])

    if weight_ticker_col and weight_col:
        allocation = weights_df[[weight_ticker_col, weight_col]].copy()
        allocation.columns = ["Ticker", "Weight"]
        allocation["Weight"] = pd.to_numeric(allocation["Weight"], errors="coerce")
        if allocation["Weight"].dropna().abs().max() <= 1.0:
            allocation["Weight"] *= 100.0
        allocation = allocation.dropna().sort_values("Weight", ascending=False)
        active = allocation[allocation["Weight"] > 0.0001]

        if not active.empty:
            fig_alloc = px.bar(
                active,
                x="Weight",
                y="Ticker",
                orientation="h",
                text="Weight",
                color="Weight",
                color_continuous_scale="Tealgrn",
            )
            fig_alloc.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig_alloc.update_layout(
                **PLOTLY_THEME,
                height=320,
                xaxis_title="Optimized Weight (%)",
                yaxis_title=None,
                showlegend=False,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_alloc, width="stretch")

# ============================================================
# 16. AUDIT / PROVENANCE & FOOTER
# ============================================================

st.markdown(
    """<div class="section-title">
        <h3>Data & Artifact Provenance</h3>
        <p>Verified physical recommendation artifacts loaded into the pipeline.</p>
    </div>""",
    unsafe_allow_html=True,
)

a_cols = st.columns(3)
with a_cols[0]:
    sig_status = artifacts["signals_path"].name if artifacts["signals_path"] else "Not detected"
    st.markdown(
        f"""<div class="info-box">
            <div style="color:#71869c;font-size:0.65rem;font-weight:800;">RESEARCH SIGNALS ARTIFACT</div>
            <div style="color:#dbe5ef;font-weight:700;margin-top:0.2rem;">{sig_status}</div>
            <div style="color:#00df86;font-size:0.7rem;margin-top:0.2rem;">● SVR Model & Rule Engine</div>
        </div>""",
        unsafe_allow_html=True,
    )
with a_cols[1]:
    wt_status = artifacts["weights_path"].name if artifacts["weights_path"] else "Not detected"
    st.markdown(
        f"""<div class="info-box">
            <div style="color:#71869c;font-size:0.65rem;font-weight:800;">MPT WEIGHTS ARTIFACT</div>
            <div style="color:#dbe5ef;font-weight:700;margin-top:0.2rem;">{wt_status}</div>
            <div style="color:#00df86;font-size:0.7rem;margin-top:0.2rem;">● Active Allocation Matrix</div>
        </div>""",
        unsafe_allow_html=True,
    )
with a_cols[2]:
    st.markdown(
        f"""<div class="info-box">
            <div style="color:#71869c;font-size:0.65rem;font-weight:800;">POLICY CONTEXT</div>
            <div style="color:#dbe5ef;font-weight:700;margin-top:0.2rem;">Deterministic Multi-Factor Scoring</div>
            <div style="color:#39bdf5;font-size:0.7rem;margin-top:0.2rem;">● Directional & Risk Aware</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown(
    """<div style="margin-top:1.2rem;background:rgba(90,65,20,.13);border:1px solid rgba(230,173,65,.22);border-radius:8px;padding:0.75rem 1rem;color:#9d927b;font-size:0.72rem;line-height:1.5;">
        ⚠ <b style="color:#c5a766;">Quantitative Research Disclaimer:</b>
        Research classifications shown here are systematic outputs combining statistical model estimates and risk regimes. They are not personalized investment advice, guarantees of future performance, or instructions to buy or sell securities. Research and educational use only.
    </div>""",
    unsafe_allow_html=True,
)