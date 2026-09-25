# NORTHGATE AI — PRICE & PREDICTION
# Quantitative Asset Research Terminal 

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Northgate AI — Price & Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. PROJECT PATHS & REPOSITORY STRUCTURE
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MODEL_DIR = RESULTS_DIR / "model_comparison"
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"

# ============================================================
# 3. UNIVERSE ASSETS
# ============================================================

PORTFOLIO_ASSETS = [
    "AAPL", "MSFT", "JPM", "XOM", "JNJ",
    "PG", "NVDA", "KO", "CAT", "HD",
]

# ============================================================
# 4. DARK TERMINAL THEME BASE (NO CONFLICTING PARAMETERS)
# ============================================================

PLOTLY_DARK = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {
        "color": "#cbd5e1",
        "family": "Inter, Arial, sans-serif",
    },
}

# ============================================================
# 5. GLOBAL STYLING (CSS)
# ============================================================

st.markdown(
    """<style>
    .stApp {
        background: radial-gradient(circle at 85% 10%, rgba(56,189,248,0.045), transparent 28%),
                    radial-gradient(circle at 10% 85%, rgba(168,85,247,0.035), transparent 25%),
                    #060b13;
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background: #070d18;
        border-right: 1px solid #172235;
    }

    h1, h2, h3, h4 {
        color: #f8fafc !important;
    }

    div[data-baseweb="select"] > div {
        background: #091322;
        border: 1px solid #1b3048;
        border-radius: 9px;
    }

    .ng-kpi {
        background: linear-gradient(145deg, rgba(15,23,42,0.96), rgba(9,19,34,0.96));
        border: 1px solid #17263a;
        border-radius: 12px;
        padding: 1rem 1rem 0.85rem 1rem;
        min-height: 118px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.16);
    }

    .ng-kpi-label {
        color: #64748b;
        font-size: 0.66rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .ng-kpi-value {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 850;
        margin-top: 0.3rem;
    }

    .ng-kpi-sub {
        color: #64748b;
        font-size: 0.68rem;
        margin-top: 0.25rem;
    }

    .ng-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.88), rgba(8,15,27,0.92));
        border: 1px solid #17263a;
        border-radius: 12px;
        padding: 1rem 1.05rem;
        box-shadow: 0 12px 30px rgba(0,0,0,0.13);
    }

    .ng-section-title {
        color: #f8fafc;
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: -0.01em;
    }

    .ng-section-sub {
        color: #64748b;
        font-size: 0.72rem;
        margin-top: 0.2rem;
    }

    .ng-badge {
        display: inline-block;
        padding: 0.35rem 0.65rem;
        border-radius: 7px;
        background: #091322;
        border: 1px solid #17304d;
        color: #38bdf8;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.05em;
    }

    .ng-metric {
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(148,163,184,0.08);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .ng-metric:last-child {
        border-bottom: none;
    }

    .ng-metric-label {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 700;
    }

    .ng-metric-value {
        color: #f8fafc;
        font-size: 0.88rem;
        font-weight: 800;
    }

    .ng-disclaimer {
        margin-top: 1.5rem;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        background: rgba(245,158,11,0.04);
        border: 1px solid rgba(245,158,11,0.15);
        color: #94a3b8;
        font-size: 0.72rem;
        line-height: 1.5;
    }
    </style>""",
    unsafe_allow_html=True,
)

# ============================================================
# 6. DATA HELPER FUNCTIONS
# ============================================================

@st.cache_data
def safe_read(path):
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()

def to_float(value, default=np.nan):
    try:
        value = pd.to_numeric(value, errors="coerce")
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default

def latest_row_for_ticker(df, ticker):
    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    temp = df[df["ticker"].astype(str).str.upper().eq(ticker.upper())].copy()
    if temp.empty:
        return pd.DataFrame()
    if "date" in temp.columns:
        temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
        temp = temp.sort_values("date", na_position="first")
    return temp.tail(1)

def load_price_history(ticker):
    if not FEATURES_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(FEATURES_PATH)
        if isinstance(df.columns, pd.MultiIndex):
            flattened = []
            for col in df.columns:
                parts = [str(part) for part in col if str(part).lower() != "nan"]
                flattened.append("_".join(parts))
            df.columns = flattened

        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})
            elif "date" not in df.columns:
                df = df.rename(columns={df.columns[0]: "date"})

        date_column = None
        for candidate in ["date", "Date", "datetime", "Datetime"]:
            if candidate in df.columns:
                date_column = candidate
                break

        if date_column is None:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df[date_column], errors="coerce")

        candidates = [
            f"Adj Close_{ticker}", f"Adj_Close_{ticker}",
            f"Adj Close {ticker}", f"Adj_Close {ticker}",
            f"Close_{ticker}", f"Close {ticker}",
        ]
        price_column = None
        for candidate in candidates:
            if candidate in df.columns:
                price_column = candidate
                break

        if price_column is None:
            ticker_upper = ticker.upper()
            for column in df.columns:
                column_text = str(column).upper()
                if ticker_upper in column_text and "ADJ" in column_text and "CLOSE" in column_text:
                    price_column = column
                    break

        if price_column is None:
            return pd.DataFrame()

        result = df[["date", price_column]].copy()
        result.columns = ["date", "price"]
        result["price"] = pd.to_numeric(result["price"], errors="coerce")
        result = result.dropna(subset=["date", "price"]).sort_values("date")
        result = result.drop_duplicates(subset=["date"], keep="last")

        # Rolling Quantitative Moving Averages
        result["ma_20"] = result["price"].rolling(window=20, min_periods=5).mean()
        result["ma_60"] = result["price"].rolling(window=60, min_periods=10).mean()
        return result
    except Exception:
        return pd.DataFrame()

# ============================================================
# 7. LOAD ARTIFACTS
# ============================================================

signals_df = safe_read(MODEL_DIR / "portfolio_research_signals.csv")
predictions_df = safe_read(MODEL_DIR / "portfolio_svr_latest_predictions.csv")
risk_df = safe_read(MODEL_DIR / "portfolio_risk_snapshot.csv")

# ============================================================
# 8. HEADER
# ============================================================

st.markdown(
    """<div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:1.25rem;">
        <div>
            <div style="color:#38bdf8; font-size:0.7rem; font-weight:800; letter-spacing:0.13em; text-transform:uppercase;">
                QUANTITATIVE ASSET TERMINAL
            </div>
            <div style="color:#f8fafc; font-size:2rem; font-weight:850; letter-spacing:-0.035em; margin-top:0.2rem;">
                Price & Prediction
            </div>
            <div style="color:#64748b; font-size:0.8rem; margin-top:0.3rem;">
                Historical market behaviour, model forecasts and quantitative risk signals
            </div>
        </div>
        <div class="ng-badge">SVR • 5-DAY HORIZON</div>
    </div>""",
    unsafe_allow_html=True,
)

# ============================================================
# 9. ASSET CONTROLS
# ============================================================

selector_col, horizon_col, model_col = st.columns([1.5, 0.9, 1.1], gap="medium")

with selector_col:
    ticker = st.selectbox(
        "Select Asset",
        PORTFOLIO_ASSETS,
        index=PORTFOLIO_ASSETS.index("AAPL") if "AAPL" in PORTFOLIO_ASSETS else 0,
        key="price_prediction_ticker",
    )

with horizon_col:
    st.markdown(
        """<div class="ng-card">
            <div class="ng-kpi-label">FORECAST HORIZON</div>
            <div style="color:#f8fafc; font-size:0.9rem; font-weight:800; margin-top:0.2rem;">5 Trading Days</div>
        </div>""",
        unsafe_allow_html=True,
    )

with model_col:
    st.markdown(
        """<div class="ng-card">
            <div class="ng-kpi-label">MODEL ENGINE</div>
            <div style="color:#00e676; font-size:0.9rem; font-weight:800; margin-top:0.2rem;">● SVR (Support Vector Regression)</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.write("")

# ============================================================
# 10. CURRENT SIGNAL & METRICS CALCULATION
# ============================================================

signal_row = latest_row_for_ticker(signals_df, ticker)
prediction_row = latest_row_for_ticker(predictions_df, ticker)
risk_row = latest_row_for_ticker(risk_df, ticker)

predicted_return = np.nan
predicted_return_pct = np.nan
direction = "N/A"
signal_magnitude = np.nan
research_signal = "N/A"
risk_state = "N/A"
momentum_state = "N/A"
return_20d = np.nan
return_60d = np.nan
return_252d = np.nan
current_dd = np.nan
max_dd = np.nan
beta = np.nan
signal_date = None

if not signal_row.empty:
    row = signal_row.iloc[0]
    if "predicted_5d_return" in row.index:
        predicted_return = to_float(row["predicted_5d_return"])
    if "predicted_5d_return_pct" in row.index:
        predicted_return_pct = to_float(row["predicted_5d_return_pct"])
    elif np.isfinite(predicted_return):
        predicted_return_pct = predicted_return * 100

    if "predicted_direction" in row.index:
        direction = str(row["predicted_direction"]).upper()
    if "prediction_strength" in row.index:
        signal_magnitude = to_float(row["prediction_strength"])
    elif np.isfinite(predicted_return):
        signal_magnitude = abs(predicted_return)

    if "research_signal" in row.index:
        research_signal = str(row["research_signal"]).replace("_", " ").title()

    if "risk_state" in row.index:
        risk_state = str(row["risk_state"]).upper()
    if "momentum_state" in row.index:
        momentum_state = str(row["momentum_state"]).upper()
    if "return_20d" in row.index:
        return_20d = to_float(row["return_20d"])
    if "return_60d" in row.index:
        return_60d = to_float(row["return_60d"])
    if "return_252d" in row.index:
        return_252d = to_float(row["return_252d"])
    if "current_drawdown" in row.index:
        current_dd = to_float(row["current_drawdown"])
    if "max_drawdown" in row.index:
        max_dd = to_float(row["max_drawdown"])
    if "beta_vs_sp500" in row.index:
        beta = to_float(row["beta_vs_sp500"])
    if "date" in row.index:
        signal_date = row["date"]

if not np.isfinite(predicted_return) and not prediction_row.empty:
    row = prediction_row.iloc[0]
    if "predicted_5d_return" in row.index:
        predicted_return = to_float(row["predicted_5d_return"])
        if np.isfinite(predicted_return):
            predicted_return_pct = predicted_return * 100

price_df = load_price_history(ticker)
latest_price = to_float(price_df.iloc[-1]["price"]) if not price_df.empty else np.nan

direction_positive = (predicted_return >= 0) if np.isfinite(predicted_return) else False
direction_color = "#00e676" if direction_positive else "#ef4444"
direction_symbol = "▲" if direction_positive else "▼"

risk_color_map = {"LOW": "#00e676", "MODERATE": "#f59e0b", "HIGH": "#ef4444"}
risk_color = risk_color_map.get(risk_state, "#94a3b8")

momentum_color = (
    "#00e676" if momentum_state == "POSITIVE"
    else "#ef4444" if momentum_state == "NEGATIVE"
    else "#f59e0b"
)

# Rigorous Decimal Formatting
forecast_display = f"{predicted_return_pct:+.2f}%" if np.isfinite(predicted_return_pct) else "N/A"

# ============================================================
# 11. 5 KPI CARDS
# ============================================================

k1, k2, k3, k4, k5 = st.columns(5, gap="medium")

with k1:
    price_text = f"${latest_price:,.2f}" if np.isfinite(latest_price) else "N/A"
    st.markdown(
        f"""<div class="ng-kpi" style="border-left:3px solid #38bdf8;">
            <div class="ng-kpi-label">LATEST PRICE</div>
            <div class="ng-kpi-value">{price_text}</div>
            <div class="ng-kpi-sub">Adjusted close • {ticker}</div>
        </div>""",
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""<div class="ng-kpi" style="border-left:3px solid {direction_color};">
            <div class="ng-kpi-label">PREDICTED 5D RETURN</div>
            <div class="ng-kpi-value" style="color:{direction_color};">{forecast_display}</div>
            <div class="ng-kpi-sub" style="color:{direction_color};">{direction_symbol} SVR FORECAST</div>
        </div>""",
        unsafe_allow_html=True,
    )

with k3:
    mag_sign = "+" if direction_positive else "-"
    mag_text = f"{mag_sign}{signal_magnitude:.3f}" if np.isfinite(signal_magnitude) else "N/A"
    st.markdown(
        f"""<div class="ng-kpi" style="border-left:3px solid #a855f7;">
            <div class="ng-kpi-label">SIGNAL MAGNITUDE</div>
            <div class="ng-kpi-value">{mag_text}</div>
            <div class="ng-kpi-sub" style="color:#a855f7;">Prediction magnitude</div>
        </div>""",
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""<div class="ng-kpi" style="border-left:3px solid {risk_color};">
            <div class="ng-kpi-label">RISK STATE</div>
            <div class="ng-kpi-value" style="color:{risk_color};">{risk_state}</div>
            <div class="ng-kpi-sub" style="color:{risk_color};">Volatility regime</div>
        </div>""",
        unsafe_allow_html=True,
    )

with k5:
    st.markdown(
        f"""<div class="ng-kpi" style="border-left:3px solid {momentum_color};">
            <div class="ng-kpi-label">MOMENTUM</div>
            <div class="ng-kpi-value" style="color:{momentum_color};">{momentum_state}</div>
            <div class="ng-kpi-sub">Trend regime</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.write("")

# ============================================================
# 12. RETURN & RISK SNAPSHOT
# ============================================================

st.markdown(
    """<div class="ng-card">
        <div class="ng-section-title">Return & Risk Snapshot</div>
        <div class="ng-section-sub">Quantitative context surrounding the current forecast</div>
    </div>""",
    unsafe_allow_html=True,
)

s1, s2, s3, s4, s5 = st.columns(5, gap="medium")
snapshot = [
    ("20D Return", return_20d, "#38bdf8"),
    ("60D Return", return_60d, "#06b6d4"),
    ("252D Return", return_252d, "#a855f7"),
    ("Current Drawdown", current_dd, "#f59e0b"),
    ("Beta vs S&P 500", beta, "#94a3b8"),
]

for column, (label, value, color) in zip([s1, s2, s3, s4, s5], snapshot):
    with column:
        if np.isfinite(value):
            text = f"{value:.2f}" if label.startswith("Beta") else f"{value * 100:+.2f}%"
        else:
            text = "N/A"
        st.markdown(
            f"""<div class="ng-card" style="margin-top:0.7rem;">
                <div class="ng-kpi-label">{label}</div>
                <div style="color:{color}; font-size:1.18rem; font-weight:850; margin-top:0.25rem;">{text}</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.write("")

# ============================================================
# 13. CHARTS: PRICE WITH MA & CLEAN FORECAST GAUGE
# ============================================================

chart_left, chart_right = st.columns([1.75, 1], gap="medium")

with chart_left:
    st.markdown(
        """<div class="ng-card">
            <div class="ng-section-title">Historical Price & Moving Averages</div>
            <div class="ng-section-sub">Adjusted close • 20D & 60D Trend Baselines</div>
        </div>""",
        unsafe_allow_html=True,
    )

    if not price_df.empty:
        fig_price = go.Figure()

        fig_price.add_trace(
            go.Scatter(
                x=price_df["date"],
                y=price_df["price"],
                mode="lines",
                name=f"{ticker} Close",
                line=dict(color="#38bdf8", width=2),
                fill="tozeroy",
                fillcolor="rgba(56,189,248,0.04)",
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Price: $%{y:,.2f}<extra></extra>",
            )
        )

        if "ma_20" in price_df.columns and price_df["ma_20"].notna().any():
            fig_price.add_trace(
                go.Scatter(
                    x=price_df["date"],
                    y=price_df["ma_20"],
                    mode="lines",
                    name="20D MA",
                    line=dict(color="#00e676", width=1.5, dash="dash"),
                    hovertemplate="20D MA: $%{y:,.2f}<extra></extra>",
                )
            )

        if "ma_60" in price_df.columns and price_df["ma_60"].notna().any():
            fig_price.add_trace(
                go.Scatter(
                    x=price_df["date"],
                    y=price_df["ma_60"],
                    mode="lines",
                    name="60D MA",
                    line=dict(color="#f59e0b", width=1.5, dash="dot"),
                    hovertemplate="60D MA: $%{y:,.2f}<extra></extra>",
                )
            )

        fig_price.update_layout(
            **PLOTLY_DARK,
            height=420,
            margin=dict(l=45, r=20, t=25, b=45),
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                y=1.08,
                x=0,
                font=dict(size=10, color="#94a3b8"),
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis_title=None,
            yaxis_title="Price ($)",
        )
        fig_price.update_xaxes(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.12)")
        fig_price.update_yaxes(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.12)")
        st.plotly_chart(fig_price, width="stretch", config={"displaylogo": False, "scrollZoom": False})
    else:
        st.warning(f"Historical price data for {ticker} could not be loaded from features.parquet.")

with chart_right:
    st.markdown(
        """<div class="ng-card">
            <div class="ng-section-title">5-Day Forecast Gauge</div>
            <div class="ng-section-sub">Centered quantitative threshold calibration</div>
        </div>""",
        unsafe_allow_html=True,
    )

    if np.isfinite(predicted_return_pct):
        fig_gauge = go.Figure()

        # Clean Linear Indicator with Strict Boundaries (-3% to +3%)
        fig_gauge.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=predicted_return_pct,
                number={
                    "valueformat": "+.2f",
                    "suffix": "%",
                    "font": {"size": 42, "color": direction_color, "family": "Inter, sans-serif"},
                },
                gauge={
                    "shape": "bullet",
                    "axis": {
                        "range": [-3.0, 3.0],
                        "tickvals": [-3.0, -1.5, 0, 1.5, 3.0],
                        "ticktext": ["-3%", "-1.5%", "0%", "+1.5%", "+3%"],
                        "tickcolor": "#64748b",
                        "tickwidth": 1,
                        "tickfont": {"size": 10, "color": "#64748b"},
                    },
                    "bar": {"color": direction_color, "thickness": 0.45},
                    "bgcolor": "rgba(15,23,42,0.8)",
                    "bordercolor": "rgba(255,255,255,0.08)",
                    "steps": [
                        {"range": [-3.0, 0], "color": "rgba(239,68,68,0.10)"},
                        {"range": [0, 3.0], "color": "rgba(0,230,118,0.10)"},
                    ],
                    "threshold": {
                        "line": {"color": "#ffffff", "width": 2},
                        "thickness": 0.75,
                        "value": predicted_return_pct,
                    },
                },
            )
        )

        fig_gauge.update_layout(
            **PLOTLY_DARK,
            height=210,
            margin=dict(l=35, r=35, t=40, b=20),
        )
        st.plotly_chart(fig_gauge, width="stretch", config={"displaylogo": False})

        st.markdown(
            f"""<div class="ng-card" style="margin-top:0.6rem; padding:0.85rem 1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.8rem;">
                    <span style="color:#64748b;">Direction:</span>
                    <b style="color:{direction_color};">{direction}</b>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.8rem; margin-top:0.45rem;">
                    <span style="color:#64748b;">Horizon:</span>
                    <span style="color:#f8fafc; font-weight:700;">5 Trading Days</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.info(f"No valid 5-day SVR prediction is available for {ticker}.")

# ============================================================
# 14. RESEARCH INTERPRETATION, RISK MONITOR & MODEL CONTEXT
# ============================================================

st.write("")
st.markdown(
    """<div class="ng-card">
        <div class="ng-section-title">Quantitative Signal & Risk Interpretation</div>
        <div class="ng-section-sub">Policy classification, volatility controls and model execution telemetry</div>
    </div>""",
    unsafe_allow_html=True,
)

interpretation_col, risk_col, context_col = st.columns([1.3, 1, 1], gap="medium")

with interpretation_col:
    signal_text = (
        "The SVR model estimates a positive 5-day return."
        if direction_positive else
        "The SVR model estimates a non-positive 5-day return."
    )
    if not np.isfinite(predicted_return):
        signal_text = "A valid 5-day model forecast is not currently available."

    if momentum_state == "POSITIVE":
        momentum_desc = "Momentum regime is classified as positive based on the project's lookback signals."
    elif momentum_state == "NEGATIVE":
        momentum_desc = "Momentum regime is classified as negative based on the project's lookback signals."
    else:
        momentum_desc = "Momentum regime remains unclassified within neutral regime boundaries."

    st.markdown(
        f"""<div class="ng-card" style="height:100%;">
            <div style="color:#f8fafc; font-size:0.95rem; font-weight:800; margin-bottom:0.75rem;">
                {ticker} Research Signal
            </div>
            <div style="display:flex; gap:0.5rem; margin-bottom:0.85rem;">
                <div style="background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.3); padding:0.25rem 0.6rem; border-radius:6px; font-size:0.72rem; color:#38bdf8; font-weight:700;">
                    POLICY: {research_signal}
                </div>
                <div style="background:{'rgba(0,230,118,0.12)' if direction_positive else 'rgba(239,68,68,0.12)'}; border:1px solid {direction_color}; padding:0.25rem 0.6rem; border-radius:6px; font-size:0.72rem; color:{direction_color}; font-weight:700;">
                    DIRECTION: {direction}
                </div>
            </div>
            <div style="color:#94a3b8; font-size:0.8rem; line-height:1.7;">
                {signal_text}
                <br><br>
                {momentum_desc}
                <br><br>
                Model forecast: <b style="color:#f8fafc;">{forecast_display}</b> over the next 5 trading days.
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

with risk_col:
    max_dd_display = f"{max_dd * 100:+.2f}%" if np.isfinite(max_dd) else "N/A"
    beta_display = f"{beta:.2f}" if np.isfinite(beta) else "N/A"

    st.markdown(
        f"""<div class="ng-card" style="height:100%;">
            <div style="color:#f8fafc; font-size:0.95rem; font-weight:800; margin-bottom:0.7rem;">Risk Monitor</div>
            <div class="ng-metric">
                <span class="ng-metric-label">RISK REGIME</span>
                <span class="ng-metric-value" style="color:{risk_color};">{risk_state}</span>
            </div>
            <div class="ng-metric">
                <span class="ng-metric-label">MAX DRAWDOWN</span>
                <span class="ng-metric-value" style="color:#f59e0b;">{max_dd_display}</span>
            </div>
            <div class="ng-metric">
                <span class="ng-metric-label">BETA VS S&P 500</span>
                <span class="ng-metric-value">{beta_display}</span>
            </div>
            <div class="ng-metric">
                <span class="ng-metric-label">MOMENTUM REGIME</span>
                <span class="ng-metric-value" style="color:{momentum_color};">{momentum_state}</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

with context_col:
    if signal_date is not None:
        try:
            display_date = pd.to_datetime(signal_date).strftime("%d %b %Y")
        except Exception:
            display_date = str(signal_date)
    else:
        display_date = "04 Sep 2026"

    st.markdown(
        f"""<div class="ng-card" style="height:100%;">
            <div style="color:#f8fafc; font-size:0.95rem; font-weight:800; margin-bottom:0.7rem;">Model Context</div>
            <div class="ng-metric">
                <span class="ng-metric-label">CORE ARCHITECTURE</span>
                <span class="ng-metric-value" style="color:#38bdf8;">SVR (RBF Kernel)</span>
            </div>
            <div class="ng-metric">
                <span class="ng-metric-label">FORECAST HORIZON</span>
                <span class="ng-metric-value">5 Trading Days</span>
            </div>
            <div class="ng-metric">
                <span class="ng-metric-label">ARTIFACT TIMESTAMP</span>
                <span class="ng-metric-value" style="color:#a855f7;">{display_date}</span>
            </div>
            <div class="ng-metric">
                <span class="ng-metric-label">ESTIMATION MAGNITUDE</span>
                <span class="ng-metric-value">{mag_text}</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

# ============================================================
# 15. ONE-LINE COMPACT DISCLAIMER
# ============================================================

st.write("")
st.markdown(
    """<div class="ng-disclaimer">
        ⚠ <b style="color:#f59e0b;">Quantitative Research Disclaimer:</b>
        Model forecasts and research classifications are statistical estimates and do not guarantee future performance. Research & educational use only. Not financial advice.
    </div>""",
    unsafe_allow_html=True,
)