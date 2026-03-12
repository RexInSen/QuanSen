"""
============================================================
  QuanSen — Streamlit GUI
  Wraps the Quantitative Portfolio Optimizer (v1.0) engine
  by Amatra Sen without modifying the engine.
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import yfinance as _yf

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="QuanSen · Portfolio Optimizer",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import engine (unchanged) ─────────────────────────────────
from portfolio_tool import (
    get_best_ticker,
    load_data,
    utility_portfolio,
    compute_frontier,
    tangency_portfolio,
    min_risk_portfolio,
    RF_ANNUAL,
)


def load_data(tickers, start_date, end_date):
    data = _yf.download(tickers, start=start_date, end=end_date,
                        auto_adjust=True, progress=False)
    
    # Flatten MultiIndex columns (new yfinance versions)
    prices = data["Close"]
    
    # Single ticker returns Series not DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    
    # Drop missing tickers
    tickers = [t for t in tickers if t in prices.columns]
    
    # ← THIS IS THE FIX: re-align to consistent order
    prices = prices[sorted(tickers)]
    tickers = list(prices.columns)
    
    returns = prices.pct_change().dropna()
    expected_returns = returns.mean()
    cov_raw = returns.cov()
    cov_matrix = (cov_raw + cov_raw.T) / 2
    
    expected_returns = expected_returns[tickers]
    cov_matrix = cov_matrix.loc[tickers, tickers]
    
    return tickers, returns, expected_returns, cov_matrix
# ══════════════════════════════════════════════════════════════
# GLOBAL STYLES
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Base reset ── */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0a0e17;
    color: #dce6f0;
}

/* ── Main container ── */
.main .block-container {
    background: #0a0e17;
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid #1c2840;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.2rem;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0a0e17 0%, #0d1a35 50%, #091429 100%);
    border: 1px solid #1c3a6e;
    border-radius: 6px;
    padding: 2.2rem 2.5rem 1.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,180,255,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #e8f4ff;
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
}
.hero-title span { color: #00b4ff; }
.hero-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #4a6a90;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,180,255,0.1);
    border: 1px solid rgba(0,180,255,0.25);
    color: #00b4ff;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 3px;
    margin-top: 0.8rem;
}

/* ── Cards ── */
.card {
    background: #0d1525;
    border: 1px solid #1a2d4d;
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00b4ff;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #1a2d4d;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }
.metric-box {
    flex: 1;
    min-width: 130px;
    background: #0a1428;
    border: 1px solid #1a2d4d;
    border-radius: 5px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-box .label {
    font-size: 0.66rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a6a90;
    margin-bottom: 0.4rem;
}
.metric-box .value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8f4ff;
    line-height: 1;
}
.metric-box .value.positive { color: #00e676; }
.metric-box .value.accent   { color: #00b4ff; }
.metric-box .value.gold     { color: #ffd54f; }

/* ── Section heading ── */
.section-heading {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #a8c8e8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-left: 3px solid #00b4ff;
    padding-left: 0.7rem;
    margin: 1.8rem 0 1rem 0;
}

/* ── Tables ── */
.quansen-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.quansen-table th {
    background: #0d1a35;
    color: #4a90d9;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.6rem 0.9rem;
    text-align: left;
    border-bottom: 1px solid #1a2d4d;
}
.quansen-table td {
    padding: 0.55rem 0.9rem;
    border-bottom: 1px solid #111e33;
    color: #c8ddf0;
    font-family: 'DM Mono', monospace;
}
.quansen-table tr:last-child td { border-bottom: none; }
.quansen-table tr:hover td { background: rgba(0,180,255,0.03); }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0066cc, #0044aa) !important;
    color: #e8f4ff !important;
    border: 1px solid #0055bb !important;
    border-radius: 4px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0077ee, #0055cc) !important;
    border-color: #0077ee !important;
    box-shadow: 0 0 16px rgba(0,120,255,0.25) !important;
}

/* ── Input fields ── */
.stTextInput input, .stNumberInput input, .stDateInput input {
    background: #0a1428 !important;
    color: #c8ddf0 !important;
    border: 1px solid #1a3050 !important;
    border-radius: 4px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #0066cc !important;
    box-shadow: 0 0 0 2px rgba(0,102,204,0.2) !important;
}

/* ── Select boxes ── */
.stSelectbox > div > div {
    background: #0a1428 !important;
    border: 1px solid #1a3050 !important;
    color: #c8ddf0 !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Slider ── */
.stSlider > div { padding: 0.2rem 0; }

/* ── Status/Alert boxes ── */
.status-box {
    border-radius: 4px;
    padding: 0.8rem 1.1rem;
    margin: 0.6rem 0;
    font-size: 0.82rem;
    border-left: 3px solid;
}
.status-success { background: rgba(0,230,118,0.06); border-color: #00e676; color: #80ffc0; }
.status-info    { background: rgba(0,180,255,0.06); border-color: #00b4ff; color: #80d8ff; }
.status-warn    { background: rgba(255,213,79,0.06); border-color: #ffd54f; color: #ffe082; }
.status-error   { background: rgba(255,82,82,0.06);  border-color: #ff5252; color: #ff8a80; }

/* ── Ticker chip ── */
.ticker-chip {
    display: inline-block;
    background: rgba(0,102,204,0.15);
    border: 1px solid rgba(0,102,204,0.4);
    color: #80c8ff;
    border-radius: 3px;
    padding: 2px 9px;
    font-size: 0.75rem;
    margin: 2px 3px;
    font-family: 'DM Mono', monospace;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #4a90d9 !important;
    background: #0d1525 !important;
    border: 1px solid #1a2d4d !important;
    border-radius: 4px !important;
}

/* ── Divider ── */
hr { border-color: #1a2d4d !important; margin: 1.5rem 0 !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: #00b4ff !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1525;
    border-bottom: 1px solid #1a2d4d;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #4a6a90 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.2rem !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #00b4ff !important;
    border-bottom-color: #00b4ff !important;
}

/* ── DataFrame ── */
.dataframe { font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important; }

/* ── Weight bar fill ── */
.weight-bar-bg {
    background: #111e33;
    border-radius: 2px;
    height: 6px;
    width: 100%;
    margin-top: 4px;
}
.weight-bar-fill {
    height: 6px;
    border-radius: 2px;
    background: linear-gradient(90deg, #0066cc, #00b4ff);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "tickers": [],
        "start_date": "2021-01-01",
        "end_date": "2025-12-31",
        "returns": None,
        "expected_returns": None,
        "cov_matrix": None,
        "weights_utility": None,
        "weights_tan": None,
        "tan_return": None,
        "tan_risk": None,
        "tan_sharpe": None,
        "weights_min": None,
        "frontier_risks": None,
        "frontier_returns": None,
        "asset_returns": None,
        "asset_risks": None,
        "min_w": 0.01,
        "max_w": 0.20,
        "data_loaded": False,
        "portfolios_computed": False,
        "frontier_computed": False,
        "search_results": [],
        "target_return": 0.15,
        "rf_override": RF_ANNUAL,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def search_ticker_api(query: str):
    """Return list of {shortname, symbol} from Yahoo Finance search."""
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json().get("quotes", [])[:10]
    except Exception as e:
        return []


def compute_max_w(n: int) -> float:
    if n <= 3:   return 0.40
    if n <= 10:  return 0.30
    if n <= 20:  return 0.20
    if n <= 40:  return 0.15
    return 0.10


def metric_card(label, value, style=""):
    return f"""
    <div class="metric-box">
        <div class="label">{label}</div>
        <div class="value {style}">{value}</div>
    </div>"""


def weight_bar(pct: float, color="#00b4ff"):
    return f"""
    <div class="weight-bar-bg">
        <div class="weight-bar-fill" style="width:{min(pct,100):.1f}%;background:linear-gradient(90deg,#0066cc,{color});"></div>
    </div>"""


def make_weights_table(tickers, weights, label_col="Weight"):
    rows = ""
    for t, w in zip(tickers, weights):
        pct = w * 100
        rows += f"""<tr>
            <td><span class="ticker-chip">{t}</span></td>
            <td style="font-family:'DM Mono'">{w:.4f}</td>
            <td style="font-family:'DM Mono';color:#00e676">{pct:.2f}%</td>
            <td style="min-width:140px">{weight_bar(pct)}</td>
        </tr>"""
    return f"""
    <table class="quansen-table">
        <thead><tr>
            <th>Ticker</th>
            <th>{label_col}</th>
            <th>Allocation %</th>
            <th>Visual</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def portfolio_metrics_row(ann_ret, ann_risk, sharpe):
    ret_class  = "positive" if ann_ret > 0 else ""
    shp_class  = "accent" if sharpe > 1 else ("positive" if sharpe > 0 else "")
    st.markdown(f"""
    <div class="metric-row">
        {metric_card("Annual Return",  f"{ann_ret*100:.2f}%",   ret_class)}
        {metric_card("Annual Risk",    f"{ann_risk*100:.2f}%",  "")}
        {metric_card("Sharpe Ratio",   f"{sharpe:.3f}",          shp_class)}
        {metric_card("Risk-Free Rate", f"{RF_ANNUAL*100:.1f}%", "accent")}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.5rem">
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:#e8f4ff;letter-spacing:-0.01em">
            QUAN<span style="color:#00b4ff">SEN</span>
        </div>
        <div style="font-size:0.62rem;color:#4a6a90;letter-spacing:0.15em;text-transform:uppercase;margin-top:2px">
            Portfolio Optimizer v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card-title">⚙ Configuration</div>', unsafe_allow_html=True)

    # ── Pre-load test portfolio ──
    if st.button("Load Test Portfolio (20 stocks)", use_container_width=True):
        st.session_state.tickers = [
            'BEL.NS','VEDL.NS','BAJFINANCE.NS','BEML.NS','ADVAIT.BO',
            'ADANIENT.NS','COALINDIA.NS','CROMPTON.NS','KINGFA.NS','KRISHNADEF.NS',
            'LT.NS','LUPIN.NS','MAZDOCK.NS','PENIND.NS','PNB.NS',
            'RELIANCE.NS','SHUKRAPHAR.BO','TARIL.NS','HDFCBANK.NS','SBIN.NS'
        ]
        st.session_state.start_date = "2021-02-01"
        st.session_state.end_date   = "2026-03-06"
        st.session_state.data_loaded = False
        st.session_state.portfolios_computed = False
        st.session_state.frontier_computed = False
        st.rerun()

    st.markdown("---")

    # ── Date range ──
    st.markdown('<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem">Date Range</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.start_date = st.text_input("Start", value=st.session_state.start_date, label_visibility="collapsed", placeholder="YYYY-MM-DD")
    with col2:
        st.session_state.end_date = st.text_input("End", value=st.session_state.end_date, label_visibility="collapsed", placeholder="YYYY-MM-DD")

    # ── Constraints ──
    st.markdown('<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;margin:0.8rem 0 0.4rem">Weight Constraints</div>', unsafe_allow_html=True)
    n_t = max(len(st.session_state.tickers), 1)
    auto_max = compute_max_w(n_t)
    st.caption(f"Auto max-weight for {n_t} assets: {auto_max*100:.0f}%")
    
    min_w_pct = st.slider("Min weight %", 0, 10, int(st.session_state.min_w * 100), 1)
    max_w_pct = st.slider("Max weight %", 5, 50, int(auto_max * 100), 5)
    st.session_state.min_w = min_w_pct / 100
    st.session_state.max_w = max_w_pct / 100

    # ── Target return ──
    st.markdown('<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;margin:0.8rem 0 0.4rem">Min-Risk Target Return</div>', unsafe_allow_html=True)
    target_pct = st.slider("Target return %", 1, 60, int(st.session_state.target_return * 100), 1)
    st.session_state.target_return = target_pct / 100

    st.markdown("---")

    # ── Current tickers display ──
    if st.session_state.tickers:
        st.markdown('<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem">Selected Tickers</div>', unsafe_allow_html=True)
        chips = "".join([f'<span class="ticker-chip">{t}</span>' for t in st.session_state.tickers])
        st.markdown(chips, unsafe_allow_html=True)
        st.caption(f"{len(st.session_state.tickers)} stocks selected")

        if st.button("🗑 Clear All Tickers", use_container_width=True):
            st.session_state.tickers = []
            st.session_state.data_loaded = False
            st.session_state.portfolios_computed = False
            st.session_state.frontier_computed = False
            st.rerun()

    st.markdown("---")
    st.markdown('<div style="font-size:0.65rem;color:#263a56;text-align:center;margin-top:1rem">MPT · CVXPY · SciPy<br>Amatra Sen — QuanSen v1.0</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">QUAN<span>SEN</span></div>
    <div class="hero-subtitle">Quantitative Portfolio Optimizer · MPT Engine</div>
    <div class="hero-badge">Efficient Frontier · Tangency Portfolio · CML · Min-Risk Solver</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab_build, tab_data, tab_portfolios, tab_frontier, tab_analysis, tab_export = st.tabs([
    "📋 Build Portfolio",
    "📊 Market Data",
    "🏆 Portfolios",
    "📈 Efficient Frontier",
    "🔬 Analysis",
    "💾 Export",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — BUILD PORTFOLIO
# ══════════════════════════════════════════════════════════════
with tab_build:
    col_search, col_manual = st.columns([1.1, 1])

    # ── Ticker search ──────────────────────────────────────────
    with col_search:
        st.markdown('<div class="section-heading">Search & Add Tickers</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        search_query = st.text_input("Company name", placeholder="e.g. Reliance Industries", key="search_input")
        search_btn = st.button("🔍 Search", key="search_btn")

        if search_btn and search_query.strip():
            with st.spinner("Querying Yahoo Finance..."):
                results = search_ticker_api(search_query.strip())
                st.session_state.search_results = results

        if st.session_state.search_results:
            st.markdown('<div style="font-size:0.7rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;margin:0.8rem 0 0.4rem">Results</div>', unsafe_allow_html=True)
            for q in st.session_state.search_results:
                name   = q.get('shortname', q.get('longname', 'N/A'))
                symbol = q.get('symbol', '')
                exch   = q.get('exchDisp', '')
                if not symbol:
                    continue
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"""
                    <div style="padding:0.35rem 0">
                        <span style="color:#e8f4ff;font-size:0.82rem">{name}</span><br>
                        <span class="ticker-chip">{symbol}</span>
                        <span style="font-size:0.68rem;color:#4a6a90;margin-left:5px">{exch}</span>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    if st.button("Add", key=f"add_{symbol}_{name[:6]}"):
                        if symbol not in st.session_state.tickers:
                            # Resolve best exchange for Indian stocks
                            with st.spinner(f"Resolving {symbol}..."):
                                best = get_best_ticker(
                                    symbol,
                                    st.session_state.start_date,
                                    st.session_state.end_date
                                )
                            st.session_state.tickers.append(best)
                            st.session_state.data_loaded = False
                            st.session_state.portfolios_computed = False
                            st.session_state.frontier_computed = False
                            st.markdown(f'<div class="status-box status-success">✔ Added {best}</div>', unsafe_allow_html=True)
                            st.rerun()
                        else:
                            st.markdown(f'<div class="status-box status-warn">Already in portfolio.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Manual entry + remove ──────────────────────────────────
    with col_manual:
        st.markdown('<div class="section-heading">Manual Entry & Manage</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        manual_ticker = st.text_input("Enter ticker directly", placeholder="e.g. AAPL, TCS.NS", key="manual_input")
        if st.button("➕ Add Manually", key="manual_btn"):
            t = manual_ticker.strip().upper()
            if t:
                if t not in st.session_state.tickers:
                    with st.spinner(f"Resolving {t}..."):
                        best = get_best_ticker(t, st.session_state.start_date, st.session_state.end_date)
                    st.session_state.tickers.append(best)
                    st.session_state.data_loaded = False
                    st.session_state.portfolios_computed = False
                    st.session_state.frontier_computed = False
                    st.markdown(f'<div class="status-box status-success">✔ Added {best}</div>', unsafe_allow_html=True)
                    st.rerun()
                else:
                    st.markdown(f'<div class="status-box status-warn">Already in portfolio.</div>', unsafe_allow_html=True)

        st.markdown("---")

        if st.session_state.tickers:
            st.markdown('<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem">Current Portfolio</div>', unsafe_allow_html=True)
            for i, t in enumerate(st.session_state.tickers):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f'<span class="ticker-chip">{t}</span>', unsafe_allow_html=True)
                with c2:
                    if st.button("✕", key=f"rm_{t}_{i}"):
                        st.session_state.tickers.remove(t)
                        st.session_state.data_loaded = False
                        st.session_state.portfolios_computed = False
                        st.session_state.frontier_computed = False
                        st.rerun()
        else:
            st.markdown('<div class="status-box status-info">No tickers yet. Search or add manually.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Load Data button ───────────────────────────────────────
    st.markdown('<div class="section-heading">Step 1 — Load Market Data</div>', unsafe_allow_html=True)

    if len(st.session_state.tickers) < 2:
        st.markdown('<div class="status-box status-warn">⚠ Add at least 2 tickers to load data.</div>', unsafe_allow_html=True)
    else:
        if st.button("⬇ Download & Load Data", use_container_width=True):
            with st.spinner("Downloading price data from Yahoo Finance…"):
                try:
                    (tickers_out,
                     returns,
                     expected_returns,
                     cov_matrix) = load_data(
                        st.session_state.tickers,
                        st.session_state.start_date,
                        st.session_state.end_date
                    )
                    st.session_state.tickers          = tickers_out
                    st.session_state.returns          = returns
                    st.session_state.expected_returns = expected_returns
                    st.session_state.cov_matrix       = cov_matrix
                    st.session_state.asset_returns    = expected_returns * 252
                    st.session_state.asset_risks      = returns.std() * np.sqrt(252)
                    st.session_state.data_loaded      = True
                    st.session_state.portfolios_computed = False
                    st.session_state.frontier_computed   = False
                    st.success(f"✔ Loaded {len(returns)} trading days for {len(tickers_out)} assets.")
                except Exception as e:
                    st.error(f"Data load failed: {e}")

    # ── Step 2: Run optimizations ──────────────────────────────
    if st.session_state.data_loaded:
        st.markdown('<div class="section-heading">Step 2 — Run Optimizations</div>', unsafe_allow_html=True)
        run_col1, run_col2, run_col3 = st.columns(3)

        with run_col1:
            if st.button("⚡ Utility Portfolio", use_container_width=True):
                with st.spinner("Optimizing utility portfolio…"):
                    try:
                        w = utility_portfolio(
                            st.session_state.expected_returns,
                            st.session_state.cov_matrix,
                            st.session_state.tickers,
                            st.session_state.min_w,
                            st.session_state.max_w
                        )
                        st.session_state.weights_utility = w
                        st.success("Utility portfolio computed.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with run_col2:
            if st.button("🌟 Tangency Portfolio", use_container_width=True):
                with st.spinner("Maximising Sharpe ratio…"):
                    try:
                        w, tr, tk, ts = tangency_portfolio(
                            st.session_state.expected_returns,
                            st.session_state.cov_matrix,
                            st.session_state.tickers,
                            st.session_state.min_w,
                            st.session_state.max_w
                        )
                        st.session_state.weights_tan  = w
                        st.session_state.tan_return   = tr
                        st.session_state.tan_risk     = tk
                        st.session_state.tan_sharpe   = ts
                        st.success("Tangency portfolio computed.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with run_col3:
            if st.button("🎯 Min-Risk Portfolio", use_container_width=True):
                with st.spinner("Minimising risk for target return…"):
                    try:
                        w = min_risk_portfolio(
                            st.session_state.expected_returns,
                            st.session_state.cov_matrix,
                            st.session_state.tickers,
                            st.session_state.target_return,
                            st.session_state.min_w,
                            st.session_state.max_w
                        )
                        st.session_state.weights_min = w
                        if w is not None:
                            st.success(f"Min-risk portfolio computed (target: {st.session_state.target_return*100:.1f}%).")
                        else:
                            st.warning("No feasible solution for this target. Try a lower target return.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        run_all = st.button("🚀 Run ALL Optimizations + Frontier", use_container_width=True)
        if run_all:
            progress = st.progress(0, text="Running all optimizations…")
            try:
                progress.progress(10, "Utility portfolio…")
                w_u = utility_portfolio(
                    st.session_state.expected_returns, st.session_state.cov_matrix,
                    st.session_state.tickers, st.session_state.min_w, st.session_state.max_w)
                st.session_state.weights_utility = w_u
                progress.progress(35, "Tangency portfolio…")
                w_t, tr, tk, ts = tangency_portfolio(
                    st.session_state.expected_returns, st.session_state.cov_matrix,
                    st.session_state.tickers, st.session_state.min_w, st.session_state.max_w)
                st.session_state.weights_tan  = w_t
                st.session_state.tan_return   = tr
                st.session_state.tan_risk     = tk
                st.session_state.tan_sharpe   = ts
                progress.progress(60, "Efficient frontier…")
                fr, ret = compute_frontier(
                    st.session_state.expected_returns, st.session_state.cov_matrix,
                    st.session_state.min_w, st.session_state.max_w)
                st.session_state.frontier_risks    = fr
                st.session_state.frontier_returns  = ret
                st.session_state.frontier_computed = True
                progress.progress(85, "Min-risk portfolio…")
                w_m = min_risk_portfolio(
                    st.session_state.expected_returns, st.session_state.cov_matrix,
                    st.session_state.tickers, st.session_state.target_return,
                    st.session_state.min_w, st.session_state.max_w)
                st.session_state.weights_min = w_m
                st.session_state.portfolios_computed = True
                progress.progress(100, "Done ✔")
                st.success("All optimizations complete. Navigate the tabs above.")
            except Exception as e:
                st.error(f"Error during run: {e}")

        # ── Frontier separately ────────────────────────────────
        if st.button("📉 Compute Efficient Frontier Only", use_container_width=True):
            with st.spinner("Computing efficient frontier (100 points)…"):
                try:
                    fr, ret = compute_frontier(
                        st.session_state.expected_returns,
                        st.session_state.cov_matrix,
                        st.session_state.min_w,
                        st.session_state.max_w
                    )
                    st.session_state.frontier_risks   = fr
                    st.session_state.frontier_returns = ret
                    st.session_state.frontier_computed = True
                    st.success(f"Frontier computed: {len(fr)} feasible points.")
                except Exception as e:
                    st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════
# TAB 2 — MARKET DATA
# ══════════════════════════════════════════════════════════════
with tab_data:
    if not st.session_state.data_loaded:
        st.markdown('<div class="status-box status-info">Load data first via the Build Portfolio tab.</div>', unsafe_allow_html=True)
    else:
        returns          = st.session_state.returns
        expected_returns = st.session_state.expected_returns
        asset_returns    = st.session_state.asset_returns
        asset_risks      = st.session_state.asset_risks
        tickers          = st.session_state.tickers

        # ── Summary table ──────────────────────────────────────
        st.markdown('<div class="section-heading">Asset Summary</div>', unsafe_allow_html=True)
        summary_df = pd.DataFrame({
            "Ticker"         : tickers,
            "Ann. Return %"  : (asset_returns.values * 100).round(2),
            "Ann. Risk %"    : (asset_risks.values   * 100).round(2),
            "Sharpe"         : ((asset_returns.values - RF_ANNUAL) / asset_risks.values).round(3),
            "Daily Mean %"   : (expected_returns.values * 100).round(4),
        })
        summary_df = summary_df.set_index("Ticker")
        st.dataframe(summary_df.style
            .background_gradient(cmap="Blues", subset=["Ann. Return %"])
            .background_gradient(cmap="Reds",  subset=["Ann. Risk %"])
            .background_gradient(cmap="Greens",subset=["Sharpe"]),
            use_container_width=True)

        # ── Correlation heatmap ────────────────────────────────
        st.markdown('<div class="section-heading">Correlation Matrix</div>', unsafe_allow_html=True)
        corr = returns.corr()

        fig_corr, ax = plt.subplots(figsize=(max(6, len(tickers)*0.7), max(5, len(tickers)*0.6)))
        fig_corr.patch.set_facecolor("#0a0e17")
        ax.set_facecolor("#0a0e17")
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f",
                    linewidths=0.5, linecolor="#1a2d4d",
                    annot_kws={"size": max(6, 9-len(tickers)//3), "color": "white"},
                    ax=ax)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        plt.xticks(rotation=45, ha="left", color="#a0b8d0", fontsize=max(6, 9-len(tickers)//4))
        plt.yticks(color="#a0b8d0", fontsize=max(6, 9-len(tickers)//4))
        ax.tick_params(colors="#4a6a90")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1a2d4d")
        plt.title("Asset Correlation Matrix", color="#80b0d0", fontsize=11, pad=20)
        plt.tight_layout()
        st.pyplot(fig_corr)

        # ── Returns distribution ───────────────────────────────
        st.markdown('<div class="section-heading">Return Distributions</div>', unsafe_allow_html=True)
        n_cols = min(4, len(tickers))
        n_rows = (len(tickers) + n_cols - 1) // n_cols
        fig_dist, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.8))
        fig_dist.patch.set_facecolor("#0a0e17")
        axes_flat = np.array(axes).flatten() if len(tickers) > 1 else [axes]
        for idx, t in enumerate(tickers):
            ax2 = axes_flat[idx]
            ax2.set_facecolor("#0d1525")
            ax2.hist(returns[t].dropna(), bins=40, color="#0066cc", alpha=0.75, edgecolor="#003388")
            ax2.axvline(returns[t].mean(), color="#00e676", linestyle="--", linewidth=1.2, label="Mean")
            ax2.set_title(t, color="#80c8ff", fontsize=8, pad=4)
            ax2.tick_params(colors="#4a6a90", labelsize=6)
            for spine in ax2.spines.values():
                spine.set_edgecolor("#1a2d4d")
        for idx in range(len(tickers), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        plt.suptitle("Daily Return Distributions", color="#80b0d0", fontsize=11, y=1.01)
        plt.tight_layout()
        st.pyplot(fig_dist)

        # ── Risk-Return scatter ────────────────────────────────
        st.markdown('<div class="section-heading">Risk–Return Scatter</div>', unsafe_allow_html=True)
        fig_scat = go.Figure()
        fig_scat.add_trace(go.Scatter(
            x=asset_risks.values  * 100,
            y=asset_returns.values * 100,
            mode='markers+text',
            text=tickers,
            textposition='top center',
            textfont=dict(size=10, color="#80c8ff"),
            marker=dict(
                size=12,
                color=(asset_returns.values - RF_ANNUAL) / asset_risks.values,
                colorscale='RdYlGn',
                colorbar=dict(title="Sharpe", tickfont=dict(color="#80b0d0")),
                showscale=True,
                line=dict(width=1, color="#001428")
            ),
            hovertemplate="<b>%{text}</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"
        ))
        fig_scat.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0e17",
            plot_bgcolor="#0d1525",
            xaxis_title="Annual Risk σ (%)",
            yaxis_title="Annual Return μ (%)",
            title=dict(text="Individual Asset Risk–Return", font=dict(color="#a0c8e8")),
            font=dict(family="DM Mono", color="#80b0d0"),
            height=440,
            margin=dict(l=50, r=30, t=50, b=50),
        )
        st.plotly_chart(fig_scat, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIOS
# ══════════════════════════════════════════════════════════════
with tab_portfolios:
    if not st.session_state.data_loaded:
        st.markdown('<div class="status-box status-info">Load data and run optimizations first.</div>', unsafe_allow_html=True)
    else:
        tickers = st.session_state.tickers
        er      = st.session_state.expected_returns
        cov     = st.session_state.cov_matrix

        # ── Utility Portfolio ──────────────────────────────────
        st.markdown('<div class="section-heading">Utility-Maximized Portfolio</div>', unsafe_allow_html=True)
        if st.session_state.weights_utility is not None:
            w = st.session_state.weights_utility
            ann_ret  = er.values @ w * 252
            ann_risk = np.sqrt(w.T @ cov.values @ w) * np.sqrt(252)
            sharpe   = (ann_ret - RF_ANNUAL) / ann_risk
            portfolio_metrics_row(ann_ret, ann_risk, sharpe)
            st.markdown(make_weights_table(tickers, w, "Weight"), unsafe_allow_html=True)

            # Pie chart
            fig_pie = go.Figure(go.Pie(
                labels=tickers,
                values=w * 100,
                hole=0.45,
                textinfo='label+percent',
                textfont=dict(size=10),
                marker=dict(colors=[f"hsl({int(i*360/len(tickers))},60%,50%)" for i in range(len(tickers))],
                             line=dict(color='#0a0e17', width=2))
            ))
            fig_pie.update_layout(
                template="plotly_dark", paper_bgcolor="#0a0e17",
                title=dict(text="Utility Portfolio Allocation", font=dict(color="#a0c8e8", size=13)),
                showlegend=True,
                legend=dict(font=dict(size=9, color="#80b0d0")),
                height=380, margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.markdown('<div class="status-box status-info">Run "Utility Portfolio" to see results.</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Tangency Portfolio ─────────────────────────────────
        st.markdown('<div class="section-heading">Tangency Portfolio (Max Sharpe)</div>', unsafe_allow_html=True)
        if st.session_state.weights_tan is not None:
            w   = st.session_state.weights_tan
            tr  = st.session_state.tan_return
            tk  = st.session_state.tan_risk
            ts  = st.session_state.tan_sharpe

            st.markdown(f"""
            <div class="metric-row">
                {metric_card("Annual Return",  f"{tr*100:.2f}%", "positive")}
                {metric_card("Annual Risk",    f"{tk*100:.2f}%", "")}
                {metric_card("Sharpe Ratio",   f"{ts:.3f}",      "gold")}
                {metric_card("Risk-Free Rate", f"{RF_ANNUAL*100:.1f}%", "accent")}
            </div>""", unsafe_allow_html=True)
            st.markdown(make_weights_table(tickers, w, "Weight"), unsafe_allow_html=True)

            fig_pie2 = go.Figure(go.Pie(
                labels=tickers, values=w * 100, hole=0.45,
                textinfo='label+percent', textfont=dict(size=10),
                marker=dict(colors=[f"hsl({int(i*360/len(tickers)+40)},65%,52%)" for i in range(len(tickers))],
                             line=dict(color='#0a0e17', width=2))
            ))
            fig_pie2.update_layout(
                template="plotly_dark", paper_bgcolor="#0a0e17",
                title=dict(text="Tangency Portfolio Allocation", font=dict(color="#a0c8e8", size=13)),
                showlegend=True, legend=dict(font=dict(size=9, color="#80b0d0")),
                height=380, margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig_pie2, use_container_width=True)
        else:
            st.markdown('<div class="status-box status-info">Run "Tangency Portfolio" to see results.</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Min-Risk Portfolio ────────────────────────────────
        st.markdown(f'<div class="section-heading">Min-Risk Portfolio (Target: {st.session_state.target_return*100:.1f}%)</div>', unsafe_allow_html=True)
        if st.session_state.weights_min is not None:
            w = st.session_state.weights_min
            ann_ret  = er.values @ w * 252
            ann_risk = np.sqrt(w.T @ cov.values @ w) * np.sqrt(252)
            sharpe   = (ann_ret - RF_ANNUAL) / ann_risk
            portfolio_metrics_row(ann_ret, ann_risk, sharpe)
            st.markdown(make_weights_table(tickers, w, "Weight"), unsafe_allow_html=True)

            fig_pie3 = go.Figure(go.Pie(
                labels=tickers, values=w * 100, hole=0.45,
                textinfo='label+percent', textfont=dict(size=10),
                marker=dict(colors=[f"hsl({int(i*360/len(tickers)+80)},60%,48%)" for i in range(len(tickers))],
                             line=dict(color='#0a0e17', width=2))
            ))
            fig_pie3.update_layout(
                template="plotly_dark", paper_bgcolor="#0a0e17",
                title=dict(text="Min-Risk Portfolio Allocation", font=dict(color="#a0c8e8", size=13)),
                showlegend=True, legend=dict(font=dict(size=9, color="#80b0d0")),
                height=380, margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig_pie3, use_container_width=True)
        else:
            st.markdown('<div class="status-box status-info">Run "Min-Risk Portfolio" to see results.</div>', unsafe_allow_html=True)

        # ── Side-by-side comparison ────────────────────────────
        any_computed = any([
            st.session_state.weights_utility is not None,
            st.session_state.weights_tan is not None,
            st.session_state.weights_min is not None,
        ])
        if any_computed:
            st.markdown("---")
            st.markdown('<div class="section-heading">Portfolio Comparison</div>', unsafe_allow_html=True)

            comp_data = {}
            if st.session_state.weights_utility is not None:
                w = st.session_state.weights_utility
                comp_data["Utility"] = {
                    "Return %": round(er.values @ w * 252 * 100, 2),
                    "Risk %":   round(np.sqrt(w @ cov.values @ w) * np.sqrt(252) * 100, 2),
                    "Sharpe":   round((er.values @ w * 252 - RF_ANNUAL) / (np.sqrt(w @ cov.values @ w) * np.sqrt(252)), 3),
                }
            if st.session_state.weights_tan is not None:
                comp_data["Tangency"] = {
                    "Return %": round(st.session_state.tan_return * 100, 2),
                    "Risk %":   round(st.session_state.tan_risk   * 100, 2),
                    "Sharpe":   round(st.session_state.tan_sharpe, 3),
                }
            if st.session_state.weights_min is not None:
                w = st.session_state.weights_min
                comp_data["Min-Risk"] = {
                    "Return %": round(er.values @ w * 252 * 100, 2),
                    "Risk %":   round(np.sqrt(w @ cov.values @ w) * np.sqrt(252) * 100, 2),
                    "Sharpe":   round((er.values @ w * 252 - RF_ANNUAL) / (np.sqrt(w @ cov.values @ w) * np.sqrt(252)), 3),
                }

            if comp_data:
                comp_df = pd.DataFrame(comp_data).T
                st.dataframe(comp_df.style
                    .background_gradient(cmap="Greens", subset=["Return %"])
                    .background_gradient(cmap="Reds_r", subset=["Risk %"])
                    .background_gradient(cmap="Blues",  subset=["Sharpe"]),
                    use_container_width=True)

                # Bar chart comparison
                fig_bar = go.Figure()
                metrics_list = ["Return %", "Risk %", "Sharpe"]
                colors = ["#00b4ff", "#ff5252", "#ffd54f"]
                for metric, color in zip(metrics_list, colors):
                    fig_bar.add_trace(go.Bar(
                        name=metric,
                        x=list(comp_data.keys()),
                        y=[comp_data[p][metric] for p in comp_data],
                        marker_color=color,
                        text=[f"{comp_data[p][metric]}" for p in comp_data],
                        textposition='outside',
                        textfont=dict(size=9),
                    ))
                fig_bar.update_layout(
                    barmode='group',
                    template="plotly_dark",
                    paper_bgcolor="#0a0e17",
                    plot_bgcolor="#0d1525",
                    title=dict(text="Portfolio Metrics Comparison", font=dict(color="#a0c8e8")),
                    font=dict(family="DM Mono", color="#80b0d0"),
                    legend=dict(font=dict(size=9)),
                    height=380,
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — EFFICIENT FRONTIER
# ══════════════════════════════════════════════════════════════
with tab_frontier:
    if not st.session_state.data_loaded:
        st.markdown('<div class="status-box status-info">Load data first.</div>', unsafe_allow_html=True)
    elif not st.session_state.frontier_computed:
        st.markdown('<div class="status-box status-warn">Compute the efficient frontier first (Build Portfolio tab).</div>', unsafe_allow_html=True)
    else:
        fr  = st.session_state.frontier_risks
        ret = st.session_state.frontier_returns
        ar  = st.session_state.asset_risks
        are = st.session_state.asset_returns
        tickers = st.session_state.tickers

        # Tangency data
        tan_risk   = st.session_state.tan_risk
        tan_return = st.session_state.tan_return
        tan_sharpe = st.session_state.tan_sharpe

        # ── Interactive Plotly (2D) ────────────────────────────
        st.markdown('<div class="section-heading">Interactive Efficient Frontier</div>', unsafe_allow_html=True)

        fr_pct   = [r * 100 for r in ret]
        fsk_pct  = [r * 100 for r in fr]
        ar_pct   = np.array([are[t] * 100 for t in tickers])
        ask_pct  = np.array([ar[t]  * 100 for t in tickers])

        fig2d = go.Figure()
        fig2d.add_trace(go.Scatter(
            x=fsk_pct, y=fr_pct, mode='lines', name='Efficient Frontier',
            line=dict(color="#00b4ff", width=3),
            hovertemplate="Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"))

        fig2d.add_trace(go.Scatter(
            x=ask_pct, y=ar_pct, mode='markers+text', text=tickers,
            textposition="top center", textfont=dict(size=9, color="#80c8ff"),
            marker=dict(size=10, color="#ff5252", line=dict(width=1, color="#ff0000")),
            name='Assets',
            hovertemplate="<b>%{text}</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"))

        if tan_risk and tan_return and tan_sharpe:
            cml_x = np.linspace(0, max(fsk_pct) * 1.3, 100)
            cml_y = RF_ANNUAL * 100 + tan_sharpe * cml_x
            fig2d.add_trace(go.Scatter(
                x=cml_x, y=cml_y, mode='lines', name='Capital Market Line',
                line=dict(color='#00e676', dash='dash', width=2),
                hovertemplate="Risk: %{x:.2f}%<br>CML: %{y:.2f}%<extra></extra>"))
            fig2d.add_trace(go.Scatter(
                x=[tan_risk * 100], y=[tan_return * 100], mode='markers',
                marker=dict(size=18, color='gold', symbol='star',
                            line=dict(width=1, color='#aa8800')),
                name='Tangency Portfolio',
                hovertemplate=(f"<b>Tangency</b><br>Sharpe: {tan_sharpe:.3f}<br>"
                               "Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>")))

        # Slider
        steps_2d = []
        for i in range(len(fr_pct)):
            step_data = {
                "x": [fsk_pct, ask_pct.tolist()],
                "y": [fr_pct,  ar_pct.tolist()],
            }
            if tan_risk:
                step_data["x"] += [cml_x.tolist(), [tan_risk * 100], [fsk_pct[i]]]
                step_data["y"] += [cml_y.tolist(), [tan_return * 100], [fr_pct[i]]]
            steps_2d.append(dict(
                method="update",
                args=[step_data],
                label=f"{fr_pct[i]:.1f}%"
            ))

        # Selected point trace
        fig2d.add_trace(go.Scatter(
            x=[fsk_pct[0]], y=[fr_pct[0]], mode='markers',
            marker=dict(size=14, color='white', symbol='circle',
                        line=dict(color='#00b4ff', width=2)),
            name='Selected Point',
            hovertemplate="Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"))

        fig2d.update_layout(
            sliders=[dict(
                active=0,
                currentvalue={"prefix": "Target Return: ", "font": {"color": "#80b0d0"}},
                pad={"t": 50}, steps=steps_2d, transition={"duration": 0},
                bgcolor="#0d1525", bordercolor="#1a2d4d",
                font=dict(color="#4a6a90", size=8)
            )],
            template="plotly_dark",
            paper_bgcolor="#0a0e17",
            plot_bgcolor="#0d1525",
            title=dict(text="Efficient Frontier · Tangency Portfolio · Capital Market Line",
                       font=dict(color="#a0c8e8", size=13)),
            xaxis_title="Annual Risk (σ) %",
            yaxis_title="Annual Return (μ) %",
            font=dict(family="DM Mono", color="#80b0d0"),
            legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0.4)"),
            height=600,
            margin=dict(l=50, r=30, t=60, b=80),
        )
        st.plotly_chart(fig2d, use_container_width=True)

        # ── 3D Plot ────────────────────────────────────────────
        if tan_sharpe:
            st.markdown('<div class="section-heading">3D Frontier — Risk / Return / Sharpe</div>', unsafe_allow_html=True)
            frontier_sharpes = [(r - RF_ANNUAL*100) / risk for r, risk in zip(fr_pct, fsk_pct)]

            fig3d = go.Figure()
            fig3d.add_trace(go.Scatter3d(
                x=fsk_pct, y=fr_pct, z=frontier_sharpes,
                mode='lines+markers',
                marker=dict(size=3.5, color=frontier_sharpes, colorscale='Plasma',
                            colorbar=dict(title="Sharpe", tickfont=dict(color="#80b0d0"))),
                line=dict(width=5, color='royalblue'),
                name="Frontier",
                hovertemplate="Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{z:.3f}<extra></extra>"))
            fig3d.add_trace(go.Scatter3d(
                x=[tan_risk * 100], y=[tan_return * 100], z=[tan_sharpe],
                mode='markers',
                marker=dict(size=10, color='gold', symbol='diamond',
                            line=dict(color='#aa8800', width=1)),
                name='Tangency Portfolio'))
            fig3d.update_layout(
                scene=dict(
                    xaxis=dict(title="Risk (%)", backgroundcolor="#0d1525",
                               gridcolor="#1a2d4d", color="#80b0d0"),
                    yaxis=dict(title="Return (%)", backgroundcolor="#0d1525",
                               gridcolor="#1a2d4d", color="#80b0d0"),
                    zaxis=dict(title="Sharpe", backgroundcolor="#0d1525",
                               gridcolor="#1a2d4d", color="#80b0d0"),
                    bgcolor="#0a0e17",
                ),
                paper_bgcolor="#0a0e17",
                title=dict(text="3D Efficient Frontier — Risk / Return / Sharpe",
                           font=dict(color="#a0c8e8")),
                font=dict(family="DM Mono", color="#80b0d0"),
                legend=dict(font=dict(size=9)),
                height=560,
                margin=dict(l=0, r=0, t=50, b=0),
            )
            st.plotly_chart(fig3d, use_container_width=True)

        # ── Static Matplotlib frontier ─────────────────────────
        st.markdown('<div class="section-heading">Static Frontier Plot</div>', unsafe_allow_html=True)
        fig_s, ax_s = plt.subplots(figsize=(10, 6))
        fig_s.patch.set_facecolor("#0a0e17")
        ax_s.set_facecolor("#0d1525")

        ax_s.plot(fsk_pct, fr_pct, color="#00b4ff", linewidth=2.5, label="Efficient Frontier")
        ax_s.scatter(ask_pct, ar_pct, color="#ff5252", s=80, zorder=5, label="Assets")

        if tan_risk and tan_sharpe:
            cml_x2 = np.linspace(0, max(fsk_pct) * 1.2, 100)
            cml_y2 = RF_ANNUAL * 100 + tan_sharpe * cml_x2
            ax_s.plot(cml_x2, cml_y2, linestyle="--", color="#00e676", linewidth=1.8, label="CML")
            ax_s.scatter([tan_risk * 100], [tan_return * 100],
                         marker="*", s=350, color="gold", zorder=6, label="Tangency")

        for t in tickers:
            ax_s.annotate(t, (ar[t] * 100, are[t] * 100),
                          xytext=(5, 5), textcoords="offset points",
                          fontsize=7, color="#80c8ff")

        ax_s.set_xlabel("Annual Risk (σ) %", color="#80b0d0")
        ax_s.set_ylabel("Annual Return (μ) %", color="#80b0d0")
        ax_s.set_title("Efficient Frontier & CML", color="#a0c8e8", fontsize=12)
        ax_s.tick_params(colors="#4a6a90")
        ax_s.legend(fontsize=9, facecolor="#0d1525", edgecolor="#1a2d4d",
                    labelcolor="#80b0d0")
        ax_s.grid(alpha=0.15, color="#1a2d4d")
        for spine in ax_s.spines.values():
            spine.set_edgecolor("#1a2d4d")
        plt.tight_layout()
        st.pyplot(fig_s)


# ══════════════════════════════════════════════════════════════
# TAB 5 — ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab_analysis:
    if not st.session_state.data_loaded:
        st.markdown('<div class="status-box status-info">Load data first.</div>', unsafe_allow_html=True)
    else:
        returns = st.session_state.returns
        tickers = st.session_state.tickers
        er      = st.session_state.expected_returns
        cov     = st.session_state.cov_matrix
        ar      = st.session_state.asset_risks
        are     = st.session_state.asset_returns

        # ── Rolling correlation ────────────────────────────────
        st.markdown('<div class="section-heading">Rolling Volatility (60-day)</div>', unsafe_allow_html=True)
        roll_vol = returns.rolling(60).std() * np.sqrt(252) * 100
        fig_vol = go.Figure()
        for i, t in enumerate(tickers):
            hue = int(i * 360 / len(tickers))
            fig_vol.add_trace(go.Scatter(
                x=roll_vol.index, y=roll_vol[t],
                mode='lines', name=t,
                line=dict(width=1.5, color=f"hsl({hue},65%,55%)"),
                hovertemplate=f"<b>{t}</b><br>%{{x|%Y-%m-%d}}<br>Volatility: %{{y:.2f}}%<extra></extra>"
            ))
        fig_vol.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0d1525",
            xaxis_title="Date", yaxis_title="Annualised Volatility %",
            title=dict(text="60-Day Rolling Annualised Volatility", font=dict(color="#a0c8e8")),
            font=dict(family="DM Mono", color="#80b0d0"),
            legend=dict(font=dict(size=8), bgcolor="rgba(0,0,0,0.4)"),
            height=420, margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # ── Cumulative returns ─────────────────────────────────
        st.markdown('<div class="section-heading">Cumulative Returns</div>', unsafe_allow_html=True)
        cum_ret = (1 + returns).cumprod() - 1
        fig_cum = go.Figure()
        for i, t in enumerate(tickers):
            hue = int(i * 360 / len(tickers))
            fig_cum.add_trace(go.Scatter(
                x=cum_ret.index, y=cum_ret[t] * 100,
                mode='lines', name=t,
                line=dict(width=1.5, color=f"hsl({hue},65%,55%)"),
                hovertemplate=f"<b>{t}</b><br>%{{x|%Y-%m-%d}}<br>Cumulative Return: %{{y:.2f}}%<extra></extra>"
            ))
        fig_cum.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0d1525",
            xaxis_title="Date", yaxis_title="Cumulative Return %",
            title=dict(text="Cumulative Returns Over Period", font=dict(color="#a0c8e8")),
            font=dict(family="DM Mono", color="#80b0d0"),
            legend=dict(font=dict(size=8), bgcolor="rgba(0,0,0,0.4)"),
            height=420, margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # ── Portfolio cumulative return if computed ────────────
        if st.session_state.portfolios_computed or st.session_state.weights_tan is not None:
            st.markdown('<div class="section-heading">Portfolio Cumulative Returns</div>', unsafe_allow_html=True)
            fig_port = go.Figure()
            port_data = {}
            if st.session_state.weights_utility is not None:
                port_data["Utility"] = (st.session_state.weights_utility, "#00b4ff")
            if st.session_state.weights_tan is not None:
                port_data["Tangency"] = (st.session_state.weights_tan, "gold")
            if st.session_state.weights_min is not None:
                port_data["Min-Risk"] = (st.session_state.weights_min, "#00e676")

            for name, (w, color) in port_data.items():
                port_ret = returns[tickers].values @ w
                port_cum = (1 + port_ret).cumprod() - 1
                fig_port.add_trace(go.Scatter(
                    x=returns.index, y=port_cum * 100,
                    mode='lines', name=name,
                    line=dict(width=2, color=color),
                    hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}%<extra></extra>"
                ))
            fig_port.update_layout(
                template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0d1525",
                xaxis_title="Date", yaxis_title="Cumulative Return %",
                title=dict(text="Optimized Portfolio Cumulative Returns", font=dict(color="#a0c8e8")),
                font=dict(family="DM Mono", color="#80b0d0"),
                legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0.4)"),
                height=420, margin=dict(l=50, r=20, t=50, b=50)
            )
            st.plotly_chart(fig_port, use_container_width=True)

        # ── Covariance heatmap ─────────────────────────────────
        st.markdown('<div class="section-heading">Covariance Matrix (Annualised)</div>', unsafe_allow_html=True)
        cov_annual = cov * 252
        fig_cov, ax_cov = plt.subplots(figsize=(max(6, len(tickers)*0.7), max(5, len(tickers)*0.6)))
        fig_cov.patch.set_facecolor("#0a0e17")
        ax_cov.set_facecolor("#0a0e17")
        sns.heatmap(cov_annual, annot=True, cmap="YlOrBr", fmt=".4f",
                    linewidths=0.4, linecolor="#1a2d4d",
                    annot_kws={"size": max(5, 8-len(tickers)//3), "color": "white"},
                    ax=ax_cov, xticklabels=tickers, yticklabels=tickers)
        ax_cov.xaxis.tick_top()
        plt.xticks(rotation=45, ha="left", color="#a0b8d0", fontsize=max(5, 8-len(tickers)//4))
        plt.yticks(color="#a0b8d0", fontsize=max(5, 8-len(tickers)//4))
        plt.title("Annualised Covariance Matrix", color="#80b0d0", fontsize=11, pad=20)
        plt.tight_layout()
        st.pyplot(fig_cov)


# ══════════════════════════════════════════════════════════════
# TAB 6 — EXPORT
# ══════════════════════════════════════════════════════════════
with tab_export:
    st.markdown('<div class="section-heading">Export Results</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.markdown('<div class="status-box status-info">Load data and run optimizations to export results.</div>', unsafe_allow_html=True)
    else:
        tickers = st.session_state.tickers
        er      = st.session_state.expected_returns
        cov     = st.session_state.cov_matrix

        # ── Weights CSV ────────────────────────────────────────
        st.markdown('<div class="card-title">Portfolio Weights</div>', unsafe_allow_html=True)

        export_rows = {"Ticker": tickers}
        if st.session_state.weights_utility is not None:
            w = st.session_state.weights_utility
            export_rows["Utility_Weight"] = w
            export_rows["Utility_Pct"]    = (w * 100).round(2)
        if st.session_state.weights_tan is not None:
            w = st.session_state.weights_tan
            export_rows["Tangency_Weight"] = w
            export_rows["Tangency_Pct"]    = (w * 100).round(2)
        if st.session_state.weights_min is not None:
            w = st.session_state.weights_min
            export_rows["MinRisk_Weight"] = w
            export_rows["MinRisk_Pct"]    = (w * 100).round(2)

        if len(export_rows) > 1:
            weights_df = pd.DataFrame(export_rows)
            st.dataframe(weights_df, use_container_width=True)
            csv_bytes = weights_df.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download Weights CSV",
                data=csv_bytes,
                file_name="quansen_weights.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.markdown('<div class="status-box status-warn">Run at least one optimization to export weights.</div>', unsafe_allow_html=True)

        # ── Asset summary CSV ──────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="card-title">Asset Summary</div>', unsafe_allow_html=True)

        asset_risks = st.session_state.asset_risks
        asset_returns = st.session_state.asset_returns
        summary_exp_df = pd.DataFrame({
            "Ticker"        : tickers,
            "Ann_Return_Pct": (asset_returns.values * 100).round(4),
            "Ann_Risk_Pct"  : (asset_risks.values   * 100).round(4),
            "Sharpe"        : ((asset_returns.values - RF_ANNUAL) / asset_risks.values).round(4),
        })
        st.dataframe(summary_exp_df, use_container_width=True)
        st.download_button(
            "⬇ Download Asset Summary CSV",
            data=summary_exp_df.to_csv(index=False).encode(),
            file_name="quansen_asset_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

        # ── Frontier CSV ───────────────────────────────────────
        if st.session_state.frontier_computed:
            st.markdown("---")
            st.markdown('<div class="card-title">Efficient Frontier Points</div>', unsafe_allow_html=True)
            frontier_df = pd.DataFrame({
                "Return_Pct": [r * 100 for r in st.session_state.frontier_returns],
                "Risk_Pct":   [r * 100 for r in st.session_state.frontier_risks],
            })
            frontier_df["Sharpe"] = ((frontier_df["Return_Pct"] - RF_ANNUAL * 100)
                                      / frontier_df["Risk_Pct"]).round(4)
            st.dataframe(frontier_df.head(20), use_container_width=True)
            st.download_button(
                "⬇ Download Frontier CSV",
                data=frontier_df.to_csv(index=False).encode(),
                file_name="quansen_frontier.csv",
                mime="text/csv",
                use_container_width=True
            )

        # ── Full stats summary ─────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="card-title">Optimizer Summary Report</div>', unsafe_allow_html=True)

        lines = [
            "QUANSEN PORTFOLIO OPTIMIZER — SUMMARY REPORT",
            "=" * 60,
            f"Tickers      : {', '.join(tickers)}",
            f"Date Range   : {st.session_state.start_date} to {st.session_state.end_date}",
            f"Risk-Free Rt : {RF_ANNUAL*100:.2f}%",
            f"Min Weight   : {st.session_state.min_w*100:.1f}%",
            f"Max Weight   : {st.session_state.max_w*100:.1f}%",
            "",
        ]
        if st.session_state.weights_utility is not None:
            w = st.session_state.weights_utility
            ann_ret  = er.values @ w * 252
            ann_risk = np.sqrt(w @ cov.values @ w) * np.sqrt(252)
            lines += [
                "UTILITY-MAXIMIZED PORTFOLIO",
                "-" * 40,
                *[f"  {t:<20} {w_:.4f}  ({w_*100:.2f}%)" for t, w_ in zip(tickers, w)],
                f"  Annual Return  : {ann_ret*100:.2f}%",
                f"  Annual Risk    : {ann_risk*100:.2f}%",
                f"  Sharpe Ratio   : {(ann_ret-RF_ANNUAL)/ann_risk:.3f}",
                "",
            ]
        if st.session_state.weights_tan is not None:
            w = st.session_state.weights_tan
            lines += [
                "TANGENCY PORTFOLIO (MAX SHARPE)",
                "-" * 40,
                *[f"  {t:<20} {w_:.4f}  ({w_*100:.2f}%)" for t, w_ in zip(tickers, w)],
                f"  Annual Return  : {st.session_state.tan_return*100:.2f}%",
                f"  Annual Risk    : {st.session_state.tan_risk*100:.2f}%",
                f"  Sharpe Ratio   : {st.session_state.tan_sharpe:.3f}",
                "",
            ]
        if st.session_state.weights_min is not None:
            w = st.session_state.weights_min
            ann_ret  = er.values @ w * 252
            ann_risk = np.sqrt(w @ cov.values @ w) * np.sqrt(252)
            lines += [
                f"MIN-RISK PORTFOLIO (Target: {st.session_state.target_return*100:.1f}%)",
                "-" * 40,
                *[f"  {t:<20} {w_:.4f}  ({w_*100:.2f}%)" for t, w_ in zip(tickers, w)],
                f"  Annual Return  : {ann_ret*100:.2f}%",
                f"  Annual Risk    : {ann_risk*100:.2f}%",
                f"  Sharpe Ratio   : {(ann_ret-RF_ANNUAL)/ann_risk:.3f}",
                "",
            ]
        lines.append("=" * 60)
        report_text = "\n".join(lines)
        st.code(report_text, language=None)
        st.download_button(
            "⬇ Download Summary Report (.txt)",
            data=report_text.encode(),
            file_name="quansen_report.txt",
            mime="text/plain",
            use_container_width=True
        )
