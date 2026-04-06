"""
============================================================
  QuanSen — Data Loading, Cached Computation & PDF Export
  Module 2 of 4: Market data fetching, optimizer wrappers,
  cached computations, plotting helpers, and report generation.
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import yfinance as _yf
import plotly.graph_objects as go

from engines.optimizers import (
    utility_portfolio,
    compute_frontier,
    tangency_portfolio,
    min_risk_portfolio,
)
from engines.config import RF_ANNUAL, SHRINKAGE_ALPHA
from engines.data_loader import load_data as _engine_load_data



# ── Data Loading ──────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)

def load_data(tickers, start_date, end_date, alpha=None):
    """
    Thin cached wrapper around the canonical engine loader.
    Keeps one authoritative data path for downloads, overlap trimming,
    shrinkage, and covariance estimation.
    """
    tickers = list(tickers)
    if alpha is None:
        alpha = SHRINKAGE_ALPHA
    tickers_out, returns, expected_returns, cov_matrix, raw_er, bm_sym = (
        _engine_load_data(tickers, start_date, end_date, alpha=alpha)
    )
    tickers_out      = list(tickers_out)
    returns          = returns[tickers_out]
    expected_returns = expected_returns[tickers_out]
    raw_er           = raw_er[tickers_out]
    cov_matrix       = cov_matrix.loc[tickers_out, tickers_out]
    return tickers_out, returns, expected_returns, cov_matrix, raw_er, bm_sym


# ── Expected Returns Resolver ─────────────────────────────────
def optimizer_expected_returns():
    """Return the exact expected-return series currently driving optimization."""
    if (st.session_state.get("momentum_enabled") and
            st.session_state.get("momentum_final_er") is not None):
        return st.session_state.momentum_final_er.reindex(st.session_state.tickers)
    return st.session_state.expected_returns



# ── Cached Market Quotes ──────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def cached_fetch_tape_quotes(symbols):
    symbols = list(symbols)
    if not symbols:
        return []

    try:
        data = _yf.download(
            symbols,
            period="7d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )

        if data.empty:
            return []

        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data[["Close"]] if "Close" in data.columns else data

        if isinstance(close, pd.Series):
            close = close.to_frame(name=symbols[0])

        rows = []
        for symbol in symbols:
            if symbol not in close.columns:
                continue
            series = close[symbol].dropna()
            if series.empty:
                continue
            price = float(series.iloc[-1])
            prev = float(series.iloc[-2]) if len(series) >= 2 else price
            chg = 0.0 if prev == 0 else ((price / prev) - 1.0) * 100.0
            rows.append({
                "symbol": symbol,
                "price": price,
                "chg": chg,
            })
        return rows
    except Exception:
        return []



# ── Cached Momentum Computation ───────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def cached_compute_momentum(tickers, start_date, end_date, shrink_er, lookback, skip, beta, auto_lookback, prev_regime_probs=None):
    from engines.momentum_engine import compute_momentum

    return compute_momentum(
        list(tickers),
        start_date,
        end_date,
        shrink_er,
        lookback=lookback,
        skip=skip,
        beta=beta,
        auto_lookback=auto_lookback,
        prev_regime_probs=prev_regime_probs,
    )



# ── Cached Optimizer Wrappers ─────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def cached_utility_portfolio(
    expected_returns,
    cov_matrix,
    tickers,
    min_w,
    max_w,
    previous_weights=None,
    turnover_penalty=0.0,
    rebalance_ratio=1.0,
):
    return utility_portfolio(
        expected_returns,
        cov_matrix,
        list(tickers),
        min_w,
        max_w,
        previous_weights=previous_weights,
        turnover_penalty=turnover_penalty,
        rebalance_ratio=rebalance_ratio,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_tangency_portfolio(
    expected_returns,
    cov_matrix,
    tickers,
    min_w,
    max_w,
    previous_weights=None,
    turnover_penalty=0.0,
    rebalance_ratio=1.0,
):
    return tangency_portfolio(
        expected_returns,
        cov_matrix,
        list(tickers),
        min_w,
        max_w,
        previous_weights=previous_weights,
        turnover_penalty=turnover_penalty,
        rebalance_ratio=rebalance_ratio,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_min_risk_portfolio(
    expected_returns,
    cov_matrix,
    tickers,
    target_return,
    min_w,
    max_w,
    previous_weights=None,
    turnover_penalty=0.0,
    rebalance_ratio=1.0,
):
    return min_risk_portfolio(
        expected_returns,
        cov_matrix,
        list(tickers),
        target_return,
        min_w,
        max_w,
        previous_weights=previous_weights,
        turnover_penalty=turnover_penalty,
        rebalance_ratio=rebalance_ratio,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_compute_frontier(expected_returns, cov_matrix, min_w, max_w, n_points=100):
    return compute_frontier(
        expected_returns,
        cov_matrix,
        min_w,
        max_w,
        n_points=n_points,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_asset_risks(returns):
    return returns.std() * np.sqrt(252)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_correlation_matrix(returns):
    return returns.corr()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_portfolio_stats(weights, expected_returns, cov_matrix):
    ann_ret = float(expected_returns.values @ weights) * 252
    variance = float(weights @ cov_matrix.values @ weights)
    ann_risk = np.sqrt(max(variance, 0.0)) * np.sqrt(252)
    sharpe = (ann_ret - RF_ANNUAL) / ann_risk if ann_risk > 1e-12 else np.nan
    return ann_ret, ann_risk, sharpe


@st.cache_data(ttl=3600, show_spinner=False)
def cached_portfolio_cumulative_returns(returns, weights):
    clean_returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    port_ret = clean_returns.values @ weights
    return pd.Series((1 + port_ret).cumprod() - 1, index=clean_returns.index)


# ── Plotly Heatmap Helper ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════
# PLOTLY HEATMAP HELPER
# ══════════════════════════════════════════════════════════════
def _abbrev(ticker: str, max_len: int = 8) -> str:
    """BAJFINANCE.NS → BAJFIN  |  short tickers unchanged."""
    base = ticker.split(".")[0]          # strip .NS / .BO / etc.
    return base[:max_len]


def plotly_heatmap(matrix: "pd.DataFrame", title: str,
                   colorscale: str = "RdBu", zmid: float = 0,
                   fmt: str = ".2f", height: int = 520) -> "go.Figure":
    """
    Interactive Plotly heatmap with:
    - Abbreviated axis labels (strip exchange suffix, truncate)
    - Full ticker name shown on hover
    - Cell value shown on hover
    - Auto-contrasting annotation text (dark on light cells, light on dark)
    - Dark QuanSen theme
    """
    tickers  = list(matrix.columns)
    abbrevs  = [_abbrev(t) for t in tickers]
    z        = matrix.values
    n        = len(tickers)

    # Build hover text: "BAJFINANCE.NS × HDFCBANK.NS\n0.43"
    hover = [[
        f"<b>{tickers[r]}</b> × <b>{tickers[c]}</b><br>value: {z[r, c]:{fmt[1:]}}"
        for c in range(n)
    ] for r in range(n)]

    # Auto-contrast annotation colour per cell
    # Normalise z to [0,1] relative to colorscale range
    z_min, z_max = float(matrix.min().min()), float(matrix.max().max())
    span = z_max - z_min if z_max != z_min else 1.0

    ann_text   = [[f"{z[r,c]:{fmt[1:]}}" for c in range(n)] for r in range(n)]
    font_colors = []
    for r in range(n):
        row_colors = []
        for c in range(n):
            norm = (z[r, c] - z_min) / span   # 0 = min colour, 1 = max colour
            # RdBu: near 0 = red (dark), near 0.5 = white (light), near 1 = blue (dark)
            # Use white text on dark ends, dark text in the middle
            if colorscale == "RdBu":
                use_white = norm < 0.25 or norm > 0.75
            else:
                # Sequential scales like YlOrBr start very light and only become dark near the top end.
                use_white = norm > 0.72
            row_colors.append("white" if use_white else "#111827")
        font_colors.append(row_colors)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=abbrevs,
        y=abbrevs,
        text=hover,
        hoverinfo="text",
        colorscale=colorscale,
        zmid=zmid,
        zmin=z_min if colorscale != "RdBu" else None,
        zmax=z_max if colorscale != "RdBu" else None,
        colorbar=dict(
            tickfont=dict(color="#80b0d0", size=10),
            outlinecolor="#1a2d4d",
            outlinewidth=1,
        ),
        xgap=1, ygap=1,
    ))

    # Annotations — adaptive font colour per cell
    annotations = []
    font_size = max(6, min(11, int(180 / max(n, 1))))
    for r in range(n):
        for c in range(n):
            annotations.append(dict(
                x=abbrevs[c], y=abbrevs[r],
                text=ann_text[r][c],
                showarrow=False,
                font=dict(size=font_size, color=font_colors[r][c],
                          family="DM Mono, monospace"),
                xref="x", yref="y",
            ))

    fig.update_layout(
        annotations=annotations,
        title=dict(text=title, font=dict(color="#a0c8e8", size=13)),
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#0d1525",
        font=dict(family="DM Mono", color="#80b0d0"),
        xaxis=dict(
            tickfont=dict(size=max(8, min(12, int(160/max(n,1)))),
                          color="#a0c8e8"),
            tickangle=-35,
            side="bottom",
            gridcolor="#1a2d4d",
        ),
        yaxis=dict(
            tickfont=dict(size=max(8, min(12, int(160/max(n,1)))),
                          color="#a0c8e8"),
            autorange="reversed",
            gridcolor="#1a2d4d",
        ),
        height=height,
        margin=dict(l=80, r=40, t=60, b=80),
    )
    return fig


# ── Ticker Search API ─────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
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



# ── Ticker Date Range ─────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_date_range(ticker: str):
    """
    Return (first_date, last_date, last_colour, last_age_days) for a ticker.
    Fetches the entire available history in one call (period='max').
    Colour-codes the last date: green <=7d, amber <=30d, red >30d.
    Falls back to 'N/A' strings on any failure.
    """
    import datetime as _dt
    try:
        valid = None
        df = _yf.download(
            ticker,
            period="max",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                close = df["Close"]
            else:
                close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            valid = close.dropna()

        # Fallback: Yahoo sometimes flakes on one path but succeeds on Ticker.history.
        if valid is None or valid.empty:
            hist = _yf.Ticker(ticker).history(period="max", auto_adjust=True)
            if not hist.empty and "Close" in hist.columns:
                valid = hist["Close"].dropna()

        if valid is None or valid.empty:
            return "N/A", "N/A", "#4a6a90", 9999

        first_dt = valid.index[0]
        last_dt  = valid.index[-1]
        age      = (_dt.date.today() - last_dt.date()).days

        if age <= 7:
            last_colour = "#00e676"   # green  — actively trading
        elif age <= 30:
            last_colour = "#ffd54f"   # amber  — recent gap / holiday
        else:
            last_colour = "#ff5252"   # red    — stale / delisted

        return first_dt.strftime("%Y-%m-%d"), last_dt.strftime("%Y-%m-%d"), last_colour, age

    except Exception:
        return "N/A", "N/A", "#4a6a90", 9999



# ── PDF Report Generator ──────────────────────────────────────
# ══════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ══════════════════════════════════════════════════════════════
def generate_pdf(ss) -> bytes:
    """Build a full QuanSen report as PDF bytes using reportlab."""
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable, PageBreak)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2*cm,
        title="QuanSen Portfolio Report"
    )

    W, H = A4
    # ── Colour palette (dark theme printed on white) ──────────────
    NAVY   = colors.HexColor("#0d1a35")
    BLUE   = colors.HexColor("#0066cc")
    LBLUE  = colors.HexColor("#4a90d9")
    TEAL   = colors.HexColor("#00b4ff")
    GREEN  = colors.HexColor("#00c853")
    RED    = colors.HexColor("#e53935")
    GOLD   = colors.HexColor("#f9a825")
    DGREY  = colors.HexColor("#263238")
    LGREY  = colors.HexColor("#eceff1")
    WHITE  = colors.white
    BLACK  = colors.HexColor("#1a1a2e")

    styles = getSampleStyleSheet()

    def S(name, **kw):
        base = styles[name]
        return ParagraphStyle(name + "_q", parent=base, **kw)

    title_st   = S("Title",   fontSize=26, textColor=NAVY,  spaceAfter=4,  leading=30)
    sub_st     = S("Normal",  fontSize=9,  textColor=LBLUE, spaceAfter=2,  leading=12)
    h1_st      = S("Heading1",fontSize=13, textColor=NAVY,  spaceBefore=14, spaceAfter=4, leading=16)
    h2_st      = S("Heading2",fontSize=10, textColor=BLUE,  spaceBefore=8,  spaceAfter=3, leading=13)
    body_st    = S("Normal",  fontSize=8.5,textColor=DGREY, spaceAfter=3,  leading=12)
    mono_st    = S("Code",    fontSize=7.5,textColor=DGREY, spaceAfter=2,  leading=10, fontName="Courier")
    caption_st = S("Normal",  fontSize=7,  textColor=LBLUE, spaceAfter=2,  leading=9,  alignment=TA_RIGHT)
    red_st     = S("Normal",  fontSize=8.5,textColor=RED,   spaceAfter=3,  leading=12)
    green_st   = S("Normal",  fontSize=8.5,textColor=GREEN, spaceAfter=3,  leading=12)

    def HR():
        return HRFlowable(width="100%", thickness=0.5, color=LBLUE, spaceAfter=6, spaceBefore=2)

    def section(title):
        return [HR(), Paragraph(title, h1_st)]

    def subsection(title):
        return [Paragraph(title, h2_st)]

    def kv(label, value, style=body_st):
        return Paragraph(f"<b>{label}:</b>  {value}", style)

    def df_to_table(df, col_widths=None):
        """Convert a DataFrame to a reportlab Table with QuanSen styling."""
        data = [list(df.columns)] + df.astype(str).values.tolist()
        tbl  = Table(data, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  NAVY),
            ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,0),  7),
            ("ALIGN",         (0,0), (-1,0),  "CENTER"),
            ("BOTTOMPADDING", (0,0), (-1,0),  5),
            ("TOPPADDING",    (0,0), (-1,0),  5),
            ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE",      (0,1), (-1,-1), 7),
            ("ALIGN",         (1,1), (-1,-1), "RIGHT"),
            ("ALIGN",         (0,1), (0,-1),  "LEFT"),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, LGREY]),
            ("GRID",          (0,0), (-1,-1), 0.25, colors.HexColor("#b0bec5")),
            ("LEFTPADDING",   (0,0), (-1,-1), 5),
            ("RIGHTPADDING",  (0,0), (-1,-1), 5),
            ("TOPPADDING",    (0,1), (-1,-1), 3),
            ("BOTTOMPADDING", (0,1), (-1,-1), 3),
        ]))
        return tbl

    story = []

    # ── Cover ─────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("QUAN<font color='#0066cc'>SEN</font>", title_st))
    story.append(Paragraph("Quantitative Portfolio Optimizer — Full Report", sub_st))
    story.append(Spacer(1, 0.3*cm))
    import datetime as _dtnow
    active_saved = ss.get("active_saved_portfolio")
    active_watch = ss.get("active_watchlist")
    story.append(Paragraph(
        f"Generated: {_dtnow.datetime.now().strftime('%Y-%m-%d  %H:%M')}  ·  "
        f"Period: {ss.start_date}  →  {ss.end_date}  ·  "
        f"RF Rate: {RF_ANNUAL*100:.2f}%",
        sub_st
    ))
    story.append(HR())
    story.append(kv("Tickers", "  |  ".join(ss.tickers)))
    story.append(kv("Weight bounds", f"{ss.min_w*100:.1f}%  –  {ss.max_w*100:.1f}%"))
    if active_saved:
        story.append(kv("Saved Portfolio Context", active_saved))
    if active_watch:
        story.append(kv("Watchlist Context", active_watch))
    cover_summary = Table([
        ["Assets", str(len(ss.tickers)), "Momentum", "On" if ss.momentum_enabled else "Off"],
        ["Min / Max Wt", f"{ss.min_w*100:.1f}% / {ss.max_w*100:.1f}%", "Target Return", f"{ss.target_return*100:.1f}%"],
        ["Shrinkage α", f"{ss.shrinkage_alpha*100:.0f}%", "Alerts", str(len(ss.get('alerts', [])))],
    ], colWidths=[3.2*cm, 3.1*cm, 3.2*cm, 3.1*cm])
    cover_summary.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#f7fbff")),
        ("BOX", (0,0), (-1,-1), 0.4, colors.HexColor("#c7d8ea")),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#d2dee9")),
        ("TEXTCOLOR", (0,0), (-1,-1), DGREY),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(Spacer(1, 0.2*cm))
    story.append(cover_summary)
    story.append(PageBreak())

    # ── 1. Asset Summary ──────────────────────────────────────────
    story += section("1. Asset Summary")
    ar = ss.asset_returns
    ak = ss.asset_risks
    # Resolve active expected returns (same as optimizer)
    if ss.get("momentum_enabled") and ss.get("momentum_final_er") is not None:
        er = ss.momentum_final_er.reindex(ss.tickers)
    else:
        er = ss.expected_returns
    tickers = ss.tickers
    asset_df = pd.DataFrame({
        "Ticker":        tickers,
        "Ann. Return %": (ar.values * 100).round(2),
        "Ann. Risk %":   (ak.values * 100).round(2),
        "Sharpe":        ((ar.values - RF_ANNUAL) / ak.values).round(3),
        "Daily Mean %":  (er.values * 100).round(4),
    })
    story.append(df_to_table(asset_df))
    story.append(Spacer(1, 0.3*cm))

    # ── 2. Correlation Matrix ─────────────────────────────────────
    story += section("2. Correlation Matrix")
    corr = cached_correlation_matrix(ss.returns).round(3)
    corr_display = corr.reset_index()
    corr_display.columns = [""] + list(corr.columns)
    n_c = len(corr_display.columns)
    cw  = (W - 4*cm) / n_c
    tbl_corr = df_to_table(corr_display, col_widths=[cw]*n_c)
    # Colour-code correlation cells
    ts = [
        ("BACKGROUND", (0,0), (-1,0),   NAVY),
        ("TEXTCOLOR",  (0,0), (-1,0),   WHITE),
        ("FONTNAME",   (0,0), (-1,0),   "Helvetica-Bold"),
    ]
    for i in range(1, len(corr_display)+1):
        for j in range(1, n_c):
            try:
                val = float(corr_display.iloc[i-1, j])
            except Exception:
                continue
            if val >= 0.7:
                bg = colors.HexColor("#ef9a9a")
            elif val >= 0.4:
                bg = colors.HexColor("#fff9c4")
            elif val <= -0.4:
                bg = colors.HexColor("#b2dfdb")
            else:
                bg = WHITE if i % 2 == 0 else LGREY
            ts.append(("BACKGROUND", (j, i), (j, i), bg))
    tbl_corr.setStyle(TableStyle(ts))
    story.append(tbl_corr)
    story.append(Paragraph("Red = high positive correlation  ·  Teal = negative correlation  ·  Yellow = moderate", caption_st))
    story.append(PageBreak())

    # ── 3. Portfolio Weights ──────────────────────────────────────
    story += section("3. Optimised Portfolio Weights")
    cov = ss.cov_matrix

    def port_block(label, w):
        ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
        story.extend(subsection(label))
        w_df = pd.DataFrame({
            "Ticker":  tickers,
            "Weight":  np.round(w, 4),
            "Alloc %": np.round(w * 100, 2),
        })
        story.append(df_to_table(w_df, col_widths=[5*cm, 3*cm, 3*cm]))
        story.append(Spacer(1, 0.15*cm))
        story.append(kv("Annual Return", f"{ann_ret*100:.2f}%"))
        story.append(kv("Annual Risk",   f"{ann_risk*100:.2f}%"))
        story.append(kv("Sharpe Ratio",  f"{sharpe:.3f}"))
        story.append(Spacer(1, 0.3*cm))

    if ss.weights_utility is not None:
        port_block("Utility-Maximised Portfolio", ss.weights_utility)
    if ss.weights_tan is not None:
        w = ss.weights_tan
        story.extend(subsection("Tangency Portfolio (Max Sharpe)"))
        w_df = pd.DataFrame({
            "Ticker":  tickers,
            "Weight":  np.round(w, 4),
            "Alloc %": np.round(w * 100, 2),
        })
        story.append(df_to_table(w_df, col_widths=[5*cm, 3*cm, 3*cm]))
        story.append(Spacer(1, 0.15*cm))
        story.append(kv("Annual Return", f"{ss.tan_return*100:.2f}%"))
        story.append(kv("Annual Risk",   f"{ss.tan_risk*100:.2f}%"))
        story.append(kv("Sharpe Ratio",  f"{ss.tan_sharpe:.3f}"))
        story.append(Spacer(1, 0.3*cm))
    if ss.weights_min is not None:
        port_block(f"Min-Risk Portfolio (Target {ss.target_return*100:.1f}%)", ss.weights_min)

    story.append(PageBreak())

    # ── 4. Efficient Frontier ─────────────────────────────────────
    if ss.frontier_computed:
        story += section("4. Efficient Frontier Points (sample)")
        fr_df = pd.DataFrame({
            "Return %": [round(r*100, 3) for r in ss.frontier_returns],
            "Risk %":   [round(r*100, 3) for r in ss.frontier_risks],
        })
        fr_df["Sharpe"] = ((fr_df["Return %"] - RF_ANNUAL*100) / fr_df["Risk %"]).round(3)
        story.append(df_to_table(fr_df.iloc[::max(1, len(fr_df)//30)].reset_index(drop=True)))
        story.append(PageBreak())

    # ── 5. Monitoring ─────────────────────────────────────────────
    if ss.get("watchlists") or ss.get("alerts"):
        story += section("5. Monitoring Setup")
        if ss.get("watchlists"):
            wl_rows = []
            for name, meta in ss.watchlists.items():
                symbols = meta.get("symbols", [])
                wl_rows.append({
                    "Watchlist": name,
                    "Symbols": ", ".join(symbols[:6]) + (" ..." if len(symbols) > 6 else ""),
                    "Count": len(symbols),
                    "Updated": meta.get("updated_at", "N/A"),
                })
            story.extend(subsection("Saved Watchlists"))
            story.append(df_to_table(pd.DataFrame(wl_rows)))
            story.append(Spacer(1, 0.2*cm))
        if ss.get("alerts"):
            alert_rows = [{
                "Symbol": a.get("symbol", ""),
                "Condition": a.get("condition", ""),
                "Threshold": f"{float(a.get('threshold', 0.0)):.3f}",
                "Note": a.get("note", ""),
            } for a in ss.alerts]
            story.extend(subsection("Active Alerts"))
            story.append(df_to_table(pd.DataFrame(alert_rows)))
            story.append(PageBreak())

    # ── 6. Backtest ───────────────────────────────────────────────
    bt = ss.bt_results
    if bt is not None:
        story += section("6. Backtest Results")
        story.extend(subsection(f"Portfolio: {bt['portfolio_name']}"))
        story.append(kv("Entry Mode",       bt["entry_mode"]))
        story.append(kv("Backtest Period",  f"{bt['overall_entry']}  →  {bt['overall_exit']}  ({bt['n_days']} days / {bt['n_years']:.1f} yrs)"))
        story.append(kv("Starting Capital", f"Rs {bt['capital']:,.2f}"))
        story.append(kv("Uninvested Cash",  f"Rs {bt['residual']:,.2f}"))
        story.append(Spacer(1, 0.2*cm))

        # Summary metrics table
        pr = bt["pct_return"]
        cg = bt["cagr"]
        ret_col  = GREEN if pr >= 0 else RED
        cagr_col = GREEN if cg >= 0 else RED
        summary_data = [
            ["Metric", "Value"],
            ["Capital Invested",  f"Rs {bt['total_invest']:>14,.2f}"],
            ["Portfolio Value",   f"Rs {bt['total_exit']:>14,.2f}"],
            ["Absolute Return",   f"Rs {bt['abs_return']:>+14,.2f}"],
            ["Total Return %",    f"{pr:>+.2f}%"],
            ["CAGR",              f"{cg:>+.2f}%"],
        ]
        s_tbl = Table(summary_data, colWidths=[6*cm, 5*cm])
        s_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  NAVY),
            ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
            ("FONTNAME",      (0,0), (-1,-1), "Helvetica"),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8.5),
            ("ALIGN",         (1,0), (1,-1),  "RIGHT"),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, LGREY]),
            ("GRID",          (0,0), (-1,-1), 0.25, colors.HexColor("#b0bec5")),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("RIGHTPADDING",  (0,0), (-1,-1), 6),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("TEXTCOLOR",     (1,3), (1,3),   ret_col),
            ("TEXTCOLOR",     (1,4), (1,4),   cagr_col),
            ("FONTNAME",      (1,3), (1,4),   "Helvetica-Bold"),
        ]))
        story.append(s_tbl)
        story.append(Spacer(1, 0.4*cm))

        # Trade blotter
        story.extend(subsection("Trade Blotter"))
        blotter = bt["blotter"].copy()
        # Format currency columns
        for col in ["Capital Deployed", "Current Value", "Buy Price", "Sell Price"]:
            if col in blotter.columns:
                blotter[col] = blotter[col].apply(lambda x: f"{x:,.2f}")
        if "Gain / Loss" in blotter.columns:
            blotter["Gain / Loss"] = bt["blotter"]["Gain / Loss"].apply(lambda x: f"{x:+,.2f}")
        if "Return %" in blotter.columns:
            blotter["Return %"] = bt["blotter"]["Return %"].apply(lambda x: f"{x:+.2f}%")
        if "Shares Bought" in blotter.columns:
            blotter["Shares Bought"] = bt["blotter"]["Shares Bought"].apply(lambda x: f"{x:.4f}")

        n_cols = len(blotter.columns)
        usable_w = W - 4*cm
        cw_blotter = [usable_w / n_cols] * n_cols
        b_tbl = df_to_table(blotter, col_widths=cw_blotter)

        # Colour Return % column green/red
        ret_col_idx = list(blotter.columns).index("Return %") if "Return %" in blotter.columns else -1
        if ret_col_idx >= 0:
            extra = []
            for i, val in enumerate(bt["blotter"]["Return %"], start=1):
                c = GREEN if val >= 0 else RED
                extra.append(("TEXTCOLOR", (ret_col_idx, i), (ret_col_idx, i), c))
            b_tbl.setStyle(TableStyle(extra))

        story.append(b_tbl)

    # ── Footer note ───────────────────────────────────────────────
    story.append(Spacer(1, 0.6*cm))
    story.append(HR())
    story.append(Paragraph(
        f"QuanSen v{APP_VERSION}  ·  Amatra Sen  ·  For informational purposes only. Not financial advice.",
        S("Normal", fontSize=7, textColor=LBLUE, alignment=TA_CENTER)
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
