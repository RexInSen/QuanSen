"""
============================================================
  Quantitative Portfolio Optimizer(QuanSen) — v1.0
  Author: Amatra Sen
  Description: MPT-based portfolio optimizer with efficient
               frontier, tangency portfolio, and CML.
============================================================
"""

import sys
import numpy as np
import pandas as pd
import cvxpy as cp
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
from scipy.optimize import minimize
import yfinance as yf
import os
import webbrowser

# ── Plotly renderer (browser works everywhere outside Jupyter) ──
pio.renderers.default = "browser"

# ── Pandas display formatting ──
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.4f}".format)

RF_ANNUAL = 0.075   # Risk-free rate (7.5% — change as needed)
RF_DAILY  = RF_ANNUAL / 252

# ════════════════════════════════════════════════════════════
# 1. TICKER SEARCH
# ════════════════════════════════════════════════════════════

def search_ticker():
    """Search Yahoo Finance for a ticker by company name."""
    query = input("Enter company name: ").strip()
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        quotes = response.json().get("quotes", [])
        if not quotes:
            print("No results found. Try a different name.")
            return search_ticker()
        for i, q in enumerate(quotes[:10], 1):
            print(f"  {i}. {q.get('shortname','N/A')}  ({q['symbol']})")
        choice = int(input("Choose (1-10): "))
        return quotes[choice - 1]["symbol"]
    except Exception as e:
        print(f"Search failed: {e}")
        sys.exit(1)

def get_best_ticker(symbol, start_date, end_date):
    """
    If ticker is Indian (.NS or .BO), compare both exchanges
    and pick the one with more data.
    For all other tickers, use as-is.
    """
    is_indian = symbol.endswith('.NS') or symbol.endswith('.BO')
    
    if not is_indian:
        # Non-Indian ticker — trust it as-is, no comparison
        print(f"  Using {symbol} as provided.")
        return symbol
    
    # Indian stock — compare NS vs BO
    base = symbol.replace('.NS', '').replace('.BO', '')
    candidates = [f"{base}.NS", f"{base}.BO"]
    
    best_ticker = symbol
    best_count  = 0
    
    print(f"  Checking NSE vs BSE data for {base}...")
    for candidate in candidates:
        try:
            data = yf.download(candidate, start=start_date, end=end_date,
                               auto_adjust=True, progress=False)
            count = len(data)
            print(f"    {candidate}: {count} trading days")
            if count > best_count:
                best_count  = count
                best_ticker = candidate
        except Exception:
            print(f"    {candidate}: no data")
            continue
    
    print(f"  ✅ Selected: {best_ticker} ({best_count} days)")
    return best_ticker

def collect_tickers():
    """Collect tickers and date range from the user."""
    n = int(input("How many stocks in the portfolio? "))
    start = input("Start date (YYYY-MM-DD): ").strip()
    end   = input("End date   (YYYY-MM-DD): ").strip()
    tickers = []
    for i in range(n):
        print(f"\nSearch stock {i+1}:")
        raw_ticker = search_ticker()
        best = get_best_ticker(raw_ticker, start, end)
        
        tickers.append(best)
    print(f"\nFinal tickers selected: {tickers}")
    
    return tickers, start, end


# ════════════════════════════════════════════════════════════
# 2. DATA LOADING  +  SHRINKAGE-ADJUSTED EXPECTED RETURNS
# ════════════════════════════════════════════════════════════
#
#  WHY SHRINKAGE?
#  Raw sample means from historical daily returns are noisy.
#  A stock that happened to run +300% in your window will show
#  a massive daily mean, and the optimizer will pile into it —
#  even though that run is unlikely to repeat.
#
#  Shrinking toward a broad market index pulls outliers back to
#  something economically defensible while preserving the
#  relative ranking of stocks (good stocks still look better
#  than bad ones, just less extreme).
#
#  Formula:  adjusted_i = α * raw_i  +  (1-α) * benchmark_mean
#
#  This is a James-Stein style shrinkage estimator.
#  α = 1.0  → pure historical mean  (no shrinkage)
#  α = 0.0  → everything collapses to the benchmark mean
#  α = 0.7  → default: 70% own history, 30% market anchor
# ────────────────────────────────────────────────────────────

# Default shrinkage weight on each stock's own history.
# Exposed as a constant so gui_portfolio.py can import and
# offer it as a slider without hard-coding the value here.
SHRINKAGE_ALPHA = 0.70

# Benchmark candidates tried in order of preference.
# The auto-detector picks the right list based on whether
# the majority of tickers are Indian (.NS / .BO) or global.
_INDIAN_BENCHMARKS = ["^NSEI", "^BSESN"]   # Nifty 50, BSE Sensex
_GLOBAL_BENCHMARKS = ["^GSPC", "^IXIC"]    # S&P 500, Nasdaq


def _detect_benchmark_candidates(tickers):
    """Return the ordered list of benchmark symbols to try."""
    indian_count = sum(1 for t in tickers
                       if t.endswith(".NS") or t.endswith(".BO"))
    if indian_count / max(len(tickers), 1) >= 0.5:
        return _INDIAN_BENCHMARKS
    return _GLOBAL_BENCHMARKS


def _fetch_benchmark_daily_mean(candidates, start_date, end_date):
    """
    Try each benchmark symbol in order.
    Returns (symbol_used, daily_mean_return) or (None, None) on total failure.
    """
    for sym in candidates:
        try:
            data = yf.download(sym, start=start_date, end=end_date,
                               auto_adjust=True, progress=False)
            if data.empty:
                continue
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]   # single-ticker download → first col
            ret = close.pct_change().dropna()
            if len(ret) < 20:              # too few rows → unreliable
                continue
            mean_daily = float(ret.mean())
            print(f"  Benchmark : {sym}  |  {len(ret)} days  |  "
                  f"daily mean = {mean_daily*100:.4f}%  "
                  f"({mean_daily*252*100:.2f}% ann.)")
            return sym, mean_daily
        except Exception:
            continue
    return None, None


def _extract_close_frame(data, requested_tickers):
    if data is None or getattr(data, "empty", True):
        return pd.DataFrame(columns=requested_tickers)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]] if "Close" in data.columns else data
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=requested_tickers[0])
    return prices


def _download_prices_with_repair(tickers, start_date, end_date):
    """
    Download a basket, then retry missing tickers individually.
    This catches Yahoo/yfinance cases where a symbol is valid on the website
    but gets dropped from a multi-ticker batch response.
    """
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    prices = _extract_close_frame(data, tickers)

    repaired = []
    for ticker in tickers:
        if ticker in prices.columns and prices[ticker].dropna().size > 0:
            continue
        try:
            single = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            single_prices = _extract_close_frame(single, [ticker])
            if ticker in single_prices.columns and single_prices[ticker].dropna().size > 0:
                prices[ticker] = single_prices[ticker]
                repaired.append(ticker)
                continue
        except Exception:
            pass
        try:
            hist = yf.Ticker(ticker).history(
                start=start_date,
                end=end_date,
                auto_adjust=True,
            )
            if not hist.empty and "Close" in hist.columns:
                prices[ticker] = hist["Close"]
                repaired.append(ticker)
        except Exception:
            continue

    if repaired:
        print(f"Recovered individually after batch download failure: {repaired}")
    return prices


def _trim_to_common_window(prices, tickers):
    first_valid = {}
    last_valid = {}
    for ticker in tickers:
        series = prices[ticker].dropna() if ticker in prices.columns else pd.Series(dtype=float)
        if series.empty:
            continue
        first_valid[ticker] = series.index[0]
        last_valid[ticker] = series.index[-1]

    valid_tickers = [t for t in tickers if t in first_valid and t in last_valid]
    if len(valid_tickers) < 2:
        raise ValueError("Not enough assets with valid price history in the selected date range.")

    common_start = max(first_valid[t] for t in valid_tickers)
    common_end = min(last_valid[t] for t in valid_tickers)
    if common_start >= common_end:
        raise ValueError(
            "No overlapping trading window exists across all selected tickers. "
            "Reduce the basket or use a later start date."
        )

    trimmed = prices.loc[common_start:common_end, valid_tickers].copy()
    return trimmed, valid_tickers, common_start, common_end


def load_data(tickers, start_date, end_date, alpha=SHRINKAGE_ALPHA):
    """
    Download price data, compute returns + covariance, and apply
    shrinkage to expected returns.

    Parameters
    ----------
    tickers    : list[str]
    start_date : str  (YYYY-MM-DD)
    end_date   : str  (YYYY-MM-DD)
    alpha      : float in [0, 1]
                 Weight on each stock's own historical mean.
                 (1-alpha) goes to the benchmark mean.
                 Default = SHRINKAGE_ALPHA = 0.70

    Returns
    -------
    tickers           : list[str]   validated & ordered
    returns           : DataFrame   daily log-price changes
    expected_returns  : Series      shrinkage-adjusted daily means
    cov_matrix        : DataFrame   symmetrised sample covariance
    raw_er            : Series      unadjusted daily means (for display)
    benchmark_symbol  : str | None  benchmark actually used
    """
    print("\nDownloading price data...")
    prices = _download_prices_with_repair(tickers, start_date, end_date)

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"Warning: no data for {missing}. They will be dropped.")
        tickers = [t for t in tickers if t not in missing]

    prices = prices[tickers].sort_index()
    prices, tickers, common_start, common_end = _trim_to_common_window(prices, tickers)
    print(
        "  Optimizer window aligned to common overlap: "
        f"{common_start.strftime('%Y-%m-%d')} → {common_end.strftime('%Y-%m-%d')}"
    )
    prices = prices.ffill()

    returns = prices.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    # Drop assets that still have too little usable history in the chosen window.
    valid_counts = returns.count()
    too_sparse = [t for t in tickers if valid_counts.get(t, 0) < 20]
    if too_sparse:
        print(f"Warning: insufficient return history for {too_sparse}. They will be dropped.")
        tickers = [t for t in tickers if t not in too_sparse]
        prices = prices[tickers]
        returns = returns[tickers]

    if len(tickers) < 2:
        raise ValueError("Not enough assets with valid price history in the selected date range.")

    if returns.empty or len(returns) < 20:
        raise ValueError(
            "No usable overlapping return history found. Try a later start date or remove newer/staler symbols."
        )

    raw_er     = returns.mean()                          # unadjusted
    cov_raw    = returns.cov()
    cov_matrix = (cov_raw + cov_raw.T) / 2              # enforce symmetry

    # ── Shrinkage toward benchmark ────────────────────────────
    print(f"\nFetching benchmark for shrinkage  (α={alpha:.2f})…")
    candidates          = _detect_benchmark_candidates(tickers)
    bm_sym, bm_mean     = _fetch_benchmark_daily_mean(candidates, start_date, end_date)

    if bm_mean is None or alpha >= 1.0:
        if bm_mean is None:
            print("  ⚠  All benchmarks failed — using raw expected returns (no shrinkage).")
        expected_returns = raw_er.copy()
        bm_sym = None
    else:
        # Core shrinkage formula
        expected_returns = alpha * raw_er + (1.0 - alpha) * bm_mean

        # ── Transparency table ────────────────────────────────
        print(f"\n  {'Ticker':<22} {'Raw ann.%':>10}  {'Adj ann.%':>10}  "
              f"{'Δ ann.pp':>9}  Note")
        print("  " + "─" * 65)
        for t in tickers:
            raw_a = float(raw_er[t])   * 252 * 100
            adj_a = float(expected_returns[t]) * 252 * 100
            delta = adj_a - raw_a
            note  = "  ◀ pulled in hard" if abs(delta) > 10 else (
                    "  ◀ adjusted"       if abs(delta) >  3 else "")
            print(f"  {t:<22} {raw_a:>10.2f}  {adj_a:>10.2f}  {delta:>+9.2f}{note}")

    print(f"\n  Loaded {len(returns)} trading days for {len(tickers)} assets.\n")
    return tickers, returns, expected_returns, cov_matrix, raw_er, bm_sym


# ════════════════════════════════════════════════════════════
# 3. UTILITY-MAXIMIZED PORTFOLIO (CVXPY)
# ════════════════════════════════════════════════════════════

def utility_portfolio(expected_returns, cov_matrix, tickers,min_w,max_w):
    """Maximize E[R] - 0.5 * Var(R) subject to long-only + cap constraints."""
    n = len(expected_returns)
    w = cp.Variable(n)

    objective   = cp.Maximize(expected_returns.values @ w - 0.5 * cp.quad_form(w, cov_matrix.values))
    constraints = [cp.sum(w) == 1, w >= min_w, w <= max_w]

    cp.Problem(objective, constraints).solve(solver=cp.OSQP)
    weights = w.value

    if weights is None:
        print("Utility optimization failed.")
        return None

    ann_ret  = expected_returns.values @ weights * 252
    ann_risk = np.sqrt(weights.T @ cov_matrix.values @ weights) * np.sqrt(252)
    sharpe   = (ann_ret - RF_ANNUAL) / ann_risk

    print("=" * 50)
    print("UTILITY-MAXIMIZED PORTFOLIO")
    print("=" * 50)
    for t, wt in zip(tickers, weights):
        print(f"  {t:<20} {wt:.4f}")
    print(f"\n  Annual Return : {ann_ret*100:.2f}%")
    print(f"  Annual Risk   : {ann_risk*100:.2f}%")
    print(f"  Sharpe Ratio  : {sharpe:.3f}")
    return weights


# ════════════════════════════════════════════════════════════
# 4. CORRELATION HEATMAP
# ════════════════════════════════════════════════════════════

def plot_correlation(returns):
    """Plot seaborn correlation heatmap."""
    corr = returns.corr()
    print("\nCorrelation Matrix:")
    print(corr.round(2))
    ax = sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.xticks(rotation=90)
    plt.title("Asset Correlation Matrix")
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════
# 5. EFFICIENT FRONTIER (CVXPY)
# ════════════════════════════════════════════════════════════

def compute_frontier(expected_returns, cov_matrix,min_w,max_w,n_points=100):
    """Compute the efficient frontier via min-variance optimization."""
    n = len(expected_returns)
    target_returns = np.linspace(
        expected_returns.min() * 252,
        expected_returns.max() * 252 * 0.9,
        n_points
    )
    frontier_risks, frontier_returns = [], []

    for r_target in target_returns:
        w_m = cp.Variable(n)
        constraints = [
            cp.sum(w_m) == 1,
            w_m >= min_w,
            w_m <=  max_w,
            expected_returns.values @ w_m >= r_target / 252
        ]
        prob = cp.Problem(cp.Minimize(cp.quad_form(w_m, cov_matrix.values)), constraints)
        prob.solve(solver=cp.OSQP)

        if w_m.value is None:
            continue
        wt = w_m.value
        risk = np.sqrt(wt.T @ cov_matrix.values @ wt) * np.sqrt(252)
        frontier_risks.append(risk)
        frontier_returns.append(r_target)

    print(f"Frontier computed: {len(frontier_risks)} points.")
    return frontier_risks, frontier_returns


# ════════════════════════════════════════════════════════════
# 6. TANGENCY PORTFOLIO (SCIPY — Max Sharpe)
# ════════════════════════════════════════════════════════════

def tangency_portfolio(expected_returns, cov_matrix, tickers, min_w, max_w):
    """Find the max-Sharpe (tangency) portfolio using a robust multi-start search."""
    n = len(expected_returns)
    cov_values = cov_matrix.values
    er_values = expected_returns.values

    def _portfolio_stats(w):
        ret = float(er_values @ w) * 252
        var = float(w.T @ cov_values @ w)
        risk = np.sqrt(max(var, 0.0)) * np.sqrt(252)
        return ret, risk

    def _neg_sharpe_safe(w):
        ret, risk = _portfolio_stats(w)
        if risk <= 1e-12 or not np.isfinite(risk):
            return 1e9
        sharpe = (ret - RF_ANNUAL) / risk
        if not np.isfinite(sharpe):
            return 1e9
        return -sharpe

    residual = 1.0 - min_w * n
    if residual < -1e-9:
        raise ValueError(
            f"Infeasible weight bounds: {n} assets with min_w={min_w:.4f} exceed 100% total allocation."
        )

    seeds = [np.full(n, 1.0 / n)]
    room = max_w - min_w
    for i in range(n):
        w0 = np.full(n, min_w)
        extra = max(residual, 0.0)
        add_i = min(room, extra)
        w0[i] += add_i
        leftover = extra - add_i
        if leftover > 1e-12:
            for j in range(n):
                if j == i:
                    continue
                add_j = min(room - (w0[j] - min_w), leftover)
                if add_j > 0:
                    w0[j] += add_j
                    leftover -= add_j
                if leftover <= 1e-12:
                    break
        if abs(w0.sum() - 1.0) <= 1e-8:
            seeds.append(w0)

    best = None
    for x0 in seeds:
        result = minimize(
            _neg_sharpe_safe,
            x0=x0,
            method='SLSQP',
            bounds=[(min_w, max_w)] * n,
            constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
            options={'ftol': 1e-12, 'maxiter': 1000}
        )
        if not result.success:
            continue
        candidate = np.clip(result.x, min_w, max_w)
        total = candidate.sum()
        if total <= 0 or not np.isfinite(total):
            continue
        candidate = candidate / total
        score = _neg_sharpe_safe(candidate)
        if best is None or score < best[0]:
            best = (score, candidate)

    if best is None:
        raise ValueError("Tangency optimisation failed to converge to a feasible solution.")

    weights = best[1]
    tan_return, tan_risk = _portfolio_stats(weights)
    tan_sharpe = (tan_return - RF_ANNUAL) / tan_risk

    print("\n" + "=" * 50)
    print("TANGENCY PORTFOLIO (MAX SHARPE)")
    print("=" * 50)
    for t, wt in zip(tickers, weights):
        print(f"  {t:<20} {wt:.4f}")
    print(f"\n  Annual Return : {tan_return*100:.2f}%")
    print(f"  Annual Risk   : {tan_risk*100:.2f}%")
    print(f"  Sharpe Ratio  : {tan_sharpe:.3f}")

    return weights, tan_return, tan_risk, tan_sharpe


# ════════════════════════════════════════════════════════════
# 7. MIN-RISK PORTFOLIO FOR TARGET RETURN (CVXPY)
# ════════════════════════════════════════════════════════════

def min_risk_portfolio(expected_returns, cov_matrix, tickers, target_return,min_w,max_w):
    """Minimize risk subject to achieving at least target_return."""
    n = len(expected_returns)
    w1 = cp.Variable(n)
    constraints = [
        cp.sum(w1) == 1,
        w1 >= min_w,
        w1 <= max_w,
        expected_returns.values @ w1 >= target_return / 252
    ]
    cp.Problem(cp.Minimize(cp.quad_form(w1, cov_matrix.values)), constraints).solve(solver=cp.OSQP)
    weights = w1.value

    if weights is None:
        print(f"No feasible portfolio for {target_return*100:.1f}% return. Try a lower target.")
        return None

    ann_ret  = expected_returns.values @ weights * 252
    ann_risk = np.sqrt(weights.T @ cov_matrix.values @ weights) * np.sqrt(252)
    sharpe   = (ann_ret - RF_ANNUAL) / ann_risk

    print("\n" + "=" * 50)
    print(f"MIN-RISK PORTFOLIO  (target ≥ {target_return*100:.1f}%)")
    print("=" * 50)
    for t, wt in zip(tickers, weights):
        print(f"  {t:<20} {wt:.4f}")
    print(f"\n  Annual Return : {ann_ret*100:.2f}%")
    print(f"  Annual Risk   : {ann_risk*100:.2f}%")
    print(f"  Sharpe Ratio  : {sharpe:.3f}")
    return weights


# ════════════════════════════════════════════════════════════
# 8. MATPLOTLIB STATIC PLOT
# ════════════════════════════════════════════════════════════

def plot_frontier_static(frontier_risks, frontier_returns, asset_risks,
                          asset_returns, tan_risk, tan_return, tan_sharpe, tickers):
    cml_x = np.linspace(0, max(frontier_risks) * 1.2, 100)
    cml_y = RF_ANNUAL + tan_sharpe * cml_x

    plt.figure(figsize=(10, 6))
    plt.plot(frontier_risks, frontier_returns, color="darkblue", linewidth=3, label="Efficient Frontier")
    plt.scatter(asset_risks, asset_returns, color="red", s=80, zorder=5, label="Assets")
    plt.plot(cml_x, cml_y, linestyle="--", color="green", linewidth=2, label="Capital Market Line")
    plt.scatter(tan_risk, tan_return, marker="*", s=350, color="gold", zorder=6, label="Tangency Portfolio")

    for ticker in tickers:
        plt.annotate(ticker, (asset_risks[ticker], asset_returns[ticker]),
                     xytext=(6, 6), textcoords="offset points", fontsize=8)

    plt.xlabel("Annual Risk (σ)", fontsize=12)
    plt.ylabel("Annual Return (μ)", fontsize=12)
    plt.title("Efficient Frontier & Capital Market Line", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════
# 9. PLOTLY INTERACTIVE PLOT + 3D
# ════════════════════════════════════════════════════════════

def plot_frontier_interactive(frontier_risks, frontier_returns, asset_risks,
                               asset_returns, tan_risk, tan_return, tan_sharpe, tickers):
    # Convert to percentages
    fr_pct   = [r * 100 for r in frontier_returns]
    fsk_pct  = [r * 100 for r in frontier_risks]
    ar_pct   = np.array([asset_returns[t] * 100 for t in tickers])
    ask_pct  = np.array([asset_risks[t]   * 100 for t in tickers])
    tan_ret_pct  = tan_return * 100
    tan_risk_pct = tan_risk   * 100
    cml_x_pct = np.linspace(0, max(fsk_pct) * 1.3, 100)
    cml_y_pct = RF_ANNUAL * 100 + tan_sharpe * cml_x_pct
    frontier_sharpes = [(r - RF_ANNUAL*100) / risk for r, risk in zip(fr_pct, fsk_pct)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fsk_pct, y=fr_pct, mode='lines', name='Efficient Frontier',
        line=dict(color="royalblue", width=3),
        hovertemplate="Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"))

    fig.add_trace(go.Scatter(x=ask_pct, y=ar_pct, mode='markers+text', text=tickers,
        textposition="top center", marker=dict(size=10, color="tomato"), name='Assets',
        hovertemplate="<b>%{text}</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"))

    fig.add_trace(go.Scatter(x=[tan_risk_pct], y=[tan_ret_pct], mode='markers',
        marker=dict(size=18, color='gold', symbol='star'), name='Tangency Portfolio',
        hovertemplate=(f"<b>Tangency</b><br>Sharpe: {tan_sharpe:.3f}<br>"
                       "Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>")))

    fig.add_trace(go.Scatter(x=cml_x_pct, y=cml_y_pct, mode='lines',
        line=dict(color='limegreen', dash='dash', width=2), name='CML',
        hovertemplate="Risk: %{x:.2f}%<br>CML Return: %{y:.2f}%<extra></extra>"))

    fig.add_trace(go.Scatter(x=[fsk_pct[0]], y=[fr_pct[0]], mode='markers',
        marker=dict(size=14, color='white', symbol='circle', line=dict(color='royalblue', width=2)),
        name='Selected Point',
        hovertemplate="Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"))

    steps = []
    for i in range(len(fr_pct)):
        steps.append(dict(
            method="update",
            args=[{"x": [fsk_pct, ask_pct.tolist(), [tan_risk_pct], cml_x_pct.tolist(), [fsk_pct[i]]],
                   "y": [fr_pct,  ar_pct.tolist(),  [tan_ret_pct],  cml_y_pct.tolist(), [fr_pct[i]]]}],
            label=f"{fr_pct[i]:.1f}%"
        ))

    fig.update_layout(
        sliders=[dict(active=0, currentvalue={"prefix": "Target Return: "},
                      pad={"t": 50}, steps=steps,transition={"duration": 0})],
        title="Efficient Frontier with Tangency Portfolio & CML",
        xaxis_title="Annual Risk (σ) %", yaxis_title="Annual Return (μ) %",
        template="plotly_dark", width=950, height=620
    )
   
    output_path    = os.path.join(os.getcwd(), "quansen_frontier.html")
    fig.write_html(output_path)
    
  
    webbrowser.open(f"file://{output_path}")
    print(f"Plot saved and opened: {output_path}")

    

    # 3D plot
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(
        x=fsk_pct, y=fr_pct, z=frontier_sharpes, mode='lines+markers',
        marker=dict(size=4, color=frontier_sharpes, colorscale='Plasma'),
        line=dict(width=5), name="Frontier",
        hovertemplate="Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{z:.3f}<extra></extra>"))
    fig3d.add_trace(go.Scatter3d(
        x=[tan_risk_pct], y=[tan_ret_pct], z=[tan_sharpe], mode='markers',
        marker=dict(size=10, color='gold', symbol='diamond'), name='Tangency'))
    fig3d.update_layout(
        scene=dict(xaxis_title="Risk (%)", yaxis_title="Return (%)", zaxis_title="Sharpe"),
        title="3D Efficient Frontier — Risk / Return / Sharpe")
    # Same for 3D plot
    output_path_3d = os.path.join(os.getcwd(), "quansen_frontier_3d.html")
    fig3d.write_html(output_path_3d)
    webbrowser.open(f"file://{output_path_3d}")

# ════════════════════════════════════════════════════════════
# 10. MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 50)
    print("  QUANTITATIVE PORTFOLIO OPTIMIZER  v1.0")
    print("=" * 50 + "\n")

    # ── Input mode ──
    mode = input("Use test portfolio? (y/n): ").strip().lower()
    if mode == 'y':
        tickers    = ['BEL.NS', 'VEDL.NS', "BAJFINANCE.NS","BEML.NS","ADVAIT.BO","ADANIENT.NS","COALINDIA.NS","CROMPTON.NS","KINGFA.NS","KRISHNADEF.NS","LT.NS","LUPIN.NS","MAZDOCK.NS","PENIND.NS","PNB.NS","RELIANCE.NS","SHUKRAPHAR.BO","TARIL.NS","HDFCBANK.NS","SBIN.NS"]
        start_date = "2021-02-01"
        end_date   = "2026-03-06"
    else:
        tickers, start_date, end_date = collect_tickers()

    # ── Load data ──
    alpha_input = input(
        f"Shrinkage alpha (0–1, default {SHRINKAGE_ALPHA})  "
        "[Enter to use default]: "
    ).strip()
    alpha = float(alpha_input) if alpha_input else SHRINKAGE_ALPHA
    alpha = max(0.0, min(1.0, alpha))   # clamp to valid range

    tickers, returns, expected_returns, cov_matrix, raw_er, bm_sym = load_data(
        tickers, start_date, end_date, alpha=alpha)
    n = len(tickers)

    if n <= 3:
        max_w = 0.40
    elif n <= 10:
        max_w = 0.30
    elif n <= 20:
        max_w = 0.20
    elif n <= 40:
        max_w = 0.15
    else:
        max_w = 0.10
    
    min_w = 0.01

    # ── Step 1: Utility-maximized portfolio ──
    weights_utility= utility_portfolio(expected_returns, cov_matrix, tickers,min_w,max_w)

    # ── Step 2: Correlation heatmap ──
    plot_correlation(returns)

    # ── Step 3: Efficient frontier ──
    frontier_risks, frontier_returns = compute_frontier(expected_returns, cov_matrix,min_w,max_w)

    # ── Step 4: Asset-level stats ──
    asset_returns = expected_returns * 252
    asset_risks   = returns.std() * np.sqrt(252)

    # ── Step 5: Tangency portfolio ──
    weights_tan, tan_return, tan_risk, tan_sharpe = tangency_portfolio(
        expected_returns, cov_matrix, tickers,min_w,max_w)

    # ── Step 6: Static plot ──
    plot_frontier_static(frontier_risks, frontier_returns, asset_risks,
                         asset_returns, tan_risk, tan_return, tan_sharpe, tickers)

    # ── Step 7: Interactive Plotly plots ──
    plot_frontier_interactive(frontier_risks, frontier_returns, asset_risks,
                              asset_returns, tan_risk, tan_return, tan_sharpe, tickers)

    # ── Step 8: Min-risk for target return ──
    print("\n")
    target = float(input("Enter target annual return (decimal, e.g. 0.15 for 15%): "))
    weights_min= min_risk_portfolio(expected_returns, cov_matrix, tickers, target,min_w,max_w)

    print("\n✅ Analysis complete.\n")

    # ── Save weights to CSV ──
    weights_df = pd.DataFrame({
        'Ticker': tickers,
        'Utility_Weight':   weights_utility,
        'Tangency_Weight':  weights_tan,
        'MinRisk_Weight':   weights_min
    })
    weights_df['Utility_Pct']  = (weights_df['Utility_Weight']  * 100).round(2)
    weights_df['Tangency_Pct'] = (weights_df['Tangency_Weight'] * 100).round(2)
    weights_df['MinRisk_Pct']  = (weights_df['MinRisk_Weight']  * 100).round(2)

    csv_path = os.path.join(os.getcwd(), "quansen_weights.csv")
    weights_df.to_csv(csv_path, index=False)
    print(f"\n✅ Weights saved to: {csv_path}")


if __name__ == "__main__":
    main()
