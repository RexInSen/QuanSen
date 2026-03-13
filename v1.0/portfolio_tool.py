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
# 2. DATA LOADING
# ════════════════════════════════════════════════════════════

def load_data(tickers, start_date, end_date):
    """Download price data and compute returns + covariance."""
    print("\nDownloading data...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    prices = data["Close"]

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"Warning: no data for {missing}. They will be dropped.")
        tickers = [t for t in tickers if t not in missing]
        prices = prices[tickers]

    returns = prices.pct_change().dropna()
    expected_returns = returns.mean()
    cov_raw = returns.cov()
    cov_matrix = (cov_raw + cov_raw.T) / 2   # enforce symmetry

    print(f"Loaded {len(returns)} trading days for {len(tickers)} assets.\n")
    return tickers, returns, expected_returns, cov_matrix


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

def tangency_portfolio(expected_returns, cov_matrix, tickers,min_w,max_w):
    """Find the max-Sharpe (tangency) portfolio using scipy SLSQP."""
    n = len(expected_returns)

    def neg_sharpe(w):
        ret  = (expected_returns.values @ w) * 252
        risk = np.sqrt(w.T @ cov_matrix.values @ w) * np.sqrt(252)
        return -(ret - RF_ANNUAL) / risk

    result = minimize(
        neg_sharpe,
        x0=np.ones(n) / n,
        method='SLSQP',
        bounds=[(min_w, max_w)] * n,
        constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
        options={'ftol': 1e-12, 'maxiter': 1000}
    )

    weights = result.x
    tan_return = (expected_returns.values @ weights) * 252
    tan_risk   = np.sqrt(weights.T @ cov_matrix.values @ weights) * np.sqrt(252)
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
    tickers, returns, expected_returns, cov_matrix = load_data(tickers, start_date, end_date)
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
