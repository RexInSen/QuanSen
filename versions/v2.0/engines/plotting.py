"""
============================================================
  QuanSen — Plotting
============================================================

  All chart and visualisation functions.

  Public API
  ----------
  plot_correlation(returns)
  plot_frontier_static(...)
  plot_frontier_interactive(...)    also saves + opens 3D plot
============================================================
"""

import os
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio

from engines.config import RF_ANNUAL

pio.renderers.default = "browser"


# ════════════════════════════════════════════════════════════
# CORRELATION HEATMAP
# ════════════════════════════════════════════════════════════

def plot_correlation(returns):
    """Print and plot a seaborn correlation heatmap."""
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
# STATIC MATPLOTLIB FRONTIER
# ════════════════════════════════════════════════════════════

def plot_frontier_static(frontier_risks, frontier_returns, asset_risks,
                          asset_returns, tan_risk, tan_return, tan_sharpe, tickers):
    """Matplotlib efficient frontier + CML + tangency point."""
    cml_x = np.linspace(0, max(frontier_risks) * 1.2, 100)
    cml_y = RF_ANNUAL + tan_sharpe * cml_x

    plt.figure(figsize=(10, 6))
    plt.plot(frontier_risks, frontier_returns,
             color="darkblue", linewidth=3, label="Efficient Frontier")
    plt.scatter(asset_risks, asset_returns,
                color="red", s=80, zorder=5, label="Assets")
    plt.plot(cml_x, cml_y,
             linestyle="--", color="green", linewidth=2, label="Capital Market Line")
    plt.scatter(tan_risk, tan_return,
                marker="*", s=350, color="gold", zorder=6, label="Tangency Portfolio")

    for ticker in tickers:
        plt.annotate(ticker, (asset_risks[ticker], asset_returns[ticker]),
                     xytext=(6, 6), textcoords="offset points", fontsize=8)

    plt.xlabel("Annual Risk (sigma)", fontsize=12)
    plt.ylabel("Annual Return (mu)", fontsize=12)
    plt.title("Efficient Frontier & Capital Market Line", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════
# INTERACTIVE PLOTLY FRONTIER  (2D slider + 3D surface)
# ════════════════════════════════════════════════════════════

def plot_frontier_interactive(frontier_risks, frontier_returns, asset_risks,
                               asset_returns, tan_risk, tan_return, tan_sharpe, tickers):
    """
    Build and open two Plotly plots:
      1. 2D interactive frontier with a target-return slider
      2. 3D frontier surface coloured by Sharpe ratio

    Both are saved as HTML files in the current working directory.
    """
    # ── Convert everything to % for display ──────────────────
    fr_pct       = [r * 100 for r in frontier_returns]
    fsk_pct      = [r * 100 for r in frontier_risks]
    ar_pct       = np.array([asset_returns[t] * 100 for t in tickers])
    ask_pct      = np.array([asset_risks[t]   * 100 for t in tickers])
    tan_ret_pct  = tan_return * 100
    tan_risk_pct = tan_risk   * 100
    cml_x_pct    = np.linspace(0, max(fsk_pct) * 1.3, 100)
    cml_y_pct    = RF_ANNUAL * 100 + tan_sharpe * cml_x_pct
    frontier_sharpes = [(r - RF_ANNUAL * 100) / risk
                        for r, risk in zip(fr_pct, fsk_pct)]

    # ── 2D interactive plot ───────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fsk_pct, y=fr_pct, mode="lines", name="Efficient Frontier",
        line=dict(color="royalblue", width=3),
        hovertemplate="Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"))

    fig.add_trace(go.Scatter(
        x=ask_pct, y=ar_pct, mode="markers+text", text=tickers,
        textposition="top center", marker=dict(size=10, color="tomato"), name="Assets",
        hovertemplate="<b>%{text}</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"))

    fig.add_trace(go.Scatter(
        x=[tan_risk_pct], y=[tan_ret_pct], mode="markers",
        marker=dict(size=18, color="gold", symbol="star"), name="Tangency Portfolio",
        hovertemplate=(f"<b>Tangency</b><br>Sharpe: {tan_sharpe:.3f}<br>"
                       "Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>")))

    fig.add_trace(go.Scatter(
        x=cml_x_pct, y=cml_y_pct, mode="lines",
        line=dict(color="limegreen", dash="dash", width=2), name="CML",
        hovertemplate="Risk: %{x:.2f}%<br>CML Return: %{y:.2f}%<extra></extra>"))

    fig.add_trace(go.Scatter(
        x=[fsk_pct[0]], y=[fr_pct[0]], mode="markers",
        marker=dict(size=14, color="white", symbol="circle",
                    line=dict(color="royalblue", width=2)),
        name="Selected Point",
        hovertemplate="Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"))

    steps = [
        dict(method="update",
             args=[{"x": [fsk_pct, ask_pct.tolist(), [tan_risk_pct],
                           cml_x_pct.tolist(), [fsk_pct[i]]],
                    "y": [fr_pct,  ar_pct.tolist(),  [tan_ret_pct],
                          cml_y_pct.tolist(), [fr_pct[i]]]}],
             label=f"{fr_pct[i]:.1f}%")
        for i in range(len(fr_pct))
    ]

    fig.update_layout(
        sliders=[dict(active=0, currentvalue={"prefix": "Target Return: "},
                      pad={"t": 50}, steps=steps, transition={"duration": 0})],
        title="Efficient Frontier with Tangency Portfolio & CML",
        xaxis_title="Annual Risk (sigma) %",
        yaxis_title="Annual Return (mu) %",
        template="plotly_dark", width=950, height=620
    )

    out_2d = os.path.join(os.getcwd(), "quansen_frontier.html")
    fig.write_html(out_2d)
    webbrowser.open(f"file://{out_2d}")
    print(f"Plot saved and opened: {out_2d}")

    # ── 3D frontier surface ───────────────────────────────────
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(
        x=fsk_pct, y=fr_pct, z=frontier_sharpes,
        mode="lines+markers",
        marker=dict(size=4, color=frontier_sharpes, colorscale="Plasma"),
        line=dict(width=5), name="Frontier",
        hovertemplate="Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{z:.3f}<extra></extra>"))

    fig3d.add_trace(go.Scatter3d(
        x=[tan_risk_pct], y=[tan_ret_pct], z=[tan_sharpe],
        mode="markers",
        marker=dict(size=10, color="gold", symbol="diamond"),
        name="Tangency"))

    fig3d.update_layout(
        scene=dict(xaxis_title="Risk (%)", yaxis_title="Return (%)", zaxis_title="Sharpe"),
        title="3D Efficient Frontier — Risk / Return / Sharpe")

    out_3d = os.path.join(os.getcwd(), "quansen_frontier_3d.html")
    fig3d.write_html(out_3d)
    webbrowser.open(f"file://{out_3d}")
