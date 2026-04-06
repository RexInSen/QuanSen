"""
============================================================
  QuanSen — Quantitative Portfolio Optimizer  v1.1
  Author: Amatra Sen

  Entry point.  Run this file to start the optimizer.

  Module map
  ----------
  config.py          constants (RF rate, shrinkage alpha, benchmarks)
  ticker_search.py   Yahoo Finance search, NSE vs BSE picker
  data_loader.py     price download, repair, trim, shrinkage
  optimizers.py      utility / tangency / min-risk / frontier
  plotting.py        correlation heatmap, static + interactive charts
  market_state_engine.py   regime detection (alpha, beta)
  momentum_engine.py       cross-sectional momentum blend
============================================================
"""

import os
import numpy as np
import pandas as pd

from engines.config              import RF_ANNUAL, RF_DAILY
from engines.ticker_search       import collect_tickers
from engines.data_loader         import load_data
from engines.optimizers          import (utility_portfolio, compute_frontier,
                                 tangency_portfolio, min_risk_portfolio)
from engines.plotting            import (plot_correlation, plot_frontier_static,
                                 plot_frontier_interactive)
from engines.market_state_engine import detect_market_state
from engines.momentum_engine     import compute_momentum

# ── Pandas display formatting ────────────────────────────────
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.4f}".format)


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 50)
    print("  QUANTITATIVE PORTFOLIO OPTIMIZER  v1.1")
    print("=" * 50 + "\n")

    # ── Step 0: Get tickers & dates ──────────────────────────
    mode = input("Use test portfolio? (y/n): ").strip().lower()
    if mode == "y":
        today = pd.Timestamp.today().normalize()
        tickers = [
            "BEL.NS", "VEDL.NS", "BAJFINANCE.NS", "BEML.NS", "ADVAIT.BO",
            "ADANIENT.NS", "COALINDIA.NS", "CROMPTON.NS", "KINGFA.NS",
            "KRISHNADEF.NS", "LT.NS", "LUPIN.NS", "MAZDOCK.NS", "PENIND.NS",
            "PNB.NS", "RELIANCE.NS", "SHUKRAPHAR.BO", "TARIL.NS",
            "HDFCBANK.NS", "SBIN.NS"
        ]
        start_date = (today - pd.Timedelta(days=365 * 5)).strftime("%Y-%m-%d")
        end_date   = today.strftime("%Y-%m-%d")
    else:
        tickers, start_date, end_date = collect_tickers()

    # ── Step 1: Detect market regime ────────────────────────
    alpha_regime, beta_regime, regime_meta = detect_market_state(
        tickers, start_date, end_date
    )
    print("\nMarket regime:")
    print(regime_meta)

    # ── Step 2: Load data + shrinkage ───────────────────────
    alpha  = float(np.clip(alpha_regime, 0.0, 1.0))
    tickers, returns, expected_returns, cov_matrix, raw_er, bm_sym = load_data(
        tickers, start_date, end_date, alpha=alpha
    )
    n = len(tickers)

    # ── Step 3: Blend momentum into expected returns ─────────
    # beta_regime controls shrinkage vs momentum weight.
    #   beta = 1.0  ->  pure shrinkage (crisis / bear regime)
    #   beta ~ 0.30 ->  momentum tilts strongly (bull regime)
    # expected_returns is replaced so all optimizers downstream
    # automatically receive the momentum-adjusted returns.
    print(f"\nRunning momentum engine  (beta={beta_regime:.3f}, auto_lookback=True)...")
    mom_scores, mom_signals, expected_returns, mom_meta = compute_momentum(
        tickers    = tickers,
        start_date = start_date,
        end_date   = end_date,
        shrink_er  = expected_returns,
        beta       = beta_regime,
        auto_lookback = True,
    )
    print(f"  Momentum window : {mom_meta['lookback']}d lookback / {mom_meta['skip']}d skip")
    print(f"  Entry : {mom_meta['entry_date']}   Exit : {mom_meta['exit_date']}")
    print("\n  Momentum signals:")
    for t, sig, sc in zip(tickers, mom_signals, mom_scores):
        icon = "[+]" if sig == "Strong" else ("[-]" if sig == "Weak" else "[~]")
        print(f"    {icon}  {t:<22}  {sig:<8}  raw={sc*100:+.1f}%")

    # ── Step 4: Position-size bounds ────────────────────────
    if   n <= 3:  max_w = 0.40
    elif n <= 10: max_w = 0.30
    elif n <= 20: max_w = 0.20
    elif n <= 40: max_w = 0.15
    else:         max_w = 0.10
    min_w = 0.01

    # ── Step 5: Run optimizers ───────────────────────────────
    weights_utility = utility_portfolio(expected_returns, cov_matrix, tickers, min_w, max_w)

    plot_correlation(returns)

    frontier_risks, frontier_returns = compute_frontier(
        expected_returns, cov_matrix, min_w, max_w
    )

    asset_returns = expected_returns * 252
    asset_risks   = returns.std() * np.sqrt(252)

    weights_tan, tan_return, tan_risk, tan_sharpe = tangency_portfolio(
        expected_returns, cov_matrix, tickers, min_w, max_w
    )

    # ── Step 6: Plots ────────────────────────────────────────
    plot_frontier_static(
        frontier_risks, frontier_returns,
        asset_risks, asset_returns,
        tan_risk, tan_return, tan_sharpe, tickers
    )
    plot_frontier_interactive(
        frontier_risks, frontier_returns,
        asset_risks, asset_returns,
        tan_risk, tan_return, tan_sharpe, tickers
    )

    # ── Step 7: Min-risk for user-specified target ───────────
    print("\n")
    target      = float(input("Enter target annual return (decimal, e.g. 0.15 for 15%): "))
    weights_min = min_risk_portfolio(
        expected_returns, cov_matrix, tickers, target, min_w, max_w
    )

    print("\nAnalysis complete.\n")

    # ── Step 8: Save weights to CSV ─────────────────────────
    weights_df = pd.DataFrame({
        "Ticker":          tickers,
        "Utility_Weight":  weights_utility,
        "Tangency_Weight": weights_tan,
        "MinRisk_Weight":  weights_min,
    })
    weights_df["Utility_Pct"]  = (weights_df["Utility_Weight"]  * 100).round(2)
    weights_df["Tangency_Pct"] = (weights_df["Tangency_Weight"] * 100).round(2)
    weights_df["MinRisk_Pct"]  = (weights_df["MinRisk_Weight"]  * 100).round(2)

    csv_path = os.path.join(os.getcwd(), "quansen_weights.csv")
    weights_df.to_csv(csv_path, index=False)
    print(f"Weights saved to: {csv_path}")


if __name__ == "__main__":
    main()
