"""
============================================================
  QuanSen — Portfolio Optimizers
============================================================

  All four optimization routines.  Every function takes
  (expected_returns, cov_matrix, tickers, min_w, max_w)
  and returns weights + stats.

  Public API
  ----------
  utility_portfolio(...)   → weights
  compute_frontier(...)    → frontier_risks, frontier_returns
  tangency_portfolio(...)  → weights, return, risk, sharpe
  min_risk_portfolio(...)  → weights
============================================================
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize

from engines.config import RF_ANNUAL


# ════════════════════════════════════════════════════════════
# UTILITY-MAXIMIZED PORTFOLIO  (CVXPY)
# Maximize E[R] - 0.5 * Var(R)  subject to long-only + cap
# ════════════════════════════════════════════════════════════

def utility_portfolio(expected_returns, cov_matrix, tickers, min_w, max_w):
    """
    Maximize mean-variance utility: E[R] - 0.5 * Var(R).
    Returns the weight array, or None if the solver fails.
    """
    n = len(expected_returns)
    w = cp.Variable(n)

    objective   = cp.Maximize(
        expected_returns.values @ w - 0.5 * cp.quad_form(w, cov_matrix.values)
    )
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
# EFFICIENT FRONTIER  (CVXPY)
# ════════════════════════════════════════════════════════════

def compute_frontier(expected_returns, cov_matrix, min_w, max_w, n_points=100):
    """
    Trace the efficient frontier by solving min-variance for a
    grid of target return levels.

    Returns
    -------
    frontier_risks   : list[float]   annualised risk per point
    frontier_returns : list[float]   annualised return per point
    """
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
            w_m <= max_w,
            expected_returns.values @ w_m >= r_target / 252
        ]
        prob = cp.Problem(cp.Minimize(cp.quad_form(w_m, cov_matrix.values)), constraints)
        prob.solve(solver=cp.OSQP)

        if w_m.value is None:
            continue
        wt   = w_m.value
        risk = np.sqrt(wt.T @ cov_matrix.values @ wt) * np.sqrt(252)
        frontier_risks.append(risk)
        frontier_returns.append(r_target)

    print(f"Frontier computed: {len(frontier_risks)} points.")
    return frontier_risks, frontier_returns


# ════════════════════════════════════════════════════════════
# TANGENCY PORTFOLIO  (SCIPY — max Sharpe)
# ════════════════════════════════════════════════════════════

def tangency_portfolio(expected_returns, cov_matrix, tickers, min_w, max_w):
    """
    Find the max-Sharpe (tangency) portfolio using a robust
    multi-start SLSQP search.

    Returns
    -------
    weights    : np.ndarray
    tan_return : float   annualised
    tan_risk   : float   annualised
    tan_sharpe : float
    """
    n          = len(expected_returns)
    cov_values = cov_matrix.values
    er_values  = expected_returns.values

    def _portfolio_stats(w):
        ret  = float(er_values @ w) * 252
        var  = float(w.T @ cov_values @ w)
        risk = np.sqrt(max(var, 0.0)) * np.sqrt(252)
        return ret, risk

    def _neg_sharpe_safe(w):
        ret, risk = _portfolio_stats(w)
        if risk <= 1e-12 or not np.isfinite(risk):
            return 1e9
        sharpe = (ret - RF_ANNUAL) / risk
        return -sharpe if np.isfinite(sharpe) else 1e9

    # Build feasible starting points
    residual = 1.0 - min_w * n
    if residual < -1e-9:
        raise ValueError(
            f"Infeasible weight bounds: {n} assets × min_w={min_w:.4f} exceeds 100%."
        )

    seeds = [np.full(n, 1.0 / n)]
    room  = max_w - min_w
    for i in range(n):
        w0       = np.full(n, min_w)
        extra    = max(residual, 0.0)
        add_i    = min(room, extra)
        w0[i]   += add_i
        leftover = extra - add_i
        if leftover > 1e-12:
            for j in range(n):
                if j == i:
                    continue
                add_j     = min(room - (w0[j] - min_w), leftover)
                if add_j > 0:
                    w0[j]    += add_j
                    leftover -= add_j
                if leftover <= 1e-12:
                    break
        if abs(w0.sum() - 1.0) <= 1e-8:
            seeds.append(w0)

    best = None
    for x0 in seeds:
        result = minimize(
            _neg_sharpe_safe, x0=x0, method="SLSQP",
            bounds=[(min_w, max_w)] * n,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
            options={"ftol": 1e-12, "maxiter": 1000}
        )
        if not result.success:
            continue
        candidate = np.clip(result.x, min_w, max_w)
        total     = candidate.sum()
        if total <= 0 or not np.isfinite(total):
            continue
        candidate = candidate / total
        score     = _neg_sharpe_safe(candidate)
        if best is None or score < best[0]:
            best = (score, candidate)

    if best is None:
        raise ValueError("Tangency optimisation failed to converge to a feasible solution.")

    weights    = best[1]
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
# MIN-RISK FOR TARGET RETURN  (CVXPY)
# ════════════════════════════════════════════════════════════

def min_risk_portfolio(expected_returns, cov_matrix, tickers, target_return, min_w, max_w):
    """
    Minimize portfolio variance subject to achieving at least
    `target_return` (annualised decimal, e.g. 0.15 for 15%).

    Returns weights array, or None if no feasible solution exists.
    """
    n  = len(expected_returns)
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
    print(f"MIN-RISK PORTFOLIO  (target >= {target_return*100:.1f}%)")
    print("=" * 50)
    for t, wt in zip(tickers, weights):
        print(f"  {t:<20} {wt:.4f}")
    print(f"\n  Annual Return : {ann_ret*100:.2f}%")
    print(f"  Annual Risk   : {ann_risk*100:.2f}%")
    print(f"  Sharpe Ratio  : {sharpe:.3f}")
    return weights
