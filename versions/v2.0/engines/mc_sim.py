"""
============================================================
  QuanSen — Monte Carlo Simulation Engine
  engines/mc_sim.py

  Three simulation modes:
    1. Standard GBM            — run_mc_simulation()
    2. Regime-Switching GBM    — run_regime_switching_simulation()
    3. Stress Scenario Engine  — run_stress_scenarios()

  Standalone — does NOT import from any other QuanSen engine.
============================================================
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional


# ══════════════════════════════════════════════════════════════
# 1 ─ STANDARD GBM
# ══════════════════════════════════════════════════════════════

def run_mc_simulation(
    mu: float,
    sigma: float,
    initial_capital: float,
    horizon_years: float,
    n_sims: int,
    annual_contribution: float = 0.0,
    dt: float = 1 / 252,
    seed=None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng            = np.random.default_rng(seed)
    n_steps        = max(1, int(round(horizon_years / dt)))
    steps_per_year = int(round(1.0 / dt))
    drift = (mu - 0.5 * sigma ** 2) * dt
    vol   = sigma * np.sqrt(dt)
    z           = rng.standard_normal((n_steps, n_sims))
    log_returns = drift + vol * z
    log_cum     = np.cumsum(log_returns, axis=0)
    multipliers = np.vstack([np.ones((1, n_sims)), np.exp(log_cum)])
    paths       = initial_capital * multipliers
    if annual_contribution != 0.0:
        for year_idx in range(1, int(horizon_years) + 1):
            step = year_idx * steps_per_year
            if step <= n_steps:
                remaining = n_steps - step
                if remaining > 0:
                    contrib_mult = np.exp(np.cumsum(log_returns[step:, :], axis=0))
                    contrib_paths = annual_contribution * np.vstack([
                        np.ones((1, n_sims)), contrib_mult
                    ])
                    paths[step:, :] += contrib_paths
    t_axis = np.linspace(0, horizon_years, n_steps + 1)
    return paths, t_axis


# ══════════════════════════════════════════════════════════════
# 2 ─ REGIME-SWITCHING GBM
# ══════════════════════════════════════════════════════════════

REGIME_DEFAULTS = {
    "bull":     {"mu":  0.18,  "sigma": 0.12, "color": "#22c55e", "label": "Bull"},
    "sideways": {"mu":  0.06,  "sigma": 0.16, "color": "#ffd166", "label": "Sideways"},
    "bear":     {"mu": -0.12,  "sigma": 0.28, "color": "#ff4d6d", "label": "Bear"},
    "crisis":   {"mu": -0.35,  "sigma": 0.50, "color": "#9b5de5", "label": "Crisis"},
}
REGIME_ORDER = ["bull", "sideways", "bear", "crisis"]


def build_transition_matrix(regime_probs: dict, stickiness: float = 0.90) -> np.ndarray:
    n     = len(REGIME_ORDER)
    T     = np.zeros((n, n))
    probs = np.array([regime_probs.get(r, 0.25) for r in REGIME_ORDER], dtype=float)
    probs = probs / probs.sum()
    for i in range(n):
        T[i, i]  = stickiness
        off_mass = 1.0 - stickiness
        denom    = max(1.0 - probs[i], 1e-9)
        for j in range(n):
            if j != i:
                T[i, j] = off_mass * probs[j] / denom
        T[i] /= T[i].sum()
    return T


def run_regime_switching_simulation(
    initial_capital: float,
    horizon_years: float,
    n_sims: int,
    regime_probs: dict,
    regime_params=None,
    stickiness: float = 0.90,
    annual_contribution: float = 0.0,
    dt_regime: float = 1 / 52,
    dt_gbm: float = 1 / 252,
    seed=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Regime-switching Monte Carlo.

    At each weekly step the regime transitions via a Markov chain whose
    stationary distribution matches regime_probs.  Within each week,
    daily GBM draws use that week's regime mu/sigma.

    Returns
    -------
    paths        : (n_gbm_steps+1, n_sims)
    t_axis       : (n_gbm_steps+1,) in years
    regime_track : (n_regime_steps, n_sims)  int regime index per week
    """
    rng    = np.random.default_rng(seed)
    params = {**REGIME_DEFAULTS, **(regime_params or {})}

    T        = build_transition_matrix(regime_probs, stickiness)
    T_cumsum = np.cumsum(T, axis=1)

    n_regime_steps  = max(1, int(round(horizon_years / dt_regime)))
    gbm_per_regime  = max(1, int(round(dt_regime / dt_gbm)))
    n_gbm_steps     = n_regime_steps * gbm_per_regime
    steps_per_year  = int(round(1.0 / dt_gbm))

    init_probs = np.array([regime_probs.get(r, 0.25) for r in REGIME_ORDER], dtype=float)
    init_probs = init_probs / init_probs.sum()
    init_regime = rng.choice(len(REGIME_ORDER), size=n_sims, p=init_probs)

    regime_track    = np.zeros((n_regime_steps, n_sims), dtype=np.int8)
    regime_track[0] = init_regime
    for t in range(1, n_regime_steps):
        u = rng.random(n_sims)
        for s in range(n_sims):
            cur = int(regime_track[t - 1, s])
            regime_track[t, s] = int(np.searchsorted(T_cumsum[cur], u[s]))

    paths    = np.zeros((n_gbm_steps + 1, n_sims))
    paths[0] = initial_capital

    for rt in range(n_regime_steps):
        gbm_start = rt * gbm_per_regime
        gbm_end   = min(gbm_start + gbm_per_regime, n_gbm_steps)
        n_steps_w = gbm_end - gbm_start

        for ri, rname in enumerate(REGIME_ORDER):
            mask = (regime_track[rt] == ri)
            if not mask.any():
                continue
            n_m    = int(mask.sum())
            mu_r   = params[rname]["mu"]
            sig_r  = params[rname]["sigma"]
            drift  = (mu_r - 0.5 * sig_r ** 2) * dt_gbm
            vol    = sig_r * np.sqrt(dt_gbm)
            z      = rng.standard_normal((n_steps_w, n_m))
            lr     = drift + vol * z
            mults  = np.exp(np.cumsum(lr, axis=0))   # (n_steps_w, n_m)
            base   = paths[gbm_start, mask]            # (n_m,)
            for s in range(n_steps_w):
                gi = gbm_start + s + 1
                if gi <= n_gbm_steps:
                    paths[gi, mask] = base * mults[s]

        # safety: fill zeros
        for gi in range(gbm_start + 1, gbm_end + 1):
            if gi <= n_gbm_steps:
                zm = paths[gi] == 0
                if zm.any():
                    paths[gi, zm] = paths[gi - 1, zm]

    if annual_contribution != 0.0:
        for year_idx in range(1, int(horizon_years) + 1):
            step = year_idx * steps_per_year
            if 0 < step <= n_gbm_steps:
                paths[step:, :] += annual_contribution

    t_axis = np.linspace(0, horizon_years, n_gbm_steps + 1)
    return paths, t_axis, regime_track


# ══════════════════════════════════════════════════════════════
# 3 ─ STRESS SCENARIO ENGINE
# ══════════════════════════════════════════════════════════════

STRESS_SCENARIOS = {
    "2008 GFC":           {"shock": -0.45, "shock_duration_months": 15, "recovery_mu": 0.18, "recovery_sigma": 0.22, "color": "#ff4d6d"},
    "2020 COVID Crash":   {"shock": -0.34, "shock_duration_months":  2, "recovery_mu": 0.32, "recovery_sigma": 0.28, "color": "#ff8a65"},
    "2000 Dot-Com Bust":  {"shock": -0.50, "shock_duration_months": 30, "recovery_mu": 0.10, "recovery_sigma": 0.20, "color": "#ffd166"},
    "Flash Crash -20%":   {"shock": -0.20, "shock_duration_months":  1, "recovery_mu": 0.15, "recovery_sigma": 0.18, "color": "#9b5de5"},
    "Stagflation":        {"shock": -0.25, "shock_duration_months": 24, "recovery_mu": 0.06, "recovery_sigma": 0.20, "color": "#4a90d9"},
    "Mild Correction":    {"shock": -0.10, "shock_duration_months":  3, "recovery_mu": 0.12, "recovery_sigma": 0.15, "color": "#00e676"},
    "Custom":             {"shock": -0.30, "shock_duration_months":  6, "recovery_mu": 0.15, "recovery_sigma": 0.20, "color": "#80c8ff"},
}


def run_stress_scenario(
    mu: float,
    sigma: float,
    initial_capital: float,
    horizon_years: float,
    n_sims: int,
    shock_pct: float,
    shock_duration_months: float,
    shock_start_month: float,
    recovery_mu: float,
    recovery_sigma: float,
    annual_contribution: float = 0.0,
    dt: float = 1 / 252,
    seed=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Three-phase simulation: pre-shock GBM → distressed crash → recovery GBM.

    The shock magnitude is baked into the distressed drift so the median
    path hits approximately shock_pct by end of shock window.
    """
    rng             = np.random.default_rng(seed)
    n_steps         = max(1, int(round(horizon_years / dt)))
    steps_per_year  = int(round(1.0 / dt))
    steps_per_month = steps_per_year / 12.0

    shock_start_step = max(1, int(round(shock_start_month * steps_per_month)))
    shock_end_step   = min(n_steps, int(round(
        (shock_start_month + shock_duration_months) * steps_per_month
    )))

    # Phase parameters
    drift_n = (mu            - 0.5 * sigma            ** 2) * dt
    vol_n   =  sigma            * np.sqrt(dt)

    shock_ann_mu    = np.log(1 + max(shock_pct, -0.99)) / max(shock_duration_months / 12.0, 1/252)
    shock_ann_sigma = recovery_sigma * 1.8
    drift_s = (shock_ann_mu    - 0.5 * shock_ann_sigma ** 2) * dt
    vol_s   =  shock_ann_sigma * np.sqrt(dt)

    drift_r = (recovery_mu    - 0.5 * recovery_sigma  ** 2) * dt
    vol_r   =  recovery_sigma * np.sqrt(dt)

    n_pre  = shock_start_step
    n_shk  = max(0, shock_end_step - shock_start_step)
    n_rec  = max(0, n_steps - shock_end_step)

    z_pre = rng.standard_normal((n_pre,  n_sims)) if n_pre  > 0 else None
    z_shk = rng.standard_normal((n_shk,  n_sims)) if n_shk  > 0 else None
    z_rec = rng.standard_normal((n_rec,  n_sims)) if n_rec  > 0 else None

    paths    = np.zeros((n_steps + 1, n_sims))
    paths[0] = initial_capital

    if n_pre > 0 and z_pre is not None:
        lr = drift_n + vol_n * z_pre
        paths[1:n_pre + 1] = initial_capital * np.exp(np.cumsum(lr, axis=0))

    if n_shk > 0 and z_shk is not None:
        base = paths[shock_start_step]
        lr   = drift_s + vol_s * z_shk
        paths[shock_start_step + 1:shock_end_step + 1] = base * np.exp(np.cumsum(lr, axis=0))

    if n_rec > 0 and z_rec is not None:
        base = paths[shock_end_step]
        lr   = drift_r + vol_r * z_rec
        paths[shock_end_step + 1:n_steps + 1] = base * np.exp(np.cumsum(lr, axis=0))

    if annual_contribution != 0.0:
        for year_idx in range(1, int(horizon_years) + 1):
            step = year_idx * steps_per_year
            if 0 < step <= n_steps:
                paths[step:, :] += annual_contribution

    t_axis = np.linspace(0, horizon_years, n_steps + 1)
    return paths, t_axis


def run_all_stress_scenarios(
    mu: float,
    sigma: float,
    initial_capital: float,
    horizon_years: float,
    n_sims: int,
    selected_scenarios: List[str],
    shock_start_month: float = 6.0,
    annual_contribution: float = 0.0,
    custom_params: dict = None,
    dt: float = 1 / 252,
    seed=None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    results    = {}
    base_seed  = seed if seed is not None else 42
    scenarios  = {**STRESS_SCENARIOS}
    if custom_params:
        scenarios["Custom"] = {**scenarios["Custom"], **custom_params}

    for i, name in enumerate(selected_scenarios):
        if name not in scenarios:
            continue
        sp = scenarios[name]
        paths, t = run_stress_scenario(
            mu=mu, sigma=sigma,
            initial_capital=initial_capital,
            horizon_years=horizon_years,
            n_sims=n_sims,
            shock_pct=sp["shock"],
            shock_duration_months=sp["shock_duration_months"],
            shock_start_month=shock_start_month,
            recovery_mu=sp["recovery_mu"],
            recovery_sigma=sp["recovery_sigma"],
            annual_contribution=annual_contribution,
            dt=dt,
            seed=base_seed + i,
        )
        results[name] = (paths, t)
    return results


# ══════════════════════════════════════════════════════════════
# SHARED ANALYTICS
# ══════════════════════════════════════════════════════════════

def compute_mc_stats(
    paths: np.ndarray,
    initial_capital: float,
    annual_contribution: float = 0.0,
    horizon_years: float = 1.0,
) -> dict:
    finals         = paths[-1, :]
    total_invested = initial_capital + annual_contribution * max(int(horizon_years), 0)
    p5, p10, p25, p50, p75, p90, p95 = np.percentile(finals, [5, 10, 25, 50, 75, 90, 95])
    mean_val = float(np.mean(finals))
    std_val  = float(np.std(finals))
    n        = len(finals)

    prob_profit = float(np.sum(finals > total_invested) / n)
    prob_double = float(np.sum(finals > 2 * initial_capital) / n)
    prob_loss   = float(np.sum(finals < initial_capital) / n)
    prob_halved = float(np.sum(finals < 0.5 * initial_capital) / n)

    with np.errstate(divide="ignore", invalid="ignore"):
        cagr_paths = np.where(
            finals > 0,
            (finals / initial_capital) ** (1.0 / max(horizon_years, 0.001)) - 1.0,
            -1.0,
        )
    cagr_median = float(np.median(cagr_paths))

    loss_pct   = (finals - initial_capital) / initial_capital * 100.0
    var_95     = float(np.percentile(loss_pct, 5))
    worst_5pct = loss_pct[loss_pct <= var_95]
    cvar_95    = float(np.mean(worst_5pct)) if len(worst_5pct) > 0 else var_95

    median_path = np.median(paths, axis=1)
    running_max = np.maximum.accumulate(median_path)
    dd          = (median_path - running_max) / np.maximum(running_max, 1e-9)
    max_dd      = float(dd.min())

    return {
        "finals":         finals,
        "mean":           mean_val,
        "median":         float(p50),
        "p5":             float(p5),
        "p10":            float(p10),
        "p25":            float(p25),
        "p75":            float(p75),
        "p90":            float(p90),
        "p95":            float(p95),
        "std":            std_val,
        "prob_profit":    prob_profit,
        "prob_double":    prob_double,
        "prob_loss":      prob_loss,
        "prob_halved":    prob_halved,
        "cagr_median":    cagr_median,
        "var_95":         var_95,
        "cvar_95":        cvar_95,
        "max_drawdown":   max_dd,
        "total_invested": total_invested,
        "n_sims":         n,
    }


def build_histogram_data(finals: np.ndarray, n_bins: int = 80):
    valid = finals[finals > 0]
    if len(valid) == 0:
        return np.array([]), np.array([])
    log_min = np.log10(max(valid.min(), 1e-3))
    log_max = np.log10(valid.max() * 1.01)
    if log_min >= log_max:
        edges = np.linspace(valid.min(), valid.max() * 1.01, n_bins + 1)
    else:
        edges = np.logspace(log_min, log_max, n_bins + 1)
    counts, edges_out = np.histogram(finals, bins=edges)
    centres = 0.5 * (edges_out[:-1] + edges_out[1:])
    return centres, counts


def sample_paths_for_plot(paths, t_axis, n_sample=100, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(paths.shape[1], size=min(n_sample, paths.shape[1]), replace=False)
    return paths[:, idx], t_axis


def horizon_to_years(value: float, unit: str) -> float:
    return {"months": value / 12.0, "years": value}.get(unit, value)


def summarize_regime_paths(regime_track: np.ndarray) -> dict:
    """Summarise simulated regime paths in plain-language stats."""
    if regime_track is None or regime_track.size == 0:
        return {
            "avg_switches": 0.0,
            "median_switches": 0.0,
            "p90_switches": 0.0,
            "occupancy": {name: 0.0 for name in REGIME_ORDER},
            "start_regime": "unknown",
            "end_regime": "unknown",
        }

    switch_counts = np.sum(regime_track[1:] != regime_track[:-1], axis=0) if regime_track.shape[0] > 1 else np.zeros(regime_track.shape[1])
    flat = regime_track.reshape(-1)
    occ = {}
    for idx, name in enumerate(REGIME_ORDER):
        occ[name] = float(np.mean(flat == idx))

    start_mode = int(pd.Series(regime_track[0]).mode().iloc[0])
    end_mode = int(pd.Series(regime_track[-1]).mode().iloc[0])
    return {
        "avg_switches": float(np.mean(switch_counts)),
        "median_switches": float(np.median(switch_counts)),
        "p90_switches": float(np.percentile(switch_counts, 90)),
        "occupancy": occ,
        "start_regime": REGIME_ORDER[start_mode],
        "end_regime": REGIME_ORDER[end_mode],
    }


def compare_scenario_stats(
    baseline_paths: np.ndarray,
    scenario_results: dict,
    initial_capital: float,
    horizon_years: float,
) -> pd.DataFrame:
    def _row(name, paths):
        st = compute_mc_stats(paths, initial_capital, horizon_years=horizon_years)
        return {
            "Scenario":   name,
            "Median (₹)": int(st["median"]),
            "Mean (₹)":   int(st["mean"]),
            "P5 (₹)":     int(st["p5"]),
            "P95 (₹)":    int(st["p95"]),
            "VaR 95%":    round(st["var_95"], 1),
            "Max DD%":    round(st["max_drawdown"] * 100, 1),
            "P(profit)":  "{:.1f}%".format(st["prob_profit"] * 100),
            "Med CAGR":   "{:.1f}%".format(st["cagr_median"] * 100),
        }
    rows = [_row("Baseline (no shock)", baseline_paths)]
    for name, (paths, _) in scenario_results.items():
        rows.append(_row(name, paths))
    return pd.DataFrame(rows).set_index("Scenario")
