"""
============================================================
  QuanSen — Streamlit GUI  (Main Entry Point)
  Module 4 of 4: Page configuration, sidebar, hero area,
  and all seven application tabs.
  
  Wraps the Quantitative Portfolio Optimizer engine
  by Amatra Sen without modifying the engine.
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import html
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as _yf

from engines.market_state_engine import detect_market_state, regime_to_parameters
from engines.ticker_search import get_best_ticker
from engines.config import RF_ANNUAL, SHRINKAGE_ALPHA

# ── Import QuanSen GUI modules ──────────────────────────────
from engines.config_and_state import (
    APP_VERSION, _ICO_B64, _LOGO_B64,
    SECTOR_PRESETS, APP_DATA_PATH,
    init_state, sync_user_store_to_session, persist_session_user_store,
    consume_bridge_symbols,
    current_portfolio_payload, load_portfolio_payload,
    reset_portfolio_outputs,
    request_action, consume_action,
    normalize_symbol_list, parse_uploaded_symbols, apply_ticker_set,
    compute_max_w, default_analysis_window,
)
from engines.data_and_compute import (
    load_data, optimizer_expected_returns,
    cached_fetch_tape_quotes, cached_compute_momentum,
    cached_utility_portfolio, cached_tangency_portfolio,
    cached_min_risk_portfolio, cached_compute_frontier,
    cached_asset_risks, cached_correlation_matrix,
    cached_portfolio_stats, cached_portfolio_cumulative_returns,
    plotly_heatmap, search_ticker_api, get_ticker_date_range,
    generate_pdf,
)
from engines.ui_components import (
    GLOBAL_CSS,
    render_perf_status_panel, render_session_strip,
    show_flash_notice, render_hero_action_hub,
    render_build_workflow_overview, render_universe_helper,
    evaluate_alerts, render_ticker_tape,
    metric_card, weight_bar, make_weights_table,
    portfolio_metrics_row, render_insight_note,
    render_portfolio_spotlight, render_comparison_spotlights,
    render_market_weather_panel, render_regime_change_panel,
    # ── Noob mode ──────────────────────────────────────────────
    NOOB_CSS,
    render_mode_toggle,
    render_noob_hero,
    render_noob_how_it_works,
    render_noob_tab_overview,
    render_noob_tab_my_money,
    render_noob_tab_my_stocks,
    render_noob_tab_the_mix,
    render_noob_recommended_pick,
    render_noob_piggy_bank,
    render_noob_slot_weights,
    render_noob_fight_club,
    render_noob_time_machine,
    render_noob_risk_gauge,
)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="QuanSen · Portfolio Optimizer",
    page_icon="\U0001f4d0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global styles ─────────────────────────────────────
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Green theme — split into two parts ───────────────────────
# Part 1: .nb-* class definitions only — safe to always inject
#          so noob components render correctly regardless of mode.
#          These classes don't touch any global elements.
# Part 2: Global overrides (sidebar bg, body bg, tabs, etc.) —
#          injected ONLY when is_noob=True, after the sidebar runs.
_GREEN_CLASSES_CSS = """
<style>
/* ── Noob sidebar header ────────────────────────────────────── */
.nb-green-sidebar-header {
    text-align: center;
    padding: 0.6rem 0 1rem;
    border-bottom: 1px solid #1a5c2a;
    margin-bottom: 1rem;
}
.nb-green-sidebar-header .nb-app-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 800;
    color: #4afa7a;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.nb-green-sidebar-header .nb-app-sub {
    font-size: 0.62rem;
    color: #3db85a;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 2px;
}
.nb-green-status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 8px #22c55e;
    margin-right: 6px;
    vertical-align: middle;
}
.nb-green-status-dot.red    { background: #ff4d6d; box-shadow: 0 0 8px #ff4d6d; }
.nb-green-status-dot.yellow { background: #ffd166; box-shadow: 0 0 8px #ffd166; }

/* ── Noob dashboard cards ───────────────────────────────────── */
.nb-dash-card {
    background: linear-gradient(135deg, #061a0c, #0a2412);
    border: 1px solid #1a5c2a;
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 2px 18px rgba(0,200,80,0.08);
    position: relative;
    overflow: hidden;
}
.nb-dash-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00c851, #4afa7a, #00c851);
}
.nb-dash-card-label {
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3db85a;
    margin-bottom: 0.25rem;
}
.nb-dash-card-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #4afa7a;
    font-family: 'DM Mono', monospace;
    line-height: 1.1;
}
.nb-dash-card-sub  { font-size: 0.7rem; color: #2e7d52; margin-top: 0.2rem; }
.nb-dash-card-icon { position: absolute; top: 0.8rem; right: 1rem; font-size: 1.8rem; opacity: 0.18; }

/* ── Stock chip ─────────────────────────────────────────────── */
.nb-stock-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(0,200,80,0.12);
    border: 1px solid #1a7a38;
    border-radius: 20px;
    padding: 3px 10px 3px 8px;
    font-size: 0.72rem;
    color: #7dd89a;
    margin: 2px 3px;
    font-family: 'DM Mono', monospace;
}

/* ── Step wizard ─────────────────────────────────────────────── */
.nb-step-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.5rem; }
.nb-step-num {
    width: 28px; height: 28px; border-radius: 50%;
    background: linear-gradient(135deg, #0d6b2a, #12882f);
    border: 1px solid #1aad44;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700; color: #c8ffd4; flex-shrink: 0;
    box-shadow: 0 0 10px rgba(26,173,68,0.3);
}
.nb-step-num.done { background: linear-gradient(135deg, #065c20, #0a7a2a); border-color: #00c851; }
.nb-step-text { font-size: 0.82rem; color: #7dd89a; }
.nb-step-text strong { color: #c8ffd4; }

/* ── Green section heading ───────────────────────────────────── */
.nb-green-heading {
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #22c55e;
    font-weight: 700;
    margin: 1.2rem 0 0.6rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #1a5c2a;
}
</style>
"""
st.markdown(_GREEN_CLASSES_CSS, unsafe_allow_html=True)

# Part 2 is injected conditionally after is_noob is known — see sidebar block below.

# ── Session State Init ────────────────────────────────────────
init_state()
_demo_start_date, _demo_end_date = default_analysis_window(years=5)


def _persist_regime_state(meta: dict | None):
    """Persist smoothed regime probabilities and latest meta across reruns."""
    if not meta:
        return
    _existing = st.session_state.get("regime_meta") or {}
    _merged = dict(_existing)
    _merged.update(meta)
    _probs = _merged.get("regime_state") or _merged.get("regime_probabilities")
    if _probs:
        st.session_state.regime_prev_probs = dict(_probs)
    st.session_state.regime_meta = _merged
    st.session_state.regime_signature = (
        tuple(st.session_state.get("tickers", [])),
        str(st.session_state.get("start_date")),
        str(st.session_state.get("end_date")),
    )


def _regime_history_for_signature(signature):
    """Only reuse smoothed regime history for the exact same basket/date window."""
    if st.session_state.get("regime_signature") != signature:
        return None
    return st.session_state.get("regime_prev_probs")


def _commit_regime_snapshot(lookback=None, source="run"):
    """Store a compact run snapshot so the UI can explain what changed."""
    meta = st.session_state.get("regime_meta") or {}
    if not meta:
        return
    current = {
        "source": source,
        "dominant_regime": meta.get("dominant_regime", "unknown"),
        "confidence": float(meta.get("confidence", 0.0) or 0.0),
        "fast_dominant_regime": meta.get("fast_dominant_regime", meta.get("dominant_regime", "unknown")),
        "fast_confidence": float(meta.get("fast_confidence", meta.get("confidence", 0.0)) or 0.0),
        "alpha": float(st.session_state.get("shrinkage_alpha") or meta.get("alpha", 0.0) or 0.0),
        "beta": float(st.session_state.get("momentum_beta") or meta.get("beta", 0.0) or 0.0),
        "lookback": int(lookback) if lookback is not None else None,
        "primary_horizon": (st.session_state.get("momentum_meta") or {}).get("primary_horizon"),
    }
    previous = st.session_state.get("regime_run_snapshot")
    if previous:
        st.session_state.regime_prev_snapshot = previous
    st.session_state.regime_run_snapshot = current


def _optimizer_tactical_controls(previous_weights=None, enabled=None):
    """Return tactical optimizer controls without disturbing the long-term baseline."""
    if enabled is None:
        enabled = bool(
            st.session_state.get("momentum_enabled")
            and st.session_state.get("momentum_meta")
            and st.session_state.get("momentum_final_er") is not None
        )
    meta = st.session_state.get("momentum_meta") or {}
    if not enabled or not meta:
        return {
            "previous_weights": previous_weights,
            "turnover_penalty": 0.0,
            "rebalance_ratio": 1.0,
        }
    return {
        "previous_weights": previous_weights,
        "turnover_penalty": float(meta.get("tactical_turnover_penalty", 0.0) or 0.0),
        "rebalance_ratio": float(meta.get("tactical_rebalance_ratio", 1.0) or 1.0),
    }


def _has_full_regime_meta(meta):
    return bool(
        isinstance(meta, dict)
        and meta.get("regime_probabilities")
        and ("trend" in meta or "trend_z" in meta)
    )


def _get_regime_state(force=False):
    """Reuse the latest regime read when the basket/date window has not changed."""
    tickers = tuple(st.session_state.get("tickers", []))
    if not tickers:
        return None, None, None
    signature = (
        tickers,
        str(st.session_state.get("start_date")),
        str(st.session_state.get("end_date")),
    )
    cached_meta = st.session_state.get("regime_meta")
    if (
        not force
        and cached_meta
        and _has_full_regime_meta(cached_meta)
        and st.session_state.get("regime_signature") == signature
    ):
        return (
            float(cached_meta.get("alpha", st.session_state.get("shrinkage_alpha", 0.55))),
            float(cached_meta.get("beta", st.session_state.get("momentum_beta", 0.55))),
            cached_meta,
        )

    alpha_regime, beta_regime, regime_meta = detect_market_state(
        tickers,
        st.session_state.start_date,
        st.session_state.end_date,
        prev_regime_probs=_regime_history_for_signature(signature),
    )
    _persist_regime_state(regime_meta)
    return alpha_regime, beta_regime, regime_meta


def _weight_series(weights, tickers):
    arr = np.asarray(weights, dtype=float).reshape(-1)
    ser = pd.Series(arr, index=list(tickers), dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total = float(ser.sum())
    if total <= 1e-12:
        return pd.Series(1.0 / max(len(ser), 1), index=ser.index, dtype=float)
    return ser / total


def _deployment_posture(defensive_blend, cash_weight, stress_score, distance_score):
    if cash_weight >= 0.18 or defensive_blend >= 0.55 or stress_score >= 0.62:
        return "Capital Preservation"
    if cash_weight >= 0.08 or defensive_blend >= 0.32 or stress_score >= 0.45:
        return "Defensive"
    if defensive_blend >= 0.16 or stress_score >= 0.28 or distance_score >= 0.22:
        return "Balanced"
    return "Risk-On"


def _deployment_posture_display(posture):
    mapping = {
        "Capital Preservation": "Capital Safe",
        "Defensive": "Defensive",
        "Balanced": "Balanced",
        "Risk-On": "Risk-On",
    }
    return mapping.get(str(posture), str(posture))


def _deployment_confidence(latest_meta, latest_overlay, relative_alpha):
    latest_meta = latest_meta or {}
    latest_overlay = latest_overlay or {}
    slow_conf = float(latest_meta.get("confidence", 0.0) or 0.0)
    fast_conf = float(latest_meta.get("fast_confidence", slow_conf) or 0.0)
    stress = float(latest_overlay.get("stress_score", 0.0) or 0.0)
    distance = float(latest_overlay.get("distance_score", 0.0) or 0.0)
    defensive_blend = float(latest_overlay.get("defensive_blend", 0.0) or 0.0)
    cash_weight = float(latest_overlay.get("cash_weight", 0.0) or 0.0)
    score = (
        42.0
        + 18.0 * slow_conf
        + 22.0 * fast_conf
        - 18.0 * stress
        - 14.0 * distance
        - 10.0 * defensive_blend
        - 16.0 * cash_weight
        + np.clip(relative_alpha, -10.0, 10.0) * 0.8
    )
    return float(np.clip(score, 15.0, 92.0))


def _bounded_weight_series(weights, tickers, min_w=0.0, max_w=1.0, target_sum=1.0, max_iter=12):
    ser = pd.Series(np.asarray(weights, dtype=float).reshape(-1), index=list(tickers), dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if ser.empty:
        return ser
    target_sum = float(max(target_sum, 0.0))
    if target_sum <= 1e-12:
        return ser * 0.0
    lower = max(0.0, float(min_w))
    upper = max(lower, float(max_w))
    ser = ser.clip(lower=0.0)
    total = float(ser.sum())
    if total <= 1e-12:
        ser = pd.Series(target_sum / len(ser), index=ser.index, dtype=float)
    else:
        ser = ser * (target_sum / total)
    upper_eff = min(upper, target_sum)
    lower_eff = min(lower, target_sum / len(ser))
    ser = ser.clip(lower=lower_eff, upper=upper_eff)
    for _ in range(max_iter):
        gap = target_sum - float(ser.sum())
        if abs(gap) <= 1e-9:
            break
        if gap > 0:
            room = (upper_eff - ser).clip(lower=0.0)
        else:
            room = (ser - lower_eff).clip(lower=0.0)
        free = room[room > 1e-10]
        if free.empty:
            break
        if gap > 0:
            step = free / free.sum() * gap
            ser.loc[free.index] = ser.loc[free.index] + step
        else:
            step = free / free.sum() * (-gap)
            ser.loc[free.index] = ser.loc[free.index] - step
        ser = ser.clip(lower=lower_eff, upper=upper_eff)
    total = float(ser.sum())
    if total > 1e-12:
        ser = ser * (target_sum / total)
    return ser.clip(lower=0.0)


def _rank_unit(series):
    ser = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if ser.empty:
        return pd.Series(dtype=float)
    ranks = ser.rank(method="average", pct=True)
    return ranks.reindex(series.index).fillna(ranks.mean())


def _model_disagreement(raw_er, shrink_er, active_er, tickers):
    frames = []
    for series in [raw_er, shrink_er, active_er]:
        if series is None:
            continue
        ser = pd.Series(series, dtype=float).reindex(tickers)
        frames.append(_rank_unit(ser))
    if len(frames) < 2:
        neutral = pd.Series(1.0, index=tickers, dtype=float)
        return neutral, 0.0
    rank_df = pd.concat(frames, axis=1).reindex(tickers).fillna(method="ffill", axis=1).fillna(method="bfill", axis=1)
    disagreement = rank_df.std(axis=1).fillna(0.0)
    disagreement = disagreement / max(float(disagreement.max()), 1e-9)
    penalty = (1.0 - 0.35 * disagreement).clip(lower=0.55, upper=1.0)
    score = float(np.clip(disagreement.mean(), 0.0, 1.0))
    return penalty, score


def _transition_risk(meta):
    meta = meta or {}
    forecast = meta.get("transition_forecast") or {}
    if forecast:
        return float(np.clip(forecast.get("transition_risk_10d", 0.0), 0.0, 1.0))
    fast_probs = meta.get("fast_regime_probabilities", {}) or {}
    if not fast_probs:
        return 0.0
    dominant_prob = max(float(v) for v in fast_probs.values())
    bear_crisis = float(fast_probs.get("bear", 0.0)) + float(fast_probs.get("crisis", 0.0))
    return float(np.clip((1.0 - dominant_prob) * 0.55 + bear_crisis * 0.45, 0.0, 1.0))


def _normalise_regime_probs(probs):
    vals = {k: max(float((probs or {}).get(k, 0.0) or 0.0), 0.0) for k in REGIME_ORDER}
    total = sum(vals.values())
    if total <= 1e-12:
        return {k: 1.0 / len(REGIME_ORDER) for k in REGIME_ORDER}
    return {k: vals[k] / total for k in REGIME_ORDER}


def _regime_sim_current_market_probs(meta):
    """Build a safer 'current market' seed for regime-switch MC from tactical state."""
    meta = meta or {}
    fast_probs = _normalise_regime_probs(
        meta.get("fast_regime_probabilities") or meta.get("regime_probabilities") or {}
    )
    slow_probs = _normalise_regime_probs(
        meta.get("regime_probabilities") or meta.get("fast_regime_probabilities") or {}
    )
    fast_label = str(meta.get("fast_dominant_regime") or meta.get("dominant_regime") or "sideways").strip().lower()
    fast_conf = float(meta.get("fast_confidence", meta.get("confidence", 0.0)) or 0.0)
    transition = _transition_risk(meta)
    stress = _regime_stress_score(meta)
    feats = meta.get("fast_regime_features", {}) or {}
    index_21d_return = float(feats.get("index_21d_return", 0.0) or 0.0)
    index_21d_drawdown = float(feats.get("index_21d_drawdown", 0.0) or 0.0)
    index_63d_drawdown = float(feats.get("index_63d_drawdown", 0.0) or 0.0)
    selloff = float(np.clip(max(-index_21d_return, 0.0) / 0.06, 0.0, 1.5))
    dd21 = float(np.clip(max(-index_21d_drawdown, 0.0) / 0.08, 0.0, 1.5))
    dd63 = float(np.clip(max(-index_63d_drawdown, 0.0) / 0.12, 0.0, 1.5))
    damage = float(np.clip(0.45 * selloff + 0.30 * dd21 + 0.25 * dd63, 0.0, 1.5))

    fast_weight = 0.82 if fast_label not in {"mixed", "bear", "crisis"} else 0.90
    seed = {
        k: fast_weight * fast_probs.get(k, 0.0) + (1.0 - fast_weight) * slow_probs.get(k, 0.0)
        for k in REGIME_ORDER
    }

    if fast_label == "mixed":
        bull_shift = min(seed["bull"], 0.10 + 0.12 * damage + 0.08 * transition)
        seed["bull"] -= bull_shift
        seed["sideways"] += bull_shift * 0.45
        seed["bear"] += bull_shift * 0.40
        seed["crisis"] += bull_shift * 0.15

    if fast_label in {"bear", "crisis"}:
        bull_cap = 0.14 if damage < 0.9 else 0.08
    elif index_21d_return <= -0.06:
        bull_cap = 0.16
    else:
        bull_cap = 1.0
    if seed["bull"] > bull_cap:
        overflow = seed["bull"] - bull_cap
        seed["bull"] = bull_cap
        seed["bear"] += overflow * 0.65
        seed["crisis"] += overflow * 0.35

    defensive_boost = float(np.clip(0.08 * damage + 0.06 * stress + 0.06 * transition, 0.0, 0.22))
    if defensive_boost > 0:
        bull_take = min(seed["bull"], defensive_boost * 0.60)
        side_take = min(seed["sideways"], defensive_boost * 0.40)
        seed["bull"] -= bull_take
        seed["sideways"] -= side_take
        seed["bear"] += bull_take * 0.70 + side_take * 0.55
        seed["crisis"] += bull_take * 0.30 + side_take * 0.45

    if fast_conf < 0.08 and seed["bull"] > seed["bear"]:
        uncertainty_shift = min(seed["bull"] - seed["bear"], 0.10)
        seed["bull"] -= uncertainty_shift
        seed["sideways"] += uncertainty_shift * 0.60
        seed["bear"] += uncertainty_shift * 0.40

    return _normalise_regime_probs(seed)


def _weight_stability_score(weight_history_df):
    if weight_history_df is None or weight_history_df.empty or len(weight_history_df) < 2:
        return 0.5
    turnover = weight_history_df.diff().abs().sum(axis=1).dropna()
    if turnover.empty:
        return 0.5
    return float(np.clip(1.0 - turnover.tail(4).mean() / 0.35, 0.0, 1.0))


def _promote_deployable_weights_to_portfolios(deployable):
    deployable = deployable or {}
    deploy_weights = deployable.get("weights") or {}
    tickers = list(st.session_state.get("tickers", []))
    if not deploy_weights or not tickers:
        st.session_state.flash_notice = ("warning", "No deployable weights are available to promote yet.")
        return False

    deploy_series = pd.Series(deploy_weights, dtype=float).reindex(tickers).fillna(0.0)
    total = float(deploy_series.sum())
    if total <= 1e-12:
        st.session_state.flash_notice = ("warning", "Deploy-now weights were empty after aligning to the current basket.")
        return False

    deploy_series = deploy_series / total
    st.session_state.weights_deploy = deploy_series.values.astype(float)
    st.session_state.deployable_meta = dict(deployable)
    st.session_state.portfolios_computed = True
    st.session_state.qs_section = "portfolios"
    st.session_state.flash_notice = ("success", "Deploy-now weights were loaded into the Portfolios view.")
    return True


def _weight_confidence_bands(weight_history_df, deploy_total_weights):
    deploy_total_weights = pd.Series(deploy_total_weights, dtype=float)
    if weight_history_df is None or weight_history_df.empty:
        df = pd.DataFrame({
            "Lower %": deploy_total_weights.mul(100.0),
            "Deploy Now %": deploy_total_weights.mul(100.0),
            "Upper %": deploy_total_weights.mul(100.0),
        })
        return df.round(2)
    hist = weight_history_df.reindex(columns=deploy_total_weights.index).fillna(0.0)
    recent = hist.tail(min(5, len(hist)))
    lower = recent.quantile(0.20).reindex(deploy_total_weights.index).fillna(deploy_total_weights)
    upper = recent.quantile(0.80).reindex(deploy_total_weights.index).fillna(deploy_total_weights)
    lower = np.minimum(lower, deploy_total_weights)
    upper = np.maximum(upper, deploy_total_weights)
    return pd.DataFrame({
        "Lower %": lower * 100.0,
        "Deploy Now %": deploy_total_weights * 100.0,
        "Upper %": upper * 100.0,
    }).round(2)


def _downside_guard_weights(returns_df, tickers, target_sum, min_w=0.0, max_w=1.0):
    if returns_df is None or returns_df.empty:
        return _bounded_weight_series(np.ones(len(tickers)), tickers, min_w=min_w, max_w=max_w, target_sum=target_sum)
    frame = returns_df.reindex(columns=tickers).dropna(how="all").fillna(0.0)
    downside = frame.clip(upper=0.0)
    downside_vol = downside.pow(2).mean().pow(0.5).replace(0.0, np.nan)
    inv_risk = (1.0 / downside_vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if float(inv_risk.sum()) <= 1e-12:
        inv_risk = pd.Series(1.0, index=tickers, dtype=float)
    return _bounded_weight_series(inv_risk.values, tickers, min_w=min_w, max_w=max_w, target_sum=target_sum)


def _apply_execution_friction(target_total_weights, previous_total_weights, target_sum, min_w=0.0, max_w=1.0, min_change=0.0075, turnover_budget=0.22):
    target_total_weights = pd.Series(target_total_weights, dtype=float)
    previous_total_weights = pd.Series(previous_total_weights, dtype=float).reindex(target_total_weights.index).fillna(0.0)
    adjusted = target_total_weights.copy()
    delta = adjusted - previous_total_weights
    adjusted.loc[delta.abs() < min_change] = previous_total_weights.loc[delta.abs() < min_change]
    delta = adjusted - previous_total_weights
    turnover = float(delta.abs().sum())
    if turnover > turnover_budget and turnover > 1e-12:
        adjusted = previous_total_weights + delta * (turnover_budget / turnover)
    return _bounded_weight_series(adjusted.values, adjusted.index, min_w=min_w, max_w=max_w, target_sum=target_sum)


def _ticker_adjustment_explanations(compare_df, confidence_bands, disagreement_penalty, latest_meta, top_n=6):
    if compare_df is None or compare_df.empty:
        return pd.DataFrame(columns=["Ticker", "Action", "Why"])
    latest_meta = latest_meta or {}
    fast_probs = latest_meta.get("fast_regime_probabilities", {}) or {}
    tactical_label = str(latest_meta.get("fast_dominant_regime", latest_meta.get("dominant_regime", "unknown"))).title()
    bear_crisis = float(fast_probs.get("bear", 0.0)) + float(fast_probs.get("crisis", 0.0))
    rows = []
    disagreement_penalty = pd.Series(disagreement_penalty, dtype=float)
    for ticker, row in compare_df.reindex(compare_df["Delta vs Original pp"].abs().sort_values(ascending=False).index).head(top_n).iterrows():
        delta = float(row.get("Delta vs Original pp", 0.0))
        deploy = float(row.get("Deploy Now %", 0.0))
        defensive = float(row.get("Defensive %", deploy))
        tactical = float(row.get("Tactical %", deploy))
        band_low = band_high = deploy
        if confidence_bands is not None and ticker in confidence_bands.index:
            band_low = float(confidence_bands.loc[ticker, "Lower %"])
            band_high = float(confidence_bands.loc[ticker, "Upper %"])
        band_width = max(band_high - band_low, 0.0)
        disagreement = float(1.0 - disagreement_penalty.get(ticker, 1.0))
        action = "Increase" if delta > 0.5 else ("Cut" if delta < -0.5 else "Hold")
        reasons = []
        if action == "Increase":
            if tactical >= defensive:
                reasons.append("tactical model still prefers it")
            if disagreement < 0.22:
                reasons.append("signals agree")
            if band_width <= 4.0:
                reasons.append("weight range is stable")
        elif action == "Cut":
            if defensive < tactical:
                reasons.append("defensive mix wants less")
            if bear_crisis >= 0.40:
                reasons.append(f"{tactical_label.lower()} tape is asking for caution")
            if disagreement >= 0.25:
                reasons.append("models disagree on conviction")
            if band_width >= 6.0:
                reasons.append("recent weight range is wide")
        else:
            reasons.append("signals are close, so friction control kept the change small")
        rows.append({
            "Ticker": ticker,
            "Action": action,
            "Why": "; ".join(reasons[:3]) if reasons else "balanced by tactical and defensive inputs",
        })
    return pd.DataFrame(rows)


def _regime_distance_score(reference_meta, current_meta):
    """Estimate how different deployment conditions are from the training regime."""
    reference_meta = reference_meta or {}
    current_meta = current_meta or {}

    def _probs(meta, key):
        probs = meta.get(key, {}) or {}
        return np.array([float(probs.get(r, 0.25)) for r in ["bull", "sideways", "bear", "crisis"]], dtype=float)

    slow_gap = float(np.abs(_probs(reference_meta, "regime_probabilities") - _probs(current_meta, "regime_probabilities")).sum() / 2.0)
    fast_gap = float(np.abs(_probs(reference_meta, "fast_regime_probabilities") - _probs(current_meta, "fast_regime_probabilities")).sum() / 2.0)
    ref_basket = reference_meta.get("basket", {}) or {}
    cur_basket = current_meta.get("basket", {}) or {}
    corr_gap = min(abs(float(cur_basket.get("avg_corr_63", 0.35)) - float(ref_basket.get("avg_corr_63", 0.35))) / 0.35, 1.0)
    downside_gap = min(abs(float(cur_basket.get("downside_pct", 25.0)) - float(ref_basket.get("downside_pct", 25.0))) / 50.0, 1.0)
    participation_gap = min(abs(float(cur_basket.get("participation_pct", 50.0)) - float(ref_basket.get("participation_pct", 50.0))) / 50.0, 1.0)
    return float(np.clip(
        0.35 * slow_gap + 0.30 * fast_gap + 0.15 * corr_gap + 0.10 * downside_gap + 0.10 * participation_gap,
        0.0,
        1.0,
    ))


def _regime_stress_score(meta):
    """Collapse tactical stress signals into one gating score."""
    meta = meta or {}
    fast_probs = meta.get("fast_regime_probabilities", meta.get("regime_probabilities", {})) or {}
    basket = meta.get("basket", {}) or {}
    fast_feats = meta.get("fast_regime_features", {}) or {}
    bearish = float(fast_probs.get("bear", 0.0)) + float(fast_probs.get("crisis", 0.0))
    avg_corr = float(basket.get("avg_corr_63", 0.35))
    downside = float(basket.get("downside_pct", 25.0)) / 100.0
    participation = float(basket.get("participation_pct", 50.0)) / 100.0
    vol_ratio = float(meta.get("vol_ratio", 1.0) or 1.0)
    index_21d_return = float(fast_feats.get("index_21d_return", 0.0) or 0.0)
    index_21d_drawdown = float(fast_feats.get("index_21d_drawdown", 0.0) or 0.0)
    index_63d_drawdown = float(fast_feats.get("index_63d_drawdown", 0.0) or 0.0)
    index_selloff = float(np.clip(max(-index_21d_return, 0.0) / 0.06, 0.0, 1.5))
    index_dd21 = float(np.clip(max(-index_21d_drawdown, 0.0) / 0.08, 0.0, 1.5))
    index_dd63 = float(np.clip(max(-index_63d_drawdown, 0.0) / 0.12, 0.0, 1.5))
    return float(np.clip(
        0.28 * bearish +
        0.14 * max(avg_corr - 0.45, 0.0) / 0.40 +
        0.14 * max(downside - 0.25, 0.0) / 0.75 +
        0.08 * max(0.55 - participation, 0.0) / 0.55 +
        0.08 * max(vol_ratio - 1.10, 0.0) / 1.40 +
        0.14 * index_selloff +
        0.08 * index_dd21 +
        0.06 * index_dd63,
        0.0,
        1.0,
    ))


def _apply_backtest_safety_overlay(selected_portfolio, target_weights, defensive_weights, current_meta, reference_meta, gate_strength=1.0):
    """Blend aggressive portfolios toward defense/cash when deployment conditions diverge."""
    target_weights = target_weights.astype(float)
    defensive_weights = defensive_weights.reindex(target_weights.index).fillna(0.0).astype(float)
    defensive_total = float(defensive_weights.sum())
    if defensive_total <= 1e-12:
        defensive_weights = target_weights.copy()
    else:
        defensive_weights = defensive_weights / defensive_total

    stress_score = _regime_stress_score(current_meta)
    distance_score = _regime_distance_score(reference_meta, current_meta)
    fast_probs = current_meta.get("fast_regime_probabilities", current_meta.get("regime_probabilities", {})) or {}
    bear_crisis = float(fast_probs.get("bear", 0.0)) + float(fast_probs.get("crisis", 0.0))

    overlay_strength = float(np.clip(
        gate_strength * (
            0.45 * stress_score +
            0.35 * distance_score +
            0.20 * max(0.0, bear_crisis - 0.45)
        ),
        0.0,
        0.90,
    ))
    if "Min-Risk" in selected_portfolio:
        overlay_strength *= 0.35

    blended = (1.0 - overlay_strength) * target_weights + overlay_strength * defensive_weights
    blended_total = float(blended.sum())
    if blended_total > 1e-12:
        blended = blended / blended_total

    cash_weight = float(np.clip(
        max(0.0, stress_score - 0.58) * 0.45 +
        max(0.0, distance_score - 0.55) * 0.30 +
        max(0.0, bear_crisis - 0.70) * 0.35,
        0.0,
        0.35,
    ))
    invested_weights = blended * (1.0 - cash_weight)
    invested_total = float(invested_weights.sum())
    if invested_total > 1e-12:
        invested_weights = invested_weights / invested_total

    return {
        "weights": invested_weights,
        "defensive_blend": overlay_strength,
        "cash_weight": cash_weight,
        "stress_score": stress_score,
        "distance_score": distance_score,
    }


def _bt_benchmark_symbol(tickers, regime_meta):
    if regime_meta and regime_meta.get("benchmark"):
        return regime_meta.get("benchmark")
    if st.session_state.get("bm_sym"):
        return st.session_state.get("bm_sym")
    indian = sum(str(t).upper().endswith(".NS") or str(t).upper().endswith(".BO") for t in tickers)
    return "^NSEI" if indian >= max(1, len(tickers) / 2) else "^GSPC"


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_download_close_prices(symbols, start_date, end_date):
    """Cached adjusted-close downloader used by walk-forward backtests."""
    data = _yf.download(
        list(symbols),
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    if data is None or getattr(data, "empty", True):
        return pd.DataFrame()
    prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else (data[["Close"]] if "Close" in data.columns else data)
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=list(symbols)[0])
    return prices.sort_index().dropna(how="all")


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_walkforward_snapshot(
    tickers,
    train_start,
    train_end,
    target_return,
    min_w,
    max_w,
    momentum_enabled,
    momentum_lookback,
    auto_regime,
    lightweight,
    short_horizon_mode=False,
    prev_regime_probs=None,
):
    """Cached training snapshot for a walk-forward rebalance step."""
    alpha_regime, beta_regime, regime_meta = detect_market_state(
        list(tickers),
        train_start,
        train_end,
        prev_regime_probs=prev_regime_probs,
    )
    regime_meta = dict(regime_meta or {})
    effective_regime_label = regime_meta.get("dominant_regime", "unknown")
    effective_regime_source = "slow"
    if short_horizon_mode:
        fast_probs = regime_meta.get("fast_regime_probabilities") or regime_meta.get("regime_probabilities") or {}
        fast_alpha, fast_beta = regime_to_parameters(fast_probs)
        alpha_regime = float(np.clip(0.35 * alpha_regime + 0.65 * fast_alpha, 0.12, 0.95))
        beta_regime = float(np.clip(0.35 * beta_regime + 0.65 * fast_beta, 0.20, 0.92))
        effective_regime_label = regime_meta.get("fast_dominant_regime", effective_regime_label)
        effective_regime_source = "fast"
    regime_meta["effective_alpha"] = round(alpha_regime, 4)
    regime_meta["effective_beta"] = round(beta_regime, 4)
    regime_meta["backtest_regime_label"] = effective_regime_label
    regime_meta["backtest_regime_source"] = effective_regime_source
    tickers_out, returns, expected_returns, cov_matrix, raw_er, bm_sym = load_data(
        list(tickers),
        train_start,
        train_end,
        alpha=alpha_regime,
    )
    active_er = expected_returns.reindex(tickers_out)
    momentum_meta = None
    if momentum_enabled and not lightweight:
        _, _, active_er, momentum_meta = cached_compute_momentum(
            tuple(tickers_out),
            train_start,
            train_end,
            expected_returns.reindex(tickers_out),
            lookback=momentum_lookback,
            skip=21,
            beta=beta_regime,
            auto_lookback=auto_regime,
            prev_regime_probs=prev_regime_probs,
        )
    return {
        "alpha_regime": alpha_regime,
        "beta_regime": beta_regime,
        "regime_meta": regime_meta,
        "tickers_out": list(tickers_out),
        "returns": returns.reindex(columns=tickers_out),
        "active_er": active_er,
        "shrink_er": expected_returns.reindex(tickers_out),
        "raw_er": raw_er.reindex(tickers_out),
        "cov_matrix": cov_matrix,
        "target_return": target_return,
        "min_w": min_w,
        "max_w": max_w,
        "momentum_meta": momentum_meta,
        "bm_sym": bm_sym,
    }


def _run_walkforward_backtest(
    selected_portfolio,
    weights_bt,
    tickers_bt,
    bt_start,
    bt_end,
    capital,
    allow_fractional,
    rebalance_step_days,
    reference_meta,
    lightweight=False,
    gate_strength=1.0,
):
    """Walk-forward backtest with regime-aware re-optimisation and defense overlays."""
    tickers_bt = list(tickers_bt)
    if len(tickers_bt) < 2:
        raise ValueError("Need at least 2 tickers for walk-forward backtesting.")

    bt_start_ts = pd.Timestamp(bt_start)
    bt_end_ts = pd.Timestamp(bt_end)
    session_start_ts = pd.Timestamp(st.session_state.start_date)
    session_end_ts = pd.Timestamp(st.session_state.end_date)
    short_horizon_mode = int((bt_end_ts - bt_start_ts).days) <= 126
    train_span_days = max(126, int((session_end_ts - session_start_ts).days))
    effective_train_span_days = max(126, min(train_span_days, 252)) if lightweight else train_span_days
    effective_rebalance_days = max(int(rebalance_step_days), 10) if lightweight else int(rebalance_step_days)
    history_start_ts = min(session_start_ts, bt_start_ts - pd.Timedelta(days=effective_train_span_days + 40))

    full_prices = _cached_download_close_prices(
        tuple(tickers_bt),
        history_start_ts.strftime("%Y-%m-%d"),
        (bt_end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    price_window = full_prices.loc[(full_prices.index >= bt_start_ts) & (full_prices.index <= bt_end_ts)].copy()
    if price_window.empty or len(price_window) < 2:
        raise ValueError("No price data returned for the walk-forward backtest window.")

    valid_tickers = []
    first_valid = {}
    last_valid = {}
    for ticker in tickers_bt:
        if ticker not in price_window.columns:
            continue
        col = price_window[ticker].dropna()
        col = col[col > 0]
        if len(col) < 2:
            continue
        valid_tickers.append(ticker)
        first_valid[ticker] = col.index[0]
        last_valid[ticker] = col.index[-1]
    if len(valid_tickers) < 2:
        raise ValueError("Not enough tickers have valid prices in the walk-forward backtest window.")

    common_entry_ts = max(first_valid[t] for t in valid_tickers)
    common_exit_ts = min(last_valid[t] for t in valid_tickers)
    if common_entry_ts >= common_exit_ts:
        raise ValueError("No overlapping trading window exists across all tickers for walk-forward mode.")

    window = price_window.loc[common_entry_ts:common_exit_ts, valid_tickers].ffill().dropna(how="all")
    if len(window) < 2:
        raise ValueError("Not enough overlapping price data for walk-forward mode.")

    rebalance_dates = list(window.index[::max(effective_rebalance_days, 1)])
    if rebalance_dates[-1] != window.index[-1]:
        rebalance_dates.append(window.index[-1])

    portfolio_value = float(capital)
    residual_cash = 0.0
    daily_values_segments = []
    rebalance_rows = []
    # Walk-forward detection should evolve from its own local history, not from
    # whatever broad regime snapshot happened to be stored in session state.
    current_prev_probs = None
    previous_weights = _weight_series(weights_bt, tickers_bt).reindex(valid_tickers).fillna(0.0)
    benchmark_symbol = _bt_benchmark_symbol(valid_tickers, reference_meta)
    latest_target_weights = None
    latest_defensive_weights = None
    latest_invested_weights = None
    latest_cash_weight = 0.0
    latest_overlay = None
    latest_regime_meta = None
    latest_rebalance_date = None
    latest_returns = None
    latest_raw_er = None
    latest_shrink_er = None
    latest_active_er = None
    latest_previous_total = None
    weight_history_rows = []
    calibration_rows = []

    for i, rebalance_date in enumerate(rebalance_dates[:-1]):
        prior_index = full_prices.index[full_prices.index < rebalance_date]
        if len(prior_index) == 0:
            continue
        train_end_ts = prior_index[-1]
        train_start_ts = max(session_start_ts, train_end_ts - pd.Timedelta(days=effective_train_span_days))
        if train_end_ts <= train_start_ts:
            continue

        snapshot = _cached_walkforward_snapshot(
            tuple(valid_tickers),
            train_start_ts.strftime("%Y-%m-%d"),
            train_end_ts.strftime("%Y-%m-%d"),
            st.session_state.target_return,
            st.session_state.min_w,
            st.session_state.max_w,
            bool(st.session_state.get("momentum_enabled")),
            st.session_state.get("momentum_lookback", 252),
            bool(st.session_state.get("auto_regime", True)),
            bool(lightweight),
            bool(short_horizon_mode),
            prev_regime_probs=current_prev_probs,
        )
        alpha_regime = snapshot["alpha_regime"]
        beta_regime = snapshot["beta_regime"]
        regime_meta = snapshot["regime_meta"]
        current_prev_probs = regime_meta.get("regime_state") or regime_meta.get("regime_probabilities")
        tickers_out = snapshot["tickers_out"]
        cov_matrix = snapshot["cov_matrix"]
        active_er = snapshot["active_er"].reindex(tickers_out)
        latest_returns = snapshot.get("returns")
        latest_raw_er = snapshot.get("raw_er")
        latest_shrink_er = snapshot.get("shrink_er")
        latest_active_er = active_er
        tactical_turnover = float(regime_meta.get("tactical_turnover_penalty", 0.0) or 0.0)
        tactical_rebalance = float(regime_meta.get("tactical_rebalance_ratio", 1.0) or 1.0)

        prev_aligned = previous_weights.reindex(tickers_out).fillna(0.0)
        try:
            defensive_arr = cached_min_risk_portfolio(
                active_er,
                cov_matrix,
                tuple(tickers_out),
                st.session_state.target_return,
                st.session_state.min_w,
                st.session_state.max_w,
                previous_weights=prev_aligned.values,
                turnover_penalty=tactical_turnover,
                rebalance_ratio=tactical_rebalance,
            )
        except Exception:
            defensive_arr = prev_aligned.values
        if defensive_arr is None:
            defensive_arr = prev_aligned.values
        defensive_weights = _weight_series(defensive_arr, tickers_out)

        if "Utility" in selected_portfolio:
            try:
                target_arr = cached_utility_portfolio(
                    active_er,
                    cov_matrix,
                    tuple(tickers_out),
                    st.session_state.min_w,
                    st.session_state.max_w,
                    previous_weights=prev_aligned.values,
                    turnover_penalty=tactical_turnover,
                    rebalance_ratio=tactical_rebalance,
                )
            except Exception:
                target_arr = prev_aligned.values
        elif "Tangency" in selected_portfolio:
            try:
                target_arr, _, _, _ = cached_tangency_portfolio(
                    active_er,
                    cov_matrix,
                    tuple(tickers_out),
                    st.session_state.min_w,
                    st.session_state.max_w,
                    previous_weights=prev_aligned.values,
                    turnover_penalty=tactical_turnover,
                    rebalance_ratio=tactical_rebalance,
                )
            except Exception:
                target_arr = prev_aligned.values
        else:
            target_arr = defensive_arr

        target_weights = _weight_series(target_arr if target_arr is not None else prev_aligned.values, tickers_out)
        overlay = _apply_backtest_safety_overlay(
            selected_portfolio,
            target_weights,
            defensive_weights,
            regime_meta,
            reference_meta,
            gate_strength=gate_strength,
        )
        invested_weights = overlay["weights"].reindex(tickers_out).fillna(0.0)
        cash_weight = overlay["cash_weight"]
        latest_previous_total = prev_aligned.reindex(valid_tickers).fillna(0.0)
        latest_target_weights = target_weights.reindex(valid_tickers).fillna(0.0)
        latest_defensive_weights = defensive_weights.reindex(valid_tickers).fillna(0.0)
        latest_invested_weights = invested_weights.reindex(valid_tickers).fillna(0.0)
        latest_cash_weight = cash_weight
        latest_overlay = dict(overlay)
        latest_regime_meta = regime_meta
        latest_rebalance_date = rebalance_date

        px0 = window.loc[rebalance_date, tickers_out].replace([np.inf, -np.inf], np.nan).ffill()
        if px0.isna().any():
            px0 = window.loc[:rebalance_date, tickers_out].ffill().iloc[-1]
        invested_capital = portfolio_value * (1.0 - cash_weight)
        target_alloc = invested_capital * invested_weights
        if allow_fractional:
            shares = (target_alloc / px0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            residual_cash = portfolio_value * cash_weight
        else:
            shares = np.floor((target_alloc / px0).replace([np.inf, -np.inf], np.nan).fillna(0.0))
            residual_cash = portfolio_value - float((shares * px0).sum())

        next_rebalance_date = rebalance_dates[i + 1]
        segment = window.loc[rebalance_date:next_rebalance_date, tickers_out].ffill()
        segment_values = segment.mul(shares, axis=1).sum(axis=1) + residual_cash
        if i > 0 and not segment_values.empty:
            segment_values = segment_values.iloc[1:]
        daily_values_segments.append(segment_values)
        segment_total_return = 0.0
        if not segment_values.empty:
            segment_total_return = float(segment_values.iloc[-1] / max(float(segment_values.iloc[0]), 1e-9) - 1.0)

        px_end = segment.iloc[-1].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        benchmark_segment_return = np.nan
        benchmark_prices_segment = _cached_download_close_prices(
            (benchmark_symbol,),
            rebalance_date.strftime("%Y-%m-%d"),
            (next_rebalance_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        if benchmark_prices_segment is not None and not benchmark_prices_segment.empty:
            benchmark_series = benchmark_prices_segment.squeeze()
            benchmark_series = pd.Series(benchmark_series).dropna()
            if len(benchmark_series) >= 2:
                benchmark_segment_return = float(benchmark_series.iloc[-1] / max(float(benchmark_series.iloc[0]), 1e-9) - 1.0)

        def _segment_weight_return(weight_series):
            if weight_series is None or px_end.isna().all():
                return np.nan
            aligned = _weight_series(weight_series, tickers_out).reindex(tickers_out).fillna(0.0)
            if aligned.sum() <= 1e-12:
                return np.nan
            price_rel = (px_end / px0.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0) - 1.0
            return float((aligned * price_rel).sum())

        tactical_segment_return = _segment_weight_return(target_weights)
        defensive_segment_return = _segment_weight_return(defensive_weights)
        deploy_segment_return = _segment_weight_return(invested_weights) * (1.0 - cash_weight)
        calibration_rows.append({
            "Date": rebalance_date.strftime("%Y-%m-%d"),
            "Posture": _deployment_posture(
                float(overlay.get("defensive_blend", 0.0) or 0.0),
                float(cash_weight or 0.0),
                float(overlay.get("stress_score", 0.0) or 0.0),
                float(overlay.get("distance_score", 0.0) or 0.0),
            ),
            "Regime Used": str(regime_meta.get("backtest_regime_label", regime_meta.get("dominant_regime", "unknown"))).title(),
            "Deploy Return %": round(deploy_segment_return * 100.0, 2) if pd.notna(deploy_segment_return) else np.nan,
            "Tactical Return %": round(tactical_segment_return * 100.0, 2) if pd.notna(tactical_segment_return) else np.nan,
            "Defensive Return %": round(defensive_segment_return * 100.0, 2) if pd.notna(defensive_segment_return) else np.nan,
            "Portfolio Return %": round(segment_total_return * 100.0, 2),
            "Benchmark Return %": round(benchmark_segment_return * 100.0, 2) if pd.notna(benchmark_segment_return) else np.nan,
            "Defense %": round(float(overlay.get("defensive_blend", 0.0) or 0.0) * 100.0, 1),
            "Cash %": round(float(cash_weight or 0.0) * 100.0, 1),
        })
        portfolio_value = float(segment_values.iloc[-1])
        previous_weights = invested_weights.reindex(valid_tickers).fillna(0.0)
        weight_history_rows.append((rebalance_date, previous_weights.reindex(valid_tickers).fillna(0.0)))

        rebalance_rows.append({
            "Date": rebalance_date.strftime("%Y-%m-%d"),
            "Strategic Regime": str(regime_meta.get("dominant_regime", "unknown")).title(),
            "Tactical Regime": str(regime_meta.get("fast_dominant_regime", regime_meta.get("dominant_regime", "unknown"))).title(),
            "Regime Used": str(regime_meta.get("backtest_regime_label", regime_meta.get("dominant_regime", "unknown"))).title(),
            "Regime Source": str(regime_meta.get("backtest_regime_source", "slow")).title(),
            "Regime Distance": round(overlay["distance_score"], 3),
            "Stress Score": round(overlay["stress_score"], 3),
            "Defensive Blend %": round(overlay["defensive_blend"] * 100, 1),
            "Cash %": round(cash_weight * 100, 1),
            "Alpha %": round(float(regime_meta.get("effective_alpha", alpha_regime)) * 100, 1),
            "Beta %": round(float(regime_meta.get("effective_beta", beta_regime)) * 100, 1),
            "Top Weight": previous_weights.idxmax() if not previous_weights.empty else "—",
            "Top Weight %": round(previous_weights.max() * 100, 1) if not previous_weights.empty else 0.0,
        })

    if not daily_values_segments:
        raise ValueError("Walk-forward engine could not build any rebalance segments.")

    daily_vals = pd.concat(daily_values_segments).sort_index()
    daily_vals = daily_vals[~daily_vals.index.duplicated(keep="last")]
    total_exit = float(daily_vals.iloc[-1])
    total_invest = float(capital)
    abs_return = total_exit - total_invest
    pct_return = abs_return / total_invest * 100.0
    n_days = int((daily_vals.index[-1].date() - daily_vals.index[0].date()).days)
    n_years = n_days / 365.25
    cagr = ((total_exit / total_invest) ** (1 / max(n_years, 0.01)) - 1) * 100.0

    benchmark_close = _cached_download_close_prices(
        (benchmark_symbol,),
        daily_vals.index[0].strftime("%Y-%m-%d"),
        (daily_vals.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    benchmark_close = benchmark_close.squeeze() if benchmark_close is not None and not benchmark_close.empty else pd.Series(dtype=float)
    benchmark_close = pd.Series(benchmark_close).dropna()
    benchmark_curve = None
    benchmark_return = None
    if not benchmark_close.empty and len(benchmark_close) >= 2:
        benchmark_close = benchmark_close.reindex(daily_vals.index).ffill().bfill()
        benchmark_curve = capital * (benchmark_close / max(float(benchmark_close.iloc[0]), 1e-9))
        benchmark_return = (float(benchmark_curve.iloc[-1]) / capital - 1.0) * 100.0

    rebalance_df = pd.DataFrame(rebalance_rows)
    original_weights = _weight_series(weights_bt, tickers_bt).reindex(valid_tickers).fillna(0.0)
    target_weights_latest = _weight_series(
        latest_target_weights if latest_target_weights is not None else original_weights,
        valid_tickers,
    )
    defensive_weights_latest = _weight_series(
        latest_defensive_weights if latest_defensive_weights is not None else original_weights,
        valid_tickers,
    )
    invested_weights_latest = _weight_series(
        latest_invested_weights if latest_invested_weights is not None else original_weights,
        valid_tickers,
    )
    deploy_now_weights = invested_weights_latest * (1.0 - latest_cash_weight)
    weight_history_df = pd.DataFrame(
        [weights for _, weights in weight_history_rows],
        index=[date for date, _ in weight_history_rows],
    ).reindex(columns=valid_tickers).fillna(0.0) if weight_history_rows else pd.DataFrame(columns=valid_tickers)
    disagreement_penalty, disagreement_score = _model_disagreement(
        latest_raw_er.reindex(valid_tickers) if latest_raw_er is not None else None,
        latest_shrink_er.reindex(valid_tickers) if latest_shrink_er is not None else None,
        latest_active_er.reindex(valid_tickers) if latest_active_er is not None else None,
        valid_tickers,
    )
    transition_risk = _transition_risk(latest_regime_meta)
    stability_score = _weight_stability_score(weight_history_df.mul(1.0 - latest_cash_weight))
    target_sum = max(1.0 - latest_cash_weight, 0.0)
    disagreement_adjusted = _bounded_weight_series(
        (deploy_now_weights.reindex(valid_tickers).fillna(0.0) * disagreement_penalty.reindex(valid_tickers).fillna(1.0)).values,
        valid_tickers,
        min_w=0.0,
        max_w=st.session_state.max_w,
        target_sum=target_sum,
    )
    downside_guard = _downside_guard_weights(
        latest_returns,
        valid_tickers,
        target_sum=target_sum,
        min_w=0.0,
        max_w=min(st.session_state.max_w, max(target_sum, 1e-6)),
    )
    base_scenario = float(np.clip(1.0 - 0.35 * transition_risk - 0.20 * float((latest_overlay or {}).get("stress_score", 0.0) or 0.0), 0.25, 0.75))
    weak_scenario = float(np.clip(0.18 + 0.25 * transition_risk + 0.10 * float((latest_overlay or {}).get("stress_score", 0.0) or 0.0), 0.15, 0.45))
    crisis_scenario = float(np.clip(1.0 - base_scenario - weak_scenario, 0.10, 0.35))
    scenario_total = base_scenario + weak_scenario + crisis_scenario
    scenario_mix = {
        "base": base_scenario / scenario_total,
        "weak": weak_scenario / scenario_total,
        "crisis": crisis_scenario / scenario_total,
    }
    weak_weights = _bounded_weight_series(
        (0.65 * disagreement_adjusted + 0.35 * defensive_weights_latest * target_sum).values,
        valid_tickers,
        min_w=0.0,
        max_w=st.session_state.max_w,
        target_sum=target_sum,
    )
    crisis_target_sum = max(target_sum * (1.0 - min(0.18 + 0.22 * transition_risk, 0.35)), 0.0)
    crisis_weights = _downside_guard_weights(
        latest_returns,
        valid_tickers,
        target_sum=crisis_target_sum,
        min_w=0.0,
        max_w=min(st.session_state.max_w, max(crisis_target_sum, 1e-6)),
    )
    scenario_blended = (
        disagreement_adjusted * scenario_mix["base"]
        + weak_weights * scenario_mix["weak"]
        + crisis_weights.reindex(valid_tickers).fillna(0.0) * scenario_mix["crisis"]
    )
    scenario_cash = float(np.clip(
        latest_cash_weight
        + 0.05 * scenario_mix["weak"]
        + 0.18 * scenario_mix["crisis"]
        + 0.08 * transition_risk,
        0.0,
        0.40,
    ))
    final_target_sum = max(1.0 - scenario_cash, 0.0)
    scenario_blended = _bounded_weight_series(
        scenario_blended.values,
        valid_tickers,
        min_w=0.0,
        max_w=st.session_state.max_w,
        target_sum=final_target_sum,
    )
    trust_score = float(np.clip(
        30.0
        + 28.0 * stability_score
        + 0.45 * _deployment_confidence(latest_regime_meta, latest_overlay, pct_return - (benchmark_return if benchmark_return is not None else 0.0))
        - 18.0 * disagreement_score
        - 16.0 * transition_risk,
        15.0,
        95.0,
    ))
    trust_scale = float(np.clip(trust_score / 100.0, 0.25, 0.95))
    blended_for_trust = scenario_blended * trust_scale + defensive_weights_latest.reindex(valid_tickers).fillna(0.0) * final_target_sum * (1.0 - trust_scale)
    final_deploy_weights = _bounded_weight_series(
        blended_for_trust.values,
        valid_tickers,
        min_w=0.0,
        max_w=st.session_state.max_w,
        target_sum=final_target_sum,
    )
    final_total_weights = _apply_execution_friction(
        final_deploy_weights,
        latest_previous_total.fillna(0.0) * max(1.0 - latest_cash_weight, 0.0) if latest_previous_total is not None else original_weights * final_target_sum,
        target_sum=final_target_sum,
        min_w=0.0,
        max_w=st.session_state.max_w,
        min_change=0.0075,
        turnover_budget=0.22,
    )
    confidence_bands = _weight_confidence_bands(weight_history_df.mul(1.0 - latest_cash_weight), final_total_weights)
    deploy_now_weights = final_total_weights
    compare_df = pd.DataFrame({
        "Original %": original_weights.mul(100.0),
        "Tactical %": target_weights_latest.mul(100.0),
        "Defensive %": defensive_weights_latest.mul(100.0),
        "Deploy Now %": deploy_now_weights.mul(100.0),
    }).fillna(0.0)
    compare_df["Delta vs Original pp"] = compare_df["Deploy Now %"] - compare_df["Original %"]
    compare_df = compare_df.sort_values("Deploy Now %", ascending=False)

    posture = _deployment_posture(
        float((latest_overlay or {}).get("defensive_blend", 0.0) or 0.0),
        float(scenario_cash or 0.0),
        float((latest_overlay or {}).get("stress_score", 0.0) or 0.0),
        float((latest_overlay or {}).get("distance_score", 0.0) or 0.0),
    )
    relative_alpha = pct_return - (benchmark_return if benchmark_return is not None else 0.0)
    confidence_score = _deployment_confidence(latest_regime_meta, latest_overlay, relative_alpha)
    top_adds = compare_df.sort_values("Delta vs Original pp", ascending=False).head(3)
    top_cuts = compare_df.sort_values("Delta vs Original pp", ascending=True).head(3)
    explanations_df = _ticker_adjustment_explanations(
        compare_df,
        confidence_bands,
        disagreement_penalty,
        latest_regime_meta,
        top_n=min(8, len(compare_df)),
    )
    calibration_df = pd.DataFrame(calibration_rows)
    if not calibration_df.empty:
        calibration_df["Relative vs Benchmark pp"] = calibration_df["Deploy Return %"] - calibration_df["Benchmark Return %"]
        calibration_df["Winner"] = np.where(
            calibration_df["Deploy Return %"] >= calibration_df["Benchmark Return %"],
            "Deploy held up better",
            "Benchmark held up better",
        )
        calibration_df = calibration_df.sort_values("Date", ascending=False).reset_index(drop=True)
    deployable = {
        "date": latest_rebalance_date.strftime("%Y-%m-%d") if latest_rebalance_date is not None else daily_vals.index[-1].strftime("%Y-%m-%d"),
        "posture": posture,
        "confidence": confidence_score,
        "trust_score": trust_score,
        "cash_weight": float(scenario_cash),
        "defensive_blend": float((latest_overlay or {}).get("defensive_blend", 0.0) or 0.0),
        "stress_score": float((latest_overlay or {}).get("stress_score", 0.0) or 0.0),
        "distance_score": float((latest_overlay or {}).get("distance_score", 0.0) or 0.0),
        "transition_risk": transition_risk,
        "stability_score": stability_score,
        "disagreement_score": disagreement_score,
        "strategic_regime": (latest_regime_meta or {}).get("dominant_regime", "unknown"),
        "tactical_regime": (latest_regime_meta or {}).get("fast_dominant_regime", (latest_regime_meta or {}).get("dominant_regime", "unknown")),
        "regime_used": (latest_regime_meta or {}).get("backtest_regime_label", (latest_regime_meta or {}).get("dominant_regime", "unknown")),
        "regime_source": (latest_regime_meta or {}).get("backtest_regime_source", "slow"),
        "weights": deploy_now_weights.round(6).to_dict(),
        "original_weights": original_weights.round(6).to_dict(),
        "tactical_weights": target_weights_latest.round(6).to_dict(),
        "defensive_weights": defensive_weights_latest.round(6).to_dict(),
        "downside_guard_weights": downside_guard.round(6).to_dict(),
        "confidence_bands": confidence_bands,
        "scenario_mix": {k: round(v * 100.0, 1) for k, v in scenario_mix.items()},
        "comparison": compare_df.round(2),
        "explanations": explanations_df,
        "calibration": calibration_df.round(2) if not calibration_df.empty else calibration_df,
        "top_adds": top_adds.round(2),
        "top_cuts": top_cuts.round(2),
    }
    return {
        "daily_vals": daily_vals,
        "rebalance_df": rebalance_df,
        "overall_entry": daily_vals.index[0].strftime("%Y-%m-%d"),
        "overall_exit": daily_vals.index[-1].strftime("%Y-%m-%d"),
        "total_invest": total_invest,
        "total_exit": total_exit,
        "abs_return": abs_return,
        "pct_return": pct_return,
        "cagr": cagr,
        "n_days": n_days,
        "n_years": n_years,
        "residual": residual_cash,
        "benchmark_symbol": benchmark_symbol,
        "benchmark_curve": benchmark_curve,
        "benchmark_return": benchmark_return,
        "deployable": deployable,
    }

# ── Card-nav session state ────────────────────────────────────
if "qs_section" not in st.session_state:
    st.session_state.qs_section = "build"


if not st.session_state.user_store_loaded:
    sync_user_store_to_session()
    st.session_state.user_store_loaded = True

consume_bridge_symbols()

try:
    _helper_mode = st.query_params.get("helper", "")
except Exception:
    _helper_mode = ""
if _helper_mode in {"upload", "sector"}:
    render_universe_helper(_helper_mode)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Mode toggle (always visible, determines everything else) ─
    # Returns "noob" or "neat"; also injects NOOB_CSS into the page.
    _app_mode = render_mode_toggle()
    is_noob = (_app_mode == "noob")

    # ── Inject global green overrides ONLY in noob mode ──────────
    # These rules target [data-testid="stSidebar"], .stApp, tabs etc.
    # — global selectors that must NOT fire in neat mode.
    if is_noob:
        st.markdown("""
<style>
/* ═══ NOOB MODE GLOBAL OVERRIDES — only active when is_noob=True ══ */

/* Sidebar background */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #031a0e 0%, #042b12 40%, #051f0d 100%) !important;
    border-right: 1.5px solid #1a5c2a !important;
}
[data-testid="stSidebar"] * { color: #a8f0b8 !important; }
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea {
    background: #071f0e !important;
    border: 1px solid #1e6b30 !important;
    color: #c8ffd4 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #0d6b2a, #0a8c32) !important;
    color: #c8ffd4 !important;
    border: 1px solid #1aad44 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #12882f, #0fad3c) !important;
    box-shadow: 0 0 14px rgba(26,173,68,0.45) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div { background: #0d6b2a !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] p { color: #7dd89a !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #4afa7a !important; }
[data-testid="stSidebar"] hr { border-color: #1a5c2a !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(10,40,18,0.7) !important;
    border: 1px solid #1a5c2a !important;
    border-radius: 10px !important;
}

/* Main body background */
.stApp, [data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #020d06 0%, #031508 50%, #041209 100%) !important;
}
.block-container { background: transparent !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #041209 !important;
    border-bottom: 1px solid #1a5c2a !important;
}
.stTabs [data-baseweb="tab"]       { color: #3db85a !important; }
.stTabs [aria-selected="true"]     { color: #4afa7a !important; border-bottom: 2px solid #22c55e !important; }

/* Expanders and DataFrames in main area */
[data-testid="stDataFrame"] {
    border: 1px solid #1a5c2a !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] {
    background: rgba(4,20,10,0.8) !important;
    border: 1px solid #1a5c2a !important;
    border-radius: 10px !important;
}
[data-testid="stAlert"] { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

    if is_noob:
        # ════════════════════════════════════════════════════════
        # NOOB SIDEBAR — clean, green, minimal
        # ════════════════════════════════════════════════════════
        st.markdown(
            f'''<div class="nb-green-sidebar-header">
                <img src="data:image/png;base64,{_LOGO_B64}"
                     style="width:64px;height:64px;border-radius:14px;
                            box-shadow:0 0 18px rgba(0,200,80,0.4);
                            margin-bottom:8px;">
                <div class="nb-app-name">QuanSen</div>
                <div class="nb-app-sub">Easy Mode · v{APP_VERSION}</div>
            </div>''',
            unsafe_allow_html=True
        )

        # ── Status dots ──────────────────────────────────────────
        _data_ok  = st.session_state.get("data_loaded", False)
        _ports_ok = st.session_state.get("portfolios_computed", False)
        _n_t      = len(st.session_state.get("tickers", []))

        st.markdown(
            f'''<div style="margin-bottom:1rem;padding:0.6rem 0.8rem;
                            background:rgba(10,40,18,0.6);border-radius:10px;
                            border:1px solid #1a5c2a;">
                <div style="font-size:0.65rem;color:#3db85a;letter-spacing:0.15em;
                            text-transform:uppercase;margin-bottom:6px">Status</div>
                <div style="font-size:0.78rem;margin-bottom:3px">
                    <span class="nb-green-status-dot {'red' if _n_t==0 else ''}"></span>
                    {_n_t} stock{"s" if _n_t!=1 else ""} selected
                </div>
                <div style="font-size:0.78rem;margin-bottom:3px">
                    <span class="nb-green-status-dot {'red' if not _data_ok else ''}"></span>
                    Data {"loaded ✓" if _data_ok else "not loaded yet"}
                </div>
                <div style="font-size:0.78rem">
                    <span class="nb-green-status-dot {'red' if not _ports_ok else ''}"></span>
                    Results {"ready ✓" if _ports_ok else "not run yet"}
                </div>
            </div>''',
            unsafe_allow_html=True
        )

        # ── Quick-load preset ────────────────────────────────────
        st.markdown('<div class="nb-green-heading">🚀 Quick Start</div>', unsafe_allow_html=True)
        if st.button("🎲 Load 20 Indian Stocks", use_container_width=True, key="nb_sb_preset"):
            st.session_state.tickers = [
                'BEL.NS','VEDL.NS','BAJFINANCE.NS','BEML.NS','ADVAIT.BO',
                'ADANIENT.NS','COALINDIA.NS','CROMPTON.NS','KINGFA.NS','KRISHNADEF.NS',
                'LT.NS','LUPIN.NS','MAZDOCK.NS','PENIND.NS','PNB.NS',
                'RELIANCE.NS','SHUKRAPHAR.BO','TARIL.NS','HDFCBANK.NS','SBIN.NS'
            ]
            st.session_state.start_date = _demo_start_date
            st.session_state.end_date   = _demo_end_date
            reset_portfolio_outputs()
            st.rerun()

        # ── Add stocks ───────────────────────────────────────────
        st.markdown('<div class="nb-green-heading">➕ Add a Stock</div>', unsafe_allow_html=True)
        _nb_sb_search = st.text_input(
            "Search or type a ticker",
            placeholder="e.g. Reliance or SBIN.NS",
            key="nb_sb_search_input",
            label_visibility="collapsed",
        )
        if st.button("🔍 Add Stock", use_container_width=True, key="nb_sb_add_btn"):
            if _nb_sb_search.strip():
                with st.spinner("Finding best match…"):
                    try:
                        _resolved = get_best_ticker(
                            _nb_sb_search.strip().upper(),
                            st.session_state.start_date,
                            st.session_state.end_date,
                        )
                        if _resolved not in st.session_state.tickers:
                            st.session_state.tickers.append(_resolved)
                            reset_portfolio_outputs()
                            st.success(f"Added {_resolved}")
                            st.rerun()
                        else:
                            st.warning(f"{_resolved} is already in your list.")
                    except Exception as _e:
                        st.error(f"Could not find: {_nb_sb_search.strip()}")

        # ── Basket preview + remove ──────────────────────────────
        if st.session_state.tickers:
            st.markdown('<div class="nb-green-heading">📋 Your Stocks</div>', unsafe_allow_html=True)
            for _i, _t in enumerate(st.session_state.tickers):
                _c1, _c2 = st.columns([4, 1])
                with _c1:
                    st.markdown(
                        f'<span class="nb-stock-chip">🟢 {_t}</span>',
                        unsafe_allow_html=True
                    )
                with _c2:
                    if st.button("✕", key=f"nb_sb_rm_{_t}_{_i}"):
                        st.session_state.tickers.remove(_t)
                        reset_portfolio_outputs()
                        st.rerun()

            if st.button("🗑 Clear All", use_container_width=True, key="nb_sb_clear"):
                st.session_state.tickers = []
                reset_portfolio_outputs()
                st.rerun()

        # ── Date range ───────────────────────────────────────────
        st.markdown('<div class="nb-green-heading">📅 Time Period</div>', unsafe_allow_html=True)
        import datetime as _nb_dt
        _nb_days = st.select_slider(
            "How far back?",
            options=[365, 730, 1095, 1825, 2555],
            value=1825,
            format_func=lambda v: f"{v//365} yr{'s' if v//365 > 1 else ''}",
            key="nb_sb_days_back",
            label_visibility="collapsed",
        )
        _nb_today  = _nb_dt.date.today()
        _nb_start  = _nb_today - _nb_dt.timedelta(days=_nb_days)
        st.session_state.start_date = _nb_start.strftime("%Y-%m-%d")
        st.session_state.end_date   = _nb_today.strftime("%Y-%m-%d")
        st.caption(f"📆 {st.session_state.start_date} → {st.session_state.end_date}")

        # ── Load & Run ───────────────────────────────────────────
        st.markdown('<div class="nb-green-heading">⚡ Actions</div>', unsafe_allow_html=True)
        if len(st.session_state.tickers) >= 2:
            if st.button("📡 Load Data", use_container_width=True, key="nb_sb_load"):
                request_action("load_data")
                st.rerun()
            if st.button(
                "🚀 Run Magic!",
                use_container_width=True,
                key="nb_sb_run",
                disabled=not st.session_state.data_loaded,
            ):
                request_action("run_all")
                st.rerun()
        else:
            st.caption("⬆ Add at least 2 stocks first.")

        st.markdown("---")
        st.markdown(
            f'<div style="font-size:0.6rem;color:#1a5c2a;text-align:center;">'
            f'QuanSen Easy Mode · v{APP_VERSION}<br>'
            f'MPT · CVXPY · by Amatra Sen</div>',
            unsafe_allow_html=True
        )

    else:
        # ════════════════════════════════════════════════════════
        # NEAT SIDEBAR — full original controls
        # ════════════════════════════════════════════════════════
        st.markdown(
            f'''<div style="text-align:center;padding:0.4rem 0 1rem">
                <img src="data:image/png;base64,{_LOGO_B64}"
                     style="width:90px;height:90px;border-radius:18px;
                            box-shadow:0 0 18px rgba(0,180,255,0.35);">
                <div style="font-family:'Syne',sans-serif;font-size:0.65rem;
                            font-weight:700;color:#4a6a90;letter-spacing:0.18em;
                            text-transform:uppercase;margin-top:6px">
                    Portfolio Optimizer v{APP_VERSION}
                </div>
            </div>''',
            unsafe_allow_html=True
        )

        st.markdown('<div class="card-title">⚙ Control Deck</div>', unsafe_allow_html=True)

        with st.expander("Universe Builder", expanded=True):
            if st.button("Load Test Portfolio (20 stocks)", use_container_width=True, key="load_test_portfolio_btn"):
                st.session_state.tickers = [
                    'BEL.NS','VEDL.NS','BAJFINANCE.NS','BEML.NS','ADVAIT.BO',
                    'ADANIENT.NS','COALINDIA.NS','CROMPTON.NS','KINGFA.NS','KRISHNADEF.NS',
                    'LT.NS','LUPIN.NS','MAZDOCK.NS','PENIND.NS','PNB.NS',
                    'RELIANCE.NS','SHUKRAPHAR.BO','TARIL.NS','HDFCBANK.NS','SBIN.NS'
                ]
                st.session_state.start_date = _demo_start_date
                st.session_state.end_date = _demo_end_date
                reset_portfolio_outputs()
                st.rerun()

            use_sector_helper = st.toggle(
                "Use helper window for sectors",
                value=True,
                key="sector_helper_toggle",
                help="Opens the sector picker in a separate tab so this sidebar stays clean.",
            )
            if use_sector_helper:
                st.markdown(
                    """
                    <a class="builder-helper-link" href="?helper=sector" target="_blank">
                        <div class="builder-helper-title">Open Sector Basket Desk</div>
                        <div class="builder-helper-copy">Choose a sector preset in a separate tab and send it back here.</div>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                sector_names = ["Custom"] + sorted(SECTOR_PRESETS.keys())
                selected_sector = st.selectbox(
                    "Sector preset",
                    options=sector_names,
                    key="sector_preset_select",
                    help="Load a ready-made basket instead of typing symbols one by one.",
                )
                sector_symbols = SECTOR_PRESETS.get(selected_sector, [])
                if sector_symbols:
                    st.caption(f"{len(sector_symbols)} symbols ready in {selected_sector}.")
                    col_sector_1, col_sector_2 = st.columns(2)
                    with col_sector_1:
                        if st.button("Replace Basket", key="replace_sector_btn", use_container_width=True):
                            apply_ticker_set(sector_symbols, mode="replace")
                            st.success(f"Loaded {selected_sector}")
                            st.rerun()
                    with col_sector_2:
                        if st.button("Append Sector", key="append_sector_btn", use_container_width=True):
                            apply_ticker_set(sector_symbols, mode="append")
                            st.success(f"Appended {selected_sector}")
                            st.rerun()

            if st.session_state.builder_prefill is not None:
                st.session_state.builder_symbols_raw = st.session_state.builder_prefill
                st.session_state.builder_prefill = None
            builder_default = st.session_state.get("builder_symbols_raw") or ", ".join(st.session_state.tickers[:20])
            builder_symbols_raw = st.text_area(
                "Paste symbols",
                value=builder_default,
                key="builder_symbols_raw",
                height=85,
                placeholder="SBIN.NS, HDFCBANK.NS, ICICIBANK.NS",
                help="Comma, newline, or semicolon separated symbols are all fine.",
            )
            use_upload_helper = st.toggle(
                "Use helper window for file upload",
                value=True,
                key="upload_helper_toggle",
                help="Opens the file uploader in a separate tab for CSV/TXT/Excel imports.",
            )
            imported_symbols = []
            if use_upload_helper:
                st.markdown(
                    """
                    <a class="builder-helper-link" href="?helper=upload" target="_blank">
                        <div class="builder-helper-title">Open File Upload Desk</div>
                        <div class="builder-helper-copy">Upload a basket file in a separate tab and send the cleaned symbols back here.</div>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                upload = st.file_uploader(
                    "Import symbols from CSV/XLSX/TXT",
                    type=["csv", "txt", "xlsx", "xls"],
                    key="builder_file_upload",
                )
                imported_symbols = parse_uploaded_symbols(upload)
            if imported_symbols:
                st.caption(f"Imported {len(imported_symbols)} symbols from file.")

            manual_symbols = normalize_symbol_list(builder_symbols_raw)
            candidate_symbols = imported_symbols or manual_symbols
            if candidate_symbols:
                st.caption(f"Builder preview: {len(candidate_symbols)} symbols")

            b1, b2 = st.columns(2)
            with b1:
                if st.button("Replace Basket", key="replace_basket_btn", use_container_width=True):
                    if candidate_symbols:
                        apply_ticker_set(candidate_symbols, mode="replace")
                        st.success(f"Loaded {len(candidate_symbols)} symbols")
                        st.rerun()
                    st.warning("Paste or import symbols first.")
            with b2:
                if st.button("Append Basket", key="append_basket_btn", use_container_width=True):
                    if candidate_symbols:
                        apply_ticker_set(candidate_symbols, mode="append")
                        st.success(f"Appended {len(candidate_symbols)} symbols")
                        st.rerun()
                    st.warning("Paste or import symbols first.")
            if st.button("Clear Builder", key="clear_builder_btn", use_container_width=True):
                st.session_state.builder_prefill = ""
                st.rerun()

            if st.session_state.tickers:
                st.markdown(
                    '<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;'
                    'margin:0.65rem 0 0.45rem">Selected Basket</div>',
                    unsafe_allow_html=True
                )
                preview = ", ".join(st.session_state.tickers[:12])
                if len(st.session_state.tickers) > 12:
                    preview += f" ... +{len(st.session_state.tickers) - 12} more"
                st.caption(preview)
                st.caption(f"{len(st.session_state.tickers)} tickers active")
                if st.button("🗑 Clear All Tickers", use_container_width=True, key="clear_all_tickers_btn"):
                    st.session_state.tickers = []
                    reset_portfolio_outputs()
                    st.rerun()

        with st.expander("Date & Rules", expanded=True):
            date_mode = st.radio(
                "Date selection mode",
                ["📅 Start / End dates", "⏱ Days back from today"],
                label_visibility="collapsed",
                horizontal=False,
                key="date_mode"
            )

            if date_mode == "📅 Start / End dates":
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Start")
                    st.session_state.start_date = st.text_input(
                        "Start", value=st.session_state.start_date,
                        label_visibility="collapsed", placeholder="YYYY-MM-DD", key="si_start"
                    )
                with col2:
                    st.caption("End")
                    st.session_state.end_date = st.text_input(
                        "End", value=st.session_state.end_date,
                        label_visibility="collapsed", placeholder="YYYY-MM-DD", key="si_end"
                    )
            else:
                import datetime as _dt
                days_back = st.number_input(
                    "Number of days back from today", min_value=30, max_value=7300,
                    value=365 * 3, step=30, key="days_back_input"
                )
                _today = _dt.date.today()
                _start = _today - _dt.timedelta(days=int(days_back))
                st.session_state.start_date = _start.strftime("%Y-%m-%d")
                st.session_state.end_date = _today.strftime("%Y-%m-%d")
                st.caption(f"▶ {st.session_state.start_date}  →  {st.session_state.end_date}  ({int(days_back)} days)")

            n_t = max(len(st.session_state.tickers), 1)
            auto_max = compute_max_w(n_t)
            st.caption(f"Auto max-weight for {n_t} assets: {auto_max*100:.0f}%")
            min_w_pct = st.slider("Min weight %", 0, 10, int(st.session_state.min_w * 100), 1)
            max_w_pct = st.slider("Max weight %", 5, 50, int(auto_max * 100), 5)
            st.session_state.min_w = min_w_pct / 100
            st.session_state.max_w = max_w_pct / 100

            target_pct = st.slider("Target return %", 1, 60, int(st.session_state.target_return * 100), 1)
            st.session_state.target_return = target_pct / 100

            auto_alpha = st.toggle(
                "Automatic α (market regime)",
                value=st.session_state.get("auto_alpha", True),
                key="auto_alpha_toggle",
                help="Automatically determine shrinkage α using the market regime engine."
            )
            st.session_state.auto_alpha = auto_alpha

            if auto_alpha and st.session_state.tickers:
                alpha_regime, _, regime_meta = _get_regime_state(force=False)
                st.session_state.shrinkage_alpha = alpha_regime
                alpha_pct = int(alpha_regime * 100)
                st.caption(
                    f"Automatic α = **{alpha_pct}%**  ·  regime: {max(regime_meta['regime_probabilities'], key=regime_meta['regime_probabilities'].get)}"
                )
            else:
                alpha_pct = st.slider(
                    "Shrinkage α",
                    0, 100,
                    int(st.session_state.get("shrinkage_alpha", 0.7) * 100),
                    5,
                    help=(
                        "α controls how much to trust each stock's own historical mean vs the market index.\n\n"
                        "α=100%  → pure history\n"
                        "α=70%   → 70% history + 30% benchmark\n"
                        "α=0%    → fully benchmark driven"
                    )
                )
                st.session_state.shrinkage_alpha = alpha_pct / 100

            _bm = st.session_state.get("bm_sym")
            st.caption(
                f"α = {int(st.session_state.shrinkage_alpha*100)}%  ·  benchmark = {_bm if _bm else '(auto after load)'}"
            )

        with st.expander("Momentum", expanded=False):
            mom_enabled = st.toggle(
                "Enable momentum tilt",
                value=st.session_state.momentum_enabled,
                key="mom_toggle",
                help="When ON, the optimizer uses β-blended expected returns."
            )

        st.session_state.momentum_enabled = mom_enabled

        if mom_enabled:
            auto_regime = st.toggle(
                "Automatic regime detection",
                value=st.session_state.get("auto_regime", True),
                key="auto_regime_toggle",
                help="Use the market_state_engine to automatically determine α and β."
            )
            st.session_state.auto_regime = auto_regime

            if auto_regime and st.session_state.tickers:
                alpha_regime, beta_regime, regime_meta = _get_regime_state(force=False)
                st.session_state.shrinkage_alpha = alpha_regime
                st.session_state.momentum_beta = beta_regime

                st.markdown("#### 📊 Market Regime")
                trend = regime_meta["trend"]
                vol_ratio = regime_meta["vol_ratio"]
                drawdown = regime_meta["drawdown"]
                probs = regime_meta["regime_probabilities"]

                st.markdown(
                    f"""
                    <div style="font-size:13px;line-height:1.7;margin-bottom:8px;">
                    <b>Trend</b> <span style="float:right;">{trend:.3f}</span><br>
                    <b>Volatility</b> <span style="float:right;">{vol_ratio:.2f}</span><br>
                    <b>Drawdown</b> <span style="float:right;">{abs(drawdown):.2%}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("**Regime probabilities**")
                colors = {"bull":"#00ff9c","sideways":"#ffd166","bear":"#ff4d6d","crisis":"#9b5de5"}
                for regime, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                    color = colors.get(regime, "#4a6a90")
                    st.markdown(
                        f"""<div style="margin-top:6px;font-size:12px;">{regime.capitalize()} — {p*100:.1f}%</div>
                        <div style="background:#1a2132;border-radius:6px;overflow:hidden;height:8px;box-shadow:0 0 6px {color};">
                            <div style="width:{p*100}%;height:100%;background:{color};box-shadow:0 0 12px {color};"></div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                dominant = max(probs, key=probs.get)
                regime_scale = {"crisis":0.1,"bear":0.3,"sideways":0.6,"bull":0.9}
                gauge_value  = regime_scale.get(dominant, 0.5)
                labels = {"bull":"🟢 Bull Market","sideways":"🟡 Sideways Market","bear":"🟠 Bear Market","crisis":"🔴 Crisis Market"}
                badge_colors = {"bull":"#00ff9c","sideways":"#ffd166","bear":"#ff4d6d","crisis":"#9b5de5"}
                badge_color  = badge_colors.get(dominant)

                st.markdown(
                    f"""<div style="padding:8px;margin-top:15px;border-radius:8px;text-align:center;
                    background:linear-gradient(135deg,{badge_color}33,#0f1724);
                    border:1px solid {badge_color};box-shadow:0 0 10px {badge_color};font-weight:500;">
                    Detected Regime: {labels[dominant]}</div>""",
                    unsafe_allow_html=True
                )

                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=gauge_value,
                    number={"suffix":" score","font":{"size":28,"color":"#e6f0ff"}},
                    gauge={
                        "axis":{"range":[0,1],"tickvals":[0.1,0.35,0.6,0.85],
                                "ticktext":["Crisis","Bear","Sideways","Bull"],"tickfont":{"size":11,"color":"#cbd5e1"}},
                        "bar":{"color":"#00e5ff","thickness":0.12},
                        "steps":[{"range":[0,0.2],"color":"#7c3aed"},{"range":[0.2,0.45],"color":"#ef4444"},
                                 {"range":[0.45,0.75],"color":"#facc15"},{"range":[0.75,1],"color":"#22c55e"}]
                    }
                ))
                fig.update_layout(height=160,margin=dict(l=20,r=20,t=10,b=0),
                                  paper_bgcolor="rgba(0,0,0,0)",font={"color":"#e2e8f0","family":"Inter"})
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("How market regime is calculated"):
                    st.markdown("""
                The regime detection engine now uses a fuller market-health read:
                **Trend** - multi-horizon direction of the market
                **Volatility** - stress and vol-of-vol
                **Drawdown** - depth and speed of recent damage
                **Breadth & participation** - how many basket names are still healthy
                **Correlation stress** - whether stocks are moving as one block
                Those signals become Bull / Sideways / Bear / Crisis probabilities, and auto-lookback then turns that mix into a shorter or longer momentum window.
                    """)
            else:
                _lookback_options = list(range(42, 253, 21))
                current_lb = st.session_state.get("momentum_lookback", 252)
                lb_choice = st.select_slider(
                    "Lookback window", options=_lookback_options,
                    value=current_lb if current_lb in _lookback_options else 252,
                    format_func=lambda v: f"{v}d (~{v//21}m)",
                    help="Measured backward from the selected end date."
                )
                st.session_state.momentum_lookback = lb_choice
                beta_pct = st.slider(
                    "β — history vs momentum", 0, 100,
                    int(st.session_state.get("momentum_beta", 0.6) * 100), 5
                )
                st.session_state.momentum_beta = beta_pct / 100
                st.caption(f"β = {beta_pct}%  ·  lookback = {lb_choice}d  ·  skip = 21d")

            if st.session_state.data_loaded:
                momentum_requested = (
                    st.button("⚡ Compute Momentum", use_container_width=True, key="quant_compute_momentum_btn")
                    or consume_action("compute_momentum")
                )
                if momentum_requested:
                    with st.spinner("Computing momentum signals…"):
                        try:
                            scores, signals, final_er, meta = cached_compute_momentum(
                                tuple(st.session_state.tickers),
                                st.session_state.start_date,
                                st.session_state.end_date,
                                st.session_state.expected_returns,
                                lookback=st.session_state.get("momentum_lookback", 252),
                                skip=21,
                                beta=st.session_state.momentum_beta,
                                auto_lookback=st.session_state.auto_regime,
                                prev_regime_probs=st.session_state.get("regime_prev_probs"),
                            )
                            st.session_state.momentum_scores   = scores
                            st.session_state.momentum_signals  = signals
                            st.session_state.momentum_final_er = final_er
                            st.session_state.momentum_meta     = meta
                            _persist_regime_state(meta.get("regime_meta"))
                            st.session_state.asset_returns     = final_er.reindex(st.session_state.tickers) * 252
                            st.session_state.portfolios_computed = False
                            st.session_state.frontier_computed   = False
                            _commit_regime_snapshot(
                                lookback=meta.get("lookback"),
                                source="quant-momentum",
                            )
                            n_strong = (signals == "Strong").sum()
                            n_weak   = (signals == "Weak").sum()
                            st.success(f"✔ Momentum computed  ·  🟢 {n_strong} Strong  ·  🔴 {n_weak} Weak")
                        except Exception as e:
                            import traceback
                            st.error(f"Momentum compute failed: {e}")
                            st.code(traceback.format_exc())
            else:
                st.caption("⚠ Load data first to compute momentum.")

        with st.expander("💾 Saved Portfolios", expanded=False):
            save_name = st.text_input("Portfolio name", key="save_portfolio_name", placeholder="e.g. India Momentum Basket")
            csave1, csave2 = st.columns(2)
            with csave1:
                if st.button("Save Current", key="save_current_portfolio", use_container_width=True):
                    if st.session_state.tickers and save_name.strip():
                        st.session_state.saved_portfolios[save_name.strip()] = {
                            **current_portfolio_payload(),
                            "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                        }
                        st.session_state.active_saved_portfolio = save_name.strip()
                        persist_session_user_store()
                        st.success(f"Saved portfolio: {save_name.strip()}")
                    else:
                        st.warning("Add tickers and enter a portfolio name first.")
            with csave2:
                if st.button("Quick Save", key="save_test_portfolio", use_container_width=True):
                    name = save_name.strip() or f"Portfolio {pd.Timestamp.now().strftime('%H%M%S')}"
                    st.session_state.saved_portfolios[name] = {
                        **current_portfolio_payload(),
                        "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    }
                    st.session_state.active_saved_portfolio = name
                    persist_session_user_store()
                    st.success(f"Saved portfolio: {name}")

            saved_names = sorted(st.session_state.saved_portfolios.keys())
            if saved_names:
                selected_saved = st.selectbox(
                    "Saved sets", options=saved_names,
                    index=saved_names.index(st.session_state.active_saved_portfolio) if st.session_state.active_saved_portfolio in saved_names else 0,
                    key="saved_portfolio_picker",
                )
                st.session_state.active_saved_portfolio = selected_saved
                meta = st.session_state.saved_portfolios[selected_saved]
                st.caption(f"{len(meta.get('tickers', []))} tickers  ·  {meta.get('start_date')} → {meta.get('end_date')}  ·  updated {meta.get('updated_at', 'N/A')}")
                pl1, pl2 = st.columns(2)
                with pl1:
                    if st.button("Load Saved", key="load_saved_portfolio", use_container_width=True):
                        load_portfolio_payload(meta)
                        st.success(f"Loaded: {selected_saved}")
                        st.rerun()
                with pl2:
                    if st.button("Delete Saved", key="delete_saved_portfolio", use_container_width=True):
                        st.session_state.saved_portfolios.pop(selected_saved, None)
                        st.session_state.active_saved_portfolio = None
                        persist_session_user_store()
                        st.rerun()
            else:
                st.caption("No saved portfolios yet.")

        with st.expander("👀 Watchlists & Alerts", expanded=False):
            wl_name = st.text_input("Watchlist name", key="watchlist_name", placeholder="e.g. Defence Radar")
            if st.session_state.watchlist_prefill is not None:
                st.session_state.watchlist_symbols_raw = st.session_state.watchlist_prefill
                st.session_state.watchlist_prefill = None
            wl_symbols_default = st.session_state.get("watchlist_symbols_raw")
            if wl_symbols_default is None:
                wl_symbols_default = ", ".join(st.session_state.tickers[:12]) if st.session_state.tickers else ""
            wl_symbols_raw = st.text_area("Symbols (comma separated)", value=wl_symbols_default, key="watchlist_symbols_raw", height=70)
            w1, w2 = st.columns(2)
            with w1:
                if st.button("Save Watchlist", key="save_watchlist", use_container_width=True):
                    symbols = [s.strip().upper() for s in wl_symbols_raw.split(",") if s.strip()]
                    if wl_name.strip() and symbols:
                        st.session_state.watchlists[wl_name.strip()] = {"symbols": symbols, "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
                        st.session_state.active_watchlist = wl_name.strip()
                        persist_session_user_store()
                        st.success(f"Saved watchlist: {wl_name.strip()}")
                    else:
                        st.warning("Enter a watchlist name and at least one symbol.")
            with w2:
                if st.button("Use Current", key="watchlist_use_current", use_container_width=True):
                    if st.session_state.tickers:
                        st.session_state.watchlist_prefill = ", ".join(st.session_state.tickers)
                        st.rerun()
            watch_names = sorted(st.session_state.watchlists.keys())
            if watch_names:
                selected_watch = st.selectbox("Saved watchlists", options=watch_names,
                    index=watch_names.index(st.session_state.active_watchlist) if st.session_state.active_watchlist in watch_names else 0,
                    key="watchlist_picker")
                st.session_state.active_watchlist = selected_watch
                watch_symbols = st.session_state.watchlists[selected_watch].get("symbols", [])
                st.caption(f"{len(watch_symbols)} symbols  ·  updated {st.session_state.watchlists[selected_watch].get('updated_at', 'N/A')}")
                watch_quotes = cached_fetch_tape_quotes(tuple(watch_symbols))
                if watch_quotes:
                    watch_df = pd.DataFrame([{"Symbol": q["symbol"],"Price": round(q["price"],3),"Chg %": round(q["chg"],2)} for q in watch_quotes])
                    st.dataframe(watch_df, use_container_width=True, height=min(220, 45 + 35 * len(watch_df)))
                ww1, ww2 = st.columns(2)
                with ww1:
                    if st.button("Load to Tickers", key="load_watchlist_tickers", use_container_width=True):
                        st.session_state.tickers = list(watch_symbols)
                        reset_portfolio_outputs()
                        st.rerun()
                with ww2:
                    if st.button("Delete Watch", key="delete_watchlist", use_container_width=True):
                        st.session_state.watchlists.pop(selected_watch, None)
                        st.session_state.active_watchlist = None
                        persist_session_user_store()
                        st.rerun()
            else:
                st.caption("No watchlists saved yet.")

            st.markdown("---")
            st.caption("Price alerts")
            alert_symbol_default = st.session_state.tickers[0] if st.session_state.tickers else ""
            a1, a2 = st.columns(2)
            with a1:
                alert_symbol = st.text_input("Symbol", value=alert_symbol_default, key="alert_symbol")
            with a2:
                alert_condition = st.selectbox("Condition", ["above", "below"], key="alert_condition")
            a3, a4 = st.columns([1.2, 1])
            with a3:
                alert_threshold = st.number_input("Threshold", min_value=0.0, value=100.0, step=1.0, key="alert_threshold")
            with a4:
                alert_note = st.text_input("Note", key="alert_note", placeholder="optional")
            if st.button("Add Alert", key="add_alert_btn", use_container_width=True):
                if alert_symbol.strip():
                    st.session_state.alerts.append({
                        "symbol": alert_symbol.strip().upper(),
                        "condition": alert_condition,
                        "threshold": float(alert_threshold),
                        "note": alert_note.strip(),
                        "enabled": True,
                    })
                    persist_session_user_store()
                    st.success(f"Alert added for {alert_symbol.strip().upper()}")
                    st.rerun()
            alert_hits, _quote_map = evaluate_alerts(st.session_state.alerts)
            if alert_hits:
                st.markdown('<div class="status-box status-warn">Triggered alerts</div>', unsafe_allow_html=True)
                for hit in alert_hits[:6]:
                    st.caption(f"{hit['symbol']} at {hit['price']:.3f} is {hit['condition']} {hit['threshold']:.3f}" + (f" · {hit['note']}" if hit['note'] else ""))
            if st.session_state.alerts:
                for idx, alert in enumerate(st.session_state.alerts):
                    c_alert1, c_alert2 = st.columns([4, 1])
                    with c_alert1:
                        st.markdown(f"`{alert['symbol']}`  {alert['condition']}  `{alert['threshold']:.3f}`" + (f"  ·  {alert['note']}" if alert.get("note") else ""))
                    with c_alert2:
                        if st.button("✕", key=f"del_alert_{idx}"):
                            st.session_state.alerts.pop(idx)
                            persist_session_user_store()
                            st.rerun()

        with st.expander("Monitoring", expanded=False):
            st.markdown('<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem">📡 Live Indices Tape</div>', unsafe_allow_html=True)
            _ALL_INDICES = {
                "Nifty 50":"^NSEI","BSE Sensex":"^BSESN","Nifty Bank":"^NSEBANK","Nifty IT":"^CNXIT","Nifty Midcap":"NIFTY_MIDCAP_100.NS",
                "S&P 500":"^GSPC","Nasdaq 100":"^NDX","Dow Jones":"^DJI","Russell 2000":"^RUT","VIX":"^VIX","FTSE 100":"^FTSE",
                "DAX":"^GDAXI","Nikkei 225":"^N225","Hang Seng":"^HSI","Shanghai Comp.":"000001.SS","Gold":"GC=F","Silver":"SI=F",
                "Crude Oil WTI":"CL=F","Brent Crude":"BZ=F","Natural Gas":"NG=F","USD/INR":"INR=X","EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X",
                "USD/JPY":"USDJPY=X","Bitcoin":"BTC-USD","Ethereum":"ETH-USD",
            }
            _default_indices = ["Nifty 50","BSE Sensex","Gold","S&P 500","Bitcoin","USD/INR"]
            _tape_presets = {
                "India":["Nifty 50","BSE Sensex","Nifty Bank","Nifty IT","Nifty Midcap"],
                "Global":["S&P 500","Nasdaq 100","Dow Jones","FTSE 100","DAX","Nikkei 225","Hang Seng"],
                "Macro":["Gold","Silver","Crude Oil WTI","Brent Crude","Natural Gas","USD/INR","VIX"],
                "Crypto":["Bitcoin","Ethereum","Gold","USD/JPY"],
            }
            if not is_noob and "tape_indices" not in st.session_state:
                st.session_state.tape_indices = _default_indices
            st.caption("Pick your own strip or load a preset.")
            p1,p2,p3,p4,p5 = st.columns(5)
            with p1:
                if st.button("India",  key="tape_preset_india",  use_container_width=True): st.session_state.tape_indices = _tape_presets["India"];  st.rerun()
            with p2:
                if st.button("Global", key="tape_preset_global", use_container_width=True): st.session_state.tape_indices = _tape_presets["Global"]; st.rerun()
            with p3:
                if st.button("Macro",  key="tape_preset_macro",  use_container_width=True): st.session_state.tape_indices = _tape_presets["Macro"];  st.rerun()
            with p4:
                if st.button("Crypto", key="tape_preset_crypto", use_container_width=True): st.session_state.tape_indices = _tape_presets["Crypto"]; st.rerun()
            with p5:
                if st.button("Clear",  key="tape_preset_clear",  use_container_width=True): st.session_state.tape_indices = []; st.rerun()
            if st.button("↻ Refresh Live Tape", key="tape_refresh_btn", use_container_width=True):
                cached_fetch_tape_quotes.clear()
                st.rerun()
            chosen = st.multiselect("Choose indices for the tape", options=list(_ALL_INDICES.keys()),
                default=st.session_state.tape_indices, key="tape_multiselect", label_visibility="collapsed")
            st.session_state.tape_indices = chosen
            st.markdown("---")
            render_perf_status_panel()

        st.markdown("---")
        st.markdown(f'<div style="font-size:0.65rem;color:#263a56;text-align:center;margin-top:1rem">MPT · CVXPY · SciPy<br>Amatra Sen — QuanSen v{APP_VERSION}</div>', unsafe_allow_html=True)




# ══════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════
# ── Hero — neat mode only ──────────────────────────────────────
if not is_noob:
    import datetime as _hero_dt
    _now_str = _hero_dt.datetime.now().strftime("%A, %d %B %Y  ·  %H:%M:%S")
    st.markdown(f'''
<div class="hero-banner" style="display:flex;align-items:center;gap:2rem;padding:1.4rem 2rem;">
    <img src="data:image/png;base64,{_LOGO_B64}"
         style="width:82px;height:82px;border-radius:16px;flex-shrink:0;
                box-shadow:0 0 24px rgba(0,180,255,0.45);
                animation:logoPulse 3s ease-in-out infinite;">
    <div style="flex:1">
        <div class="hero-title" style="margin-bottom:0.15rem">
            QUAN<span>SEN</span>
        </div>
        <div class="hero-subtitle">Quantitative Portfolio Optimizer · MPT Engine</div>
        <div class="hero-badge">Ultimate Risk Management Tool</div>
    </div>
    <div style="text-align:right;flex-shrink:0">
        <div style="font-size:0.7rem;color:#4a6a90;letter-spacing:0.1em;
                    text-transform:uppercase;margin-bottom:2px">Session time</div>
        <div id="qs-clock" style="font-family:'DM Mono',monospace;font-size:0.85rem;
                    color:#00b4ff;letter-spacing:0.05em">{_now_str}</div>
    </div>
</div>
<script>
(function() {{
    function pad(n){{ return n<10?'0'+n:n; }}
    function tick(){{
        var d=new Date();
        var days=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
        var months=['January','February','March','April','May','June',
                    'July','August','September','October','November','December'];
        var s=days[d.getDay()]+', '+pad(d.getDate())+' '+months[d.getMonth()]+' '+d.getFullYear()
             +'  ·  '+pad(d.getHours())+':'+pad(d.getMinutes())+':'+pad(d.getSeconds());
        var el=document.getElementById('qs-clock');
        if(el) el.textContent=s;
    }}
    tick(); setInterval(tick,1000);
}})();
</script>
''', unsafe_allow_html=True)

show_flash_notice()

if not is_noob:
    render_hero_action_hub()
    render_session_strip()

# ── Live Ticker Tape — neat mode only ────────────────────────
if not is_noob:
    render_ticker_tape(
        st.session_state.get("tape_indices", []),
        {
            "Nifty 50":"^NSEI","BSE Sensex":"^BSESN","Nifty Bank":"^NSEBANK",
            "Nifty IT":"^CNXIT","Nifty Midcap":"NIFTY_MIDCAP_100.NS",
            "S&P 500":"^GSPC","Nasdaq 100":"^NDX","Dow Jones":"^DJI",
            "Russell 2000":"^RUT","VIX":"^VIX","FTSE 100":"^FTSE",
            "DAX":"^GDAXI","Nikkei 225":"^N225","Hang Seng":"^HSI",
            "Shanghai Comp.":"000001.SS","Gold":"GC=F","Silver":"SI=F",
            "Crude Oil WTI":"CL=F","Brent Crude":"BZ=F","Natural Gas":"NG=F",
            "USD/INR":"INR=X","EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X",
            "USD/JPY":"USDJPY=X","Bitcoin":"BTC-USD","Ethereum":"ETH-USD",
        }
    )


# ══════════════════════════════════════════════════════════════
# CARD-BASED NAVIGATION
# ══════════════════════════════════════════════════════════════

# CSS for the card nav
st.markdown('''<style>
.qs-nav-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 10px;
    margin: 0.8rem 0 1.4rem;
}
.qs-nav-card {
    background: rgba(10,18,32,0.7);
    border: 1px solid #1a2d4d;
    border-radius: 12px;
    padding: 0.75rem 0.5rem 0.65rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.18s;
    text-decoration: none;
    display: block;
}
.qs-nav-card:hover {
    border-color: #00b4ff;
    background: rgba(0,180,255,0.08);
    transform: translateY(-2px);
    box-shadow: 0 4px 18px rgba(0,180,255,0.15);
}
.qs-nav-card.active {
    border-color: #00b4ff;
    background: rgba(0,180,255,0.12);
    box-shadow: 0 0 0 1px #00b4ff33, 0 4px 20px rgba(0,180,255,0.2);
}
.qs-nav-card .nav-icon { font-size: 1.5rem; margin-bottom: 4px; }
.qs-nav-card .nav-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a6a90;
}
.qs-nav-card.active .nav-label { color: #00b4ff; }
.qs-nav-card .nav-status {
    font-size: 0.58rem;
    color: #263a56;
    margin-top: 2px;
}
.qs-nav-card.active .nav-status { color: #4a90d9; }
/* Noob green theme overrides */
.nb-nav-card {
    background: rgba(4,20,10,0.8);
    border: 1px solid #1a5c2a;
    border-radius: 12px;
    padding: 0.75rem 0.5rem 0.65rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.18s;
    display: block;
}
.nb-nav-card:hover {
    border-color: #22c55e;
    background: rgba(34,197,94,0.1);
    transform: translateY(-2px);
}
.nb-nav-card.active {
    border-color: #4afa7a;
    background: rgba(74,250,122,0.1);
    box-shadow: 0 0 0 1px #4afa7a33, 0 4px 20px rgba(74,250,122,0.15);
}
.nb-nav-card .nav-label { color: #3db85a; }
.nb-nav-card.active .nav-label { color: #4afa7a; }
/* nav buttons styled below via scoped class */
.qs-nav-grid div[data-testid="stButton"] > button {
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.45rem 0.1rem !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    height: 36px !important;
    line-height: 1 !important;
    transition: all 0.15s !important;
    width: 100% !important;
}
</style>''', unsafe_allow_html=True)

# ── Section definitions ───────────────────────────────────────
if is_noob:
    _QS_SECTIONS = [
        ("build",      "🏠", "Home",     lambda: len(st.session_state.tickers) > 0),
        ("data",       "📊", "Stocks",   lambda: st.session_state.get("data_loaded", False)),
        ("portfolios", "🎰", "Mix",      lambda: st.session_state.get("portfolios_computed", False)),
        ("frontier",   "📈", "Frontier", lambda: st.session_state.get("frontier_computed", False)),
        ("analysis",   "🔍", "Analysis", lambda: st.session_state.get("data_loaded", False)),
        ("backtest",   "💰", "Backtest", lambda: st.session_state.get("portfolios_computed", False)),
        ("mc",         "🎲", "Simulate", lambda: st.session_state.get("portfolios_computed", False)),
        ("tracker",    "📡", "Tracker",  lambda: bool(st.session_state.get("lpt_holdings"))),
        ("export",     "💾", "Export",   lambda: st.session_state.get("data_loaded", False)),
    ]
else:
    _QS_SECTIONS = [
        ("build",      "📋", "Build",     lambda: len(st.session_state.tickers) > 0),
        ("data",       "📊", "Data",      lambda: st.session_state.get("data_loaded", False)),
        ("portfolios", "🏆", "Portfolio", lambda: st.session_state.get("portfolios_computed", False)),
        ("frontier",   "📈", "Frontier",  lambda: st.session_state.get("frontier_computed", False)),
        ("analysis",   "🔬", "Analysis",  lambda: st.session_state.get("data_loaded", False)),
        ("backtest",   "🧪", "Backtest",  lambda: st.session_state.get("portfolios_computed", False)),
        ("mc",         "🎲", "MonteCarlo",lambda: st.session_state.get("portfolios_computed", False)),
        ("tracker",    "📡", "Tracker",   lambda: bool(st.session_state.get("lpt_holdings"))),
        ("export",     "💾", "Export",    lambda: st.session_state.get("data_loaded", False)),
    ]

# ── Render card grid ──────────────────────────────────────────
_qs_active = st.session_state.qs_section

# Wrap nav in a div so scoped CSS can target it
st.markdown('<div class="qs-nav-grid">', unsafe_allow_html=True)
_nav_cols = st.columns(len(_QS_SECTIONS))
for _ci, (_sec_id, _sec_icon, _sec_label, _sec_ready) in enumerate(_QS_SECTIONS):
    with _nav_cols[_ci]:
        _is_active = (_sec_id == _qs_active)
        _is_ready  = _sec_ready()
        _btn_label = "{} {}".format(_sec_icon, _sec_label)
        if st.button(
            _btn_label,
            key="nav_btn_{}".format(_sec_id),
            use_container_width=True,
            type="primary" if _is_active else "secondary",
            help="{} · {}".format(_sec_label, "ready ✓" if _is_ready else "not ready yet"),
        ):
            st.session_state.qs_section = _sec_id
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<hr style="border-color:#1a2d4d;margin:0.4rem 0 1.2rem;">', unsafe_allow_html=True)

# ── Section router — replaces tab contexts ────────────────────
_qs_s = st.session_state.qs_section

# Section content below — routed by _qs_s



# ══════════════════════════════════════════════════════════════
# TAB 1 — BUILD PORTFOLIO  /  DASHBOARD (Noob)
# ══════════════════════════════════════════════════════════════
if _qs_s == "build":
    if is_noob:
        # ── Noob handlers (must fire before any UI renders) ──────
        _noob_run_all = consume_action("run_all")
        _noob_load    = consume_action("load_data")

        if _noob_load and len(st.session_state.tickers) >= 2:
            with st.spinner("📡 Connecting to markets…"):
                try:
                    _nb_alpha_r, _nb_beta_r, _nb_regime_m = _get_regime_state(force=True)
                    from engines.data_and_compute import load_data, cached_asset_risks
                    (_nb_tickers_out, _nb_returns, _nb_expected_returns,
                     _nb_cov_matrix, _nb_raw_er, _nb_bm_sym) = load_data(
                        tuple(st.session_state.tickers),
                        st.session_state.start_date,
                        st.session_state.end_date,
                        alpha=_nb_alpha_r,
                    )
                    st.session_state.tickers           = _nb_tickers_out
                    st.session_state.returns           = _nb_returns
                    st.session_state.expected_returns  = _nb_expected_returns
                    st.session_state.cov_matrix        = _nb_cov_matrix
                    st.session_state.raw_er            = _nb_raw_er
                    st.session_state.bm_sym            = _nb_bm_sym
                    st.session_state.shrinkage_alpha   = _nb_alpha_r
                    st.session_state.momentum_beta     = _nb_beta_r
                    st.session_state.alpha_regime      = _nb_alpha_r
                    st.session_state.beta_regime       = _nb_beta_r
                    _persist_regime_state(_nb_regime_m)
                    st.session_state.asset_returns     = _nb_expected_returns * 252
                    st.session_state.asset_risks       = cached_asset_risks(_nb_returns)
                    st.session_state.data_loaded       = True
                    st.session_state.price_data        = (1 + _nb_returns).cumprod()
                    st.session_state.portfolios_computed = False
                    st.session_state.frontier_computed   = False
                    st.session_state.momentum_scores   = None
                    st.session_state.momentum_signals  = None
                    st.session_state.momentum_final_er = None
                    st.session_state.momentum_meta     = None
                    # Store regime info for persistent display in the dashboard
                    st.session_state.noob_regime_meta  = _nb_regime_m
                    _commit_regime_snapshot(source="noob-load")
                    _dom = _nb_regime_m.get("dominant_regime", "unknown")
                    st.success(f"✅ Loaded {len(_nb_tickers_out)} stocks · regime: **{_dom}** · α={_nb_alpha_r:.0%}")
                except Exception as _nb_le:
                    st.error(f"Data load failed: {_nb_le}")

        if _noob_run_all and st.session_state.data_loaded:
            _nb_progress = st.progress(0, text="✨ Making magic happen…")
            try:
                _nb_er = optimizer_expected_returns()
                _nb_progress.progress(10, "Computing momentum…")
                try:
                    _nb_scores, _nb_signals, _nb_final_er, _nb_meta = cached_compute_momentum(
                        tuple(st.session_state.tickers),
                        st.session_state.start_date, st.session_state.end_date,
                        st.session_state.expected_returns,
                        lookback=st.session_state.get("momentum_lookback", 252),
                        skip=21,
                        beta=st.session_state.get("momentum_beta", 0.6),
                        auto_lookback=True,
                        prev_regime_probs=st.session_state.get("regime_prev_probs"),
                    )
                    st.session_state.momentum_enabled  = True
                    st.session_state.momentum_scores   = _nb_scores
                    st.session_state.momentum_signals  = _nb_signals
                    st.session_state.momentum_final_er = _nb_final_er
                    st.session_state.momentum_meta     = _nb_meta
                    _persist_regime_state(_nb_meta.get("regime_meta"))
                    st.session_state.noob_regime_meta  = _nb_meta.get("regime_meta") or st.session_state.get("noob_regime_meta")
                    _commit_regime_snapshot(
                        lookback=_nb_meta.get("lookback"),
                        source="noob-momentum",
                    )
                    st.session_state.asset_returns     = _nb_final_er.reindex(st.session_state.tickers) * 252
                    _nb_er = _nb_final_er.reindex(st.session_state.tickers)
                except Exception:
                    pass
                _nb_progress.progress(25, "Building utility portfolio…")
                _nb_w_u = cached_utility_portfolio(
                    _nb_er,
                    st.session_state.cov_matrix,
                    tuple(st.session_state.tickers),
                    st.session_state.min_w,
                    st.session_state.max_w,
                    **_optimizer_tactical_controls(
                        previous_weights=st.session_state.get("weights_utility"),
                        enabled=True,
                    ),
                )
                st.session_state.weights_utility = _nb_w_u
                _nb_progress.progress(50, "Finding best Sharpe mix…")
                _nb_w_t, _nb_tr, _nb_tk, _nb_ts = cached_tangency_portfolio(
                    _nb_er,
                    st.session_state.cov_matrix,
                    tuple(st.session_state.tickers),
                    st.session_state.min_w,
                    st.session_state.max_w,
                    **_optimizer_tactical_controls(
                        previous_weights=st.session_state.get("weights_tan"),
                        enabled=True,
                    ),
                )
                st.session_state.weights_tan   = _nb_w_t
                st.session_state.tan_return    = _nb_tr
                st.session_state.tan_risk      = _nb_tk
                st.session_state.tan_sharpe    = _nb_ts
                _nb_progress.progress(70, "Drawing the frontier…")
                _nb_fr, _nb_ret = cached_compute_frontier(_nb_er, st.session_state.cov_matrix, st.session_state.min_w, st.session_state.max_w)
                st.session_state.frontier_risks    = _nb_fr
                st.session_state.frontier_returns  = _nb_ret
                st.session_state.frontier_computed = True
                _nb_progress.progress(90, "Minimising risk…")
                _nb_w_m = cached_min_risk_portfolio(
                    _nb_er,
                    st.session_state.cov_matrix,
                    tuple(st.session_state.tickers),
                    st.session_state.target_return,
                    st.session_state.min_w,
                    st.session_state.max_w,
                    **_optimizer_tactical_controls(
                        previous_weights=st.session_state.get("weights_min"),
                        enabled=True,
                    ),
                )
                st.session_state.weights_min      = _nb_w_m
                st.session_state.portfolios_computed = True
                _nb_progress.progress(100, "Done ✔")
                st.success("🎉 All done! Head to **🎰 The Mix** tab to see your portfolios.")
            except Exception as _nb_e:
                st.error(f"Something went wrong: {_nb_e}")

        # ── Green dashboard header ───────────────────────────────
        import datetime as _nb_now_dt
        _nb_now_str = _nb_now_dt.datetime.now().strftime("%A %d %B  ·  %H:%M")
        _nb_n_t    = len(st.session_state.tickers)
        _nb_loaded = st.session_state.get("data_loaded", False)
        _nb_ran    = st.session_state.get("portfolios_computed", False)

        st.markdown(
            f'''<div style="background:linear-gradient(135deg,#031a0e,#0a2d14);
                border:1px solid #1a5c2a;border-radius:16px;padding:1.2rem 1.6rem;
                margin-bottom:1.2rem;box-shadow:0 4px 24px rgba(0,180,60,0.12);">
                <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.8rem;">
                    <div>
                        <div style="font-size:0.6rem;letter-spacing:0.22em;text-transform:uppercase;color:#3db85a;margin-bottom:4px">
                            QuanSen · Easy Mode
                        </div>
                        <div style="font-size:1.45rem;font-weight:800;color:#4afa7a;font-family:Syne,sans-serif;letter-spacing:0.04em;">
                            Portfolio Dashboard
                        </div>
                        <div style="font-size:0.72rem;color:#2e7d52;margin-top:2px">{_nb_now_str}</div>
                    </div>
                    <div style="display:flex;gap:1rem;flex-wrap:wrap;">
                        <div style="text-align:center;padding:0.5rem 1rem;background:rgba(0,200,80,0.08);
                                    border:1px solid #1a5c2a;border-radius:10px;min-width:70px;">
                            <div style="font-size:1.5rem;font-weight:800;color:#4afa7a;font-family:'DM Mono',monospace">{_nb_n_t}</div>
                            <div style="font-size:0.58rem;color:#3db85a;text-transform:uppercase;letter-spacing:0.12em">Stocks</div>
                        </div>
                        <div style="text-align:center;padding:0.5rem 1rem;background:rgba(0,200,80,0.08);
                                    border:1px solid {"#22c55e" if _nb_loaded else "#1a5c2a"};border-radius:10px;min-width:70px;">
                            <div style="font-size:1.1rem;font-weight:800;color:{"#4afa7a" if _nb_loaded else "#1a5c2a"}">{"✓" if _nb_loaded else "○"}</div>
                            <div style="font-size:0.58rem;color:#3db85a;text-transform:uppercase;letter-spacing:0.12em">Data</div>
                        </div>
                        <div style="text-align:center;padding:0.5rem 1rem;background:rgba(0,200,80,0.08);
                                    border:1px solid {"#22c55e" if _nb_ran else "#1a5c2a"};border-radius:10px;min-width:70px;">
                            <div style="font-size:1.1rem;font-weight:800;color:{"#4afa7a" if _nb_ran else "#1a5c2a"}">{"✓" if _nb_ran else "○"}</div>
                            <div style="font-size:0.58rem;color:#3db85a;text-transform:uppercase;letter-spacing:0.12em">Results</div>
                        </div>
                    </div>
                </div>
            </div>''',
            unsafe_allow_html=True
        )

        # ── Step progress bar ────────────────────────────────────
        _steps = [
            ("Add stocks",  _nb_n_t >= 2),
            ("Load data",   _nb_loaded),
            ("Run Magic!",  _nb_ran),
            ("See results", _nb_ran),
        ]
        st.markdown(
            '<div style="display:flex;gap:0.5rem;margin-bottom:1.4rem;align-items:center;flex-wrap:wrap;">' +
            "".join(
                f'<div style="display:flex;align-items:center;gap:0.4rem;">'
                f'<div style="width:26px;height:26px;border-radius:50%;'
                f'background:{"linear-gradient(135deg,#0d6b2a,#12882f)" if done else "#0a1f0e"};'
                f'border:1px solid {"#22c55e" if done else "#1a3a1e"};'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-size:0.65rem;font-weight:700;color:{"#c8ffd4" if done else "#1a5c2a"}">'
                f'{"✓" if done else str(i+1)}</div>'
                f'<span style="font-size:0.75rem;color:{"#7dd89a" if done else "#1e6b30"}">{label}</span>'
                f'{"<span style=\"color:#1a5c2a;margin:0 2px\">›</span>" if i < len(_steps)-1 else ""}'
                f'</div>'
                for i, (label, done) in enumerate(_steps)
            ) +
            '</div>',
            unsafe_allow_html=True
        )

        # ── Two-column layout: search + basket ───────────────────
        _nb_col_search, _nb_col_basket = st.columns([1.1, 1], gap="medium")

        with _nb_col_search:
            st.markdown('<div class="nb-dash-card">', unsafe_allow_html=True)
            st.markdown('<span class="nb-green-status-dot"></span><span style="font-size:0.7rem;color:#3db85a;text-transform:uppercase;letter-spacing:0.15em;font-weight:700">Search & Add Stocks</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            _nb_q = st.text_input(
                "Company name or ticker",
                placeholder="e.g. Reliance Industries  or  HDFCBANK.NS",
                key="nb_tab_search_q",
                label_visibility="collapsed",
            )
            _nb_col_s1, _nb_col_s2 = st.columns([2, 1])
            with _nb_col_s1:
                _nb_do_search = st.button("🔍 Search", use_container_width=True, key="nb_tab_do_search")
            with _nb_col_s2:
                if st.button("🎲 Load 20", use_container_width=True, key="nb_tab_load20"):
                    st.session_state.tickers = [
                        'BEL.NS','VEDL.NS','BAJFINANCE.NS','BEML.NS','ADVAIT.BO',
                        'ADANIENT.NS','COALINDIA.NS','CROMPTON.NS','KINGFA.NS','KRISHNADEF.NS',
                        'LT.NS','LUPIN.NS','MAZDOCK.NS','PENIND.NS','PNB.NS',
                        'RELIANCE.NS','SHUKRAPHAR.BO','TARIL.NS','HDFCBANK.NS','SBIN.NS'
                    ]
                    st.session_state.start_date = _demo_start_date
                    st.session_state.end_date   = _demo_end_date
                    reset_portfolio_outputs()
                    st.rerun()

            if _nb_do_search and _nb_q.strip():
                with st.spinner("Searching…"):
                    _nb_results = search_ticker_api(_nb_q.strip())
                    st.session_state.nb_search_results = _nb_results

            if st.session_state.get("nb_search_results"):
                for _nb_r in st.session_state.nb_search_results[:6]:
                    _nb_sym  = _nb_r.get("symbol", "")
                    _nb_name = _nb_r.get("shortname") or _nb_r.get("longname") or _nb_sym
                    _nb_exch = _nb_r.get("exchDisp", "")
                    if not _nb_sym:
                        continue
                    _nb_rc1, _nb_rc2 = st.columns([4, 1])
                    with _nb_rc1:
                        st.markdown(
                            f'<div style="padding:4px 0">'
                            f'<span style="color:#c8ffd4;font-size:0.8rem">{html.escape(_nb_name[:38])}</span><br>'
                            f'<span class="nb-stock-chip">{_nb_sym}</span>'
                            f'<span style="font-size:0.62rem;color:#3db85a;margin-left:5px">{_nb_exch}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    with _nb_rc2:
                        if st.button("＋", key=f"nb_add_{_nb_sym}", use_container_width=True):
                            if _nb_sym not in st.session_state.tickers:
                                with st.spinner("Resolving…"):
                                    _nb_best = get_best_ticker(_nb_sym, st.session_state.start_date, st.session_state.end_date)
                                st.session_state.tickers.append(_nb_best)
                                reset_portfolio_outputs()
                                st.session_state.nb_search_results = []
                                st.rerun()
                            else:
                                st.warning("Already added.")

            st.markdown('<div class="nb-green-heading">Or type a ticker directly</div>', unsafe_allow_html=True)
            _nb_direct_c1, _nb_direct_c2 = st.columns([3, 1])
            with _nb_direct_c1:
                _nb_direct = st.text_input("Direct ticker", placeholder="SBIN.NS", key="nb_direct_ticker", label_visibility="collapsed")
            with _nb_direct_c2:
                if st.button("Add", key="nb_add_direct", use_container_width=True):
                    _t = _nb_direct.strip().upper()
                    if _t and _t not in st.session_state.tickers:
                        with st.spinner(f"Adding {_t}…"):
                            _nb_best2 = get_best_ticker(_t, st.session_state.start_date, st.session_state.end_date)
                        st.session_state.tickers.append(_nb_best2)
                        reset_portfolio_outputs()
                        st.rerun()

        with _nb_col_basket:
            st.markdown('<div class="nb-dash-card">', unsafe_allow_html=True)
            st.markdown('<span class="nb-green-status-dot"></span><span style="font-size:0.7rem;color:#3db85a;text-transform:uppercase;letter-spacing:0.15em;font-weight:700">Your Basket</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.session_state.tickers:
                for _nb_ti, _nb_tt in enumerate(st.session_state.tickers):
                    _nb_bc1, _nb_bc2 = st.columns([5, 1])
                    with _nb_bc1:
                        st.markdown(f'<span class="nb-stock-chip">🟢 {_nb_tt}</span>', unsafe_allow_html=True)
                    with _nb_bc2:
                        if st.button("✕", key=f"nb_tab_rm_{_nb_tt}_{_nb_ti}"):
                            st.session_state.tickers.remove(_nb_tt)
                            reset_portfolio_outputs()
                            st.rerun()
                st.markdown("---")
                if st.button("🗑 Clear All Stocks", use_container_width=True, key="nb_tab_clear"):
                    st.session_state.tickers = []
                    reset_portfolio_outputs()
                    st.rerun()
            else:
                st.markdown(
                    '<div style="padding:1.5rem;text-align:center;color:#1a5c2a;font-size:0.82rem;">'
                    '🌱 No stocks yet.<br>Search above or hit <b>Load 20</b> to get started!'
                    '</div>',
                    unsafe_allow_html=True
                )

        # ── Action row ───────────────────────────────────────────
        st.markdown('<div class="nb-green-heading">⚡ Actions</div>', unsafe_allow_html=True)
        _nb_act1, _nb_act2 = st.columns(2)
        with _nb_act1:
            if st.button(
                "📡 Load Market Data",
                use_container_width=True,
                key="nb_tab_load_btn",
                disabled=len(st.session_state.tickers) < 2,
            ):
                request_action("load_data")
                st.rerun()
        with _nb_act2:
            if st.button(
                "🚀 Run Magic!",
                use_container_width=True,
                key="nb_tab_run_btn",
                disabled=not st.session_state.data_loaded,
            ):
                request_action("run_all")
                st.rerun()

        if len(st.session_state.tickers) < 2:
            st.caption("⬆ Add at least 2 stocks, then hit **Load Market Data**.")
        elif not _nb_loaded:
            st.caption("👆 Hit **Load Market Data** to analyse your stocks.")
        elif not _nb_ran:
            st.caption("👆 Hit **Run Magic!** to generate your portfolios.")
        else:
            st.success("✅ All done! Go to **🎰 The Mix** tab to see your results.")

        # ── Quick stats dashboard (shown after data loaded) ──────
        if _nb_loaded and st.session_state.get("asset_risks") is not None:
            st.markdown('<div class="nb-green-heading">📊 Quick Stats</div>', unsafe_allow_html=True)
            _nb_tickers_for_dash = st.session_state.tickers
            _nb_ar = st.session_state.asset_returns
            _nb_risks = st.session_state.asset_risks

            _nb_best_ret_idx = int(np.argmax(_nb_ar.values))
            _nb_lowest_risk_idx = int(np.argmin(_nb_risks.values))

            _nb_dc1, _nb_dc2, _nb_dc3, _nb_dc4 = st.columns(4)
            with _nb_dc1:
                st.markdown(
                    f'<div class="nb-dash-card">'
                    f'<div class="nb-dash-card-icon">📈</div>'
                    f'<div class="nb-dash-card-label">Stocks analysed</div>'
                    f'<div class="nb-dash-card-value">{len(_nb_tickers_for_dash)}</div>'
                    f'<div class="nb-dash-card-sub">{st.session_state.start_date} → {st.session_state.end_date}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with _nb_dc2:
                st.markdown(
                    f'<div class="nb-dash-card">'
                    f'<div class="nb-dash-card-icon">🏆</div>'
                    f'<div class="nb-dash-card-label">Best returner</div>'
                    f'<div class="nb-dash-card-value" style="font-size:1.1rem">{_nb_tickers_for_dash[_nb_best_ret_idx]}</div>'
                    f'<div class="nb-dash-card-sub">{_nb_ar.values[_nb_best_ret_idx]*100:.1f}% / yr</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with _nb_dc3:
                st.markdown(
                    f'<div class="nb-dash-card">'
                    f'<div class="nb-dash-card-icon">🛡</div>'
                    f'<div class="nb-dash-card-label">Lowest risk</div>'
                    f'<div class="nb-dash-card-value" style="font-size:1.1rem">{_nb_tickers_for_dash[_nb_lowest_risk_idx]}</div>'
                    f'<div class="nb-dash-card-sub">{_nb_risks.values[_nb_lowest_risk_idx]*100:.1f}% vol</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with _nb_dc4:
                _regime_label_map = {"bull":"🟢 Bull","sideways":"🟡 Sideways","bear":"🔴 Bear","crisis":"🔴 Crisis"}
                _dom_r = st.session_state.get("bm_sym", "?")
                st.markdown(
                    f'<div class="nb-dash-card">'
                    f'<div class="nb-dash-card-icon">🌐</div>'
                    f'<div class="nb-dash-card-label">Benchmark</div>'
                    f'<div class="nb-dash-card-value" style="font-size:1.1rem">{_dom_r}</div>'
                    f'<div class="nb-dash-card-sub">α = {st.session_state.shrinkage_alpha:.0%}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # ── Market regime panel ──────────────────────────────────
            # Shows what the market state engine detected so the user
            # knows the engine ran and what parameters it chose.
            _nb_regime_display = st.session_state.get("noob_regime_meta")
            if _nb_regime_display:
                _nb_probs   = _nb_regime_display.get("regime_probabilities", {})
                _nb_dom     = _nb_regime_display.get("dominant_regime", "unknown")
                _nb_alpha_d = st.session_state.get("shrinkage_alpha", 0.7)
                _nb_beta_d  = st.session_state.get("momentum_beta", 0.6)

                _regime_colors = {
                    "bull":     ("#22c55e", "🟢", "Bull Market"),
                    "sideways": ("#ffd166", "🟡", "Sideways"),
                    "bear":     ("#ff4d6d", "🔴", "Bear Market"),
                    "crisis":   ("#9b5de5", "🔴", "Crisis"),
                }
                _nb_rc, _nb_emoji, _nb_label = _regime_colors.get(
                    _nb_dom, ("#7dd89a", "⚪", _nb_dom.capitalize())
                )

                # Plain-English alpha explanation
                if _nb_alpha_d >= 0.85:
                    _alpha_plain = "trusting your stocks' own history heavily"
                elif _nb_alpha_d >= 0.65:
                    _alpha_plain = "balancing stock history with the market benchmark"
                else:
                    _alpha_plain = "leaning on the market benchmark for safety"

                st.markdown('<div class="nb-green-heading">🧠 Market Brain — Auto Mode</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'''<div style="background:linear-gradient(135deg,#031a0e,#042214);
                                   border:1px solid #1a5c2a;border-radius:14px;
                                   padding:1rem 1.2rem;margin-bottom:0.8rem;">
                        <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
                            <div style="flex:1;min-width:180px;">
                                <div style="font-size:0.6rem;letter-spacing:0.16em;text-transform:uppercase;
                                            color:#3db85a;margin-bottom:4px">Detected Market Regime</div>
                                <div style="font-size:1.4rem;font-weight:800;color:{_nb_rc}">
                                    {_nb_emoji} {_nb_label}
                                </div>
                                <div style="font-size:0.72rem;color:#3db85a;margin-top:4px">
                                    The engine analysed trend, volatility, drawdown, basket breadth and correlation stress
                                </div>
                            </div>
                            <div style="display:flex;gap:0.8rem;flex-wrap:wrap;">
                                <div style="text-align:center;padding:0.5rem 0.9rem;
                                            background:rgba(74,250,122,0.07);
                                            border:1px solid #1a5c2a;border-radius:10px;">
                                    <div style="font-size:1.2rem;font-weight:800;color:#4afa7a;
                                                font-family:\'DM Mono\',monospace">{_nb_alpha_d:.0%}</div>
                                    <div style="font-size:0.58rem;color:#3db85a;text-transform:uppercase;
                                                letter-spacing:0.1em">Stock trust (α)</div>
                                </div>
                                <div style="text-align:center;padding:0.5rem 0.9rem;
                                            background:rgba(74,250,122,0.07);
                                            border:1px solid #1a5c2a;border-radius:10px;">
                                    <div style="font-size:1.2rem;font-weight:800;color:#4afa7a;
                                                font-family:\'DM Mono\',monospace">{_nb_beta_d:.0%}</div>
                                    <div style="font-size:0.58rem;color:#3db85a;text-transform:uppercase;
                                                letter-spacing:0.1em">History weight (β)</div>
                                </div>
                            </div>
                        </div>
                        <!-- Probability bars -->
                        <div style="margin-top:0.9rem;display:grid;
                                    grid-template-columns:repeat(4,1fr);gap:0.5rem;">''',
                    unsafe_allow_html=True
                )

                _bar_colors = {"bull":"#22c55e","sideways":"#ffd166","bear":"#ff4d6d","crisis":"#9b5de5"}
                _bars_html = ""
                for _rname in ["bull","sideways","bear","crisis"]:
                    _rp = _nb_probs.get(_rname, 0.0)
                    _bc = _bar_colors[_rname]
                    _bold = "font-weight:800;" if _rname == _nb_dom else ""
                    _bars_html += (
                        f'<div style="text-align:center;">'
                        f'<div style="font-size:0.62rem;color:#3db85a;text-transform:uppercase;'
                        f'letter-spacing:0.08em;{_bold}">{_rname}</div>'
                        f'<div style="background:#041209;border-radius:6px;height:6px;'
                        f'overflow:hidden;margin:3px 0;">'
                        f'<div style="width:{_rp*100:.1f}%;height:100%;background:{_bc};'
                        f'border-radius:6px;"></div></div>'
                        f'<div style="font-size:0.7rem;color:{_bc};{_bold}">{_rp*100:.0f}%</div>'
                        f'</div>'
                    )

                st.markdown(
                    _bars_html +
                    f'</div>'
                    f'<div style="margin-top:0.7rem;font-size:0.75rem;color:#3db85a;">'
                    f'💡 In <strong style="color:#c8ffd4">{_nb_label}</strong> conditions the engine is '
                    f'<strong style="color:#c8ffd4">{_alpha_plain}</strong> '
                    f'(α={_nb_alpha_d:.0%}) and weighting momentum signals at β={_nb_beta_d:.0%}.'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

    else:
        render_build_workflow_overview()
        st.markdown(
            f"""
            <div class="build-action-rail">
                <div class="build-action-card">
                    <div class="workflow-kicker">Quick Flow</div>
                    <div class="headline">Build → Load → Tilt → Run</div>
                    <div class="copy">Use the search desk, lock the common market window, optionally compute momentum, then fire the full optimizer stack.</div>
                </div>
                <div class="build-action-card">
                    <div class="workflow-kicker">Basket</div>
                    <div class="headline">{len(st.session_state.tickers)} symbols</div>
                    <div class="copy">Enough names for a proper optimization set? Aim for at least 5 if you want diversification to show up.</div>
                </div>
                <div class="build-action-card">
                    <div class="workflow-kicker">Data</div>
                    <div class="headline">{'Loaded' if st.session_state.data_loaded else 'Awaiting load'}</div>
                    <div class="copy">The optimizer only trusts the strict common-overlap window after a successful data load.</div>
                </div>
                <div class="build-action-card">
                    <div class="workflow-kicker">Desk</div>
                    <div class="headline">{'Results live' if st.session_state.portfolios_computed or st.session_state.frontier_computed else 'Ready to stage'}</div>
                    <div class="copy">Run `ALL` when you want the quickest comparison across utility, tangency, frontier, and min-risk.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        render_market_weather_panel(st.session_state.get("regime_meta"))
        render_regime_change_panel()

        st.markdown(
            """
            <div class="workflow-section-shell">
                <div class="workflow-section-topline">
                    <div>
                        <div class="workflow-section-kicker">Stage 1</div>
                        <div class="section-heading" style="margin:0.2rem 0 0 0">Build The Universe</div>
                    </div>
                    <div class="workflow-section-copy">Search fast, add manually, then tighten the basket before you load the window.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        col_search, col_manual = st.columns([1.1, 1])

        # ── Ticker search ──────────────────────────────────────────
        with col_search:
            st.markdown('<div class="section-heading">Search & Add Tickers</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="status-box status-info">Desk search: press Enter or click Search to query Yahoo Finance.</div>',
                unsafe_allow_html=True
            )
            with st.form("ticker_search_form", clear_on_submit=False):
                search_query = st.text_input(
                    "Company name",
                    placeholder="e.g. Reliance Industries",
                    key="search_input"
                )
                search_btn = st.form_submit_button("🔍 Search", use_container_width=True)

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
                                reset_portfolio_outputs()
                                st.markdown(f'<div class="status-box status-success">✔ Added {best}</div>', unsafe_allow_html=True)
                                st.rerun()
                            else:
                                st.markdown(f'<div class="status-box status-warn">Already in portfolio.</div>', unsafe_allow_html=True)

        # ── Manual entry + remove ──────────────────────────────────
        with col_manual:
            st.markdown('<div class="section-heading">Manual Entry & Manage</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="status-box status-info">Quick add: broker-style symbols like `SBIN` will be auto-resolved when possible.</div>',
                unsafe_allow_html=True
            )
            with st.form("manual_add_form", clear_on_submit=False):
                manual_ticker = st.text_input(
                    "Enter ticker directly",
                    placeholder="e.g. AAPL, TCS.NS",
                    key="manual_input"
                )
                manual_btn = st.form_submit_button("➕ Add Manually", use_container_width=True)
            if manual_btn:
                t = manual_ticker.strip().upper()
                if t:
                    if t not in st.session_state.tickers:
                        with st.spinner(f"Resolving {t}..."):
                            best = get_best_ticker(t, st.session_state.start_date, st.session_state.end_date)
                        st.session_state.tickers.append(best)
                        reset_portfolio_outputs()
                        st.markdown(f'<div class="status-box status-success">✔ Added {best}</div>', unsafe_allow_html=True)
                        st.rerun()
                    else:
                        st.markdown(f'<div class="status-box status-warn">Already in portfolio.</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="workflow-section-shell">
                <div class="workflow-section-topline">
                    <div>
                        <div class="workflow-section-kicker">Stage 2</div>
                        <div class="section-heading" style="margin:0.2rem 0 0 0">Load And Execute</div>
                    </div>
                    <div class="workflow-section-copy">Freeze the market window, confirm the return model, then fire the optimizer stack.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if len(st.session_state.tickers) < 2:
            st.markdown('<div class="status-box status-warn">⚠ Add at least 2 tickers to load data.</div>', unsafe_allow_html=True)
        else:
            load_requested = (
                st.button("⬇ Download & Load Data", use_container_width=True, key="quant_load_data_btn") or
                consume_action("load_data")
            )
            if load_requested:
                with st.spinner("Downloading price data and computing shrinkage-adjusted returns…"):
                    try:
                    
                        # Detect market regime
                        alpha_regime, beta_regime, regime_meta = _get_regime_state(force=True)

                        st.session_state.beta_regime = beta_regime
                        st.session_state.beta_regime = max(0.25, min(0.9, beta_regime))
                        st.session_state.alpha_regime = alpha_regime
                        st.session_state.shrinkage_alpha = alpha_regime
                        st.session_state.momentum_beta = st.session_state.beta_regime
                        _persist_regime_state(regime_meta)

                        (tickers_out,
                         returns,
                         expected_returns,
                         cov_matrix,
                         raw_er,
                         bm_sym) = load_data(
                            tuple(st.session_state.tickers),
                            st.session_state.start_date,
                            st.session_state.end_date,
                            alpha=alpha_regime,)
                    
                        st.session_state.tickers          = tickers_out
                        st.session_state.returns          = returns
                        st.session_state.expected_returns = expected_returns
                        st.session_state.cov_matrix       = cov_matrix
                        st.session_state.raw_er           = raw_er
                        st.session_state.bm_sym           = bm_sym
                        st.session_state.asset_returns    = expected_returns * 252
                        st.session_state.asset_risks      = cached_asset_risks(returns)
                        st.session_state.data_loaded      = True
                        st.session_state.portfolios_computed = False
                        st.session_state.frontier_computed   = False
                        # price_data: cumulative price index reconstructed from returns
                        # Used by the Noob mode Time Machine feature
                        st.session_state.price_data = (1 + returns).cumprod()
                        # Clear momentum — stale signals don't apply to new data
                        st.session_state.momentum_scores    = None
                        st.session_state.momentum_signals   = None
                        st.session_state.momentum_final_er  = None
                        st.session_state.momentum_meta      = None
                        _commit_regime_snapshot(source="quant-load")

                        alpha = st.session_state.shrinkage_alpha
                        if len(returns) <= 0:
                            raise ValueError(
                                "No usable return history remained after cleaning. "
                                "Try a later start date or reduce the basket."
                            )
                        overlap_note = ""
                        if len(returns.index) > 0:
                            overlap_note = (
                                f" Effective optimizer window: "
                                f"**{returns.index[0].strftime('%Y-%m-%d')}** -> "
                                f"**{returns.index[-1].strftime('%Y-%m-%d')}**."
                            )
                        if bm_sym:
                            message = (
                                f"✔ Loaded {len(returns)} trading days for {len(tickers_out)} assets. "
                                f"Returns shrunk toward **{bm_sym}**  (α={alpha:.0%})."
                                f"{overlap_note}"
                            )
                            level = "success"
                        else:
                            message = (
                                f"✔ Loaded {len(returns)} trading days for {len(tickers_out)} assets. "
                                "Benchmark unavailable — using raw expected returns."
                                f"{overlap_note}"
                            )
                            level = "warning"
                        st.session_state.flash_notice = (level, message)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Data load failed: {e}")
                        import traceback; st.code(traceback.format_exc())

        # ── Step 2: Run optimizations ──────────────────────────────
        # Resolve which expected returns the optimizer should use
        def _active_er():
            return optimizer_expected_returns()

        if st.session_state.data_loaded:
            exec_info_1, exec_info_2, exec_info_3 = st.columns(3)
            with exec_info_1:
                st.markdown(
                    f"""
                    <div class="workflow-metric">
                        <div class="workflow-metric-label">Data State</div>
                        <div class="workflow-metric-value">Loaded</div>
                        <div class="workflow-metric-copy">{len(st.session_state.returns)} trading rows across {len(st.session_state.tickers)} assets</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with exec_info_2:
                active_model = "Momentum blend" if (
                    st.session_state.momentum_enabled and st.session_state.momentum_final_er is not None
                ) else "Shrinkage core"
                st.markdown(
                    f"""
                    <div class="workflow-metric">
                        <div class="workflow-metric-label">Model In Play</div>
                        <div class="workflow-metric-value">{active_model}</div>
                        <div class="workflow-metric-copy">the optimizer is using {active_model.lower()} expected returns</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with exec_info_3:
                momentum_state = (
                    "Ready" if st.session_state.momentum_final_er is not None else
                    "Enabled" if st.session_state.momentum_enabled else
                    "Off"
                )
                st.markdown(
                    f"""
                    <div class="workflow-metric">
                        <div class="workflow-metric-label">Momentum Desk</div>
                        <div class="workflow-metric-value">{momentum_state}</div>
                        <div class="workflow-metric-copy">compute momentum when you want the optimizer to tilt off raw shrinkage</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown('<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.12em;text-transform:uppercase;margin:1rem 0 0.6rem">Optimization Actions</div>', unsafe_allow_html=True)
            run_col1, run_col2, run_col3 = st.columns(3)

            with run_col1:
                if st.button("⚡ Utility Portfolio", use_container_width=True, key="quant_utility_btn"):
                    with st.spinner("Optimizing utility portfolio…"):
                        try:
                            w = cached_utility_portfolio(
                                _active_er(),
                                st.session_state.cov_matrix,
                                tuple(st.session_state.tickers),
                                st.session_state.min_w,
                                st.session_state.max_w,
                                **_optimizer_tactical_controls(
                                    previous_weights=st.session_state.get("weights_utility")
                                ),
                            )
                            st.session_state.weights_utility = w
                            st.success("Utility portfolio computed.")
                        except Exception as e:
                            st.error(f"Error: {e}")

            with run_col2:
                if st.button("🌟 Tangency Portfolio", use_container_width=True, key="quant_tangency_btn"):
                    with st.spinner("Maximising Sharpe ratio…"):
                        try:
                            w, tr, tk, ts = cached_tangency_portfolio(
                                _active_er(),
                                st.session_state.cov_matrix,
                                tuple(st.session_state.tickers),
                                st.session_state.min_w,
                                st.session_state.max_w,
                                **_optimizer_tactical_controls(
                                    previous_weights=st.session_state.get("weights_tan")
                                ),
                            )
                            st.session_state.weights_tan  = w
                            st.session_state.tan_return   = tr
                            st.session_state.tan_risk     = tk
                            st.session_state.tan_sharpe   = ts
                            st.success("Tangency portfolio computed.")
                        except Exception as e:
                            st.error(f"Error: {e}")

            with run_col3:
                if st.button("🎯 Min-Risk Portfolio", use_container_width=True, key="quant_minrisk_btn"):
                    with st.spinner("Minimising risk for target return…"):
                        try:
                            w = cached_min_risk_portfolio(
                                _active_er(),
                                st.session_state.cov_matrix,
                                tuple(st.session_state.tickers),
                                st.session_state.target_return,
                                st.session_state.min_w,
                                st.session_state.max_w,
                                **_optimizer_tactical_controls(
                                    previous_weights=st.session_state.get("weights_min")
                                ),
                            )
                            st.session_state.weights_min = w
                            if w is not None:
                                st.success(f"Min-risk portfolio computed (target: {st.session_state.target_return*100:.1f}%).")
                            else:
                                st.warning("No feasible solution for this target. Try a lower target return.")
                        except Exception as e:
                            st.error(f"Error: {e}")

            run_all = (
                st.button("🚀 Run ALL Optimizations + Frontier", use_container_width=True, key="quant_run_all_btn") or
                consume_action("run_all")
            )
            if run_all:
                progress = st.progress(0, text="Running all optimizations…")
                try:
                    progress.progress(10, "Utility portfolio…")
                    w_u = cached_utility_portfolio(
                        _active_er(), st.session_state.cov_matrix,
                        tuple(st.session_state.tickers), st.session_state.min_w, st.session_state.max_w,
                        **_optimizer_tactical_controls(
                            previous_weights=st.session_state.get("weights_utility")
                        ),
                    )
                    st.session_state.weights_utility = w_u
                    progress.progress(35, "Tangency portfolio…")
                    w_t, tr, tk, ts = cached_tangency_portfolio(
                        _active_er(), st.session_state.cov_matrix,
                        tuple(st.session_state.tickers), st.session_state.min_w, st.session_state.max_w,
                        **_optimizer_tactical_controls(
                            previous_weights=st.session_state.get("weights_tan")
                        ),
                    )
                    st.session_state.weights_tan  = w_t
                    st.session_state.tan_return   = tr
                    st.session_state.tan_risk     = tk
                    st.session_state.tan_sharpe   = ts
                    progress.progress(60, "Efficient frontier…")
                    fr, ret = cached_compute_frontier(
                        _active_er(), st.session_state.cov_matrix,
                        st.session_state.min_w, st.session_state.max_w)
                    st.session_state.frontier_risks    = fr
                    st.session_state.frontier_returns  = ret
                    st.session_state.frontier_computed = True
                    progress.progress(85, "Min-risk portfolio…")
                    w_m = cached_min_risk_portfolio(
                        _active_er(), st.session_state.cov_matrix,
                        tuple(st.session_state.tickers), st.session_state.target_return,
                        st.session_state.min_w, st.session_state.max_w,
                        **_optimizer_tactical_controls(
                            previous_weights=st.session_state.get("weights_min")
                        ),
                    )
                    st.session_state.weights_min = w_m
                    st.session_state.portfolios_computed = True
                    progress.progress(100, "Done ✔")
                    st.success("All optimizations complete. Navigate the tabs above.")
                except Exception as e:
                    st.error(f"Error during run: {e}")

            # ── Frontier separately ────────────────────────────────
            frontier_requested = (
                st.button("📉 Compute Efficient Frontier Only", use_container_width=True, key="quant_frontier_only_btn") or
                consume_action("frontier")
            )
            if frontier_requested:
                with st.spinner("Computing efficient frontier (100 points)…"):
                    try:
                        fr, ret = cached_compute_frontier(
                            _active_er(),
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

        st.markdown(
            """
            <div class="workflow-section-shell">
                <div class="workflow-section-topline">
                    <div>
                        <div class="workflow-section-kicker">Stage 3</div>
                        <div class="section-heading" style="margin:0.2rem 0 0 0">Audit The Basket</div>
                    </div>
                    <div class="workflow-section-copy">Review listing coverage, prune weak names, and sanity-check the sample before deeper analysis.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.session_state.tickers:
            st.markdown(
                '<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem">Ticker Coverage Audit</div>',
                unsafe_allow_html=True
            )
            for i, t in enumerate(st.session_state.tickers):
                first_date, last_date, colour, age = get_ticker_date_range(t)
                age_label = f"{age}d ago" if age < 9999 else "unknown"
                try:
                    import datetime as _dt2
                    chosen_start = _dt2.date.fromisoformat(st.session_state.start_date)
                    ticker_start = _dt2.date.fromisoformat(first_date) if first_date != "N/A" else None
                    start_warn = ticker_start and chosen_start < ticker_start
                except Exception:
                    start_warn = False
                c1, c2 = st.columns([5, 1])
                with c1:
                    warn_html = (
                        f'<span style="font-size:0.65rem;color:#ff5252;margin-left:6px">'
                        f'⚠ data starts {first_date} — your start date is earlier</span>'
                    ) if start_warn else ""
                    st.markdown(
                        f'<span class="ticker-chip">{t}</span>'
                        f'<span style="font-size:0.7rem;color:#4a90d9;margin-left:8px">from </span>'
                        f'<span style="font-size:0.7rem;color:#a0c8e8;">{first_date}</span>'
                        f'<span style="font-size:0.7rem;color:#4a90d9;margin-left:6px"> to </span>'
                        f'<span style="font-size:0.7rem;color:{colour};">{last_date}</span>'
                        f'<span style="font-size:0.65rem;color:#4a6a90;margin-left:5px">({age_label})</span>'
                        + warn_html,
                        unsafe_allow_html=True
                    )
                with c2:
                    if st.button("✕", key=f"rm_{t}_{i}"):
                        st.session_state.tickers.remove(t)
                        reset_portfolio_outputs()
                        st.rerun()
        else:
            st.markdown('<div class="status-box status-info">No tickers yet. Search or add manually.</div>', unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════
# TAB 2 — MARKET DATA  /  MY STOCKS (Noob)
# ══════════════════════════════════════════════════════════════
if _qs_s == "data":
    if is_noob:
        render_noob_tab_my_stocks()
    elif not st.session_state.data_loaded:
        render_insight_note("This tab becomes your diagnostic layer after a data load. It explains what the return engine believes about each asset before the optimizer starts assigning weights.")
    else:
        returns          = st.session_state.returns
        expected_returns = st.session_state.expected_returns
        asset_returns    = st.session_state.asset_returns
        asset_risks      = st.session_state.asset_risks
        tickers          = st.session_state.tickers

        # ── Shrinkage comparison panel ─────────────────────────
        raw_er = st.session_state.raw_er
        bm_sym = st.session_state.bm_sym
        alpha  = st.session_state.shrinkage_alpha

        if raw_er is not None and bm_sym is not None:
            st.markdown('<div class="section-heading">Return Shrinkage Adjustment</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="status-box status-info">' +
                f'Shrinkage  <b>α = {alpha:.0%}</b>  ·  Benchmark anchored to  <b>{bm_sym}</b>.<br>' +
                f'Each stock\'s expected return = {alpha:.0%} × its own mean  +  {1-alpha:.0%} × {bm_sym} mean.' +
                '</div>',
                unsafe_allow_html=True
            )
            shrink_df = pd.DataFrame({
                "Ticker":       tickers,
                "Raw ann. %":   (raw_er.values * 252 * 100).round(2),
                "Adj ann. %":   (expected_returns.values * 252 * 100).round(2),
                "Δ (pp)":       ((expected_returns.values - raw_er.values) * 252 * 100).round(2),
            }).set_index("Ticker")

            def _colour_delta(val):
                if val < -5:   return "color: #ff5252; font-weight:600"
                elif val < 0:  return "color: #ff8a65"
                elif val > 5:  return "color: #69f0ae; font-weight:600"
                elif val > 0:  return "color: #b9f6ca"
                return ""

            st.dataframe(
                shrink_df.style
                    .background_gradient(cmap="Blues",  subset=["Adj ann. %"])
                    .background_gradient(cmap="Oranges", subset=["Raw ann. %"])
                    .applymap(_colour_delta, subset=["Δ (pp)"]),
                use_container_width=True
            )
            st.caption(
                "Δ (pp) = adjusted − raw.  "
                "Negative = return pulled down (outlier tamed).  "
                "Large red values = stocks the raw optimizer would have over-weighted."
            )
            st.markdown("---")
        elif raw_er is not None and bm_sym is None:
            st.markdown(
                '<div class="status-box status-warn">⚠ Benchmark unavailable — raw expected returns used (no shrinkage applied).</div>',
                unsafe_allow_html=True
            )

        # ── Momentum Signal Table ──────────────────────────────
        if (st.session_state.momentum_enabled and
                st.session_state.momentum_signals is not None):

            scores   = st.session_state.momentum_scores
            signals  = st.session_state.momentum_signals
            final_er = st.session_state.momentum_final_er
            meta     = st.session_state.momentum_meta or {}
            beta     = st.session_state.momentum_beta
            lb       = int(meta.get("lookback", st.session_state.momentum_lookback))
            skip_days = int(meta.get("skip", 21))
            lb_mode = "auto" if meta.get("auto_lookback") else "manual"

            st.markdown('<div class="section-heading">Momentum Signals</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="status-box status-info">' +
                f'Lookback <b>{lb} trading days</b> ({lb_mode}) from the selected end date  ·  Skip last <b>{skip_days} days</b>  ·  ' +
                f'β = <b>{beta:.0%}</b>  (history weight)<br>' +
                f'μ_final = {beta:.0%} × μ_shrinkage  +  {1-beta:.0%} × momentum_scaled' +
                '</div>',
                unsafe_allow_html=True
            )
            if meta.get("entry_date") and meta.get("exit_date"):
                st.caption(
                    f"Effective momentum window: {meta['entry_date']} → {meta['exit_date']} "
                    f"(selected sample end date: {meta.get('end_date', st.session_state.end_date)})."
                )
            if meta.get("auto_lookback") and meta.get("regime_meta"):
                _lb_regime = meta["regime_meta"]
                _lb_dom = _lb_regime.get("dominant_regime", "unknown")
                _lb_conf = float(_lb_regime.get("confidence", 0.0))
                _lb_drivers = _lb_regime.get("drivers", {})
                _fast_dom = _lb_regime.get("fast_dominant_regime", _lb_dom)
                _fast_conf = float(_lb_regime.get("fast_confidence", _lb_conf))
                st.caption(
                    f"Auto-lookback driver: {_lb_dom} regime, confidence {_lb_conf:.0%}, "
                    f"short-term pulse {_fast_dom} at {_fast_conf:.0%}, "
                    f"raw weighted lookback {_lb_regime.get('lookback_raw', lb):.1f}d."
                )
                if _lb_drivers:
                    st.markdown(
                        f"""
                        <div class="status-box status-info" style="margin-top:0.45rem;">
                            <b>Why {lb}d?</b><br>
                            Base regime mix:
                            bull {float(_lb_drivers.get('bull_anchor', 0.0)):.1f}d +
                            sideways {float(_lb_drivers.get('sideways_anchor', 0.0)):.1f}d +
                            bear {float(_lb_drivers.get('bear_anchor', 0.0)):.1f}d +
                            crisis {float(_lb_drivers.get('crisis_anchor', 0.0)):.1f}d<br>
                            Stress adjustments:
                            downside -{float(_lb_drivers.get('downside_penalty', 0.0)):.1f}d,
                            correlation -{float(_lb_drivers.get('corr_penalty', 0.0)):.1f}d,
                            participation -{float(_lb_drivers.get('participation_penalty', 0.0)):.1f}d,
                            conviction +{float(_lb_drivers.get('conviction_bonus', 0.0)):.1f}d,
                            confidence +{float(_lb_drivers.get('confidence_bonus', 0.0)):.1f}d.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            if meta.get("horizon_weights"):
                _hz = meta.get("horizon_weights", {})
                _primary = meta.get("primary_horizon")
                _hz_parts = ", ".join(
                    f"{int(h)}d {float(w):.0%}" for h, w in sorted(_hz.items(), key=lambda kv: int(kv[0]))
                )
                st.caption(
                    f"Short-term engine blend: primary horizon {_primary}d  ·  {_hz_parts}."
                )
            if meta.get("tactical_rebalance_ratio") is not None:
                st.caption(
                    f"Tactical rebalance ratio {float(meta.get('tactical_rebalance_ratio', 1.0)):.0%}  ·  "
                    f"turnover penalty {float(meta.get('tactical_turnover_penalty', 0.0)):.3f}."
                )

            def _signal_icon(s):
                return {"Strong": "🟢 Strong", "Neutral": "🟡 Neutral", "Weak": "🔴 Weak"}.get(s, s)

            mom_df = pd.DataFrame({
                "Ticker":           tickers,
                "Momentum %":       (scores.reindex(tickers).values * 100).round(2),
                "Signal":           [_signal_icon(signals.get(t, "Neutral")) for t in tickers],
                "μ_shrinkage ann%": (expected_returns.values * 252 * 100).round(2),
                "μ_final ann%":     (final_er.reindex(tickers).values * 252 * 100).round(2),
                "Δ vs shrinkage":   ((final_er.reindex(tickers).values - expected_returns.values) * 252 * 100).round(2),
            }).set_index("Ticker")

            def _col_momentum(val):
                if val > 15:   return "color:#00e676;font-weight:600"
                elif val > 0:  return "color:#b9f6ca"
                elif val < -15:return "color:#ff5252;font-weight:600"
                elif val < 0:  return "color:#ff8a65"
                return ""

            def _col_delta(val):
                if val > 2:    return "color:#00e676;font-weight:600"
                elif val > 0:  return "color:#b9f6ca"
                elif val < -2: return "color:#ff5252;font-weight:600"
                elif val < 0:  return "color:#ff8a65"
                return ""

            st.dataframe(
                mom_df.style
                    .applymap(_col_momentum, subset=["Momentum %"])
                    .applymap(_col_delta,    subset=["Δ vs shrinkage"])
                    .background_gradient(cmap="RdYlGn", subset=["μ_final ann%"]),
                use_container_width=True
            )

            # Summary counts
            n_strong  = (signals == "Strong").sum()
            n_neutral = (signals == "Neutral").sum()
            n_weak    = (signals == "Weak").sum()
            st.caption(
                f"🟢 Strong: {n_strong}  ·  🟡 Neutral: {n_neutral}  ·  🔴 Weak: {n_weak}  ·  "
                f"Optimizer is using μ_final (momentum-blended) returns."
            )
            st.markdown("---")

        elif st.session_state.momentum_enabled and st.session_state.momentum_signals is None:
            st.markdown(
                '<div class="status-box status-warn">⚠ Momentum enabled but not yet computed — ' +
                'click ⚡ Compute Momentum in the sidebar.</div>',
                unsafe_allow_html=True
            )

        # ── Summary table ──────────────────────────────────────
        # Always reflect the active expected returns:
        # momentum-blended if enabled+computed, else shrinkage
        _mom_on = (st.session_state.momentum_enabled and
                   st.session_state.momentum_final_er is not None)
        _active  = (st.session_state.momentum_final_er.reindex(tickers)
                    if _mom_on else expected_returns)
        _er_label = "μ_final ann% (mom)" if _mom_on else "μ_shrinkage ann%"

        st.markdown('<div class="section-heading">Asset Summary</div>', unsafe_allow_html=True)
        if _mom_on:
            st.caption("⚡ Showing momentum-blended μ_final — these are the returns the optimizer uses.")
        summary_df = pd.DataFrame({
            "Ticker"        : tickers,
            _er_label       : (_active.values * 252 * 100).round(2),
            "Ann. Risk %"   : (asset_risks.values * 100).round(2),
            "Sharpe"        : ((_active.values * 252 - RF_ANNUAL) / asset_risks.values).round(3),
            "Daily Mean %"  : (_active.values * 100).round(4),
        })
        summary_df = summary_df.set_index("Ticker")
        st.dataframe(summary_df.style
            .background_gradient(cmap="Blues",  subset=[_er_label])
            .background_gradient(cmap="Reds",   subset=["Ann. Risk %"])
            .background_gradient(cmap="Greens", subset=["Sharpe"]),
            use_container_width=True)

        # ── Correlation heatmap (Plotly) ───────────────────────
        st.markdown('<div class="section-heading">Correlation Matrix</div>', unsafe_allow_html=True)
        render_insight_note("High positive clusters here are where diversification starts to disappear. If many names move together, the optimizer has fewer true ways to spread risk.")
        corr = cached_correlation_matrix(returns)
        fig_corr = plotly_heatmap(
            corr, "Asset Correlation Matrix",
            colorscale="RdBu", zmid=0, fmt=".2f",
            height=max(420, len(tickers) * 28 + 120)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Hover over any cell for full ticker names and exact value.  "
                   "Labels show abbreviated ticker (exchange suffix stripped).")

        # ── Returns distribution ───────────────────────────────
        st.markdown('<div class="section-heading">Return Distributions</div>', unsafe_allow_html=True)
        render_insight_note("These histograms show how noisy each name is day to day. Wide, skewed, or jumpy distributions often explain why a stock gets capped or downweighted.")
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
        render_insight_note("This is the raw opportunity map for individual assets. The optimizer generally prefers points that sit higher for the same level of risk.")
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
# TAB 3 — PORTFOLIOS  /  THE MIX (Noob)
# ══════════════════════════════════════════════════════════════
if _qs_s == "portfolios":
    if is_noob:
        render_noob_tab_the_mix()
    elif not st.session_state.data_loaded:
        render_insight_note("Load data in the Build tab first. Once the common market window is locked, this desk will show allocation spotlights and portfolio rankings.")
    else:
        tickers = st.session_state.tickers
        er      = optimizer_expected_returns()
        cov     = st.session_state.cov_matrix
        render_insight_note("This desk is for allocation decisions. Utility shows the most comfortable risk-adjusted utility mix, Tangency shows the cleanest excess-return efficiency, and Min-Risk shows the calmest feasible path to your target.")

        comparison_seed = {}
        if st.session_state.weights_utility is not None:
            w = st.session_state.weights_utility
            ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
            comparison_seed["Utility"] = {"Return %": ann_ret * 100, "Risk %": ann_risk * 100, "Sharpe": sharpe}
        if st.session_state.weights_tan is not None:
            comparison_seed["Tangency"] = {
                "Return %": st.session_state.tan_return * 100,
                "Risk %": st.session_state.tan_risk * 100,
                "Sharpe": st.session_state.tan_sharpe,
            }
        if st.session_state.weights_min is not None:
            w = st.session_state.weights_min
            ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
            comparison_seed["Min-Risk"] = {"Return %": ann_ret * 100, "Risk %": ann_risk * 100, "Sharpe": sharpe}
        if st.session_state.get("weights_deploy") is not None:
            w = np.asarray(st.session_state.weights_deploy, dtype=float)
            ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
            comparison_seed["Deploy-Now"] = {"Return %": ann_ret * 100, "Risk %": ann_risk * 100, "Sharpe": sharpe}

        best_sharpe_name = None
        if comparison_seed:
            best_sharpe_name = max(comparison_seed.items(), key=lambda item: item[1]["Sharpe"])[0]

        st.markdown('<div class="section-heading">Utility-Maximized Portfolio</div>', unsafe_allow_html=True)
        if st.session_state.weights_utility is not None:
            w = st.session_state.weights_utility
            ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
            fig_pie = go.Figure(go.Pie(
                labels=tickers, values=w * 100, hole=0.45, textinfo='label+percent', textfont=dict(size=10),
                marker=dict(colors=[f"hsl({int(i*360/len(tickers))},60%,50%)" for i in range(len(tickers))],
                            line=dict(color='#0a0e17', width=2))
            ))
            fig_pie.update_layout(template="plotly_dark", paper_bgcolor="#0a0e17",
                                  title=dict(text="Utility Portfolio Allocation", font=dict(color="#a0c8e8", size=13)),
                                  showlegend=True, legend=dict(font=dict(size=9, color="#80b0d0")),
                                  height=380, margin=dict(l=10, r=10, t=50, b=10))
            render_portfolio_spotlight(
                "Utility-Maximized Portfolio",
                "Best when you want the optimizer to respect the risk penalty and still keep expected return in play.",
                w, ann_ret, ann_risk, sharpe, fig_pie,
                badge="Risk utility lens",
                badge_gold=(best_sharpe_name == "Utility")
            )
        else:
            render_insight_note("Utility is waiting for its first run. Use this when you want the most comfortable risk-adjusted mix under your constraints.")

        st.markdown("---")
        st.markdown('<div class="section-heading">Tangency Portfolio (Max Sharpe)</div>', unsafe_allow_html=True)
        if st.session_state.weights_tan is not None:
            w = st.session_state.weights_tan
            tr = st.session_state.tan_return
            tk = st.session_state.tan_risk
            ts = st.session_state.tan_sharpe
            fig_pie2 = go.Figure(go.Pie(
                labels=tickers, values=w * 100, hole=0.45, textinfo='label+percent', textfont=dict(size=10),
                marker=dict(colors=[f"hsl({int(i*360/len(tickers)+40)},65%,52%)" for i in range(len(tickers))],
                            line=dict(color='#0a0e17', width=2))
            ))
            fig_pie2.update_layout(template="plotly_dark", paper_bgcolor="#0a0e17",
                                   title=dict(text="Tangency Portfolio Allocation", font=dict(color="#a0c8e8", size=13)),
                                   showlegend=True, legend=dict(font=dict(size=9, color="#80b0d0")),
                                   height=380, margin=dict(l=10, r=10, t=50, b=10))
            render_portfolio_spotlight(
                "Tangency Portfolio",
                "Best when you want the highest excess return per unit of risk and a clean benchmark against every other portfolio.",
                w, tr, tk, ts, fig_pie2,
                badge="Best Sharpe" if best_sharpe_name == "Tangency" else "Efficiency lens",
                badge_gold=(best_sharpe_name == "Tangency")
            )
        else:
            render_insight_note("Tangency has not been computed yet. This is the portfolio to use when you want the sharpest return-to-risk benchmark.")

        st.markdown("---")
        st.markdown(f'<div class="section-heading">Min-Risk Portfolio (Target: {st.session_state.target_return*100:.1f}%)</div>', unsafe_allow_html=True)
        if st.session_state.weights_min is not None:
            w = st.session_state.weights_min
            ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
            fig_pie3 = go.Figure(go.Pie(
                labels=tickers, values=w * 100, hole=0.45, textinfo='label+percent', textfont=dict(size=10),
                marker=dict(colors=[f"hsl({int(i*360/len(tickers)+80)},60%,48%)" for i in range(len(tickers))],
                            line=dict(color='#0a0e17', width=2))
            ))
            fig_pie3.update_layout(template="plotly_dark", paper_bgcolor="#0a0e17",
                                   title=dict(text="Min-Risk Portfolio Allocation", font=dict(color="#a0c8e8", size=13)),
                                   showlegend=True, legend=dict(font=dict(size=9, color="#80b0d0")),
                                   height=380, margin=dict(l=10, r=10, t=50, b=10))
            render_portfolio_spotlight(
                "Min-Risk Portfolio",
                "Best when the brief is capital preservation first and you only want enough return to clear the target hurdle.",
                w, ann_ret, ann_risk, sharpe, fig_pie3,
                badge="Target discipline",
                badge_gold=(best_sharpe_name == "Min-Risk")
            )
        else:
            render_insight_note("Min-Risk will appear once you solve a feasible target-return problem. Lower the target if the desk cannot find a clean solution.")

        if st.session_state.get("weights_deploy") is not None:
            st.markdown("---")
            st.markdown('<div class="section-heading">Deploy-Now Portfolio</div>', unsafe_allow_html=True)
            w = np.asarray(st.session_state.weights_deploy, dtype=float)
            ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
            _deploy_meta = st.session_state.get("deployable_meta") or {}
            _deploy_posture = _deployment_posture_display(_deploy_meta.get("posture", "Balanced"))
            _deploy_date = str(_deploy_meta.get("date", "latest rebalance"))
            fig_pie4 = go.Figure(go.Pie(
                labels=tickers, values=w * 100, hole=0.45, textinfo='label+percent', textfont=dict(size=10),
                marker=dict(colors=[f"hsl({int(i*360/len(tickers)+120)},62%,50%)" for i in range(len(tickers))],
                            line=dict(color='#0a0e17', width=2))
            ))
            fig_pie4.update_layout(template="plotly_dark", paper_bgcolor="#0a0e17",
                                   title=dict(text="Deploy-Now Allocation", font=dict(color="#a0c8e8", size=13)),
                                   showlegend=True, legend=dict(font=dict(size=9, color="#80b0d0")),
                                   height=380, margin=dict(l=10, r=10, t=50, b=10))
            render_portfolio_spotlight(
                "Deploy-Now Portfolio",
                f"Latest walk-forward adjusted mix as of {_deploy_date}. This is the version that already reflects tactical rebalance, defensive blend, and any cash sleeve from the recent backtest.",
                w, ann_ret, ann_risk, sharpe, fig_pie4,
                badge=f"{_deploy_posture} · Backtest-adjusted",
                badge_gold=(best_sharpe_name == "Deploy-Now")
            )
        elif st.session_state.get("deployable_meta"):
            render_insight_note("A deploy-now recommendation is available in Backtest. Promote it once from that tab and it will appear here as a first-class portfolio.")

        # ── Side-by-side comparison ────────────────────────────
        any_computed = any([
            st.session_state.weights_utility is not None,
            st.session_state.weights_tan is not None,
            st.session_state.weights_min is not None,
            st.session_state.get("weights_deploy") is not None,
        ])
        if any_computed:
            st.markdown("---")
            st.markdown('<div class="section-heading">Portfolio Comparison</div>', unsafe_allow_html=True)

            comp_data = {}
            if st.session_state.weights_utility is not None:
                w = st.session_state.weights_utility
                ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
                comp_data["Utility"] = {
                    "Return %": round(ann_ret * 100, 2),
                    "Risk %":   round(ann_risk * 100, 2),
                    "Sharpe":   round(sharpe, 3),
                }
            if st.session_state.weights_tan is not None:
                comp_data["Tangency"] = {
                    "Return %": round(st.session_state.tan_return * 100, 2),
                    "Risk %":   round(st.session_state.tan_risk   * 100, 2),
                    "Sharpe":   round(st.session_state.tan_sharpe, 3),
                }
            if st.session_state.weights_min is not None:
                w = st.session_state.weights_min
                ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
                comp_data["Min-Risk"] = {
                    "Return %": round(ann_ret * 100, 2),
                    "Risk %":   round(ann_risk * 100, 2),
                    "Sharpe":   round(sharpe, 3),
                }
            if st.session_state.get("weights_deploy") is not None:
                w = np.asarray(st.session_state.weights_deploy, dtype=float)
                ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
                comp_data["Deploy-Now"] = {
                    "Return %": round(ann_ret * 100, 2),
                    "Risk %":   round(ann_risk * 100, 2),
                    "Sharpe":   round(sharpe, 3),
                }

            if comp_data:
                comp_df = pd.DataFrame(comp_data).T
                render_insight_note("Use the cards first for ranking intuition, then the table and chart for exact deltas. In most sessions, Tangency is the benchmark portfolio to beat.")
                render_comparison_spotlights(comp_df)
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
# TAB 4 — EFFICIENT FRONTIER  (both modes)
# ══════════════════════════════════════════════════════════════
if _qs_s == "frontier":
    if is_noob:
        st.markdown(
            '<div class="nb-info-box" style="margin-bottom:1rem">📈 <strong>The Frontier</strong> — '
            'This chart shows every possible mix of your stocks. The curved line is the "sweet spot" '
            'where you get the most growth for the least drama. The gold star ⭐ is your best balanced pick.</div>',
            unsafe_allow_html=True
        )
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
# TAB 5 — ANALYSIS  /  GAINS OVER TIME (Noob)
# ══════════════════════════════════════════════════════════════
if _qs_s == "analysis":
    if is_noob:
        st.markdown('<div class="nb-section">📈 How Did Your Portfolio Do Over Time?</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="nb-info-box">This chart shows how ₹1 invested would have grown over time '
            'with your portfolio mix. The higher the line goes, the better! '
            'Dips are normal — what matters is the overall direction.</div>',
            unsafe_allow_html=True
        )
        ss = st.session_state
        if not ss.get("data_loaded"):
            st.markdown(
                '<div class="nb-info-box">Load your data and run the optimizer to see the growth chart!</div>',
                unsafe_allow_html=True
            )
        else:
            returns_df = ss.get("returns")
            tickers_nb = ss.get("tickers", [])
            if returns_df is not None and (
                ss.get("weights_tan") is not None or
                ss.get("weights_utility") is not None
            ):
                fig_nb_cum = go.Figure()
                _nb_port_data = {}
                if ss.get("weights_utility") is not None:
                    _nb_port_data["Max Growth"] = (ss.weights_utility, "#00e676")
                if ss.get("weights_tan") is not None:
                    _nb_port_data["Best Balanced"] = (ss.weights_tan, "gold")
                if ss.get("weights_min") is not None:
                    _nb_port_data["Smoothest Ride"] = (ss.weights_min, "#7dd3fc")
                for _nb_name, (_nb_w, _nb_col) in _nb_port_data.items():
                    _nb_cum = cached_portfolio_cumulative_returns(returns_df[tickers_nb], _nb_w)
                    fig_nb_cum.add_trace(go.Scatter(
                        x=_nb_cum.index, y=_nb_cum.values * 100,
                        mode='lines', name=_nb_name,
                        line=dict(width=2.5, color=_nb_col),
                        hovertemplate=f"<b>{_nb_name}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}%<extra></extra>"
                    ))
                fig_nb_cum.update_layout(
                    template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0d1525",
                    xaxis_title="Date", yaxis_title="Cumulative Return %",
                    title=dict(text="How ₹1 Grew Over Time", font=dict(color="#a0c8e8")),
                    font=dict(family="DM Mono", color="#80b0d0"),
                    legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0.4)"),
                    height=420, margin=dict(l=50, r=20, t=50, b=50)
                )
                st.plotly_chart(fig_nb_cum, use_container_width=True)
                # Piggy bank sim right beneath the chart
                _nb_wt = ss.get("weights_tan")
                _nb_best_w = _nb_wt if _nb_wt is not None else ss.get("weights_utility")
                if _nb_best_w is not None:
                    _nb_cum_best = cached_portfolio_cumulative_returns(returns_df[tickers_nb], _nb_best_w)
                    _nb_cum_df = _nb_cum_best.to_frame(name="portfolio") + 1  # convert % cumret to multiplier
                    render_noob_piggy_bank(weights=_nb_best_w, cumulative_returns_df=_nb_cum_df, key_prefix="nb_analysis_tab")
            else:
                st.markdown(
                    '<div class="nb-info-box">Hit <strong>Run ALL</strong> in the sidebar '
                    'to see your portfolio growth chart!</div>',
                    unsafe_allow_html=True
                )
    elif not st.session_state.data_loaded:
        render_insight_note("Analysis turns the optimizer inputs into a story: volatility regimes, wealth paths, and covariance structure. Load data first to unlock it.")
    else:
        returns = st.session_state.returns
        tickers = st.session_state.tickers
        er      = optimizer_expected_returns()
        cov     = st.session_state.cov_matrix
        ar      = st.session_state.asset_risks
        are     = st.session_state.asset_returns

        # ── Rolling correlation ────────────────────────────────
        st.markdown('<div class="section-heading">Rolling Volatility (60-day)</div>', unsafe_allow_html=True)
        render_insight_note("Use this to see regime changes. If one asset's volatility suddenly spikes, it can dominate covariance and pull the optimizer away from it.")
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
        render_insight_note("This is the path each asset actually took over your sample window. It helps separate smooth compounders from names that only look good in average-return form.")
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
            render_insight_note("This is often the most decision-relevant chart in the app. If two portfolios have similar Sharpe but one compounds more smoothly, that's usually the one investors tolerate better.")
            fig_port = go.Figure()
            port_data = {}
            if st.session_state.weights_utility is not None:
                port_data["Utility"] = (st.session_state.weights_utility, "#00b4ff")
            if st.session_state.weights_tan is not None:
                port_data["Tangency"] = (st.session_state.weights_tan, "gold")
            if st.session_state.weights_min is not None:
                port_data["Min-Risk"] = (st.session_state.weights_min, "#00e676")

            for name, (w, color) in port_data.items():
                port_cum = cached_portfolio_cumulative_returns(returns[tickers], w)
                fig_port.add_trace(go.Scatter(
                    x=port_cum.index, y=port_cum.values * 100,
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

        # ── Covariance heatmap (Plotly) ────────────────────────
        st.markdown('<div class="section-heading">Covariance Matrix (Annualised)</div>', unsafe_allow_html=True)
        render_insight_note("Covariance is the engine room of the optimizer. Large off-diagonal blocks mean the desk sees those assets as moving together, which reduces diversification benefit.")
        cov_annual = cov * 252
        fig_cov = plotly_heatmap(
            cov_annual, "Annualised Covariance Matrix",
            colorscale="YlOrBr", zmid=None, fmt=".4f",
            height=max(420, len(tickers) * 28 + 120)
        )
        st.plotly_chart(fig_cov, use_container_width=True)
        st.caption("Hover over any cell for full ticker names and exact value.  "
                   "Diagonal = each stock's own annualised variance.")


# ══════════════════════════════════════════════════════════════
# TAB 6 — BACKTEST  /  MY MONEY (Noob)
# ══════════════════════════════════════════════════════════════
if _qs_s == "backtest":
    import datetime as _dt
    import yfinance as _yf2

    if is_noob:
        # Build cumulative returns df for noob piggy bank if data is ready
        _nb_cum_df_for_money = None
        _ss = st.session_state
        if _ss.get("data_loaded") and _ss.get("returns") is not None:
            _nb_wt_m = _ss.get("weights_tan")
            _nb_best_w_money = _nb_wt_m if _nb_wt_m is not None else _ss.get("weights_utility")
            if _nb_best_w_money is not None:
                try:
                    _nb_raw_cum = cached_portfolio_cumulative_returns(
                        _ss.returns[_ss.tickers], _nb_best_w_money
                    )
                    # Convert cumulative return series to multiplier (1.0 = start)
                    _nb_cum_df_for_money = (_nb_raw_cum + 1).to_frame(name="portfolio")
                except Exception:
                    _nb_cum_df_for_money = None
        render_noob_tab_my_money(cumulative_returns_df=_nb_cum_df_for_money)
    else:
        st.markdown('<div class="section-heading">Portfolio Backtest — Capital Simulation</div>', unsafe_allow_html=True)

        if not st.session_state.data_loaded:
            st.markdown('<div class="status-box status-info">Load market data and run at least one optimisation first.</div>', unsafe_allow_html=True)
        else:
            available_portfolios = {}
            if st.session_state.weights_utility is not None:
                available_portfolios["Utility Maximiser"] = st.session_state.weights_utility
            if st.session_state.weights_tan is not None:
                available_portfolios["Tangency (Max Sharpe)"] = st.session_state.weights_tan
            if st.session_state.weights_min is not None:
                available_portfolios["Min-Risk"] = st.session_state.weights_min

            if not available_portfolios:
                st.markdown('<div class="status-box status-warn">Run at least one portfolio optimisation before backtesting.</div>', unsafe_allow_html=True)
            else:
                col_cfg, col_res = st.columns([1, 2])

                with col_cfg:
                    st.markdown('<div class="card-title">Backtest Settings</div>', unsafe_allow_html=True)

                    selected_portfolio = st.selectbox(
                        "Portfolio to backtest",
                        list(available_portfolios.keys()),
                        key="bt_portfolio_name"
                    )
                    weights_bt = available_portfolios[selected_portfolio]
                    tickers_bt = st.session_state.tickers

                    capital = st.number_input(
                        "Starting Capital (₹ or $)",
                        min_value=1000.0, max_value=1_000_000_000.0,
                        value=100_000.0, step=1000.0, format="%.2f",
                        key="bt_capital"
                    )

                    st.markdown("**Backtest period**")
                    use_session_dates = st.checkbox(
                        "Use the same dates as data window", value=True, key="bt_use_session"
                    )
                    if use_session_dates:
                        bt_start = st.session_state.start_date
                        bt_end   = st.session_state.end_date
                        st.caption(f"▶ {bt_start}  →  {bt_end}")
                    else:
                        bt_mode = st.radio(
                            "Date mode", ["Start / End", "Days back from today"],
                            horizontal=True, key="bt_date_mode"
                        )
                        if bt_mode == "Start / End":
                            bt_start = st.text_input("Backtest start", value=st.session_state.start_date, key="bt_s")
                            bt_end   = st.text_input("Backtest end",   value=st.session_state.end_date,   key="bt_e")
                        else:
                            bt_days  = st.number_input("Days back", min_value=30, max_value=7300,
                                                        value=365*3, step=30, key="bt_days")
                            bt_end   = _dt.date.today().strftime("%Y-%m-%d")
                            bt_start = (_dt.date.today() - _dt.timedelta(days=int(bt_days))).strftime("%Y-%m-%d")
                            st.caption(f"▶ {bt_start}  →  {bt_end}")

                    st.markdown("---")
                    bt_engine = st.radio(
                        "Backtest engine",
                        ["Walk-forward adaptive (Recommended)", "Static hold"],
                        key="bt_engine_mode",
                        help="Walk-forward re-estimates the portfolio through the backtest window. Static hold simply carries the original weights forward.",
                    )

                    allow_fractional = st.checkbox(
                        "Allow fractional shares", value=False, key="bt_frac",
                        help="If OFF, floor to whole shares and leave residual as cash."
                    )

                    if bt_engine.startswith("Walk-forward"):
                        st.markdown("---")
                        try:
                            _bt_days_span = max((pd.Timestamp(bt_end) - pd.Timestamp(bt_start)).days, 0)
                        except Exception:
                            _bt_days_span = 0
                        rebalance_freq = st.selectbox(
                            "Rebalance frequency",
                            ["Weekly (5d)", "Biweekly (10d)", "Monthly (21d)"],
                            index=2,
                            key="bt_rebalance_freq",
                        )
                        wf_speed_mode = st.selectbox(
                            "Walk-forward detail",
                            ["Full fidelity", "Lightweight (faster)"],
                            index=1 if _bt_days_span <= 120 else 0,
                            key="bt_wf_speed",
                            help="Lightweight mode uses shorter training windows and skips the expensive momentum refresh inside each rebalance step.",
                        )
                        gate_strength = st.slider(
                            "Regime safety strength",
                            min_value=0.0, max_value=2.0, value=1.0, step=0.1,
                            key="bt_gate_strength",
                            help="Higher values blend aggressive portfolios faster toward min-risk/cash when the deployment regime diverges from the training regime.",
                        )
                        st.caption(
                            "Walk-forward mode re-estimates the selected portfolio through the test window, "
                            "checks current regime distance vs the training regime, and can blend toward min-risk/cash when conditions deteriorate. "
                            + ("Lightweight mode skips intra-step momentum refresh for speed." if wf_speed_mode.startswith("Lightweight") else "Full fidelity keeps the tactical refresh turned on.")
                        )
                    else:
                        # ── Entry mode ─────────────────────────────────────
                        st.markdown("---")
                        st.markdown('<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem">Entry Mode</div>', unsafe_allow_html=True)
                        entry_mode = st.radio(
                            "entry_mode_radio",
                            [
                                "📅 Common start — all stocks bought on the same day",
                                "📈 Staggered — each stock bought on its own first listed day",
                            ],
                            label_visibility="collapsed",
                            key="bt_entry_mode"
                        )
                        if entry_mode.startswith("📅"):
                            st.caption(
                                "All stocks are bought on the **latest** first-available date among "
                                "the portfolio (so every stock has a price on day 1). "
                                "Returns are directly comparable across the same window."
                            )
                        else:
                            st.caption(
                                "Each stock is bought on its own first trading day in the window. "
                                "Capital for that stock sits as uninvested cash until then. "
                                "Per-stock returns cover different periods — portfolio-level CAGR "
                                "is measured from the earliest buy to the latest sell."
                            )

                    # ── Pre-flight data availability panel ─────────────
                    st.markdown("---")
                    st.markdown('<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem">Ticker Data Availability</div>', unsafe_allow_html=True)
                    try:
                        chosen_start = _dt.date.fromisoformat(bt_start)
                        chosen_end   = _dt.date.fromisoformat(bt_end)
                    except Exception:
                        chosen_start = chosen_end = None

                    for t in tickers_bt:
                        first_date, last_date, colour, age = get_ticker_date_range(t)
                        warn_start = warn_end = False
                        try:
                            if first_date != "N/A" and chosen_start:
                                warn_start = chosen_start < _dt.date.fromisoformat(first_date)
                            if last_date != "N/A" and chosen_end:
                                warn_end   = chosen_end > _dt.date.fromisoformat(last_date)
                        except Exception:
                            pass
                        warn_html = ""
                        if warn_start:
                            warn_html += f'<br><span style="font-size:0.63rem;color:#ff5252">⚠ data starts {first_date} — entry shifts to first available day</span>'
                        if warn_end:
                            warn_html += f'<br><span style="font-size:0.63rem;color:#ffd54f">⚠ last data {last_date} — exit will use that date</span>'
                        st.markdown(
                            f'<div style="margin-bottom:0.45rem">'
                            f'<span class="ticker-chip">{t}</span> '
                            f'<span style="font-size:0.68rem;color:#4a90d9">from </span>'
                            f'<span style="font-size:0.68rem;color:#a0c8e8">{first_date}</span>'
                            f'<span style="font-size:0.68rem;color:#4a90d9"> → </span>'
                            f'<span style="font-size:0.68rem;color:{colour}">{last_date}</span>'
                            + warn_html + '</div>',
                            unsafe_allow_html=True
                        )

                    run_bt = st.button("▶ Run Backtest", use_container_width=True, key="run_bt_btn")

                # ── Backtest engine ────────────────────────────────────
                with col_res:
                    if run_bt:
                        if bt_engine.startswith("Walk-forward"):
                            _rebalance_step_map = {
                                "Weekly (5d)": 5,
                                "Biweekly (10d)": 10,
                                "Monthly (21d)": 21,
                            }
                            with st.spinner("Running walk-forward regime-aware backtest…"):
                                try:
                                    _wf = _run_walkforward_backtest(
                                        selected_portfolio=selected_portfolio,
                                        weights_bt=weights_bt,
                                        tickers_bt=tickers_bt,
                                        bt_start=bt_start,
                                        bt_end=bt_end,
                                        capital=capital,
                                        allow_fractional=allow_fractional,
                                        rebalance_step_days=_rebalance_step_map.get(rebalance_freq, 21),
                                        reference_meta=st.session_state.get("regime_meta") or {},
                                        lightweight=wf_speed_mode.startswith("Lightweight"),
                                        gate_strength=gate_strength,
                                    )

                                    _wf_daily = _wf["daily_vals"]
                                    _wf_rebal = _wf["rebalance_df"]
                                    _wf_total_exit = _wf["total_exit"]
                                    _wf_total_invest = _wf["total_invest"]
                                    _wf_abs = _wf["abs_return"]
                                    _wf_pct = _wf["pct_return"]
                                    _wf_cagr = _wf["cagr"]
                                    _wf_bm_ret = _wf.get("benchmark_return")
                                    _wf_bm_sym = _wf.get("benchmark_symbol")
                                    _wf_deploy = _wf.get("deployable") or {}

                                    st.markdown(f"""
                                    <div class="metric-row">
                                        {metric_card("Capital Invested", f"₹{_wf_total_invest:,.0f}", "accent")}
                                        {metric_card("Portfolio Value",  f"₹{_wf_total_exit:,.0f}",   "positive" if _wf_pct>0 else "")}
                                        {metric_card("Absolute Return",  f"₹{_wf_abs:+,.0f}",  "positive" if _wf_abs>0 else "")}
                                        {metric_card("Total Return",     f"{_wf_pct:+.2f}%",   "positive" if _wf_pct>0 else "")}
                                        {metric_card("CAGR",             f"{_wf_cagr:+.2f}%",  "positive" if _wf_cagr>0 else "")}
                                        {metric_card("Benchmark",        f"{_wf_bm_ret:+.2f}%" if _wf_bm_ret is not None else "—", "accent")}
                                    </div>""", unsafe_allow_html=True)

                                    if _wf_bm_ret is not None:
                                        _rel = _wf_pct - _wf_bm_ret
                                        st.caption(
                                            f"Walk-forward mode rebalanced through {_wf['n_days']} days ({_wf['n_years']:.2f} years)  ·  "
                                            f"benchmark {_wf_bm_sym}: {_wf_bm_ret:+.2f}%  ·  relative alpha: {_rel:+.2f}%."
                                        )
                                    else:
                                        st.caption(
                                            f"Walk-forward mode rebalanced through {_wf['n_days']} days ({_wf['n_years']:.2f} years)."
                                        )

                                    st.markdown('<div class="section-heading">Adaptive Portfolio Value Over Time</div>', unsafe_allow_html=True)
                                    fig_wf = go.Figure()
                                    fig_wf.add_trace(go.Scatter(
                                        x=_wf_daily.index, y=_wf_daily.values,
                                        mode='lines', name="Adaptive Portfolio",
                                        fill='tozeroy',
                                        line=dict(color="#00b4ff", width=2.2),
                                        fillcolor="rgba(0,180,255,0.08)",
                                        hovertemplate="%{x|%Y-%m-%d}<br>₹%{y:,.2f}<extra></extra>"
                                    ))
                                    if _wf.get("benchmark_curve") is not None:
                                        _wf_bm_curve = _wf["benchmark_curve"].reindex(_wf_daily.index).ffill().bfill()
                                        fig_wf.add_trace(go.Scatter(
                                            x=_wf_bm_curve.index, y=_wf_bm_curve.values,
                                            mode='lines', name=f"Benchmark ({_wf_bm_sym})",
                                            line=dict(color="#ffd54f", width=1.6, dash="dot"),
                                            hovertemplate="%{x|%Y-%m-%d}<br>₹%{y:,.2f}<extra></extra>"
                                        ))
                                    fig_wf.add_hline(
                                        y=capital, line_dash="dash",
                                        line_color="rgba(255,213,79,0.5)",
                                        annotation_text="Starting capital",
                                        annotation_font_color="#ffd54f"
                                    )
                                    fig_wf.update_layout(
                                        template="plotly_dark", paper_bgcolor="#0a0e17",
                                        plot_bgcolor="#0d1525",
                                        xaxis_title="Date", yaxis_title="Portfolio Value (₹)",
                                        title=dict(text=f"Walk-Forward Backtest: {selected_portfolio}", font=dict(color="#a0c8e8")),
                                        font=dict(family="DM Mono", color="#80b0d0"),
                                        height=420, margin=dict(l=50, r=20, t=50, b=50)
                                    )
                                    st.plotly_chart(fig_wf, use_container_width=True)

                                    if _wf_deploy:
                                        _deploy_cmp = _wf_deploy.get("comparison")
                                        _deploy_cash = float(_wf_deploy.get("cash_weight", 0.0) or 0.0)
                                        _deploy_blend = float(_wf_deploy.get("defensive_blend", 0.0) or 0.0)
                                        _deploy_conf = float(_wf_deploy.get("confidence", 0.0) or 0.0)
                                        _deploy_trust = float(_wf_deploy.get("trust_score", 0.0) or 0.0)
                                        _deploy_transition = float(_wf_deploy.get("transition_risk", 0.0) or 0.0)
                                        _deploy_stability = float(_wf_deploy.get("stability_score", 0.0) or 0.0)
                                        _deploy_disagreement = float(_wf_deploy.get("disagreement_score", 0.0) or 0.0)
                                        _deploy_posture = _deployment_posture_display(_wf_deploy.get("posture", "Balanced"))
                                        st.markdown("""
                                        <style>
                                        .deploy-now-row { gap: 0.7rem; }
                                        .deploy-now-row .metric-box { min-width: 108px; padding: 0.78rem 0.85rem; }
                                        .deploy-now-row .metric-box .label { font-size: 0.56rem; letter-spacing: 0.09em; }
                                        .deploy-now-row .metric-box .value { font-size: 1.12rem; line-height: 1.05; }
                                        </style>
                                        """, unsafe_allow_html=True)
                                        st.markdown('<div class="section-heading">Suggested Investment Weights Today</div>', unsafe_allow_html=True)
                                        render_insight_note(
                                            "This is the latest deployable allocation from the walk-forward engine. It starts from the selected optimizer portfolio, applies the current tactical rebalance, then blends toward defense and cash if the recent market tape looks less friendly than the training regime."
                                        )
                                        st.markdown(f"""
                                        <div class="metric-row deploy-now-row">
                                            {metric_card("Posture", str(_deploy_posture), "accent")}
                                            {metric_card("Confidence", f"{_deploy_conf:.0f}/100", "positive" if _deploy_conf >= 60 else "")}
                                            {metric_card("Trust", f"{_deploy_trust:.0f}/100", "positive" if _deploy_trust >= 60 else "")}
                                            {metric_card("Cash", f"{_deploy_cash*100:.1f}%", "accent")}
                                            {metric_card("Defense", f"{_deploy_blend*100:.1f}%", "accent")}
                                            {metric_card("Regime", str(_wf_deploy.get("regime_used", "unknown")).title(), "accent")}
                                            {metric_card("As Of", str(_wf_deploy.get("date", "—")), "accent")}
                                        </div>""", unsafe_allow_html=True)
                                        st.caption(
                                            f"Strategic regime: {str(_wf_deploy.get('strategic_regime', 'unknown')).title()}  ·  "
                                            f"tactical regime: {str(_wf_deploy.get('tactical_regime', 'unknown')).title()}  ·  "
                                            f"decision source: {str(_wf_deploy.get('regime_source', 'slow')).title()}."
                                        )
                                        _sig1, _sig2, _sig3 = st.columns(3)
                                        with _sig1:
                                            st.markdown(
                                                f"**Transition Risk**: {_deploy_transition*100:.1f}%  \n"
                                                f"**Stability**: {_deploy_stability*100:.1f}%"
                                            )
                                        with _sig2:
                                            st.markdown(
                                                f"**Model Disagreement**: {_deploy_disagreement*100:.1f}%  \n"
                                                f"**Stress**: {float(_wf_deploy.get('stress_score', 0.0) or 0.0):.3f}"
                                            )
                                        with _sig3:
                                            _mix = _wf_deploy.get("scenario_mix") or {}
                                            st.markdown(
                                                f"**Scenario Mix**: Base {_mix.get('base', 0):.1f}%  \n"
                                                f"Weak {_mix.get('weak', 0):.1f}%  ·  Crisis {_mix.get('crisis', 0):.1f}%"
                                            )
                                        if _deploy_cmp is not None and not _deploy_cmp.empty:
                                            st.dataframe(
                                                _deploy_cmp.style.format({
                                                    "Original %": "{:.2f}%",
                                                    "Tactical %": "{:.2f}%",
                                                    "Defensive %": "{:.2f}%",
                                                    "Deploy Now %": "{:.2f}%",
                                                    "Delta vs Original pp": "{:+.2f} pp",
                                                }),
                                                use_container_width=True,
                                                height=min(420, 60 + 35 * len(_deploy_cmp)),
                                            )
                                        _deploy_explanations = _wf_deploy.get("explanations")
                                        if _deploy_explanations is not None and not _deploy_explanations.empty:
                                            st.markdown("**Why These Changes Happened**")
                                            st.dataframe(
                                                _deploy_explanations,
                                                use_container_width=True,
                                                height=min(320, 60 + 35 * len(_deploy_explanations)),
                                            )
                                        _bands = _wf_deploy.get("confidence_bands")
                                        if _bands is not None and not _bands.empty:
                                            st.markdown("**Weight Confidence Bands**")
                                            st.dataframe(
                                                _bands.style.format({
                                                    "Lower %": "{:.2f}%",
                                                    "Deploy Now %": "{:.2f}%",
                                                    "Upper %": "{:.2f}%",
                                                }),
                                                use_container_width=True,
                                                height=min(320, 60 + 35 * len(_bands)),
                                            )
                                        _deploy_calibration = _wf_deploy.get("calibration")
                                        if _deploy_calibration is not None and not _deploy_calibration.empty:
                                            st.markdown("**Recent Calibration Examples**")
                                            st.dataframe(
                                                _deploy_calibration.head(5).style.format({
                                                    "Deploy Return %": "{:+.2f}%",
                                                    "Tactical Return %": "{:+.2f}%",
                                                    "Defensive Return %": "{:+.2f}%",
                                                    "Portfolio Return %": "{:+.2f}%",
                                                    "Benchmark Return %": "{:+.2f}%",
                                                    "Relative vs Benchmark pp": "{:+.2f} pp",
                                                    "Defense %": "{:.1f}%",
                                                    "Cash %": "{:.1f}%",
                                                }),
                                                use_container_width=True,
                                                height=min(280, 60 + 35 * len(_deploy_calibration.head(5))),
                                            )
                                        if st.button("View Weights In Portfolio View", key="use_deployable_weights_btn", use_container_width=True):
                                            _promote_deployable_weights_to_portfolios(_wf_deploy)
                                            st.rerun()
                                        _adds = _wf_deploy.get("top_adds")
                                        _cuts = _wf_deploy.get("top_cuts")
                                        _c1, _c2 = st.columns(2)
                                        with _c1:
                                            st.markdown("**Top Increases vs Original**")
                                            if _adds is not None and not _adds.empty:
                                                st.dataframe(
                                                    _adds[["Deploy Now %", "Delta vs Original pp"]].style.format({
                                                        "Deploy Now %": "{:.2f}%",
                                                        "Delta vs Original pp": "{:+.2f} pp",
                                                    }),
                                                    use_container_width=True,
                                                    height=min(220, 60 + 35 * len(_adds)),
                                                )
                                        with _c2:
                                            st.markdown("**Top Cuts vs Original**")
                                            if _cuts is not None and not _cuts.empty:
                                                st.dataframe(
                                                    _cuts[["Deploy Now %", "Delta vs Original pp"]].style.format({
                                                        "Deploy Now %": "{:.2f}%",
                                                        "Delta vs Original pp": "{:+.2f} pp",
                                                    }),
                                                    use_container_width=True,
                                                    height=min(220, 60 + 35 * len(_cuts)),
                                                )

                                    st.markdown('<div class="section-heading">Regime Defense Log</div>', unsafe_allow_html=True)
                                    render_insight_note(
                                        "Each rebalance shows the strategic regime, the tactical short-term regime, which one the short-horizon engine actually used, how far conditions had drifted from the original training regime, and how much the engine blended the selected portfolio toward min-risk or cash."
                                    )
                                    st.dataframe(
                                        _wf_rebal.style.format({
                                            "Regime Distance": "{:.3f}",
                                            "Stress Score": "{:.3f}",
                                            "Defensive Blend %": "{:.1f}%",
                                            "Cash %": "{:.1f}%",
                                            "Alpha %": "{:.1f}%",
                                            "Beta %": "{:.1f}%",
                                            "Top Weight %": "{:.1f}%",
                                        }),
                                        use_container_width=True,
                                        height=min(420, 60 + 35 * len(_wf_rebal)),
                                    )

                                    st.session_state.bt_results = {
                                        "portfolio_name": f"{selected_portfolio} · Walk-Forward",
                                        "capital": capital,
                                        "entry_mode": "Walk-Forward Adaptive",
                                        "overall_entry": _wf["overall_entry"],
                                        "overall_exit": _wf["overall_exit"],
                                        "n_days": _wf["n_days"],
                                        "n_years": _wf["n_years"],
                                        "total_invest": _wf_total_invest,
                                        "total_exit": _wf_total_exit,
                                        "abs_return": _wf_abs,
                                        "pct_return": _wf_pct,
                                        "cagr": _wf_cagr,
                                        "residual": _wf.get("residual", 0.0),
                                        "allow_fractional": allow_fractional,
                                        "blotter": _wf_rebal,
                                        "deployable": _wf_deploy,
                                    }

                                    st.download_button(
                                        "⬇ Download Rebalance Log CSV",
                                        data=_wf_rebal.to_csv(index=False).encode(),
                                        file_name="quansen_walkforward_backtest.csv",
                                        mime="text/csv",
                                        use_container_width=True,
                                    )
                                except Exception as e:
                                    st.error(f"Walk-forward backtest failed: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
                        else:
                            with st.spinner("Downloading price data for backtest window…"):
                                bt_data = _yf2.download(
                                    tickers_bt, start=bt_start, end=bt_end,
                                    auto_adjust=True, progress=False
                                )
                                bt_prices = bt_data["Close"]
                                if isinstance(bt_prices, pd.Series):
                                    bt_prices = bt_prices.to_frame(name=tickers_bt[0])

                                present = [t for t in tickers_bt if t in bt_prices.columns]
                                bt_prices = bt_prices[present]

                                # Drop rows where ALL tickers are NaN but keep partial rows —
                                # each ticker gets its own first/last valid date below.
                                bt_prices = bt_prices.dropna(how="all")

                                if bt_prices.empty or len(bt_prices) < 2:
                                    st.error("No price data returned for the backtest window.")
                                else:
                                    # ── Step 1: find each ticker's own first & last valid date ──
                                    ticker_first_valid = {}   # first day with real price in window
                                    ticker_last_valid  = {}   # last  day with real price in window
                                    valid_tickers = []

                                    for t in present:
                                        col = bt_prices[t].dropna()
                                        col = col[col > 0]
                                        if len(col) < 2:
                                            continue
                                        valid_tickers.append(t)
                                        ticker_first_valid[t] = col.index[0]
                                        ticker_last_valid[t]  = col.index[-1]

                                    skipped = [t for t in present if t not in valid_tickers]
                                    if skipped:
                                        st.warning(f"⚠ Skipped (no valid prices in window): {skipped}")

                                    if not valid_tickers:
                                        st.error("No tickers had valid prices in the backtest window.")
                                    else:
                                        use_common = entry_mode.startswith("📅")

                                        # ── Step 2: decide the common entry date if needed ─────
                                        if use_common:
                                            # Common start = latest first-available date across all tickers
                                            # This guarantees every stock has a real price on day 0
                                            common_entry_ts = max(ticker_first_valid[t] for t in valid_tickers)
                                            common_exit_ts  = min(ticker_last_valid[t]  for t in valid_tickers)

                                            if common_entry_ts >= common_exit_ts:
                                                st.error(
                                                    "No overlapping trading window exists across all tickers. "
                                                    "Try Staggered mode or remove tickers with very different listing dates."
                                                )
                                                st.stop()

                                            # Warn if common start is much later than requested window
                                            days_lost = (common_entry_ts.date() - _dt.date.fromisoformat(bt_start)).days
                                            if days_lost > 30:
                                                latest_t = max(valid_tickers, key=lambda t: ticker_first_valid[t])
                                                st.info(
                                                    f"ℹ Common start is **{common_entry_ts.strftime('%Y-%m-%d')}** "
                                                    f"because **{latest_t}** only has data from that date. "
                                                    f"{days_lost} days of earlier history are excluded to keep all stocks on equal footing."
                                                )

                                            # ── Resolve actual prices at common dates ──
                                            # Use .asof() which finds the last valid value AT or BEFORE
                                            # the target date — immune to timezone drift and sparse indices.
                                            def safe_price(series, target_ts):
                                                """Return the closest real price at or before target_ts."""
                                                clean = series.dropna()
                                                clean = clean[clean > 0]
                                                if clean.empty:
                                                    return float("nan")
                                                p = clean.asof(target_ts)
                                                if pd.isna(p):
                                                    # target_ts is before first data point; take first real price
                                                    p = float(clean.iloc[0])
                                                return float(p)

                                            entry_prices = {t: safe_price(bt_prices[t], common_entry_ts) for t in valid_tickers}
                                            exit_prices  = {t: safe_price(bt_prices[t], common_exit_ts)  for t in valid_tickers}

                                            # Resolve actual index labels closest to common dates
                                            def nearest_label(index, target_ts):
                                                pos = index.get_indexer([target_ts], method="nearest")[0]
                                                return index[pos]

                                            actual_entry_ts = nearest_label(bt_prices.index, common_entry_ts)
                                            actual_exit_ts  = nearest_label(bt_prices.index, common_exit_ts)

                                            entry_dates = {t: actual_entry_ts.strftime("%Y-%m-%d") for t in valid_tickers}
                                            exit_dates  = {t: actual_exit_ts.strftime("%Y-%m-%d")  for t in valid_tickers}
                                            overall_entry = actual_entry_ts.strftime("%Y-%m-%d")
                                            overall_exit  = actual_exit_ts.strftime("%Y-%m-%d")

                                            # Re-check: drop any ticker that still has NaN price after safe_price
                                            bad = [t for t in valid_tickers if pd.isna(entry_prices[t]) or pd.isna(exit_prices[t])]
                                            if bad:
                                                st.warning(f"⚠ Dropped tickers with no price on common dates: {bad}")
                                                valid_tickers = [t for t in valid_tickers if t not in bad]
                                                entry_prices  = {t: entry_prices[t] for t in valid_tickers}
                                                exit_prices   = {t: exit_prices[t]  for t in valid_tickers}
                                                entry_dates   = {t: entry_dates[t]  for t in valid_tickers}
                                                exit_dates    = {t: exit_dates[t]   for t in valid_tickers}
                                                if not valid_tickers:
                                                    st.error("No tickers had valid prices on the common entry/exit dates.")
                                                    st.stop()

                                        else:
                                            # Staggered: each stock enters on its own first day
                                            # Use safe_price / nearest_label for same robustness
                                            def safe_price(series, target_ts):
                                                clean = series.dropna()
                                                clean = clean[clean > 0]
                                                if clean.empty:
                                                    return float("nan")
                                                p = clean.asof(target_ts)
                                                if pd.isna(p):
                                                    p = float(clean.iloc[0])
                                                return float(p)

                                            entry_dates  = {t: ticker_first_valid[t].strftime("%Y-%m-%d") for t in valid_tickers}
                                            exit_dates   = {t: ticker_last_valid[t].strftime("%Y-%m-%d")  for t in valid_tickers}
                                            entry_prices = {t: safe_price(bt_prices[t], ticker_first_valid[t]) for t in valid_tickers}
                                            exit_prices  = {t: safe_price(bt_prices[t], ticker_last_valid[t])  for t in valid_tickers}

                                            bad = [t for t in valid_tickers if pd.isna(entry_prices[t]) or pd.isna(exit_prices[t])]
                                            if bad:
                                                st.warning(f"⚠ Dropped tickers with no resolvable price: {bad}")
                                                valid_tickers = [t for t in valid_tickers if t not in bad]
                                                entry_prices  = {t: entry_prices[t] for t in valid_tickers}
                                                exit_prices   = {t: exit_prices[t]  for t in valid_tickers}
                                                entry_dates   = {t: entry_dates[t]  for t in valid_tickers}
                                                exit_dates    = {t: exit_dates[t]   for t in valid_tickers}
                                                if not valid_tickers:
                                                    st.error("No tickers had valid prices.")
                                                    st.stop()

                                            all_entry_ts = sorted(ticker_first_valid[t] for t in valid_tickers)
                                            all_exit_ts  = sorted(ticker_last_valid[t]  for t in valid_tickers)
                                            overall_entry = all_entry_ts[0].strftime("%Y-%m-%d")
                                            overall_exit  = all_exit_ts[-1].strftime("%Y-%m-%d")

                                        valid_weights_raw = np.array([weights_bt[tickers_bt.index(t)] for t in valid_tickers])
                                        valid_weights     = valid_weights_raw / valid_weights_raw.sum()

                                        # ── Step 3: allocate capital → shares ──────
                                        alloc = {t: capital * w for t, w in zip(valid_tickers, valid_weights)}
                                        if allow_fractional:
                                            shares   = {t: alloc[t] / entry_prices[t] for t in valid_tickers}
                                            residual = 0.0
                                        else:
                                            shares   = {t: float(int(alloc[t] / entry_prices[t])) for t in valid_tickers}
                                            residual = sum(alloc[t] - shares[t] * entry_prices[t] for t in valid_tickers)

                                        # ── Step 4: exit values & portfolio totals ──
                                        exit_vals    = {t: shares[t] * exit_prices[t] for t in valid_tickers}
                                        total_exit   = sum(exit_vals.values()) + residual
                                        total_invest = capital
                                        abs_return   = total_exit - total_invest
                                        pct_return   = abs_return / total_invest * 100

                                        n_days  = (
                                            _dt.date.fromisoformat(overall_exit) -
                                            _dt.date.fromisoformat(overall_entry)
                                        ).days
                                        n_years = n_days / 365.25
                                        cagr    = ((total_exit / total_invest) ** (1 / max(n_years, 0.01)) - 1) * 100

                                        # ── Step 5: trade blotter ──────────────────
                                        rows = []
                                        for t in valid_tickers:
                                            ep  = entry_prices[t]
                                            xp  = exit_prices[t]
                                            sh  = shares[t]
                                            inv = sh * ep
                                            val = sh * xp
                                            g   = val - inv
                                            hold_days = (
                                                _dt.date.fromisoformat(exit_dates[t]) -
                                                _dt.date.fromisoformat(entry_dates[t])
                                            ).days
                                            rows.append({
                                                "Ticker":           t,
                                                "Weight %":         round(valid_weights[valid_tickers.index(t)] * 100, 2),
                                                "Buy Date":         entry_dates[t],
                                                "Buy Price":        round(ep, 2),
                                                "Shares Bought":    round(sh, 4),
                                                "Capital Deployed": round(inv, 2),
                                                "Sell Date":        exit_dates[t],
                                                "Sell Price":       round(xp, 2),
                                                "Current Value":    round(val, 2),
                                                "Gain / Loss":      round(g, 2),
                                                "Hold (days)":      hold_days,
                                                "Return %":         round((xp - ep) / ep * 100, 2),
                                            })
                                        bt_df = pd.DataFrame(rows)

                                        # ── Step 6: daily portfolio value curve ────
                                        # For common mode: slice to [common_entry, common_exit], ffill gaps
                                        # For staggered:   each ticker contributes from its own entry onward
                                        if use_common:
                                            window = bt_prices.loc[common_entry_ts:common_exit_ts].copy()
                                            window = window[valid_tickers].ffill()
                                            daily_vals = pd.Series(0.0, index=window.index)
                                            for t in valid_tickers:
                                                daily_vals += shares[t] * window[t]
                                            daily_vals += residual
                                        else:
                                            daily_vals = pd.Series(0.0, index=bt_prices.index)
                                            for t in valid_tickers:
                                                col_raw   = bt_prices[t].copy()
                                                e_idx     = bt_prices.index.get_loc(ticker_first_valid[t])
                                                col_slice = col_raw.iloc[e_idx:].ffill()
                                                daily_vals.iloc[e_idx:e_idx + len(col_slice)] += shares[t] * col_slice.values
                                            daily_vals = daily_vals.replace(0.0, np.nan).ffill().bfill()
                                            daily_vals += residual

                                        # ── Summary metrics ────────────────────────
                                        st.markdown(f"""
                                        <div class="metric-row">
                                            {metric_card("Capital Invested", f"₹{total_invest:,.0f}", "accent")}
                                            {metric_card("Portfolio Value",  f"₹{total_exit:,.0f}",   "positive" if pct_return>0 else "")}
                                            {metric_card("Absolute Return",  f"₹{abs_return:+,.0f}",  "positive" if abs_return>0 else "")}
                                            {metric_card("Total Return",     f"{pct_return:+.2f}%",   "positive" if pct_return>0 else "")}
                                            {metric_card("CAGR",             f"{cagr:+.2f}%",          "positive" if cagr>0 else "")}
                                            {metric_card("Period",           f"{n_days}d / {n_years:.1f}y", "accent")}
                                        </div>""", unsafe_allow_html=True)

                                        st.caption(
                                            f"Earliest buy: **{overall_entry}**  ·  Latest sell: **{overall_exit}**  ·  "
                                            f"{'Fractional' if allow_fractional else 'Whole'} shares  ·  "
                                            f"Uninvested cash: ₹{residual:,.2f}"
                                        )

                                        # ── Trade blotter table ────────────────────
                                        st.markdown('<div class="section-heading">Trade Blotter — Buy & Sell Detail</div>', unsafe_allow_html=True)
                                        st.dataframe(
                                            bt_df.style
                                                .background_gradient(cmap="RdYlGn", subset=["Return %", "Gain / Loss"])
                                                .format({
                                                    "Capital Deployed": "₹{:,.2f}",
                                                    "Current Value":    "₹{:,.2f}",
                                                    "Gain / Loss":      "₹{:+,.2f}",
                                                    "Buy Price":        "{:,.2f}",
                                                    "Sell Price":       "{:,.2f}",
                                                    "Shares Bought":    "{:.4f}",
                                                    "Return %":         "{:+.2f}%",
                                                }),
                                            use_container_width=True,
                                            height=min(400, 60 + 35 * len(bt_df))
                                        )

                                        # ── Portfolio value over time chart ────────
                                        st.markdown('<div class="section-heading">Portfolio Value Over Time</div>', unsafe_allow_html=True)
                                        fig_bt = go.Figure()
                                        fig_bt.add_trace(go.Scatter(
                                            x=daily_vals.index, y=daily_vals.values,
                                            mode='lines', name="Portfolio Value",
                                            fill='tozeroy',
                                            line=dict(color="#00b4ff", width=2),
                                            fillcolor="rgba(0,180,255,0.07)",
                                            hovertemplate="%{x|%Y-%m-%d}<br>₹%{y:,.2f}<extra></extra>"
                                        ))
                                        fig_bt.add_hline(
                                            y=capital, line_dash="dash",
                                            line_color="rgba(255,213,79,0.5)",
                                            annotation_text="Invested Capital",
                                            annotation_font_color="#ffd54f"
                                        )
                                        fig_bt.update_layout(
                                            template="plotly_dark", paper_bgcolor="#0a0e17",
                                            plot_bgcolor="#0d1525",
                                            xaxis_title="Date", yaxis_title="Portfolio Value (₹)",
                                            title=dict(text=f"Backtest: {selected_portfolio}", font=dict(color="#a0c8e8")),
                                            font=dict(family="DM Mono", color="#80b0d0"),
                                            height=400, margin=dict(l=50, r=20, t=50, b=50)
                                        )
                                        st.plotly_chart(fig_bt, use_container_width=True)

                                        # ── Per-stock return bar chart ─────────────
                                        st.markdown('<div class="section-heading">Per-Stock Return %</div>', unsafe_allow_html=True)
                                        bt_df_sorted = bt_df.sort_values("Return %", ascending=False)
                                        colors_bar   = ["#00e676" if r > 0 else "#ff5252" for r in bt_df_sorted["Return %"]]
                                        fig_bar = go.Figure(go.Bar(
                                            x=bt_df_sorted["Ticker"], y=bt_df_sorted["Return %"],
                                            marker_color=colors_bar,
                                            text=[f"{r:+.1f}%" for r in bt_df_sorted["Return %"]],
                                            textposition="outside",
                                            hovertemplate="<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>"
                                        ))
                                        fig_bar.update_layout(
                                            template="plotly_dark", paper_bgcolor="#0a0e17",
                                            plot_bgcolor="#0d1525",
                                            xaxis_title="", yaxis_title="Return %",
                                            font=dict(family="DM Mono", color="#80b0d0"),
                                            height=320, margin=dict(l=50, r=20, t=30, b=60),
                                            uniformtext_minsize=7, uniformtext_mode="hide"
                                        )
                                        fig_bar.add_hline(y=0, line_color="rgba(255,255,255,0.15)")
                                        st.plotly_chart(fig_bar, use_container_width=True)

                                        # ── Persist results for PDF export ───────
                                        st.session_state.bt_results = {
                                            "portfolio_name": selected_portfolio,
                                            "capital":        capital,
                                            "entry_mode":     "Common Start" if use_common else "Staggered",
                                            "overall_entry":  overall_entry,
                                            "overall_exit":   overall_exit,
                                            "n_days":         n_days,
                                            "n_years":        n_years,
                                            "total_invest":   total_invest,
                                            "total_exit":     total_exit,
                                            "abs_return":     abs_return,
                                            "pct_return":     pct_return,
                                            "cagr":           cagr,
                                            "residual":       residual,
                                            "allow_fractional": allow_fractional,
                                            "blotter":        bt_df,
                                        }

                                        bt_csv = bt_df.to_csv(index=False).encode()
                                        st.download_button(
                                            "⬇ Download Trade Blotter CSV", data=bt_csv,
                                            file_name="quansen_backtest.csv", mime="text/csv",
                                            use_container_width=True
                                        )

                    _saved_bt = st.session_state.get("bt_results") or {}
                    _saved_deploy = _saved_bt.get("deployable") or {}
                    if _saved_deploy and _saved_bt.get("entry_mode") == "Walk-Forward Adaptive":
                        st.markdown("---")
                        render_insight_note(
                            "Latest walk-forward deploy-now weights are saved from the most recent run. "
                            "Use the button below to jump straight to the Portfolios view with that mix loaded."
                        )
                        if st.button("View Weights In Portfolio View", key="use_deployable_weights_saved_btn", use_container_width=True):
                            _promote_deployable_weights_to_portfolios(_saved_deploy)
                            st.rerun()


    

# ══════════════════════════════════════════════════════════════
# TAB 7 — MONTE CARLO SIMULATOR  (both modes)
# ══════════════════════════════════════════════════════════════
if _qs_s == "mc":
    import datetime as _mc_dt
    from engines.mc_sim import (
        run_mc_simulation, compute_mc_stats,
        build_histogram_data, sample_paths_for_plot,
        horizon_to_years,
        run_regime_switching_simulation,
        build_transition_matrix, summarize_regime_paths,
        run_all_stress_scenarios, compare_scenario_stats,
        STRESS_SCENARIOS, REGIME_DEFAULTS, REGIME_ORDER,
    )
    # Sub-tabs: Standard GBM | Regime-Switching | Stress Scenarios
    if is_noob:
        _mc_subtabs = st.tabs(["🎲 Simulator", "🌡 Stress Test"])
        _mc_sub_gbm, _mc_sub_stress = _mc_subtabs
        _mc_sub_regime = None
    else:
        _mc_subtabs = st.tabs(["📊 Standard GBM", "🔄 Regime-Switching", "💥 Stress Scenarios"])
        _mc_sub_gbm, _mc_sub_regime, _mc_sub_stress = _mc_subtabs

    # ── Wrap all GBM content in the first sub-tab ────────────────
    with _mc_sub_gbm:
      # ── Noob mode: friendly wrapper ──────────────────────────────
      if is_noob:
        st.markdown(
            '''<div style="background:linear-gradient(135deg,#031a0e,#0a2d14);
                border:1px solid #1a5c2a;border-radius:16px;padding:1.2rem 1.6rem;
                margin-bottom:1.2rem;box-shadow:0 4px 24px rgba(0,180,60,0.12);">
                <div style="font-size:0.6rem;letter-spacing:0.22em;text-transform:uppercase;color:#3db85a;margin-bottom:4px">
                    QuanSen · Easy Mode
                </div>
                <div style="font-size:1.3rem;font-weight:800;color:#4afa7a;font-family:Syne,sans-serif;">
                    🎲 What-If Simulator
                </div>
                <div style="font-size:0.78rem;color:#3db85a;margin-top:4px">
                    See thousands of possible futures for your money
                </div>
            </div>''',
            unsafe_allow_html=True,
        )
        st.markdown(
            '''<div style="background:rgba(10,40,18,0.7);border:1px solid #1a5c2a;
                border-radius:12px;padding:0.9rem 1.1rem;margin-bottom:1rem;font-size:0.82rem;color:#7dd89a;">
                💡 <strong style="color:#c8ffd4">How it works:</strong>
                We run thousands of "coin-flip" futures for your portfolio. Some go well, some don't.
                The spread shows the range of realistic outcomes. The gold line is the most likely result.
            </div>''',
            unsafe_allow_html=True,
        )

      # ── Quant-mode header ──────────────────────────────────────
      else:
        st.markdown('<div class="section-heading">Monte Carlo Wealth Simulation — GBM Engine</div>', unsafe_allow_html=True)
        render_insight_note(
            "GBM paths are drawn from the portfolio's own annualised μ and σ. "
            "Select any computed portfolio as the base — the engine uses that portfolio's exact return and risk profile. "
            "Contributions are injected at each annual anniversary step."
        )

    # ── Portfolio selector ───────────────────────────────────────
    # Helper: safely compute portfolio stats, guarding against stale weight
    # vectors whose length no longer matches the current er / cov after a
    # ticker was dropped.  Returns (ann_ret, ann_risk) or None on mismatch.
    def _safe_port_stats(weights, er_series, cov_df):
        try:
            n_er  = len(er_series)
            n_cov = len(cov_df)
            n_w   = len(weights)
            if n_w != n_er or n_w != n_cov:
                # Trim weights to the common length (optimizer may have
                # dropped one ticker that failed to download).
                _min = min(n_w, n_er, n_cov)
                weights   = weights[:_min]
                er_series = er_series.iloc[:_min]
                cov_df    = cov_df.iloc[:_min, :_min]
                total = weights.sum()
                if total > 0:
                    weights = weights / total
            from engines.data_and_compute import cached_portfolio_stats as _cps_inner
            ann_ret, ann_risk, sharpe = _cps_inner(weights, er_series, cov_df)
            return float(ann_ret), float(ann_risk)
        except Exception:
            return None, None

    _mc_er_now  = optimizer_expected_returns()
    _mc_cov_now = st.session_state.cov_matrix

    _mc_port_opts = {}
    if st.session_state.get("weights_tan") is not None:
        # Tangency stats are already stored — use them directly, no matmul needed
        _r = st.session_state.tan_return
        _k = st.session_state.tan_risk
        if _r is not None and _k is not None:
            _mc_port_opts[f"Tangency  (μ={_r*100:.1f}%  σ={_k*100:.1f}%)"] = ("tan", _r, _k)
    if st.session_state.get("weights_utility") is not None:
        _r, _k = _safe_port_stats(st.session_state.weights_utility, _mc_er_now, _mc_cov_now)
        if _r is not None:
            _mc_port_opts[f"Utility   (μ={_r*100:.1f}%  σ={_k*100:.1f}%)"] = ("utility", _r, _k)
    if st.session_state.get("weights_min") is not None:
        _r, _k = _safe_port_stats(st.session_state.weights_min, _mc_er_now, _mc_cov_now)
        if _r is not None:
            _mc_port_opts[f"Min-Risk  (μ={_r*100:.1f}%  σ={_k*100:.1f}%)"] = ("min", _r, _k)
    if st.session_state.get("weights_deploy") is not None:
        _r, _k = _safe_port_stats(st.session_state.weights_deploy, _mc_er_now, _mc_cov_now)
        if _r is not None:
            _deploy_meta = st.session_state.get("deployable_meta") or {}
            _deploy_posture = str(_deploy_meta.get("posture") or "Deploy-Now").strip()
            _mc_port_opts[f"Deploy-Now [{_deploy_posture}]  (μ={_r*100:.1f}%  σ={_k*100:.1f}%)"] = ("deploy", _r, _k)

    _mc_has_portfolio = bool(_mc_port_opts)

    if not st.session_state.get("data_loaded"):
        st.markdown('<div class="status-box status-info">Load market data and run at least one optimisation first.</div>', unsafe_allow_html=True)
        st.stop()

    if not _mc_has_portfolio:
        st.markdown('<div class="status-box status-warn">Run at least one portfolio optimisation to seed the simulator.</div>', unsafe_allow_html=True)
        st.stop()

    # ── Config panel ─────────────────────────────────────────────
    _mc_cfg_col, _mc_res_col = st.columns([1, 2.2], gap="large")

    with _mc_cfg_col:
        if is_noob:
            st.markdown('<div class="nb-green-heading">⚙ Setup</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card-title">Simulation Settings</div>', unsafe_allow_html=True)

        # Portfolio pick
        _mc_port_label = st.selectbox(
            "Base portfolio",
            list(_mc_port_opts.keys()),
            key="mc_port_select",
            help="The annualised μ and σ of the selected portfolio seed the GBM.",
        )
        _, _mc_mu, _mc_sigma = _mc_port_opts[_mc_port_label]
        if _mc_port_label.startswith("Deploy-Now"):
            st.caption("Deploy-Now uses the latest walk-forward adjusted mix from the Backtest tab.")

        # Manual override toggle (quant only)
        _mc_override = False
        if not is_noob:
            _mc_override = st.toggle(
                "Override μ / σ manually",
                value=False,
                key="mc_override_toggle",
                help="Plug in any return / risk assumptions, independent of the optimizer output.",
            )

        if _mc_override:
            _mc_mu_input    = st.number_input("Annual return μ (%)", min_value=-50.0, max_value=200.0,
                                              value=round(_mc_mu * 100, 2), step=0.1, key="mc_mu_inp")
            _mc_sigma_input = st.number_input("Annual volatility σ (%)", min_value=0.1, max_value=200.0,
                                              value=round(_mc_sigma * 100, 2), step=0.1, key="mc_sig_inp")
            _mc_mu    = _mc_mu_input    / 100.0
            _mc_sigma = _mc_sigma_input / 100.0

        st.markdown("---")

        # Capital
        _mc_capital = st.number_input(
            "Initial capital (₹ / $)",
            min_value=1_000.0,
            max_value=1_000_000_000.0,
            value=float(st.session_state.get("bt_capital", 100_000.0)),
            step=1_000.0,
            format="%.0f",
            key="mc_capital",
        )

        # Annual contribution
        _mc_contrib = st.number_input(
            "Annual top-up (₹ / $)",
            min_value=0.0,
            max_value=100_000_000.0,
            value=0.0,
            step=1_000.0,
            format="%.0f",
            key="mc_contrib",
            help="Added at each year anniversary. Set 0 to disable.",
        )

        st.markdown("---")

        # Horizon
        _mc_h_unit = st.radio(
            "Horizon unit",
            ["months", "years"],
            index=1,
            horizontal=True,
            key="mc_h_unit",
        )
        if _mc_h_unit == "months":
            _mc_h_val = st.slider("Horizon (months)", 1, 60, 36, 1, key="mc_h_months")
            _mc_horizon_years = _mc_h_val / 12.0
            _mc_horizon_label = f"{_mc_h_val} month{'s' if _mc_h_val != 1 else ''}"
        else:
            _mc_h_val = st.slider("Horizon (years)", 1, 40, 10, 1, key="mc_h_years")
            _mc_horizon_years = float(_mc_h_val)
            _mc_horizon_label = f"{_mc_h_val} year{'s' if _mc_h_val != 1 else ''}"

        # Number of simulations
        _mc_nsims = st.select_slider(
            "Simulations",
            options=[100, 250, 500, 1000, 2500, 5000, 10000],
            value=5000,
            key="mc_nsims",
            help="More simulations = more precise distribution but slower render.",
        )

        # Seed toggle (quant only)
        _mc_seed = None
        if not is_noob:
            if st.toggle("Fix random seed", value=False, key="mc_seed_toggle",
                         help="Reproducible run — same seed gives same paths."):
                _mc_seed = st.number_input("Seed value", min_value=0, max_value=99999,
                                            value=42, step=1, key="mc_seed_val")
        _mc_light_visuals = st.toggle(
            "Light visuals (faster)",
            value=True,
            key="mc_light_visuals",
            help="Draw fewer paths and lighter histograms for faster rendering.",
        )

        # Run button
        _mc_run = st.button(
            "▶ Run Simulation" if not is_noob else "🚀 Show My Futures!",
            use_container_width=True,
            key="mc_run_btn",
        )

        # Live stats for selected portfolio
        st.markdown("---")
        if is_noob:
            st.markdown('<div class="nb-green-heading">📌 Your Portfolio Stats</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem">Seeded Parameters</div>', unsafe_allow_html=True)

        st.markdown(
            f'''<div style="background:rgba(0,180,255,0.05);border:0.5px solid #1a2d4d;
                border-radius:10px;padding:0.8rem 1rem;font-family:'DM Mono',monospace;font-size:0.8rem;">
                <div style="color:#4a6a90;margin-bottom:4px;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em">GBM inputs</div>
                <div style="color:#a0c8e8">μ  =  <span style="color:#00e676;font-weight:600">{_mc_mu*100:.2f}%</span> / yr</div>
                <div style="color:#a0c8e8">σ  =  <span style="color:#ff8a65;font-weight:600">{_mc_sigma*100:.2f}%</span> / yr</div>
                <div style="color:#a0c8e8">T  =  <span style="color:#ffd54f;font-weight:600">{_mc_horizon_label}</span></div>
                <div style="color:#a0c8e8">N  =  <span style="color:#80c8ff;font-weight:600">{_mc_nsims:,}</span> paths</div>
            </div>''',
            unsafe_allow_html=True,
        )

    # ── Results panel ────────────────────────────────────────────
    with _mc_res_col:
        if _mc_run:
            with st.spinner(f"Running {_mc_nsims:,} simulations over {_mc_horizon_label}…"):

                _mc_paths, _mc_t = run_mc_simulation(
                    mu=_mc_mu,
                    sigma=_mc_sigma,
                    initial_capital=_mc_capital,
                    horizon_years=_mc_horizon_years,
                    n_sims=_mc_nsims,
                    annual_contribution=_mc_contrib,
                    seed=_mc_seed,
                )
                _mc_stats = compute_mc_stats(
                    _mc_paths, _mc_capital,
                    annual_contribution=_mc_contrib,
                    horizon_years=_mc_horizon_years,
                )

            # Persist results in session for the charts below
            st.session_state["mc_last_paths"]  = _mc_paths
            st.session_state["mc_last_t"]      = _mc_t
            st.session_state["mc_last_stats"]  = _mc_stats
            st.session_state["mc_last_capital"] = _mc_capital
            st.session_state["mc_last_horizon"] = _mc_horizon_label
            st.session_state["mc_last_nsims"]   = _mc_nsims

        # Render results if we have them
        if st.session_state.get("mc_last_stats") is not None:
            _st      = st.session_state["mc_last_stats"]
            _paths   = st.session_state["mc_last_paths"]
            _t       = st.session_state["mc_last_t"]
            _cap     = st.session_state["mc_last_capital"]
            _hor_lbl = st.session_state["mc_last_horizon"]
            _nsims_d = st.session_state["mc_last_nsims"]
            _mc_light = bool(st.session_state.get("mc_light_visuals", True))

            # ── Summary metric cards ────────────────────────────────
            _mc_v_median = "₹{:,.0f}".format(_st['median'])
            _mc_v_mean   = "₹{:,.0f}".format(_st['mean'])
            _mc_v_p95    = "₹{:,.0f}".format(_st['p95'])
            _mc_v_p5     = "₹{:,.0f}".format(_st['p5'])
            _mc_v_cagr   = "{:.2f}%".format(_st['cagr_median'] * 100)
            _mc_v_var    = "{:.1f}%".format(_st['var_95'])
            _mc_cagr_cls = "positive" if _st["cagr_median"] > 0 else ""
            st.markdown(
                "<div class=\"metric-row\">"
                + metric_card("Median Outcome",   _mc_v_median, "accent")
                + metric_card("Mean Outcome",      _mc_v_mean,   "")
                + metric_card("Optimistic (95%)",  _mc_v_p95,    "positive")
                + metric_card("Pessimistic (5%)",  _mc_v_p5,     "")
                + metric_card("Median CAGR",       _mc_v_cagr,   _mc_cagr_cls)
                + metric_card("VaR 95%",           _mc_v_var,    "")
                + "</div>",
                unsafe_allow_html=True,
            )

            # ── Probability bands ───────────────────────────────────
            _col_p1, _col_p2, _col_p3, _col_p4 = st.columns(4)
            _prob_items = [
                (_col_p1, "P(profit)",     f"{_st['prob_profit']*100:.1f}%",  "#00e676"),
                (_col_p2, "P(2× initial)", f"{_st['prob_double']*100:.1f}%",  "#ffd54f"),
                (_col_p3, "P(any loss)",   f"{_st['prob_loss']*100:.1f}%",    "#ff8a65"),
                (_col_p4, "P(<50% left)",  f"{_st['prob_halved']*100:.1f}%",  "#ff5252"),
            ]
            for _col, _lbl, _val, _col_c in _prob_items:
                with _col:
                    st.markdown(
                        f'''<div style="background:rgba(10,20,40,0.6);border:0.5px solid #1a2d4d;
                            border-radius:10px;padding:0.7rem 1rem;text-align:center;">
                            <div style="font-size:0.62rem;color:#4a6a90;text-transform:uppercase;
                                        letter-spacing:0.1em;margin-bottom:4px">{_lbl}</div>
                            <div style="font-size:1.35rem;font-weight:700;color:{_col_c};
                                        font-family:'DM Mono',monospace">{_val}</div>
                        </div>''',
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

            # ── Fan chart — wealth paths ────────────────────────────
            st.markdown('<div class="section-heading">Simulated Wealth Paths</div>', unsafe_allow_html=True)
            if not is_noob:
                render_insight_note(
                    "Each thin line is one simulated portfolio path. Percentile bands show the envelope. "
                    "The gold line is the median outcome — half of all paths finish above it."
                )

            # Sample 200 paths for display
            _disp_paths, _ = sample_paths_for_plot(_paths, _t, n_sample=60 if _mc_light else 200)

            fig_fan = go.Figure()

            # Thin path traces
            _n_disp = min(60 if _mc_light else 200, _disp_paths.shape[1])
            for _pi in range(_n_disp):
                _path_col = "rgba(0,120,220,0.08)" if _disp_paths[-1, _pi] >= _cap else "rgba(220,60,60,0.06)"
                fig_fan.add_trace(go.Scatter(
                    x=_t, y=_disp_paths[:, _pi],
                    mode="lines",
                    line=dict(width=0.6, color=_path_col),
                    showlegend=False,
                    hoverinfo="skip",
                ))

            # Percentile band fills
            _p5_path   = np.percentile(_paths, 5,  axis=1)
            _p25_path  = np.percentile(_paths, 25, axis=1)
            _p75_path  = np.percentile(_paths, 75, axis=1)
            _p95_path  = np.percentile(_paths, 95, axis=1)
            _med_path  = np.percentile(_paths, 50, axis=1)

            fig_fan.add_trace(go.Scatter(
                x=np.concatenate([_t, _t[::-1]]),
                y=np.concatenate([_p95_path, _p5_path[::-1]]),
                fill="toself", fillcolor="rgba(0,120,200,0.07)",
                line=dict(width=0), name="90% range", showlegend=True,
                hoverinfo="skip",
            ))
            fig_fan.add_trace(go.Scatter(
                x=np.concatenate([_t, _t[::-1]]),
                y=np.concatenate([_p75_path, _p25_path[::-1]]),
                fill="toself", fillcolor="rgba(0,180,255,0.12)",
                line=dict(width=0), name="50% range", showlegend=True,
                hoverinfo="skip",
            ))
            fig_fan.add_trace(go.Scatter(
                x=_t, y=_med_path, mode="lines",
                line=dict(color="gold", width=2.5), name="Median",
            ))
            fig_fan.add_trace(go.Scatter(
                x=_t, y=_p95_path, mode="lines",
                line=dict(color="#00e676", width=1.5, dash="dot"), name="95th pctile",
            ))
            fig_fan.add_trace(go.Scatter(
                x=_t, y=_p5_path, mode="lines",
                line=dict(color="#ff5252", width=1.5, dash="dot"), name="5th pctile",
            ))
            # Invested capital line
            fig_fan.add_hline(
                y=_st["total_invested"],
                line_dash="dash", line_color="rgba(255,213,79,0.4)",
                annotation_text=f"Total invested  ₹{_st['total_invested']:,.0f}",
                annotation_font_color="#ffd54f",
            )

            fig_fan.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a0e17",
                plot_bgcolor="#0d1525",
                xaxis_title=f"Time (years)  ·  horizon = {_hor_lbl}",
                yaxis_title="Portfolio Value (₹)",
                title=dict(text=f"Monte Carlo Fan Chart  ·  {_nsims_d:,} simulations",
                           font=dict(color="#a0c8e8")),
                font=dict(family="DM Mono", color="#80b0d0"),
                legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0.4)"),
                height=480,
                margin=dict(l=60, r=20, t=55, b=55),
            )
            st.plotly_chart(fig_fan, use_container_width=True)

            # ── Final value distribution histogram ──────────────────
            st.markdown('<div class="section-heading">Final Wealth Distribution</div>', unsafe_allow_html=True)
            if not is_noob:
                render_insight_note(
                    "The histogram of terminal portfolio values. The green zone is above your total invested capital; "
                    "the red zone is below. Skew to the right is normal under GBM — downside is bounded, upside is not."
                )

            _centres, _counts = build_histogram_data(_st["finals"], n_bins=40 if _mc_light else 80)

            if len(_centres) > 0:
                _bar_colors = [
                    "#00e676" if c >= _cap else "#ff5252"
                    for c in _centres
                ]
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Bar(
                    x=_centres, y=_counts,
                    marker_color=_bar_colors,
                    marker_line_width=0,
                    hovertemplate="Value: ₹%{x:,.0f}<br>Count: %{y}<extra></extra>",
                    name="Frequency",
                ))
                # Percentile lines
                for _pv, _pc, _pl in [
                    (_st["p5"],  "#ff5252", "5th"),
                    (_st["p25"], "#ff8a65", "25th"),
                    (_st["median"], "gold", "Median"),
                    (_st["p75"], "#b9f6ca", "75th"),
                    (_st["p95"], "#00e676", "95th"),
                ]:
                    fig_hist.add_vline(
                        x=_pv, line_dash="dot",
                        line_color=_pc,
                        annotation_text=f"{_pl} ₹{_pv:,.0f}",
                        annotation_font_color=_pc,
                        annotation_font_size=10,
                    )
                fig_hist.add_vline(
                    x=_cap, line_dash="dash",
                    line_color="rgba(255,213,79,0.5)",
                    annotation_text="Initial capital",
                    annotation_font_color="#ffd54f",
                )
                fig_hist.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0a0e17",
                    plot_bgcolor="#0d1525",
                    xaxis_title="Terminal Portfolio Value (₹)  [log scale]",
                    yaxis_title="Number of simulations",
                    xaxis_type="log",
                    title=dict(text="Distribution of Terminal Wealth",
                               font=dict(color="#a0c8e8")),
                    font=dict(family="DM Mono", color="#80b0d0"),
                    showlegend=False,
                    height=380,
                    margin=dict(l=60, r=20, t=50, b=55),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # ── Percentile table ────────────────────────────────────
            if not is_noob:
                st.markdown('<div class="section-heading">Percentile Table</div>', unsafe_allow_html=True)
                _pct_df = pd.DataFrame({
                    "Percentile": ["5th", "10th", "25th", "50th (Median)", "75th", "90th", "95th"],
                    "Terminal Value (₹)": [
                        f"₹{_st['p5']:,.0f}",
                        f"₹{_st['p10']:,.0f}",
                        f"₹{_st['p25']:,.0f}",
                        f"₹{_st['median']:,.0f}",
                        f"₹{_st['p75']:,.0f}",
                        f"₹{_st['p90']:,.0f}",
                        f"₹{_st['p95']:,.0f}",
                    ],
                    "Return vs Initial": [
                        f"{(_st['p5']-_cap)/_cap*100:+.1f}%",
                        f"{(_st['p10']-_cap)/_cap*100:+.1f}%",
                        f"{(_st['p25']-_cap)/_cap*100:+.1f}%",
                        f"{(_st['median']-_cap)/_cap*100:+.1f}%",
                        f"{(_st['p75']-_cap)/_cap*100:+.1f}%",
                        f"{(_st['p90']-_cap)/_cap*100:+.1f}%",
                        f"{(_st['p95']-_cap)/_cap*100:+.1f}%",
                    ],
                    "CAGR": [
                        f"{(max(_st['p5'],0.01)/_cap)**(1/max(_mc_horizon_years,0.001))-1:.2%}",
                        f"{(max(_st['p10'],0.01)/_cap)**(1/max(_mc_horizon_years,0.001))-1:.2%}",
                        f"{(max(_st['p25'],0.01)/_cap)**(1/max(_mc_horizon_years,0.001))-1:.2%}",
                        f"{(max(_st['median'],0.01)/_cap)**(1/max(_mc_horizon_years,0.001))-1:.2%}",
                        f"{(max(_st['p75'],0.01)/_cap)**(1/max(_mc_horizon_years,0.001))-1:.2%}",
                        f"{(max(_st['p90'],0.01)/_cap)**(1/max(_mc_horizon_years,0.001))-1:.2%}",
                        f"{(max(_st['p95'],0.01)/_cap)**(1/max(_mc_horizon_years,0.001))-1:.2%}",
                    ],
                }).set_index("Percentile")
                st.dataframe(_pct_df, use_container_width=True)

                # ── CVaR / Risk summary ─────────────────────────────
                st.markdown('<div class="section-heading">Risk Analytics</div>', unsafe_allow_html=True)
                _ra_c1, _ra_c2, _ra_c3 = st.columns(3)
                with _ra_c1:
                    st.markdown(
                        f'''<div style="background:rgba(255,82,82,0.08);border:0.5px solid #3d1515;
                            border-radius:10px;padding:0.9rem 1rem;">
                            <div style="font-size:0.65rem;color:#ff8a65;text-transform:uppercase;letter-spacing:0.1em">VaR 95%</div>
                            <div style="font-size:1.4rem;font-weight:700;color:#ff5252;font-family:'DM Mono',monospace">{_st['var_95']:.1f}%</div>
                            <div style="font-size:0.68rem;color:#9e4a4a">Worst-case return at 95% confidence</div>
                        </div>''',
                        unsafe_allow_html=True,
                    )
                with _ra_c2:
                    st.markdown(
                        f'''<div style="background:rgba(255,82,82,0.08);border:0.5px solid #3d1515;
                            border-radius:10px;padding:0.9rem 1rem;">
                            <div style="font-size:0.65rem;color:#ff8a65;text-transform:uppercase;letter-spacing:0.1em">CVaR 95%</div>
                            <div style="font-size:1.4rem;font-weight:700;color:#ff5252;font-family:'DM Mono',monospace">{_st['cvar_95']:.1f}%</div>
                            <div style="font-size:0.68rem;color:#9e4a4a">Expected loss in worst 5% of scenarios</div>
                        </div>''',
                        unsafe_allow_html=True,
                    )
                with _ra_c3:
                    st.markdown(
                        f'''<div style="background:rgba(0,180,255,0.06);border:0.5px solid #1a2d4d;
                            border-radius:10px;padding:0.9rem 1rem;">
                            <div style="font-size:0.65rem;color:#4a90d9;text-transform:uppercase;letter-spacing:0.1em">Std Dev (terminal)</div>
                            <div style="font-size:1.4rem;font-weight:700;color:#80c8ff;font-family:'DM Mono',monospace">₹{_st['std']:,.0f}</div>
                            <div style="font-size:0.68rem;color:#3a6a90">Spread of final wealth values</div>
                        </div>''',
                        unsafe_allow_html=True,
                    )

                # ── Export ──────────────────────────────────────────
                st.markdown("---")
                _mc_finals_df = pd.DataFrame({"final_wealth": _st["finals"]})
                st.download_button(
                    "⬇ Download Final Wealth Distribution CSV",
                    data=_mc_finals_df.to_csv(index=False).encode(),
                    file_name="quansen_mc_finals.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="mc_dl_finals",
                )

        else:
            # No simulation run yet — show idle state
            st.markdown(
                '''<div style="padding:4rem 2rem;text-align:center;color:#263a56;">
                    <div style="font-size:3rem;margin-bottom:1rem">🎲</div>
                    <div style="font-size:1.1rem;color:#4a6a90;margin-bottom:0.5rem">
                        Configure the simulation settings and hit <strong>Run</strong>
                    </div>
                    <div style="font-size:0.82rem;color:#1a2d4d">
                        The GBM engine will simulate thousands of wealth paths<br>
                        using your optimized portfolio's μ and σ as inputs.
                    </div>
                </div>''',
                unsafe_allow_html=True,
            )
    # ═══ end _mc_sub_gbm ═══════════════════════════════════════

    # ══════════════════════════════════════════════════════════
    # REGIME-SWITCHING SUB-TAB  (quant only)
    # ══════════════════════════════════════════════════════════
    if not is_noob and _mc_sub_regime is not None:
      with _mc_sub_regime:
        st.markdown('<div class="section-heading">Regime-Switching Monte Carlo — Market Story Simulator</div>', unsafe_allow_html=True)
        render_insight_note(
            "Instead of assuming one steady market, this simulator lets the portfolio move through Bull, Sideways, Bear, and Crisis phases over time. "
            "Use a preset first, then open advanced controls only if you want to tune the story."
        )

        if not st.session_state.get("data_loaded"):
            st.markdown('<div class="status-box status-info">Load market data first so the simulator can seed itself from the live regime engine.</div>', unsafe_allow_html=True)
            st.stop()

        _rs_mc_port_opts = {}
        if st.session_state.get("weights_tan") is not None:
            _rs_mc_port_opts["Tangency"] = (st.session_state.tan_return, st.session_state.tan_risk)
        if st.session_state.get("weights_utility") is not None:
            try:
                from engines.data_and_compute import cached_portfolio_stats as _rs_cps
                _rs_r, _rs_k, _ = _rs_cps(
                    st.session_state.weights_utility,
                    optimizer_expected_returns(),
                    st.session_state.cov_matrix,
                )
                _rs_mc_port_opts["Utility"] = (float(_rs_r), float(_rs_k))
            except Exception:
                pass
        if st.session_state.get("weights_min") is not None:
            try:
                from engines.data_and_compute import cached_portfolio_stats as _rs_cps
                _rs_r, _rs_k, _ = _rs_cps(
                    st.session_state.weights_min,
                    optimizer_expected_returns(),
                    st.session_state.cov_matrix,
                )
                _rs_mc_port_opts["Min-Risk"] = (float(_rs_r), float(_rs_k))
            except Exception:
                pass
        if st.session_state.get("weights_deploy") is not None:
            try:
                from engines.data_and_compute import cached_portfolio_stats as _rs_cps
                _rs_r, _rs_k, _ = _rs_cps(
                    st.session_state.weights_deploy,
                    optimizer_expected_returns(),
                    st.session_state.cov_matrix,
                )
                _deploy_meta = st.session_state.get("deployable_meta") or {}
                _deploy_posture = str(_deploy_meta.get("posture") or "Deploy-Now").strip()
                _rs_mc_port_opts[f"Deploy-Now [{_deploy_posture}]"] = (float(_rs_r), float(_rs_k))
            except Exception:
                pass
        if not _rs_mc_port_opts:
            st.markdown('<div class="status-box status-warn">Run at least one optimisation to seed the regime simulator from a portfolio.</div>', unsafe_allow_html=True)
            st.stop()

        _rs_cfg, _rs_res = st.columns([1.05, 2.15], gap="large")

        with _rs_cfg:
            st.markdown('<div class="card-title">Quick Setup</div>', unsafe_allow_html=True)

            _rs_port_lbl = st.selectbox("Base portfolio", list(_rs_mc_port_opts.keys()), key="rs_port_sel")
            _rs_mu, _rs_sigma = _rs_mc_port_opts[_rs_port_lbl]
            st.caption(f"Seeded from {_rs_port_lbl}: μ {_rs_mu*100:.1f}%  ·  σ {_rs_sigma*100:.1f}%")
            if _rs_port_lbl.startswith("Deploy-Now"):
                st.caption("Deploy-Now uses the latest walk-forward adjusted mix from the Backtest tab.")

            st.markdown("---")
            _rs_regime_meta = st.session_state.get("regime_meta") or st.session_state.get("noob_regime_meta") or {}
            _rs_auto_probs = _regime_sim_current_market_probs(_rs_regime_meta)
            _rs_fast_label = str(
                _rs_regime_meta.get("fast_dominant_regime")
                or _rs_regime_meta.get("dominant_regime")
                or "unknown"
            ).strip().title()
            _rs_fast_feats = _rs_regime_meta.get("fast_regime_features", {}) or {}
            _rs_index_21d = float(_rs_fast_feats.get("index_21d_return", 0.0) or 0.0)
            _rs_presets = {
                "Current market (Recommended)": _rs_auto_probs,
                "Optimistic": {"bull": 0.55, "sideways": 0.25, "bear": 0.15, "crisis": 0.05},
                "Balanced": {"bull": 0.35, "sideways": 0.35, "bear": 0.20, "crisis": 0.10},
                "Defensive": {"bull": 0.20, "sideways": 0.30, "bear": 0.35, "crisis": 0.15},
                "Crisis prep": {"bull": 0.08, "sideways": 0.18, "bear": 0.34, "crisis": 0.40},
                "Custom": _rs_auto_probs,
            }
            _rs_preset = st.selectbox(
                "Starting market story",
                list(_rs_presets.keys()),
                key="rs_preset",
                help="Pick a ready-made regime mix, or switch to Custom if you want full manual control.",
            )
            if _rs_preset == "Custom":
                st.markdown('<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem">Starting Regime Mix</div>', unsafe_allow_html=True)
                _rs_bull_p = st.slider("Bull %", 0, 100, int(_rs_auto_probs.get("bull", 0.25) * 100), 5, key="rs_bull")
                _rs_sideways_p = st.slider("Sideways %", 0, 100, int(_rs_auto_probs.get("sideways", 0.35) * 100), 5, key="rs_side")
                _rs_bear_p = st.slider("Bear %", 0, 100, int(_rs_auto_probs.get("bear", 0.25) * 100), 5, key="rs_bear")
                _rs_crisis_p = st.slider("Crisis %", 0, 100, int(_rs_auto_probs.get("crisis", 0.15) * 100), 5, key="rs_crisis")
                _rs_probs_raw = _normalise_regime_probs({
                    "bull": _rs_bull_p,
                    "sideways": _rs_sideways_p,
                    "bear": _rs_bear_p,
                    "crisis": _rs_crisis_p,
                })
                st.caption("Custom inputs are auto-normalised back to 100%.")
            else:
                _rs_probs_raw = _normalise_regime_probs(_rs_presets[_rs_preset])
                if _rs_preset == "Current market (Recommended)":
                    st.caption(f"Tactical regime read: {_rs_fast_label}  ·  Nifty 21D: {_rs_index_21d*100:+.1f}%")
                _rs_prob_tiles = st.columns(4)
                for _rci, _rn in enumerate(REGIME_ORDER):
                    with _rs_prob_tiles[_rci]:
                        _rp = _rs_probs_raw.get(_rn, 0.0)
                        _rc = REGIME_DEFAULTS[_rn]["color"]
                        st.markdown(
                            '<div style="text-align:center;padding:0.55rem;background:rgba(10,20,40,0.5);'
                            'border:0.5px solid #1a2d4d;border-radius:10px;">'
                            '<div style="font-size:0.6rem;color:#4a6a90;text-transform:uppercase;letter-spacing:0.08em">{}</div>'
                            '<div style="font-size:1.05rem;font-weight:700;color:{};font-family:\'DM Mono\',monospace">{:.0%}</div>'
                            '</div>'.format(REGIME_DEFAULTS[_rn]["label"], _rc, _rp),
                            unsafe_allow_html=True,
                        )

            st.markdown("---")
            _rs_stickiness = st.slider(
                "How sticky is each regime?",
                min_value=0.70, max_value=0.99, value=0.90, step=0.01,
                key="rs_stickiness",
                help="Higher means the market tends to stay in the same phase for longer before switching.",
            )
            _rs_avg_spell = 1.0 / max(1e-6, 1.0 - _rs_stickiness)
            _rs_transition_matrix = build_transition_matrix(_rs_probs_raw, _rs_stickiness)

            st.markdown("---")
            with st.expander("Advanced market tuning", expanded=False):
                st.caption("Use offsets from the selected portfolio instead of typing raw regime return/vol values from scratch.")
                _rs_custom_assumptions = st.toggle(
                    "Customize per-regime return and volatility",
                    value=False,
                    key="rs_custom_assumptions",
                )
                _rs_regime_params = None
                if _rs_custom_assumptions:
                    _rs_regime_params = {}
                    for _rn in REGIME_ORDER:
                        _def = REGIME_DEFAULTS[_rn]
                        _delta_default = (_def["mu"] - _rs_mu) * 100
                        _vol_mult_default = _def["sigma"] / max(_rs_sigma, 1e-6)
                        st.markdown(f"<div style='font-size:0.72rem;color:#9bb0cd;margin:0.65rem 0 0.35rem'>{REGIME_DEFAULTS[_rn]['label']}</div>", unsafe_allow_html=True)
                        _rc1, _rc2 = st.columns(2)
                        with _rc1:
                            _rdelta = st.number_input(
                                f"{REGIME_DEFAULTS[_rn]['label']} return offset (%)",
                                value=round(_delta_default, 1),
                                step=0.5,
                                key=f"rs_delta_{_rn}",
                            )
                        with _rc2:
                            _rmult = st.number_input(
                                f"{REGIME_DEFAULTS[_rn]['label']} vol multiple",
                                min_value=0.1,
                                value=round(_vol_mult_default, 2),
                                step=0.05,
                                key=f"rs_mult_{_rn}",
                            )
                        _rs_regime_params[_rn] = {
                            **_def,
                            "mu": _rs_mu + _rdelta / 100.0,
                            "sigma": max(0.001, _rs_sigma * _rmult),
                        }
                        st.caption(
                            f"{REGIME_DEFAULTS[_rn]['label']} now uses μ {_rs_regime_params[_rn]['mu']*100:.1f}% and σ {_rs_regime_params[_rn]['sigma']*100:.1f}%."
                        )

            st.markdown("---")
            _rs_capital = st.number_input("Initial capital (₹)", min_value=1000.0, value=100000.0, step=1000.0, format="%.0f", key="rs_capital")
            _rs_contrib = st.number_input("Annual top-up (₹)", min_value=0.0, value=0.0, step=1000.0, format="%.0f", key="rs_contrib")
            _rs_h_unit = st.radio("Horizon unit", ["months", "years"], horizontal=True, key="rs_h_unit")
            if _rs_h_unit == "months":
                _rs_h_val = st.slider("Horizon (months)", 3, 120, 36, 1, key="rs_h_months")
            else:
                _rs_h_val = st.slider("Horizon (years)", 1, 40, 10, 1, key="rs_h_years")
            _rs_horizon_years = horizon_to_years(float(_rs_h_val), _rs_h_unit)
            _rs_horizon_label = f"{int(_rs_h_val)} {_rs_h_unit[:-1] if int(_rs_h_val) == 1 else _rs_h_unit}"
            _rs_nsims = st.select_slider("Simulations", options=[500, 1000, 2500, 5000], value=2500, key="rs_nsims")
            _rs_seed = None
            if st.toggle("Fix random seed", value=False, key="rs_seed_toggle", help="Use the same seed to make runs reproducible."):
                _rs_seed = st.number_input("Seed value", min_value=0, max_value=99999, value=42, step=1, key="rs_seed_val")
            _rs_light_visuals = st.toggle(
                "Light visuals (faster)",
                value=True,
                key="rs_light_visuals",
                help="Use lighter charts and fewer path traces for faster rendering.",
            )
            _rs_run = st.button("▶ Run Market Story Simulation", use_container_width=True, key="rs_run_btn")

            st.markdown("---")
            _rs_dom = max(_rs_probs_raw, key=_rs_probs_raw.get)
            _rs_dom_color = REGIME_DEFAULTS[_rs_dom]["color"]
            st.markdown(
                f"""
                <div class="status-box status-info">
                    <b>What this simulation assumes</b><br>
                    Starts with a <span style="color:{_rs_dom_color};font-weight:700">{REGIME_DEFAULTS[_rs_dom]["label"]}</span>-leaning market,
                    usually stays in one regime for about <b>{_rs_avg_spell:.1f} weeks</b>,
                    shifts regimes weekly,
                    and simulates portfolio returns daily inside each regime.<br><br>
                    Base portfolio: <b>{_rs_port_lbl}</b> at μ <b>{_rs_mu*100:.1f}%</b> and σ <b>{_rs_sigma*100:.1f}%</b>.<br>
                    Horizon: <b>{_rs_horizon_label}</b> across <b>{_rs_nsims:,}</b> simulated futures.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with _rs_res:
            if _rs_run:
                with st.spinner("Running {:,} regime-switching simulations…".format(_rs_nsims)):
                    try:
                        _rs_paths, _rs_t, _rs_rtrack = run_regime_switching_simulation(
                            initial_capital=_rs_capital,
                            horizon_years=float(_rs_horizon_years),
                            n_sims=_rs_nsims,
                            regime_probs=_rs_probs_raw,
                            regime_params=_rs_regime_params,
                            stickiness=_rs_stickiness,
                            annual_contribution=_rs_contrib,
                            seed=_rs_seed,
                        )
                        _rs_stats = compute_mc_stats(_rs_paths, _rs_capital, _rs_contrib, float(_rs_horizon_years))
                        _rs_summary = summarize_regime_paths(_rs_rtrack)
                        st.session_state["rs_last_paths"] = _rs_paths
                        st.session_state["rs_last_t"] = _rs_t
                        st.session_state["rs_last_stats"] = _rs_stats
                        st.session_state["rs_last_rtrack"] = _rs_rtrack
                        st.session_state["rs_last_capital"] = _rs_capital
                        st.session_state["rs_last_horizon"] = _rs_horizon_years
                        st.session_state["rs_last_horizon_label"] = _rs_horizon_label
                        st.session_state["rs_last_probs"] = _rs_probs_raw
                        st.session_state["rs_last_transition"] = _rs_transition_matrix
                        st.session_state["rs_last_summary"] = _rs_summary
                        st.session_state["rs_last_stickiness"] = _rs_stickiness
                        st.session_state["rs_last_preset"] = _rs_preset
                    except Exception as _rs_e:
                        st.error("Regime simulation failed: {}".format(_rs_e))
                        import traceback; st.code(traceback.format_exc())

            if st.session_state.get("rs_last_stats") is not None:
                _rst = st.session_state["rs_last_stats"]
                _rspaths = st.session_state["rs_last_paths"]
                _rst_ax = st.session_state["rs_last_t"]
                _rscap = st.session_state["rs_last_capital"]
                _rsh = st.session_state["rs_last_horizon"]
                _rsh_lbl = st.session_state.get("rs_last_horizon_label", f"{_rsh:.1f} years")
                _rsprbs = st.session_state["rs_last_probs"]
                _rs_tmat = st.session_state.get("rs_last_transition")
                _rs_summary = st.session_state.get("rs_last_summary", {})
                _rs_total_invested = float(_rst.get("total_invested", _rscap))
                _rs_prob_below_invested = max(0.0, 1.0 - float(_rst.get("prob_profit", 0.0)))
                _rs_occ = _rs_summary.get("occupancy", {})
                _rs_occ_dom = max(_rs_occ, key=_rs_occ.get) if _rs_occ else "unknown"
                _rs_light = bool(st.session_state.get("rs_light_visuals", True))

                st.markdown(
                    f"""
                    <div class="status-box status-info">
                        <b>Simulation story</b><br>
                        Most likely outcome after <b>{_rsh_lbl}</b>: <b>₹{_rst['median']:,.0f}</b>.<br>
                        Bad-but-plausible outcome (10th percentile): <b>₹{_rst['p10']:,.0f}</b>.<br>
                        Chance of finishing below total invested capital (₹{_rs_total_invested:,.0f}): <b>{_rs_prob_below_invested:.0%}</b>.<br>
                        Typical market path: starts most often in <b>{str(_rs_summary.get('start_regime', 'unknown')).title()}</b>,
                        ends most often in <b>{str(_rs_summary.get('end_regime', 'unknown')).title()}</b>,
                        and spends the most time in <b>{str(_rs_occ_dom).title()}</b>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                _rs_vm = "₹{:,.0f}".format(_rst["median"])
                _rs_vp10 = "₹{:,.0f}".format(_rst["p10"])
                _rs_vprofit = "{:.0f}%".format((1.0 - _rs_prob_below_invested) * 100)
                _rs_vcg = "{:.2f}%".format(_rst["cagr_median"] * 100)
                _rs_vflip = "{:.1f}".format(_rs_summary.get("avg_switches", 0.0))
                _rs_vspell = "{:.1f}w".format(1.0 / max(1e-6, 1.0 - st.session_state.get("rs_last_stickiness", 0.90)))
                _rs_cg_cls = "positive" if _rst["cagr_median"] > 0 else ""
                st.markdown(
                    "<div class=\"metric-row\">"
                    + metric_card("Most Likely", _rs_vm, "accent")
                    + metric_card("Bad But Plausible", _rs_vp10, "")
                    + metric_card("Chance of Profit", _rs_vprofit, "positive")
                    + metric_card("Median CAGR", _rs_vcg, _rs_cg_cls)
                    + metric_card("Avg Regime Flips", _rs_vflip, "")
                    + metric_card("Avg Spell Length", _rs_vspell, "")
                    + "</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("---")
                _rs_info_left, _rs_info_right = st.columns([1.05, 1.3], gap="large")
                with _rs_info_left:
                    st.markdown('<div class="section-heading">Starting Regime Mix</div>', unsafe_allow_html=True)
                    _rs_prob_cols = st.columns(4)
                    for _rci, _rn in enumerate(REGIME_ORDER):
                        with _rs_prob_cols[_rci]:
                            _rp = _rsprbs.get(_rn, 0)
                            _rc = REGIME_DEFAULTS[_rn]["color"]
                            st.markdown(
                                '<div style="text-align:center;padding:0.6rem;background:rgba(10,20,40,0.5);'
                                'border:0.5px solid #1a2d4d;border-radius:10px;">'
                                '<div style="font-size:0.62rem;color:#4a6a90;text-transform:uppercase;letter-spacing:0.1em">{}</div>'
                                '<div style="font-size:1.3rem;font-weight:700;color:{};font-family:\'DM Mono\',monospace">{:.0%}</div>'
                                '</div>'.format(REGIME_DEFAULTS[_rn]["label"], _rc, _rp),
                                unsafe_allow_html=True,
                            )
                    st.caption(f"Preset used: {st.session_state.get('rs_last_preset', 'Custom')}.")

                with _rs_info_right:
                    st.markdown('<div class="section-heading">How Regime Shifts Work</div>', unsafe_allow_html=True)
                    render_insight_note("Rows show the current regime. Columns show the next regime one week later. Bright diagonal cells mean the market tends to persist before changing.")
                    _rs_tdf = pd.DataFrame(
                        _rs_tmat,
                        index=[REGIME_DEFAULTS[_rn]["label"] for _rn in REGIME_ORDER],
                        columns=[REGIME_DEFAULTS[_rn]["label"] for _rn in REGIME_ORDER],
                    )
                    fig_t = go.Figure(go.Heatmap(
                        z=_rs_tdf.values,
                        x=_rs_tdf.columns,
                        y=_rs_tdf.index,
                        colorscale="Blues",
                        zmin=0,
                        zmax=1,
                        text=[[f"{v:.0%}" for v in row] for row in _rs_tdf.values],
                        texttemplate="%{text}",
                        hovertemplate="From %{y}<br>To %{x}<br>%{z:.1%}<extra></extra>",
                        colorbar=dict(title="Prob."),
                    ))
                    fig_t.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0a0e17",
                        plot_bgcolor="#0d1525",
                        font=dict(family="DM Mono", color="#80b0d0"),
                        height=320,
                        margin=dict(l=40, r=20, t=20, b=40),
                    )
                    st.plotly_chart(fig_t, use_container_width=True)

                if not _rs_light:
                    st.markdown('<div class="section-heading">Regime Occupancy Over Time</div>', unsafe_allow_html=True)
                    render_insight_note("Each stacked bar shows how much of the simulated market is in Bull, Sideways, Bear, or Crisis at that point in time.")
                    _rs_rtrack = st.session_state["rs_last_rtrack"]
                    _rs_n_weeks = _rs_rtrack.shape[0]
                    _rs_downsample = max(1, _rs_n_weeks // 60)
                    _rs_weeks_idx  = np.arange(0, _rs_n_weeks, _rs_downsample)
                    _rs_occ_curve = np.zeros((len(_rs_weeks_idx), 4))
                    for _wi, _wt in enumerate(_rs_weeks_idx):
                        for _ri in range(4):
                            _rs_occ_curve[_wi, _ri] = np.mean(_rs_rtrack[_wt] == _ri)

                    fig_occ = go.Figure()
                    for _ri, _rn in enumerate(REGIME_ORDER):
                        _rc = REGIME_DEFAULTS[_rn]["color"]
                        fig_occ.add_trace(go.Bar(
                            x=_rs_weeks_idx / 52,
                            y=_rs_occ_curve[:, _ri],
                            name=REGIME_DEFAULTS[_rn]["label"],
                            marker_color=_rc,
                        ))
                    fig_occ.update_layout(
                        barmode="stack",
                        template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0d1525",
                        xaxis_title="Time (years)", yaxis_title="Fraction of simulations",
                        title=dict(text="Regime Occupancy", font=dict(color="#a0c8e8")),
                        font=dict(family="DM Mono", color="#80b0d0"),
                        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0.4)"),
                        height=300, margin=dict(l=50, r=20, t=45, b=45),
                    )
                    st.plotly_chart(fig_occ, use_container_width=True)
                else:
                    st.caption("Light visuals mode is on, so the occupancy timeline is skipped for a faster render.")

                st.markdown('<div class="section-heading">Regime-Switching Wealth Paths</div>', unsafe_allow_html=True)
                render_insight_note("The fan chart shows the full range of possible portfolio paths once the market is allowed to switch states over time.")

                _rs_disp, _ = sample_paths_for_plot(_rspaths, _rst_ax, n_sample=60 if _rs_light else 150)
                _rs_p5p   = np.percentile(_rspaths, 5,  axis=1)
                _rs_p25p  = np.percentile(_rspaths, 25, axis=1)
                _rs_p75p  = np.percentile(_rspaths, 75, axis=1)
                _rs_p95p  = np.percentile(_rspaths, 95, axis=1)
                _rs_medp  = np.percentile(_rspaths, 50, axis=1)

                fig_rs = go.Figure()
                for _pi in range(min(60 if _rs_light else 150, _rs_disp.shape[1])):
                    _pv = _rs_disp[-1, _pi]
                    _pc = "rgba(0,120,220,0.07)" if _pv >= _rscap else "rgba(220,60,60,0.05)"
                    fig_rs.add_trace(go.Scatter(
                        x=_rst_ax, y=_rs_disp[:, _pi],
                        mode="lines", line=dict(width=0.5, color=_pc),
                        showlegend=False, hoverinfo="skip",
                    ))
                fig_rs.add_trace(go.Scatter(
                    x=np.concatenate([_rst_ax, _rst_ax[::-1]]),
                    y=np.concatenate([_rs_p95p, _rs_p5p[::-1]]),
                    fill="toself", fillcolor="rgba(0,120,200,0.08)",
                    line=dict(width=0), name="90% range",
                ))
                fig_rs.add_trace(go.Scatter(
                    x=np.concatenate([_rst_ax, _rst_ax[::-1]]),
                    y=np.concatenate([_rs_p75p, _rs_p25p[::-1]]),
                    fill="toself", fillcolor="rgba(0,180,255,0.13)",
                    line=dict(width=0), name="50% range",
                ))
                fig_rs.add_trace(go.Scatter(x=_rst_ax, y=_rs_medp, mode="lines",
                    line=dict(color="gold", width=2.5), name="Median"))
                fig_rs.add_trace(go.Scatter(x=_rst_ax, y=_rs_p95p, mode="lines",
                    line=dict(color="#22c55e", width=1.2, dash="dot"), name="95th pctile"))
                fig_rs.add_trace(go.Scatter(x=_rst_ax, y=_rs_p5p, mode="lines",
                    line=dict(color="#ff4d6d", width=1.2, dash="dot"), name="5th pctile"))
                fig_rs.add_hline(y=_rscap, line_dash="dash",
                    line_color="rgba(255,213,79,0.4)",
                    annotation_text="Initial capital",
                    annotation_font_color="#ffd54f")
                fig_rs.add_hline(y=_rs_total_invested, line_dash="dot",
                    line_color="rgba(128,200,255,0.35)",
                    annotation_text="Total invested",
                    annotation_font_color="#80c8ff")
                fig_rs.update_layout(
                    template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0d1525",
                    xaxis_title="Time (years)  ·  horizon = {}".format(_rsh_lbl),
                    yaxis_title="Portfolio Value (₹)",
                    title=dict(text="Regime-Switching Fan Chart  ·  {:,} simulations".format(len(_rspaths[0])),
                               font=dict(color="#a0c8e8")),
                    font=dict(family="DM Mono", color="#80b0d0"),
                    legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0.4)"),
                    height=500, margin=dict(l=60, r=20, t=55, b=55),
                )
                st.plotly_chart(fig_rs, use_container_width=True)

            else:
                st.markdown(
                    '<div style="padding:4rem 2rem;text-align:center;color:#263a56;">'
                    '<div style="font-size:3rem;margin-bottom:1rem">🔄</div>'
                    '<div style="font-size:1.1rem;color:#4a6a90;margin-bottom:0.5rem">'
                    'Pick a market story and hit <strong>Run Market Story Simulation</strong></div>'
                    '<div style="font-size:0.82rem;color:#1a2d4d">'
                    'QuanSen will start from your chosen market mix, let regimes switch over time,<br>'
                    'and show how that changes the portfolio range of outcomes.'
                    '</div></div>',
                    unsafe_allow_html=True,
                )

    # ══════════════════════════════════════════════════════════
    # STRESS SCENARIOS SUB-TAB  (both modes)
    # ══════════════════════════════════════════════════════════
    with _mc_sub_stress:
        if is_noob:
            st.markdown(
                '<div style="background:linear-gradient(135deg,#031a0e,#0a2d14);'
                'border:1px solid #1a5c2a;border-radius:16px;padding:1.2rem 1.6rem;margin-bottom:1rem;">'
                '<div style="font-size:0.6rem;letter-spacing:0.22em;text-transform:uppercase;color:#3db85a;margin-bottom:4px">QuanSen · Easy Mode</div>'
                '<div style="font-size:1.3rem;font-weight:800;color:#4afa7a;font-family:Syne,sans-serif;">💥 What if a crash hits?</div>'
                '<div style="font-size:0.78rem;color:#3db85a;margin-top:4px">See how your money survives historical market crashes</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="section-heading">Stress Scenario Engine</div>', unsafe_allow_html=True)
            render_insight_note(
                "Each scenario injects a deterministic drawdown at a chosen month, "
                "runs a distressed GBM through the crash window, then switches to a recovery drift. "
                "The baseline (no shock) runs in parallel for direct comparison. "
                "All scenarios share the same portfolio μ / σ and capital settings."
            )

        if not st.session_state.get("data_loaded"):
            st.markdown('<div class="status-box status-info">Load market data first.</div>', unsafe_allow_html=True)
            st.stop()

        _ss_port_opts = {}
        if st.session_state.get("weights_tan") is not None:
            _ss_port_opts["Tangency"] = (st.session_state.tan_return, st.session_state.tan_risk)
        if st.session_state.get("weights_utility") is not None:
            try:
                from engines.data_and_compute import cached_portfolio_stats as _ss_cps
                _ss_r, _ss_k, _ = _ss_cps(
                    st.session_state.weights_utility,
                    optimizer_expected_returns(),
                    st.session_state.cov_matrix,
                )
                _ss_port_opts["Utility"] = (float(_ss_r), float(_ss_k))
            except Exception:
                pass
        if not _ss_port_opts:
            st.markdown('<div class="status-box status-warn">Run at least one optimisation first.</div>', unsafe_allow_html=True)
            st.stop()

        _ss_cfg, _ss_res = st.columns([1, 2.2], gap="large")

        with _ss_cfg:
            if is_noob:
                st.markdown('<div class="nb-green-heading">⚙ Crash Settings</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="card-title">Scenario Settings</div>', unsafe_allow_html=True)

            _ss_port_lbl = st.selectbox("Base portfolio", list(_ss_port_opts.keys()), key="ss_port_sel")
            _ss_mu, _ss_sigma = _ss_port_opts[_ss_port_lbl]

            st.markdown("---")
            _ss_all_names  = list(STRESS_SCENARIOS.keys())
            _ss_default    = [s for s in _ss_all_names if s != "Custom"]
            _ss_selected   = st.multiselect(
                "Select scenarios to run",
                options=_ss_all_names,
                default=_ss_default[:4],
                key="ss_selected",
                help="Custom lets you specify your own shock parameters below.",
            )

            _ss_custom_params = None
            if "Custom" in _ss_selected:
                with st.expander("Custom scenario parameters", expanded=True):
                    _ss_c1, _ss_c2 = st.columns(2)
                    with _ss_c1:
                        _ss_cshock = st.number_input("Shock size %",  min_value=-99.0, max_value=-1.0,  value=-30.0, step=1.0, key="ss_cshock") / 100.0
                        _ss_cdur   = st.number_input("Duration (months)", min_value=1.0, max_value=60.0, value=6.0,  step=1.0, key="ss_cdur")
                    with _ss_c2:
                        _ss_crmu   = st.number_input("Recovery μ %",   min_value=-20.0, max_value=60.0,  value=15.0, step=0.5, key="ss_crmu")  / 100.0
                        _ss_crsig  = st.number_input("Recovery σ %",   min_value=1.0,   max_value=80.0,  value=20.0, step=0.5, key="ss_crsig") / 100.0
                    _ss_custom_params = {
                        "shock": _ss_cshock,
                        "shock_duration_months": _ss_cdur,
                        "recovery_mu": _ss_crmu,
                        "recovery_sigma": _ss_crsig,
                    }

            st.markdown("---")
            _ss_shock_start = st.slider(
                "Shock starts at month",
                min_value=1, max_value=48, value=6, step=1,
                key="ss_shock_start",
                help="Month within the simulation horizon at which the shock hits.",
            )

            _ss_capital = st.number_input("Initial capital (₹)", min_value=1000.0, value=100000.0, step=1000.0, format="%.0f", key="ss_capital")
            _ss_contrib = st.number_input("Annual top-up (₹)",   min_value=0.0,    value=0.0,      step=1000.0, format="%.0f", key="ss_contrib")
            _ss_h_val   = st.slider("Horizon (years)", 1, 40, 10, 1, key="ss_horizon")
            _ss_nsims   = st.select_slider("Simulations", options=[500, 1000, 2500, 5000], value=2500, key="ss_nsims")
            _ss_run     = st.button(
                "▶ Run Stress Tests" if not is_noob else "💥 Simulate the Crashes!",
                use_container_width=True, key="ss_run_btn",
            )

            # Scenario summary table
            st.markdown("---")
            _ss_tbl_rows = []
            for _sn in (_ss_selected if _ss_selected else list(STRESS_SCENARIOS.keys())[:4]):
                _sp = {**STRESS_SCENARIOS.get(_sn, {}), **((_ss_custom_params or {}) if _sn == "Custom" else {})}
                if _sp:
                    _ss_tbl_rows.append({
                        "Scenario": _sn,
                        "Shock":    "{:.0%}".format(_sp.get("shock", 0)),
                        "Duration": "{}m".format(int(_sp.get("shock_duration_months", 0))),
                        "Rec. μ":  "{:.0%}".format(_sp.get("recovery_mu", 0)),
                    })
            if _ss_tbl_rows:
                st.dataframe(pd.DataFrame(_ss_tbl_rows).set_index("Scenario"), use_container_width=True)

        with _ss_res:
            if _ss_run and _ss_selected:
                with st.spinner("Running {:,} paths × {} scenarios…".format(_ss_nsims, len(_ss_selected))):
                    try:
                        # Baseline
                        _ss_base_paths, _ss_base_t = run_mc_simulation(
                            mu=_ss_mu, sigma=_ss_sigma,
                            initial_capital=_ss_capital,
                            horizon_years=float(_ss_h_val),
                            n_sims=_ss_nsims,
                            annual_contribution=_ss_contrib,
                            seed=99,
                        )
                        # All scenarios
                        _ss_results = run_all_stress_scenarios(
                            mu=_ss_mu, sigma=_ss_sigma,
                            initial_capital=_ss_capital,
                            horizon_years=float(_ss_h_val),
                            n_sims=_ss_nsims,
                            selected_scenarios=_ss_selected,
                            shock_start_month=float(_ss_shock_start),
                            annual_contribution=_ss_contrib,
                            custom_params=_ss_custom_params,
                            seed=42,
                        )
                        st.session_state["ss_base_paths"]    = _ss_base_paths
                        st.session_state["ss_base_t"]        = _ss_base_t
                        st.session_state["ss_results"]       = _ss_results
                        st.session_state["ss_stored_capital"]= _ss_capital
                        st.session_state["ss_stored_horizon"]= _ss_h_val
                        st.session_state["ss_stored_shock"]  = _ss_shock_start
                    except Exception as _ss_e:
                        st.error("Stress simulation failed: {}".format(_ss_e))
                        import traceback; st.code(traceback.format_exc())

            if st.session_state.get("ss_results") is not None:
                _ss_bp    = st.session_state["ss_base_paths"]
                _ss_bt    = st.session_state["ss_base_t"]
                _ss_res_d = st.session_state["ss_results"]
                _ss_cap   = st.session_state["ss_stored_capital"]
                _ss_h     = st.session_state["ss_stored_horizon"]
                _ss_ss    = st.session_state["ss_stored_shock"]

                # ── Comparison table ────────────────────────────────
                st.markdown('<div class="section-heading">Scenario Comparison</div>', unsafe_allow_html=True)
                _ss_comp_df = compare_scenario_stats(_ss_bp, _ss_res_d, _ss_cap, float(_ss_h))

                def _ss_colour_median(val):
                    try:
                        if int(val) > _ss_cap: return "color:#00e676;font-weight:600"
                        return "color:#ff5252"
                    except Exception: return ""
                def _ss_colour_var(val):
                    try:
                        v = float(str(val).replace("%",""))
                        if v < -30: return "color:#ff5252;font-weight:600"
                        if v < -15: return "color:#ff8a65"
                        return "color:#b9f6ca"
                    except Exception: return ""

                st.dataframe(
                    _ss_comp_df.style
                        .background_gradient(cmap="RdYlGn", subset=["Median (₹)", "Mean (₹)", "P95 (₹)"])
                        .background_gradient(cmap="Reds_r",  subset=["P5 (₹)"]),
                    use_container_width=True,
                )

                # ── Median path overlay chart ───────────────────────
                st.markdown('<div class="section-heading">Median Wealth Path — All Scenarios vs Baseline</div>', unsafe_allow_html=True)
                if not is_noob:
                    render_insight_note(
                        "Each line is the median wealth path for that scenario. "
                        "The shock onset (dashed vertical) marks where the crash hits. "
                        "Gap between baseline and scenario at the end = cost of the crash."
                    )

                fig_stress = go.Figure()

                # Baseline median
                _ss_b_med = np.median(_ss_bp, axis=1)
                fig_stress.add_trace(go.Scatter(
                    x=_ss_bt, y=_ss_b_med,
                    mode="lines", name="Baseline (no shock)",
                    line=dict(color="#80c8ff", width=2.5, dash="dash"),
                    hovertemplate="Baseline<br>Year %{x:.1f}<br>₹%{y:,.0f}<extra></extra>",
                ))

                # Each scenario: median + P5/P95 band
                for _sn, (_sp_paths, _sp_t) in _ss_res_d.items():
                    _sc = STRESS_SCENARIOS.get(_sn, {}).get("color", "#80c8ff")
                    _sp_med = np.median(_sp_paths, axis=1)
                    _sp_p5  = np.percentile(_sp_paths, 5,  axis=1)
                    _sp_p95 = np.percentile(_sp_paths, 95, axis=1)

                    # Shaded band — convert hex to rgba for transparency
                    def _hex_to_rgba(h, alpha=0.08):
                        h = h.lstrip("#")
                        if len(h) == 6:
                            r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
                            return "rgba({},{},{},{})".format(r, g, b, alpha)
                        return "rgba(120,120,120,{})".format(alpha)
                    _band_fill = _hex_to_rgba(_sc) if _sc.startswith("#") else "rgba(120,120,120,0.08)"
                    fig_stress.add_trace(go.Scatter(
                        x=np.concatenate([_sp_t, _sp_t[::-1]]),
                        y=np.concatenate([_sp_p95, _sp_p5[::-1]]),
                        fill="toself",
                        fillcolor=_band_fill,
                        line=dict(width=0),
                        showlegend=False, hoverinfo="skip",
                    ))
                    fig_stress.add_trace(go.Scatter(
                        x=_sp_t, y=_sp_med,
                        mode="lines", name=_sn,
                        line=dict(color=_sc, width=2),
                        hovertemplate=_sn + "<br>Year %{x:.1f}<br>₹%{y:,.0f}<extra></extra>",
                    ))

                # Shock onset line
                fig_stress.add_vline(
                    x=_ss_ss / 12.0,
                    line_dash="dot", line_color="rgba(255,213,79,0.6)",
                    annotation_text="Shock @ month {}".format(_ss_ss),
                    annotation_font_color="#ffd54f",
                    annotation_position="top right",
                )
                fig_stress.add_hline(
                    y=_ss_cap, line_dash="dash",
                    line_color="rgba(255,255,255,0.12)",
                    annotation_text="Initial capital",
                    annotation_font_color="#4a6a90",
                )

                fig_stress.update_layout(
                    template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0d1525",
                    xaxis_title="Time (years)", yaxis_title="Portfolio Value (₹)",
                    title=dict(text="Stress Scenario Median Paths  ·  {:,} simulations each".format(_ss_nsims),
                               font=dict(color="#a0c8e8")),
                    font=dict(family="DM Mono", color="#80b0d0"),
                    legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0.4)"),
                    height=500, margin=dict(l=60, r=20, t=55, b=55),
                )
                st.plotly_chart(fig_stress, use_container_width=True)

                # ── Final distribution overlay histogram ────────────
                st.markdown('<div class="section-heading">Terminal Wealth Distribution Overlay</div>', unsafe_allow_html=True)

                fig_hdist = go.Figure()
                # Baseline
                _ss_b_finals = _ss_bp[-1, :]
                fig_hdist.add_trace(go.Histogram(
                    x=_ss_b_finals,
                    name="Baseline",
                    opacity=0.55,
                    marker_color="#80c8ff",
                    nbinsx=60,
                    hovertemplate="Baseline<br>₹%{x:,.0f}<br>Count: %{y}<extra></extra>",
                ))
                for _sn, (_sp_paths, _) in _ss_res_d.items():
                    _sc = STRESS_SCENARIOS.get(_sn, {}).get("color", "#aaaaaa")
                    fig_hdist.add_trace(go.Histogram(
                        x=_sp_paths[-1, :],
                        name=_sn,
                        opacity=0.45,
                        marker_color=_sc,
                        nbinsx=60,
                        hovertemplate=_sn + "<br>₹%{x:,.0f}<br>Count: %{y}<extra></extra>",
                    ))
                fig_hdist.update_layout(
                    barmode="overlay",
                    template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0d1525",
                    xaxis_title="Terminal Portfolio Value (₹)",
                    yaxis_title="Count",
                    title=dict(text="Terminal Wealth Distributions", font=dict(color="#a0c8e8")),
                    font=dict(family="DM Mono", color="#80b0d0"),
                    legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0.4)"),
                    height=380, margin=dict(l=60, r=20, t=50, b=55),
                )
                st.plotly_chart(fig_hdist, use_container_width=True)

                # ── Export ──────────────────────────────────────────
                st.markdown("---")
                _ss_export_rows = [{"Scenario": "Baseline (no shock)", "final_wealth_sample": _ss_bp[-1, :].tolist()}]
                for _sn, (_sp_paths, _) in _ss_res_d.items():
                    _ss_export_rows.append({"Scenario": _sn, "final_wealth_sample": _sp_paths[-1, :].tolist()})
                _ss_export_df = pd.DataFrame({
                    row["Scenario"]: row["final_wealth_sample"] for row in _ss_export_rows
                })
                st.download_button(
                    "⬇ Download Terminal Wealth CSV (all scenarios)",
                    data=_ss_export_df.to_csv(index=False).encode(),
                    file_name="quansen_stress_finals.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="ss_dl_finals",
                )

            else:
                st.markdown(
                    '<div style="padding:4rem 2rem;text-align:center;color:#263a56;">'
                    '<div style="font-size:3rem;margin-bottom:1rem">💥</div>'
                    '<div style="font-size:1.1rem;color:#4a6a90;margin-bottom:0.5rem">'
                    'Select scenarios and hit <strong>Run Stress Tests</strong></div>'
                    '<div style="font-size:0.82rem;color:#1a2d4d">'
                    'Each scenario injects a historical-style crash at your chosen month<br>'
                    'then simulates recovery — baseline runs in parallel for comparison.'
                    '</div></div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════
# TAB — LIVE PORTFOLIO TRACKER  (both modes)
# ══════════════════════════════════════════════════════════════
if _qs_s == "tracker":
    from engines.portfolio_tracker import (
        fetch_live_quotes, fetch_sparklines,
        compute_portfolio_pnl, compute_drift,
        compute_rebalance_trades,
        empty_holding, holdings_to_df, df_to_holdings,
        parse_holdings_csv, export_holdings_csv,
    )

    # ── Session state init ────────────────────────────────────
    if "lpt_holdings" not in st.session_state:
        st.session_state.lpt_holdings = []
    if "lpt_quotes" not in st.session_state:
        st.session_state.lpt_quotes = {}
    if "lpt_last_refresh" not in st.session_state:
        st.session_state.lpt_last_refresh = None
    if "lpt_show_rebalance" not in st.session_state:
        st.session_state.lpt_show_rebalance = False

    # ── Header ────────────────────────────────────────────────
    if is_noob:
        st.markdown(
            '''<div style="background:linear-gradient(135deg,#031a0e,#0a2d14);
                border:1px solid #1a5c2a;border-radius:16px;padding:1.2rem 1.6rem;
                margin-bottom:1.2rem;">
                <div style="font-size:0.6rem;letter-spacing:0.22em;text-transform:uppercase;
                            color:#3db85a;margin-bottom:4px">QuanSen · Easy Mode</div>
                <div style="font-size:1.3rem;font-weight:800;color:#4afa7a;font-family:Syne,sans-serif;">
                    📡 My Live Portfolio
                </div>
                <div style="font-size:0.78rem;color:#3db85a;margin-top:4px">
                    Track your actual holdings — see live prices, gains & losses in real time
                </div>
            </div>''',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="section-heading">Live Portfolio Tracker</div>', unsafe_allow_html=True)
        render_insight_note(
            "Enter your actual holdings — shares owned and average buy price. "
            "The tracker fetches live quotes and shows unrealised P&L, day change, "
            "weight drift vs any computed optimizer portfolio, and a rebalancing trade list."
        )

    # ══════════════════════════════════════════════════════════
    # Layout: left config | right live display
    # ══════════════════════════════════════════════════════════
    _lpt_left, _lpt_right = st.columns([1, 2.4], gap="large")

    # ── LEFT: Holdings Manager ────────────────────────────────
    with _lpt_left:
        if is_noob:
            st.markdown('<div class="nb-green-heading">➕ Add a Holding</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card-title">Holdings Manager</div>', unsafe_allow_html=True)

        # Quick-add from current optimizer basket
        if st.session_state.get("tickers") and not is_noob:
            if st.button("📥 Import tickers from current basket", use_container_width=True, key="lpt_import_basket"):
                _lpt_existing = {h["ticker"] for h in st.session_state.lpt_holdings}
                _lpt_added = 0
                for _t in st.session_state.tickers:
                    if _t not in _lpt_existing:
                        st.session_state.lpt_holdings.append(empty_holding(_t))
                        _lpt_added += 1
                if _lpt_added:
                    st.success("Added {} tickers — fill in shares and cost.".format(_lpt_added))
                    st.rerun()

        # Add single holding form
        with st.form("lpt_add_form", clear_on_submit=True):
            _lpt_f1, _lpt_f2 = st.columns(2)
            with _lpt_f1:
                _lpt_new_ticker = st.text_input("Ticker", placeholder="SBIN.NS", key="lpt_new_ticker")
                _lpt_new_shares = st.number_input("Shares", min_value=0.0001, value=1.0, step=0.5, format="%.4f", key="lpt_new_shares")
            with _lpt_f2:
                _lpt_new_cost   = st.number_input("Avg buy price", min_value=0.01, value=100.0, step=0.5, format="%.2f", key="lpt_new_cost")
                _lpt_new_date   = st.text_input("Buy date", value=pd.Timestamp.today().strftime("%Y-%m-%d"), key="lpt_new_date")
            _lpt_new_notes  = st.text_input("Notes (optional)", placeholder="e.g. SIP tranche 1", key="lpt_new_notes")
            _lpt_add_submit = st.form_submit_button("➕ Add Holding", use_container_width=True)

        if _lpt_add_submit and _lpt_new_ticker.strip():
            _lpt_t = _lpt_new_ticker.strip().upper()
            # Check for duplicate — if exists, update shares/cost as weighted average
            _lpt_existing_idx = next(
                (i for i, h in enumerate(st.session_state.lpt_holdings) if h["ticker"] == _lpt_t), None
            )
            if _lpt_existing_idx is not None:
                _lpt_old = st.session_state.lpt_holdings[_lpt_existing_idx]
                _lpt_total_shares = _lpt_old["shares"] + _lpt_new_shares
                _lpt_new_avg = (
                    (_lpt_old["shares"] * _lpt_old["avg_cost"] + _lpt_new_shares * _lpt_new_cost)
                    / _lpt_total_shares
                )
                st.session_state.lpt_holdings[_lpt_existing_idx]["shares"]   = round(_lpt_total_shares, 4)
                st.session_state.lpt_holdings[_lpt_existing_idx]["avg_cost"] = round(_lpt_new_avg, 4)
                st.success("Updated {} — averaged cost to ₹{:.2f}".format(_lpt_t, _lpt_new_avg))
            else:
                _lpt_h = empty_holding(_lpt_t)
                _lpt_h["shares"]   = _lpt_new_shares
                _lpt_h["avg_cost"] = _lpt_new_cost
                _lpt_h["buy_date"] = _lpt_new_date
                _lpt_h["notes"]    = _lpt_new_notes
                st.session_state.lpt_holdings.append(_lpt_h)
                st.success("Added {}".format(_lpt_t))
            st.session_state.lpt_quotes = {}
            st.rerun()

        # CSV import/export
        st.markdown("---")
        if is_noob:
            st.markdown('<div class="nb-green-heading">📂 Import / Export</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem">Bulk Import / Export</div>', unsafe_allow_html=True)

        st.caption("Supports broker statements (Zerodha, Groww, CDSL, NSDL) and QuanSen native CSV/XLSX")
        _lpt_upload = st.file_uploader(
            "Upload holdings file",
            type=["csv", "xlsx", "xls"], key="lpt_csv_upload",
            label_visibility="collapsed",
        )
        if _lpt_upload:
            _lpt_fname = _lpt_upload.name.lower()
            _lpt_raw   = _lpt_upload.read()
            _lpt_is_xl = _lpt_fname.endswith(".xlsx") or _lpt_fname.endswith(".xls")
            _lpt_parsed, _lpt_errs = parse_holdings_csv(
                csv_bytes  = b"" if _lpt_is_xl else _lpt_raw,
                xlsx_bytes = _lpt_raw if _lpt_is_xl else None,
            )
            # Show warnings (non-blocking) in yellow, hard errors in red
            _lpt_hard_errs = [e for e in _lpt_errs if not e.startswith("Derived") and not e.startswith("ISIN")]
            _lpt_warnings  = [e for e in _lpt_errs if e.startswith("Derived") or e.startswith("ISIN")]
            for w in _lpt_warnings:
                st.warning(w)
            for e in _lpt_hard_errs:
                st.error(e)
            if _lpt_parsed:
                st.success("Parsed {} holdings from {}".format(len(_lpt_parsed), _lpt_upload.name))
                # Preview table
                _lpt_preview = pd.DataFrame([
                    {"Ticker": h["ticker"], "Shares": h["shares"],
                     "Avg Cost": h["avg_cost"], "Notes": h["notes"]}
                    for h in _lpt_parsed
                ])
                st.dataframe(_lpt_preview, use_container_width=True,
                             height=min(300, 45 + 35*len(_lpt_preview)))
                st.caption("⚠ Verify tickers above — broker names are auto-mapped to NSE symbols.")
                _lpt_mode = st.radio("Import mode", ["Replace all", "Append"],
                                     horizontal=True, key="lpt_import_mode")
                if st.button("✅ Confirm Import ({} holdings)".format(len(_lpt_parsed)),
                             use_container_width=True, key="lpt_confirm_import"):
                    if _lpt_mode == "Replace all":
                        st.session_state.lpt_holdings = _lpt_parsed
                    else:
                        _lpt_tickers_existing = {h["ticker"] for h in st.session_state.lpt_holdings}
                        for _lh in _lpt_parsed:
                            if _lh["ticker"] not in _lpt_tickers_existing:
                                st.session_state.lpt_holdings.append(_lh)
                    st.session_state.lpt_quotes = {}
                    st.rerun()

        if st.session_state.lpt_holdings:
            st.download_button(
                "⬇ Export Holdings CSV",
                data=export_holdings_csv(st.session_state.lpt_holdings),
                file_name="quansen_holdings.csv",
                mime="text/csv",
                use_container_width=True,
                key="lpt_dl_csv",
            )

        # Holdings list with remove buttons
        if st.session_state.lpt_holdings:
            st.markdown("---")
            if is_noob:
                st.markdown('<div class="nb-green-heading">📋 Your Holdings</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem">Current Holdings ({} positions)</div>'.format(len(st.session_state.lpt_holdings)), unsafe_allow_html=True)

            for _lpt_i, _lpt_h in enumerate(st.session_state.lpt_holdings):
                _lpt_q      = st.session_state.lpt_quotes.get(_lpt_h["ticker"], {})
                _lpt_ok     = _lpt_q.get("ok", False)
                _lpt_px     = _lpt_q.get("price")
                _lpt_islive = _lpt_q.get("is_live", False)
                _lpt_chg= _lpt_q.get("chg_pct")
                _lpt_cv = _lpt_h["shares"] * _lpt_px if _lpt_ok and _lpt_px else None
                _lpt_cb = _lpt_h["shares"] * _lpt_h["avg_cost"]
                _lpt_unr= (_lpt_cv - _lpt_cb) if _lpt_cv is not None else None
                _lpt_unr_col = "#00e676" if (_lpt_unr or 0) >= 0 else "#ff5252"
                _lpt_chg_col = "#00e676" if (_lpt_chg or 0) >= 0 else "#ff5252"

                _lpt_hc1, _lpt_hc2 = st.columns([5, 1])
                with _lpt_hc1:
                    st.markdown(
                        '<div style="padding:5px 0;border-bottom:0.5px solid #1a2d4d;">' +
                        '<span style="color:#80c8ff;font-size:0.82rem;font-weight:600;font-family:DM Mono,monospace">{}</span>'.format(_lpt_h["ticker"]) +
                        '<span style="color:#4a6a90;font-size:0.72rem;margin-left:8px">{:.0f} sh @ ₹{:.2f}</span>'.format(_lpt_h["shares"], _lpt_h["avg_cost"]) +
                        ('<span style="color:{};font-size:0.72rem;margin-left:8px;font-family:DM Mono,monospace">₹{:,.0f}</span>'.format(_lpt_unr_col, _lpt_unr) if _lpt_unr is not None else "") +
                        ('<span style="color:{};font-size:0.68rem;margin-left:6px">{:+.1f}%</span>'.format(_lpt_chg_col, _lpt_chg) if _lpt_chg is not None else "") +
                        '</div>',
                        unsafe_allow_html=True,
                    )
                with _lpt_hc2:
                    if st.button("✕", key="lpt_rm_{}_{}".format(_lpt_h["ticker"], _lpt_i)):
                        st.session_state.lpt_holdings.pop(_lpt_i)
                        st.session_state.lpt_quotes = {}
                        st.rerun()

            if st.button("🗑 Clear All Holdings", use_container_width=True, key="lpt_clear_all"):
                st.session_state.lpt_holdings = []
                st.session_state.lpt_quotes   = {}
                st.rerun()

        # Refresh controls
        st.markdown("---")
        _lpt_ref_col1, _lpt_ref_col2 = st.columns(2)
        with _lpt_ref_col1:
            _lpt_do_refresh = st.button(
                "🔄 Refresh Quotes" if not is_noob else "📡 Get Live Prices!",
                use_container_width=True, key="lpt_refresh_btn",
                disabled=len(st.session_state.lpt_holdings) == 0,
            )
        with _lpt_ref_col2:
            _lpt_auto_refresh = st.toggle("Auto on load", value=True, key="lpt_auto_refresh")

        if st.session_state.lpt_last_refresh:
            st.caption("Last refreshed: {}".format(st.session_state.lpt_last_refresh))

    # ── RIGHT: Live Display ───────────────────────────────────
    with _lpt_right:

        # Auto-refresh on first load if holdings exist and quotes are stale
        _lpt_needs_refresh = (
            _lpt_do_refresh or
            (_lpt_auto_refresh and
             st.session_state.lpt_holdings and
             not st.session_state.lpt_quotes)
        )

        if _lpt_needs_refresh and st.session_state.lpt_holdings:
            _lpt_tickers_to_fetch = list({h["ticker"] for h in st.session_state.lpt_holdings})
            with st.spinner("Fetching live quotes for {} tickers…".format(len(_lpt_tickers_to_fetch))):
                st.session_state.lpt_quotes = fetch_live_quotes(_lpt_tickers_to_fetch)
                st.session_state.lpt_last_refresh = pd.Timestamp.now().strftime("%Y-%m-%d  %H:%M:%S")

        if not st.session_state.lpt_holdings:
            st.markdown(
                '''<div style="padding:5rem 2rem;text-align:center;color:#263a56;">
                    <div style="font-size:3rem;margin-bottom:1rem">📡</div>
                    <div style="font-size:1.1rem;color:#4a6a90;margin-bottom:0.5rem">
                        Add your first holding on the left
                    </div>
                    <div style="font-size:0.82rem;color:#1a2d4d">
                        Enter ticker, shares owned, and your average buy price.<br>
                        Hit Refresh Quotes to see live P&amp;L instantly.
                    </div>
                </div>''',
                unsafe_allow_html=True,
            )
        elif not st.session_state.lpt_quotes:
            st.markdown(
                '<div class="status-box status-info">Holdings loaded — hit <strong>Refresh Quotes</strong> to fetch live prices.</div>',
                unsafe_allow_html=True,
            )
        else:
            # ── Market status banner ────────────────────────────
            _lpt_live_count  = sum(1 for q in st.session_state.lpt_quotes.values() if q.get("is_live"))
            _lpt_total_q     = sum(1 for q in st.session_state.lpt_quotes.values() if q.get("ok"))
            _lpt_stale_count = _lpt_total_q - _lpt_live_count
            if _lpt_total_q > 0:
                if _lpt_live_count == _lpt_total_q:
                    st.markdown(
                        '<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(0,230,118,0.08);' +
                        'border:0.5px solid #00e676;border-radius:8px;padding:5px 14px;font-size:0.75rem;color:#00e676;margin-bottom:10px;">' +
                        '<span style="width:7px;height:7px;border-radius:50%;background:#00e676;display:inline-block;box-shadow:0 0 6px #00e676"></span>' +
                        'Market Open — live prices</div>',
                        unsafe_allow_html=True,
                    )
                elif _lpt_live_count == 0:
                    # All stale — find the most recent date across quotes
                    _lpt_dates = [
                        q.get("price_label","").replace("Last Close (","").replace(")","")
                        for q in st.session_state.lpt_quotes.values()
                        if q.get("ok") and "Last Close" in q.get("price_label","")
                    ]
                    _lpt_last_date = _lpt_dates[0] if _lpt_dates else "last trading day"
                    st.markdown(
                        '<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(255,213,79,0.08);' +
                        'border:0.5px solid #ffd54f;border-radius:8px;padding:5px 14px;font-size:0.75rem;color:#ffd54f;margin-bottom:10px;">' +
                        '<span style="width:7px;height:7px;border-radius:50%;background:#ffd54f;display:inline-block;"></span>' +
                        'Market Closed — showing last close ({})</div>'.format(_lpt_last_date),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(255,213,79,0.08);' +
                        'border:0.5px solid #ffd54f;border-radius:8px;padding:5px 14px;font-size:0.75rem;color:#ffd54f;margin-bottom:10px;">' +
                        '<span style="width:7px;height:7px;border-radius:50%;background:#ffd54f;display:inline-block;"></span>' +
                        '{} live · {} last close</div>'.format(_lpt_live_count, _lpt_stale_count),
                        unsafe_allow_html=True,
                    )
            _lpt_df, _lpt_sum = compute_portfolio_pnl(
                st.session_state.lpt_holdings,
                st.session_state.lpt_quotes,
            )

            # ── Summary metric row ──────────────────────────────
            _lpt_val_str  = "₹{:,.0f}".format(_lpt_sum["total_value"])
            _lpt_cost_str = "₹{:,.0f}".format(_lpt_sum["total_cost"])
            _lpt_unr_str  = "₹{:+,.0f}".format(_lpt_sum["total_unr"])
            _lpt_pct_str  = "{:+.2f}%".format(_lpt_sum["total_unr_pct"])
            _lpt_day_str  = "₹{:+,.0f}".format(_lpt_sum["total_day_chg"])
            _lpt_pnl_cls  = "positive" if _lpt_sum["total_unr"] >= 0 else ""
            _lpt_day_cls  = "positive" if _lpt_sum["total_day_chg"] >= 0 else ""
            st.markdown(
                "<div class=\"metric-row\">"
                + metric_card("Portfolio Value",  _lpt_val_str,  "accent")
                + metric_card("Total Invested",   _lpt_cost_str, "")
                + metric_card("Unrealised P&L",   _lpt_unr_str,  _lpt_pnl_cls)
                + metric_card("Return %",         _lpt_pct_str,  _lpt_pnl_cls)
                + metric_card("Day Change",        _lpt_day_str,  _lpt_day_cls)
                + metric_card("Positions",         str(_lpt_sum["n_holdings"]), "")
                + "</div>",
                unsafe_allow_html=True,
            )

            # Winners / Losers badge row
            st.markdown(
                '<div style="display:flex;gap:1rem;margin:0.6rem 0 1rem;flex-wrap:wrap;">' +
                '<div style="background:rgba(0,230,118,0.1);border:0.5px solid #00e676;border-radius:8px;padding:4px 14px;font-size:0.75rem;color:#00e676;">🟢 {} winner{}</div>'.format(_lpt_sum["n_winners"], "s" if _lpt_sum["n_winners"] != 1 else "") +
                '<div style="background:rgba(255,82,82,0.1);border:0.5px solid #ff5252;border-radius:8px;padding:4px 14px;font-size:0.75rem;color:#ff5252;">🔴 {} loser{}</div>'.format(_lpt_sum["n_losers"], "s" if _lpt_sum["n_losers"] != 1 else "") +
                '<div style="background:rgba(255,213,79,0.08);border:0.5px solid #ffd54f;border-radius:8px;padding:4px 14px;font-size:0.75rem;color:#ffd54f;">🏆 Best: {}</div>'.format(_lpt_sum["best_ticker"]) +
                '<div style="background:rgba(255,82,82,0.06);border:0.5px solid #ff5252;border-radius:8px;padding:4px 14px;font-size:0.75rem;color:#ff8a65;">📉 Worst: {}</div>'.format(_lpt_sum["worst_ticker"]) +
                '</div>',
                unsafe_allow_html=True,
            )

            # ── Holdings detail table ───────────────────────────
            # Rename "Live Price" col to reflect market state
            _lpt_price_col = "Live Price" if _lpt_live_count > 0 else "Last Close"
            _lpt_df_disp   = _lpt_df.rename(columns={"Live Price": _lpt_price_col})
            st.markdown('<div class="section-heading">Holdings Detail</div>', unsafe_allow_html=True)

            _lpt_display_df = _lpt_df_disp[[
                "Ticker","Shares","Avg Cost", _lpt_price_col,
                "Day Chg %","Cost Basis","Curr Value",
                "Unrealised ₹","Unrealised %","52W High","52W Low",
            ]].copy()

            def _lpt_colour_unr(val):
                try:
                    v = float(val)
                    if v > 10:   return "color:#00e676;font-weight:600"
                    if v > 0:    return "color:#b9f6ca"
                    if v < -10:  return "color:#ff5252;font-weight:600"
                    if v < 0:    return "color:#ff8a65"
                except Exception: pass
                return ""

            def _lpt_colour_day(val):
                try:
                    v = float(val)
                    return "color:#00e676" if v >= 0 else "color:#ff5252"
                except Exception: return ""

            st.dataframe(
                _lpt_display_df.style
                    .applymap(_lpt_colour_unr, subset=["Unrealised %"])
                    .applymap(_lpt_colour_day, subset=["Day Chg %"])
                    .format({
                        "Avg Cost":          "{:,.2f}",
                        _lpt_price_col:      "{:,.2f}",
                        "Cost Basis":   "₹{:,.0f}",
                        "Curr Value":   "₹{:,.0f}",
                        "Unrealised ₹": "₹{:+,.0f}",
                        "Unrealised %": "{:+.2f}%",
                        "Day Chg %":    "{:+.2f}%",
                        "52W High":     "{:,.2f}",
                        "52W Low":      "{:,.2f}",
                    }, na_rep="—"),
                use_container_width=True,
                height=min(500, 60 + 35 * len(_lpt_display_df)),
            )

            # ── Allocation donut chart ──────────────────────────
            st.markdown("---")
            _lpt_chart_c1, _lpt_chart_c2 = st.columns(2)

            with _lpt_chart_c1:
                st.markdown('<div class="section-heading">Allocation</div>', unsafe_allow_html=True)
                _lpt_valid_df = _lpt_df[_lpt_df["Curr Value"].notna()].copy()
                if not _lpt_valid_df.empty:
                    fig_donut = go.Figure(go.Pie(
                        labels=_lpt_valid_df["Ticker"],
                        values=_lpt_valid_df["Curr Value"],
                        hole=0.52,
                        textinfo="label+percent",
                        textfont=dict(size=10),
                        marker=dict(
                            colors=["hsl({},60%,52%)".format(int(i*360/len(_lpt_valid_df))) for i in range(len(_lpt_valid_df))],
                            line=dict(color="#0a0e17", width=2),
                        ),
                    ))
                    fig_donut.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0a0e17",
                        showlegend=False,
                        height=300,
                        margin=dict(l=10, r=10, t=20, b=10),
                        annotations=[dict(
                            text="₹{:,.0f}".format(_lpt_sum["total_value"]),
                            x=0.5, y=0.5, font_size=13, showarrow=False,
                            font=dict(color="#a0c8e8", family="DM Mono"),
                        )],
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

            with _lpt_chart_c2:
                st.markdown('<div class="section-heading">Unrealised P&L by Position</div>', unsafe_allow_html=True)
                _lpt_pnl_df = _lpt_df[_lpt_df["Unrealised ₹"].notna()].sort_values("Unrealised ₹")
                if not _lpt_pnl_df.empty:
                    _lpt_bar_colors = ["#00e676" if v >= 0 else "#ff5252" for v in _lpt_pnl_df["Unrealised ₹"]]
                    fig_pnl = go.Figure(go.Bar(
                        x=_lpt_pnl_df["Unrealised ₹"],
                        y=_lpt_pnl_df["Ticker"],
                        orientation="h",
                        marker_color=_lpt_bar_colors,
                        text=["₹{:+,.0f}".format(v) for v in _lpt_pnl_df["Unrealised ₹"]],
                        textposition="outside",
                        textfont=dict(size=9),
                        hovertemplate="<b>%{y}</b><br>₹%{x:+,.0f}<extra></extra>",
                    ))
                    fig_pnl.add_vline(x=0, line_color="rgba(255,255,255,0.15)")
                    fig_pnl.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0a0e17",
                        plot_bgcolor="#0d1525",
                        xaxis_title="Unrealised P&L (₹)",
                        font=dict(family="DM Mono", color="#80b0d0"),
                        height=300,
                        margin=dict(l=10, r=40, t=20, b=30),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)

            # ── 52-week position chart ──────────────────────────
            _lpt_has_52w = _lpt_df["52W High"].notna().any()
            if _lpt_has_52w and not is_noob:
                st.markdown('<div class="section-heading">52-Week Price Position</div>', unsafe_allow_html=True)
                render_insight_note("The dot shows where the current price sits between the 52-week low and high. Far right = near highs, far left = near lows.")
                _lpt_52_df = _lpt_df[
                    _lpt_df["52W High"].notna() &
                    _lpt_df["52W Low"].notna() &
                    _lpt_df["Live Price"].notna()
                ].copy()
                if not _lpt_52_df.empty:
                    _lpt_52_df["Position %"] = (
                        (_lpt_52_df["Live Price"] - _lpt_52_df["52W Low"]) /
                        (_lpt_52_df["52W High"] - _lpt_52_df["52W Low"]).replace(0, np.nan) * 100
                    ).clip(0, 100)
                    fig_52 = go.Figure()
                    for _, _row in _lpt_52_df.iterrows():
                        _pos = _row["Position %"]
                        _col = "#00e676" if _pos > 60 else "#ffd54f" if _pos > 30 else "#ff5252"
                        fig_52.add_trace(go.Scatter(
                            x=[0, 100],
                            y=[_row["Ticker"], _row["Ticker"]],
                            mode="lines",
                            line=dict(color="rgba(255,255,255,0.08)", width=8),
                            showlegend=False, hoverinfo="skip",
                        ))
                        fig_52.add_trace(go.Scatter(
                            x=[_pos],
                            y=[_row["Ticker"]],
                            mode="markers+text",
                            marker=dict(size=14, color=_col, line=dict(color="#0a0e17", width=2)),
                            text=["{:.0f}%".format(_pos)],
                            textposition="middle right",
                            textfont=dict(size=9, color=_col),
                            showlegend=False,
                            hovertemplate="<b>{}</b><br>Price: ₹{:,.2f}<br>52W Low: ₹{:,.2f}<br>52W High: ₹{:,.2f}<br>Position: {:.0f}%<extra></extra>".format(
                                _row["Ticker"], _row["Live Price"], _row["52W Low"], _row["52W High"], _pos
                            ),
                        ))
                    fig_52.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0a0e17",
                        plot_bgcolor="#0d1525",
                        xaxis=dict(title="Position between 52W Low → High (%)", range=[-5, 115],
                                   tickvals=[0, 25, 50, 75, 100],
                                   ticktext=["52W Low", "25%", "50%", "75%", "52W High"]),
                        yaxis=dict(title=""),
                        font=dict(family="DM Mono", color="#80b0d0"),
                        height=max(220, len(_lpt_52_df) * 38 + 80),
                        margin=dict(l=20, r=60, t=20, b=50),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_52, use_container_width=True)

            # ── Drift & Rebalance (quant only) ──────────────────
            if not is_noob:
                st.markdown("---")
                st.markdown('<div class="section-heading">Weight Drift vs Optimizer Portfolio</div>', unsafe_allow_html=True)

                _lpt_port_opts = {}
                if st.session_state.get("weights_tan") is not None:
                    _lpt_port_opts["Tangency (Max Sharpe)"] = st.session_state.weights_tan
                if st.session_state.get("weights_utility") is not None:
                    _lpt_port_opts["Utility Maximiser"]     = st.session_state.weights_utility
                if st.session_state.get("weights_min") is not None:
                    _lpt_port_opts["Min-Risk"]               = st.session_state.weights_min

                if not _lpt_port_opts:
                    st.markdown('<div class="status-box status-info">Run an optimisation first to compare drift against a target portfolio.</div>', unsafe_allow_html=True)
                else:
                    _lpt_drift_port = st.selectbox(
                        "Compare drift against",
                        list(_lpt_port_opts.keys()),
                        key="lpt_drift_port_sel",
                    )
                    _lpt_tgt_w_arr = _lpt_port_opts[_lpt_drift_port]
                    _lpt_tickers   = st.session_state.tickers
                    _lpt_tgt_dict  = {
                        _lpt_tickers[i]: float(_lpt_tgt_w_arr[i])
                        for i in range(min(len(_lpt_tickers), len(_lpt_tgt_w_arr)))
                    }

                    _lpt_drift_df = compute_drift(
                        st.session_state.lpt_holdings,
                        st.session_state.lpt_quotes,
                        target_weights=_lpt_tgt_dict,
                    )

                    if not _lpt_drift_df.empty:
                        def _lpt_col_drift(val):
                            try:
                                v = float(val)
                                if v >  5: return "color:#ff8a65;font-weight:600"
                                if v >  2: return "color:#ffd54f"
                                if v < -5: return "color:#ff5252;font-weight:600"
                                if v < -2: return "color:#ff8a65"
                                return "color:#00e676"
                            except Exception: return ""

                        st.dataframe(
                            _lpt_drift_df.style
                                .applymap(_lpt_col_drift, subset=["Drift pp"])
                                .format({
                                    "Curr Value":      "₹{:,.0f}",
                                    "Curr Weight %":   "{:.2f}%",
                                    "Target Weight %": "{:.2f}%",
                                    "Drift pp":        "{:+.2f}pp",
                                }, na_rep="—"),
                            use_container_width=True,
                            height=min(400, 60 + 35 * len(_lpt_drift_df)),
                        )

                        # Drift bar chart
                        _lpt_drifts = _lpt_drift_df.dropna(subset=["Drift pp"])
                        if not _lpt_drifts.empty:
                            _lpt_drift_colors = [
                                "#ff5252" if d < -2 else "#ffd54f" if abs(d) <= 2 else "#ff8a65"
                                for d in _lpt_drifts["Drift pp"]
                            ]
                            fig_drift = go.Figure(go.Bar(
                                x=_lpt_drifts["Ticker"],
                                y=_lpt_drifts["Drift pp"],
                                marker_color=_lpt_drift_colors,
                                text=["{:+.1f}pp".format(d) for d in _lpt_drifts["Drift pp"]],
                                textposition="outside",
                                textfont=dict(size=9),
                                hovertemplate="<b>%{x}</b><br>Drift: %{y:+.2f}pp<extra></extra>",
                            ))
                            fig_drift.add_hline(y=0, line_color="rgba(255,255,255,0.15)")
                            fig_drift.add_hline(y=2,  line_dash="dot", line_color="rgba(255,213,79,0.3)")
                            fig_drift.add_hline(y=-2, line_dash="dot", line_color="rgba(255,213,79,0.3)")
                            fig_drift.update_layout(
                                template="plotly_dark",
                                paper_bgcolor="#0a0e17", plot_bgcolor="#0d1525",
                                yaxis_title="Drift (percentage points)",
                                font=dict(family="DM Mono", color="#80b0d0"),
                                height=280,
                                margin=dict(l=50, r=20, t=20, b=50),
                                showlegend=False,
                            )
                            st.plotly_chart(fig_drift, use_container_width=True)
                            st.caption("Dashed lines at ±2pp. Outside those bands = consider rebalancing.")

                    # Rebalance trade list
                    st.markdown("---")
                    st.markdown('<div class="section-heading">Rebalancing Trade List</div>', unsafe_allow_html=True)
                    render_insight_note(
                        "Trades required to move from your current actual holdings back to the target weights. "
                        "Uses current live prices. Toggle fractional shares below."
                    )

                    _lpt_frac = st.toggle("Allow fractional shares", value=False, key="lpt_frac_rebal")
                    _lpt_extra_cap = st.number_input(
                        "Additional capital to deploy (₹)",
                        min_value=0.0, value=0.0, step=1000.0, format="%.0f",
                        key="lpt_extra_cap",
                        help="Add fresh capital on top of current portfolio value before rebalancing.",
                    )

                    _lpt_rebal_total = (
                        sum(
                            h["shares"] * st.session_state.lpt_quotes.get(h["ticker"], {}).get("price", 0)
                            for h in st.session_state.lpt_holdings
                            if st.session_state.lpt_quotes.get(h["ticker"], {}).get("ok")
                        ) + _lpt_extra_cap
                    )

                    _lpt_trades_df = compute_rebalance_trades(
                        st.session_state.lpt_holdings,
                        st.session_state.lpt_quotes,
                        target_weights=_lpt_tgt_dict,
                        total_capital=_lpt_rebal_total if _lpt_rebal_total > 0 else None,
                        allow_fractional=_lpt_frac,
                    )

                    if not _lpt_trades_df.empty:
                        _lpt_buy_val  = _lpt_trades_df.loc[_lpt_trades_df["Action"] == "BUY",  "Trade Value"].sum()
                        _lpt_sell_val = _lpt_trades_df.loc[_lpt_trades_df["Action"] == "SELL", "Trade Value"].sum()
                        _lpt_tr1, _lpt_tr2, _lpt_tr3 = st.columns(3)
                        with _lpt_tr1:
                            st.markdown('<div style="background:rgba(0,230,118,0.08);border:0.5px solid #00e676;border-radius:10px;padding:0.8rem;text-align:center;"><div style="font-size:0.62rem;color:#00e676;text-transform:uppercase;letter-spacing:0.1em">Total Buys</div><div style="font-size:1.2rem;font-weight:700;color:#00e676;font-family:DM Mono,monospace">₹{:,.0f}</div></div>'.format(_lpt_buy_val), unsafe_allow_html=True)
                        with _lpt_tr2:
                            st.markdown('<div style="background:rgba(255,82,82,0.08);border:0.5px solid #ff5252;border-radius:10px;padding:0.8rem;text-align:center;"><div style="font-size:0.62rem;color:#ff5252;text-transform:uppercase;letter-spacing:0.1em">Total Sells</div><div style="font-size:1.2rem;font-weight:700;color:#ff5252;font-family:DM Mono,monospace">₹{:,.0f}</div></div>'.format(_lpt_sell_val), unsafe_allow_html=True)
                        with _lpt_tr3:
                            st.markdown('<div style="background:rgba(0,180,255,0.06);border:0.5px solid #1a2d4d;border-radius:10px;padding:0.8rem;text-align:center;"><div style="font-size:0.62rem;color:#4a90d9;text-transform:uppercase;letter-spacing:0.1em">Net Cash Flow</div><div style="font-size:1.2rem;font-weight:700;color:#80c8ff;font-family:DM Mono,monospace">₹{:+,.0f}</div></div>'.format(_lpt_buy_val - _lpt_sell_val), unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)

                        def _lpt_colour_action(val):
                            return "color:#00e676;font-weight:600" if val == "BUY" else "color:#ff5252;font-weight:600"

                        st.dataframe(
                            _lpt_trades_df.style
                                .applymap(_lpt_colour_action, subset=["Action"])
                                .format({
                                    "Trade Value":   "₹{:,.0f}",
                                    "Price":         "{:,.2f}",
                                    "Curr Weight%":  "{:.2f}%",
                                    "Tgt Weight%":   "{:.2f}%",
                                }),
                            use_container_width=True,
                            height=min(400, 60 + 35 * len(_lpt_trades_df)),
                        )
                        st.download_button(
                            "⬇ Download Trade List CSV",
                            data=_lpt_trades_df.to_csv(index=False).encode(),
                            file_name="quansen_rebalance_trades.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="lpt_dl_trades",
                        )
                    else:
                        st.markdown('<div class="status-box status-success">✅ Portfolio is already aligned with target weights — no trades needed.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 7 — EXPORT  (both modes)
# ══════════════════════════════════════════════════════════════
if _qs_s == "export":
    if is_noob:
        st.markdown('<div class="nb-section">💾 Save Your Results</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="nb-info-box">Once you\'ve run the optimizer, you can download your results here. '
            'Share them with a friend, open them in Excel, or just keep them for your records!</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div class="section-heading">Export Results</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.markdown('<div class="status-box status-info">Load data and run optimizations to export results.</div>', unsafe_allow_html=True)
    else:
        # ── PDF Report ─────────────────────────────────────────
        st.markdown('<div class="card-title">📄 Full PDF Report</div>', unsafe_allow_html=True)
        has_bt = st.session_state.bt_results is not None
        has_opt = any([
            st.session_state.weights_utility is not None,
            st.session_state.weights_tan is not None,
            st.session_state.weights_min is not None,
        ])
        if not has_opt:
            st.markdown('<div class="status-box status-warn">Run at least one optimisation to generate the PDF.</div>', unsafe_allow_html=True)
        else:
            if not has_bt:
                st.caption("ℹ Run a backtest first to include the Trade Blotter section in the PDF.")
            if st.button("📄 Generate & Download Full PDF Report", use_container_width=True, key="pdf_btn"):
                with st.spinner("Building PDF…"):
                    try:
                        pdf_bytes = generate_pdf(st.session_state)
                        st.download_button(
                            label="⬇ Click here to download  quansen_report.pdf",
                            data=pdf_bytes,
                            file_name="quansen_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="pdf_download"
                        )
                        st.success("✔ PDF ready — click the button above to download.")
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        st.markdown("---")
        tickers = st.session_state.tickers
        er      = optimizer_expected_returns()
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
            ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
            lines += [
                "UTILITY-MAXIMIZED PORTFOLIO",
                "-" * 40,
                *[f"  {t:<20} {w_:.4f}  ({w_*100:.2f}%)" for t, w_ in zip(tickers, w)],
                f"  Annual Return  : {ann_ret*100:.2f}%",
                f"  Annual Risk    : {ann_risk*100:.2f}%",
                f"  Sharpe Ratio   : {sharpe:.3f}",
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
            ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
            lines += [
                f"MIN-RISK PORTFOLIO (Target: {st.session_state.target_return*100:.1f}%)",
                "-" * 40,
                *[f"  {t:<20} {w_:.4f}  ({w_*100:.2f}%)" for t, w_ in zip(tickers, w)],
                f"  Annual Return  : {ann_ret*100:.2f}%",
                f"  Annual Risk    : {ann_risk*100:.2f}%",
                f"  Sharpe Ratio   : {sharpe:.3f}",
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
