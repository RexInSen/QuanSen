"""
============================================================
  QuanSen — Momentum Engine  (v1.1)
  Computes cross-sectional + time-series momentum signals
  and blends them with shrinkage-adjusted expected returns.

  Pipeline:
    1. Download price history for the lookback window
    2. Compute raw momentum:  Price[-skip] / Price[-lookback] - 1
    3. Cross-sectional z-score normalisation across all tickers
    4. Scale z-scores to return space (std of shrinkage returns)
    5. Blend:  μ_final = β × μ_shrinkage  +  (1-β) × momentum_scaled
    6. Classify signals: Strong / Neutral / Weak (by z-score tercile)

  Returns: (scores, signals, final_expected_returns, meta)
============================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st


# ── Thresholds for signal classification ─────────────────────
# Based on cross-sectional z-score after normalisation
_STRONG_THRESH =  0.50   # z > +0.50  →  Strong  🟢
_WEAK_THRESH   = -0.50   # z < -0.50  →  Weak    🔴
# else                   →  Neutral   🟡


def _infer_market_index(tickers: list) -> str:
    """Pick a regime anchor that matches the user's basket."""
    if not tickers:
        return "^GSPC"
    indian = sum(
        str(t).upper().endswith(".NS") or str(t).upper().endswith(".BO")
        for t in tickers
    )
    return "^NSEI" if indian >= max(1, len(tickers) / 2) else "^GSPC"


def _snap_lookback(value: float) -> int:
    """Snap to the nearest 21-trading-day bucket for cleaner UX."""
    buckets = np.array([42, 63, 84, 105, 126, 147, 168, 189, 210, 231, 252], dtype=int)
    return int(buckets[np.argmin(np.abs(buckets - value))])


def _zscore_cross_section(series: pd.Series, index: list[str]) -> pd.Series:
    """Cross-sectional z-score with stable zero fallback."""
    s = series.reindex(index)
    std = float(s.std())
    if std <= 1e-12 or np.isnan(std):
        return pd.Series(0.0, index=index, dtype=float)
    return ((s - float(s.mean())) / std).reindex(index, fill_value=0.0)


def _horizon_skip(horizon: int, base_skip: int) -> int:
    """Shorter horizons should not inherit the full long-horizon skip."""
    if horizon <= 10:
        return 1
    if horizon <= 21:
        return min(5, base_skip)
    if horizon <= 63:
        return min(10, base_skip)
    return base_skip


def _compute_horizon_momentum(prices: pd.DataFrame, tickers: list[str], horizon: int, base_skip: int) -> tuple[pd.Series, dict]:
    """Compute raw momentum for one horizon using a horizon-specific skip."""
    skip = _horizon_skip(horizon, base_skip)
    trading_days = prices.index
    exit_pos = len(trading_days) - 1 - skip
    entry_pos = max(0, exit_pos - horizon)
    exit_date = trading_days[exit_pos]
    entry_date = trading_days[entry_pos]

    raw = {}
    for t in tickers:
        if t not in prices.columns:
            raw[t] = np.nan
            continue
        col = prices[t].dropna()
        if col.empty or col.index[-1] < exit_date or col.index[0] > entry_date:
            raw[t] = np.nan
            continue
        p_exit = float(col.asof(exit_date))
        p_entry = float(col.asof(entry_date))
        if np.isnan(p_exit) or np.isnan(p_entry) or p_entry <= 0:
            raw[t] = np.nan
        else:
            raw[t] = p_exit / p_entry - 1.0

    ser = pd.Series(raw, dtype=float)
    cs_mean = ser.mean()
    ser = ser.fillna(cs_mean if not np.isnan(cs_mean) else 0.0).reindex(tickers, fill_value=0.0)
    meta = {
        "entry_date": str(pd.Timestamp(entry_date).date()),
        "exit_date": str(pd.Timestamp(exit_date).date()),
        "skip": int(skip),
    }
    return ser, meta


@st.cache_data(ttl=3600)
def dynamic_lookback(anchor, start_date, end_date, prev_regime_probs=None):
    """
    Regime-aware lookback selector.

    Instead of a hard-coded single-market heuristic, this now uses the
    probabilistic regime detector and converts regime probabilities into a
    weighted lookback. That makes the auto mode much less likely to get
    stuck at 252d unless the regime is genuinely very bullish.
    """
    if isinstance(anchor, str):
        tickers = [anchor]
    else:
        tickers = list(anchor)

    try:
        from engines.market_state_engine import detect_market_state

        _, _, regime_meta = detect_market_state(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            prev_regime_probs=prev_regime_probs,
        )
        probs = regime_meta.get("regime_probabilities", {})
        bull = float(probs.get("bull", 0.25))
        sideways = float(probs.get("sideways", 0.25))
        bear = float(probs.get("bear", 0.25))
        crisis = float(probs.get("crisis", 0.25))
        basket = regime_meta.get("basket", {})
        participation = float(basket.get("participation_pct", 50.0)) / 100.0
        avg_corr_63 = float(basket.get("avg_corr_63", 0.35))
        downside_share = float(basket.get("downside_pct", 25.0)) / 100.0
        confidence = float(regime_meta.get("confidence", 0.0))

        # Base regime-weighted anchor:
        # bull still supports long trend capture, but sideways/bear/crisis now
        # pull much shorter than before so auto mode reacts faster.
        base_lookback = (
            231 * bull +
            126 * sideways +
            63 * bear +
            42 * crisis
        )

        # Basket-level stress shortens the window aggressively when the basket
        # itself is deteriorating even if the broad index is not in full crisis.
        downside_penalty = 63.0 * max(downside_share - 0.25, 0.0) / 0.75
        corr_penalty = 42.0 * max(avg_corr_63 - 0.45, 0.0) / 0.40
        participation_penalty = 42.0 * max(0.55 - participation, 0.0) / 0.55
        conviction_bonus = 21.0 * max(bull - 0.55, 0.0) * max(0.0, 1.0 - downside_share)
        confidence_bonus = 10.0 * max(confidence - 0.20, 0.0) * max(0.0, bull - bear)

        raw_lookback = base_lookback - downside_penalty - corr_penalty - participation_penalty
        raw_lookback += conviction_bonus + confidence_bonus
        raw_lookback = float(np.clip(raw_lookback, 42.0, 252.0))
        lookback = _snap_lookback(raw_lookback)
        skip = 21

        meta = {
            "market_index": _infer_market_index(tickers),
            "dominant_regime": regime_meta.get("dominant_regime", "unknown"),
            "confidence": float(regime_meta.get("confidence", 0.0)),
            "regime_probabilities": probs,
            "base_lookback": float(base_lookback),
            "lookback_raw": float(raw_lookback),
            "lookback": int(lookback),
            "skip": int(skip),
            "drivers": {
                "bull_anchor": round(231 * bull, 1),
                "sideways_anchor": round(126 * sideways, 1),
                "bear_anchor": round(63 * bear, 1),
                "crisis_anchor": round(42 * crisis, 1),
                "downside_penalty": round(downside_penalty, 1),
                "corr_penalty": round(corr_penalty, 1),
                "participation_penalty": round(participation_penalty, 1),
                "conviction_bonus": round(conviction_bonus, 1),
                "confidence_bonus": round(confidence_bonus, 1),
            },
        }
        return lookback, skip, meta
    except Exception:
        return 126, 21, {
            "market_index": _infer_market_index(tickers),
            "dominant_regime": "unknown",
            "confidence": 0.0,
            "regime_probabilities": {},
            "base_lookback": 126.0,
            "lookback_raw": 126.0,
            "lookback": 126,
            "skip": 21,
            "drivers": {},
        }

def compute_momentum(
    tickers:       list,
    start_date:    str,
    end_date:      str,
    shrink_er:     "pd.Series",
    lookback:      int = 252,
    skip:          int = 21,
    beta:          float = 0.60,
    auto_lookback: bool = False,
    prev_regime_probs: dict | None = None,
) -> tuple:
    """
    Parameters
    ----------
    tickers    : list of ticker strings (sorted, same order as shrink_er)
    start_date : data window start (YYYY-MM-DD) — used to determine end ref date
    end_date   : data window end   (YYYY-MM-DD)
    shrink_er  : pd.Series of shrinkage-adjusted daily expected returns (index=tickers)
    lookback   : trading days to look back from (end_date - skip). Default 252.
    skip       : trading days to exclude before end_date. Default 21 (1 month).
    beta       : weight on shrink_er vs momentum signal. 1.0 = ignore momentum.
    auto_lookback : if True, use dynamic_lookback() to select lookback/skip from
                    market regime instead of the caller-supplied values.

    Returns
    -------
    scores   : pd.Series — raw momentum return per ticker  (index=tickers)
    signals  : pd.Series — 'Strong' / 'Neutral' / 'Weak'  (index=tickers)
    final_er : pd.Series — blended daily expected returns  (index=tickers)
    meta     : dict      — effective momentum window metadata
    """
    # ── Regime-aware lookback ──────────────────────────────────────
    if auto_lookback:
        lookback, skip, regime_meta = dynamic_lookback(
            tickers,
            start_date,
            end_date,
            prev_regime_probs=prev_regime_probs,
        )
    else:
        regime_meta = {
            "vol_ratio":  None,
            "trend_dist": None,
            "drawdown":   None,
            "lookback":   lookback,
            "skip":       skip,
        }

    # ── Step 1: Download prices ─────────────────────────────────
    # Need (lookback + skip) trading days before the selected sample end date.
    # Trading days ≈ 252/year → multiply by 1.45 calendar-day buffer
    # for weekends, holidays, and gaps.
    # IMPORTANT: never mix `period` with `end` in yfinance — use start+end only.
    import datetime as _dt
    _end_dt   = pd.Timestamp(end_date)
    _cal_days = int((lookback + skip) * 1.45) + 30   # generous buffer
    _start_dt = _end_dt - pd.Timedelta(days=_cal_days)
    _fetch_start = _start_dt.strftime("%Y-%m-%d")
    _fetch_end = (_end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    data = yf.download(
        tickers,
        start=_fetch_start,
        end=_fetch_end,
        auto_adjust=True, progress=False
    )

    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]] if "Close" in data.columns else data

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    # Keep only requested tickers that have data
    valid = [t for t in tickers if t in prices.columns]
    prices = prices[valid].ffill().bfill().dropna(how="all")

    if len(prices) < lookback + skip + 5:
        # Not enough history — return raw shrinkage, neutral signals
        neutral = pd.Series("Neutral", index=tickers)
        zeros   = pd.Series(0.0,       index=tickers)
        meta = {
            "entry_date": None,
            "exit_date": None,
            "lookback": lookback,
            "skip": skip,
            "end_date": str(_end_dt.date()),
            "auto_lookback": bool(auto_lookback),
            "regime_meta": regime_meta,
        }
        return zeros, neutral, shrink_er.copy(), meta

    # ── Step 2: Tactical multi-horizon momentum ────────────────
    slow_probs = regime_meta.get("regime_probabilities", {})
    fast_probs = regime_meta.get("fast_regime_probabilities", slow_probs)

    base_horizon_weights = {
        5:   0.22 * fast_probs.get("bull", 0.0) + 0.30 * fast_probs.get("bear", 0.0) + 0.35 * fast_probs.get("crisis", 0.0) + 0.10 * slow_probs.get("sideways", 0.0),
        10:  0.22 * fast_probs.get("bull", 0.0) + 0.25 * fast_probs.get("bear", 0.0) + 0.20 * fast_probs.get("crisis", 0.0) + 0.12 * slow_probs.get("sideways", 0.0),
        21:  0.15 * slow_probs.get("bull", 0.0) + 0.24 * fast_probs.get("bull", 0.0) + 0.22 * slow_probs.get("sideways", 0.0) + 0.18 * fast_probs.get("sideways", 0.0) + 0.15 * fast_probs.get("bear", 0.0),
        63:  0.34 * slow_probs.get("bull", 0.0) + 0.26 * slow_probs.get("sideways", 0.0) + 0.18 * fast_probs.get("bull", 0.0) + 0.08 * fast_probs.get("bear", 0.0),
        126: 0.38 * slow_probs.get("bull", 0.0) + 0.18 * slow_probs.get("sideways", 0.0) + 0.10 * fast_probs.get("bull", 0.0),
    }
    horizon_weight_sum = sum(base_horizon_weights.values())
    if horizon_weight_sum <= 1e-12:
        horizon_weights = {21: 0.35, 63: 0.40, 126: 0.25}
    else:
        horizon_weights = {h: w / horizon_weight_sum for h, w in base_horizon_weights.items() if w > 1e-8}

    horizon_raw_scores = {}
    horizon_z_scores = {}
    horizon_meta = {}
    for horizon, weight_h in horizon_weights.items():
        raw_h, meta_h = _compute_horizon_momentum(prices, tickers, horizon, skip)
        horizon_raw_scores[horizon] = raw_h
        horizon_z_scores[horizon] = _zscore_cross_section(raw_h, tickers)
        horizon_meta[horizon] = meta_h

    z_scores = pd.Series(0.0, index=tickers, dtype=float)
    scores_raw = pd.Series(0.0, index=tickers, dtype=float)
    for horizon, weight_h in horizon_weights.items():
        z_scores = z_scores.add(horizon_z_scores[horizon] * weight_h, fill_value=0.0)
        scores_raw = scores_raw.add(horizon_raw_scores[horizon] * weight_h, fill_value=0.0)

    # Tactical impulse: react to very recent regime shifts without replacing the
    # slow long-horizon anchor. Small regime changes now matter, but only as an overlay.
    fast_bull = float(fast_probs.get("bull", 0.25))
    fast_bear = float(fast_probs.get("bear", 0.25))
    fast_crisis = float(fast_probs.get("crisis", 0.25))
    fast_sideways = float(fast_probs.get("sideways", 0.25))
    tactical_impulse = float(
        0.18 * fast_bull
        - 0.12 * fast_sideways
        - 0.20 * fast_bear
        - 0.28 * fast_crisis
    )
    tactical_strength = float(regime_meta.get("tactical_momentum_weight", 0.20))
    if 5 in horizon_z_scores:
        z_scores = z_scores + tactical_strength * tactical_impulse * horizon_z_scores[5]
    if 10 in horizon_z_scores:
        z_scores = z_scores + 0.5 * tactical_strength * tactical_impulse * horizon_z_scores[10]

    # Re-normalise the blended score.
    z_scores = _zscore_cross_section(z_scores, tickers)

    # ── Step 5: Classify signals ────────────────────────────────
    def _classify(z):
        if z > _STRONG_THRESH:  return "Strong"
        if z < _WEAK_THRESH:    return "Weak"
        return "Neutral"

    signals = z_scores.map(_classify)

    # ── Step 6: Scale z-scores to return space ──────────────────
    # We want momentum_scaled to live in the same daily-return space
    # as shrink_er, but anchored to a ROBUST scale — not the raw std
    # of shrink_er, which is contaminated by outliers like KRISHNADEF.
    #
    # Robust anchor: median absolute deviation of shrink_er, converted
    # to std-equivalent (MAD × 1.4826 ≈ σ for a Normal distribution).
    # This is insensitive to the 1-2 extreme stocks that inflate raw std.
    #
    # Further cap: daily scale cannot exceed 0.0015 (≈ 37% ann.) —
    # momentum should tilt weights, not dominate them.
    mad = float(np.median(np.abs(shrink_er.values - np.median(shrink_er.values))))
    robust_std = mad * 1.4826
    if robust_std < 1e-7 or np.isnan(robust_std):
        # Fallback: 10th–90th percentile range / 2.56
        p10, p90 = np.percentile(shrink_er.values, [10, 90])
        robust_std = max((p90 - p10) / 2.56, 1e-6)

    scale = min(robust_std, 0.0015)   # cap at ~37% ann.
    momentum_scaled = z_scores * scale   # daily return units, bounded

    # ── Step 7: Blend ───────────────────────────────────────────
    # μ_final = β × μ_shrinkage  +  (1-β) × momentum_scaled
    final_er = beta * shrink_er + (1.0 - beta) * momentum_scaled

    # Sanity clip: ±0.5% daily = ±126% ann. Hard ceiling.
    final_er = final_er.clip(-0.005, 0.005)

    primary_horizon = max(horizon_weights, key=horizon_weights.get)
    primary_meta = horizon_meta.get(primary_horizon, {})
    meta = {
        "entry_date": primary_meta.get("entry_date"),
        "exit_date": primary_meta.get("exit_date"),
        "lookback": lookback,
        "skip": skip,
        "end_date": str(_end_dt.date()),
        "auto_lookback": bool(auto_lookback),
        "regime_meta": regime_meta,
        "fast_regime_meta": {
            "dominant": regime_meta.get("fast_dominant_regime", regime_meta.get("dominant_regime", "unknown")),
            "confidence": regime_meta.get("fast_confidence", regime_meta.get("confidence", 0.0)),
            "probabilities": fast_probs,
        },
        "tactical_strength": tactical_strength,
        "tactical_turnover_penalty": float(regime_meta.get("tactical_turnover_penalty", 0.0) or 0.0),
        "tactical_rebalance_ratio": float(regime_meta.get("tactical_rebalance_ratio", 1.0) or 1.0),
        "horizon_weights": {int(h): round(float(w), 4) for h, w in horizon_weights.items()},
        "horizon_windows": {str(h): horizon_meta[h] for h in horizon_meta},
        "primary_horizon": int(primary_horizon),
    }

    return scores_raw, signals, final_er, meta
