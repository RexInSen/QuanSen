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


# ── Thresholds for signal classification ─────────────────────
# Based on cross-sectional z-score after normalisation
_STRONG_THRESH =  0.50   # z > +0.50  →  Strong  🟢
_WEAK_THRESH   = -0.50   # z < -0.50  →  Weak    🔴
# else                   →  Neutral   🟡


def compute_momentum(
    tickers:       list,
    start_date:    str,
    end_date:      str,
    shrink_er:     "pd.Series",
    lookback:      int = 252,
    skip:          int = 21,
    beta:          float = 0.60,
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

    Returns
    -------
    scores   : pd.Series — raw momentum return per ticker  (index=tickers)
    signals  : pd.Series — 'Strong' / 'Neutral' / 'Weak'  (index=tickers)
    final_er : pd.Series — blended daily expected returns  (index=tickers)
    meta     : dict      — effective momentum window metadata
    """

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
        }
        return zeros, neutral, shrink_er.copy(), meta

    # ── Step 2: Identify reference dates ───────────────────────
    # exit_idx  = most recent date (row index -1 after skip)
    # entry_idx = exit_idx - lookback rows
    trading_days = prices.index

    # Anchor momentum to the selected sample end date, not the machine's current date.
    # `trading_days[-1]` is the last available row on or before the user-selected end date.
    exit_pos  = len(trading_days) - 1 - skip      # skip last `skip` trading days from sample end
    entry_pos = exit_pos - lookback                # go back `lookback` trading days

    if entry_pos < 0:
        entry_pos = 0   # graceful fallback for short histories

    exit_date  = trading_days[exit_pos]
    entry_date = trading_days[entry_pos]

    # ── Step 3: Compute raw momentum per ticker ─────────────────
    raw_momentum = {}
    for t in valid:
        col = prices[t].dropna()
        if col.index[-1] < exit_date or col.index[0] > entry_date:
            raw_momentum[t] = np.nan
            continue
        p_exit  = float(col.asof(exit_date))
        p_entry = float(col.asof(entry_date))
        if np.isnan(p_exit) or np.isnan(p_entry) or p_entry <= 0:
            raw_momentum[t] = np.nan
        else:
            raw_momentum[t] = p_exit / p_entry - 1.0

    scores_raw = pd.Series(raw_momentum, dtype=float)

    # Fill any NaN tickers with cross-sectional mean (neutral fallback)
    cs_mean = scores_raw.mean()
    scores_raw = scores_raw.fillna(cs_mean if not np.isnan(cs_mean) else 0.0)

    # Reindex to match input tickers order (some may have been missing)
    scores_raw = scores_raw.reindex(tickers, fill_value=cs_mean if not np.isnan(cs_mean) else 0.0)

    # ── Step 4: Cross-sectional z-score normalisation ───────────
    cs_std = scores_raw.std()
    if cs_std == 0 or np.isnan(cs_std):
        z_scores = pd.Series(0.0, index=tickers)
    else:
        z_scores = (scores_raw - scores_raw.mean()) / cs_std

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

    meta = {
        "entry_date": str(pd.Timestamp(entry_date).date()),
        "exit_date": str(pd.Timestamp(exit_date).date()),
        "lookback": lookback,
        "skip": skip,
        "end_date": str(_end_dt.date()),
    }

    return scores_raw, signals, final_er, meta
