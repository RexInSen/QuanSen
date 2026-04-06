"""
============================================================
QuanSen — Probabilistic Market State Engine  (v2.0)
============================================================


1.  Multi-timeframe trend  — EMA50/150/200 consensus + ADX-style
    directional strength replaces the single ad-hoc EMA150 formula.

2.  Volatility regime  — Vol-of-vol (realised vol acceleration) and a
    VIX-proxy (GARCH-1,1 approximation) are added alongside the
    short/long vol ratio.  Baseline is the 252-day vol so crisis
    spikes read correctly even when 120-day vol is already elevated.

3.  Drawdown quality  — separate 1yr / 3yr peaks, drawdown speed
    (how fast did we fall), and a recovery-momentum flag.

4.  Market breadth proxy  — percentage of the lookback window spent
    above the EMA150; a narrow index can look bullish while most
    stocks are already in a bear market.

5.  Credit / liquidity stress proxy  — yield-curve slope via the
    US10Y-US2Y spread (^TNX minus ^IRX) or GSEC10 for Indian
    portfolios.  Inversion is historically the single best leading
    recession indicator.

6.  Regime scoring — replaced arbitrary power laws with a
    calibrated logistic (sigmoid) combination of standardised
    z-scores.  Each feature is mapped to [-1, +1] before scoring
    so no single dimension can dominate because of scale.

7.  Regime persistence (EMA smoothing on probabilities) — prevents
    daily flip-flopping by blending the current reading with the
    exponentially-weighted history of the last 10 sessions.

8.  Graceful degradation  — every external data pull is wrapped so
    the engine still returns sensible defaults when a data source
    is unavailable (common for Indian yield-curve proxies).

9.  Richer meta dict  — all intermediate signals are returned so
    momentum_engine.py and portfolio_tool.py can log and plot them.

10. Bug fix  — the original `rolling_peak.iloc[-1]` could be a
    scalar OR a 1-element Series depending on the yfinance version.
    Now always cast to float explicitly.
============================================================
"""

from __future__ import annotations

import warnings
from functools import lru_cache
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Benchmark index + yield-curve tickers per market
_MARKET_CONFIG = {
    "india": {
        "index":      "^NSEI",
        "index_alt":  "^BSESN",
        "yc_long":    "^TNX",   # fallback to US 10Y when Indian not available
        "yc_short":   "^IRX",
    },
    "us": {
        "index":      "^GSPC",
        "index_alt":  "^NDX",
        "yc_long":    "^TNX",
        "yc_short":   "^IRX",
    },
}

# Regime persistence: EMA half-life in trading days
_PERSISTENCE_HALFLIFE = 10   # ~2 calendar weeks
_FAST_PERSISTENCE_HALFLIFE = 4
_MIXED_CONFIDENCE_THRESHOLD = 0.12
_BULL_PROB_MIXED_CAP = 0.58

# Sigmoid sharpness: controls how fast scores saturate
_SIG_K = 3.5


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _sig(x: float) -> float:
    """Sigmoid mapping ℝ → (0,1).  Centred so _sig(0)=0.5."""
    return 1.0 / (1.0 + np.exp(-_SIG_K * float(x)))


def _zscore_clip(series: pd.Series, window: int = 252, cap: float = 2.5) -> float:
    """Rolling z-score of the last observation, clipped to ±cap."""
    if len(series) < window // 2:
        return 0.0
    mu  = series.rolling(window, min_periods=window // 2).mean().iloc[-1]
    sig = series.rolling(window, min_periods=window // 2).std().iloc[-1]
    if sig <= 0 or np.isnan(sig) or np.isnan(mu):
        return 0.0
    return float(np.clip((series.iloc[-1] - mu) / sig, -cap, cap))


def _safe_float(x) -> float:
    """Convert scalar-or-Series to float safely."""
    if isinstance(x, pd.Series):
        x = x.iloc[-1]
    v = float(x)
    return 0.0 if np.isnan(v) else v


@lru_cache(maxsize=256)
def _fetch_cached(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download with auto_adjust; return empty DF on failure."""
    try:
        df = yf.download(
            symbol, start=start, end=end,
            auto_adjust=True, progress=False
        )
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = _fetch_cached(symbol, start, end)
    return df.copy() if not df.empty else pd.DataFrame()


def _extract_close_frame(data, requested_symbols: list[str]) -> pd.DataFrame:
    """Normalise yfinance output to a Close-price frame."""
    if data is None or getattr(data, "empty", True):
        return pd.DataFrame(columns=requested_symbols)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]] if "Close" in data.columns else data
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=requested_symbols[0])
    return prices


@lru_cache(maxsize=64)
def _fetch_price_frame_cached(symbols_tuple: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Download a basket of prices with single-symbol repair fallbacks."""
    symbols = list(dict.fromkeys(symbols_tuple))
    if not symbols:
        return pd.DataFrame()

    try:
        raw = yf.download(
            symbols, start=start, end=end,
            auto_adjust=True, progress=False, threads=True
        )
        prices = _extract_close_frame(raw, symbols)
    except Exception:
        prices = pd.DataFrame(columns=symbols)

    for symbol in symbols:
        if symbol in prices.columns and prices[symbol].dropna().size > 0:
            continue
        try:
            single = _fetch(symbol, start, end)
            single_prices = _extract_close_frame(single, [symbol])
            if symbol in single_prices.columns and single_prices[symbol].dropna().size > 0:
                prices[symbol] = single_prices[symbol]
        except Exception:
            continue

    return prices.sort_index()


def _fetch_price_frame(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    prices = _fetch_price_frame_cached(tuple(dict.fromkeys(symbols)), start, end)
    return prices.copy() if not prices.empty else pd.DataFrame()


def _avg_pairwise_corr(frame: pd.DataFrame) -> float:
    """Average off-diagonal correlation for a returns frame."""
    if frame is None or frame.empty or frame.shape[1] < 2:
        return 0.35
    corr = frame.corr().replace([np.inf, -np.inf], np.nan)
    vals = corr.to_numpy(dtype=float)
    mask = ~np.eye(vals.shape[0], dtype=bool)
    off_diag = vals[mask]
    off_diag = off_diag[np.isfinite(off_diag)]
    if off_diag.size == 0:
        return 0.35
    return float(np.clip(off_diag.mean(), -1.0, 1.0))


# ═══════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def _trend_features(price: pd.Series) -> dict:
    """
    Multi-timeframe trend signals.

    Returns a dict of raw values and a composite trend_z in [-1, +1]:
      +1  →  strongly trending up
      -1  →  strongly trending down
       0  →  no trend / sideways
    """
    ema50  = price.ewm(span=50,  adjust=False).mean()
    ema150 = price.ewm(span=150, adjust=False).mean()
    ema200 = price.ewm(span=200, adjust=False).mean()

    p   = _safe_float(price.iloc[-1])
    e50 = _safe_float(ema50.iloc[-1])
    e150= _safe_float(ema150.iloc[-1])
    e200= _safe_float(ema200.iloc[-1])

    # ── 1. Price position relative to each EMA (normalised) ──
    dist50  = (p - e50)  / e50  if e50  > 0 else 0.0
    dist150 = (p - e150) / e150 if e150 > 0 else 0.0
    dist200 = (p - e200) / e200 if e200 > 0 else 0.0

    # ── 2. EMA alignment score (bull stack = +1, bear stack = -1) ──
    alignment = 0.0
    if e50 > e150 > e200:
        alignment = +1.0
    elif e50 < e150 < e200:
        alignment = -1.0
    else:
        alignment = np.sign(e50 - e200) * 0.4   # partial alignment

    # ── 3. EMA slope (rate of change over 21 days, % basis) ──
    slope150 = 0.0
    if len(ema150) > 21:
        e_prev = _safe_float(ema150.iloc[-22])
        slope150 = (e150 - e_prev) / e_prev if e_prev > 0 else 0.0

    # ── 4. Composite trend score (weighted average of normalised signals) ──
    # Each component is a z-like value; we take their weighted mean then
    # map through sigmoid to get a probability-style [0,1] output.
    raw = (
        0.30 * dist150 +
        0.20 * dist200 +
        0.15 * dist50  +
        0.20 * alignment +
        0.15 * slope150 * 10   # slope is small; scale up
    )

    # Standardise raw to ~unit scale using a calibrated divisor
    # (empirically dist150 ranges ~[-0.25, +0.25] in normal markets)
    trend_z = float(np.clip(raw / 0.20, -2.5, 2.5))

    return {
        "dist50": dist50,
        "dist150": dist150,
        "dist200": dist200,
        "ema_alignment": alignment,
        "slope150": slope150,
        "trend_z": trend_z,
    }


def _volatility_features(returns: pd.Series) -> dict:
    """
    Volatility regime: short/long ratio, vol-of-vol, GARCH(1,1) proxy.
    """
    vol10  = _safe_float(returns.rolling(10,  min_periods=5 ).std().iloc[-1])
    vol20  = _safe_float(returns.rolling(20,  min_periods=10).std().iloc[-1])
    vol63  = _safe_float(returns.rolling(63,  min_periods=30).std().iloc[-1])
    vol252 = _safe_float(returns.rolling(252, min_periods=120).std().iloc[-1])

    baseline = vol252 if vol252 > 1e-6 else vol63

    # Short/long vol ratio — elevated = stressed
    vol_ratio = vol20 / baseline if baseline > 0 else 1.0
    vol_ratio = float(np.clip(vol_ratio, 0.5, 3.5))

    # Vol-of-vol: standard deviation of the rolling 20-day vol series
    # High vol-of-vol = regime transition / unstable
    rolling_vol20 = returns.rolling(20, min_periods=10).std()
    vov = _safe_float(rolling_vol20.rolling(63, min_periods=30).std().iloc[-1])
    vov_baseline  = _safe_float(rolling_vol20.rolling(252, min_periods=120).std().iloc[-1])
    vov_ratio = vov / vov_baseline if vov_baseline > 1e-8 else 1.0
    vov_ratio = float(np.clip(vov_ratio, 0.5, 3.0))

    # GARCH(1,1)-style: ω + α*r²[t-1] + β*σ²[t-1]
    # We use a fast approximation: exponentially-weighted variance
    ewvar = float(returns.ewm(span=30, adjust=False).var().iloc[-1])
    garch_proxy = float(np.sqrt(max(ewvar, 0.0)))

    # Vol regime z-score: how elevated is current vol vs its own history?
    vol_z = _zscore_clip(rolling_vol20.dropna(), window=252, cap=3.0)

    return {
        "vol20":       vol20,
        "vol252":      vol252,
        "vol_ratio":   vol_ratio,
        "vov_ratio":   vov_ratio,
        "garch_proxy": garch_proxy,
        "vol_z":       vol_z,         # +ve = elevated vol (bearish), –ve = calm
    }


def _drawdown_features(price: pd.Series) -> dict:
    """
    Drawdown features: severity, speed, and recovery momentum.
    """
    p = _safe_float(price.iloc[-1])

    # Peak over 1yr and 3yr
    peak_1y = _safe_float(price.rolling(252,  min_periods=120).max().iloc[-1])
    peak_3y = _safe_float(price.rolling(756,  min_periods=252).max().iloc[-1])

    dd_1y = (p - peak_1y) / peak_1y if peak_1y > 0 else 0.0
    dd_3y = (p - peak_3y) / peak_3y if peak_3y > 0 else 0.0

    dd_1y = float(np.clip(dd_1y, -0.60, 0.0))
    dd_3y = float(np.clip(dd_3y, -0.80, 0.0))

    # Drawdown speed: how much did price fall in the last 21 trading days?
    p_21ago = _safe_float(price.iloc[-22]) if len(price) > 22 else p
    dd_speed = (p - p_21ago) / p_21ago if p_21ago > 0 else 0.0
    dd_speed = float(np.clip(dd_speed, -0.15, 0.15))

    # Recovery momentum: price vs 63-day low
    low_63 = _safe_float(price.rolling(63, min_periods=20).min().iloc[-1])
    recovery = (p - low_63) / low_63 if low_63 > 0 else 0.0
    recovery = float(np.clip(recovery, 0.0, 0.30))

    # Composite drawdown z (negative = in drawdown, magnitude matters)
    dd_z = (0.5 * dd_1y + 0.3 * dd_3y + 0.2 * dd_speed) / 0.10
    dd_z = float(np.clip(dd_z, -3.0, 0.5))

    return {
        "dd_1y":     dd_1y,
        "dd_3y":     dd_3y,
        "dd_speed":  dd_speed,
        "recovery":  recovery,
        "dd_z":      dd_z,     # 0=no drawdown, –3=severe drawdown
    }


def _breadth_feature(price: pd.Series) -> dict:
    """
    Breadth proxy using a single index: fraction of the past 252 days
    where the price was above EMA150.  <0.4 = narrow bear market.
    """
    ema150 = price.ewm(span=150, adjust=False).mean()
    window = min(252, len(price))
    above  = (price.iloc[-window:] > ema150.iloc[-window:]).sum()
    breadth = float(above / window)   # 0.0 to 1.0

    # Transform to [-1, +1]: 0.5 = neutral
    breadth_z = float(np.clip((breadth - 0.5) / 0.25, -2.0, 2.0))

    return {
        "breadth":   breadth,
        "breadth_z": breadth_z,
    }


def _basket_condition_features(prices: pd.DataFrame) -> dict:
    """
    Cross-sectional market-health features from the actual basket.

    This improves regime quality versus using only the broad index:
      - participation: how many stocks are still rising?
      - breadth: how many remain above medium/long trend?
      - correlation stress: are names suddenly moving as one block?
      - downside share: how much of the basket is already in drawdown?
    """
    default = {
        "basket_size": 0,
        "breadth": 0.5,
        "breadth_z": 0.0,
        "participation": 0.5,
        "participation_z": 0.0,
        "avg_corr_63": 0.35,
        "avg_corr_252": 0.35,
        "corr_z": 0.0,
        "downside_share": 0.25,
        "downside_z": 0.0,
        "above_50": 0.5,
        "above_200": 0.5,
        "up_21": 0.5,
        "up_63": 0.5,
    }
    if prices is None or prices.empty:
        return default

    px = prices.sort_index().ffill().dropna(how="all")
    eligible = [c for c in px.columns if px[c].count() >= 80]
    if len(eligible) < 2:
        return default
    px = px[eligible]

    last = px.ffill().iloc[-1]
    ema50 = px.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = px.ewm(span=200, adjust=False).mean().iloc[-1]

    ret21 = px.pct_change(21).replace([np.inf, -np.inf], np.nan).iloc[-1]
    ret63 = px.pct_change(63).replace([np.inf, -np.inf], np.nan).iloc[-1]
    rolling_peak = px.rolling(252, min_periods=60).max().iloc[-1]
    drawdowns = (last / rolling_peak) - 1.0

    above_50 = float((last > ema50).mean())
    above_200 = float((last > ema200).mean())
    up_21 = float((ret21 > 0).mean()) if ret21.notna().any() else 0.5
    up_63 = float((ret63 > 0).mean()) if ret63.notna().any() else 0.5
    mild_dd = float((drawdowns < -0.10).mean()) if drawdowns.notna().any() else 0.25
    severe_dd = float((drawdowns < -0.20).mean()) if drawdowns.notna().any() else 0.10

    breadth = float(np.clip(
        0.40 * above_200 + 0.25 * above_50 + 0.20 * up_63 + 0.15 * up_21,
        0.0, 1.0
    ))
    participation = float(np.clip(0.60 * up_63 + 0.40 * up_21, 0.0, 1.0))
    downside_share = float(np.clip(0.60 * severe_dd + 0.40 * mild_dd, 0.0, 1.0))

    returns = px.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    recent = returns.tail(63).dropna(axis=1, thresh=20)
    baseline = returns.tail(252).dropna(axis=1, thresh=60)
    avg_corr_63 = _avg_pairwise_corr(recent)
    avg_corr_252 = _avg_pairwise_corr(baseline)
    corr_delta = avg_corr_63 - avg_corr_252

    return {
        "basket_size": len(eligible),
        "breadth": breadth,
        "breadth_z": float(np.clip((breadth - 0.50) / 0.18, -3.0, 3.0)),
        "participation": participation,
        "participation_z": float(np.clip((participation - 0.50) / 0.18, -3.0, 3.0)),
        "avg_corr_63": avg_corr_63,
        "avg_corr_252": avg_corr_252,
        "corr_z": float(np.clip(corr_delta / 0.12, -2.5, 3.0)),
        "downside_share": downside_share,
        "downside_z": float(np.clip((downside_share - 0.25) / 0.18, -2.5, 3.0)),
        "above_50": above_50,
        "above_200": above_200,
        "up_21": up_21,
        "up_63": up_63,
    }


def _fast_regime_features(price: pd.Series, basket_feats: dict) -> dict:
    """
    Tactical short-horizon features for detecting small regime shifts.

    These deliberately react faster than the slow regime block and are used
    only for tactical overlays, not to replace the strategic long-term engine.
    """
    default = {
        "fast_trend": 0.0,
        "fast_vol": 0.0,
        "fast_dd": 0.0,
        "fast_breadth": 0.0,
        "fast_corr": 0.0,
        "fast_recovery": 0.0,
        "index_21d_return": 0.0,
        "index_21d_drawdown": 0.0,
        "index_63d_drawdown": 0.0,
        "index_return_stress": 0.0,
        "index_drawdown_21_stress": 0.0,
        "index_drawdown_63_stress": 0.0,
    }
    if price is None or len(price) < 30:
        return default

    px = price.astype(float).dropna()
    ret = px.pct_change().dropna()
    if ret.empty:
        return default

    r5 = _safe_float(px.pct_change(5).iloc[-1]) if len(px) > 5 else 0.0
    r10 = _safe_float(px.pct_change(10).iloc[-1]) if len(px) > 10 else r5
    r21 = _safe_float(px.pct_change(21).iloc[-1]) if len(px) > 21 else r10
    ema20 = px.ewm(span=20, adjust=False).mean()
    e20 = _safe_float(ema20.iloc[-1])
    p = _safe_float(px.iloc[-1])
    trend_gap = (p - e20) / e20 if e20 > 0 else 0.0
    fast_trend = float(np.clip((0.45 * r5 + 0.35 * r10 + 0.20 * r21 + 0.30 * trend_gap) / 0.05, -3.0, 3.0))

    vol5 = _safe_float(ret.rolling(5, min_periods=3).std().iloc[-1])
    vol21 = _safe_float(ret.rolling(21, min_periods=10).std().iloc[-1])
    fast_vol = float(np.clip(((vol5 / max(vol21, 1e-8)) - 1.0) / 0.35, -2.5, 3.0))

    peak21 = _safe_float(px.rolling(21, min_periods=10).max().iloc[-1])
    peak63 = _safe_float(px.rolling(63, min_periods=20).max().iloc[-1])
    dd21 = (p - peak21) / peak21 if peak21 > 0 else 0.0
    dd63 = (p - peak63) / peak63 if peak63 > 0 else 0.0
    dd10 = _safe_float(px.pct_change(10).iloc[-1]) if len(px) > 10 else 0.0
    fast_dd = float(np.clip((0.45 * dd21 + 0.35 * dd63 + 0.20 * dd10) / 0.06, -3.0, 2.0))
    index_return_stress = float(np.clip(max(-r21, 0.0) / 0.06, 0.0, 3.0))
    index_drawdown_21_stress = float(np.clip(max(-dd21, 0.0) / 0.08, 0.0, 3.0))
    index_drawdown_63_stress = float(np.clip(max(-dd63, 0.0) / 0.12, 0.0, 3.0))

    short_breadth = 0.5 * float(basket_feats.get("up_21", 0.5)) + 0.5 * float(basket_feats.get("above_50", 0.5))
    fast_breadth = float(np.clip((short_breadth - 0.5) / 0.18, -3.0, 3.0))
    fast_corr = float(np.clip((float(basket_feats.get("avg_corr_63", 0.35)) - 0.35) / 0.15, -2.5, 3.0))
    fast_recovery = float(np.clip((float(basket_feats.get("participation", 0.5)) - float(basket_feats.get("downside_share", 0.25))) / 0.35, -2.5, 2.5))

    return {
        "fast_trend": fast_trend,
        "fast_vol": fast_vol,
        "fast_dd": fast_dd,
        "fast_breadth": fast_breadth,
        "fast_corr": fast_corr,
        "fast_recovery": fast_recovery,
        "index_21d_return": r21,
        "index_21d_drawdown": dd21,
        "index_63d_drawdown": dd63,
        "index_return_stress": index_return_stress,
        "index_drawdown_21_stress": index_drawdown_21_stress,
        "index_drawdown_63_stress": index_drawdown_63_stress,
    }


def _yield_curve_feature(yc_long_sym: str, yc_short_sym: str,
                          start: str, end: str) -> dict:
    """
    Yield curve slope proxy (10Y – 2Y spread).
    Inversion (<0) is historically the strongest recession leading indicator.
    Returns neutral values if data is unavailable.
    """
    default = {"yc_spread": 1.5, "yc_z": 0.0, "inverted": False}
    try:
        long_df  = _fetch(yc_long_sym,  start, end)
        short_df = _fetch(yc_short_sym, start, end)
        if long_df.empty or short_df.empty:
            return default

        long_rate  = long_df["Close"].squeeze()
        short_rate = short_df["Close"].squeeze()

        # Align on common dates
        spread = long_rate - short_rate
        spread = spread.dropna()
        if len(spread) < 20:
            return default

        current_spread = _safe_float(spread.iloc[-1])
        spread_z       = _zscore_clip(spread, window=252, cap=2.5)
        inverted       = current_spread < 0.0

        return {
            "yc_spread": current_spread,
            "yc_z":      spread_z,   # –ve = flattening/inversion = bearish
            "inverted":  inverted,
        }
    except Exception:
        return default


# ═══════════════════════════════════════════════════════════════
#  REGIME PROBABILITY MODEL
# ═══════════════════════════════════════════════════════════════

def _compute_regime_scores(
    trend_z: float,
    vol_z: float,
    dd_z: float,
    breadth_z: float,
    participation_z: float,
    corr_z: float,
    downside_z: float,
    yc_z: float,
    vov_ratio: float,
    recovery: float,
) -> dict[str, float]:
    """
    Converts 6 standardised signals into 4 regime probabilities.

    Logic matrix (sign → regime):
    ─────────────────────────────────────────────────────────────
    Signal          Bull   Sideways   Bear   Crisis
    trend_z > 0      +++      +        –      ––
    vol_z < 0        +++      ++       –      –––
    dd_z ≈ 0         +++      ++       –      –––
    breadth_z > 0    ++       +        –      –
    yc_z > 0         +        +        –      ––
    vov_ratio ≈ 1    +        ++       –      –––
    ─────────────────────────────────────────────────────────────

    Each score is built as a weighted logistic of the relevant signals,
    then the four scores are softmax-normalised to probabilities.
    """

    # ── Bull: rising trend + low vol + no drawdown + good breadth ──
    stress_corr = max(0.0, corr_z)

    bull_logit = (
        +0.32 * trend_z
        -0.22 * vol_z
        +0.18 * breadth_z
        +0.14 * participation_z
        -0.10 * stress_corr
        -0.10 * abs(dd_z)
        +0.06 * yc_z
    )
    bull_score = _sig(bull_logit - 0.18)

    # ── Sideways: weak trend + moderate vol + shallow drawdown ──
    sideways_logit = (
        -0.35 * abs(trend_z)
        -0.14 * abs(participation_z)
        -0.12 * abs(breadth_z)
        -0.14 * max(vol_z, 0.0)
        -0.10 * abs(dd_z)
        -0.08 * stress_corr
        +0.10 * (1.0 - min(abs(breadth_z) / 2.5, 1.0))
    )
    sideways_score = _sig(sideways_logit + 0.30)

    # ── Bear: falling trend + elevated vol + in drawdown + poor breadth ──
    bear_logit = (
        -0.30 * trend_z
        +0.18 * vol_z
        -0.20 * dd_z
        -0.15 * breadth_z
        -0.18 * participation_z
        +0.12 * stress_corr
        +0.10 * downside_z
        -0.08 * yc_z
    )
    bear_score = _sig(bear_logit - 0.10)

    # ── Crisis: extreme vol spike + rapid drawdown + yield-curve inversion ──
    crisis_logit = (
        +0.28 * vol_z
        +0.22 * max(0.0, vov_ratio - 1.30)
        -0.18 * dd_z
        +0.18 * stress_corr
        +0.16 * downside_z
        -0.10 * trend_z
        -0.10 * yc_z
        -0.08 * recovery
        -0.06 * breadth_z
    )
    crisis_score = _sig(crisis_logit - 0.45)

    scores = np.array([bull_score, sideways_score, bear_score, crisis_score])

    # Softmax normalisation
    scores = scores - scores.max()           # numerical stability
    exp_s  = np.exp(scores * 2.7)
    probs  = exp_s / exp_s.sum()

    regimes = ["bull", "sideways", "bear", "crisis"]
    return dict(zip(regimes, probs.tolist()))


def _compute_fast_regime_scores(
    fast_trend: float,
    fast_vol: float,
    fast_dd: float,
    fast_breadth: float,
    fast_corr: float,
    fast_recovery: float,
    index_21d_return: float,
    index_21d_drawdown: float,
    index_63d_drawdown: float,
    index_return_stress: float,
    index_drawdown_21_stress: float,
    index_drawdown_63_stress: float,
) -> dict[str, float]:
    """
    Tactical short-horizon regime detector.

    This is intentionally more reactive than the slow regime block so it can
    catch small weekly/daily state shifts without disturbing strategic posture.
    """
    bull_logit = (
        +0.34 * fast_trend
        -0.18 * fast_vol
        +0.16 * fast_breadth
        -0.10 * max(fast_corr, 0.0)
        +0.12 * fast_recovery
        -0.22 * index_return_stress
        -0.16 * index_drawdown_21_stress
        -0.10 * index_drawdown_63_stress
    )
    sideways_logit = (
        -0.28 * abs(fast_trend)
        -0.16 * max(fast_vol, 0.0)
        -0.10 * abs(fast_breadth)
        +0.12 * (1.0 - min(abs(fast_corr) / 2.5, 1.0))
        -0.10 * max(index_return_stress - 0.8, 0.0)
    )
    bear_logit = (
        -0.30 * fast_trend
        +0.18 * fast_vol
        -0.16 * fast_breadth
        +0.14 * max(fast_corr, 0.0)
        -0.16 * fast_dd
        -0.08 * fast_recovery
        +0.24 * index_return_stress
        +0.18 * index_drawdown_21_stress
        +0.12 * index_drawdown_63_stress
    )
    crisis_logit = (
        +0.24 * fast_vol
        +0.22 * max(fast_corr, 0.0)
        -0.20 * fast_dd
        -0.12 * fast_trend
        -0.10 * fast_breadth
        -0.08 * fast_recovery
        +0.16 * max(index_return_stress - 1.0, 0.0)
        +0.18 * index_drawdown_21_stress
        +0.14 * index_drawdown_63_stress
    )

    bull_score = _sig(bull_logit - 0.08)
    sideways_score = _sig(sideways_logit + 0.25)
    bear_score = _sig(bear_logit - 0.05)
    crisis_score = _sig(crisis_logit - 0.30)

    # Hard cap: a sharp 21-day index selloff cannot be called bull.
    if index_21d_return <= -0.06:
        bearish_shock = min(max((-index_21d_return - 0.06) / 0.08, 0.0), 1.5)
        bull_score = min(bull_score, 1e-4)
        sideways_score *= max(0.55 - 0.15 * bearish_shock, 0.20)
        bear_score += 0.22 + 0.18 * bearish_shock
        crisis_score += 0.12 + 0.12 * bearish_shock

    scores = np.array([bull_score, sideways_score, bear_score, crisis_score])
    scores = scores - scores.max()
    probs = np.exp(scores * 3.1)
    probs = probs / probs.sum()
    regimes = ["bull", "sideways", "bear", "crisis"]
    return dict(zip(regimes, probs.tolist()))


def _display_regime_label(probs: dict[str, float], confidence: float) -> str:
    """Prefer 'mixed' over a weak bullish argmax so the UI does not overstate conviction."""
    if not probs:
        return "unknown"
    dominant = max(probs, key=probs.get)
    if dominant == "bull" and (confidence < _MIXED_CONFIDENCE_THRESHOLD or float(probs.get("bull", 0.0)) < _BULL_PROB_MIXED_CAP):
        return "mixed"
    return dominant


def _transition_forecast(
    slow_probs: dict[str, float],
    fast_probs: dict[str, float],
    fast_feats: dict,
    basket_feats: dict,
    confidence: float,
    fast_confidence: float,
    dominant: str,
) -> dict:
    """Short-horizon transition forecast used by the UI and deploy-now logic."""
    slow_probs = slow_probs or {}
    fast_probs = fast_probs or {}
    fast_feats = fast_feats or {}
    basket_feats = basket_feats or {}

    downside_share = float(basket_feats.get("downside_share", 0.25))
    participation = float(basket_feats.get("participation", 0.50))
    corr_stress = max(float(basket_feats.get("corr_z", 0.0)), 0.0) / 3.0
    index_stress = float(np.clip(
        0.45 * float(fast_feats.get("index_return_stress", 0.0))
        + 0.35 * float(fast_feats.get("index_drawdown_21_stress", 0.0))
        + 0.20 * float(fast_feats.get("index_drawdown_63_stress", 0.0)),
        0.0,
        2.5,
    )) / 2.5
    bullish_recovery = float(np.clip(
        0.45 * max(float(fast_feats.get("fast_trend", 0.0)), 0.0) / 3.0
        + 0.35 * max(float(fast_feats.get("fast_recovery", 0.0)), 0.0) / 2.5
        + 0.20 * max(float(fast_feats.get("fast_breadth", 0.0)), 0.0) / 3.0,
        0.0,
        1.0,
    ))

    next_scores = {
        "bull": (
            0.55 * float(slow_probs.get("bull", 0.25))
            + 0.85 * float(fast_probs.get("bull", 0.25))
            + 0.18 * bullish_recovery
            - 0.22 * index_stress
            - 0.10 * corr_stress
            - 0.08 * downside_share
        ),
        "sideways": (
            0.60 * float(slow_probs.get("sideways", 0.25))
            + 0.70 * float(fast_probs.get("sideways", 0.25))
            + 0.06 * (1.0 - index_stress)
        ),
        "bear": (
            0.55 * float(slow_probs.get("bear", 0.25))
            + 0.90 * float(fast_probs.get("bear", 0.25))
            + 0.20 * index_stress
            + 0.12 * corr_stress
            + 0.10 * downside_share
            + 0.08 * max(0.55 - participation, 0.0) / 0.55
        ),
        "crisis": (
            0.50 * float(slow_probs.get("crisis", 0.25))
            + 0.95 * float(fast_probs.get("crisis", 0.25))
            + 0.22 * index_stress
            + 0.16 * corr_stress
            + 0.10 * downside_share
            + 0.10 * max(float(fast_feats.get("fast_vol", 0.0)), 0.0) / 3.0
        ),
    }
    scores = np.array([max(next_scores[r], 1e-6) for r in ["bull", "sideways", "bear", "crisis"]], dtype=float)
    scores = scores - scores.max()
    probs = np.exp(scores * 2.9)
    probs = probs / probs.sum()
    next_probs = dict(zip(["bull", "sideways", "bear", "crisis"], probs.tolist()))

    current_regime = dominant if dominant in next_probs else max(slow_probs or next_probs, key=(slow_probs or next_probs).get)
    stay_prob = float(next_probs.get(current_regime, max(next_probs.values())))
    bearish_shift = float(np.clip(next_probs["bear"] + next_probs["crisis"], 0.0, 1.0))
    crisis_risk = float(np.clip(next_probs["crisis"], 0.0, 1.0))
    transition_risk = float(np.clip(
        (1.0 - stay_prob) * 0.45
        + bearish_shift * 0.35
        + crisis_risk * 0.20
        + max(0.20 - fast_confidence, 0.0),
        0.0,
        1.0,
    ))
    expected_next = _display_regime_label(next_probs, max(confidence, fast_confidence))

    if crisis_risk >= 0.24:
        summary = "Crisis risk is rising"
    elif bearish_shift >= 0.58:
        summary = "Risk of slipping toward bear"
    elif stay_prob >= 0.62:
        summary = "Current regime is likely to persist"
    elif bullish_recovery >= 0.45 and bearish_shift < 0.45:
        summary = "Recovery pressure is building"
    else:
        summary = "Short-term tape looks unstable"

    return {
        "next_10d_probabilities": {k: round(v, 4) for k, v in next_probs.items()},
        "expected_next_regime": expected_next,
        "stay_prob_10d": round(stay_prob, 4),
        "bearish_shift_prob_10d": round(bearish_shift, 4),
        "crisis_risk_10d": round(crisis_risk, 4),
        "transition_risk_10d": round(transition_risk, 4),
        "summary": summary,
    }


def _smooth_regime_probs(
    current_probs: dict,
    history: dict | None,
    halflife: int = _PERSISTENCE_HALFLIFE,
) -> dict:
    """
    Exponential smoothing of regime probabilities to prevent daily flip-flopping.
    `history` should be the smoothed probs from the previous call (or None on first call).
    """
    if history is None:
        return current_probs

    alpha = 1.0 - np.exp(-np.log(2) / halflife)   # EMA decay factor

    smoothed = {}
    for regime in current_probs:
        prev = history.get(regime, current_probs[regime])
        smoothed[regime] = alpha * current_probs[regime] + (1 - alpha) * prev

    # Re-normalise (smoothing can slightly violate sum-to-1)
    total = sum(smoothed.values())
    return {k: v / total for k, v in smoothed.items()}


# ═══════════════════════════════════════════════════════════════
#  PARAMETER CONVERSION
# ═══════════════════════════════════════════════════════════════

def regime_to_parameters(regime_probs: dict) -> tuple[float, float]:
    """
    Convert regime probabilities → (alpha, beta) for downstream engines.

    alpha: stock-picking confidence (high = trust individual stock signals)
    beta:  shrinkage weight vs momentum (high = trust mean-reversion / shrinkage)

    The parameter mappings are regime-weighted averages of per-regime anchors:

    Regime     alpha   beta   Rationale
    ──────     ─────   ────   ─────────────────────────────────────────────────
    Bull       0.95    0.25   High stock dispersion; momentum works well.
    Sideways   0.65    0.55   Mixed; moderate both signals.
    Bear       0.35    0.75   Low dispersion; shrinkage safer; avoid stock risk.
    Crisis     0.12    0.92   Near-zero stock alpha; stay close to shrinkage.
    """
    bull    = regime_probs["bull"]
    sideways= regime_probs["sideways"]
    bear    = regime_probs["bear"]
    crisis  = regime_probs["crisis"]

    alpha = (0.95 * bull + 0.65 * sideways + 0.35 * bear + 0.12 * crisis)
    beta  = (0.25 * bull + 0.55 * sideways + 0.75 * bear + 0.92 * crisis)

    alpha = float(np.clip(alpha, 0.12, 0.95))   # raised ceiling to allow full bull alpha
    beta  = float(np.clip(beta,  0.20, 0.92))

    return alpha, beta


# ═══════════════════════════════════════════════════════════════
#  MASTER FUNCTION
# ═══════════════════════════════════════════════════════════════

def detect_market_state(
    tickers:        list,
    start_date:     str,
    end_date:       str,
    prev_regime_probs: dict | None = None,   # pass smoothed probs from last call
) -> tuple[float, float, dict]:
    """
    Detect probabilistic market regime and return adaptive parameters.

    Parameters
    ----------
    tickers           : list of ticker symbols (used to infer market)
    start_date        : history window start (YYYY-MM-DD)
    end_date          : history window end   (YYYY-MM-DD)
    prev_regime_probs : smoothed regime probs from the previous session
                        (enables regime persistence smoothing).  Pass None
                        on first call or when you want a fresh read.

    Returns
    -------
    alpha  : float  — stock-picking confidence [0.12, 0.92]
    beta   : float  — shrinkage weight [0.25, 0.92]
    meta   : dict   — all intermediate signals and regime probabilities
    """

    # ── Select benchmark market ──────────────────────────────────
    indian = sum(t.endswith(".NS") or t.endswith(".BO") for t in tickers)
    market = "india" if indian >= len(tickers) / 2 else "us"
    cfg    = _MARKET_CONFIG[market]
    index_symbol = cfg["index"]

    # ── Fetch index price history ────────────────────────────────
    # Need at least 3 years for 3yr drawdown and 252-day rolling stats.
    import datetime as _dt
    _end_dt   = pd.Timestamp(end_date)
    _start_dt = _end_dt - pd.Timedelta(days=int(252 * 3.2))
    fetch_start = _start_dt.strftime("%Y-%m-%d")

    raw = _fetch(index_symbol, fetch_start, end_date)

    # Fallback to alt index if primary unavailable
    if raw.empty:
        raw = _fetch(cfg["index_alt"], fetch_start, end_date)
        index_symbol = cfg["index_alt"]

    if raw.empty:
        # Cannot compute anything — return neutral defaults
        neutral_probs = {"bull": 0.25, "sideways": 0.25, "bear": 0.25, "crisis": 0.25}
        return 0.55, 0.55, {
            "benchmark": index_symbol,
            "market": market,
            "regime_probabilities": neutral_probs,
            "dominant_regime": "unknown",
            "alpha": 0.55,
            "beta": 0.55,
            "confidence": 0.0,
            "error": "index data unavailable",
        }

    # Unwrap MultiIndex if needed
    if isinstance(raw.columns, pd.MultiIndex):
        price = raw["Close"].squeeze()
    else:
        price = raw["Close"].squeeze()

    if isinstance(price, pd.DataFrame):
        price = price.iloc[:, 0]

    price = price.astype(float).dropna()

    # Clip to the requested window for signal computation
    price = price[price.index <= end_date]

    returns = price.pct_change().dropna()
    basket_prices = _fetch_price_frame(tickers, fetch_start, end_date)

    # ── Compute all feature groups ───────────────────────────────
    trend_feats   = _trend_features(price)
    vol_feats     = _volatility_features(returns)
    dd_feats      = _drawdown_features(price)
    index_breadth_feats = _breadth_feature(price)
    basket_feats  = _basket_condition_features(basket_prices)
    if basket_feats["basket_size"] >= 2:
        breadth = float(np.clip(
            0.65 * basket_feats["breadth"] + 0.35 * index_breadth_feats["breadth"],
            0.0, 1.0
        ))
        breadth_z = float(np.clip(
            0.65 * basket_feats["breadth_z"] + 0.35 * index_breadth_feats["breadth_z"],
            -3.0, 3.0
        ))
    else:
        breadth = index_breadth_feats["breadth"]
        breadth_z = index_breadth_feats["breadth_z"]
    yc_feats      = _yield_curve_feature(
                        cfg["yc_long"], cfg["yc_short"], fetch_start, end_date
                    )

    # ── Compute raw regime probabilities ────────────────────────
    raw_probs = _compute_regime_scores(
        trend_z   = trend_feats["trend_z"],
        vol_z     = vol_feats["vol_z"],
        dd_z      = dd_feats["dd_z"],
        breadth_z = breadth_z,
        participation_z = basket_feats["participation_z"],
        corr_z = basket_feats["corr_z"],
        downside_z = basket_feats["downside_z"],
        yc_z      = yc_feats["yc_z"],
        vov_ratio = vol_feats["vov_ratio"],
        recovery  = dd_feats["recovery"],
    )

    fast_feats = _fast_regime_features(price, basket_feats)
    fast_raw_probs = _compute_fast_regime_scores(
        fast_trend=fast_feats["fast_trend"],
        fast_vol=fast_feats["fast_vol"],
        fast_dd=fast_feats["fast_dd"],
        fast_breadth=fast_feats["fast_breadth"],
        fast_corr=fast_feats["fast_corr"],
        fast_recovery=fast_feats["fast_recovery"],
        index_21d_return=fast_feats["index_21d_return"],
        index_21d_drawdown=fast_feats["index_21d_drawdown"],
        index_63d_drawdown=fast_feats["index_63d_drawdown"],
        index_return_stress=fast_feats["index_return_stress"],
        index_drawdown_21_stress=fast_feats["index_drawdown_21_stress"],
        index_drawdown_63_stress=fast_feats["index_drawdown_63_stress"],
    )

    # ── Apply regime persistence smoothing ──────────────────────
    slow_history = prev_regime_probs
    fast_history = None
    if isinstance(prev_regime_probs, dict) and "slow" in prev_regime_probs:
        slow_history = prev_regime_probs.get("slow")
        fast_history = prev_regime_probs.get("fast")
    smoothed_probs = _smooth_regime_probs(raw_probs, slow_history)
    fast_probs = _smooth_regime_probs(
        fast_raw_probs,
        fast_history,
        halflife=_FAST_PERSISTENCE_HALFLIFE,
    )

    # ── Derive parameters ────────────────────────────────────────
    alpha, beta = regime_to_parameters(smoothed_probs)
    _, fast_beta = regime_to_parameters(fast_probs)

    # Dominant regime label
    sorted_probs = sorted(smoothed_probs.values(), reverse=True)
    confidence = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0])
    dominant = _display_regime_label(smoothed_probs, confidence)
    fast_sorted_probs = sorted(fast_probs.values(), reverse=True)
    fast_confidence = float(fast_sorted_probs[0] - fast_sorted_probs[1]) if len(fast_sorted_probs) > 1 else float(fast_sorted_probs[0])
    fast_dominant = _display_regime_label(fast_probs, fast_confidence)
    transition_forecast = _transition_forecast(
        smoothed_probs,
        fast_probs,
        fast_feats,
        basket_feats,
        confidence,
        fast_confidence,
        dominant,
    )
    tactical_turnover_penalty = float(np.clip(
        0.01
        + 0.05 * fast_probs["sideways"]
        + 0.06 * fast_probs["crisis"]
        + 0.03 * max(0.0, basket_feats["avg_corr_63"] - 0.45) / 0.40,
        0.0,
        0.14,
    ))
    tactical_rebalance_ratio = float(np.clip(
        0.30
        + 0.40 * fast_confidence
        + 0.20 * (fast_probs["bull"] + fast_probs["bear"] + fast_probs["crisis"])
        - 0.10 * fast_probs["sideways"],
        0.20,
        0.90,
    ))
    tactical_momentum_weight = float(np.clip(1.0 - fast_beta, 0.08, 0.80))

    # ── Build meta dict ──────────────────────────────────────────
    meta = {
        # Identification
        "benchmark":         index_symbol,
        "market":            market,
        # Composite z-scores (each ~ N(0,1) in normal markets)
        "trend_z":           round(trend_feats["trend_z"],   4),
        "vol_z":             round(vol_feats["vol_z"],       4),
        "dd_z":              round(dd_feats["dd_z"],         4),
        "breadth_z":         round(breadth_z, 4),
        "yc_z":              round(yc_feats["yc_z"],         4),
        "participation_z":   round(basket_feats["participation_z"], 4),
        "corr_z":            round(basket_feats["corr_z"],   4),
        "downside_z":        round(basket_feats["downside_z"], 4),
        # Detailed sub-features
        "trend_detail": {
            "dist50":        round(trend_feats["dist50"],    4),
            "dist150":       round(trend_feats["dist150"],   4),
            "dist200":       round(trend_feats["dist200"],   4),
            "ema_alignment": round(trend_feats["ema_alignment"], 3),
            "slope150":      round(trend_feats["slope150"],  6),
        },
        "volatility": {
            "vol20_ann":     round(vol_feats["vol20"] * np.sqrt(252) * 100, 2),
            "vol252_ann":    round(vol_feats["vol252"] * np.sqrt(252) * 100, 2),
            "vol_ratio":     round(vol_feats["vol_ratio"],  3),
            "vov_ratio":     round(vol_feats["vov_ratio"],  3),
            "garch_proxy":   round(vol_feats["garch_proxy"] * np.sqrt(252) * 100, 2),
        },
        "drawdown_detail": {
            "dd_1y_pct":     round(dd_feats["dd_1y"] * 100, 2),
            "dd_3y_pct":     round(dd_feats["dd_3y"] * 100, 2),
            "dd_speed_pct":  round(dd_feats["dd_speed"] * 100, 2),
            "recovery_pct":  round(dd_feats["recovery"] * 100, 2),
        },
        "breadth":           round(breadth * 100, 1),
        "basket": {
            "n_assets":      basket_feats["basket_size"],
            "above_50_pct":  round(basket_feats["above_50"] * 100, 1),
            "above_200_pct": round(basket_feats["above_200"] * 100, 1),
            "up_21_pct":     round(basket_feats["up_21"] * 100, 1),
            "up_63_pct":     round(basket_feats["up_63"] * 100, 1),
            "participation_pct": round(basket_feats["participation"] * 100, 1),
            "avg_corr_63":   round(basket_feats["avg_corr_63"], 3),
            "avg_corr_252":  round(basket_feats["avg_corr_252"], 3),
            "downside_pct":  round(basket_feats["downside_share"] * 100, 1),
        },
        "yc_spread":         round(yc_feats["yc_spread"], 3),
        "yc_inverted":       yc_feats["inverted"],
        # Regime output
        "regime_probabilities_raw":      {k: round(v, 4) for k, v in raw_probs.items()},
        "regime_probabilities":          {k: round(v, 4) for k, v in smoothed_probs.items()},
        "dominant_regime":   dominant,
        "confidence":        round(confidence, 4),
        "fast_regime_features": {k: round(v, 4) for k, v in fast_feats.items()},
        "fast_regime_probabilities_raw": {k: round(v, 4) for k, v in fast_raw_probs.items()},
        "fast_regime_probabilities": {k: round(v, 4) for k, v in fast_probs.items()},
        "fast_dominant_regime": fast_dominant,
        "fast_confidence": round(fast_confidence, 4),
        "transition_forecast": transition_forecast,
        # Parameters
        "alpha":             round(alpha, 4),
        "beta":              round(beta,  4),
        "fast_beta":         round(fast_beta, 4),
        "tactical_momentum_weight": round(tactical_momentum_weight, 4),
        "tactical_turnover_penalty": round(tactical_turnover_penalty, 4),
        "tactical_rebalance_ratio": round(tactical_rebalance_ratio, 4),
        "regime_state": {
            "slow": {k: round(v, 4) for k, v in smoothed_probs.items()},
            "fast": {k: round(v, 4) for k, v in fast_probs.items()},
        },
        # ── Backward-compatibility flat keys (v1.0 GUI contract) ──
        # These mirror the old meta dict so existing GUI code that does
        # regime_meta["vol_ratio"], regime_meta["trend"], etc. keeps working
        # without any changes.
        "trend":             round(trend_feats["trend_z"],        4),
        "vol_ratio":         round(vol_feats["vol_ratio"],        4),
        "drawdown":          round(dd_feats["dd_1y"],             4),
        "regime_probs":      {k: round(v, 4) for k, v in smoothed_probs.items()},
    }

    return alpha, beta, meta
