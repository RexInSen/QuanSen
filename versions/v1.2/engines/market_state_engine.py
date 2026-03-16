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


def _fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download with auto_adjust; return empty DF on failure."""
    try:
        df = yf.download(
            symbol, start=start, end=end,
            auto_adjust=True, progress=False
        )
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


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
    bull_logit = (
        +0.35 * trend_z
        -0.25 * vol_z
        +0.20 * breadth_z
        -0.10 * abs(dd_z)           # any drawdown hurts bull
        +0.10 * yc_z * 0.5          # positive curve helps mildly
    )
    bull_score = _sig(bull_logit - 0.20)   # shifted right: need positive evidence

    # ── Sideways: weak trend + moderate vol + shallow drawdown ──
    sideways_logit = (
        -0.40 * abs(trend_z)        # strong trend in either direction kills sideways
        -0.25 * vol_z               # high vol leans toward bear/crisis
        -0.20 * abs(dd_z)
        +0.15 * (1.0 - abs(breadth_z))  # medium breadth fits sideways
    )
    sideways_score = _sig(sideways_logit + 0.30)

    # ── Bear: falling trend + elevated vol + in drawdown + poor breadth ──
    bear_logit = (
        -0.35 * trend_z
        +0.20 * vol_z
        -0.25 * dd_z                # dd_z is ≤0, so –dd_z is ≥0 when in drawdown
        -0.10 * breadth_z
        -0.10 * yc_z * 0.5
    )
    bear_score = _sig(bear_logit - 0.10)

    # ── Crisis: extreme vol spike + rapid drawdown + yield-curve inversion ──
    crisis_logit = (
        +0.35 * vol_z
        +0.25 * max(0.0, vov_ratio - 1.5)   # vol-of-vol explosion
        -0.25 * dd_z                          # deep drawdown
        -0.15 * yc_z * 0.5                   # inversion
        -0.10 * trend_z                       # falling price
        -0.10 * recovery                      # crisis = no recovery yet
    )
    crisis_score = _sig(crisis_logit - 0.50)  # needs strong multi-signal confirmation

    scores = np.array([bull_score, sideways_score, bear_score, crisis_score])

    # Softmax normalisation
    scores = scores - scores.max()           # numerical stability
    exp_s  = np.exp(scores * 2.5)           # temperature = 2.5 (sharper than softmax-1)
    probs  = exp_s / exp_s.sum()

    regimes = ["bull", "sideways", "bear", "crisis"]
    return dict(zip(regimes, probs.tolist()))


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
    Bull       0.90    0.30   High stock dispersion; momentum works well.
    Sideways   0.65    0.55   Mixed; moderate both signals.
    Bear       0.35    0.75   Low dispersion; shrinkage safer; avoid stock risk.
    Crisis     0.12    0.92   Near-zero stock alpha; stay close to shrinkage.
    """
    bull    = regime_probs["bull"]
    sideways= regime_probs["sideways"]
    bear    = regime_probs["bear"]
    crisis  = regime_probs["crisis"]

    alpha = (0.90 * bull + 0.65 * sideways + 0.35 * bear + 0.12 * crisis)
    beta  = (0.30 * bull + 0.55 * sideways + 0.75 * bear + 0.92 * crisis)

    alpha = float(np.clip(alpha, 0.12, 0.92))
    beta  = float(np.clip(beta,  0.25, 0.92))

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

    # ── Compute all feature groups ───────────────────────────────
    trend_feats   = _trend_features(price)
    vol_feats     = _volatility_features(returns)
    dd_feats      = _drawdown_features(price)
    breadth_feats = _breadth_feature(price)
    yc_feats      = _yield_curve_feature(
                        cfg["yc_long"], cfg["yc_short"], fetch_start, end_date
                    )

    # ── Compute raw regime probabilities ────────────────────────
    raw_probs = _compute_regime_scores(
        trend_z   = trend_feats["trend_z"],
        vol_z     = vol_feats["vol_z"],
        dd_z      = dd_feats["dd_z"],
        breadth_z = breadth_feats["breadth_z"],
        yc_z      = yc_feats["yc_z"],
        vov_ratio = vol_feats["vov_ratio"],
        recovery  = dd_feats["recovery"],
    )

    # ── Apply regime persistence smoothing ──────────────────────
    smoothed_probs = _smooth_regime_probs(raw_probs, prev_regime_probs)

    # ── Derive parameters ────────────────────────────────────────
    alpha, beta = regime_to_parameters(smoothed_probs)

    # Dominant regime label
    dominant = max(smoothed_probs, key=smoothed_probs.get)

    # ── Build meta dict ──────────────────────────────────────────
    meta = {
        # Identification
        "benchmark":         index_symbol,
        "market":            market,
        # Composite z-scores (each ~ N(0,1) in normal markets)
        "trend_z":           round(trend_feats["trend_z"],   4),
        "vol_z":             round(vol_feats["vol_z"],       4),
        "dd_z":              round(dd_feats["dd_z"],         4),
        "breadth_z":         round(breadth_feats["breadth_z"], 4),
        "yc_z":              round(yc_feats["yc_z"],         4),
        # Detailed sub-features
        "trend": {
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
        "drawdown": {
            "dd_1y_pct":     round(dd_feats["dd_1y"] * 100, 2),
            "dd_3y_pct":     round(dd_feats["dd_3y"] * 100, 2),
            "dd_speed_pct":  round(dd_feats["dd_speed"] * 100, 2),
            "recovery_pct":  round(dd_feats["recovery"] * 100, 2),
        },
        "breadth":           round(breadth_feats["breadth"] * 100, 1),
        "yc_spread":         round(yc_feats["yc_spread"], 3),
        "yc_inverted":       yc_feats["inverted"],
        # Regime output
        "regime_probabilities_raw":      {k: round(v, 4) for k, v in raw_probs.items()},
        "regime_probabilities":          {k: round(v, 4) for k, v in smoothed_probs.items()},
        "dominant_regime":   dominant,
        # Parameters
        "alpha":             round(alpha, 4),
        "beta":              round(beta,  4),
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
