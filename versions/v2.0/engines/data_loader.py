"""
============================================================
  QuanSen — Data Loader & Shrinkage-Adjusted Expected Returns
============================================================

  WHY SHRINKAGE?
  Raw sample means from historical daily returns are noisy.
  A stock that happened to run +300% in your window will show
  a massive daily mean, and the optimizer will pile into it —
  even though that run is unlikely to repeat.

  Shrinking toward a broad market index pulls outliers back to
  something economically defensible while preserving the
  relative ranking of stocks (good stocks still look better
  than bad ones, just less extreme).

  Formula:  adjusted_i = α * raw_i  +  (1-α) * benchmark_mean

  This is a James-Stein style shrinkage estimator.
    α = 1.0  → pure historical mean  (no shrinkage)
    α = 0.0  → everything collapses to the benchmark mean
    α = 0.7  → default: 70% own history, 30% market anchor

  Public API
  ----------
  load_data(tickers, start_date, end_date, alpha) → 6-tuple
============================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
from functools import lru_cache

from engines.config import SHRINKAGE_ALPHA, INDIAN_BENCHMARKS, GLOBAL_BENCHMARKS


# ════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ════════════════════════════════════════════════════════════

def _detect_benchmark_candidates(tickers):
    """Return the ordered list of benchmark symbols to try."""
    indian_count = sum(1 for t in tickers if t.endswith(".NS") or t.endswith(".BO"))
    if indian_count / max(len(tickers), 1) >= 0.5:
        return INDIAN_BENCHMARKS
    return GLOBAL_BENCHMARKS


def _robust_daily_mean(ret: pd.Series) -> float:
    """Blend clipped sample and EWMA means so one lucky run does not dominate ERs."""
    if ret is None:
        return 0.0
    ret = pd.Series(ret, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if ret.empty:
        return 0.0
    if len(ret) >= 10:
        lower = ret.quantile(0.02)
        upper = ret.quantile(0.98)
        ret = ret.clip(lower=lower, upper=upper)
    sample_mean = float(ret.mean())
    if len(ret) < 20:
        return sample_mean
    span = int(np.clip(len(ret) // 3, 21, 63))
    ewma_mean = float(ret.ewm(span=span, adjust=False).mean().iloc[-1])
    return 0.65 * sample_mean + 0.35 * ewma_mean


def _stabilize_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Shrink sample covariance toward a correlation-stabilized EWMA view."""
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if clean.empty:
        return pd.DataFrame()

    sample_cov = clean.cov().fillna(0.0)
    sample_cov = (sample_cov + sample_cov.T) / 2.0

    span = int(np.clip(len(clean) // 3, 21, 63))
    ewma_vol = clean.ewm(span=span, adjust=False).std().iloc[-1].fillna(clean.std().fillna(0.0))
    ewma_vol = ewma_vol.replace(0.0, np.nan).fillna(clean.std().replace(0.0, np.nan)).fillna(1e-4)

    corr = clean.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr.values, 1.0)
    corr_shrunk = 0.80 * corr + 0.20 * np.eye(len(corr))
    vol_outer = np.outer(ewma_vol.values, ewma_vol.values)
    corr_cov = pd.DataFrame(vol_outer * corr_shrunk, index=clean.columns, columns=clean.columns)

    diag_cov = pd.DataFrame(np.diag(np.diag(sample_cov.values)), index=clean.columns, columns=clean.columns)
    cov_matrix = 0.55 * sample_cov + 0.30 * corr_cov + 0.15 * diag_cov
    cov_matrix = (cov_matrix + cov_matrix.T) / 2.0
    ridge = max(float(np.nanmean(np.diag(cov_matrix.values))), 1e-8) * 1e-4
    cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * ridge
    return cov_matrix


@lru_cache(maxsize=64)
def _fetch_benchmark_daily_mean_cached(candidates_tuple, start_date, end_date):
    """
    Try each benchmark in order.
    Returns (symbol_used, daily_mean_return) or (None, None) on total failure.
    """
    for sym in candidates_tuple:
        try:
            data = yf.download(sym, start=start_date, end=end_date,
                               auto_adjust=True, progress=False)
            if data.empty:
                continue
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            ret = close.pct_change().dropna()
            if len(ret) < 20:
                continue
            mean_daily = _robust_daily_mean(ret)
            print(f"  Benchmark : {sym}  |  {len(ret)} days  |  "
                  f"daily mean = {mean_daily*100:.4f}%  "
                  f"({mean_daily*252*100:.2f}% ann.)")
            return sym, mean_daily
        except Exception:
            continue
    return None, None


def _fetch_benchmark_daily_mean(candidates, start_date, end_date):
    return _fetch_benchmark_daily_mean_cached(tuple(candidates), start_date, end_date)


def _extract_close_frame(data, requested_tickers):
    """Normalise yfinance output to a plain DataFrame of Close prices."""
    if data is None or getattr(data, "empty", True):
        return pd.DataFrame(columns=requested_tickers)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]] if "Close" in data.columns else data
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=requested_tickers[0])
    return prices


@lru_cache(maxsize=64)
def _download_prices_with_repair_cached(tickers_tuple, start_date, end_date):
    """
    Batch download, then retry any missing tickers individually.
    Handles Yahoo/yfinance cases where a valid symbol is silently
    dropped from a multi-ticker batch response.
    """
    tickers = list(tickers_tuple)
    data   = yf.download(tickers, start=start_date, end=end_date,
                         auto_adjust=True, progress=False, threads=True)
    prices = _extract_close_frame(data, tickers)

    repaired = []
    for ticker in tickers:
        if ticker in prices.columns and prices[ticker].dropna().size > 0:
            continue
        # First retry: plain download
        try:
            single        = yf.download(ticker, start=start_date, end=end_date,
                                        auto_adjust=True, progress=False, threads=False)
            single_prices = _extract_close_frame(single, [ticker])
            if ticker in single_prices.columns and single_prices[ticker].dropna().size > 0:
                prices[ticker] = single_prices[ticker]
                repaired.append(ticker)
                continue
        except Exception:
            pass
        # Second retry: Ticker.history()
        try:
            hist = yf.Ticker(ticker).history(start=start_date, end=end_date,
                                             auto_adjust=True)
            if not hist.empty and "Close" in hist.columns:
                prices[ticker] = hist["Close"]
                repaired.append(ticker)
        except Exception:
            continue

    if repaired:
        print(f"Recovered individually after batch download failure: {repaired}")
    return prices


def _download_prices_with_repair(tickers, start_date, end_date):
    prices = _download_prices_with_repair_cached(tuple(tickers), start_date, end_date)
    return prices.copy()


def _trim_to_common_window(prices, tickers):
    """
    Trim the price DataFrame to the overlapping date range
    across all tickers that have data.
    """
    first_valid = {}
    last_valid  = {}
    for ticker in tickers:
        series = prices[ticker].dropna() if ticker in prices.columns else pd.Series(dtype=float)
        if series.empty:
            continue
        first_valid[ticker] = series.index[0]
        last_valid[ticker]  = series.index[-1]

    valid_tickers = [t for t in tickers if t in first_valid and t in last_valid]
    if len(valid_tickers) < 2:
        raise ValueError("Not enough assets with valid price history in the selected date range.")

    common_start = max(first_valid[t] for t in valid_tickers)
    common_end   = min(last_valid[t]  for t in valid_tickers)
    if common_start >= common_end:
        raise ValueError(
            "No overlapping trading window exists across all selected tickers. "
            "Reduce the basket or use a later start date."
        )

    trimmed = prices.loc[common_start:common_end, valid_tickers].copy()
    return trimmed, valid_tickers, common_start, common_end


# ════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════

def load_data(tickers, start_date, end_date, alpha=SHRINKAGE_ALPHA):
    """
    Download price data, compute returns + covariance, and apply
    James-Stein shrinkage to expected returns.

    Parameters
    ----------
    tickers    : list[str]
    start_date : str  (YYYY-MM-DD)
    end_date   : str  (YYYY-MM-DD)
    alpha      : float in [0, 1]
                 Weight on each stock's own historical mean.
                 (1-alpha) goes to the benchmark mean.
                 Default = SHRINKAGE_ALPHA from config.py

    Returns
    -------
    tickers          : list[str]    validated & ordered
    returns          : DataFrame    daily pct changes
    expected_returns : Series       shrinkage-adjusted daily means
    cov_matrix       : DataFrame    symmetrised sample covariance
    raw_er           : Series       unadjusted daily means (for display)
    benchmark_symbol : str | None   benchmark actually used
    """
    print("\nDownloading price data...")
    prices = _download_prices_with_repair(tickers, start_date, end_date)

    # Drop tickers with no data at all
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"Warning: no data for {missing}. They will be dropped.")
        tickers = [t for t in tickers if t not in missing]

    prices, tickers, common_start, common_end = _trim_to_common_window(
        prices[tickers].sort_index(), tickers
    )
    print(
        "  Optimizer window aligned to common overlap: "
        f"{common_start.strftime('%Y-%m-%d')} → {common_end.strftime('%Y-%m-%d')}"
    )
    prices  = prices.ffill()
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")

    # Drop tickers with too little return history in the overlap window
    valid_counts = returns.count()
    too_sparse   = [t for t in tickers if valid_counts.get(t, 0) < 20]
    if too_sparse:
        print(f"Warning: insufficient return history for {too_sparse}. They will be dropped.")
        tickers = [t for t in tickers if t not in too_sparse]
        prices  = prices[tickers]
        returns = returns[tickers]

    if len(tickers) < 2:
        raise ValueError("Not enough assets with valid price history in the selected date range.")
    if returns.empty or len(returns) < 20:
        raise ValueError(
            "No usable overlapping return history found. "
            "Try a later start date or remove newer/staler symbols."
        )

    clipped_returns = returns.copy()
    if len(clipped_returns) >= 10:
        lower = clipped_returns.quantile(0.02)
        upper = clipped_returns.quantile(0.98)
        clipped_returns = clipped_returns.clip(lower=lower, upper=upper, axis=1)
    sample_mean = clipped_returns.mean()
    ewma_span = int(np.clip(len(clipped_returns) // 3, 21, 63))
    ewma_mean = clipped_returns.ewm(span=ewma_span, adjust=False).mean().iloc[-1]
    raw_er = 0.65 * sample_mean + 0.35 * ewma_mean
    cov_matrix = _stabilize_covariance(clipped_returns)

    # ── Shrinkage toward benchmark ────────────────────────────
    print(f"\nFetching benchmark for shrinkage  (alpha={alpha:.2f})...")
    candidates      = _detect_benchmark_candidates(tickers)
    bm_sym, bm_mean = _fetch_benchmark_daily_mean(candidates, start_date, end_date)

    if bm_mean is None or alpha >= 1.0:
        if bm_mean is None:
            print("  All benchmarks failed — using raw expected returns (no shrinkage).")
        expected_returns = raw_er.copy()
        bm_sym = None
    else:
        expected_returns = alpha * raw_er + (1.0 - alpha) * bm_mean

        # Transparency table
        print(f"\n  {'Ticker':<22} {'Raw ann.%':>10}  {'Adj ann.%':>10}  {'Delta ann.pp':>13}  Note")
        print("  " + "─" * 70)
        for t in tickers:
            raw_a = float(raw_er[t])             * 252 * 100
            adj_a = float(expected_returns[t])   * 252 * 100
            delta = adj_a - raw_a
            note  = "  pulled in hard" if abs(delta) > 10 else (
                    "  adjusted"       if abs(delta) >  3 else "")
            print(f"  {t:<22} {raw_a:>10.2f}  {adj_a:>10.2f}  {delta:>+13.2f}{note}")

    print(f"\n  Loaded {len(returns)} trading days for {len(tickers)} assets.\n")
    return tickers, returns, expected_returns, cov_matrix, raw_er, bm_sym
