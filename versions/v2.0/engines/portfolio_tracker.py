"""
============================================================
  QuanSen — Live Portfolio Tracker Engine
  engines/portfolio_tracker.py

  Handles holdings CRUD, live quote fetching, P&L
  computation, drift vs target weights, rebalancing
  trade list, and sparkline data.

  Standalone — safe to import anywhere.
============================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo


# ── Holdings schema ───────────────────────────────────────────
# Each holding is a dict:
# {
#   "ticker":     str,
#   "shares":     float,
#   "avg_cost":   float,   # per share, in portfolio currency
#   "buy_date":   str,     # YYYY-MM-DD
#   "notes":      str,     # optional label
# }


def empty_holding(ticker: str = "") -> dict:
    return {
        "ticker":   ticker.upper().strip(),
        "shares":   0.0,
        "avg_cost": 0.0,
        "buy_date": date.today().strftime("%Y-%m-%d"),
        "notes":    "",
    }


def _exchange_timezone(ticker: str) -> ZoneInfo:
    """Return the exchange-local timezone for the symbol."""
    suffix = ticker.upper()
    if suffix.endswith(".NS") or suffix.endswith(".BO"):
        return ZoneInfo("Asia/Kolkata")
    return ZoneInfo("America/New_York")


# ── Live quote fetcher ────────────────────────────────────────

def _is_market_open(ticker: str) -> bool:
    """
    Heuristic: market is likely open if the most recent data row
    has a timestamp from today (UTC-adjusted for IST/EST).
    """
    try:
        local_date = datetime.now(_exchange_timezone(ticker)).date()
        df = yf.download(ticker, period="1d", interval="1m",
                         auto_adjust=True, progress=False)
        if df.empty:
            return False
        last_ts = df.index[-1]
        if getattr(last_ts, "tzinfo", None) is not None:
            last_ts = last_ts.tz_convert(_exchange_timezone(ticker))
        if hasattr(last_ts, "date"):
            return last_ts.date() == local_date
        return False
    except Exception:
        return False


def fetch_live_quotes(tickers: List[str]) -> Dict[str, dict]:
    """
    Fetch the most recent available price for each ticker.

    Strategy:
      1. Download last 5 trading days of daily OHLCV — always gives
         at least one valid close even across weekends / holidays.
      2. Use col.iloc[-1] as "price" (most recent close available).
      3. Use col.iloc[-2] as "prev_close" for day-change calculation.
      4. Set is_live=True only when the latest bar's date matches today
         in the exchange's local timezone; otherwise is_live=False and
         price_label="Last Close" so the UI can show the distinction.

    Returns dict keyed by ticker:
    {
      "price":       float,   most recent close (live or last close)
      "prev_close":  float,
      "chg_pct":     float,
      "chg_abs":     float,
      "high_52w":    float | None,
      "low_52w":     float | None,
      "volume":      int   | None,
      "mkt_cap":     float | None,
      "name":        str,
      "currency":    str,
      "is_live":     bool,   True = intraday / today's close available
      "price_label": str,    "Live" or "Last Close (YYYY-MM-DD)"
      "ok":          bool,
    }
    """
    if not tickers:
        return {}

    results = {}

    try:
        # 5-day window guarantees data across any weekend or 3-day holiday
        # Use per-ticker download to avoid MultiIndex issues across yfinance versions
        data = yf.download(
            tickers,
            period="5d",
            auto_adjust=True,
            progress=False,
        )
        if data.empty:
            for t in tickers:
                results[t] = _failed_quote(t)
            return results

        # ── Normalise to a simple {ticker: Series} close map ──────
        # yfinance >= 0.2 with multiple tickers returns MultiIndex
        # columns: ('Close','BEL.NS'), ('Open','BEL.NS'), ...
        # yfinance < 0.2 returns flat columns under data['Close'][ticker]
        # We handle both by extracting a plain ticker→Series dict.
        close = {}
        cols = data.columns
        is_multi = isinstance(cols, pd.MultiIndex)

        if len(tickers) == 1:
            # Single ticker — always flat
            t = tickers[0]
            if "Close" in data.columns:
                close[t] = data["Close"].dropna()
            else:
                close[t] = pd.Series(dtype=float)
        elif is_multi:
            # New yfinance MultiIndex: level 0 = field, level 1 = ticker
            if "Close" in cols.get_level_values(0):
                close_df = data["Close"]
                for t in tickers:
                    if t in close_df.columns:
                        close[t] = close_df[t].dropna()
                    else:
                        close[t] = pd.Series(dtype=float)
            else:
                for t in tickers:
                    close[t] = pd.Series(dtype=float)
        else:
            # Old flat MultiIndex style: columns are tickers under data['Close']
            try:
                close_df = data["Close"]
                if isinstance(close_df, pd.Series):
                    close[tickers[0]] = close_df.dropna()
                else:
                    for t in tickers:
                        close[t] = close_df[t].dropna() if t in close_df.columns else pd.Series(dtype=float)
            except Exception:
                for t in tickers:
                    close[t] = pd.Series(dtype=float)

        for t in tickers:
            try:
                col = close.get(t, pd.Series(dtype=float)).dropna()
                if col.empty:
                    results[t] = _failed_quote(t)
                    continue

                price = float(col.iloc[-1])
                prev  = float(col.iloc[-2]) if len(col) >= 2 else price

                # Determine if this is a live/today price or stale close
                last_date = col.index[-1]
                if hasattr(last_date, "date"):
                    last_date = last_date.date()
                elif hasattr(last_date, "to_pydatetime"):
                    last_date = last_date.to_pydatetime().date()

                is_ns = t.upper().endswith(".NS") or t.upper().endswith(".BO")
                exchange_tz = _exchange_timezone(t)
                if getattr(col.index, "tz", None) is not None:
                    last_date = pd.Timestamp(col.index[-1]).tz_convert(exchange_tz).date()
                local_today = datetime.now(exchange_tz).date()

                is_live    = (last_date == local_today)
                price_label = "Live" if is_live else "Last Close ({})".format(
                    last_date.strftime("%d %b %Y") if hasattr(last_date, "strftime") else str(last_date)
                )

                chg_abs = price - prev
                chg_pct = chg_abs / prev * 100 if prev != 0 else 0.0

                results[t] = {
                    "price":       price,
                    "prev_close":  prev,
                    "chg_pct":     round(chg_pct, 2),
                    "chg_abs":     round(chg_abs, 4),
                    "high_52w":    None,
                    "low_52w":     None,
                    "volume":      None,
                    "mkt_cap":     None,
                    "name":        t,
                    "currency":    "INR" if is_ns else "USD",
                    "is_live":     is_live,
                    "price_label": price_label,
                    "ok":          True,
                }
            except Exception:
                results[t] = _failed_quote(t)

    except Exception:
        for t in tickers:
            results[t] = _failed_quote(t)

    # Enrich with fast_info (52W high/low, market cap, name)
    for t in tickers:
        if not results.get(t, {}).get("ok"):
            continue
        try:
            fi = yf.Ticker(t).fast_info
            results[t]["high_52w"] = round(float(fi.year_high), 2) if fi.year_high else None
            results[t]["low_52w"]  = round(float(fi.year_low),  2) if fi.year_low  else None
            results[t]["mkt_cap"]  = fi.market_cap
            results[t]["name"]     = fi.name if hasattr(fi, "name") and fi.name else t
        except Exception:
            pass

    return results


def _failed_quote(ticker: str) -> dict:
    return {
        "price": None, "prev_close": None, "chg_pct": None,
        "chg_abs": None, "high_52w": None, "low_52w": None,
        "volume": None, "mkt_cap": None, "name": ticker,
        "currency": "?", "is_live": False, "price_label": "Unavailable",
        "ok": False,
    }


# ── Sparkline data ────────────────────────────────────────────

def fetch_sparklines(tickers: List[str], days: int = 30) -> Dict[str, pd.Series]:
    """
    Return a dict of {ticker: pd.Series of close prices} for the last `days` days.
    Version-safe: handles both flat and MultiIndex yfinance column structures.
    """
    if not tickers:
        return {}
    result = {}
    try:
        raw = yf.download(
            tickers, period="{:d}d".format(days),
            auto_adjust=True, progress=False,
        )
        if raw.empty:
            return {t: pd.Series(dtype=float) for t in tickers}

        cols     = raw.columns
        is_multi = isinstance(cols, pd.MultiIndex)

        if len(tickers) == 1:
            t = tickers[0]
            result[t] = raw["Close"].dropna() if "Close" in raw.columns else pd.Series(dtype=float)
        elif is_multi and "Close" in cols.get_level_values(0):
            close_df = raw["Close"]
            for t in tickers:
                result[t] = close_df[t].dropna() if t in close_df.columns else pd.Series(dtype=float)
        else:
            try:
                close_df = raw["Close"]
                for t in tickers:
                    result[t] = close_df[t].dropna() if t in close_df.columns else pd.Series(dtype=float)
            except Exception:
                for t in tickers:
                    result[t] = pd.Series(dtype=float)
    except Exception:
        for t in tickers:
            result[t] = pd.Series(dtype=float)
    return result


# ── P&L engine ────────────────────────────────────────────────

def compute_portfolio_pnl(
    holdings: List[dict],
    quotes: Dict[str, dict],
) -> Tuple[pd.DataFrame, dict]:
    """
    Compute per-holding and aggregate P&L.

    Returns
    -------
    df      : DataFrame with one row per holding + computed columns
    summary : dict of aggregate metrics
    """
    rows = []
    for h in holdings:
        t     = h["ticker"]
        q     = quotes.get(t, {})
        price = q.get("price") if q.get("ok") else None

        cost_basis  = h["shares"] * h["avg_cost"]
        curr_value  = h["shares"] * price if price is not None else None
        unrealised  = (curr_value - cost_basis) if curr_value is not None else None
        unreal_pct  = (unrealised / cost_basis * 100) if (cost_basis > 0 and unrealised is not None) else None
        day_chg_abs = h["shares"] * q.get("chg_abs", 0) if price is not None else None

        rows.append({
            "Ticker":       t,
            "Name":         q.get("name", t),
            "Shares":       round(h["shares"], 4),
            "Avg Cost":     round(h["avg_cost"], 2),
            "Live Price":   round(price, 2) if price else None,
            "Day Chg %":    q.get("chg_pct"),
            "Day Chg ₹":    round(day_chg_abs, 2) if day_chg_abs is not None else None,
            "Cost Basis":   round(cost_basis, 2),
            "Curr Value":   round(curr_value, 2) if curr_value is not None else None,
            "Unrealised ₹": round(unrealised, 2) if unrealised is not None else None,
            "Unrealised %": round(unreal_pct, 2) if unreal_pct is not None else None,
            "52W High":     q.get("high_52w"),
            "52W Low":      q.get("low_52w"),
            "Buy Date":     h.get("buy_date", ""),
            "Notes":        h.get("notes", ""),
            "_ok":          q.get("ok", False),
        })

    df = pd.DataFrame(rows)

    total_cost  = df["Cost Basis"].sum()
    valid       = df[df["Curr Value"].notna()]
    total_value = valid["Curr Value"].sum() if not valid.empty else 0.0
    total_unr   = valid["Unrealised ₹"].sum() if not valid.empty else 0.0
    total_day   = valid["Day Chg ₹"].sum() if "Day Chg ₹" in valid.columns else 0.0

    summary = {
        "total_cost":    round(total_cost, 2),
        "total_value":   round(total_value, 2),
        "total_unr":     round(total_unr, 2),
        "total_unr_pct": round(total_unr / total_cost * 100, 2) if total_cost > 0 else 0.0,
        "total_day_chg": round(total_day, 2),
        "n_holdings":    len(holdings),
        "n_winners":     int((df["Unrealised %"] > 0).sum()),
        "n_losers":      int((df["Unrealised %"] < 0).sum()),
        "best_ticker":   df.loc[df["Unrealised %"].idxmax(), "Ticker"] if not df["Unrealised %"].isna().all() else "—",
        "worst_ticker":  df.loc[df["Unrealised %"].idxmin(), "Ticker"] if not df["Unrealised %"].isna().all() else "—",
    }
    return df, summary


# ── Drift & rebalancing ───────────────────────────────────────

def compute_drift(
    holdings: List[dict],
    quotes: Dict[str, dict],
    target_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute current portfolio weights and drift from target.

    target_weights: {ticker: weight (0-1)} from optimizer.
                    If None, drift column is omitted.
    """
    values = {}
    for h in holdings:
        q = quotes.get(h["ticker"], {})
        if q.get("ok") and q.get("price"):
            values[h["ticker"]] = h["shares"] * q["price"]

    total = sum(values.values())
    if total == 0:
        return pd.DataFrame()

    rows = []
    for t, v in values.items():
        curr_w  = v / total
        tgt_w   = target_weights.get(t, 0.0) if target_weights else None
        drift   = curr_w - tgt_w if tgt_w is not None else None
        rows.append({
            "Ticker":         t,
            "Curr Value":     round(v, 2),
            "Curr Weight %":  round(curr_w * 100, 2),
            "Target Weight %": round(tgt_w * 100, 2) if tgt_w is not None else None,
            "Drift pp":        round(drift * 100, 2) if drift is not None else None,
        })

    return pd.DataFrame(rows).sort_values("Curr Weight %", ascending=False)


def compute_rebalance_trades(
    holdings: List[dict],
    quotes: Dict[str, dict],
    target_weights: Dict[str, float],
    total_capital: Optional[float] = None,
    allow_fractional: bool = False,
) -> pd.DataFrame:
    """
    Compute the trades needed to rebalance to target_weights.

    total_capital: if None, uses current portfolio value.
    Returns a DataFrame of BUY/SELL trades with share counts and values.
    """
    values = {}
    for h in holdings:
        q = quotes.get(h["ticker"], {})
        if q.get("ok") and q.get("price"):
            values[h["ticker"]] = {
                "price":  q["price"],
                "shares": h["shares"],
                "value":  h["shares"] * q["price"],
            }

    port_value = total_capital or sum(v["value"] for v in values.values())
    if port_value == 0:
        return pd.DataFrame()

    rows = []
    all_tickers = set(list(values.keys()) + list(target_weights.keys()))

    for t in all_tickers:
        tgt_w     = target_weights.get(t, 0.0)
        tgt_val   = port_value * tgt_w
        curr_val  = values[t]["value"]  if t in values else 0.0
        price     = values[t]["price"]  if t in values else quotes.get(t, {}).get("price", 0)
        curr_shr  = values[t]["shares"] if t in values else 0.0

        delta_val  = tgt_val - curr_val
        if price and price > 0:
            delta_shr = delta_val / price
            if not allow_fractional:
                delta_shr = int(delta_shr)   # floor toward zero
            delta_val = delta_shr * price
        else:
            delta_shr = 0.0
            delta_val = 0.0

        if abs(delta_shr) < 0.001:
            continue

        rows.append({
            "Ticker":       t,
            "Action":       "BUY" if delta_shr > 0 else "SELL",
            "Shares":       abs(round(delta_shr, 4)),
            "Price":        round(price, 2) if price else None,
            "Trade Value":  round(abs(delta_val), 2),
            "Curr Weight%": round(curr_val / port_value * 100, 2),
            "Tgt Weight%":  round(tgt_w * 100, 2),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Action", "Trade Value"], ascending=[True, False])
    return df


# ── Helpers ───────────────────────────────────────────────────

def holdings_to_df(holdings: List[dict]) -> pd.DataFrame:
    if not holdings:
        return pd.DataFrame(columns=["ticker","shares","avg_cost","buy_date","notes"])
    return pd.DataFrame(holdings)


def df_to_holdings(df: pd.DataFrame) -> List[dict]:
    return df.to_dict(orient="records")


# ── ISIN → NSE ticker map (Indian equities & ETFs) ───────────
_ISIN_MAP = {
    # Large / mid caps
    "INE002A01018": "RELIANCE.NS",   "INE040A01034": "HDFCBANK.NS",
    "INE090A01021": "ICICIBANK.NS",  "INE467B01029": "TCS.NS",
    "INE009A01021": "INFY.NS",       "INE397D01024": "HCLTECH.NS",
    "INE081A01020": "TATASTEEL.NS",  "INE205A01025": "VEDL.NS",
    "INE263A01024": "BEL.NS",        "INE752E01010": "POWERGRID.NS",
    "INE536H01010": "CIEINDIA.NS",   "INE758A01072": "AVANCE.BO",
    "INE066A01021": "ONGC.NS",       "INE585B01010": "BHARTIARTL.NS",
    "INE030A01027": "BAJFINANCE.NS", "INE029A01011": "BAJAJFINSV.NS",
    "INE021A01026": "SBIN.NS",       "INE238A01034": "AXISBANK.NS",
    "INE028A01039": "BANKBARODA.NS", "INE008A01015": "IDBI.NS",
    "INE883A01011": "ADANIENT.NS",   "INE423A01024": "ADANIPORTS.NS",
    "INE070A01015": "LT.NS",         "INE774D01024": "NESTLEIND.NS",
    "INE455K01017": "BAJAJ-AUTO.NS", "INE343H01029": "COALINDIA.NS",
    "INE213A01029": "NTPC.NS",       "INE242A01010": "IOC.NS",
    "INE101A01026": "WIPRO.NS",      "INE196A01026": "TECHM.NS",
    "INE860A01027": "SUNPHARMA.NS",  "INE058A01010": "DRREDDY.NS",
    "INE361B01024": "CIPLA.NS",      "INE044A01036": "HDFCLIFE.NS",
    "INE795G01014": "SBILIFE.NS",    "INE176B01034": "MUTHOOTFIN.NS",
    "INE322A01013": "LUPIN.NS",      "INE117A01022": "DIVISLAB.NS",
    "INE179A01014": "TITAN.NS",      "INE296A01024": "BRITANNIA.NS",
    "INE494B01023": "MARICO.NS",     "INE021A01026": "SBIN.NS",
    "INE848E01016": "NAUKRI.NS",     "INE745G01035": "ZOMATO.NS",
    "INE192R01011": "PAYTM.NS",      "INE018A01030": "MARUTI.NS",
    "INE0J1Y01017": "TATATECH.NS",   "INE155A01022": "TATAMOTORS.NS",
    "INE028A01039": "BANKBARODA.NS", "INE752E01010": "POWERGRID.NS",
    # ETFs / Mutual Fund units
    "INF204KC1089": "PHARMABEES.NS",  "INF277KA1984": "TATSILV.NS",
    "INF204KB17I05":"NIFTYBEES.NS",  "INF204KA1BZ7": "BANKBEES.NS",
    "INF732E01612": "GOLDBEES.NS",   "INF789F01XY0": "SILVERBEES.NS",
    "INF200KA1UM0": "MON100.NS",
}

# Column alias maps — broker → standard name
_COL_ALIASES = {
    # QuanSen native format
    "ticker":             "ticker",
    "shares":             "shares",
    "avg_cost":           "avg_cost",
    # Common broker formats
    "stock name":         "stock_name",
    "scrip name":         "stock_name",
    "symbol":             "ticker",
    "trading symbol":     "ticker",
    "isin":               "isin",
    "quantity":           "shares",
    "qty":                "shares",
    "net qty":            "shares",
    "average buy price":  "avg_cost",
    "avg. buy price":     "avg_cost",
    "average price":      "avg_cost",
    "avg price":          "avg_cost",
    "buy price":          "avg_cost",
    "purchase price":     "avg_cost",
    "average cost":       "avg_cost",
    "buy value":          "buy_value",
    "closing price":      "closing_price",
    "closing value":      "closing_value",
    "unrealised p&l":     "unrealised_pnl",
    "unrealized p&l":     "unrealised_pnl",
    "p&l":                "unrealised_pnl",
    "buy date":           "buy_date",
    "purchase date":      "buy_date",
    "notes":              "notes",
    "remarks":            "notes",
}


def _find_data_header_row(raw: pd.DataFrame) -> int:
    """
    Scan rows to find the first one that looks like a data header.
    Looks for rows containing quantity/shares/price keywords.
    """
    keywords = {"quantity", "qty", "shares", "price", "symbol",
                "stock name", "scrip", "isin", "ticker"}
    for i, row in raw.iterrows():
        vals = {str(v).strip().lower() for v in row.values if pd.notna(v)}
        if len(vals & keywords) >= 2:
            return i
    return 0


def _stock_name_to_ticker(name: str) -> str:
    """
    Convert a broker stock name to a best-guess NSE ticker.
    Applies common suffix/word removals and appends .NS.
    """
    import re
    name = str(name).strip().upper()
    # Remove common suffixes
    for pat in [r"LIMITED$", r"LTD\.?$", r"LTD$", r"CORPORATION$",
                r"CORP\.?$", r"INDUSTRIES$", r"INDUSTRY$",
                r"ENTERPRISES$", r"HOLDINGS$", r"GROUP$",
                r"& SONS$", r"AND SONS$", r"\(INDIA\)$", r"INDIA$",
                r"PVT\.?$", r"PRIVATE$", r"CO\.?$"]:
        name = re.sub(pat, "", name).strip()
    # Remove punctuation and extra spaces
    name = re.sub(r"[^A-Z0-9\-]", "", name)
    # Common known mappings by cleaned name fragment
    # Values are full ticker symbols including exchange suffix
    _NAME_FRAGMENTS = {
        "BHARATELEC":       "BEL.NS",
        "BHARATELECTRONICS": "BEL.NS",
        "TATASTEEL":        "TATASTEEL.NS",
        "VEDANTA":          "VEDL.NS",
        "POWERGRID":        "POWERGRID.NS",
        "POWERGRIDCORP":    "POWERGRID.NS",
        "CIEAUTOMOTIVE":    "CIEINDIA.NS",
        "NIPPONAMC":        "PHARMABEES.NS",   # Nippon Pharma BeES ETF
        "NIPPONPHARMA":     "PHARMABEES.NS",
        "PHARMABEES":       "PHARMABEES.NS",
        "TATAAML":          "TATSILV.NS",     # Tata Silver ETF — BSE only
        "TATASILV":         "TATSILV.NS",
        "TATASILVER":       "TATSILV.NS",
        "AVANCETECHNOLOGIES": "AVANCE.BO",     # Avance Tech — BSE only
        "AVANCETECH":       "AVANCE.BO",
        "AVANCE":           "AVANCE.BO",
    }
    clean = name.replace("-", "").replace(" ", "")
    for frag, full_ticker in _NAME_FRAGMENTS.items():
        if frag in clean:
            return full_ticker
    # Fallback: first 10 chars + .NS (best guess for unlisted names)
    return clean[:10] + ".NS"


def parse_holdings_csv(
    csv_bytes: bytes,
    xlsx_bytes: bytes = None,
) -> Tuple[List[dict], List[str]]:
    """
    Parse a holdings CSV/XLSX — handles both:
      1. QuanSen native format  (ticker, shares, avg_cost[, buy_date, notes])
      2. Indian broker statements (Zerodha, Groww, CDSL, NSDL, etc.)
         — auto-detects header row, normalises column names,
           resolves ISIN → NSE ticker, derives ticker from stock name.

    Returns (holdings_list, errors_list).
    """
    import io, re
    errors = []
    result = []

    try:
        # ── Load raw dataframe ────────────────────────────────
        if xlsx_bytes is not None:
            raw = pd.read_excel(io.BytesIO(xlsx_bytes), header=None)
        else:
            raw = pd.read_csv(io.BytesIO(csv_bytes), header=None)

        # ── Find actual data header row ───────────────────────
        header_idx = _find_data_header_row(raw)
        if xlsx_bytes is not None:
            df = pd.read_excel(io.BytesIO(xlsx_bytes), skiprows=header_idx)
        else:
            df = pd.read_csv(io.BytesIO(csv_bytes), skiprows=header_idx)

        # ── Normalise column names ────────────────────────────
        df.columns = [str(c).strip().lower() for c in df.columns]
        col_map = {}
        for col in df.columns:
            std = _COL_ALIASES.get(col)
            if std:
                col_map[col] = std
        df = df.rename(columns=col_map)

        # Drop completely empty rows
        df = df.dropna(how="all")

        # ── Detect format ─────────────────────────────────────
        has_ticker     = "ticker"     in df.columns
        has_isin       = "isin"       in df.columns
        has_stock_name = "stock_name" in df.columns
        has_shares     = "shares"     in df.columns
        has_avg_cost   = "avg_cost"   in df.columns

        if not has_shares:
            return [], ["Could not find a quantity/shares column. "
                        "Expected one of: Quantity, Qty, Shares."]
        if not has_avg_cost:
            return [], ["Could not find a price column. "
                        "Expected one of: Average buy price, Avg price, Purchase price."]

        # ── Resolve ticker for each row ───────────────────────
        warnings = []
        for _, row in df.iterrows():
            try:
                shares   = float(row["shares"])
                avg_cost = float(row["avg_cost"])
                if shares <= 0 or avg_cost <= 0:
                    continue

                ticker = None

                # Priority 1: explicit ticker column
                if has_ticker:
                    t = str(row["ticker"]).strip().upper()
                    if t and t != "NAN":
                        ticker = t if ("." in t) else t + ".NS"

                # Priority 2: ISIN lookup
                if ticker is None and has_isin:
                    isin = str(row.get("isin", "")).strip().upper()
                    if isin and isin != "NAN":
                        ticker = _ISIN_MAP.get(isin)
                        if ticker is None:
                            warnings.append(
                                "ISIN {} not in map — using name fallback".format(isin)
                            )

                # Priority 3: derive from stock name
                if ticker is None and has_stock_name:
                    name = str(row.get("stock_name", "")).strip()
                    if name and name != "nan":
                        ticker = _stock_name_to_ticker(name)
                        warnings.append(
                            "Derived ticker {} from name '{}' — verify it is correct".format(
                                ticker, name
                            )
                        )

                if ticker is None:
                    errors.append("Could not resolve ticker for row: {}".format(dict(row)))
                    continue

                h = empty_holding(ticker)
                h["shares"]   = round(shares, 4)
                h["avg_cost"] = round(avg_cost, 4)
                h["buy_date"] = str(row.get("buy_date", date.today().strftime("%Y-%m-%d")))
                h["notes"]    = str(row.get("notes", row.get("stock_name", "")))
                if h["notes"] == "nan":
                    h["notes"] = ""
                result.append(h)

            except Exception as e:
                errors.append("Row parse error: {}".format(e))

        errors = warnings + errors   # warnings first, hard errors last

    except Exception as e:
        errors.append("File parse failed: {}".format(e))

    return result, errors


def export_holdings_csv(holdings: List[dict]) -> bytes:
    return holdings_to_df(holdings).to_csv(index=False).encode()
