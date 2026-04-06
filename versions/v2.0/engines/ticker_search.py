"""
============================================================
  QuanSen — Ticker Search & Input Collection
============================================================
  Handles all user-facing input:
    - search_ticker()     search Yahoo Finance by company name
    - get_best_ticker()   pick NSE vs BSE based on data coverage
    - collect_tickers()   full interactive input session
============================================================
"""

import sys
import requests
import yfinance as yf


def search_ticker():
    """Search Yahoo Finance for a ticker by company name."""
    query = input("Enter company name: ").strip()
    url   = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        quotes = response.json().get("quotes", [])
        if not quotes:
            print("No results found. Try a different name.")
            return search_ticker()
        for i, q in enumerate(quotes[:10], 1):
            print(f"  {i}. {q.get('shortname', 'N/A')}  ({q['symbol']})")
        choice = int(input("Choose (1-10): "))
        return quotes[choice - 1]["symbol"]
    except Exception as e:
        print(f"Search failed: {e}")
        sys.exit(1)


def get_best_ticker(symbol, start_date, end_date):
    """
    For Indian stocks (.NS / .BO), compare both exchanges and pick
    the one with more trading days in the requested window.
    For all other tickers, return as-is.
    """
    is_indian = symbol.endswith(".NS") or symbol.endswith(".BO")

    if not is_indian:
        print(f"  Using {symbol} as provided.")
        return symbol

    base       = symbol.replace(".NS", "").replace(".BO", "")
    candidates = [f"{base}.NS", f"{base}.BO"]

    best_ticker = symbol
    best_count  = 0

    print(f"  Checking NSE vs BSE data for {base}...")
    for candidate in candidates:
        try:
            data  = yf.download(candidate, start=start_date, end=end_date,
                                auto_adjust=True, progress=False)
            count = len(data)
            print(f"    {candidate}: {count} trading days")
            if count > best_count:
                best_count  = count
                best_ticker = candidate
        except Exception:
            print(f"    {candidate}: no data")

    print(f"  Selected: {best_ticker} ({best_count} days)")
    return best_ticker


def collect_tickers():
    """
    Interactive session: ask the user how many stocks, dates,
    then search + validate each ticker.

    Returns
    -------
    tickers    : list[str]
    start_date : str  (YYYY-MM-DD)
    end_date   : str  (YYYY-MM-DD)
    """
    n     = int(input("How many stocks in the portfolio? "))
    start = input("Start date (YYYY-MM-DD): ").strip()
    end   = input("End date   (YYYY-MM-DD): ").strip()

    tickers = []
    for i in range(n):
        print(f"\nSearch stock {i + 1}:")
        raw    = search_ticker()
        best   = get_best_ticker(raw, start, end)
        tickers.append(best)

    print(f"\nFinal tickers selected: {tickers}")
    return tickers, start, end
