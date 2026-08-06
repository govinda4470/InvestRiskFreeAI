"""Data loading for NSE stocks.

Sources (tried in order):
1. yfinance  - fresh daily + 5-minute data (works on the user's machine; NSE delayed ~15 min)
2. NSE API   - official NSE quotes (best-effort; requires working session cookies)
3. bundled   - the real NSE daily dataset shipped in data/bundled/nse_daily (offline fallback)
"""
from __future__ import annotations

import glob
import os
from functools import lru_cache

import pandas as pd

from ..config import get


def _read_csv_robust(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "Date" not in df.columns:
        raise KeyError(f"no Date column in {path}")
    # bundled NSE bhavcopy files use DD-MMM-YYYY ("15-May-2017");
    # yfinance-style CSVs use ISO YYYY-MM-DD. Parse ISO FIRST (never use
    # dayfirst with format='mixed' on ISO strings - it swaps day/month!).
    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.drop_duplicates(subset="Date", keep="last")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    # ---- data quality cleaning ----
    # NSE never trades weekends; drop any stray weekend bars
    df = df[df["Date"].dt.dayofweek < 5]
    # drop impossible single-day moves (>25%) - data errors, not real moves
    ret = df["Close"].pct_change()
    df = df[(ret.abs() < 0.25) | ret.isna()]
    # OHLC sanity: high >= max(open, close) >= min(open, close) >= low
    df = df[(df["High"] >= df[["Open", "Close"]].max(axis=1) - 1e-9) &
            (df["Low"] <= df[["Open", "Close"]].min(axis=1) + 1e-9) &
            (df["High"] > 0) & (df["Low"] > 0)]
    # old NSE bhavcopy has 'Close Price' etc.
    df.attrs["symbol"] = os.path.basename(path).rsplit(".", 1)[0]
    if "Close" not in df.columns and "Close Price" in df.columns:
        df = df.rename(
            columns={
                "Open Price": "Open",
                "High Price": "High",
                "Low Price": "Low",
                "Close Price": "Close",
                "Total Traded Quantity": "Volume",
            }
        )
    return df.set_index("Date")


@lru_cache(maxsize=512)
def load_daily(symbol: str, source: str | None = None) -> pd.DataFrame:
    """Daily OHLCV for an NSE symbol. symbol like 'RELIANCE' (without .NS)."""
    source = source or get("universe.source", "bundled")
    bundled_dir = get("data.bundled_dir")
    cache_dir = get("data.cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"{symbol}.csv")
    if os.path.exists(cache_path):
        try:
            return _read_csv_robust(cache_path)
        except Exception:
            pass

    if source in ("yfinance", "auto"):
        try:
            import yfinance as yf

            df = yf.download(
                f"{symbol}.NS",
                period="10y",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if df is not None and len(df) > 100:
                df = df.rename(columns=str.capitalize)
                df.columns = [c.split()[0] for c in df.columns]
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                df.index = pd.to_datetime(df.index)
                df.index.name = "Date"
                df.to_csv(cache_path)
                return df
        except Exception:
            pass

    # bundled real NSE data (offline fallback)
    p = os.path.join(bundled_dir, f"{symbol}.csv")
    if os.path.exists(p):
        return _read_csv_robust(p)
    raise FileNotFoundError(f"No data for {symbol} (checked cache, yfinance, bundled)")


def load_index_daily(symbol: str = "^NSEI", source: str | None = None) -> pd.DataFrame:
    """Nifty index daily data. Bundled index file covers 2017-2019 only;
    on the user's machine yfinance provides the full history."""
    if source in ("yfinance", "auto"):
        try:
            import yfinance as yf

            df = yf.download(
                symbol, period="10y", interval="1d", progress=False, auto_adjust=True
            )
            if df is not None and len(df) > 100:
                df.columns = [c.split()[0] for c in df.columns]
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                df.index = pd.to_datetime(df.index)
                df.index.name = "Date"
                return df
        except Exception:
            pass
    p = os.path.join(get("data.bundled_dir"), "NIFTY50_IDX.csv")
    if os.path.exists(p):
        df = _read_csv_robust(p)
        return df.rename(columns={"Shares Traded": "Volume"})
    raise FileNotFoundError("no index data")


def fetch_quote(symbol: str) -> float | None:
    """Latest tradable price for an NSE symbol (yfinance live), or None.

    Works on the user's machine / Streamlit Cloud where outbound internet is
    available. Returns None when no live feed is reachable (offline sandbox),
    so callers fall back to the last close of the bundled dataset.
    """
    try:
        import yfinance as yf

        t = yf.Ticker(f"{symbol}.NS")
        try:
            fi = t.fast_info
            px = fi.get("last_price")
            if px is not None and float(px) > 0:
                return float(px)
        except Exception:
            pass
        h = t.history(period="2d", interval="1d", auto_adjust=True)
        if h is not None and len(h) > 0:
            px = float(h["Close"].iloc[-1])
            if px > 0:
                return px
    except Exception:
        pass
    return None


def list_bundled_symbols() -> list[str]:
    """All symbols with bundled data (excludes the index file)."""
    out = []
    for p in glob.glob(os.path.join(get("data.bundled_dir"), "*.csv")):
        name = os.path.basename(p)[:-4]
        if name != "NIFTY50_IDX":
            out.append(name)
    return sorted(out)
