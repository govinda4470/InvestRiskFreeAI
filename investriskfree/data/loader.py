"""Data loading for NSE stocks.

Sources (tried in order):
1. yfinance  - fresh daily + 5-minute data (works on the user's machine; NSE delayed ~15 min)
2. NSE API   - official NSE quotes (best-effort; requires working session cookies)
3. bundled   - the real NSE daily dataset shipped in data/bundled/nse_daily (offline fallback)
"""
from __future__ import annotations

import glob
import os
import time

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


def _normalize_yfinance(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """Normalize both flat and MultiIndex yfinance download results."""
    if isinstance(df.columns, pd.MultiIndex):
        # Recent yfinance versions return (field, ticker) columns even for one symbol.
        if ticker and ticker in df.columns.get_level_values(-1):
            try:
                df = df.xs(ticker, axis=1, level=-1)
            except KeyError:
                pass
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    rename = {str(column): str(column).strip().capitalize() for column in df.columns}
    df = df.rename(columns=rename)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(column in df for column in required):
        raise ValueError("yfinance response is missing OHLCV fields")
    out = df[required].copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(None)
    out.index.name = "Date"
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def load_daily(symbol: str, source: str | None = None) -> pd.DataFrame:
    """Daily OHLCV for an NSE symbol (without the ``.NS`` suffix).

    Explicit ``source='yfinance'`` never silently pretends the bundled 2022 data
    is live.  A fresh (18-hour) yfinance cache is allowed; otherwise a failed
    download raises so auto-trading cannot execute against stale history.
    """
    source = source or get("universe.source", "bundled")
    bundled_dir = get("data.bundled_dir")
    cache_dir = os.path.join(get("data.cache_dir"), "daily")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{symbol}.csv")

    if source in ("yfinance", "auto"):
        fresh = os.path.exists(cache_path) and time.time() - os.path.getmtime(cache_path) < 18 * 3600
        if fresh:
            try:
                return _read_csv_robust(cache_path)
            except Exception:
                pass
        error = None
        try:
            import yfinance as yf

            ticker = f"{symbol}.NS"
            downloaded = yf.download(
                ticker,
                period="10y",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if downloaded is None or len(downloaded) <= 100:
                raise RuntimeError("fewer than 100 daily bars returned")
            frame = _normalize_yfinance(downloaded, ticker)
            frame.to_csv(cache_path)
            return frame
        except Exception as exc:
            error = exc
        if source == "yfinance":
            raise RuntimeError(f"live yfinance daily data unavailable for {symbol}: {error}")
        # ``auto`` may use an older downloaded cache before the bundled fallback.
        if os.path.exists(cache_path):
            try:
                return _read_csv_robust(cache_path)
            except Exception:
                pass

    p = os.path.join(bundled_dir, f"{symbol}.csv")
    if os.path.exists(p):
        return _read_csv_robust(p)
    raise FileNotFoundError(f"No data for {symbol} (checked yfinance and bundled)")


def load_index_daily(symbol: str = "^NSEI", source: str | None = None) -> pd.DataFrame:
    """Nifty index daily data with the same stale-data rules as ``load_daily``."""
    source = source or get("universe.source", "bundled")
    cache_dir = os.path.join(get("data.cache_dir"), "daily")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "NIFTY50_LIVE.csv")
    if source in ("yfinance", "auto"):
        fresh = os.path.exists(cache_path) and time.time() - os.path.getmtime(cache_path) < 18 * 3600
        if fresh:
            try:
                return _read_csv_robust(cache_path)
            except Exception:
                pass
        error = None
        try:
            import yfinance as yf

            downloaded = yf.download(
                symbol, period="10y", interval="1d", progress=False,
                auto_adjust=True, threads=False,
            )
            if downloaded is None or len(downloaded) <= 100:
                raise RuntimeError("fewer than 100 index bars returned")
            frame = _normalize_yfinance(downloaded, symbol)
            frame.to_csv(cache_path)
            return frame
        except Exception as exc:
            error = exc
        if source == "yfinance":
            raise RuntimeError(f"live yfinance index data unavailable: {error}")
        if os.path.exists(cache_path):
            try:
                return _read_csv_robust(cache_path)
            except Exception:
                pass
    path = os.path.join(get("data.bundled_dir"), "NIFTY50_IDX.csv")
    if os.path.exists(path):
        frame = _read_csv_robust(path)
        return frame.rename(columns={"Shares Traded": "Volume"})
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
