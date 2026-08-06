"""Technical indicators implemented with pure pandas/numpy (no TA-Lib dependency).

All functions accept a pandas Series/DataFrame of prices and return aligned Series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period, min_periods=period).mean()


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = line - sig
    return pd.DataFrame({"macd": line, "signal": sig, "hist": hist})


def bollinger(close: pd.Series, period: int = 20, dev: float = 2.0) -> pd.DataFrame:
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return pd.DataFrame(
        {"mid": mid, "upper": mid + dev * std, "lower": mid - dev * std}
    )


def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    high = df["High"].rolling(period, min_periods=period).max()
    low = df["Low"].rolling(period, min_periods=period).min()
    return pd.DataFrame({"upper": high, "lower": low, "mid": (high + low) / 2})


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (trend strength, 0-100)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )
    tr = atr(df, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price (for intraday frames)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_vol = df["Volume"].cumsum()
    return (tp * df["Volume"]).cumsum() / cum_vol.replace(0, np.nan)


def vwap_std(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling std of price around VWAP (intraday deviation bands)."""
    v = vwap(df)
    diff = (df["Close"] - v) / v.replace(0, np.nan)
    return diff.rolling(period, min_periods=period).std(ddof=0)


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    return close.pct_change(period) * 100.0


def volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Annualised volatility of daily log returns."""
    rets = np.log(close / close.shift(1))
    return rets.rolling(period, min_periods=period).std(ddof=0) * np.sqrt(252)


def zscore(s: pd.Series, period: int = 20) -> pd.Series:
    mean = s.rolling(period, min_periods=period).mean()
    std = s.rolling(period, min_periods=period).std(ddof=0)
    return (s - mean) / std.replace(0, np.nan)


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def volume_z(df: pd.DataFrame, period: int = 20) -> pd.Series:
    vol = df["Volume"]
    mean = vol.rolling(period, min_periods=period).mean()
    std = vol.rolling(period, min_periods=period).std(ddof=0)
    return (vol - mean) / std.replace(0, np.nan)
