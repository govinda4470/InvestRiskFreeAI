"""Investment / position style - buy quality liquid large-caps on dips in a
confirmed long-term uptrend. Weeks-to-months holding, minimal screen time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ta
from .base import Strategy
from .intraday import IntradayORB, IntradayVWAP
from .swing import SwingBreakout, SwingMeanRev, SwingTrend


class InvestDip(Strategy):
    """TREND DIP INVESTOR.

    Rules:
      * close > SMA200 and SMA200 rising     (long-term uptrend)
      * close within pullback_pct of 20-day high  (a controlled dip)
      * RSI(14) between 35 and 60            (not falling knife, not chasing)
    Stop : close < SMA200 (trend broken)
    Target: +15% or max_hold days (120).
    """

    def __init__(self, params: dict | None = None):
        defaults = dict(ema_trend=50, sma_long=200, pullback_pct=5.0,
                        rsi_min=35, rsi_max=60, target_pct=15.0,
                        max_hold_days=120)
        defaults.update(params or {})
        super().__init__("invest", "invest", "daily", defaults)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        p = self.params
        d["sma_long"] = ta.sma(d["Close"], p["sma_long"])
        d["sma_long_prev"] = d["sma_long"].shift(20)
        d["high20"] = d["High"].rolling(20, min_periods=20).max()
        d["rsi"] = ta.rsi(d["Close"], 14)
        d["atr"] = ta.atr(d, 14)
        return d

    def signals(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self.prepare(df)
        out = self.blank(d)
        p = self.params
        uptrend = (d["Close"] > d["sma_long"]) & (d["sma_long"] > d["sma_long_prev"])
        dip = (d["High"] / d["high20"] - 1) * 100 <= p["pullback_pct"]
        rsi_ok = d["rsi"].between(p["rsi_min"], p["rsi_max"])
        cond = uptrend & dip & rsi_ok
        entry = cond & ~cond.shift(1, fill_value=False)
        out.loc[entry, "entry"] = 1.0
        out.loc[entry, "reason"] = "TrendDip: buy controlled dip in uptrend"
        # stop = trend break (checked on close), target = +15%
        out.loc[entry, "sl"] = d.loc[entry.index, "sma_long"]
        out.loc[entry, "target"] = d.loc[entry.index, "Close"] * (1 + p["target_pct"] / 100)
        out.loc[entry, "rr"] = np.nan  # stop is dynamic; engine treats as close-based exit
        out.loc[d["Close"] < d["sma_long"], "exit"] = 1.0
        out["max_hold"] = p["max_hold_days"]
        out["sl_on_close"] = 1.0  # trend-break stop checked on close, not intrabar wicks
        return out


class StrategyRegistry:
    """Factory of all built-in strategies."""

    @staticmethod
    def all() -> dict[str, Strategy]:
        return {
            "swing_trend": SwingTrend(),
            "swing_meanrev": SwingMeanRev(),
            "swing_breakout": SwingBreakout(),
            "intraday_orb": IntradayORB(),
            "intraday_vwap": IntradayVWAP(),
            "invest": InvestDip(),
        }

    @staticmethod
    def get(name: str) -> Strategy:
        return StrategyRegistry.all()[name]
