"""Swing-trading strategies (daily bars, 2-25 day holding period)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ta
from .base import Strategy


class SwingTrend(Strategy):
    """TREND RIDER - buy the pullback inside a confirmed uptrend.

    Rules (all must be true to trigger):
      * close > EMA20 > EMA50           (medium-term uptrend)
      * ADX(14) > 18                    (real trend, not chop)
      * RSI(14) < 72                    (not over-bought / chasing)
      * close within 0.7*ATR of EMA20   (a fresh pullback, not extended)
    Stop : entry - 2.2*ATR
    Target: entry + 3.5*ATR
    Exit : close < EMA20 (trend broken) or max_hold days
    """

    def __init__(self, params: dict | None = None):
        defaults = dict(ema_fast=20, ema_slow=50, atr_period=14, atr_sl_mult=2.5,
                        atr_target_mult=4.0, max_hold_days=20, adx_min=18, rsi_max=72)
        defaults.update(params or {})
        super().__init__("swing_trend", "swing", "daily", defaults)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        p = self.params
        d["ema_fast"] = ta.ema(d["Close"], p["ema_fast"])
        d["ema_slow"] = ta.ema(d["Close"], p["ema_slow"])
        d["atr"] = ta.atr(d, p["atr_period"])
        d["adx"] = ta.adx(d, p["atr_period"])
        d["rsi"] = ta.rsi(d["Close"], 14)
        return d

    def signals(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self.prepare(df)
        out = self.blank(d)
        p = self.params
        cond = (
            (d["Close"] > d["ema_fast"])
            & (d["ema_fast"] > d["ema_slow"])
            & (d["adx"] > p["adx_min"])
            & (d["rsi"] < p["rsi_max"])
            & ((d["Close"] - d["ema_fast"]) <= 0.7 * d["atr"])
        )
        entry = cond & ~cond.shift(1, fill_value=False)
        out.loc[entry, "entry"] = 1.0
        out.loc[entry, "reason"] = "TrendRider: EMA20/50 uptrend pullback"
        self.fill_sl_target(out, entry, d["Close"], d["atr"],
                            p["atr_sl_mult"], p["atr_target_mult"])
        # exit when the MEDIUM trend breaks (EMA50), not the fast one -
        # exiting on EMA20 whipsaws good trades (validated by sweep)
        out.loc[d["Close"] < d["ema_slow"], "exit"] = 1.0
        out["max_hold"] = p["max_hold_days"]
        return out


class SwingMeanRev(Strategy):
    """DIP BUYER - short-term RSI(2) mean reversion inside an uptrend.

    Rules:
      * close > EMA50                    (only buy dips in an uptrend)
      * RSI(2) < 15                      (short-term panic / oversold)
    Stop : entry - 1.5*ATR   (tight - capital protection first)
    Target: entry + 3.0*ATR
    Exit : RSI(2) >= 60 (bounce complete) or close < EMA50 or max_hold days.
    High win-rate, smaller winners; the tight stop keeps losses tiny.
    """

    def __init__(self, params: dict | None = None):
        defaults = dict(rsi_period=2, rsi_buy_below=15, rsi_exit_above=60,
                        ema_trend=50, atr_period=14, atr_sl_mult=1.5,
                        atr_target_mult=3.0, max_hold_days=15)
        defaults.update(params or {})
        super().__init__("swing_meanrev", "swing", "daily", defaults)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        p = self.params
        d["rsi_short"] = ta.rsi(d["Close"], p["rsi_period"])
        d["ema_trend"] = ta.ema(d["Close"], p["ema_trend"])
        d["atr"] = ta.atr(d, p["atr_period"])
        return d

    def signals(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self.prepare(df)
        out = self.blank(d)
        p = self.params
        cond = (d["Close"] > d["ema_trend"]) & (d["rsi_short"] < p["rsi_buy_below"])
        entry = cond & ~cond.shift(1, fill_value=False)
        out.loc[entry, "entry"] = 1.0
        out.loc[entry, "reason"] = "DipBuyer: RSI(2) oversold in uptrend"
        self.fill_sl_target(out, entry, d["Close"], d["atr"],
                            p["atr_sl_mult"], p["atr_target_mult"])
        out.loc[(d["rsi_short"] >= p["rsi_exit_above"]) | (d["Close"] < d["ema_trend"]), "exit"] = 1.0
        out["max_hold"] = p["max_hold_days"]
        return out


class SwingBreakout(Strategy):
    """RANGE BREAKER - Donchian breakout with volume confirmation.

    Rules:
      * close > highest high of last 20 bars
      * volume > 1.5x the 20-day average    (real participation)
      * RSI(14) > 55                        (momentum agreeing)
      * close > EMA50                       (only from above the trend line)
    Stop : entry - 2.0*ATR
    Target: entry + 3.0*ATR
    Exit : close < EMA20 or max_hold days.
    """

    def __init__(self, params: dict | None = None):
        defaults = dict(donchian_period=20, vol_mult=1.5, atr_period=14,
                        atr_sl_mult=3.0, atr_target_mult=4.0,
                        max_hold_days=25, rsi_min=55, ema_guard=50)
        defaults.update(params or {})
        super().__init__("swing_breakout", "swing", "daily", defaults)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        p = self.params
        dc = ta.donchian(d, p["donchian_period"])
        d["dc_high"] = dc["upper"].shift(1)          # prior period high (avoid self-inclusion)
        d["vol_avg"] = d["Volume"].rolling(p["donchian_period"], min_periods=5).mean()
        d["atr"] = ta.atr(d, p["atr_period"])
        d["rsi"] = ta.rsi(d["Close"], 14)
        d["ema_guard"] = ta.ema(d["Close"], p["ema_guard"])
        d["ema_exit"] = ta.ema(d["Close"], 20)
        return d

    def signals(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self.prepare(df)
        out = self.blank(d)
        p = self.params
        cond = (
            (d["Close"] > d["dc_high"])
            & (d["Volume"] > p["vol_mult"] * d["vol_avg"])
            & (d["rsi"] > p["rsi_min"])
            & (d["Close"] > d["ema_guard"])
        )
        entry = cond & ~cond.shift(1, fill_value=False)
        out.loc[entry, "entry"] = 1.0
        out.loc[entry, "reason"] = "RangeBreaker: Donchian(20) breakout + volume"
        self.fill_sl_target(out, entry, d["Close"], d["atr"],
                            p["atr_sl_mult"], p["atr_target_mult"])
        out.loc[d["Close"] < d["ema_exit"], "exit"] = 1.0
        out["max_hold"] = p["max_hold_days"]
        return out
