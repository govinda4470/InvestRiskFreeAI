"""Intraday strategies (5-minute bars, forced square-off at 15:25).

Intraday is the HIGHEST-risk style (SEBI: ~71% of intraday traders lose money),
so these strategies get the tightest stops and the brain gates them hardest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ta
from .base import Strategy


def _daily_atr_for_intraday(d: pd.DataFrame) -> pd.Series:
    """ATR of the daily bar containing each 5m bar (approximated per day)."""
    daily = d.resample("1D").agg({"High": "max", "Low": "min", "Close": "last"})
    daily_atr = ta.atr(daily, 14)
    d2 = d.copy()
    d2["day"] = d2.index.normalize()
    daily_atr.name = "daily_atr"
    d2 = d2.join(daily_atr, on="day")
    return d2["daily_atr"]


class IntradayORB(Strategy):
    """OPENING RANGE BREAKOUT (ORB).

    Rules:
      * 9:15-9:30 (first 3 bars) sets the opening range High/Low
      * long entry: close breaks above range_high + 0.35*ATR (entries until 12:00)
      * Stop : entry - 1.2*ATR   Target: entry + 2.2*ATR
      * forced square-off at 15:25 (never carry intraday risk overnight)
    """

    def __init__(self, params: dict | None = None):
        defaults = dict(opening_range_minutes=15, breakout_buffer_atr=0.35,
                        sl_atr=1.2, target_atr=2.2, entry_deadline="12:00",
                        atr_period=14)
        defaults.update(params or {})
        super().__init__("intraday_orb", "intraday", "5m", defaults)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["day"] = d.index.normalize()
        d["time"] = d.index.strftime("%H:%M")
        d["daily_atr"] = _daily_atr_for_intraday(d)
        # opening range per day (first 3 bars = 9:15-9:30)
        first = d.groupby("day").head(3)
        or_high = first.groupby("day")["High"].max().rename("or_high")
        or_low = first.groupby("day")["Low"].min().rename("or_low")
        d = d.join(or_high, on="day").join(or_low, on="day")
        return d

    def signals(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self.prepare(df)
        out = self.blank(d)
        p = self.params
        within_deadline = d["time"] <= p["entry_deadline"]
        after_open = d["time"] > "09:30"
        break_high = d["Close"] > d["or_high"] + p["breakout_buffer_atr"] * d["daily_atr"]
        cond = break_high & after_open & within_deadline
        # don't re-enter if already entered that day
        entry = cond & ~cond.shift(1, fill_value=False)
        out.loc[entry, "entry"] = 1.0
        out.loc[entry, "reason"] = "ORB: opening-range breakout"
        self.fill_sl_target(out, entry, d["Close"], d["daily_atr"],
                            p["sl_atr"], p["target_atr"])
        out["max_hold"] = 75  # day length in 5m bars; square-off forces the real exit
        out["square_off"] = (d["time"] >= "15:25").astype(float)
        return out


class IntradayVWAP(Strategy):
    """VWAP REVERSION - fade extreme deviations from the day's VWAP.

    Rules:
      * long entry: close falls below VWAP - 1.8*std, then next close back above VWAP
        (a snap-back bounce, only taken inside a rising session)
      * Stop : entry - 0.8*ATR (very tight)  Target: entry + 1.4*ATR
      * forced square-off at 15:25
    """

    def __init__(self, params: dict | None = None):
        defaults = dict(dev_mult=1.8, sl_atr=0.8, target_atr=1.4,
                        vwap_std_period=20, atr_period=14)
        defaults.update(params or {})
        super().__init__("intraday_vwap", "intraday", "5m", defaults)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["day"] = d.index.normalize()
        d["time"] = d.index.strftime("%H:%M")
        d["daily_atr"] = _daily_atr_for_intraday(d)
        d["vwap"] = d.groupby("day")["Close"].transform(
            lambda x: ta.vwap(pd.DataFrame({
                "High": x, "Low": x, "Close": x,
                "Volume": d.loc[x.index, "Volume"],
            }))
        )
        # deviation of price from vwap, rolling std per day
        dev = (d["Close"] - d["vwap"]) / d["vwap"].replace(0, np.nan)
        d["dev"] = dev
        period = self.params["vwap_std_period"]
        d["dev_std"] = dev.groupby(d["day"]).transform(
            lambda x: x.rolling(period, min_periods=5).std(ddof=0)
        )
        return d

    def signals(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self.prepare(df)
        out = self.blank(d)
        p = self.params
        below_band = d["dev"] < -p["dev_mult"] * d["dev_std"]
        back_above = (d["dev"] > 0) & below_band.shift(1, fill_value=False)
        cond = back_above & (d["time"] >= "09:45")
        entry = cond & ~cond.shift(1, fill_value=False)
        out.loc[entry, "entry"] = 1.0
        out.loc[entry, "reason"] = "VWAPReversion: snap-back above VWAP"
        self.fill_sl_target(out, entry, d["Close"], d["daily_atr"],
                            p["sl_atr"], p["target_atr"])
        out["max_hold"] = 75
        out["square_off"] = (d["time"] >= "15:25").astype(float)
        return out
