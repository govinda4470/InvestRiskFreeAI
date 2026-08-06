"""Synthetic intraday bars from real daily OHLC (Brownian bridge).

Only used when no real intraday feed is available (e.g. offline demo in this
sandbox). On the user's machine the scanner/backtester uses REAL 5-minute bars
from yfinance. Everywhere synthetic data is shown, the UI labels it as DEMO data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_to_intraday(
    daily: pd.DataFrame,
    interval_min: int = 5,
    seed: int = 42,
    sessions_per_day: int = 75,  # 9:15-15:30 = 75 x 5min bars
) -> pd.DataFrame:
    """Create 5-minute bars inside each daily bar using a seeded Brownian bridge
    that respects the real open/high/low/close/volume of the day."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, d in daily.iterrows():
        o, h, l, c = float(d["Open"]), float(d["High"]), float(d["Low"]), float(d["Close"])
        vol = float(d["Volume"]) if "Volume" in d and not np.isnan(d["Volume"]) else 1e6
        n = sessions_per_day
        # bridge from open to close, scaled so extremes hit the daily range
        t = np.linspace(0, 1, n + 1)
        bridge = np.zeros(n + 1)
        z = rng.normal(0, 1, n)
        bridge[1:] = np.cumsum(z) / np.sqrt(n)
        bridge = bridge - t * bridge[-1]  # start=0, end=0
        base = o + (c - o) * t
        price = base + bridge * max((h - l), (c - o).__abs__() * 0.5) * 0.5
        price = np.clip(price, min(o, l) * 0.999, max(o, h) * 1.001)
        price[-1] = c
        price[0] = o
        open_ = price[:-1]
        close_ = price[1:]
        high_ = np.maximum(open_, close_) * (1 + rng.uniform(0, 0.0006, n))
        low_ = np.minimum(open_, close_) * (1 - rng.uniform(0, 0.0006, n))
        high_ = np.maximum(high_, np.maximum(open_, close_))
        low_ = np.minimum(low_, np.minimum(open_, close_))
        vol_split = np.full(n, vol / n)
        start = pd.Timestamp(d.name)
        if hasattr(start, "tz") and start.tzinfo is None:
            start = start.tz_localize("Asia/Kolkata")
        times = [
            start.replace(hour=9, minute=15) + pd.Timedelta(minutes=interval_min) * i
            for i in range(n)
        ]
        rows.append(
            pd.DataFrame(
                {
                    "Open": open_,
                    "High": high_,
                    "Low": low_,
                    "Close": close_,
                    "Volume": vol_split,
                },
                index=times,
            )
        )
    out = pd.concat(rows)
    out.index.name = "datetime"
    return out
