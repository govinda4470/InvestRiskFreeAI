"""The AI Brain: market regime, confluence scoring, confidence, position sizing.

The brain is the capital-protection layer:
  1. REGIME  - is the market in a tradable state at all? (RISK-ON / NEUTRAL / RISK-OFF)
  2. CONFIDENCE - how strongly do multiple independent signals agree? (0-100)
  3. ML GATE - does a walk-forward-trained model predict P(win) >= baseline?
  4. SIZING  - position size derived from the risk budget, never from greed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ta
from .config import get


# ---------------------------------------------------------------- regime
def market_regime(index_df: pd.DataFrame, breadth: pd.Series | None = None) -> pd.Series:
    """Daily regime series: True = longs allowed (index above 200SMA)."""
    close = index_df["Close"]
    sma200 = ta.sma(close, 200)
    ok = close > sma200
    return ok.fillna(False)


def market_breadth(stock_closes: pd.DataFrame) -> pd.Series:
    """Fraction of universe stocks trading above their 200SMA (0-1)."""
    frac = stock_closes.apply(
        lambda s: (s > ta.sma(s, 200)).astype(float), axis=0
    ).mean(axis=1, skipna=True)
    return frac


def regime_label(index_df: pd.DataFrame, breadth: float | None = None) -> tuple[str, float, float]:
    """Current regime label + index-vs-200SMA distance + breadth."""
    close = index_df["Close"]
    sma200 = ta.sma(close, 200).iloc[-1]
    last = close.iloc[-1]
    dist = (last / sma200 - 1) * 100 if sma200 == sma200 else 0.0
    b = breadth if breadth is not None else 0.5
    if dist > 0 and b >= get("brain.regime.min_breadth", 0.4):
        label = "RISK-ON" if (dist > 2 and b > 0.6) else "NEUTRAL"
    else:
        label = "RISK-OFF"
    return label, float(dist), float(b)


# ---------------------------------------------------------------- confidence
def confidence_score(
    stock_df: pd.DataFrame, breadth: float, regime_ok: bool,
    confluence: int = 1, p_win: float | None = None,
) -> float:
    """0-100 confluence confidence. Sub-55 = no trade (capital protection)."""
    d = stock_df
    close = d["Close"]
    atr_pct = float(ta.atr(d, 14).iloc[-1] / close.iloc[-1] * 100)
    rsi = float(ta.rsi(close, 14).iloc[-1])
    ema20, ema50 = float(ta.ema(close, 20).iloc[-1]), float(ta.ema(close, 50).iloc[-1])
    sma200 = float(ta.sma(close, 200).iloc[-1])
    adx = float(ta.adx(d, 14).iloc[-1])
    vol20 = float(d["Volume"].tail(20).mean())
    vol200 = float(d["Volume"].tail(200).mean())

    score = 0.0
    # trend (25 pts)
    t = 0
    if close.iloc[-1] > ema20:
        t += 8
    if ema20 > ema50:
        t += 9
    if close.iloc[-1] > sma200:
        t += 8
    score += t
    # momentum (15)
    m = 0
    if rsi > 50:
        m += 6
    if adx > 20:
        m += 9
    score += m
    # volatility fit (15): not dead, not wild
    v = 0
    if 0.8 <= atr_pct <= 3.5:
        v += 15
    elif atr_pct < 4.5:
        v += 8
    score += v
    # liquidity (15)
    liq = 0
    if vol20 > 1_000_000:
        liq += 15
    elif vol20 > 300_000:
        liq += 9
    score += liq
    # volume trend (5)
    if vol20 > vol200:
        score += 5
    # regime (15)
    if regime_ok:
        score += 15
    elif breadth >= get("brain.regime.min_breadth", 0.4):
        score += 7
    # confluence bonus (10): multiple strategies agreeing
    score += min(10, (confluence - 1) * 5)
    # ML agreement (optional)
    if p_win is not None and p_win == p_win:
        if p_win >= 0.55:
            score += 5
        elif p_win < 0.45:
            score -= 8
    return float(np.clip(score, 0, 100))


# ---------------------------------------------------------------- sizing
def position_size(
    capital: float, entry: float, sl: float,
    risk_pct: float | None = None, max_pos_pct: float | None = None,
    min_value: float | None = None,
) -> dict:
    """Risk-budget position sizing. NEVER risks more than risk_pct of capital."""
    risk_pct = risk_pct if risk_pct is not None else get("capital.risk_per_trade_pct", 0.5) / 100
    max_pos_pct = max_pos_pct if max_pos_pct is not None else get("capital.max_position_pct", 25) / 100
    min_value = min_value if min_value is not None else get("capital.min_position_value", 2500)
    if entry <= 0 or sl <= 0 or entry <= sl:
        return {"qty": 0, "risk_rs": 0.0, "pos_value": 0.0, "blocked": "invalid levels"}
    risk_per_share = entry - sl
    risk_amt = capital * risk_pct
    qty = int(risk_amt // risk_per_share)
    pos_value = qty * entry
    if pos_value > capital * max_pos_pct:
        qty = int(capital * max_pos_pct // entry)
        pos_value = qty * entry
    if qty <= 0:
        return {"qty": 0, "risk_rs": 0.0, "pos_value": 0.0,
                "blocked": f"capital too small: risk ₹{risk_amt:.0f} < 1 share's risk ₹{risk_per_share:.0f}"}
    if pos_value < min_value:
        return {"qty": qty, "risk_rs": risk_amt, "pos_value": pos_value,
                "blocked": f"position ₹{pos_value:,.0f} below min ₹{min_value:,.0f} (costs would eat profits)"}
    return {"qty": qty, "risk_rs": risk_amt, "pos_value": pos_value, "blocked": None}
