"""Low-risk stock universe.

We only trade liquid, large-cap NSE names: deep liquidity means tight spreads,
fewer manipulation games, and realistic fills. Small caps with thin volume are
excluded even if they look cheap — they are where retail capital gets trapped.
"""
from __future__ import annotations

import pandas as pd

from ..config import get
from .loader import list_bundled_symbols, load_daily


def _liquidity_ok(df: pd.DataFrame) -> tuple[bool, dict]:
    stats = {
        "avg_volume": float(df["Volume"].tail(60).mean()),
        "avg_close": float(df["Close"].tail(60).mean()),
        "avg_turnover_cr": float(
            (df["Close"] * df["Volume"]).tail(60).mean() / 1e7
        ),
        "max_price": float(df["Close"].tail(60).max()),
        "min_price": float(df["Close"].tail(60).min()),
    }
    if stats["min_price"] < get("universe.min_price", 20):
        return False, stats
    if stats["max_price"] > get("universe.max_price", 5000):
        return False, stats
    if stats["avg_volume"] < get("universe.min_avg_volume", 100_000):
        return False, stats
    if stats["avg_turnover_cr"] < get("universe.min_avg_turnover_cr", 5):
        return False, stats
    return True, stats


def get_universe() -> list[dict]:
    """Return a list of {symbol, stats} passing the low-risk liquidity filter."""
    exclude = set(get("universe.exclude", []))
    out = []
    for sym in list_bundled_symbols():
        if sym in exclude:
            continue
        try:
            df = load_daily(sym)
            if len(df) < 500:  # need enough history for 200SMA etc.
                continue
            ok, stats = _liquidity_ok(df)
            if ok:
                stats["symbol"] = sym
                out.append(stats)
        except Exception:
            continue
    out.sort(key=lambda s: -s["avg_turnover_cr"])
    return out
