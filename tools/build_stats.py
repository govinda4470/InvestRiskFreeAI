#!/usr/bin/env python3
"""Build the strategy performance registry used by the scanner.

Runs every enabled strategy over the full stock universe (real NSE daily data)
and saves per-strategy + per-symbol aggregate stats to data/strategy_stats.json.
The scanner then shows these numbers on every signal:
  - profit probability (win rate)
  - expected trade duration (median hold days)
  - expectancy per trade, max drawdown, profit factor

Usage:
    python tools/build_stats.py [--symbols RELIANCE,HDFCBANK] [--years 10]

The registry is a cold-start estimate: it is static until the user re-runs it
against LIVE data on their machine (yfinance source gives 2015->today).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from investriskfree.backtest import Backtester  # noqa: E402
from investriskfree.brain import market_regime  # noqa: E402
from investriskfree.config import get  # noqa: E402
from investriskfree.data.loader import list_bundled_symbols, load_daily  # noqa: E402
from investriskfree.strategies import StrategyRegistry  # noqa: E402

OUT_PATH = os.path.join(get("data.repo_root"), "data", "strategy_stats.json")


def _regime_map(stocks: list[str]) -> dict[str, np.ndarray]:
    """Regime proxy: equal-weight index of the universe above 200SMA."""
    closes = {}
    for s in stocks:
        try:
            closes[s] = load_daily(s)["Close"]
        except Exception:
            continue
    proxy = pd.DataFrame(closes).mean(axis=1).to_frame("Close")
    proxy["Open"] = proxy["High"] = proxy["Low"] = proxy["Close"]
    proxy["Volume"] = 1
    regime = market_regime(proxy)
    return {s: regime.reindex(load_daily(s).index).ffill().fillna(False).to_numpy()
            for s in closes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", help="comma-separated subset")
    ap.add_argument("--years", type=int, default=10, help="years of data per stock")
    args = ap.parse_args()

    symbols = args.symbols.split(",") if args.symbols else list_bundled_symbols()
    registry = StrategyRegistry.all()
    strategies = {k: v for k, v in registry.items() if get(f"strategies.{k}.enabled", False)}
    if not strategies:
        print("No enabled strategies in config.yaml - nothing to build.")
        return
    regimes = _regime_map(symbols)
    t0 = time.time()

    by_strat: dict[str, list[dict]] = {k: [] for k in strategies}
    by_pair: dict[str, dict] = {}
    demo_intraday: dict[str, dict] = {}

    for i, sym in enumerate(symbols):
        try:
            daily = load_daily(sym)
            if args.years:
                daily = daily.tail(args.years * 252)
            reg = regimes.get(sym)
        except Exception:
            continue
        for name, strat in strategies.items():
            try:
                if strat.style == "intraday":
                    # intraday needs 5m bars: use real NSE daily -> synthetic
                    # 5m bridge, explicitly marked demo. On the user's machine
                    # the scanner can use REAL 5m bars from yfinance instead.
                    from investriskfree.data.synthetic import daily_to_intraday
                    intra = daily_to_intraday(daily.tail(700), seed=42)
                    res = Backtester().run(intra, strat, capital=100_000)
                else:
                    res = Backtester().run(daily, strat, regime_ok=reg, capital=100_000)
                st = res.stats
                rec = {"symbol": sym, **{k: (round(v, 3) if isinstance(v, float) else v)
                                         for k, v in st.items()}}
                if strat.style == "intraday":
                    demo_intraday[name] = rec  # aggregate later (demo)
                else:
                    by_strat[name].append(rec)
                    by_pair[f"{sym}|{name}"] = rec
            except Exception:
                continue
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(symbols)} symbols ({time.time()-t0:.0f}s)", flush=True)

    out = {"meta": {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": "bundled real NSE daily 2012-2022 (offline demo)",
        "costs": "full Indian cost model (brokerage+STT+exchange+GST+SEBI+stamp+slippage)",
        "capital": 100_000, "risk_per_trade_pct": get("capital.risk_per_trade_pct"),
    }}
    for name, recs in by_strat.items():
        if not recs:
            continue
        df = pd.DataFrame(recs)
        agg = {}
        for col in ("win_rate", "profit_probability", "expectancy_pct",
                    "total_return_pct", "max_drawdown_pct", "profit_factor",
                    "median_hold_days", "avg_hold_days", "sharpe",
                    "profitable_month_prob", "p_up_3m", "p_up_6m", "p_up_12m"):
            if col in df:
                agg[col] = round(float(df[col].mean(skipna=True)), 2)
        agg["positive_stocks"] = int((df["expectancy_pct"] > 0).sum())
        agg["n_stocks"] = len(df)
        agg["n_trades"] = int(df["trades"].sum())
        out[name] = agg
    out["by_pair"] = {k: v for k, v in by_pair.items() if v["trades"] >= 10}
    # intraday stats are demo-only (synthetic 5m bars from real daily data)
    for name, rec in demo_intraday.items():
        agg = {k: round(float(rec[k]), 2) for k in
               ("win_rate", "expectancy_pct", "max_drawdown_pct",
                "median_hold_days", "profit_factor", "total_return_pct")
               if k in rec}
        agg["demo_data"] = True
        out[name] = agg

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {OUT_PATH}")
    for name in strategies:
        if name in out:
            a = out[name]
            if "n_stocks" in a:
                print(f"  {name:16s} n_stocks={a['n_stocks']:3d} n_trades={a['n_trades']:5d} "
                      f"win%={a['win_rate']:5.2f} exp%={a['expectancy_pct']:+.2f} "
                      f"ret%={a['total_return_pct']:+.1f} dd%={a['max_drawdown_pct']:.1f}")
            else:
                print(f"  {name:16s} DEMO (synthetic 5m) trades={a.get('n_trades', '?')} "
                      f"win%={a.get('win_rate', 0)} exp%={a.get('expectancy_pct', 0):+.2f}")


if __name__ == "__main__":
    main()
