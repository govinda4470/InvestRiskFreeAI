#!/usr/bin/env python3
"""InvestRiskFreeAI command-line interface.

Examples:
    python -m investriskfree scan                       # today's AI signals
    python -m investriskfree backtest RELIANCE swing_trend
    python -m investriskfree backtest RELIANCE --all    # all strategies
    python -m investriskfree stats                      # strategy registry
    python -m investriskfree paper new 100000           # create paper account
    python -m investriskfree paper summary
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .backtest import Backtester
from .brain import market_regime
from .config import get
from .data.loader import list_bundled_symbols, load_daily
from .data.universe import get_universe
from .strategies import StrategyRegistry


def _regime_array(symbol: str):
    """Regime proxy from the equal-weight universe index (offline fallback)."""
    from .data.loader import load_daily as ld
    closes = pd.DataFrame()
    for s in list_bundled_symbols()[:40]:
        try:
            closes[s] = ld(s)["Close"]
        except Exception:
            continue
    proxy = closes.mean(axis=1).to_frame("Close")
    proxy["Open"] = proxy["High"] = proxy["Low"] = proxy["Close"]
    proxy["Volume"] = 1
    regime = market_regime(proxy)
    return regime.reindex(load_daily(symbol).index).ffill().fillna(False).to_numpy()


def cmd_backtest(args) -> int:
    df = load_daily(args.symbol)
    df.attrs["symbol"] = args.symbol
    reg = _regime_array(args.symbol)
    if args.all:
        names = list(StrategyRegistry.all())
    else:
        names = args.strategy.split(",")
    for name in names:
        strat = StrategyRegistry.get(name)
        if args.style and strat.style != args.style:
            continue
        res = Backtester().run(df, strat, regime_ok=reg, capital=args.capital)
        s = res.stats
        print(f"\n=== {args.symbol} | {name} ({strat.style}) | "
              f"{df.index[0].date()} -> {df.index[-1].date()} ===")
        for k, v in s.items():
            print(f"  {k:24s} {v if not isinstance(v, float) else round(v, 3)}")
        if args.trades:
            frame = res.to_frame()
            print(f"\n  TRADES ({len(frame)}):")
            cols = ["entry_date", "exit_date", "entry_price", "exit_price",
                    "net_pnl_pct", "hold_days", "reason"]
            if len(frame):
                print(frame[cols].to_string(index=False)[:4000])
    return 0


def cmd_scan(args) -> int:
    from .scanner import scan
    signals = scan(capital=args.capital, styles=tuple(args.styles.split(",")))
    print(f"{'SYM':12s} {'STYLE':9s} {'STRATEGY':18s} {'CONF':>5s} {'LVL':8s} "
          f"{'ENTRY':>8s} {'SL':>8s} {'TGT':>8s} {'RR':>4s} {'P(win)':>7s} STATUS")
    for s in signals:
        pwin = f"{s['profit_prob_pct']:.0f}%" if s.get("profit_prob_pct") is not None else "n/a"
        status = s["blocked"] or "ACTION"
        print(f"{s['symbol']:12s} {s['style']:9s} {s['strategy']:18s} "
              f"{s['confidence']:5.1f} {s['level']:8s} {s['entry_ref']:8.2f} "
              f"{s['sl']:8.2f} {s['target']:8.2f} {s['rr']:4.2f} {pwin:7s} {status}")
    blocked = sum(1 for s in signals if s["blocked"])
    print(f"\n{len(signals) - blocked} actionable signals, {blocked} blocked by risk rules")
    return 0


def cmd_stats(args) -> int:
    from .scanner import load_stats_registry
    reg = load_stats_registry()
    print("Strategy performance registry (built from real NSE daily backtests)\n")
    for name, d in reg.items():
        if name in ("meta", "by_pair"):
            continue
        if isinstance(d, dict):
            print(f"{name:18s} {json.dumps(d)}")
    return 0


def cmd_paper(args) -> int:
    from .paper import PaperBroker
    broker = PaperBroker()
    if args.action == "new":
        broker.create_account(args.capital)
        print(f"Paper account created with virtual capital ₹{args.capital:,.0f}")
    elif args.action == "summary":
        s = broker.summary()
        if not s:
            print("No account. Run: python -m investriskfree paper new 100000")
            return 1
        for k, v in s.items():
            print(f"  {k:18s} {v:,.2f}" if isinstance(v, float) else f"  {k:18s} {v}")
    elif args.action == "positions":
        for p in broker.positions():
            print(f"  #{p['id']} {p['symbol']} {p['strategy']} qty={p['qty']} "
                  f"entry={p['entry_price']:.2f} sl={p['sl']} tgt={p['target']}")
    elif args.action == "trades":
        for t in broker.trades(limit=20):
            print(f"  {t['symbol']:10s} {t['entry_date'][:16]} -> {t['exit_date'][:16]} "
                  f"pnl={t['net_pnl']:+.2f} ({t['net_pnl_pct']:+.2f}%) {t['reason']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="investriskfree", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backtest", help="backtest a stock+strategy")
    b.add_argument("symbol")
    b.add_argument("strategy", nargs="?", default="swing_trend")
    b.add_argument("--all", action="store_true")
    b.add_argument("--style", choices=["swing", "intraday", "invest"])
    b.add_argument("--capital", type=float, default=100_000)
    b.add_argument("--trades", action="store_true", help="print every trade")
    b.set_defaults(fn=cmd_backtest)

    s = sub.add_parser("scan", help="scan universe for today's signals")
    s.add_argument("--capital", type=float, default=100_000)
    s.add_argument("--styles", default="swing,invest,intraday")
    s.set_defaults(fn=cmd_scan)

    st = sub.add_parser("stats", help="show strategy registry")
    st.set_defaults(fn=cmd_stats)

    p = sub.add_parser("paper", help="paper trading account")
    pa = p.add_subparsers(dest="action", required=True)
    pa.add_parser("new").add_argument("capital", type=float)
    pa.add_parser("summary")
    pa.add_parser("positions")
    pa.add_parser("trades")
    p.set_defaults(fn=cmd_paper)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
