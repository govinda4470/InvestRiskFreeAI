#!/usr/bin/env python3
"""Train and evaluate the walk-forward ML win-probability gate.

For each enabled strategy, collects every backtest entry with its realised
outcome (win/loss), builds features available at entry time, trains a logistic
regression on past years, and evaluates strictly OUT-OF-SAMPLE on future years.
Writes data/ml_report.json shown in the Strategy Research page.

Usage:
    python tools/train_ml.py [--symbols RELIANCE,HDFCBANK] [--splits 3]

Honest expectations: the OOS accuracy will be only slightly above the base rate.
The gate's job is NOT prediction magic - it is to filter out the worst decile of
signals. If the model cannot beat the base rate, the gate stays neutral.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from investriskfree.backtest import Backtester  # noqa: E402
from investriskfree.brain import market_regime  # noqa: E402
from investriskfree.config import get  # noqa: E402
from investriskfree.data.loader import list_bundled_symbols, load_daily  # noqa: E402
from investriskfree.ml import FEATURE_COLS, WinProbModel, build_features  # noqa: E402
from investriskfree.strategies import StrategyRegistry  # noqa: E402

OUT_PATH = os.path.join(get("data.repo_root"), "data", "ml_report.json")
MODEL_PATH = os.path.join(get("data.repo_root"), "data", "ml_models.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", help="comma-separated subset")
    ap.add_argument("--splits", type=int, default=3)
    args = ap.parse_args()
    symbols = args.symbols.split(",") if args.symbols else list_bundled_symbols()

    closes = {}
    for s in symbols[:40]:
        try:
            closes[s] = load_daily(s)["Close"]
        except Exception:
            continue
    proxy = pd.DataFrame(closes).mean(axis=1).to_frame("Close")
    proxy["Open"] = proxy["High"] = proxy["Low"] = proxy["Close"]
    proxy["Volume"] = 1
    regime = market_regime(proxy)
    regs = {s: regime.reindex(load_daily(s).index).ffill().fillna(False).to_numpy()
            for s in closes}

    built_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    report = {"meta": {
        "built_at": built_at,
        "symbols": len(symbols),
        "splits": args.splits,
        "method": "expanding train / non-overlapping future test windows",
        "label_alignment": "signal-close features mapped to next-open trade outcome",
    }}
    model_registry = {"meta": {"built_at": built_at, "feature_cols": FEATURE_COLS}}
    for name, strat in StrategyRegistry.all().items():
        if strat.style == "intraday" or not get(f"strategies.{name}.enabled", False):
            continue
        feats_all, labels_all = [], []
        for sym in closes:
            try:
                df = load_daily(sym)
                res = Backtester().run(df, strat, regime_ok=regs[sym], capital=100_000)
                sig = strat.signals(df)
                fe = build_features(df, sig, breadth=None)
                # Backtester executes a close-generated signal at the next open.
                # Associate the outcome with the PREVIOUS (signal) bar so the
                # feature row cannot see the entry day's close/high/low.
                outcomes = {}
                for trade in res.trades:
                    location = df.index.get_indexer([pd.Timestamp(trade.entry_date)])[0]
                    if location > 0:
                        outcomes[pd.Timestamp(df.index[location - 1])] = int(trade.net_pnl > 0)
                lab = pd.Series(outcomes, name="label", dtype=float)
                fe = fe.loc[fe.index.isin(lab.index)]
                lab = lab.reindex(fe.index)
                # unique (symbol, date) index so duplicate entry dates across
                # stocks never misalign the walk-forward split
                fe.index = pd.MultiIndex.from_arrays(
                    [[sym] * len(fe), fe.index], names=["symbol", "date"])
                lab.index = fe.index
                feats_all.append(fe)
                labels_all.append(lab)
            except Exception:
                continue
        if not feats_all:
            continue
        fe = pd.concat(feats_all)
        lab = pd.concat(labels_all)
        model = WinProbModel()
        rep = model.walk_forward(fe, lab, n_splits=args.splits,
                                 min_train_years=get("brain.ml.min_train_years", 3))
        rep["n_entries"] = int(len(fe))
        report[name] = rep
        if rep.get("trained"):
            baseline = float(rep["baseline_win_rate"]) / 100
            model_registry[name] = model.to_dict(baseline=baseline)
            model_registry[name]["oos_report"] = rep
            print(f"{name:18s} OOS n={rep['n_oos']} base={rep['baseline_win_rate']:.1f}% "
                  f"acc={rep['accuracy']:.1f}% prec={rep['precision_when_win']:.1f}% "
                  f"lift={rep['lift']:.2f}x")
        else:
            print(f"{name:18s} not trained: {rep.get('reason', '?')}")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    with open(MODEL_PATH, "w") as f:
        json.dump(model_registry, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    print(f"Wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()
