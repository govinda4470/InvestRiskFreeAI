#!/usr/bin/env python3
"""Smoke tests for InvestRiskFreeAI. Run: python -m pytest tests/ -q"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from investriskfree.backtest import Backtester, CostModel
from investriskfree.brain import market_regime, position_size
from investriskfree.config import get
from investriskfree.data.loader import list_bundled_symbols, load_daily
from investriskfree.data.synthetic import daily_to_intraday
from investriskfree.data.universe import get_universe
from investriskfree.paper import PaperBroker
from investriskfree.strategies import StrategyRegistry


@pytest.fixture(scope="module")
def daily():
    return load_daily("RELIANCE")


def test_data_clean(daily):
    assert len(daily) > 2000
    assert (daily.index.dayofweek < 5).all(), "weekend bars present"
    rets = daily["Close"].pct_change().abs()
    assert (rets.dropna() < 0.25).all(), "impossible moves present"


def test_universe():
    uni = get_universe()
    assert len(uni) >= 30, f"universe too small: {len(uni)}"
    for u in uni:
        assert u["avg_turnover_cr"] >= get("universe.min_avg_turnover_cr", 5)


def test_all_strategies_signal_shape(daily):
    for name, strat in StrategyRegistry.all().items():
        sig = strat.signals(daily)
        for col in ("entry", "exit", "sl", "target", "reason", "rr"):
            assert col in sig.columns, f"{name} missing {col}"
        assert sig["entry"].isin([0, 1]).all()


def test_backtest_runs_all(daily):
    closes = pd.DataFrame({s: load_daily(s)["Close"] for s in list_bundled_symbols()[:20]})
    proxy = closes.mean(axis=1).to_frame("Close")
    proxy["Open"] = proxy["High"] = proxy["Low"] = proxy["Close"]
    proxy["Volume"] = 1
    regime = market_regime(proxy).reindex(daily.index).ffill().fillna(False).to_numpy()
    for name, strat in StrategyRegistry.all().items():
        if strat.style == "intraday":
            continue
        res = Backtester().run(daily, strat, regime_ok=regime, capital=100_000)
        s = res.stats
        assert "win_rate" in s and "expectancy_pct" in s
        assert res.equity is not None and len(res.equity) == len(daily)
        if res.trades:
            t = res.trades[0]
            assert t.net_pnl_pct is not None


def test_intraday_synthetic():
    daily = load_daily("RELIANCE").tail(300)
    intra = daily_to_intraday(daily, seed=42)
    assert len(intra) >= 300 * 70
    assert intra.index.tz is not None
    strat = StrategyRegistry.get("intraday_orb")
    res = Backtester().run(intra, strat, capital=100_000)
    assert res.stats["trades"] > 10
    # every intraday trade must be squared off same day: hold <= 1 day
    for t in res.trades:
        assert (t.exit_date - t.entry_date).days <= 1


def test_position_sizing():
    # 0.5% risk budget on 100k = ₹500; risk per share 5 => 100 shares
    s = position_size(100_000, entry=100, sl=95)
    assert s["qty"] == 100
    assert s["risk_rs"] == 500
    # max position cap respected
    s_big = position_size(10_000_000, entry=100, sl=99)
    assert s_big["pos_value"] <= 10_000_000 * 0.25
    # invalid levels blocked
    s2 = position_size(100_000, entry=100, sl=105)
    assert s2["qty"] == 0 and s2["blocked"]
    # tiny capital -> position too small to clear costs -> blocked
    s3 = position_size(1_000, entry=100, sl=99)
    assert s3["blocked"] and "below min" in s3["blocked"]


def test_cost_model_sanity():
    cm = CostModel()
    buy = cm.buy_charges(10, 1000, "swing")
    sell = cm.sell_charges(10, 1050, "swing")
    assert buy > 0 and sell > 0
    # intraday STT lower than delivery
    buy_i = cm.buy_charges(10, 1000, "intraday")
    assert buy_i < buy


def test_paper_broker(tmp_path):
    db = str(tmp_path / "paper.db")
    b = PaperBroker(db_path=db)
    b.create_account(50_000)
    res = b.buy("ITC", "swing", "swing_trend", 10, 300, 290, 330, "test", 80)
    assert res["ok"]
    assert b.summary()["n_positions"] == 1
    res2 = b.sell(1, 320, "test exit")
    assert res2["ok"]
    s = b.summary()
    assert s["realized_pnl"] > 0
    assert s["equity"] > 50_000


def test_scanner():
    from investriskfree.scanner import scan
    sigs = scan(capital=100_000, styles=("swing", "invest"))
    for s in sigs:
        assert "symbol" in s and "confidence" in s and "blocked" in s
    # blocked signals carry a reason
    for s in sigs:
        if s["blocked"]:
            assert isinstance(s["blocked"], str) and len(s["blocked"]) > 3
