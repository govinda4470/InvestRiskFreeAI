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

from investriskfree.auth import AuthStore
from investriskfree.autotrade import AutoTradeAgent
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


def test_paper_topup(tmp_path):
    db = str(tmp_path / "paper2.db")
    b = PaperBroker(db_path=db)
    b.create_account(50_000)
    r = b.topup(25_000)
    assert r["ok"] and r["new_capital"] == 75_000
    s = b.summary()
    assert s["capital"] == 75_000 and s["cash"] == 75_000
    evs = b.capital_events()
    assert len(evs) == 2  # INITIAL + TOPUP
    assert evs[0]["type"] == "DEPOSIT" and evs[0]["amount"] == 25_000


def test_suggest_position_from_cash():
    from investriskfree.brain import suggest_position_from_cash
    # Slippage is included, so quantity stays just below the nominal 100 shares.
    sugg = suggest_position_from_cash(100_000, 100_000, 100, 95)
    assert sugg["qty"] == 99 and not sugg["blocked"]
    assert sugg["pos_value"] <= 100_000
    # cash too small to afford even 1 share
    sugg2 = suggest_position_from_cash(50, 100_000, 100, 95)
    assert sugg2["qty"] == 0 and sugg2["blocked"]
    # clamps down when cash < suggested position value
    sugg3 = suggest_position_from_cash(8_000, 100_000, 100, 95)
    assert sugg3["qty"] <= 80  # 8000/100 = 80 max affordable
    assert sugg3["pos_value"] + sugg3["cost"] <= 8_000


def test_signup_and_hashed_authentication(tmp_path):
    auth = AuthStore(str(tmp_path / "auth.db"))
    result = auth.register("new_trader", "trader@example.com", "SafePassword42", "New Trader")
    assert result["ok"]
    assert auth.authenticate("NEW_TRADER", "SafePassword42")["ok"]
    assert not auth.authenticate("new_trader", "wrong-password")["ok"]
    # Password material is never stored in plaintext.
    import sqlite3
    with sqlite3.connect(str(tmp_path / "auth.db")) as conn:
        password_hash, salt = conn.execute(
            "SELECT password_hash, salt FROM users WHERE username='new_trader'"
        ).fetchone()
    assert "SafePassword42" not in password_hash
    assert len(password_hash) == 64 and len(salt) == 48
    duplicate = auth.register("new_trader", "other@example.com", "OtherPassword42")
    assert not duplicate["ok"]


def test_user_capital_withdrawal_and_equity_snapshots(tmp_path):
    broker = PaperBroker(db_path=str(tmp_path / "capital.db"))
    broker.create_account(100_000)
    assert broker.adjust_capital(20_000)["ok"]
    assert broker.adjust_capital(-10_000)["ok"]
    assert broker.summary()["capital"] == 110_000
    assert not broker.adjust_capital(-200_000)["ok"]
    curve = broker.equity_curve()
    assert len(curve) == 3
    assert curve.iloc[-1]["net_deposits"] == 110_000


def test_auto_agent_is_risk_sized_and_idempotent(tmp_path, monkeypatch):
    broker = PaperBroker(db_path=str(tmp_path / "agent.db"))
    broker.create_account(100_000)
    agent = AutoTradeAgent(broker)
    signal = {
        "symbol": "ITC", "style": "swing", "strategy": "swing_trend",
        "entry_ref": 100.0, "last_price": 100.0, "sl": 95.0, "target": 110.0,
        "rr": 2.0, "confidence": 80.0, "blocked": None,
        "reason": "test signal", "as_of": "2026-08-07",
        "gap_from_entry_pct": 0.0, "profit_prob_pct": 55.0,
    }
    monkeypatch.setattr("investriskfree.autotrade.scan", lambda **_: [signal])
    monkeypatch.setattr(AutoTradeAgent, "_latest_price", staticmethod(lambda *_, **__: 100.0))
    preview = agent.run_once(force=True)
    assert preview["ok"] and preview["entries"] == 0
    assert not broker.positions()
    agent.update_config(enabled=True, min_confidence=65, risk_pct=0.5)
    first = agent.run_once()
    assert first["ok"] and first["entries"] == 1
    positions = broker.positions()
    assert len(positions) == 1
    # Expected-fill slippage is included, keeping actual risk <= ₹500.
    assert positions[0]["qty"] == 99
    second = agent.run_once()
    assert second["entries"] == 0
    assert len(broker.positions()) == 1


def test_kronos_input_adapter_without_optional_runtime(daily):
    from investriskfree.kronos_forecast import future_business_timestamps, prepare_kline_frame
    prepared = prepare_kline_frame(daily, lookback=128)
    assert list(prepared.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert len(prepared) == 128
    future = future_business_timestamps(prepared.index[-1], 5)
    assert len(future) == 5 and (future.dayofweek < 5).all()


def test_ml_walk_forward_windows_do_not_repeat_rows():
    from investriskfree.ml import FEATURE_COLS, WinProbModel
    rng = np.random.default_rng(7)
    dates = pd.date_range("2012-01-01", "2022-12-31", periods=500)
    features = pd.DataFrame(rng.normal(size=(500, len(FEATURE_COLS))),
                            index=dates, columns=FEATURE_COLS)
    labels = pd.Series((features["rsi14"] + rng.normal(size=500) > 0).astype(int), index=dates)
    model = WinProbModel(epochs=30)
    report = model.walk_forward(features, labels, n_splits=3, min_train_years=3)
    assert report["trained"]
    assert report["n_oos"] <= len(features)
    assert sum(fold["test_rows"] for fold in report["folds"]) == report["n_oos"]
