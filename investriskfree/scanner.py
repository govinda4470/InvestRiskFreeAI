"""EOD / live signal scanner.

For every stock in the low-risk universe, run all enabled strategies through the
AI brain and return ranked, capital-protected signals.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from .brain import (
    confidence_score,
    market_breadth,
    market_regime,
    position_size,
    regime_label,
)
from .config import get
from .data.loader import load_daily, load_index_daily
from .data.synthetic import daily_to_intraday
from .data.universe import get_universe
from .ml import load_model_registry, predict_latest
from .strategies import StrategyRegistry

STATS_PATH = os.path.join(get("data.repo_root"), "data", "strategy_stats.json")


def load_stats_registry() -> dict:
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH) as f:
            return json.load(f)
    return {}


def _signal_on_last_bar(strategy, df: pd.DataFrame) -> dict | None:
    sig = strategy.signals(df)
    if "entry" not in sig or float(sig["entry"].iloc[-1]) != 1.0:
        return None
    last = df.iloc[-1]
    return {
        "entry_px_ref": float(last["Close"]),
        "last_price": float(last["Close"]),
        "as_of": last.name.date() if hasattr(last.name, "date") else str(last.name),
        "sl": float(sig["sl"].iloc[-1]),
        "target": float(sig["target"].iloc[-1]),
        "rr": float(sig["rr"].iloc[-1]),
        "reason": str(sig["reason"].iloc[-1]),
        "_signal_frame": sig,
    }


def scan(
    capital: float = 100_000.0,
    styles: tuple[str, ...] = ("swing", "invest", "intraday"),
    use_ml: bool = True,
    real_intraday: bool = False,
    demo_intraday: bool = True,
    progress_cb=None,
    live_data: bool | None = None,
) -> list[dict]:
    # ``real_intraday`` was the original UI's live-data flag. Keep it as a
    # compatibility alias while making daily live-data behavior explicit.
    live_data = real_intraday if live_data is None else bool(live_data)
    universe = get_universe()
    stats_reg = load_stats_registry()
    ml_registry = load_model_registry() if use_ml and get("brain.ml.enabled", True) else {}
    registry = StrategyRegistry.all()
    strategies = {k: v for k, v in registry.items() if v.style in styles and get(f"strategies.{k}.enabled", True)}
    if not strategies:
        return []

    # index + regime + breadth
    try:
        source = "yfinance" if live_data else None
        idx = load_index_daily(source=source)
        closes = pd.DataFrame({
            s["symbol"]: load_daily(s["symbol"], source=source)["Close"]
            for s in universe[:60]
        })
        breadth = market_breadth(closes)
        regime = market_regime(idx)
        regime_ok_last = bool(regime.iloc[-1]) if len(regime) else False
        label, dist, br = regime_label(idx, float(breadth.iloc[-1]) if len(breadth) else 0.5)
    except Exception:
        regime, regime_ok_last, label, br = None, True, "NEUTRAL (no index data)", 0.5
        breadth = None

    signals: list[dict] = []
    n = len(universe)
    for k, s in enumerate(universe):
        sym = s["symbol"]
        if progress_cb:
            progress_cb(k / n, sym)
        try:
            daily = load_daily(sym, source="yfinance" if live_data else None)
            daily.attrs["symbol"] = sym
            for strat_name, strat in strategies.items():
                style = strat.style
                # intraday: prefer real 5m data, else labeled synthetic demo
                if style == "intraday":
                    if real_intraday:
                        try:
                            import yfinance as yf

                            intra = yf.download(f"{sym}.NS", period="2d", interval="5m",
                                                progress=False, auto_adjust=True, threads=False)
                            if intra is None or len(intra) < 80:
                                raise RuntimeError("no intraday data")
                            intra = intra.rename(columns=str.capitalize)
                            intra.columns = [c.split()[0] for c in intra.columns]
                            intra.index = pd.to_datetime(intra.index)
                            intra = intra[["Open", "High", "Low", "Close", "Volume"]]
                            df = intra
                            demo = False
                        except Exception:
                            if not demo_intraday:
                                continue
                            df = daily_to_intraday(daily.tail(30), seed=hash(sym) % 1000)
                            demo = True
                    else:
                        if not demo_intraday:
                            continue
                        df = daily_to_intraday(daily.tail(30), seed=hash(sym) % 1000)
                        demo = True
                else:
                    df = daily
                    demo = False

                sig = _signal_on_last_bar(strat, df)
                if not sig:
                    continue
                # ---- AI brain gating ----
                if regime is not None and not regime_ok_last and label == "RISK-OFF":
                    signals.append(_mk_signal(
                        sym, strat, sig, capital, demo,
                        blocked="RISK-OFF regime: market below 200SMA, longs blocked",
                        label=label, br=br, live=live_data, stats_reg=stats_reg,
                    ))
                    continue

                ml_result = {"available": False, "reason": "ML disabled or model unavailable"}
                if use_ml and style != "intraday" and ml_registry:
                    ml_result = predict_latest(
                        strat_name,
                        daily,
                        sig["_signal_frame"],
                        breadth=breadth,
                        registry=ml_registry,
                    )
                p_win = ml_result.get("p_win") if ml_result.get("available") else None
                conf = confidence_score(
                    daily, br, regime_ok_last, confluence=1, p_win=p_win
                )
                if conf < get("brain.min_confidence", 55):
                    signals.append(_mk_signal(
                        sym, strat, sig, capital, demo,
                        blocked=f"confidence {conf:.0f} < {get('brain.min_confidence', 55)}",
                        label=label, br=br, conf=conf, live=live_data,
                        stats_reg=stats_reg, ml_result=ml_result,
                    ))
                    continue
                if ml_result.get("available") and not ml_result.get("allow"):
                    signals.append(_mk_signal(
                        sym, strat, sig, capital, demo,
                        blocked=(
                            f"ML gate: P(win) {ml_result['p_win']:.1%} below "
                            f"required baseline {ml_result['baseline']:.1%}"
                        ),
                        label=label, br=br, conf=conf, live=live_data,
                        stats_reg=stats_reg, ml_result=ml_result,
                    ))
                    continue
                signals.append(_mk_signal(
                    sym, strat, sig, capital, demo,
                    label=label, br=br, conf=conf, stats_reg=stats_reg,
                    live=live_data, ml_result=ml_result,
                ))
        except Exception:
            continue
    if progress_cb:
        progress_cb(1.0, "")
    signals.sort(key=lambda x: (x["blocked"] is not None, -x.get("confidence", 0)))
    return signals


def _mk_signal(sym, strat, sig, capital, demo, blocked=None, label="", br=0.5,
               conf=None, stats_reg: dict | None = None, live: bool = False,
               ml_result: dict | None = None) -> dict:
    stats_reg = stats_reg or {}
    ml_result = ml_result or {"available": False}
    style = strat.style
    entry_ref = sig["entry_px_ref"]
    sl, target, rr = sig["sl"], sig["target"], sig["rr"]
    sizing = position_size(capital, entry_ref, sl) if blocked is None else None
    key = f"{strat.name}"
    reg = stats_reg.get(key, {})
    conf = conf if conf is not None else 50.0
    level = "STRONG" if conf >= get("brain.strong_confidence", 70) else \
            ("MODERATE" if conf >= get("brain.min_confidence", 55) else "WEAK")

    # ---- current / last price ----
    last_price = sig.get("last_price", entry_ref)
    as_of = sig.get("as_of", "")  # signal candle time; stable for deduplication
    quote_as_of = str(as_of)
    if live:
        # A quote update must not change the signal id/date, otherwise an agent
        # could execute the same daily signal again on every polling cycle.
        from .data.loader import fetch_quote
        q = fetch_quote(sym)
        if q is not None and q > 0:
            last_price = q
            quote_as_of = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d %H:%M %Z")
    gap_from_entry = (last_price / entry_ref - 1) * 100 if entry_ref else 0.0
    dist_to_sl = (last_price / sl - 1) * 100 if sl and sl > 0 else 0.0
    dist_to_target = (target / last_price - 1) * 100 if target and last_price else 0.0
    return {
        "symbol": sym, "style": style, "strategy": strat.name,
        "action": "LONG", "entry_ref": entry_ref, "sl": sl, "target": target,
        "last_price": round(float(last_price), 2), "as_of": str(as_of),
        "quote_as_of": quote_as_of,
        "gap_from_entry_pct": round(float(gap_from_entry), 2),
        "dist_to_sl_pct": round(float(dist_to_sl), 2),
        "dist_to_target_pct": round(float(dist_to_target), 2),
        "rr": float(rr) if pd.notna(rr) else 0.0, "confidence": round(conf, 1),
        "level": level, "blocked": blocked, "reason": sig["reason"],
        "regime": label, "breadth": round(br, 3), "demo_data": demo,
        "profit_prob_pct": reg.get("profit_probability", None),
        "expected_duration_days": reg.get("median_hold_days", None),
        "win_rate_pct": reg.get("win_rate", None),
        "ml_available": bool(ml_result.get("available")),
        "ml_p_win_pct": (
            round(float(ml_result["p_win"]) * 100, 1)
            if ml_result.get("available") else None
        ),
        "ml_baseline_pct": (
            round(float(ml_result["baseline"]) * 100, 1)
            if ml_result.get("available") else None
        ),
        "sizing": sizing,
    }
