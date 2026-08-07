"""Risk-gated paper auto-trading agent.

The agent consumes the same scanner signals shown in the UI, sizes each order
from the current user's *equity and cash*, manages stops/targets, deduplicates
signals, and writes an immutable event log.  It never places a real broker order;
real execution requires a separately reviewed broker adapter and explicit user
consent.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from .brain import suggest_position_from_cash
from .data.loader import fetch_quote, load_daily
from .paper import PaperBroker
from .scanner import scan
from .strategies import StrategyRegistry

DEFAULT_CONFIG = {
    "enabled": False,
    "styles": ["swing", "invest"],
    "strategies": ["swing_trend", "swing_breakout", "invest"],
    "min_confidence": 65.0,
    "risk_pct": 0.5,
    "max_orders_per_cycle": 1,
    "max_daily_loss_pct": 1.5,
    "max_entry_gap_pct": 2.0,
    "require_kronos": False,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def signal_fingerprint(signal: dict) -> str:
    """Stable id for exactly-once handling of a scanner signal."""
    payload = "|".join(
        [
            str(signal.get("symbol", "")),
            str(signal.get("strategy", "")),
            str(signal.get("as_of", "")),
            f"{float(signal.get('entry_ref', 0) or 0):.4f}",
            str(signal.get("action", "LONG")),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AutoTradeAgent:
    def __init__(self, broker: PaperBroker):
        self.broker = broker
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return self.broker._conn()  # same per-user ledger and transaction boundary

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS automation_config (
                    id INTEGER PRIMARY KEY CHECK (id=1),
                    enabled INTEGER NOT NULL DEFAULT 0,
                    styles TEXT NOT NULL,
                    strategies TEXT NOT NULL,
                    min_confidence REAL NOT NULL,
                    risk_pct REAL NOT NULL,
                    max_orders_per_cycle INTEGER NOT NULL,
                    max_daily_loss_pct REAL NOT NULL,
                    max_entry_gap_pct REAL NOT NULL,
                    require_kronos INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS agent_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL, run_id TEXT NOT NULL,
                    level TEXT NOT NULL, action TEXT NOT NULL,
                    symbol TEXT, strategy TEXT, signal_id TEXT,
                    status TEXT NOT NULL, details TEXT
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS processed_signals (
                    signal_id TEXT PRIMARY KEY, first_seen TEXT NOT NULL,
                    executed_at TEXT, status TEXT NOT NULL, details TEXT
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL,
                    finished_at TEXT, status TEXT NOT NULL,
                    signals_seen INTEGER NOT NULL DEFAULT 0,
                    entries INTEGER NOT NULL DEFAULT 0,
                    exits INTEGER NOT NULL DEFAULT 0,
                    message TEXT
                )"""
            )
            row = conn.execute("SELECT id FROM automation_config WHERE id=1").fetchone()
            if not row:
                conn.execute(
                    """INSERT INTO automation_config(
                        id, enabled, styles, strategies, min_confidence,
                        risk_pct, max_orders_per_cycle, max_daily_loss_pct,
                        max_entry_gap_pct, require_kronos, updated_at
                    ) VALUES (1,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        0,
                        json.dumps(DEFAULT_CONFIG["styles"]),
                        json.dumps(DEFAULT_CONFIG["strategies"]),
                        DEFAULT_CONFIG["min_confidence"],
                        DEFAULT_CONFIG["risk_pct"],
                        DEFAULT_CONFIG["max_orders_per_cycle"],
                        DEFAULT_CONFIG["max_daily_loss_pct"],
                        DEFAULT_CONFIG["max_entry_gap_pct"],
                        0,
                        _now(),
                    ),
                )

    def config(self) -> dict:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM automation_config WHERE id=1").fetchone()
        if not row:
            return dict(DEFAULT_CONFIG)
        return {
            "enabled": bool(row["enabled"]),
            "styles": json.loads(row["styles"]),
            "strategies": json.loads(row["strategies"]),
            "min_confidence": float(row["min_confidence"]),
            "risk_pct": float(row["risk_pct"]),
            "max_orders_per_cycle": int(row["max_orders_per_cycle"]),
            "max_daily_loss_pct": float(row["max_daily_loss_pct"]),
            "max_entry_gap_pct": float(row["max_entry_gap_pct"]),
            "require_kronos": bool(row["require_kronos"]),
            "updated_at": row["updated_at"],
        }

    def update_config(self, **values) -> dict:
        cfg = {**self.config(), **values}
        cfg["styles"] = [s for s in cfg["styles"] if s in {"swing", "invest", "intraday"}]
        valid_strategies = set(StrategyRegistry.all())
        cfg["strategies"] = [s for s in cfg["strategies"] if s in valid_strategies]
        cfg["risk_pct"] = min(1.0, max(0.25, float(cfg["risk_pct"])))
        cfg["min_confidence"] = min(100.0, max(55.0, float(cfg["min_confidence"])))
        cfg["max_orders_per_cycle"] = min(5, max(1, int(cfg["max_orders_per_cycle"])))
        cfg["max_daily_loss_pct"] = min(10.0, max(0.25, float(cfg["max_daily_loss_pct"])))
        cfg["max_entry_gap_pct"] = min(10.0, max(0.25, float(cfg["max_entry_gap_pct"])))
        with self._conn() as conn:
            conn.execute(
                """UPDATE automation_config SET enabled=?, styles=?, strategies=?,
                   min_confidence=?, risk_pct=?, max_orders_per_cycle=?,
                   max_daily_loss_pct=?, max_entry_gap_pct=?, require_kronos=?,
                   updated_at=? WHERE id=1""",
                (
                    int(bool(cfg["enabled"])),
                    json.dumps(cfg["styles"]),
                    json.dumps(cfg["strategies"]),
                    cfg["min_confidence"],
                    cfg["risk_pct"],
                    cfg["max_orders_per_cycle"],
                    cfg["max_daily_loss_pct"],
                    cfg["max_entry_gap_pct"],
                    int(bool(cfg.get("require_kronos", False))),
                    _now(),
                ),
            )
        return self.config()

    def _event(
        self,
        run_id: str,
        action: str,
        status: str,
        *,
        level: str = "INFO",
        signal: dict | None = None,
        details: dict | str | None = None,
    ) -> None:
        signal = signal or {}
        serialized = details if isinstance(details, str) else json.dumps(details or {}, default=str)
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO agent_events(
                    date, run_id, level, action, symbol, strategy,
                    signal_id, status, details
                ) VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    _now(),
                    run_id,
                    level,
                    action,
                    signal.get("symbol"),
                    signal.get("strategy"),
                    signal.get("signal_id"),
                    status,
                    serialized,
                ),
            )

    def events(self, limit: int = 500) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_events ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            try:
                item["details"] = json.loads(item["details"] or "{}")
            except json.JSONDecodeError:
                pass
            output.append(item)
        return output

    def runs(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
            return [dict(row) for row in rows]

    def _processed(self, signal_id: str) -> bool:
        with self._conn() as conn:
            return bool(
                conn.execute(
                    "SELECT 1 FROM processed_signals WHERE signal_id=?", (signal_id,)
                ).fetchone()
            )

    def _claim_signal(self, signal_id: str) -> bool:
        """Atomically claim a signal so concurrent workers cannot both buy it."""
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO processed_signals(
                    signal_id, first_seen, status, details
                ) VALUES (?,?,?,?)""",
                (signal_id, _now(), "PROCESSING", "{}"),
            )
            return cursor.rowcount == 1

    def _mark_processed(self, signal_id: str, status: str, details: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO processed_signals(
                    signal_id, first_seen, executed_at, status, details
                ) VALUES (?, COALESCE((SELECT first_seen FROM processed_signals
                                      WHERE signal_id=?), ?), ?, ?, ?)""",
                (
                    signal_id,
                    signal_id,
                    _now(),
                    _now() if status == "EXECUTED" else None,
                    status,
                    json.dumps(details, default=str),
                ),
            )

    @staticmethod
    def _latest_price(symbol: str, live_data: bool, fallback: float | None = None) -> float | None:
        if live_data:
            quote = fetch_quote(symbol)
            if quote and quote > 0:
                return float(quote)
        try:
            source = "yfinance" if live_data else None
            daily = load_daily(symbol, source=source)
            return float(daily["Close"].iloc[-1])
        except Exception:
            return float(fallback) if fallback and fallback > 0 else None

    def _manage_exits(self, run_id: str, live_data: bool) -> tuple[int, dict[str, float]]:
        exits = 0
        quotes: dict[str, float] = {}
        now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
        for position in self.broker.positions():
            price = self._latest_price(
                position["symbol"], live_data, position.get("last_price") or position["entry_price"]
            )
            if price is None:
                self._event(
                    run_id,
                    "EXIT_CHECK",
                    "NO_QUOTE",
                    level="WARNING",
                    signal=position,
                    details="No price available; position left unchanged",
                )
                continue
            quotes[position["symbol"]] = price
            reason = None
            if position.get("sl") and price <= float(position["sl"]):
                reason = "auto stop-loss"
            elif position.get("target") and price >= float(position["target"]):
                reason = "auto target"
            elif position["style"] == "intraday":
                entry_time = pd.Timestamp(position["entry_date"])
                if entry_time.tzinfo is None:
                    entry_time = entry_time.tz_localize("UTC")
                entry_date_ist = entry_time.tz_convert("Asia/Kolkata").date()
                if entry_date_ist < now_ist.date():
                    reason = "auto emergency square-off (overnight intraday position)"
                elif now_ist.weekday() < 5 and (now_ist.hour, now_ist.minute) >= (15, 25):
                    reason = "auto square-off 15:25"
            if reason is None:
                try:
                    strategy = StrategyRegistry.get(position["strategy"])
                    max_hold = int(strategy.params.get("max_hold_days", 20))
                    age_days = max(0, (pd.Timestamp.now(tz="UTC") - pd.Timestamp(position["entry_date"])).days)
                    if position["style"] != "intraday" and age_days >= max_hold:
                        reason = f"auto max hold ({max_hold} days)"
                except Exception:
                    pass
            if reason:
                result = self.broker.sell(position["id"], price, reason, source="auto")
                status = "EXECUTED" if result.get("ok") else "REJECTED"
                self._event(
                    run_id,
                    "SELL",
                    status,
                    level="INFO" if result.get("ok") else "ERROR",
                    signal={**position, "signal_id": position.get("signal_id")},
                    details={"price": price, "reason": reason, **result},
                )
                exits += int(bool(result.get("ok")))
        return exits, quotes

    def run_once(
        self,
        *,
        live_data: bool = False,
        force: bool = False,
        kronos_gate=None,
    ) -> dict:
        """Run one idempotent monitor/scan/execute cycle.

        ``force`` permits a user-initiated dry run while the agent is disarmed;
        it does not bypass any capital or risk limit.  A disarmed forced cycle
        manages exits and evaluates signals but never opens a new position.
        """
        cfg = self.config()
        run_id = uuid.uuid4().hex
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO agent_runs(run_id, started_at, status) VALUES (?,?,?)",
                (run_id, _now(), "RUNNING"),
            )
        account = self.broker.account()
        if not account:
            message = "No paper account"
            self._finish_run(run_id, "BLOCKED", 0, 0, 0, message)
            return {"ok": False, "run_id": run_id, "message": message, "entries": 0, "exits": 0}
        if not cfg["enabled"] and not force:
            message = "Agent is disarmed"
            self._finish_run(run_id, "DISARMED", 0, 0, 0, message)
            return {"ok": False, "run_id": run_id, "message": message, "entries": 0, "exits": 0}

        exits, quotes = self._manage_exits(run_id, live_data)
        summary = self.broker.summary(quotes)
        capital = max(float(summary["capital"]), 1.0)
        today = datetime.now(timezone.utc).date().isoformat()
        daily_pnl = self.broker.realized_pnl_since(today)
        daily_floor = -capital * cfg["max_daily_loss_pct"] / 100
        if daily_pnl <= daily_floor:
            message = (
                f"Daily loss guard: ₹{daily_pnl:,.0f} <= "
                f"-₹{abs(daily_floor):,.0f}; no new entries"
            )
            self._event(run_id, "RISK_GUARD", "BLOCKED", level="WARNING", details=message)
            self.broker.record_snapshot(quotes, "agent cycle - daily loss guard")
            self._finish_run(run_id, "COMPLETED", 0, 0, exits, message)
            return {"ok": True, "run_id": run_id, "message": message, "entries": 0, "exits": exits}

        try:
            signals = scan(
                capital=float(summary["equity"]),
                styles=tuple(cfg["styles"]),
                real_intraday=live_data,
                live_data=live_data,
                demo_intraday=not live_data,
            )
        except Exception as exc:
            message = f"Scanner failed: {exc}"
            self._event(run_id, "SCAN", "ERROR", level="ERROR", details=message)
            self._finish_run(run_id, "ERROR", 0, 0, exits, message)
            return {"ok": False, "run_id": run_id, "message": message, "entries": 0, "exits": exits}

        actionable = [
            signal
            for signal in signals
            if not signal.get("blocked")
            and signal.get("strategy") in cfg["strategies"]
            and float(signal.get("confidence", 0)) >= cfg["min_confidence"]
        ]
        actionable.sort(
            key=lambda signal: (
                -float(signal.get("confidence", 0)),
                -float(signal.get("ml_p_win_pct") or signal.get("profit_prob_pct") or 0),
            )
        )
        entries = 0
        can_enter = bool(cfg["enabled"])
        for signal in actionable:
            if entries >= cfg["max_orders_per_cycle"]:
                break
            signal = dict(signal)
            signal_id = signal_fingerprint(signal)
            signal["signal_id"] = signal_id
            if can_enter:
                if not self._claim_signal(signal_id):
                    continue
            elif self._processed(signal_id):
                continue
            if any(position["symbol"] == signal["symbol"] for position in self.broker.positions()):
                if can_enter:
                    self._mark_processed(signal_id, "SKIPPED", {"reason": "symbol already open"})
                self._event(
                    run_id, "BUY", "SKIPPED", signal=signal, details="Symbol already open"
                )
                continue
            price = float(signal.get("last_price") or signal.get("entry_ref") or 0)
            stop = float(signal.get("sl") or 0)
            target = float(signal.get("target") or 0)
            gap = abs(float(signal.get("gap_from_entry_pct") or 0))
            if price <= stop or (target > 0 and price >= target):
                if can_enter:
                    self._mark_processed(signal_id, "SKIPPED", {"reason": "levels already crossed"})
                self._event(
                    run_id, "BUY", "SKIPPED", signal=signal, details="Stop/target already crossed"
                )
                continue
            if live_data and gap > cfg["max_entry_gap_pct"]:
                if can_enter:
                    self._mark_processed(
                        signal_id, "SKIPPED", {"reason": f"entry gap {gap:.2f}% too large"}
                    )
                self._event(
                    run_id,
                    "BUY",
                    "SKIPPED",
                    signal=signal,
                    details=f"Entry gap {gap:.2f}% exceeds limit",
                )
                continue
            if cfg["require_kronos"]:
                if kronos_gate is None:
                    details = "Kronos gate required but no model service is available"
                    if can_enter:
                        self._mark_processed(signal_id, "SKIPPED", {"reason": details})
                    self._event(
                        run_id,
                        "KRONOS_GATE",
                        "BLOCKED",
                        level="WARNING",
                        signal=signal,
                        details=details,
                    )
                    continue
                try:
                    verdict = kronos_gate(signal)
                except Exception as exc:
                    verdict = {"allow": False, "reason": f"Kronos inference failed: {exc}"}
                    if can_enter:
                        self._mark_processed(signal_id, "SKIPPED", {"kronos": verdict})
                    self._event(
                        run_id, "KRONOS_GATE", "ERROR", level="ERROR",
                        signal=signal, details=verdict,
                    )
                    continue
                if not verdict.get("allow", False):
                    if can_enter:
                        self._mark_processed(signal_id, "SKIPPED", {"kronos": verdict})
                    self._event(
                        run_id, "KRONOS_GATE", "BLOCKED", signal=signal, details=verdict
                    )
                    continue

            current = self.broker.summary(quotes)
            suggestion = suggest_position_from_cash(
                float(current["cash"]),
                float(current["equity"]),
                price,
                stop,
                signal["style"],
                cfg["risk_pct"] / 100,
            )
            if suggestion.get("blocked"):
                if can_enter:
                    self._mark_processed(signal_id, "SKIPPED", {"reason": suggestion["blocked"]})
                self._event(
                    run_id, "BUY", "BLOCKED", signal=signal, details=suggestion["blocked"]
                )
                continue
            if not can_enter:
                self._event(
                    run_id,
                    "BUY",
                    "DRY_RUN",
                    signal=signal,
                    details={"suggested_qty": suggestion["qty"], "price": price},
                )
                continue
            result = self.broker.buy(
                signal["symbol"],
                signal["style"],
                signal["strategy"],
                suggestion["qty"],
                price,
                stop,
                target,
                signal.get("reason", "AI signal"),
                signal.get("confidence"),
                signal.get("ml_p_win_pct") or signal.get("profit_prob_pct"),
                source="auto",
                signal_id=signal_id,
                risk_pct=cfg["risk_pct"] / 100,
            )
            if result.get("ok"):
                entries += 1
                self._mark_processed(signal_id, "EXECUTED", result)
                self._event(run_id, "BUY", "EXECUTED", signal=signal, details=result)
            else:
                self._mark_processed(signal_id, "REJECTED", result)
                self._event(
                    run_id, "BUY", "REJECTED", level="WARNING", signal=signal, details=result
                )

        self.broker.record_snapshot(quotes, "agent cycle")
        message = (
            f"Checked {len(signals)} signals; {len(actionable)} passed gates; "
            f"opened {entries}, closed {exits}"
        )
        self._finish_run(run_id, "COMPLETED", len(signals), entries, exits, message)
        return {
            "ok": True,
            "run_id": run_id,
            "message": message,
            "signals_seen": len(signals),
            "actionable": len(actionable),
            "entries": entries,
            "exits": exits,
            "paper_only": True,
        }

    def _finish_run(
        self,
        run_id: str,
        status: str,
        signals_seen: int,
        entries: int,
        exits: int,
        message: str,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE agent_runs SET finished_at=?, status=?, signals_seen=?,
                   entries=?, exits=?, message=? WHERE run_id=?""",
                (_now(), status, signals_seen, entries, exits, message, run_id),
            )
