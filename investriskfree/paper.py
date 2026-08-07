"""Persistent, user-isolated paper-trading broker.

The broker models slippage and Indian equity costs, enforces the hard portfolio
limits, keeps a complete audit trail, and records mark-to-market snapshots for
performance charts.  It intentionally executes virtual orders only.
"""
from __future__ import annotations

import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import CostModel
from .config import get


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PaperBroker:
    def __init__(self, db_path: str | None = None, user_id: str | None = None):
        if db_path and user_id:
            raise ValueError("pass db_path or user_id, not both")
        if user_id:
            from .auth import user_paper_db_path

            db_path = user_paper_db_path(user_id)
        self.db_path = db_path or os.path.join(get("data.repo_root"), "data", "paper.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.costs = CostModel()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        return conn

    @staticmethod
    def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}

    @classmethod
    def _add_columns(cls, conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
        existing = cls._columns(conn, table)
        for name, definition in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY, name TEXT, capital REAL, cash REAL,
                    created_at TEXT, updated_at TEXT)"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, account_id INTEGER,
                    symbol TEXT, style TEXT, strategy TEXT, qty INTEGER,
                    entry_price REAL, entry_date TEXT, sl REAL, target REAL,
                    reason TEXT, confidence REAL, p_win REAL, last_price REAL,
                    entry_cost REAL NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT 'manual', signal_id TEXT,
                    FOREIGN KEY(account_id) REFERENCES account(id))"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, account_id INTEGER,
                    symbol TEXT, style TEXT, strategy TEXT, qty INTEGER,
                    entry_price REAL, exit_price REAL, entry_date TEXT,
                    exit_date TEXT, gross_pnl REAL, costs REAL, net_pnl REAL,
                    net_pnl_pct REAL, reason TEXT, hold_days INTEGER,
                    entry_cost REAL NOT NULL DEFAULT 0,
                    exit_cost REAL NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT 'manual', signal_id TEXT,
                    confidence REAL, p_win REAL,
                    FOREIGN KEY(account_id) REFERENCES account(id))"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS equity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, equity REAL,
                    cash REAL, position_value REAL, net_deposits REAL,
                    realized_pnl REAL, unrealized_pnl REAL, reason TEXT)"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS capital_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT,
                    type TEXT, amount REAL, note TEXT)"""
            )
            # Migrate ledgers created by v0.1 without destroying user history.
            self._add_columns(
                conn,
                "positions",
                {
                    "entry_cost": "REAL NOT NULL DEFAULT 0",
                    "source": "TEXT NOT NULL DEFAULT 'manual'",
                    "signal_id": "TEXT",
                },
            )
            self._add_columns(
                conn,
                "trades",
                {
                    "entry_cost": "REAL NOT NULL DEFAULT 0",
                    "exit_cost": "REAL NOT NULL DEFAULT 0",
                    "source": "TEXT NOT NULL DEFAULT 'manual'",
                    "signal_id": "TEXT",
                    "confidence": "REAL",
                    "p_win": "REAL",
                },
            )
            self._add_columns(
                conn,
                "equity_snapshots",
                {
                    "cash": "REAL",
                    "position_value": "REAL",
                    "net_deposits": "REAL",
                    "realized_pnl": "REAL",
                    "unrealized_pnl": "REAL",
                    "reason": "TEXT",
                },
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol, strategy)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_exit ON trades(exit_date)")

    # ---------------- account ----------------
    def create_account(self, capital: float, name: str = "Paper Trader") -> None:
        capital = float(capital)
        if not math.isfinite(capital) or capital <= 0:
            raise ValueError("capital must be positive")
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM positions")
            conn.execute("DELETE FROM trades")
            conn.execute("DELETE FROM equity_snapshots")
            conn.execute("DELETE FROM capital_events")
            conn.execute("DELETE FROM account")
            cur = conn.execute(
                "INSERT INTO account(name, capital, cash, created_at, updated_at) VALUES (?,?,?,?,?)",
                (name[:80], capital, capital, _now(), _now()),
            )
            conn.execute(
                "INSERT INTO capital_events(date, type, amount, note) VALUES (?,?,?,?)",
                (_now(), "INITIAL", capital, "account created"),
            )
            self._insert_snapshot(
                conn,
                account_id=cur.lastrowid,
                quotes={},
                reason="account created",
            )

    def adjust_capital(self, amount: float, note: str = "") -> dict:
        """Deposit (positive) or withdraw (negative) virtual funds."""
        amount = float(amount)
        if not math.isfinite(amount) or amount == 0:
            return {"ok": False, "error": "amount must be non-zero"}
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            acct = conn.execute("SELECT * FROM account LIMIT 1").fetchone()
            if not acct:
                raise RuntimeError("no account - create one first")
            new_capital = float(acct["capital"]) + amount
            new_cash = float(acct["cash"]) + amount
            if amount < 0 and -amount > float(acct["cash"]):
                return {"ok": False, "error": "withdrawal cannot exceed available cash"}
            if new_capital <= 0:
                return {"ok": False, "error": "net contributed capital must remain positive"}
            event_type = "DEPOSIT" if amount > 0 else "WITHDRAWAL"
            conn.execute(
                "UPDATE account SET capital=?, cash=?, updated_at=? WHERE id=?",
                (new_capital, new_cash, _now(), acct["id"]),
            )
            conn.execute(
                "INSERT INTO capital_events(date, type, amount, note) VALUES (?,?,?,?)",
                (_now(), event_type, amount, note or event_type.lower()),
            )
            self._insert_snapshot(conn, acct["id"], {}, event_type.lower())
        return {"ok": True, "new_capital": new_capital, "new_cash": new_cash}

    def topup(self, amount: float, note: str = "") -> dict:
        """Backward-compatible deposit helper."""
        if amount <= 0:
            return {"ok": False, "error": "amount must be positive"}
        return self.adjust_capital(float(amount), note)

    def capital_events(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM capital_events ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
            return [dict(row) for row in rows]

    def account(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM account LIMIT 1").fetchone()
            return dict(row) if row else None

    def positions(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM positions ORDER BY id").fetchall()
            return [dict(row) for row in rows]

    def trades(self, limit: int = 500) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
            return [dict(row) for row in rows]

    @staticmethod
    def _summary_from_conn(
        conn: sqlite3.Connection, quotes: dict[str, float] | None = None
    ) -> dict:
        acct = conn.execute("SELECT * FROM account LIMIT 1").fetchone()
        if not acct:
            return {}
        quotes = quotes or {}
        rows = conn.execute("SELECT * FROM positions").fetchall()
        position_value = 0.0
        open_pnl = 0.0
        for row in rows:
            quote = quotes.get(row["symbol"], row["last_price"] or row["entry_price"])
            try:
                price = float(quote)
            except (TypeError, ValueError):
                price = float(row["entry_price"])
            if not math.isfinite(price) or price <= 0:
                price = float(row["entry_price"])
            position_value += price * row["qty"]
            open_pnl += (price - row["entry_price"]) * row["qty"] - float(
                row["entry_cost"] or 0
            )
            conn.execute("UPDATE positions SET last_price=? WHERE id=?", (price, row["id"]))
        realized = float(
            conn.execute("SELECT COALESCE(SUM(net_pnl),0) FROM trades").fetchone()[0]
        )
        cash = float(acct["cash"])
        equity = cash + position_value
        capital = float(acct["capital"])
        return {
            "capital": capital,
            "cash": cash,
            "pos_value": position_value,
            "equity": equity,
            "open_pnl": open_pnl,
            "realized_pnl": realized,
            # Equity minus net deposits reconciles exactly, including all entry costs.
            "total_pnl": equity - capital,
            "total_return_pct": (equity / capital - 1) * 100 if capital else 0.0,
            "n_positions": len(rows),
        }

    def summary(self, quotes: dict[str, float] | None = None) -> dict:
        with self._conn() as conn:
            return self._summary_from_conn(conn, quotes)

    def _insert_snapshot(
        self,
        conn: sqlite3.Connection,
        account_id: int,
        quotes: dict[str, float] | None,
        reason: str,
    ) -> dict:
        summary = self._summary_from_conn(conn, quotes)
        if summary:
            conn.execute(
                """INSERT INTO equity_snapshots(
                    date, equity, cash, position_value, net_deposits,
                    realized_pnl, unrealized_pnl, reason
                ) VALUES (?,?,?,?,?,?,?,?)""",
                (
                    _now(),
                    summary["equity"],
                    summary["cash"],
                    summary["pos_value"],
                    summary["capital"],
                    summary["realized_pnl"],
                    summary["open_pnl"],
                    reason[:120],
                ),
            )
        return summary

    def record_snapshot(
        self, quotes: dict[str, float] | None = None, reason: str = "mark to market"
    ) -> dict:
        with self._conn() as conn:
            acct = conn.execute("SELECT id FROM account LIMIT 1").fetchone()
            if not acct:
                return {}
            return self._insert_snapshot(conn, acct["id"], quotes, reason)

    # ---------------- trading ----------------
    def buy(
        self,
        symbol: str,
        style: str,
        strategy: str,
        qty: int,
        price: float,
        sl: float | None,
        target: float | None,
        reason: str = "",
        confidence: float | None = None,
        p_win: float | None = None,
        *,
        source: str = "manual",
        signal_id: str | None = None,
        risk_pct: float | None = None,
        enforce_limits: bool = True,
    ) -> dict:
        qty = int(qty)
        price = float(price)
        sl = float(sl) if sl is not None else None
        target = float(target) if target is not None else None
        if qty <= 0 or not math.isfinite(price) or price <= 0:
            return {"ok": False, "error": "quantity and price must be positive"}
        if sl is None or not math.isfinite(sl) or sl <= 0 or sl >= price:
            return {"ok": False, "error": "a valid stop below the entry price is required"}

        slip = (
            get("costs.slippage_intraday")
            if style == "intraday"
            else get("costs.slippage_daily")
        )
        fill = price * (1 + slip)
        entry_cost = self.costs.buy_charges(qty, fill, style)
        total = fill * qty + entry_cost
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            acct = conn.execute("SELECT * FROM account LIMIT 1").fetchone()
            if not acct:
                raise RuntimeError("no account - create one first")
            positions = conn.execute("SELECT * FROM positions").fetchall()
            if enforce_limits:
                max_positions = int(get("capital.max_open_positions", 3))
                if len(positions) >= max_positions:
                    return {"ok": False, "error": f"maximum {max_positions} open positions reached"}
                if any(row["symbol"] == symbol for row in positions):
                    return {
                        "ok": False,
                        "error": f"{symbol} is already open; averaging down is not allowed",
                    }
                summary = self._summary_from_conn(conn, {})
                equity = float(summary["equity"])
                max_position = equity * float(get("capital.max_position_pct", 25)) / 100
                if fill * qty > max_position + 1e-6:
                    return {
                        "ok": False,
                        "error": f"position exceeds {get('capital.max_position_pct', 25):g}% equity cap",
                    }
                allowed_risk_pct = (
                    float(risk_pct)
                    if risk_pct is not None
                    else float(get("capital.risk_per_trade_pct", 0.5)) / 100
                )
                # Public API accepts decimal fractions (0.005 = 0.5%).
                if allowed_risk_pct > 0.05:
                    allowed_risk_pct /= 100
                trade_risk = max(0.0, fill - sl) * qty
                if trade_risk > equity * allowed_risk_pct + 1e-6:
                    return {
                        "ok": False,
                        "error": (
                            f"risk ₹{trade_risk:,.0f} exceeds "
                            f"{allowed_risk_pct * 100:.2f}% budget"
                        ),
                    }
                min_value = float(get("capital.min_position_value", 2500))
                if fill * qty < min_value:
                    return {
                        "ok": False,
                        "error": f"position below ₹{min_value:,.0f} minimum",
                    }
            if total > float(acct["cash"]):
                return {
                    "ok": False,
                    "error": f"insufficient cash: need ₹{total:,.0f}, have ₹{acct['cash']:,.0f}",
                }
            conn.execute(
                "UPDATE account SET cash=cash-?, updated_at=? WHERE id=?",
                (total, _now(), acct["id"]),
            )
            cur = conn.execute(
                """INSERT INTO positions(
                    account_id, symbol, style, strategy, qty, entry_price,
                    entry_date, sl, target, reason, confidence, p_win,
                    last_price, entry_cost, source, signal_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    acct["id"],
                    symbol,
                    style,
                    strategy,
                    qty,
                    fill,
                    _now(),
                    sl,
                    target,
                    reason,
                    confidence,
                    p_win,
                    fill,
                    entry_cost,
                    source,
                    signal_id,
                ),
            )
            summary = self._insert_snapshot(conn, acct["id"], {symbol: fill}, f"{source} buy")
        return {
            "ok": True,
            "position_id": cur.lastrowid,
            "fill": fill,
            "costs": entry_cost,
            "equity": summary.get("equity"),
        }

    def sell(
        self, position_id: int, price: float, reason: str = "manual", source: str | None = None
    ) -> dict:
        price = float(price)
        if not math.isfinite(price) or price <= 0:
            return {"ok": False, "error": "price must be positive"}
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM positions WHERE id=?", (position_id,)).fetchone()
            if not row:
                return {"ok": False, "error": "position not found"}
            style = row["style"]
            slip = (
                get("costs.slippage_intraday")
                if style == "intraday"
                else get("costs.slippage_daily")
            )
            fill = price * (1 - slip)
            exit_cost = self.costs.sell_charges(row["qty"], fill, style)
            entry_cost = float(row["entry_cost"] or 0)
            gross = (fill - row["entry_price"]) * row["qty"]
            total_cost = entry_cost + exit_cost
            net = gross - total_cost
            entry_dt = pd.Timestamp(row["entry_date"])
            exit_dt = pd.Timestamp(_now())
            hold = max(0, (exit_dt - entry_dt).days)
            trade_source = source or row["source"] or "manual"
            conn.execute(
                """INSERT INTO trades(
                    account_id, symbol, style, strategy, qty, entry_price,
                    exit_price, entry_date, exit_date, gross_pnl, costs,
                    net_pnl, net_pnl_pct, reason, hold_days, entry_cost,
                    exit_cost, source, signal_id, confidence, p_win
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    row["account_id"],
                    row["symbol"],
                    row["style"],
                    row["strategy"],
                    row["qty"],
                    row["entry_price"],
                    fill,
                    row["entry_date"],
                    _now(),
                    gross,
                    total_cost,
                    net,
                    net / (row["entry_price"] * row["qty"]) * 100,
                    reason,
                    hold,
                    entry_cost,
                    exit_cost,
                    trade_source,
                    row["signal_id"],
                    row["confidence"],
                    row["p_win"],
                ),
            )
            conn.execute("DELETE FROM positions WHERE id=?", (position_id,))
            conn.execute(
                "UPDATE account SET cash=cash+?, updated_at=? WHERE id=?",
                (fill * row["qty"] - exit_cost, _now(), row["account_id"]),
            )
            summary = self._insert_snapshot(
                conn, row["account_id"], {}, f"{trade_source} sell: {reason}"
            )
        return {
            "ok": True,
            "fill": fill,
            "net_pnl": net,
            "net_pnl_pct": net / (row["entry_price"] * row["qty"]) * 100,
            "costs": total_cost,
            "equity": summary.get("equity"),
        }

    def close_all_intraday(self, quotes: dict[str, float]) -> list[dict]:
        results = []
        for position in self.positions():
            if position["style"] == "intraday" and position["symbol"] in quotes:
                results.append(
                    self.sell(
                        position["id"],
                        quotes[position["symbol"]],
                        "auto square-off 15:25",
                        source="auto",
                    )
                )
        return results

    # ---------------- analytics ----------------
    def equity_curve(self) -> pd.DataFrame:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM equity_snapshots ORDER BY id").fetchall()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "date",
                    "equity",
                    "cash",
                    "position_value",
                    "net_deposits",
                    "realized_pnl",
                    "unrealized_pnl",
                    "reason",
                ]
            )
        frame = pd.DataFrame([dict(row) for row in rows])
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["date", "equity"])
        # Legacy snapshots did not include deposits. Fill from account capital as
        # a display fallback; new snapshots always carry the exact value.
        if "net_deposits" in frame:
            frame["net_deposits"] = frame["net_deposits"].ffill().bfill()
        return frame

    def strategy_performance(self) -> pd.DataFrame:
        trades = self.trades(limit=100_000)
        if not trades:
            return pd.DataFrame()
        frame = pd.DataFrame(trades)
        rows: list[dict[str, Any]] = []
        for (strategy, style), group in frame.groupby(["strategy", "style"], dropna=False):
            pnl = pd.to_numeric(group["net_pnl"], errors="coerce").fillna(0)
            losses = -pnl[pnl < 0].sum()
            rows.append(
                {
                    "strategy": strategy,
                    "style": style,
                    "trades": int(len(group)),
                    "wins": int((pnl > 0).sum()),
                    "win_rate_pct": float((pnl > 0).mean() * 100),
                    "net_pnl": float(pnl.sum()),
                    "avg_pnl": float(pnl.mean()),
                    "profit_factor": float(pnl[pnl > 0].sum() / losses) if losses else float("inf"),
                    "avg_hold_days": float(
                        pd.to_numeric(group["hold_days"], errors="coerce").fillna(0).mean()
                    ),
                }
            )
        return pd.DataFrame(rows).sort_values("net_pnl", ascending=False)

    def realized_pnl_since(self, date_prefix: str) -> float:
        with self._conn() as conn:
            return float(
                conn.execute(
                    "SELECT COALESCE(SUM(net_pnl),0) FROM trades WHERE exit_date LIKE ?",
                    (f"{date_prefix}%",),
                ).fetchone()[0]
            )
