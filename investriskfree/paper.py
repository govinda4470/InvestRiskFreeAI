"""Paper-trading broker: virtual money, realistic fills (slippage + full cost
model), persistent SQLite storage. The ONLY place trades are executed before
the user switches to real money.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from .backtest import CostModel
from .config import get


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


class PaperBroker:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or os.path.join(get("data.repo_root"), "data", "paper.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.costs = CostModel()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY, name TEXT, capital REAL, cash REAL,
                    created_at TEXT, updated_at TEXT)"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, account_id INTEGER,
                    symbol TEXT, style TEXT, strategy TEXT, qty INTEGER,
                    entry_price REAL, entry_date TEXT, sl REAL, target REAL,
                    reason TEXT, confidence REAL, p_win REAL, last_price REAL)"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, account_id INTEGER,
                    symbol TEXT, style TEXT, strategy TEXT, qty INTEGER,
                    entry_price REAL, exit_price REAL, entry_date TEXT,
                    exit_date TEXT, gross_pnl REAL, costs REAL, net_pnl REAL,
                    net_pnl_pct REAL, reason TEXT, hold_days INTEGER)"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS equity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, equity REAL)"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS capital_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT,
                    type TEXT, amount REAL, note TEXT)"""
            )

    # ---------------- account ----------------
    def create_account(self, capital: float, name: str = "Paper Trader") -> None:
        with self._conn() as c:
            c.execute("DELETE FROM positions")
            c.execute("DELETE FROM trades")
            c.execute("DELETE FROM equity_snapshots")
            c.execute("DELETE FROM capital_events")
            c.execute("DELETE FROM account")
            c.execute(
                "INSERT INTO account(name, capital, cash, created_at, updated_at) VALUES (?,?,?,?,?)",
                (name, capital, capital, _now(), _now()),
            )
            c.execute(
                "INSERT INTO capital_events(date, type, amount, note) VALUES (?,?,?,?)",
                (_now(), "INITIAL", capital, "account created"),
            )
            c.execute("INSERT INTO equity_snapshots(date, equity) VALUES (?,?)",
                      (_now(), capital))

    def topup(self, amount: float, note: str = "") -> dict:
        """Add virtual capital to the paper account (top-up)."""
        acct = self.account()
        if not acct:
            raise RuntimeError("no account - create one first")
        if amount <= 0:
            return {"ok": False, "error": "amount must be positive"}
        with self._conn() as c:
            c.execute(
                "UPDATE account SET capital=capital+?, cash=cash+?, updated_at=? WHERE id=?",
                (amount, amount, _now(), acct["id"]),
            )
            c.execute(
                "INSERT INTO capital_events(date, type, amount, note) VALUES (?,?,?,?)",
                (_now(), "TOPUP", amount, note or "manual top-up"),
            )
            c.execute(
                "INSERT INTO equity_snapshots(date, equity) VALUES (?,?)",
                (_now(), acct["cash"] + amount),
            )
        return {"ok": True, "new_capital": acct["capital"] + amount}

    def capital_events(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM capital_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def account(self) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM account LIMIT 1").fetchone()
            return dict(row) if row else None

    def positions(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM positions").fetchall()
            return [dict(r) for r in rows]

    def trades(self, limit: int = 500) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def summary(self, quotes: dict[str, float] | None = None) -> dict:
        acct = self.account()
        if not acct:
            return {}
        quotes = quotes or {}
        cash = acct["cash"]
        pos_value = 0.0
        open_pnl = 0.0
        with self._conn() as c:
            rows = c.execute("SELECT * FROM positions").fetchall()
            for r in rows:
                px = quotes.get(r["symbol"], r["last_price"] or r["entry_price"])
                pos_value += px * r["qty"]
                open_pnl += (px - r["entry_price"]) * r["qty"]
                c.execute("UPDATE positions SET last_price=? WHERE id=?", (px, r["id"]))
        equity = cash + pos_value
        realized = c.execute("SELECT COALESCE(SUM(net_pnl),0) FROM trades").fetchone()[0]
        return {
            "capital": acct["capital"], "cash": cash, "pos_value": pos_value,
            "equity": equity, "open_pnl": open_pnl, "realized_pnl": realized,
            "total_pnl": realized + open_pnl,
            "total_return_pct": (equity / acct["capital"] - 1) * 100 if acct["capital"] else 0.0,
            "n_positions": len(rows),
        }

    # ---------------- trading ----------------
    def buy(
        self, symbol: str, style: str, strategy: str, qty: int, price: float,
        sl: float | None, target: float | None, reason: str = "",
        confidence: float | None = None, p_win: float | None = None,
    ) -> dict:
        acct = self.account()
        if not acct:
            raise RuntimeError("no account - create one first")
        slip = get("costs.slippage_intraday") if style == "intraday" else get("costs.slippage_daily")
        fill = price * (1 + slip)
        cost = self.costs.buy_charges(qty, fill, style)
        total = fill * qty + cost
        if total > acct["cash"]:
            return {"ok": False, "error": f"insufficient cash: need ₹{total:,.0f}, have ₹{acct['cash']:,.0f}"}
        with self._conn() as c:
            c.execute(
                "UPDATE account SET cash=cash-?, updated_at=? WHERE id=?",
                (total, _now(), acct["id"]),
            )
            cur = c.execute(
                """INSERT INTO positions(account_id, symbol, style, strategy, qty,
                    entry_price, entry_date, sl, target, reason, confidence, p_win, last_price)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (acct["id"], symbol, style, strategy, qty, fill, _now(), sl, target,
                 reason, confidence, p_win, fill),
            )
            c.execute(
                "INSERT INTO equity_snapshots(date, equity) VALUES (?,?)",
                (_now(), acct["cash"] - total + fill * qty),
            )
        return {"ok": True, "position_id": cur.lastrowid, "fill": fill, "costs": cost}

    def sell(self, position_id: int, price: float, reason: str = "manual") -> dict:
        with self._conn() as c:
            r = c.execute("SELECT * FROM positions WHERE id=?", (position_id,)).fetchone()
            if not r:
                return {"ok": False, "error": "position not found"}
            style = r["style"]
            slip = get("costs.slippage_intraday") if style == "intraday" else get("costs.slippage_daily")
            fill = price * (1 - slip)
            cost = self.costs.sell_charges(r["qty"], fill, style)
            gross = (fill - r["entry_price"]) * r["qty"]
            net = gross - cost
            entry_dt = pd.Timestamp(r["entry_date"])
            exit_dt = pd.Timestamp(_now())
            hold = max(0, (exit_dt - entry_dt).days)
            c.execute(
                """INSERT INTO trades(account_id, symbol, style, strategy, qty,
                    entry_price, exit_price, entry_date, exit_date, gross_pnl,
                    costs, net_pnl, net_pnl_pct, reason, hold_days)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (r["account_id"], r["symbol"], r["style"], r["strategy"], r["qty"],
                 r["entry_price"], fill, r["entry_date"], _now(), gross, cost, net,
                 net / (r["entry_price"] * r["qty"]) * 100, reason, hold),
            )
            c.execute("DELETE FROM positions WHERE id=?", (position_id,))
            c.execute("UPDATE account SET cash=cash+?, updated_at=? WHERE id=?",
                      (fill * r["qty"] - cost, _now(), r["account_id"]))
        return {"ok": True, "fill": fill, "net_pnl": net, "net_pnl_pct": net / (r["entry_price"] * r["qty"]) * 100}

    def close_all_intraday(self, quotes: dict[str, float]) -> list[dict]:
        out = []
        for p in self.positions():
            if p["style"] == "intraday" and p["symbol"] in quotes:
                out.append(self.sell(p["id"], quotes[p["symbol"]], "auto square-off 15:25"))
        return out

    def equity_curve(self) -> pd.DataFrame:
        with self._conn() as c:
            rows = c.execute("SELECT date, equity FROM equity_snapshots ORDER BY id").fetchall()
            if not rows:
                return pd.DataFrame(columns=["date", "equity"])
            df = pd.DataFrame([dict(r) for r in rows])
            df["date"] = pd.to_datetime(df["date"])
            return df
