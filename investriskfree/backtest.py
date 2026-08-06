"""Event-driven backtester with a realistic Indian market cost model.

Costs modelled (all configurable in config.yaml -> costs):
  brokerage (flat or %), STT (delivery 0.1%, intraday sell 0.025%),
  NSE exchange charges, GST on charges, SEBI fees, stamp duty,
  slippage (0.05% daily / 0.10% intraday).

Execution assumptions (realistic, slightly conservative):
  * signals computed on bar CLOSE -> executed at NEXT bar OPEN
  * stops/targets checked intrabar; gaps fill at the open
  * stop fills are always at the WORSE price (sl minus slippage)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import get


# ---------------------------------------------------------------- costs
class CostModel:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or get("costs")

    def _brokerage(self, value: float) -> float:
        flat = self.cfg["brokerage_per_order"]
        pct = self.cfg["brokerage_pct"] * value
        return min(flat, pct) if flat > 0 else pct

    def buy_charges(self, qty: int, price: float, style: str) -> float:
        value = qty * price
        brk = self._brokerage(value)
        exch = self.cfg["exchange_charges"] * value
        if style == "intraday":
            stamp = self.cfg["stamp_duty_intraday"] * value
            stt = 0.0
        else:
            stamp = self.cfg["stamp_duty_delivery"] * value
            stt = self.cfg["stt_delivery"] * value
        gst = self.cfg["gst_pct"] * (brk + exch)
        sebi = self.cfg["sebi_fees"] * value
        return brk + exch + stamp + stt + gst + sebi

    def sell_charges(self, qty: int, price: float, style: str) -> float:
        value = qty * price
        brk = self._brokerage(value)
        exch = self.cfg["exchange_charges"] * value
        stt = self.cfg["stt_intraday_sell"] * value if style == "intraday" \
            else self.cfg["stt_delivery"] * value
        gst = self.cfg["gst_pct"] * (brk + exch)
        sebi = self.cfg["sebi_fees"] * value
        return brk + exch + stt + gst + sebi

    def total_charges(self, qty: int, buy_px: float, sell_px: float, style: str) -> float:
        return self.buy_charges(qty, buy_px, style) + self.sell_charges(qty, sell_px, style)


# ---------------------------------------------------------------- result
@dataclass
class Trade:
    symbol: str
    style: str
    strategy: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: int
    gross_pnl: float
    costs: float
    net_pnl: float
    net_pnl_pct: float
    hold_days: int
    reason: str
    rr: float


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    style: str
    trades: list[Trade] = field(default_factory=list)
    equity: pd.Series = None
    monthly: pd.Series = None
    stats: dict = field(default_factory=dict)
    filtered_by_regime: int = 0

    def to_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ for t in self.trades])


# ---------------------------------------------------------------- engine
class Backtester:
    def __init__(self, cost_cfg: dict | None = None, slippage: float | None = None):
        self.costs = CostModel(cost_cfg)
        self.slippage = slippage

    def run(
        self,
        df: pd.DataFrame,
        strategy,
        regime_ok: np.ndarray | None = None,
        capital: float = 100_000.0,
        risk_pct: float | None = None,
        max_pos_pct: float | None = None,
        min_position_value: float | None = None,
    ) -> BacktestResult:
        risk_pct = risk_pct if risk_pct is not None else get("capital.risk_per_trade_pct", 0.5) / 100
        max_pos_pct = max_pos_pct if max_pos_pct is not None else get("capital.max_position_pct", 25) / 100
        min_position_value = min_position_value if min_position_value is not None else get("capital.min_position_value", 2500)
        style = strategy.style
        slip = self.slippage if self.slippage is not None else (
            get("costs.slippage_intraday") if style == "intraday" else get("costs.slippage_daily")
        )
        sig = strategy.signals(df)
        o = df["Open"].to_numpy(float)
        h = df["High"].to_numpy(float)
        l = df["Low"].to_numpy(float)
        c = df["Close"].to_numpy(float)
        entry = sig["entry"].to_numpy(float)
        exit_ = sig["exit"].to_numpy(float)
        sl_arr = sig["sl"].to_numpy(float)
        tg_arr = sig["target"].to_numpy(float)
        reason_arr = sig["reason"].to_numpy(object)
        rr_arr = sig["rr"].to_numpy(float)
        max_hold = int(sig["max_hold"].iloc[0]) if "max_hold" in sig else 20
        sqoff = sig["square_off"].to_numpy(float) if "square_off" in sig else None
        sl_on_close = bool(sig["sl_on_close"].iloc[0]) if "sl_on_close" in sig else False

        n = len(df)
        dates = df.index
        cash = float(capital)
        pos = None
        trades: list[Trade] = []
        filtered = 0
        eq = np.zeros(n)
        equity_val = capital

        def enter_at(idx: int):
            nonlocal cash, pos, filtered
            e = o[idx] * (1 + slip)
            sl_px = sl_arr[idx - 1]
            tg_px = tg_arr[idx - 1]
            if not np.isfinite(sl_px) or sl_px <= 0:
                return
            risk_amt = equity_val * risk_pct
            risk_per_share = e - sl_px
            if risk_per_share <= 0:
                return
            qty = int(np.floor(risk_amt / risk_per_share))
            if qty <= 0:
                return
            pos_value = qty * e
            if pos_value > equity_val * max_pos_pct:
                qty = int(np.floor(equity_val * max_pos_pct / e))
                pos_value = qty * e
            if qty <= 0 or pos_value < min_position_value:
                return
            cost = self.costs.buy_charges(qty, e, style)
            if pos_value + cost >= cash:
                return
            cash -= pos_value + cost
            pos = {
                "qty": qty, "entry_px": e, "sl": sl_px, "target": tg_px,
                "entry_i": idx, "entry_date": dates[idx], "cost": cost,
                "exit_pending": False, "reason": str(reason_arr[idx - 1]),
                "rr": float(rr_arr[idx - 1]) if np.isfinite(rr_arr[idx - 1]) else 0.0,
                "exit_px": None, "exit_i": None, "exit_reason": None,
            }

        def exit_pos(idx: int, px: float, reason: str):
            nonlocal cash, pos
            p = pos
            sell_px = px * (1 - slip)
            cost = self.costs.sell_charges(p["qty"], sell_px, style)
            gross = (sell_px - p["entry_px"]) * p["qty"]
            net = gross - cost - p["cost"]
            cash += sell_px * p["qty"] - cost
            hold = (dates[idx] - p["entry_date"]).days
            trades.append(Trade(
                symbol=df.attrs.get("symbol", "?"), style=style,
                strategy=strategy.name,
                entry_date=p["entry_date"], exit_date=dates[idx],
                entry_price=p["entry_px"], exit_price=sell_px, qty=p["qty"],
                gross_pnl=gross, costs=cost + p["cost"], net_pnl=net,
                net_pnl_pct=net / (p["entry_px"] * p["qty"]) * 100,
                hold_days=hold, reason=reason, rr=p["rr"],
            ))
            pos = None

        for i in range(1, n):
            # --- manage open position ---
            if pos is not None:
                if pos["exit_pending"]:
                    exit_pos(i, o[i], f"{pos['exit_reason']} (next open)")
                    pos = None
                else:
                    sl_px, tg_px = pos["sl"], pos["target"]
                    oi, hi, li = o[i], h[i], l[i]
                    if sl_on_close:
                        if c[i] < sl_px:
                            exit_pos(i, c[i], "SL (trend break, on close)")
                        elif tg_px and h[i] >= tg_px:
                            exit_pos(i, tg_px, "TARGET")
                    else:
                        if oi <= sl_px:
                            exit_pos(i, oi, "SL (gap)")
                        elif tg_px and oi >= tg_px:
                            exit_pos(i, oi, "TARGET (gap)")
                        elif li <= sl_px:
                            exit_pos(i, sl_px, "SL")
                        elif tg_px and hi >= tg_px:
                            exit_pos(i, tg_px, "TARGET")
                        else:
                            hold_bars = i - pos["entry_i"]
                            if sqoff is not None and sqoff[i]:
                                # intraday: square off at the SAME bar's close
                                # (never carry risk into the next session)
                                exit_pos(i, c[i], "Square-off (EOD 15:25)")
                            elif exit_[i]:
                                pos["exit_pending"] = True
                                pos["exit_reason"] = "Signal exit"
                            elif hold_bars >= max_hold:
                                pos["exit_pending"] = True
                                pos["exit_reason"] = f"Max hold ({max_hold})"
            # --- new entry from previous close signal ---
            if pos is None and entry[i - 1] == 1.0:
                if regime_ok is not None and not regime_ok[i - 1]:
                    filtered += 1
                else:
                    enter_at(i)
                    # entry-day intrabar stop/target check (entered at the open)
                    if pos is not None:
                        if sl_on_close:
                            if c[i] < pos["sl"]:
                                exit_pos(i, c[i], "SL (same bar close)")
                            elif pos["target"] and h[i] >= pos["target"]:
                                exit_pos(i, pos["target"], "TARGET (same bar)")
                        else:
                            if o[i] <= pos["sl"]:
                                exit_pos(i, o[i], "SL (entry gap)")
                            elif pos["target"] and o[i] >= pos["target"]:
                                exit_pos(i, o[i], "TARGET (entry gap)")
                            elif l[i] <= pos["sl"]:
                                exit_pos(i, pos["sl"], "SL (same bar)")
                            elif pos["target"] and h[i] >= pos["target"]:
                                exit_pos(i, pos["target"], "TARGET (same bar)")
            # --- mark to market ---
            if pos is not None:
                equity_val = cash + pos["qty"] * c[i]
            else:
                equity_val = cash
            eq[i] = equity_val

        if pos is not None:  # liquidate any leftover at last close
            exit_pos(n - 1, c[n - 1], "End of data")

        equity = pd.Series(eq, index=dates)
        result = BacktestResult(
            symbol=df.attrs.get("symbol", "?"), strategy=strategy.name,
            style=style, trades=trades, equity=equity,
            filtered_by_regime=filtered,
        )
        result.monthly = equity.resample("ME").last().pct_change().dropna()
        result.stats = compute_stats(result, capital)
        return result


# ---------------------------------------------------------------- stats
def compute_stats(res: BacktestResult, capital: float) -> dict:
    eq = res.equity
    trades = res.trades
    n = len(trades)
    s: dict = {"trades": n}
    if n == 0:
        s.update(dict(win_rate=0.0, profit_probability=0.0, avg_win_pct=0.0,
                      avg_loss_pct=0.0, profit_factor=0.0, expectancy_pct=0.0,
                      total_return_pct=0.0, cagr=0.0, max_drawdown_pct=0.0,
                      avg_hold_days=0.0, median_hold_days=0.0))
        return s
    pnl = np.array([t.net_pnl for t in trades])
    pnl_pct = np.array([t.net_pnl_pct for t in trades])
    wins = pnl > 0
    s["win_rate"] = float(wins.mean() * 100)
    s["profit_probability"] = s["win_rate"]  # P(trade is profitable)
    s["avg_win_pct"] = float(pnl_pct[wins].mean()) if wins.any() else 0.0
    s["avg_loss_pct"] = float(pnl_pct[~wins].mean()) if (~wins).any() else 0.0
    gross_win = pnl[wins].sum()
    gross_loss = -pnl[~wins].sum()
    s["profit_factor"] = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")
    s["expectancy_pct"] = float(pnl_pct.mean())
    s["expectancy_rs"] = float(pnl.mean())
    s["total_costs"] = float(sum(t.costs for t in trades))
    s["total_return_pct"] = float((eq.iloc[-1] / capital - 1) * 100)
    days = (eq.index[-1] - eq.index[0]).days
    if days > 0 and eq.iloc[-1] > 0:
        s["cagr"] = float(((eq.iloc[-1] / capital) ** (365.25 / days) - 1) * 100)
    else:
        s["cagr"] = 0.0
    cummax = eq.cummax()
    dd = (eq - cummax) / cummax
    s["max_drawdown_pct"] = float(dd.min() * 100)
    rets = eq.pct_change().dropna()
    if rets.std() > 0:
        s["sharpe"] = float(rets.mean() / rets.std() * np.sqrt(252))
    else:
        s["sharpe"] = 0.0
    holds = np.array([t.hold_days for t in trades])
    s["avg_hold_days"] = float(holds.mean())
    s["median_hold_days"] = float(np.median(holds))
    s["max_hold_days"] = float(holds.max())
    s["best_trade_pct"] = float(pnl_pct.max())
    s["worst_trade_pct"] = float(pnl_pct.min())
    s["total_turnover"] = float(sum(t.entry_price * t.qty + t.exit_price * t.qty for t in trades))
    s["cost_drag_pct"] = float(s["total_costs"] / s["total_turnover"] * 100) if s["total_turnover"] else 0.0
    # probability of a profitable month / rolling windows
    m = res.monthly
    s["profitable_month_prob"] = float((m > 0).mean() * 100) if len(m) else 0.0
    # P(capital higher after X months) from equity curve
    for months in (1, 3, 6, 12):
        days = months * 21
        vals = eq.values
        if len(vals) > days:
            with np.errstate(divide="ignore", invalid="ignore"):
                fwd = vals[days:] / vals[:-days]
            s[f"p_up_{months}m"] = float((fwd > 1).mean() * 100)
        else:
            s[f"p_up_{months}m"] = float("nan")
    return s
