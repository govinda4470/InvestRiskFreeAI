"""
InvestRiskFreeAI - Capital-protection-first AI signal & paper trading system
for the Indian (NSE) stock market.

Run:  streamlit run app.py

All signals are AI-assisted. Paper-trade with virtual money before ever using
real money. Nothing here is financial advice.
"""
from __future__ import annotations

import hmac
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="InvestRiskFreeAI", page_icon="🛡️", layout="wide")

from investriskfree.auth import AuthStore  # noqa: E402
from investriskfree.autotrade import AutoTradeAgent  # noqa: E402
from investriskfree.backtest import Backtester  # noqa: E402
from investriskfree.brain import market_regime, regime_label  # noqa: E402
from investriskfree.config import get  # noqa: E402
from investriskfree.data.loader import load_daily, load_index_daily  # noqa: E402
from investriskfree.data.synthetic import daily_to_intraday  # noqa: E402
from investriskfree.data.universe import get_universe  # noqa: E402
from investriskfree.paper import PaperBroker  # noqa: E402
from investriskfree.scanner import scan  # noqa: E402
from investriskfree.strategies import StrategyRegistry  # noqa: E402

STYLE_META = {
    "swing": ("🌊 Swing", "2-25 day holds, low stress"),
    "intraday": ("⚡ Intraday", "same-day, highest risk"),
    "invest": ("🏦 Invest", "weeks-months, buy quality dips"),
}
STRATEGY_INFO = {
    "swing_trend": "TrendRider - buy the pullback inside an EMA20/50 uptrend with ADX confirmation.",
    "swing_breakout": "RangeBreaker - Donchian 20-day breakout with volume confirmation.",
    "swing_meanrev": "DipBuyer - RSI(2) oversold bounce (LAB ONLY - near zero edge after costs).",
    "intraday_orb": "Opening Range Breakout - 9:15-9:30 range break, square-off by 15:25.",
    "intraday_vwap": "VWAP Reversion - fade extreme deviations (LAB ONLY - negative edge).",
    "invest": "TrendDip Investor - buy controlled dips in stocks above a rising 200SMA.",
}


# ----------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False, ttl=600)
def cached_scan(capital: float, styles: tuple, real_intraday: bool) -> list[dict]:
    pb = st.progress(0.0, text="Scanning universe...")
    def cb(frac, sym):
        pb.progress(float(frac), text=f"Scanning {sym or '...'}")
    sigs = scan(capital=capital, styles=styles, real_intraday=real_intraday,
                live_data=real_intraday, demo_intraday=True, progress_cb=cb)
    pb.empty()
    return sigs


@st.cache_data(show_spinner=False, ttl=600)
def cached_backtest(symbol: str, strategy: str) -> dict:
    df = load_daily(symbol)
    df.attrs["symbol"] = symbol
    strat = StrategyRegistry.get(strategy)
    reg = None
    if strat.style != "intraday":
        try:
            closes = pd.DataFrame()
            from investriskfree.data.loader import list_bundled_symbols
            for s in list_bundled_symbols()[:40]:
                closes[s] = load_daily(s)["Close"]
            proxy = closes.mean(axis=1).to_frame("Close")
            proxy["Open"] = proxy["High"] = proxy["Low"] = proxy["Close"]
            proxy["Volume"] = 1
            reg = market_regime(proxy).reindex(df.index).ffill().fillna(False).to_numpy()
        except Exception:
            reg = None
    if strat.style == "intraday":
        df = daily_to_intraday(df.tail(700), seed=42)
    res = Backtester().run(df, strat, regime_ok=reg, capital=100_000)
    return {
        "stats": res.stats,
        "trades": res.to_frame(),
        "equity": res.equity,
        "filtered": res.filtered_by_regime,
        "symbol": symbol,
        "strategy": strategy,
        "style": strat.style,
    }


def fmt(x, dec=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{x:,.{dec}f}"
    except Exception:
        return "—"


def metric_card(col, label, value, help=None, delta=None):
    col.metric(label, value, delta=delta, help=help)


def current_broker() -> PaperBroker:
    """Return the signed-in user's isolated virtual ledger."""
    user_id = st.session_state.get("user_id")
    return PaperBroker(user_id=user_id) if user_id else PaperBroker()


# ----------------------------------------------------------------- pages
def page_dashboard(broker: PaperBroker):
    st.title("🛡️ InvestRiskFreeAI — Dashboard")
    st.caption("Capital protection first · AI signals · Paper trade before real money · India NSE")
    st.markdown(
        "<div style='background:#fff3cd;border:1px solid #ffe08a;border-radius:8px;padding:10px 14px;'>"
        "⚠️ <b>Honesty first:</b> there is no 'risk-free' return in markets. This system's job is to "
        "keep your losses tiny (0.5–1% risk per trade) and let probability work over many trades. "
        "Demo signals in this sandbox use real NSE history up to 2022. On your machine, enable "
        "<b>live data</b> in the sidebar for current prices.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.2, 1, 1])
    # regime status
    try:
        idx = load_index_daily()
        closes = pd.DataFrame()
        for s in get_universe()[:40]:
            closes[s["symbol"]] = load_daily(s["symbol"])["Close"]
        breadth = (closes > closes.rolling(200).mean()).mean(axis=1)
        label, dist, br = regime_label(idx, float(breadth.iloc[-1]))
    except Exception:
        label, dist, br = "UNKNOWN", 0.0, 0.0
    with c1:
        st.subheader("🌐 Market Regime")
        color = {"RISK-ON": "green", "NEUTRAL": "orange", "RISK-OFF": "red"}.get(label, "gray")
        st.markdown(f"<div style='font-size:2rem;color:{color};font-weight:700'>{label}</div>",
                    unsafe_allow_html=True)
        st.write(f"Nifty vs 200SMA: **{dist:+.1f}%** · Breadth: **{br:.0%}** of stocks above 200SMA")
        if label == "RISK-OFF":
            st.warning("Longs are blocked in RISK-OFF. Capital protection = stay in cash.")
    with c2:
        st.subheader("💰 Capital Protection Rules")
        st.markdown(
            f"- Risk per trade: **{get('capital.risk_per_trade_pct')}%**\n"
            f"- Max position: **{get('capital.max_position_pct')}%** of capital\n"
            f"- Max open positions: **{get('capital.max_open_positions')}**\n"
            f"- Min position: ₹{get('capital.min_position_value'):,.0f} (costs)"
        )
    with c3:
        st.subheader("🧠 AI Brain")
        st.markdown(
            f"- Min confidence: **{get('brain.min_confidence')}** / 100\n"
            f"- Strong signal ≥ **{get('brain.strong_confidence')}**\n"
            f"- ML gate: walk-forward P(win) ≥ baseline\n"
            f"- Signals below threshold → **blocked**"
        )

    st.divider()
    st.subheader("📡 Today's AI Signals")
    col_left, col_right = st.columns([3, 1])
    with col_right:
        styles_sel = st.multiselect("Styles", ["swing", "intraday", "invest"],
                                    default=["swing", "invest"],
                                    format_func=lambda s: STYLE_META[s][0])
        live = st.toggle("Live data (yfinance)", value=False,
                         help="On your machine: fetch fresh NSE data. In this sandbox only bundled 2012-22 data works.")
        refresh = st.button("🔄 Re-scan", width="stretch")
    with col_left:
        if refresh:
            cached_scan.clear()
        account_summary = broker.summary()
        scan_capital = float(account_summary.get("equity", 100_000))
        signals = cached_scan(scan_capital, tuple(styles_sel), live)
        actionable = [s for s in signals if not s["blocked"]]
        blocked = [s for s in signals if s["blocked"]]
        st.markdown(f"**{len(actionable)} actionable** · {len(blocked)} blocked by risk rules")
        if actionable:
            rows = []
            for s in actionable:
                pwin = f"{s['profit_prob_pct']:.0f}%" if s.get("profit_prob_pct") else "n/a"
                dur = f"~{s['expected_duration_days']:.0f}d" if s.get("expected_duration_days") else "—"
                gap = s.get("gap_from_entry_pct")
                gap_str = f"{gap:+.1f}%" if gap is not None else "—"
                rows.append({
                    "Symbol": s["symbol"], "Style": STYLE_META[s["style"]][0],
                    "Strategy": s["strategy"], "Action": "BUY",
                    "Last ₹": fmt(s.get("last_price")),
                    "Entry≈": fmt(s["entry_ref"]),
                    "vs Entry": gap_str,
                    "Stop": fmt(s["sl"]), "Target": fmt(s["target"]),
                    "R:R": f"1:{s['rr']:.1f}",
                    "Confidence": f"{s['confidence']:.0f}%", "Level": s["level"],
                    "P(win)": pwin, "Duration": dur,
                    "From backtest": f"{s['win_rate_pct']:.0f}% win / {s['expected_duration_days']:.0f}d"
                    if s.get("win_rate_pct") else "—",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("**Last ₹** = latest available price (live quote when 'Live data' is on, else the "
                       "last close in the dataset). **vs Entry** = how far the current price is from the "
                       "signal entry. Entry is reference price (signal close); paper trade at next bar's open "
                       "in reality. RR 0.00 means the stop is dynamic (e.g. trend line).")
        else:
            st.info("No actionable signals right now. Waiting is a position — capital protected.")
        if blocked:
            with st.expander(f"🚫 {len(blocked)} signals blocked by the risk brain"):
                for s in blocked:
                    st.markdown(f"- **{s['symbol']}** ({s['strategy']}) → {s['blocked']}")
    st.divider()
    st.subheader("📊 Strategy performance (real NSE backtest, 2012–2022)")
    try:
        from investriskfree.scanner import load_stats_registry
        reg = load_stats_registry()
        rows = []
        for name, d in reg.items():
            if name in ("meta", "by_pair") or not isinstance(d, dict) or "win_rate" not in d:
                continue
            rows.append({
                "Strategy": name, "Style": STYLE_META.get(StrategyRegistry.get(name).style, ("", ""))[0],
                "P(profit)": f"{d['win_rate']:.1f}%",
                "Expectancy/trade": f"{d['expectancy_pct']:+.2f}%",
                "Med duration": f"{d['median_hold_days']:.0f}d",
                "Max DD": f"{d['max_drawdown_pct']:.1f}%",
                "P(up in 3m)": f"{d.get('p_up_3m', 0):.0f}%",
                "Profit factor": f"{d['profit_factor']:.2f}",
                "Trades": d.get("n_trades", 0),
                "Demo data": "yes" if d.get("demo_data") else "",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("Run `python tools/build_stats.py` to build the registry.")
    except Exception as e:
        st.info(f"Registry not ready: {e}")


def page_paper(broker: PaperBroker):
    st.title("💰 Paper Trading — Virtual Money")
    st.caption("Your private virtual ledger. Realistic brokerage, STT and slippage are charged, "
               "and every order is capped by your selected risk budget.")
    acct = broker.account()
    if acct is None:
        st.info("Create your virtual account to start.")
        c1, c2 = st.columns([1, 2])
        cap = c1.number_input("Starting virtual capital (₹)", 10_000, 10_000_000, 100_000, step=10_000)
        if c1.button("Create paper account", type="primary", width="stretch"):
            broker.create_account(float(cap))
            st.rerun()
    else:
        live = st.toggle(
            "Use live yfinance data",
            value=False,
            help="Uses delayed yfinance candles/quotes and refuses to label bundled 2022 data as live.",
        )
        quotes = {}
        for position in broker.positions():
            try:
                if live:
                    from investriskfree.data.loader import fetch_quote
                    quote = fetch_quote(position["symbol"])
                    if quote is None:
                        raise RuntimeError("quote unavailable")
                    quotes[position["symbol"]] = float(quote)
                else:
                    frame = load_daily(position["symbol"])
                    quotes[position["symbol"]] = float(frame["Close"].iloc[-1])
            except Exception:
                # Keep the broker's last known mark instead of substituting a
                # bundled historical price into a live position.
                quotes[position["symbol"]] = float(
                    position.get("last_price") or position["entry_price"]
                )
        summ = broker.summary(quotes)
        cash = float(summ["cash"])
        equity = float(summ["equity"])

        # ---------------- risk preference ----------------
        rc1, rc2 = st.columns([2, 1])
        risk_pct = rc1.slider("Risk per trade (% of equity)", 0.25, 1.0,
                              float(get("capital.risk_per_trade_pct")), 0.05,
                              help="The suggestion engine never risks more than this % of equity on one trade.")
        with rc2:
            st.markdown(f"**Risk budget:** ₹{equity * risk_pct / 100:,.0f} per trade")

        # ---------------- capital updates ----------------
        with st.expander("💳 Update my virtual capital", expanded=False):
            tc1, tc2, tc3, tc4 = st.columns([1, 1.5, 1.5, 1])
            capital_action = tc1.selectbox("Action", ["Deposit", "Withdraw"])
            capital_amount = tc2.number_input(
                "Amount (₹)", 1_000, 100_000_000, 50_000, step=10_000,
                key="capital_adjustment_amount",
            )
            capital_note = tc3.text_input(
                "Note (optional)", "", key="capital_adjustment_note",
                placeholder="e.g. monthly allocation",
            )
            if tc4.button("Update", type="primary", width="stretch"):
                signed_amount = float(capital_amount) * (1 if capital_action == "Deposit" else -1)
                result = broker.adjust_capital(signed_amount, capital_note)
                if result["ok"]:
                    st.success(
                        f"Capital updated. Net contributed capital is now "
                        f"₹{result['new_capital']:,.0f}."
                    )
                    st.rerun()
                else:
                    st.error(result["error"])
            events = broker.capital_events()
            if events:
                event_frame = pd.DataFrame(events)[["date", "type", "amount", "note"]]
                event_frame["amount"] = event_frame["amount"].map(lambda value: f"₹{value:+,.0f}")
                st.caption("Capital history (kept separate from strategy P&L):")
                st.dataframe(event_frame.head(10), width="stretch", hide_index=True)

        st.divider()
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Capital", f"₹{summ['capital']:,.0f}")
        m2.metric("Available Cash", f"₹{cash:,.0f}")
        m3.metric("Position Value", f"₹{summ['pos_value']:,.0f}")
        m4.metric("Total P&L", f"₹{summ['total_pnl']:+,.0f}",
                  delta=f"{summ['total_return_pct']:+.2f}%")
        m5.metric("Open Positions", f"{summ['n_positions']}/{get('capital.max_open_positions', 3)}")
        m6.metric("Risk Budget", f"₹{equity * risk_pct / 100:,.0f}")

        # ---------------- signals ----------------
        signals = cached_scan(equity, ("swing", "invest", "intraday"), live)
        actionable = [s for s in signals if not s["blocked"]]
        max_positions = int(get("capital.max_open_positions", 3))
        at_position_limit = summ["n_positions"] >= max_positions

        # ---- suggestion engine: best buys given available capital ----
        def _suggest(s, risk_p):
            """Return (score, suggestion dict) for ranking + display."""
            from investriskfree.brain import suggest_position_from_cash
            tradable_price = float(s.get("last_price") or s["entry_ref"])
            sugg = suggest_position_from_cash(cash, equity, tradable_price, s["sl"],
                                              s["style"], risk_p / 100)
            score = s["confidence"] * 0.6
            if s.get("profit_prob_pct"):
                score += min(float(s["profit_prob_pct"]), 100) * 0.25
            score += min(float(s["rr"] or 0), 3.0) * 8
            if sugg.get("blocked"):
                score -= 25
            else:
                score += 15  # affordable bonus
            return score, sugg

        ranked = []
        for s in actionable:
            score, sugg = _suggest(s, risk_pct)
            ranked.append((score, s, sugg))
        ranked.sort(key=lambda x: -x[0])

        st.divider()
        st.subheader("🏆 Best buys for your current capital")
        st.caption("Ranked by confidence + backtested profit probability + reward:risk + "
                   "whether your available cash can afford the suggested position. "
                   "**Scores refresh automatically when you top up capital.**")
        if not ranked:
            st.info("No actionable signals right now. Holding cash is a position too — capital protected.")
        else:
            best_score, best_sig, best_sugg = ranked[0]
            with st.container(border=True):
                st.markdown(f"### ⭐ Top pick: **{best_sig['symbol']}** — {best_sig['strategy']} "
                            f"(score {best_score:.0f})")
                cols = st.columns(6)
                cols[0].markdown(f"**Last ₹** {fmt(best_sig.get('last_price'))}")
                cols[1].markdown(f"**Entry ≈** ₹{fmt(best_sig['entry_ref'])}")
                cols[2].markdown(f"**SL** ₹{fmt(best_sig['sl'])} · **TGT** ₹{fmt(best_sig['target'])}")
                cols[3].markdown(f"**R:R** 1:{best_sig['rr']:.1f} · **P(win)** "
                                 f"{best_sig.get('profit_prob_pct', '—')}%")
                cols[4].markdown(f"**Confidence** {best_sig['confidence']:.0f}% ({best_sig['level']})")
                if best_sugg.get("blocked"):
                    cols[5].markdown(f"⚠️ {best_sugg['blocked']}")
                else:
                    cols[5].markdown(f"**Suggested:** {best_sugg['qty']} qty · "
                                     f"₹{best_sugg['pos_value']:,.0f} "
                                     f"({best_sugg['pos_value'] / equity * 100:.1f}% of equity)")
                st.caption(f"{best_sig['reason']} · Last price as of {best_sig.get('as_of', '?')}")
            rows = []
            for score, s, sugg in ranked:
                rows.append({
                    "Rank": "#" + str(ranked.index((score, s, sugg)) + 1),
                    "Symbol": s["symbol"], "Strategy": s["strategy"],
                    "Score": f"{score:.0f}", "Confidence": f"{s['confidence']:.0f}%",
                    "Last ₹": fmt(s.get("last_price")),
                    "Entry ₹": fmt(s["entry_ref"]),
                    "P(win)": f"{s.get('profit_prob_pct', '—')}%",
                    "R:R": f"1:{s['rr']:.1f}",
                    "Suggested qty": sugg["qty"] if not sugg.get("blocked") else "—",
                    "Needed ₹": f"{sugg['pos_value']:,.0f}" if not sugg.get("blocked") else "—",
                    "Note": sugg.get("blocked") or f"{sugg['pos_value'] / equity * 100:.1f}% of equity",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        # ---------------- trade tab ----------------
        st.divider()
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📡 Signals to trade", "📂 Open positions", "📜 Trade journal",
            "💹 Equity & drawdown", "📊 Strategy analytics",
        ])
        with tab1:
            if at_position_limit:
                st.warning(f"You already have {summ['n_positions']} open positions "
                           f"(max {max_positions}). Close one before buying — capital protection rule 4.")
            if not actionable:
                st.info("No actionable signals. Holding cash is a position too.")
            for score, s, sugg in ranked:
                with st.container(border=True):
                    pwin = f"{s['profit_prob_pct']:.0f}%" if s.get("profit_prob_pct") else "—"
                    st.markdown(f"**{s['symbol']}** · {s['strategy']} · {s['level']} "
                                f"({s['confidence']:.0f} conf) · P(win) {pwin} · "
                                f"R:R 1:{s['rr']:.1f} · {s['reason']}")
                    cols = st.columns([2, 1.2, 1.2, 1.2, 1.2, 1])
                    gap = s.get("gap_from_entry_pct")
                    cols[0].markdown(f"**Last ₹{fmt(s.get('last_price'))}** ({gap:+.1f}% vs entry) · "
                                     f"Entry ≈ ₹{fmt(s['entry_ref'])} · SL ₹{fmt(s['sl'])} · "
                                     f"TGT ₹{fmt(s['target'])}")
                    if sugg.get("blocked"):
                        cols[1].markdown(f"⚠️ **{sugg['blocked']}**")
                    else:
                        cols[1].markdown(f"✅ Suggested: **{sugg['qty']} qty** "
                                         f"(₹{sugg['pos_value']:,.0f})")
                    max_affordable = sugg["qty"] if not sugg.get("blocked") else 0
                    qty = cols[2].number_input("Quantity", min_value=1, max_value=max(10000, max_affordable),
                                               value=max(1, max_affordable), step=1,
                                               key=f"qty_{s['symbol']}_{s['strategy']}",
                                               label_visibility="collapsed",
                                               help="Enter how many shares YOU want. The suggested value "
                                                    "is the safest size for your capital.")
                    qty = int(qty)
                    trade_price = float(s.get("last_price") or s["entry_ref"])
                    est_value = qty * trade_price
                    est_risk = qty * max(0, trade_price - s["sl"])
                    from investriskfree.backtest import CostModel as _CM
                    est_cost = _CM().buy_charges(qty, trade_price, s["style"])
                    cols[3].markdown(f"₹{est_value:,.0f} · risk ₹{est_risk:,.0f}")
                    over = (est_value + est_cost) > cash
                    risk_over = est_risk > equity * risk_pct / 100 + 1e-6
                    position_over = est_value > equity * get("capital.max_position_pct", 25) / 100
                    if over:
                        cols[4].markdown("🔴 exceeds cash")
                    elif risk_over:
                        cols[4].markdown("🔴 exceeds risk budget")
                    elif position_over:
                        cols[4].markdown("🔴 exceeds position cap")
                    else:
                        cols[4].markdown(f"{(est_value + est_cost) / cash * 100:.1f}% of cash")
                    if cols[5].button("Buy 📈", key=f"buy_{s['symbol']}_{s['strategy']}",
                                      type="primary", disabled=bool(
                                          at_position_limit or over or risk_over or position_over or qty <= 0
                                      )):
                        res = broker.buy(s["symbol"], s["style"], s["strategy"], qty,
                                         trade_price, s["sl"], s["target"],
                                         reason=s["reason"], confidence=s["confidence"],
                                         p_win=s.get("ml_p_win_pct") or s.get("profit_prob_pct"),
                                         risk_pct=risk_pct / 100)
                        if res["ok"]:
                            st.success(f"Paper BUY {s['symbol']} x{qty} @ {res['fill']:.2f} "
                                       f"(costs ₹{res['costs']:.2f})")
                            st.rerun()
                        else:
                            st.error(res["error"])
        with tab2:
            pos = broker.positions()
            if not pos:
                st.info("No open positions.")
            for p in pos:
                with st.container(border=True):
                    cols = st.columns([3, 2, 2, 2, 2, 2, 1])
                    last = quotes.get(p["symbol"], p["last_price"] or p["entry_price"])
                    pnl = (last - p["entry_price"]) * p["qty"]
                    cols[0].markdown(f"**#{p['id']}** {p['symbol']} · {p['strategy']} ({p['style']})")
                    cols[1].markdown(f"Qty {p['qty']} @ {fmt(p['entry_price'])}")
                    cols[2].markdown(f"Last {fmt(last)}")
                    cols[3].markdown(f"SL {fmt(p['sl'])} · TGT {fmt(p['target'])}")
                    cols[4].markdown(f"P&L **₹{pnl:+,.0f}**")
                    cols[5].markdown(f"{p['reason'][:40]}")
                    if cols[6].button("Sell ✋", key=f"sell_{p['id']}"):
                        res = broker.sell(p["id"], last, "manual exit")
                        if res["ok"]:
                            st.success(f"Sold at {res['fill']:.2f}, P&L ₹{res['net_pnl']:+,.0f}")
                            st.rerun()
            if any(p["style"] == "intraday" for p in pos):
                if st.button("⏰ Square off all intraday positions (15:25 rule)"):
                    results = broker.close_all_intraday(quotes)
                    st.success(f"Closed {len(results)} positions")
                    st.rerun()
        with tab3:
            tr = broker.trades(limit=200)
            if tr:
                df = pd.DataFrame(tr)
                df["net_pnl_pct"] = df["net_pnl_pct"].round(2)
                st.dataframe(df[["symbol", "strategy", "source", "entry_date", "exit_date", "qty",
                                 "entry_price", "exit_price", "costs", "net_pnl", "net_pnl_pct",
                                 "reason", "hold_days"]], width="stretch", hide_index=True)
                w = (df["net_pnl"] > 0).mean() * 100
                m1, m2, m3 = st.columns(3)
                m1.metric("Win rate", f"{w:.1f}%")
                m2.metric("Total realized", f"₹{df['net_pnl'].sum():+,.0f}")
                m3.metric("Trades", f"{len(df)}")
            else:
                st.info("No closed trades yet.")
        with tab4:
            eq = broker.equity_curve()
            if len(eq):
                eq = eq.sort_values("date").copy()
                eq["peak"] = eq["equity"].cummax()
                eq["drawdown_pct"] = (eq["equity"] / eq["peak"] - 1) * 100
                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                    subplot_titles=("Virtual equity vs net deposits", "Drawdown"),
                )
                fig.add_trace(
                    go.Scatter(x=eq["date"], y=eq["equity"], mode="lines+markers",
                               name="Equity", line=dict(color="#2e7d32", width=2)),
                    row=1, col=1,
                )
                if "net_deposits" in eq and eq["net_deposits"].notna().any():
                    fig.add_trace(
                        go.Scatter(x=eq["date"], y=eq["net_deposits"], mode="lines",
                                   name="Net deposits", line=dict(color="#607d8b", dash="dash")),
                        row=1, col=1,
                    )
                fig.add_trace(
                    go.Scatter(x=eq["date"], y=eq["drawdown_pct"], mode="lines",
                               name="Drawdown %", fill="tozeroy",
                               line=dict(color="#c62828", width=1)),
                    row=2, col=1,
                )
                fig.update_layout(height=470, margin=dict(l=10, r=10, t=45, b=10))
                fig.update_yaxes(title_text="₹", row=1, col=1)
                fig.update_yaxes(title_text="%", row=2, col=1)
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "The dashed line separates deposits/withdrawals from trading performance. "
                    "Snapshots are recorded on every order, capital event and agent cycle."
                )
            else:
                st.info("No equity history yet.")
        with tab5:
            performance = broker.strategy_performance()
            trades_for_chart = broker.trades(limit=100_000)
            if performance.empty:
                st.info("Close at least one trade to see strategy performance.")
            else:
                show = performance.copy()
                show["win_rate_pct"] = show["win_rate_pct"].round(1)
                show["net_pnl"] = show["net_pnl"].round(2)
                show["avg_pnl"] = show["avg_pnl"].round(2)
                show["profit_factor"] = show["profit_factor"].replace(np.inf, np.nan).round(2)
                st.dataframe(
                    show.rename(columns={
                        "strategy": "Strategy", "style": "Style", "trades": "Trades",
                        "wins": "Wins", "win_rate_pct": "Win rate %",
                        "net_pnl": "Net P&L ₹", "avg_pnl": "Avg P&L ₹",
                        "profit_factor": "Profit factor", "avg_hold_days": "Avg hold days",
                    }),
                    width="stretch", hide_index=True,
                )
                strategy_fig = go.Figure(go.Bar(
                    x=performance["strategy"], y=performance["net_pnl"],
                    marker_color=["#2e7d32" if value >= 0 else "#c62828"
                                  for value in performance["net_pnl"]],
                    text=[f"₹{value:+,.0f}" for value in performance["net_pnl"]],
                    textposition="outside",
                ))
                strategy_fig.update_layout(
                    title="Realized net P&L by strategy (all costs included)",
                    yaxis_title="Net P&L ₹", height=330,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(strategy_fig, width="stretch")

                trade_frame = pd.DataFrame(trades_for_chart)
                trade_frame["exit_date"] = pd.to_datetime(
                    trade_frame["exit_date"], utc=True, errors="coerce"
                )
                trade_frame = trade_frame.sort_values("exit_date")
                trade_frame["cumulative_pnl"] = trade_frame.groupby("strategy")["net_pnl"].cumsum()
                cumulative_fig = go.Figure()
                for strategy, group in trade_frame.groupby("strategy"):
                    cumulative_fig.add_trace(go.Scatter(
                        x=group["exit_date"], y=group["cumulative_pnl"],
                        mode="lines+markers", name=strategy,
                    ))
                cumulative_fig.update_layout(
                    title="Cumulative realized P&L history by strategy",
                    yaxis_title="Cumulative P&L ₹", height=360,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(cumulative_fig, width="stretch")

def page_auto_trade(broker: PaperBroker):
    st.title("🤖 AI Auto-Trade Agent")
    st.caption(
        "Automatically monitors accepted signals, opens risk-sized virtual positions, "
        "and closes them at stop, target, max-hold or intraday square-off."
    )
    st.warning(
        "**Paper execution only.** This page cannot place real broker orders. Prove the "
        "strategy for at least three months before considering a separately reviewed live adapter."
    )
    if not broker.account():
        st.info("Create your user-specific virtual account in **Paper Trading** first.")
        return

    agent = AutoTradeAgent(broker)
    cfg = agent.config()
    summary = broker.summary()
    status_color = "🟢" if cfg["enabled"] else "⚪"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Agent", f"{status_color} {'ARMED' if cfg['enabled'] else 'DISARMED'}")
    m2.metric("User equity", f"₹{summary['equity']:,.0f}")
    m3.metric("Available cash", f"₹{summary['cash']:,.0f}")
    m4.metric("Open positions", f"{summary['n_positions']}/{get('capital.max_open_positions', 3)}")

    all_strategies = StrategyRegistry.all()
    with st.form("agent_settings"):
        st.subheader("Agent guardrails")
        c1, c2 = st.columns(2)
        styles = c1.multiselect(
            "Allowed trading styles",
            ["swing", "invest", "intraday"],
            default=cfg["styles"],
            format_func=lambda value: STYLE_META[value][0],
        )
        strategies = c2.multiselect(
            "Allowed strategies",
            list(all_strategies),
            default=cfg["strategies"],
            format_func=lambda value: f"{value} ({all_strategies[value].style})",
        )
        c1, c2, c3 = st.columns(3)
        min_confidence = c1.slider(
            "Minimum confidence", 55.0, 90.0, float(cfg["min_confidence"]), 1.0
        )
        risk_pct = c2.slider(
            "Risk per trade (% equity)", 0.25, 1.0, float(cfg["risk_pct"]), 0.05
        )
        max_orders = c3.number_input(
            "Max entries per cycle", 1, 5, int(cfg["max_orders_per_cycle"])
        )
        c1, c2, c3 = st.columns(3)
        daily_loss = c1.slider(
            "Daily realized-loss stop (%)", 0.25, 5.0,
            float(cfg["max_daily_loss_pct"]), 0.25,
        )
        max_gap = c2.slider(
            "Max live-price gap from signal (%)", 0.25, 5.0,
            float(cfg["max_entry_gap_pct"]), 0.25,
        )
        from investriskfree.kronos_forecast import kronos_dependencies_available
        kronos_ready, kronos_message = kronos_dependencies_available()
        require_kronos = c3.checkbox(
            "Require Kronos forecast confirmation",
            value=bool(cfg["require_kronos"]),
            disabled=not kronos_ready,
            help=kronos_message,
        )
        desired_enabled = st.toggle("Arm paper auto-trading", value=bool(cfg["enabled"]))
        confirmation = st.text_input(
            "To arm, type ARM PAPER AGENT",
            value="" if desired_enabled and not cfg["enabled"] else "ARM PAPER AGENT",
            type="password",
        )
        save_agent = st.form_submit_button("Save guardrails", type="primary")
        if save_agent:
            if desired_enabled and not cfg["enabled"] and confirmation != "ARM PAPER AGENT":
                st.error("Type the exact confirmation phrase before arming the agent.")
            elif not styles or not strategies:
                st.error("Select at least one style and strategy.")
            else:
                cfg = agent.update_config(
                    enabled=desired_enabled,
                    styles=styles,
                    strategies=strategies,
                    min_confidence=min_confidence,
                    risk_pct=risk_pct,
                    max_orders_per_cycle=max_orders,
                    max_daily_loss_pct=daily_loss,
                    max_entry_gap_pct=max_gap,
                    require_kronos=require_kronos,
                )
                st.success(f"Agent is now {'ARMED' if cfg['enabled'] else 'DISARMED'}.")
                st.rerun()

    st.divider()
    run_col, data_col = st.columns([1, 2])
    live_data = data_col.toggle(
        "Use live yfinance data for this cycle",
        value=True,
        help=(
            "Live mode refuses to silently fall back to the bundled 2022 candles. "
            "yfinance is delayed and is not a broker-grade feed."
        ),
    )
    if not live_data:
        data_col.warning("Demo mode uses the bundled dataset ending in 2022; orders are historical simulations.")
    run_label = "Run agent cycle now" if cfg["enabled"] else "Preview cycle (no new entries)"
    if run_col.button(run_label, type="primary", width="stretch"):
        kronos_gate = None
        if cfg.get("require_kronos"):
            from investriskfree.kronos_forecast import KronosForecastService
            service = KronosForecastService()

            def kronos_gate(signal):
                frame = load_daily(
                    signal["symbol"], source="yfinance" if live_data else None
                )
                return service.long_signal_gate(signal, frame)
        with st.spinner("Monitoring positions and scanning for signals..."):
            result = agent.run_once(
                live_data=live_data, force=True, kronos_gate=kronos_gate
            )
        if result.get("ok"):
            st.success(result["message"])
        else:
            st.error(result["message"])
        cached_scan.clear()

    st.info(
        "**Continuous operation:** this page runs a cycle on demand. On an always-on host, run "
        "`python tools/run_agents.py --loop --interval 300 --live` to monitor every five minutes. "
        "Streamlit Community Cloud does not guarantee background workers."
    )

    run_tab, event_tab, performance_tab = st.tabs(
        ["🕒 Cycle history", "🧾 Decision audit", "📈 Auto-trade results"]
    )
    with run_tab:
        runs = agent.runs()
        if runs:
            st.dataframe(pd.DataFrame(runs), width="stretch", hide_index=True)
        else:
            st.info("No agent cycle has run yet.")
    with event_tab:
        events = agent.events()
        if events:
            event_frame = pd.DataFrame(events)
            event_frame["details"] = event_frame["details"].map(
                lambda value: json.dumps(value, default=str) if isinstance(value, dict) else str(value)
            )
            st.dataframe(
                event_frame[["date", "action", "status", "symbol", "strategy", "details"]],
                width="stretch", hide_index=True,
            )
        else:
            st.info("No decisions recorded yet.")
    with performance_tab:
        auto_trades = [trade for trade in broker.trades(limit=100_000) if trade.get("source") == "auto"]
        if not auto_trades:
            st.info("No closed auto-trades yet. Open entries remain visible in Paper Trading.")
        else:
            frame = pd.DataFrame(auto_trades)
            frame["exit_date"] = pd.to_datetime(frame["exit_date"], utc=True, errors="coerce")
            frame = frame.sort_values("exit_date")
            frame["cumulative_pnl"] = frame["net_pnl"].cumsum()
            cols = st.columns(4)
            cols[0].metric("Closed auto-trades", len(frame))
            cols[1].metric("Win rate", f"{(frame['net_pnl'] > 0).mean() * 100:.1f}%")
            cols[2].metric("Net P&L", f"₹{frame['net_pnl'].sum():+,.0f}")
            cols[3].metric("Costs", f"₹{frame['costs'].sum():,.0f}")
            fig = go.Figure(go.Scatter(
                x=frame["exit_date"], y=frame["cumulative_pnl"],
                mode="lines+markers", line=dict(color="#1565c0", width=2),
            ))
            fig.update_layout(
                title="Auto-agent cumulative realized P&L",
                yaxis_title="Net P&L ₹", height=350,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, width="stretch")


def page_backtest():
    st.title("🔬 Backtest Lab")
    st.caption("Realistic backtests with the full Indian cost model. This is where 'profit probability %' "
               "and 'duration' numbers come from — every signal on the dashboard is linked to these stats.")
    symbols = [u["symbol"] for u in get_universe()]
    if not symbols:
        st.warning("No universe - check data/bundled/nse_daily")
        return
    c1, c2, c3 = st.columns([2, 2, 1])
    symbol = c1.selectbox("Stock", symbols, index=symbols.index("RELIANCE") if "RELIANCE" in symbols else 0)
    strategies = {k: v for k, v in StrategyRegistry.all().items()}
    strat_choice = c2.selectbox(
        "Strategy", list(strategies),
        format_func=lambda k: f"{k} ({strategies[k].style})")
    run = c3.button("Run backtest", type="primary", width="stretch")
    if run:
        cached_backtest.clear()
    if symbol and strat_choice:
        with st.spinner("Running backtest..."):
            r = cached_backtest(symbol, strat_choice)
        s = r["stats"]
        style = r["style"]
        st.markdown(f"### {symbol} · {strat_choice} ({style})")
        cols = st.columns(6)
        cols[0].metric("Trades", s["trades"])
        cols[1].metric("Profit probability", f"{s['win_rate']:.1f}%")
        cols[2].metric("Expectancy/trade", f"{s['expectancy_pct']:+.2f}%")
        cols[3].metric("Total return", f"{s['total_return_pct']:+.1f}%")
        cols[4].metric("Max drawdown", f"{s['max_drawdown_pct']:.1f}%")
        cols[5].metric("Med duration", f"{s['median_hold_days']:.0f} days")
        cols = st.columns(6)
        cols[0].metric("Profit factor", f"{s['profit_factor']:.2f}")
        cols[1].metric("Sharpe", f"{s.get('sharpe', 0):.2f}")
        cols[2].metric("Avg win", f"{s['avg_win_pct']:+.2f}%")
        cols[3].metric("Avg loss", f"{s['avg_loss_pct']:+.2f}%")
        cols[4].metric("Costs paid", f"₹{s['total_costs']:,.0f}")
        cols[5].metric("P(up in 6m)", f"{s.get('p_up_6m', float('nan')):.0f}%")
        st.caption(f"Regime-filtered signals skipped: {r['filtered']}. "
                   "Demo-data note: intraday uses synthetic 5m bars from real daily data (sandbox); "
                   "real 5m bars available via yfinance on your machine.")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                            subplot_titles=("Equity curve (₹100k start, after all costs)", "Drawdown"))
        eq = r["equity"]
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Equity",
                                 line=dict(color="#2e7d32", width=2)), row=1, col=1)
        dd = (eq / eq.cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="DD",
                                 fill="tozeroy", line=dict(color="#c62828", width=1)), row=2, col=1)
        fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")

        trades = r["trades"]
        if len(trades):
            st.subheader(f"Trade list ({len(trades)})")
            t = trades[["entry_date", "exit_date", "entry_price", "exit_price",
                        "net_pnl_pct", "hold_days", "reason"]].copy()
            for c in ("entry_price", "exit_price", "net_pnl_pct"):
                t[c] = t[c].round(2)
            st.dataframe(t.sort_values("entry_date", ascending=False),
                         width="stretch", hide_index=True)
            # duration histogram
            fig2 = go.Figure(go.Histogram(x=trades["hold_days"], nbinsx=20))
            fig2.update_layout(title="Trade duration distribution (days)",
                               height=280, margin=dict(l=10, r=10, t=40, b=10),
                               xaxis_title="Holding period (days)", yaxis_title="Trades")
            st.plotly_chart(fig2, width="stretch")


def page_research():
    st.title("🧠 Strategy Research & AI Brain")
    st.caption("Walk-forward ML evaluation and honest strategy comparisons.")
    from investriskfree.scanner import load_stats_registry
    reg = load_stats_registry()
    st.subheader("Strategy registry (built from real NSE backtests)")
    if reg.get("meta"):
        st.caption(f"Built: {reg['meta']['built_at']} · {reg['meta']['data_source']}")
    for name, d in reg.items():
        if name in ("meta", "by_pair") or not isinstance(d, dict) or "win_rate" not in d:
            continue
        with st.container(border=True):
            style = StrategyRegistry.get(name).style if name in StrategyRegistry.all() else "?"
            st.markdown(f"**{name}** — {STYLE_META.get(style, ('', ''))[0]} · {STRATEGY_INFO.get(name, '')}")
            c = st.columns(8)
            c[0].metric("P(profit)", f"{d['win_rate']:.1f}%")
            c[1].metric("Expectancy", f"{d['expectancy_pct']:+.2f}%")
            c[2].metric("Med duration", f"{d['median_hold_days']:.0f}d")
            c[3].metric("Max DD", f"{d['max_drawdown_pct']:.1f}%")
            c[4].metric("Profit factor", f"{d['profit_factor']:.2f}")
            c[5].metric("Sharpe", f"{d.get('sharpe', 0):.2f}")
            c[6].metric("P(up 3m)", f"{d.get('p_up_3m', 0):.0f}%")
            c[7].metric("Positive stocks", f"{d.get('positive_stocks', '?')}/{d.get('n_stocks', '?')}"
                        if "positive_stocks" in d else "—")
            if d.get("demo_data"):
                st.caption("⚠️ These intraday numbers come from synthetic 5m bars (demo). "
                           "Rebuild with real 5m data on your machine.")

    st.divider()
    st.subheader("🤖 Walk-Forward ML gate")
    st.markdown(
        "The ML layer trains a logistic-regression win-probability model on past years "
        "and tests it strictly on future years it never saw (walk-forward). It never overrides "
        "risk rules — it only blocks trades where predicted P(win) < historical baseline.")
    try:
        import json
        with open(os.path.join(get("data.repo_root"), "data", "ml_report.json")) as f:
            mlr = json.load(f)
        if mlr.get("meta"):
            st.caption(f"Built: {mlr['meta']['built_at']} · {mlr['meta']['symbols']} symbols · "
                       f"{mlr['meta']['splits']} chronological folds")
        ml_rows = []
        for name, d in mlr.items():
            if name == "meta" or not isinstance(d, dict) or not d.get("trained"):
                continue
            ml_rows.append({
                "Strategy": name,
                "OOS accuracy": f"{d['accuracy']:.1f}%",
                "Baseline win rate": f"{d['baseline_win_rate']:.1f}%",
                "Precision (when says win)": f"{d['precision_when_win']:.1f}%",
                "Lift": f"{d['lift']:.2f}x",
                "OOS trades": d["n_oos"],
            })
        if ml_rows:
            st.dataframe(pd.DataFrame(ml_rows), width="stretch", hide_index=True)
            st.info(
                "Read precision and lift strategy by strategy. A lift below 1.0 means the model "
                "did not improve on the baseline and should not be trusted as an edge."
            )
        else:
            st.info("Run `python tools/train_ml.py` to produce the report and deployable models.")
    except FileNotFoundError:
        st.info("Run `python tools/train_ml.py` to produce `data/ml_report.json`.")
    st.caption(
        "The scanner uses persisted model coefficients from `data/ml_models.json`. "
        "If a model is unavailable, the signal explicitly reports ML unavailable instead of "
        "pretending the gate ran."
    )

    st.divider()
    st.subheader("🔭 Kronos candlestick foundation-model forecast (optional)")
    st.markdown(
        "Kronos tokenizes OHLCV candles and autoregressively forecasts future candles. "
        "InvestRiskFreeAI uses it only as an optional confluence gate—never as a direct order "
        "generator and never as a replacement for stops, sizing or out-of-sample backtests."
    )
    from investriskfree.kronos_forecast import (
        KronosForecastService,
        kronos_dependencies_available,
    )
    kronos_ready, kronos_status = kronos_dependencies_available()
    if not kronos_ready:
        st.info(f"{kronos_status}. Install with `pip install -r requirements-kronos.txt`.")
    symbols = [item["symbol"] for item in get_universe()]
    kc1, kc2, kc3 = st.columns([2, 1, 1])
    kronos_symbol = kc1.selectbox("Forecast stock", symbols, key="kronos_symbol")
    kronos_horizon = kc2.slider("Future daily bars", 3, 20, 5)
    kronos_live = kc3.toggle("Live history", value=False, key="kronos_live")
    if st.button("Run Kronos forecast", disabled=not kronos_ready):
        try:
            history = load_daily(
                kronos_symbol, source="yfinance" if kronos_live else None
            )
            with st.spinner("Loading Kronos-mini and forecasting candles..."):
                forecast = KronosForecastService().forecast(
                    kronos_symbol, history, horizon=kronos_horizon
                )
            metrics = st.columns(4)
            metrics[0].metric("Direction", forecast.direction)
            metrics[1].metric("Forecast return", f"{forecast.expected_return_pct:+.2f}%")
            metrics[2].metric("Forecast low", f"{forecast.forecast_low_pct:+.2f}%")
            metrics[3].metric("Forecast high", f"{forecast.forecast_high_pct:+.2f}%")
            chart = go.Figure()
            recent = history.tail(100)
            chart.add_trace(go.Scatter(
                x=recent.index, y=recent["Close"], name="Historical close",
                line=dict(color="#607d8b"),
            ))
            chart.add_trace(go.Scatter(
                x=forecast.forecast.index, y=forecast.forecast["close"],
                name="Kronos forecast close", line=dict(color="#7b1fa2", width=2),
            ))
            chart.add_trace(go.Scatter(
                x=forecast.forecast.index, y=forecast.forecast["high"],
                name="Forecast high", line=dict(width=0), showlegend=False,
            ))
            chart.add_trace(go.Scatter(
                x=forecast.forecast.index, y=forecast.forecast["low"],
                name="Forecast range", fill="tonexty",
                line=dict(width=0), fillcolor="rgba(123,31,162,0.15)",
            ))
            chart.update_layout(
                title=f"{kronos_symbol}: history and {kronos_horizon}-bar Kronos forecast",
                yaxis_title="Price ₹", height=430,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(chart, width="stretch")
            st.warning("Forecast candles are uncertain model outputs, not profit probabilities.")
        except Exception as exc:
            st.error(f"Kronos forecast failed: {exc}")


def page_guide():
    st.title("📚 Guide — How this system protects your capital")
    st.markdown("""
### The 5 capital-protection rules (non-negotiable)
1. **Risk 0.5–1% of capital per trade.** A losing streak of 10 trades costs ~5–10%,
   never 50%. This is why you still have money to trade tomorrow.
2. **Never trade without a stop-loss.** Every signal ships with one. If the price
   doesn't behave, you pay a tiny fixed price and move on.
3. **Never average down.** Adding to a losing position is how accounts die.
4. **Paper trade first.** You are doing this now. Real money only after 3+ months
   of consistent virtual profit.
5. **Skip the trade when unsure.** The brain blocks signals below 55/100 confidence.
   Sitting in cash is a position.

### The strategies (all long-only, all backtested)
| Strategy | What it does | Typical duration | P(profit)* |
|---|---|---|---|
| TrendRider (swing) | Pullback buy in EMA20/50 uptrend, ADX filter | 3–15 days | ~42% |
| RangeBreaker (swing) | 20-day Donchian breakout + volume | 5–25 days | ~40% |
| TrendDip (invest) | Buy dips in stocks above rising 200SMA | 30–120 days | ~46% |
| ORB (intraday) | 9:15–9:30 range breakout, square off 15:25 | < 1 day | ~55–67% (demo) |
| DipBuyer / VWAP | Lab experiments — **disabled**, edge ≤ 0 after costs | — | — |

*From real NSE 2012–2022 backtests with full costs, 0.5% risk per trade, 100k capital.
Intraday numbers are synthetic-5m demo data until you rebuild with live 5m bars.

### Why win rate < 50% can still make money
TrendRider wins ~42% of trades but targets 1:1.6 R:R. Math: `0.42×1.6 − 0.58×1 = +0.09R`
per trade, minus costs ≈ +0.6% per trade on average in the backtest. **The system makes
money from risk-reward, not from being right often.**

### What you need for real-time data (on your machine)
- Default: `yfinance` — free, NSE data delayed ~15 min. Enough for swing/invest.
- Better: Zerodha Kite Connect (₹2,000/mo) or Angel One SmartAPI (free) for live
  ticks + order placement. The code has a clean data layer to plug either in.

### Growth plan (from ₹1k–5k)
| Capital | Realistic role | Notes |
|---|---|---|
| ₹1,000–5,000 | Learn + practice | costs eat 1–2% per trade; trade tiny or wait |
| ₹10,000–25,000 | Small real positions | 0.5% risk = ₹50–125 per trade risk |
| ₹50,000+ | Full system | proper diversification, 3 strategies at once |

### Disclosures
- Nothing here is SEBI-registered investment advice. Markets can lose money.
- Backtested results ≠ future results. The cost model is realistic but not perfect.
- The ML model is a gate, not a crystal ball. It cannot predict crashes or news.
- **After you're profitable, share the system and mentor others — that's the mission.**
""")
    st.success("Mission: first make yourself profitable with protected capital, "
               "then help others who lost money in trading.")


def _start_user_session(user: dict) -> None:
    st.session_state["authenticated"] = True
    st.session_state["user_id"] = user["id"]
    st.session_state["username"] = user["username"]
    st.session_state["display_name"] = user.get("display_name") or user["username"]


def require_login() -> bool:
    """Password-hashed sign-in/sign-up gate with isolated user ledgers."""
    auth = AuthStore()
    if st.session_state.get("authenticated", False):
        user_id = st.session_state.get("user_id")
        if user_id and auth.get_user(user_id):
            return True
        for key in ("authenticated", "user_id", "username", "display_name"):
            st.session_state.pop(key, None)

    st.markdown("""
        <style>
        [data-testid="stSidebar"] { display: none; }
        </style>
    """, unsafe_allow_html=True)
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("## 🛡️ InvestRiskFreeAI")
        st.markdown("#### Your private AI paper-trading workspace")
        st.caption(
            "Create an account to keep your capital, positions, agent settings and history "
            "separate from every other user."
        )
        login_tab, signup_tab = st.tabs(["🔐 Log in", "✨ Sign up"])
        with login_tab:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="your username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button(
                    "Log in", type="primary", use_container_width=True
                )
            if submit:
                result = auth.authenticate(username, password)
                if not result["ok"]:
                    # Deployment-secret users are optional and must match both
                    # a configured username and password. There are no hardcoded
                    # universal demo credentials.
                    secret_user = os.environ.get("ADMIN_USERNAME", "admin")
                    secret_password = os.environ.get("ADMIN_PASSWORD")
                    try:
                        if "auth" in st.secrets:
                            secret_user = str(st.secrets["auth"].get("username", secret_user))
                            secret_password = str(st.secrets["auth"].get("password", ""))
                    except Exception:
                        pass
                    external_valid = bool(
                        secret_password
                        and username.strip().lower() == secret_user.strip().lower()
                        and hmac.compare_digest(password, secret_password)
                    )
                    if external_valid:
                        try:
                            user = auth.ensure_external_user(secret_user)
                            result = {"ok": True, "user": user}
                        except ValueError as exc:
                            result = {"ok": False, "error": str(exc)}
                if result["ok"]:
                    _start_user_session(result["user"])
                    st.rerun()
                else:
                    st.error(result["error"])
            st.caption(
                "Deployment administrators can configure `[auth] username` and `password` "
                "in Streamlit Secrets. No default password is embedded in the source."
            )

        with signup_tab:
            with st.form("signup_form", clear_on_submit=False):
                display_name = st.text_input("Display name", placeholder="How should we greet you?")
                new_username = st.text_input(
                    "Choose a username", placeholder="3–32 letters, numbers, dot, dash or underscore"
                )
                email = st.text_input("E-mail", placeholder="you@example.com")
                new_password = st.text_input(
                    "Choose a password", type="password",
                    help="At least 10 characters with a letter and a number.",
                )
                confirm_password = st.text_input("Confirm password", type="password")
                accepted = st.checkbox(
                    "I understand this is paper-trading software, not financial advice."
                )
                register = st.form_submit_button(
                    "Create my account", type="primary", use_container_width=True
                )
            if register:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif not accepted:
                    st.error("Please accept the paper-trading disclosure.")
                else:
                    result = auth.register(
                        new_username, email, new_password, display_name=display_name
                    )
                    if result["ok"]:
                        _start_user_session(result["user"])
                        st.success("Account created. Your private workspace is ready.")
                        st.rerun()
                    else:
                        st.error(result["error"])
        st.divider()
        st.caption(
            "Passwords use salted PBKDF2 hashing. Market trading is never risk-free; "
            "start with virtual capital."
        )
    return False


def main():
    if not require_login():
        return

    broker = current_broker()
    st.sidebar.title("🛡️ InvestRiskFreeAI")
    st.sidebar.caption("v0.2 · NSE India · Capital-first")

    display_name = st.session_state.get("display_name", st.session_state.get("username", "user"))
    st.sidebar.caption(f"👤 Signed in as **{display_name}**")
    if st.sidebar.button("🚪 Log Out", use_container_width=True):
        for key in ("authenticated", "user_id", "username", "display_name"):
            st.session_state.pop(key, None)
        st.rerun()
    st.sidebar.divider()

    page = st.sidebar.radio("Navigate", [
        "🏠 Dashboard", "💰 Paper Trading", "🤖 Auto-Trade Agent",
        "🔬 Backtest Lab", "🧠 Strategy Research", "📚 Guide",
    ])
    st.sidebar.divider()
    try:
        summary = broker.summary()
        if summary:
            st.sidebar.metric(
                "My virtual equity", f"₹{summary['equity']:,.0f}",
                delta=f"₹{summary['total_pnl']:+,.0f} P&L",
            )
            agent_cfg = AutoTradeAgent(broker).config()
            st.sidebar.caption(
                f"Agent: {'🟢 ARMED' if agent_cfg['enabled'] else '⚪ disarmed'} · paper only"
            )
        else:
            st.sidebar.info("No paper account yet — create one in Paper Trading.")
    except Exception:
        pass
    st.sidebar.caption("Offline: bundled NSE daily 2012–2022 · Live: yfinance (delayed)")
    if page.startswith("🏠"):
        page_dashboard(broker)
    elif page.startswith("💰"):
        page_paper(broker)
    elif page.startswith("🤖"):
        page_auto_trade(broker)
    elif page.startswith("🔬"):
        page_backtest()
    elif page.startswith("🧠"):
        page_research()
    else:
        page_guide()


main()
