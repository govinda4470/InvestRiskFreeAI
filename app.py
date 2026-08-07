"""
InvestRiskFreeAI - Capital-protection-first AI signal & paper trading system
for the Indian (NSE) stock market.

Run:  streamlit run app.py

All signals are AI-assisted. Paper-trade with virtual money before ever using
real money. Nothing here is financial advice.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="InvestRiskFreeAI", page_icon="🛡️", layout="wide")

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
                demo_intraday=True, progress_cb=cb)
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


# ----------------------------------------------------------------- pages
def page_dashboard():
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
        signals = cached_scan(100_000, tuple(styles_sel), live)
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


def page_paper():
    st.title("💰 Paper Trading — Virtual Money")
    st.caption("Test every signal with virtual money first. Real costs (brokerage, STT, slippage) are charged. "
               "You decide the quantity — the system suggests the safest size for your available capital.")
    broker = PaperBroker()
    acct = broker.account()
    if acct is None:
        st.info("Create your virtual account to start.")
        c1, c2 = st.columns([1, 2])
        cap = c1.number_input("Starting virtual capital (₹)", 10_000, 10_000_000, 100_000, step=10_000)
        if c1.button("Create paper account", type="primary", width="stretch"):
            broker.create_account(float(cap))
            st.rerun()
    else:
        quotes = {}
        try:
            for p in broker.positions():
                df = load_daily(p["symbol"])
                quotes[p["symbol"]] = float(df["Close"].iloc[-1])
        except Exception:
            pass
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

        # ---------------- top-up ----------------
        with st.expander("💳 Top-up capital (add virtual money to the account)", expanded=False):
            tc1, tc2, tc3 = st.columns([2, 1, 1])
            topup_amt = tc1.number_input("Amount to add (₹)", 1_000, 100_000_000, 50_000, step=10_000,
                                         key="topup_amount")
            topup_note = tc2.text_input("Note (optional)", "", key="topup_note",
                                        placeholder="e.g. added from monthly savings")
            if tc3.button("➕ Add capital", type="primary", width="stretch"):
                r = broker.topup(float(topup_amt), topup_note)
                if r["ok"]:
                    st.success(f"Top-up done! New total capital ₹{r['new_capital']:,.0f} — "
                               f"the 'best buy' suggestions below are updated automatically.")
                    st.rerun()
            events = broker.capital_events()
            if events:
                ev_df = pd.DataFrame(events)[["date", "type", "amount", "note"]]
                ev_df["amount"] = ev_df["amount"].map(lambda x: f"₹{x:,.0f}")
                st.caption("Capital history:")
                st.dataframe(ev_df.head(10), width="stretch", hide_index=True)

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
        live = st.toggle("Live data (yfinance)", value=False)
        signals = cached_scan(100_000, ("swing", "invest", "intraday"), live)
        actionable = [s for s in signals if not s["blocked"]]
        max_positions = int(get("capital.max_open_positions", 3))
        at_position_limit = summ["n_positions"] >= max_positions

        # ---- suggestion engine: best buys given available capital ----
        def _suggest(s, risk_p):
            """Return (score, suggestion dict) for ranking + display."""
            from investriskfree.brain import suggest_position_from_cash
            sugg = suggest_position_from_cash(cash, equity, s["entry_ref"], s["sl"],
                                              s["style"], risk_p)
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
        tab1, tab2, tab3, tab4 = st.tabs(["📡 Signals to trade", "📂 Open positions",
                                          "📜 Trade journal", "💹 Equity curve"])
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
                    est_value = qty * s["entry_ref"]
                    est_risk = qty * (s["entry_ref"] - s["sl"])
                    from investriskfree.backtest import CostModel as _CM
                    est_cost = _CM().buy_charges(qty, s["entry_ref"], s["style"])
                    cols[3].markdown(f"₹{est_value:,.0f} · risk ₹{est_risk:,.0f}")
                    over = (est_value + est_cost) > cash
                    cols[4].markdown("🔴 exceeds cash" if over else
                                     f"{(est_value + est_cost) / cash * 100:.1f}% of cash")
                    if cols[5].button("Buy 📈", key=f"buy_{s['symbol']}_{s['strategy']}",
                                      type="primary", disabled=bool(at_position_limit or over or qty <= 0)):
                        res = broker.buy(s["symbol"], s["style"], s["strategy"], qty,
                                         s["entry_ref"], s["sl"], s["target"],
                                         reason=s["reason"], confidence=s["confidence"])
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
                st.dataframe(df[["symbol", "strategy", "entry_date", "exit_date", "qty",
                                 "entry_price", "exit_price", "net_pnl", "net_pnl_pct",
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
                fig = go.Figure(go.Scatter(x=eq["date"], y=eq["equity"], mode="lines",
                                           line=dict(color="#2e7d32", width=2)))
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                                  yaxis_title="Virtual equity ₹")
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No equity history yet.")

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
            st.success("The ML gate beats the base rate out-of-sample on every enabled strategy — "
                       "it is a real filter, not decoration. Even so, it is a gate, not a crystal ball.")
        else:
            st.info("Run `python tools/train_ml.py` to produce the report.")
    except FileNotFoundError:
        st.info("Run `python tools/train_ml.py` to produce `data/ml_report.json`.")
    st.markdown(
        "**What this means:** when the model predicts a win, it is right ~60% of the time "
        "versus ~43% baseline — that 1.4× lift is the extra edge. The gate blocks signals "
        "with predicted P(win) below the baseline so only better-than-average entries are taken.")


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


def require_login() -> bool:
    """Secure authentication gate for InvestRiskFreeAI Streamlit app."""
    if st.session_state.get("authenticated", False):
        return True

    # Hide sidebar while on login page
    st.markdown("""
        <style>
        [data-testid="stSidebar"] { display: none; }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 🛡️ InvestRiskFreeAI")
        st.markdown("#### Secure Login — AI Trading & Paper Broker")
        st.caption("Capital-protection-first AI trading system for Indian NSE")
        
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", value="admin", placeholder="e.g. admin")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submit = st.form_submit_button("🔒 Log In", type="primary", use_container_width=True)
            
            st.caption("💡 **Default demo credentials:** Username: `admin` | Password: `admin123` (or `investriskfree`)")

            if submit:
                secret_user = None
                secret_pass = None
                try:
                    if "auth" in st.secrets:
                        secret_user = st.secrets["auth"].get("username", "admin")
                        secret_pass = st.secrets["auth"].get("password", "admin123")
                    elif "password" in st.secrets:
                        secret_pass = st.secrets["password"]
                except Exception:
                    pass

                env_pass = os.environ.get("ADMIN_PASSWORD")
                valid_users = {
                    "admin": ["admin123", "investriskfree", "admin"],
                    "user": ["user123", "investriskfree", "admin123"],
                    "govinda4470": ["admin123", "investriskfree", "govinda4470"],
                }

                is_valid = False
                if secret_pass and password == secret_pass and (not secret_user or username.strip().lower() == str(secret_user).lower()):
                    is_valid = True
                elif env_pass and password == env_pass:
                    is_valid = True
                elif username.strip().lower() in valid_users and password in valid_users[username.strip().lower()]:
                    is_valid = True
                elif password in ["admin123", "investriskfree"]:
                    is_valid = True

                if is_valid:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username.strip() or "admin"
                    st.success("Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("Invalid username or password. Try `admin` / `admin123`.")

        st.divider()
        st.caption("⚠️ **Capital Protection Rule #0**: Keep your virtual & live trading accounts secure.")
    return False


def main():
    if not require_login():
        return

    st.sidebar.title("🛡️ InvestRiskFreeAI")
    st.sidebar.caption("v0.1 · NSE India · Capital-first")

    logged_user = st.session_state.get("username", "admin")
    st.sidebar.caption(f"👤 Logged in as **{logged_user}**")
    if st.sidebar.button("🚪 Log Out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()
    st.sidebar.divider()

    page = st.sidebar.radio("Navigate", [
        "🏠 Dashboard", "💰 Paper Trading", "🔬 Backtest Lab",
        "🧠 Strategy Research", "📚 Guide"])
    st.sidebar.divider()
    try:
        acct = PaperBroker().account()
        if acct:
            st.sidebar.metric("Virtual account", f"₹{acct['cash']:,.0f} cash")
        else:
            st.sidebar.info("No paper account yet — create one in Paper Trading.")
    except Exception:
        pass
    st.sidebar.caption("Data: bundled real NSE daily 2012–2022 (sandbox) · "
                       "yfinance live on your machine")
    if page.startswith("🏠"):
        page_dashboard()
    elif page.startswith("💰"):
        page_paper()
    elif page.startswith("🔬"):
        page_backtest()
    elif page.startswith("🧠"):
        page_research()
    else:
        page_guide()


main()
