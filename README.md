# 🛡️ InvestRiskFreeAI

**Capital-protection-first, AI-assisted trading system for the Indian (NSE) stock market.**
Swing trading, intraday, and investment styles — every signal comes with its own
backtested *profit probability %* and *expected duration*, you paper-trade with
virtual money first, and hard risk rules make sure your capital survives.

> 📖 **New here? Read the [step-by-step USER GUIDE](USER_GUIDE.md) first** — it
> walks you from opening the app to your first paper trade in ~10 minutes.
>
> 🌐 **Try it live now:** 👉 **https://investriskfreeai-goroy.streamlit.app/**
> (deployed on Streamlit Community Cloud — works from any browser, no setup).
> See **[DEPLOY.md](DEPLOY.md)** to re-deploy or update it.
>
> 🔒 **Secure Login (Demo Credentials):** Username: `admin` | Password: `admin123` (or `investriskfree`).
> You can also configure your own private credentials in Streamlit Cloud Secrets.

> ⚠️ **Honest words up front (read this twice):** there is no such thing as a
> risk-free return in markets. Anyone promising one is lying to you. What this
> system actually does — and what the user asked for — is:
> **protect capital first** (0.5% risk per trade, always a stop-loss, market-regime
> filter), **validate everything with realistic backtests**, and **paper-trade
> before real money**. It is not SEBI-registered investment advice.

---

## 🎯 The mission (in the user's own words)

1. Start small (₹1,000–5,000), grow the capital step by step.
2. **Protect capital at all costs** — most of the money was already lost in options.
3. Everything is AI-based signals with backtested profit-probability % and duration.
4. Test with **virtual money first**, then real money.
5. Realtime data access + an intelligent decision brain.
6. **After we become profitable, help others who lost money in trading.**

The system is built in exactly that order of priorities.

---

## 🏗️ What was built (this repo)

```
InvestRiskFreeAI/
├── app.py                        # Streamlit app (Dashboard, Paper Trading, Backtest Lab, Research, Guide)
├── config.yaml                   # ALL risk & strategy settings in one file
├── requirements.txt
├── investriskfree/
│   ├── backtest.py               # Event-driven backtester + full Indian cost model
│   ├── brain.py                  # AI Brain: regime, confluence confidence, position sizing
│   ├── ml.py                     # Walk-forward ML win-probability gate (logistic regression)
│   ├── scanner.py                # Universe scanner -> ranked, risk-gated signals
│   ├── paper.py                  # Paper-trading broker (virtual money, SQLite)
│   ├── indicators.py             # Pure-pandas indicators (EMA, RSI, ATR, ADX, VWAP, ...)
│   ├── strategies/               # 6 strategies (3 swing, 2 intraday, 1 invest)
│   └── data/                     # data loaders (yfinance live | NSE | bundled offline)
├── data/
│   ├── bundled/nse_daily/        # REAL NSE daily data, 50 stocks × 2012–2022 (offline demo)
│   ├── strategy_stats.json       # backtested P(profit), duration, expectancy per strategy
│   └── ml_report.json            # walk-forward out-of-sample ML results
├── tools/
│   ├── build_stats.py            # rebuild the performance registry
│   └── train_ml.py               # train + evaluate the ML gate
└── tests/                        # smoke tests (pytest)
```

**Data note:** this sandbox ships 10 years of *real NSE daily data* (Nifty-50
stocks, 2012–2022) so everything works offline. On your machine the data layer
can pull **fresh live data** via `yfinance` (free, ~15 min delayed — fine for
swing/invest) and real **5-minute intraday bars**. For true realtime + auto
trading, plug in Zerodha Kite Connect / Angel One SmartAPI via the clean data
interface.

---

## 🚀 Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Web app — signals, paper trading, backtest lab
streamlit run app.py

# 2) CLI — today's signals (offline bundled data)
python -m investriskfree scan --capital 100000

# 2b) CLI — today's signals with live yfinance Last Price & vs-Entry gap %
python -m investriskfree scan --capital 100000 --live

# 3) CLI — backtest a stock + strategy
python -m investriskfree backtest RELIANCE swing_trend --trades

# 4) Paper account (virtual money)
python -m investriskfree paper new 100000
python -m investriskfree paper summary

# 5) Rebuild performance registry & ML report (after adding live data)
python tools/build_stats.py
python tools/train_ml.py

# 6) Tests
python -m pytest tests/ -q
```

---

## 🧠 The AI Brain (how signals are gated)

Every raw strategy signal passes through four protection layers before it can
reach your paper account:

| Layer | What it does | Config |
|---|---|---|
| **1. Market regime** | If the market is below its 200-SMA and breadth is bad (RISK-OFF), longs are **blocked**. Staying in cash is a position. | `brain.regime.*` |
| **2. Confidence score** | 0–100 confluence of trend, momentum, volatility-fit, liquidity, volume trend, regime. `< 55` = **no trade**. `≥ 70` = STRONG. | `brain.min_confidence`, `brain.strong_confidence` |
| **3. ML gate** | A logistic-regression model trained on past years, tested **strictly out-of-sample** on future years, blocks trades with predicted P(win) below baseline. | `brain.ml.*` |
| **4. Position sizing** | Size comes from the risk budget (0.5% of capital), never from greed. Max 25% of capital per position. | `capital.*` |

The ML gate's honest out-of-sample numbers (10 Nifty stocks, 3 chronological folds):

| Strategy | OOS accuracy | Baseline win rate | Precision when "win" | Lift |
|---|---|---|---|---|
| swing_trend | 63.6% | 42.8% | 59.9% | **1.40×** |
| swing_breakout | 59.6% | 38.0% | 45.5% | **1.20×** |
| invest | 65.5% | 36.7% | 52.6% | **1.43×** |

The model is a *filter*, not a crystal ball — it cannot predict crashes or news.

---

## 📊 The strategies and their REAL backtest numbers

Backtests: real NSE daily data 2012–2022, 50 stocks, ₹1,00,000 capital,
0.5% risk/trade, full cost model (brokerage + STT + exchange + GST + SEBI +
stamp + 0.05% slippage), regime-filtered. **Costs are always deducted.**

| Strategy | Style | Logic | P(profit) | Expectancy/trade | Med duration | Max DD | Status |
|---|---|---|---|---|---|---|---|
| **TrendRider** (`swing_trend`) | Swing | Pullback to EMA20 in EMA20>EMA50 uptrend, ADX>18, RSI<72. Stop 2.5×ATR, target 4×ATR, exit on EMA50 break | **41.5%** | **+0.63%** | ~5 d | −2.8% | ✅ enabled |
| **TrendDip** (`invest`) | Invest | Buy dips (≤5% off 20d high) in stocks above a rising 200SMA | **46.2%** | **+2.65%** | ~35 d | −4.0% | ✅ enabled |
| **RangeBreaker** (`swing_breakout`) | Swing | Donchian-20 breakout + 1.5× volume + RSI>55. Stop 3×ATR, target 4×ATR | **39.9%** | **+0.32%** | ~14 d | −2.3% | ✅ enabled |
| **ORB** (`intraday_orb`) | Intraday | 9:15–9:30 opening-range breakout, entries till 12:00, stop 1.2×ATR, target 2.2×ATR, **forced square-off 15:25** | **47.1%** | **+0.25%** | < 1 d | −2.5% | ⚠️ demo data |
| DipBuyer (`swing_meanrev`) | Swing | RSI(2)<15 oversold bounce | 52% | ~0 after costs | — | — | 🔬 lab only |
| VWAP fade (`intraday_vwap`) | Intraday | Fade VWAP deviations | 43% | **negative** | — | — | 🔬 lab only |

**Why win rate < 50% still makes money:** the system profits from risk-reward,
not from being right often. TrendRider wins ~41% at ~1:1.6 R:R →
`0.41 × 1.6 − 0.59 × 1 = +0.07R` per trade after costs. Small, repeated, capped.

> ⚠️ Intraday numbers come from **synthetic 5-minute bars** (a Brownian bridge
> seeded from real daily OHLC) because this sandbox cannot reach a real intraday
> feed. On your machine, set `real_intraday=True` in the scanner and rebuild with
> **real 5-minute bars** from yfinance/Kite before trusting intraday trades.

---

## 💰 Paper trading flow (use this before real money)

1. **Create a virtual account** in the Paper Trading page (default ₹1,00,000).
2. **Scan** → signals appear with entry, stop, target, confidence, P(profit), duration.
3. **Buy** → the system computes the position size from your risk budget and charges
   the full real cost model, so your virtual results include brokerage/STT/slippage.
4. **Monitor** → stop-loss and target are tracked; intraday positions must be
   squared off at 15:25 (button + auto rule in backtests).
5. **Sell / journal** → every trade is recorded with its P&L and reason.

**Rule of thumb:** go to real money only after **3+ consecutive months of
consistent virtual profit** (not luck — the journal proves it).

---

## 🔌 Realtime data & live trading (on your machine)

| Source | Cost | Latency | Use for |
|---|---|---|---|
| yfinance (default) | Free | ~15 min delayed | Swing + invest signals, 5m intraday bars |
| Zerodha Kite Connect | ~₹2,000/mo | Realtime ticks + orders | Full live system, auto stops (GTT) |
| Angel One SmartAPI | Free | Realtime | Full live system |
| Upstox / Fyers | Free | Realtime | Full live system |

The data layer (`investriskfree/data/loader.py`) is the single place to plug a
broker SDK in. Strategy signals are pure price/volume math, so they work on any
feed. Live execution is intentionally **not** wired to real money in v0.1 — the
paper broker is the only executor until the user has a proven record.

---

## 🛡️ The 5 non-negotiable capital-protection rules

1. **Risk 0.5% of capital per trade** (max 1%). Ten losses in a row = −5%, not −50%.
2. **Every signal ships with a stop-loss.** No stop = no trade.
3. **Never average down** a losing position.
4. **Paper trade first**, real money only after 3+ profitable virtual months.
5. **Skip when unsure.** Confidence < 55 is an automatic no. Cash is a position.

---

## 🗺️ Roadmap

- [x] Realistic backtester with full Indian cost model
- [x] 6 strategies across swing / intraday / invest
- [x] AI Brain: regime + confidence + walk-forward ML gate
- [x] Paper trading with virtual money + trade journal
- [x] 10 years of real NSE data bundled for offline demo
- [x] Live data mode (yfinance) — works on your machine, one toggle
- [x] Signals: real-time yfinance Last Price + vs-Entry gap %
- [ ] Kite Connect / SmartAPI broker adapters (live execution)
- [ ] Telegram signal alerts
- [ ] Multi-strategy portfolio backtest (capital allocation across strategies)
- [ ] Community mode: share anonymized results, mentor losers → **the mission**

---

## 🙏 Inspiration & references

- **[optionxi](https://github.com/optionxi)** — open-source virtual trading
  platform for India ("practice before you risk") — the paper-first philosophy
  this system shares.
- **[OpenByteInc/QuantDinger](https://github.com/OpenByteInc/QuantDinger)** —
  multi-agent AI quant platform (backtesting, live trading, market data,
  research agents) — the architecture inspiration for signal + backtest + data
  separation.

Both projects reinforce the same honest message: **validate, simulate, then
risk small.**

---

## ⚖️ Disclaimer

This is an educational/quant research project. Nothing here is financial advice.
Past performance (backtested or real) does not guarantee future results. Trading
involves risk of loss. Do not trade money you cannot afford to lose. If you are
in India, consider consulting a SEBI-registered investment advisor before
trading real money.
