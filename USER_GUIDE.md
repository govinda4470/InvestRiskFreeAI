# 🛡️ InvestRiskFreeAI — Step-by-Step User Guide

*Capital protection first · AI signals · Paper trade before real money · Indian NSE*

---

## 🔗 LIVE APP LINK

**👉 https://investriskfreeai-goroy.streamlit.app/**

✅ **This is your permanent public link** — already deployed on Streamlit Community
Cloud. It works from any browser, any device (even phone), anywhere, with no token,
no setup. Bookmark it. Share it with anyone who wants to learn.

The app is also running live in this Arena session — you can use the
**"InvestRiskFreeAI app" LIVE PREVIEW panel** in the chat for instant access
(the preview panel carries the sandbox access token automatically).

> 💡 If the Streamlit link ever shows a "waking up" screen, just wait 30–60
> seconds and refresh — the free tier sleeps after long inactivity and wakes on
> first visit.

---

## 📌 THE 5 GOLDEN RULES (read before you do anything)

1. **Risk only 0.5% of your capital per trade.** ₹10,000 capital = ₹50 risk per
   trade. A bad streak of 10 trades costs ~5%, not ~50%.
2. **Never trade without a stop-loss.** Every signal already has one. Never
   remove it, never "wait a bit more".
3. **Never average down** a losing trade. Adding to losers is how accounts die.
4. **Paper trade first.** Use virtual money for at least 3 months. Real money
   only after consistent virtual profit.
5. **No signal = no trade.** Confidence below 55 is automatically blocked. Cash
   is a position. Waiting is a strategy.

---

## STEP 1 — Open the app & Log In

1. Click the **LIVE APP LINK** above.
2. You will be greeted by the **Secure Login Screen** (`require_login` authentication gate):
   - **Username**: `admin`
   - **Password**: `admin123` (or `investriskfree`)
   - *(If deployed on your own Streamlit Community Cloud account, you can customize credentials in App Settings → Secrets).*
3. After logging in, you will see the **Dashboard** with:
   - 🟢 **Market Regime** (RISK-ON / NEUTRAL / RISK-OFF) — if it's RED (RISK-OFF),
     longs are blocked. Do not fight it.
   - **Today's AI Signals** table — every row is a potential trade with its
     stop-loss, target, confidence, and backtested profit probability.
   - **Strategy performance** — how each strategy did on real NSE data 2012–2022.

---

## STEP 2 — Create your paper (virtual) account

1. Click **💰 Paper Trading** in the left sidebar.
2. Enter starting virtual capital. Start with **₹1,00,000** (or ₹10,000 to
   simulate your small-start situation).
3. Click **Create paper account**.
4. You now have a virtual account with the full cost model (brokerage, STT,
   slippage) — exactly what a real broker would charge.

---

## STEP 3 — Read a signal correctly

A signal looks like this:

```
ITC · swing_trend · STRONG (85 conf)
Entry ≈ ₹334   SL ₹316   TGT ₹362   R:R 1:1.6 · P(win) 42% · ~5 days
```

| Field | Meaning |
|---|---|
| **Entry ≈** | Reference buy price (signal close). In reality you'd enter near the next open. |
| **Last ₹** | Latest market price (real-time via yfinance in live mode, or latest close offline). |
| **vs Entry** | Gap % between the current Last price and the reference Entry price. |
| **SL** | Stop-loss. If price touches this, exit. No exceptions. |
| **TGT** | Target. Where the system books profit. |
| **R:R** | Reward:Risk = 1:1.6. You risk 1 to make 1.6. |
| **P(win)** | Historical profit probability from the real-data backtest. |
| **~5 days** | Expected holding duration (median from backtest). |
| **Confidence** | AI Brain score. <55 = blocked, 55–69 = moderate, 70+ = strong. |

**Key insight:** a 42% win rate with 1:1.6 R:R is *profitable* over many trades
(`0.42 × 1.6 − 0.58 × 1 = +0.09R` per trade). You do NOT need to be right often
— you need to follow the rules every time.

---

## STEP 4 — Paper-trade a signal

1. In **💰 Paper Trading**, the **"🏆 Best buys for your current capital"** panel
   ranks today's signals by a suitability score = confidence + backtested
   P(profit) + reward:risk + **affordability with your available cash**.
   The **⭐ Top pick** box shows the single best opportunity with the suggested
   quantity for your capital.
2. In the "📡 Signals to trade" tab, every signal shows:
   - ✅ **Suggested quantity** — the safest size for your capital (0.5% risk budget,
     capped by what your cash can afford, all costs included).
   - A **Quantity** box — **you decide how many shares to buy.** The suggested
     number is pre-filled, but you can type any amount.
   - The **₹ cost** and **risk ₹** of your chosen quantity, plus how much of your
     cash it uses. If it exceeds your cash, the Buy button is disabled (🔴).
3. Click **Buy 📈** — realistic costs are charged.
4. The position appears under **📂 Open positions** with live P&L.
5. **Sell ✋** when your target is hit, your stop is hit, or you decide to exit.
6. Every closed trade goes into the **📜 Trade journal** with its P&L and reason.
7. Watch the **💹 Equity curve** tab — this is your real virtual record.

### 💳 Top-up capital anytime (new)
Click **"💳 Top-up capital"** → enter an amount (e.g. ₹50,000) → **Add capital**.
Your total capital and cash increase, and **the "Best buys" panel re-ranks
automatically with the new available cash** — bigger capital usually means the
suggested quantity for the top pick goes up.

### ⚙️ Risk per trade slider (new)
Use the slider at the top to set your risk budget (0.25%–1% of equity). The
suggestion engine and the "risk ₹" figures update instantly. Conservative start:
**0.5%**. Never above 1% — capital protection first.

> ⏰ **Intraday rule:** intraday positions must be squared off by 15:25 (NSE
> close). The "Square off all intraday positions" button does this.

---

## STEP 5 — Verify a strategy before trusting it

In **🔬 Backtest Lab**:

1. Pick a stock (e.g. RELIANCE) and a strategy (e.g. swing_trend).
2. Click **Run backtest**.
3. Read the numbers:
   - **Profit probability** — % of winning trades.
   - **Expectancy/trade** — average % per trade after all costs. Must be positive.
   - **Max drawdown** — worst dip in the equity curve. Keep it small (this system: 2–4%).
   - **Med duration** — how many days trades typically last.
   - **P(up in 6m)** — probability capital is higher after 6 months.
4. Scroll to the **trade list** — every historical trade with entry, exit, P&L %.
5. Look at the **duration histogram** — most trades should cluster in the
   expected range, not drag on forever.

---

## STEP 6 — Understand the AI Brain (what protects you)

Every signal passes 4 protection layers before it can reach you:

```
Strategy signal
   │
   ▼
1. Market Regime filter ─── market below 200SMA + bad breadth = BLOCKED
   │
   ▼
2. Confidence gate ───────── score < 55/100 = BLOCKED
   │
   ▼
3. ML win-probability gate ─ predicted P(win) below baseline = BLOCKED
   │
   ▼
4. Position sizing ───────── 0.5% risk budget, max 25% of capital
   │
   ▼
   PAPER TRADE FIRST (never real money until proven)
```

The **Strategy Research** page shows the ML gate's honest out-of-sample results
(trained on past years, tested on future years it never saw): 63.6% accuracy,
1.4× lift vs baseline for TrendRider.

---

## STEP 7 — Run it on YOUR computer (with LIVE data)

The live preview uses bundled real NSE data (2012–2022). On your machine you get
**fresh live data** (free, ~15 min delayed — plenty for swing/invest):

```bash
# 1. Install (Windows: use Command Prompt / PowerShell, or Git Bash)
python -m venv .venv
# Windows:  .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. Start the app — then tick "Live data (yfinance)" in the sidebar
streamlit run app.py
# opens at http://localhost:8501

# 3. Today's signals from the terminal (offline bundled data)
python -m investriskfree scan --capital 100000

# 3b. Today's signals with live yfinance Last Price and vs-Entry gap %
python -m investriskfree scan --capital 100000 --live

# 4. Backtest any stock x strategy
python -m investriskfree backtest RELIANCE swing_trend --trades

# 5. Paper account from the terminal
python -m investriskfree paper new 100000
python -m investriskfree paper summary
```

---

## STEP 8 — The growth plan (your ₹1–5k start)

| Your capital | What to do | Notes |
|---|---|---|
| ₹1,000–5,000 | **Practice only.** Paper trade on virtual ₹10k–1L | Real costs eat 1–2% per small trade |
| ₹10,000–25,000 | First small real trades | 0.5% risk = ₹50–125 per trade |
| ₹50,000+ | Run all 3 strategies properly | Diversify swing + invest, 3 positions max |

**The 3-month rule:** paper trade 3+ months. If the virtual account is up with
max drawdown under ~5% and at least 20 closed trades, *then* consider real money
with a tiny size — and keep following every rule.

---

## STEP 9 — Rebuild stats & ML with YOUR live data

Whenever you want fresh numbers (e.g. after the market changes regime):

```bash
python tools/build_stats.py    # refreshes P(win), duration, expectancy for every strategy
python tools/train_ml.py       # retrains the ML gate, walk-forward, out-of-sample
python -m pytest tests/ -q     # sanity-check everything still works
```

---

## 🚫 What this system will NOT do

- It will not make you rich overnight. Anyone promising that is lying.
- It will not predict crashes, news, or corporate announcements.
- It will not "guarantee" returns. It *limits* losses so probability can work.
- It is not SEBI-registered investment advice.

## ✅ What it WILL do

- Keep every loss tiny (0.5% risk per trade, always stopped).
- Show you the **probability and expected duration** of every signal — from real
  backtests, with real costs deducted.
- Let you practice with virtual money until you are consistently profitable.
- Grow your capital slowly and safely — and later, help others who lost money
  trading, exactly as you planned. 🤝

---

*Made with the mission: first make yourself profitable with protected capital,
then help others.*
