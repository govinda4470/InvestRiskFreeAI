# Code audit and implementation notes

Date: 2026-08-07  
Scope: all tracked Python application modules, Streamlit flows, tests, deployment
docs, and the relevant runtime/training/example code in
`shiyu-coder/Kronos` at commit
`67b630e67f6a18c9e9be918d9b4337c960db1e9a`.

This is an engineering review, not a formal security certification or financial
model validation.

## High-impact findings fixed

| Finding | Impact | Resolution |
|---|---|---|
| Login accepted several hardcoded passwords, including a password-only fallback for any username | Anyone who knew a source-code password could enter the shared workspace | Replaced with PBKDF2-HMAC-SHA256 signup/login, unique salts, constant-time comparison and temporary lockout. Deployment-secret login now requires both configured username and password. |
| No signup option and one shared `paper.db` | Users could not register; capital/trades were not isolated | Added signup UI, user registry and separate `data/users/<opaque-id>/paper.db` ledgers. |
| Dashboard/Paper scanner used hardcoded ₹100,000 | Suggested sizes did not follow the signed-in user's updated capital | Scans and agent sizing now use current per-user equity/cash after deposits, withdrawals, P&L and costs. |
| UI passed `0.5` to a sizing function expecting `0.005` | Suggested risk could be interpreted as 50% instead of 0.5% | Normalize percentage/fraction boundaries and cover with tests. |
| Position sizing ignored expected entry slippage | A nominally safe quantity could exceed the risk budget at the actual paper fill | Size from the conservative slipped fill; broker re-validates risk transactionally. |
| Paper trade `net_pnl` omitted entry charges; sell/buy snapshots were incomplete | Journal totals could disagree with account equity and graphs | Persist entry and exit costs, reconcile total P&L as equity minus net deposits, and snapshot every capital/order/agent event. |
| Equity backtest started at zero and final liquidation costs were not reflected in the final point | Drawdown/returns could be distorted | Initialize every curve at starting capital and update the last point after liquidation. |
| “Live” daily mode could silently use an old cache/bundled 2022 data | A stale historical signal could be presented as live | Explicit yfinance mode now requires fresh/downloaded data and raises on failure; the signal candle timestamp stays separate from quote time. |
| Scanner's `use_ml` flag was not used | UI claimed an ML gate that was not executing | Persist final model parameters to `data/ml_models.json`, calculate latest-bar probability, show availability, and block below threshold. |
| ML walk-forward tests repeatedly included the entire future after each cutoff | OOS rows were duplicated and metrics overstated | Use expanding training with non-overlapping future test windows. |
| ML outcome was aligned to next-open execution date features | Entry-day close/high/low leaked future information into a close-generated signal | Align each result to the previous signal-close bar. |
| Scanner sorted blocked signals before actionable signals | Ranking was confusing and error-prone for consumers | Actionable signals now sort first by confidence. |
| No automated signal executor or decision history | Signals required manual entry and there was no “why did it trade?” audit | Added an idempotent paper agent, guardrails, stops/targets/max-hold/square-off, signal fingerprinting, run/event audit and worker. |

## Kronos review and integration

Kronos provides a tokenizer plus autoregressive transformer for OHLCV/K-line
forecasting. The useful inference runtime (`model/kronos.py`, `model/module.py`)
was vendored under its MIT license and wrapped in a lazy optional adapter.

Integrated:

- NSE OHLCV-to-Kronos input normalization and timestamp generation;
- Kronos-mini/Tokenizer-2k loading from Hugging Face;
- forecast summary and history/forecast graph;
- optional positive-forecast confluence gate for paper entries;
- lazy optional dependencies so normal Streamlit deployment remains lightweight;
- upstream license and attribution.

Not integrated:

- generated example artifacts and Chinese A-share data;
- the upstream example backtester, which uses all-in sizing and does not model
  this project's Indian costs/risk rules;
- fine-tuning pipelines, which require a separate GPU/data validation project;
- raw forecast direction as a direct trade signal.

A forecast is not a calibrated probability of profit. Kronos remains disabled by
default and cannot override this project's regime, confidence, stop or sizing
rules.

## Validation performed

- `python -m py_compile` on application, tools and vendored runtime.
- Ruff fatal/undefined-name checks (`F,E9`) on app, package, tools and tests.
- 16 tests covering data, strategies, costs, backtesting, account isolation
  primitives, password hashing, capital events, risk sizing, auto-agent
  idempotency, equity history, ML fold uniqueness and Kronos input conversion.
- Streamlit AppTest for login/signup, paper-account creation and auto-agent pages.
- Streamlit server smoke run bound to `0.0.0.0`.
- Full 50-stock ML registry rebuild using corrected chronology.

## Deliberate remaining limits

1. **Paper only:** there is no real-money broker executor. Adding one safely needs
   a chosen broker, instrument mapping, exchange order reconciliation, idempotency
   keys, broker-native stops, kill switch, credential vault and explicit legal/
   user review.
2. **Delayed data:** yfinance is suitable for research/paper swing workflows, not
   guaranteed realtime execution. Intraday synthetic bars remain demo-only.
3. **SQLite deployment:** separate files provide local user isolation, but
   Streamlit Community Cloud storage is not durable. Production should use a
   persistent volume or managed database and an external identity provider.
4. **Background operation:** the UI runs an immediate cycle. Continuous paper
   monitoring requires the supervised `tools/run_agents.py` process on an
   always-on host.
5. **No guaranteed edge:** corrected OOS results are historical and can decay.
   Backtest and model output are not financial advice or future guarantees.
