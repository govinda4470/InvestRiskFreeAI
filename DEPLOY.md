# 🚀 Deploying InvestRiskFreeAI to a permanent public URL

## ✅ Status: DEPLOYED

**Live public link: https://investriskfreeai-goroy.streamlit.app/**

The app is already running on Streamlit Community Cloud (free tier, created by
`govinda4470`). This page documents how it was done and how to update/redo it.

---

## Option 1 — Streamlit Community Cloud (FREE) ← used for the live link

1. Push the code to GitHub: `github.com/govinda4470/InvestRiskFreeAI`
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **New app** → paste:
   - Repo: `govinda4470/InvestRiskFreeAI`
   - Branch: `main` after merging the current Arena pull request
   - Main file path: `app.py`
4. Click **Deploy**. In ~3 minutes you get your permanent URL.
5. To update the deployed app after new commits: on the app page click
   **Manage app** → **Reboot** (or "Rerun" picks up new commits automatically).

### 🔒 Signup, Login & Streamlit Secrets
The welcome screen offers **Log in** and **Sign up**. Local passwords are stored
as salted PBKDF2 hashes, and every user receives a separate paper-trading SQLite
ledger under `data/users/` (runtime data, excluded from Git).

There are **no hardcoded demo credentials**. To add a deployment-administrator
login on Streamlit Community Cloud:

1. In the app dashboard click **⋮ (Settings) → Settings → Secrets**.
2. Add:
   ```toml
   [auth]
   username = "my_private_admin"
   password = "use-a-long-unique-password"
   ```
3. Save. The configured administrator is created in the user registry after its
   first successful login.

> Streamlit Community Cloud's local filesystem is not a durable production
> database. For accounts/history that must survive rebuilds, deploy on a host
> with a persistent volume or replace SQLite with a managed database.

**Notes**
- The cloud sandbox cannot reach NSE/Yahoo feeds, so it automatically uses the
  bundled real NSE dataset (2012–2022) — perfect for demos and learning.
- The app is fully self-contained: `app.py` + `requirements.txt` + `data/` are
  all in the repo.
- Free tier: apps sleep after inactivity; wake them by clicking the URL again.
- The **Run agent cycle** button works in the UI, but Community Cloud does not
  guarantee a continuous background worker. For 24/7 paper monitoring, use a
  persistent host and supervise `python tools/run_agents.py --loop --interval 300 --live`.
- The optional Kronos/PyTorch stack is intentionally excluded from the normal
  deployment requirements. Install `requirements-kronos.txt` only on a host with
  enough memory; model weights download on first use.

---

## Option 2 — Hugging Face Spaces (FREE alternative)

1. Go to **https://huggingface.co/new-space**
2. Name: `investriskfreeai` · SDK: **Streamlit** · Hardware: CPU basic (free).
3. Create the Space, then push:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USER/investriskfreeai
   cp -r /path/to/InvestRiskFreeAI/* .
   git add -A && git commit -m "deploy" && git push
   ```
   (Requires your Hugging Face token: Settings → Access Tokens.)
4. Public URL: `https://YOUR_USER-investriskfreeai.hf.space`

---

## Option 3 — Run it yourself (your own server / laptop)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py             # opens at http://localhost:8501
```

For 24/7 hosting on a VPS, add a systemd unit or run:
`streamlit run app.py --server.address 0.0.0.0 --server.port 8501`

---

## What the deployed app includes

| Section | What you get |
|---|---|
| 🏠 Dashboard | Market regime, today's AI signals (with P(win)% + duration), strategy stats |
| 💰 Paper Trading | Private virtual account, realistic costs, history, drawdown + strategy graphs |
| 🤖 Auto-Trade Agent | User-capital sizing, stops/targets, guardrails and full decision audit (paper only) |
| 🔬 Backtest Lab | Any stock × strategy, full stats + trade list + duration histogram |
| 🧠 Strategy Research | Walk-forward ML results + optional Kronos forecast lab |
| 📚 Guide | The 5 golden rules + growth plan |
