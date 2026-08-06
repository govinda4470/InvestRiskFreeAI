# 🚀 Deploying InvestRiskFreeAI to a permanent public URL

The app runs anywhere Streamlit runs. Pick one:

---

## Option 1 — Streamlit Community Cloud (FREE, recommended)

1. **Merge the PR** so `main` has the code:
   `https://github.com/govinda4470/InvestRiskFreeAI/pull/1` → **Merge pull request**.
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **New app** → paste:
   - Repo: `govinda4470/InvestRiskFreeAI`
   - Branch: `main`
   - Main file path: `app.py`
4. Click **Deploy**. In ~3 minutes:
   ```
   https://investriskfreeai.streamlit.app
   ```
5. That URL works from any browser/phone, with no token, forever. 🎉

**Notes**
- The cloud sandbox cannot reach NSE/Yahoo feeds, so it automatically uses the
  bundled real NSE dataset (2012–2022) — perfect for demos and learning.
- The app is fully self-contained: `app.py` + `requirements.txt` + `data/` are
  all in the repo.
- Free tier: apps sleep after ~7 days of inactivity; wake them by clicking the
  URL again.

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
| 💰 Paper Trading | Virtual account, realistic costs, trade journal, equity curve |
| 🔬 Backtest Lab | Any stock × strategy, full stats + trade list + duration histogram |
| 🧠 Strategy Research | Walk-forward ML gate results, honest strategy comparison |
| 📚 Guide | The 5 golden rules + growth plan |
