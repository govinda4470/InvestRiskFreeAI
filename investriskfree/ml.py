"""Lightweight ML win-probability model (logistic regression in pure numpy).

Honest design:
  * Features are computed ONLY from data available at entry time (no lookahead).
  * Walk-forward evaluation: the model is trained on past years and tested on
    the future years it never saw. Reported OOS stats are the real deal.
  * The model NEVER overrides the risk rules; it only gates trades whose
    predicted P(win) is below the historical baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "atr_pct", "rsi14", "rsi2", "macd_hist_norm", "ema20_slope",
    "dist_200sma", "vol_z", "breadth", "adx", "roc10", "dist_52w_high",
]


def build_features(df: pd.DataFrame, signals: pd.DataFrame, breadth: pd.Series | None = None) -> pd.DataFrame:
    """Per-entry-bar feature matrix (only look-back indicators)."""
    from . import indicators as ta

    d = df
    feats = pd.DataFrame(index=d.index)
    atr = ta.atr(d, 14)
    feats["atr_pct"] = atr / d["Close"] * 100
    feats["rsi14"] = ta.rsi(d["Close"], 14)
    feats["rsi2"] = ta.rsi(d["Close"], 2)
    macd_df = ta.macd(d["Close"])
    feats["macd_hist_norm"] = macd_df["hist"] / d["Close"] * 100
    ema20 = ta.ema(d["Close"], 20)
    feats["ema20_slope"] = ema20.pct_change(5) * 100
    feats["dist_200sma"] = (d["Close"] / ta.sma(d["Close"], 200) - 1) * 100
    feats["vol_z"] = ta.volume_z(d, 20)
    feats["adx"] = ta.adx(d, 14)
    feats["roc10"] = ta.roc(d["Close"], 10)
    feats["dist_52w_high"] = (d["Close"] / d["High"].rolling(252, min_periods=60).max() - 1) * 100
    if breadth is not None:
        feats["breadth"] = breadth.reindex(d.index).ffill()
    else:
        feats["breadth"] = 0.5
    feats["price"] = d["Close"]
    feats["entry"] = signals["entry"].to_numpy()
    return feats


class WinProbModel:
    """Logistic regression with walk-forward evaluation."""

    def __init__(self, lr: float = 0.5, epochs: int = 400, seed: int = 7):
        self.lr = lr
        self.epochs = epochs
        self.rng = np.random.default_rng(seed)
        self.w: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.oos_report: dict | None = None

    # ---------------- helpers ----------------
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self.mu is None:
            self.mu = X.mean(axis=0)
            self.sd = X.std(axis=0) + 1e-9
        return (X - self.mu) / self.sd

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        for _ in range(self.epochs):
            z = X @ w + b
            p = self._sigmoid(z)
            grad_w = X.T @ (p - y) / n
            grad_b = (p - y).mean()
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        self.w = w
        self.b = b

    def _fit_scaler(self, X: np.ndarray) -> np.ndarray:
        """Standardize X, fitting mu/sd on X (call once per training fold)."""
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-9
        return (X - self.mu) / self.sd

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.w + self.b)

    # ---------------- walk-forward ----------------
    def walk_forward(
        self, feats: pd.DataFrame, labels: pd.Series, n_splits: int = 3,
        min_train_years: int = 3,
    ) -> dict:
        """Train on past, test on future, in n_splits chronological folds.

        `feats` may contain duplicate index entries (same entry date across
        different stocks) - we join the label into the frame and use integer
        positions to stay perfectly aligned.
        """
        fe = feats.copy()
        fe["__label__"] = labels.reindex(fe.index).to_numpy()
        fe = fe.dropna(subset=FEATURE_COLS + ["__label__"])
        if len(fe) < 100:
            return {"trained": False, "reason": "not enough data"}
        dates = fe.index
        if isinstance(dates, pd.MultiIndex):
            dates = dates.get_level_values(-1)
        start = dates.min()
        years = (dates - start).days / 365.25
        total_years = float(years.max())
        if total_years < min_train_years + 1:
            return {"trained": False, "reason": f"only {total_years:.1f}y of data"}
        split_pts = np.linspace(min_train_years, total_years - 0.5, n_splits + 1)[:-1]
        oos_preds: list[float] = []
        oos_labels: list[int] = []
        for cutoff_y in split_pts:
            cutoff = start + pd.Timedelta(days=cutoff_y * 365.25)
            tr = fe[years < cutoff_y]
            te = fe[years >= cutoff_y]
            if len(tr) < 60 or len(te) < 20:
                continue
            Xtr = self._fit_scaler(tr[FEATURE_COLS].to_numpy(float))
            self._fit(Xtr, tr["__label__"].to_numpy(float))
            # standardize test with the TRAINING fold's scaler (no leakage)
            Xte = (te[FEATURE_COLS].to_numpy(float) - self.mu) / self.sd
            oos_preds.extend(self.predict_proba(Xte).tolist())
            oos_labels.extend(te["__label__"].tolist())
        if not oos_preds:
            return {"trained": False, "reason": "walk-forward produced no test set"}
        oos_preds = np.array(oos_preds)
        oos_labels = np.array(oos_labels)
        base = oos_labels.mean()
        pred_win = oos_preds >= 0.5
        acc = (pred_win == oos_labels).mean()
        # lift: how much better than baseline when the model says "win"
        agree = pred_win & (oos_labels == 1)
        tp = agree.sum()
        fp = (pred_win & (oos_labels == 0)).sum()
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        self.oos_report = {
            "trained": True,
            "n_oos": int(len(oos_labels)),
            "baseline_win_rate": float(base * 100),
            "oos_win_rate": float(oos_labels.mean() * 100),
            "accuracy": float(acc * 100),
            "precision_when_win": float(precision * 100) if np.isfinite(precision) else float("nan"),
            "lift": float(precision / base) if base > 0 and np.isfinite(precision) else float("nan"),
        }
        return self.oos_report

    def gate(self, p_win: float, baseline: float) -> bool:
        """Allow the trade only if the model's P(win) >= baseline (or >= 0.5)."""
        if p_win != p_win:  # nan
            return False
        return p_win >= max(baseline, 0.5) - 0.02
