"""Lightweight, persisted ML win-probability gate (pure NumPy logistic regression).

Design constraints:
* Features use only candles available when a signal closes.
* Walk-forward test windows are chronological and non-overlapping.
* The final model is fitted only after out-of-sample metrics are calculated.
* ML may block a trade; it can never override regime or portfolio risk rules.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from .config import get

FEATURE_COLS = [
    "atr_pct",
    "rsi14",
    "rsi2",
    "macd_hist_norm",
    "ema20_slope",
    "dist_200sma",
    "vol_z",
    "breadth",
    "adx",
    "roc10",
    "dist_52w_high",
]
MODEL_REGISTRY_PATH = os.path.join(get("data.repo_root"), "data", "ml_models.json")


def build_features(
    df: pd.DataFrame, signals: pd.DataFrame, breadth: pd.Series | None = None
) -> pd.DataFrame:
    """Build per-bar features from current/past data only."""
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
    feats["dist_52w_high"] = (
        d["Close"] / d["High"].rolling(252, min_periods=60).max() - 1
    ) * 100
    if breadth is not None:
        feats["breadth"] = breadth.reindex(d.index).ffill()
    else:
        feats["breadth"] = 0.5
    feats["price"] = d["Close"]
    feats["entry"] = signals["entry"].to_numpy()
    return feats


class WinProbModel:
    """Small logistic-regression model with honest walk-forward evaluation."""

    def __init__(self, lr: float = 0.2, epochs: int = 500, seed: int = 7):
        self.lr = lr
        self.epochs = epochs
        self.rng = np.random.default_rng(seed)
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.oos_report: dict | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, dimensions = X.shape
        weights = np.zeros(dimensions)
        bias = 0.0
        # Small L2 penalty reduces unstable coefficients on correlated indicators.
        regularization = 1e-3
        for _ in range(self.epochs):
            probabilities = self._sigmoid(X @ weights + bias)
            gradient = X.T @ (probabilities - y) / n + regularization * weights
            bias_gradient = (probabilities - y).mean()
            weights -= self.lr * gradient
            bias -= self.lr * bias_gradient
        self.w = weights
        self.b = float(bias)

    def _fit_scaler(self, X: np.ndarray) -> np.ndarray:
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-9
        return (X - self.mu) / self.sd

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "WinProbModel":
        frame = features[FEATURE_COLS].copy()
        frame["__label__"] = labels.reindex(frame.index).to_numpy()
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < 20:
            raise ValueError("not enough clean observations to fit model")
        X = self._fit_scaler(frame[FEATURE_COLS].to_numpy(float))
        self._fit(X, frame["__label__"].to_numpy(float))
        return self

    def predict_proba(self, standardized_X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("model is not fitted")
        return self._sigmoid(standardized_X @ self.w + self.b)

    def predict_frame(self, features: pd.DataFrame) -> np.ndarray:
        if self.w is None or self.mu is None or self.sd is None:
            raise RuntimeError("model is not fitted")
        X = features[FEATURE_COLS].to_numpy(float)
        return self.predict_proba((X - self.mu) / self.sd)

    def walk_forward(
        self,
        feats: pd.DataFrame,
        labels: pd.Series,
        n_splits: int = 3,
        min_train_years: int = 3,
    ) -> dict:
        """Expanding-train/non-overlapping-test chronological evaluation."""
        frame = feats.copy()
        frame["__label__"] = labels.reindex(frame.index).to_numpy()
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
            subset=FEATURE_COLS + ["__label__"]
        )
        if len(frame) < 100:
            return {"trained": False, "reason": "not enough data"}
        dates = frame.index
        if isinstance(dates, pd.MultiIndex):
            dates = dates.get_level_values(-1)
        dates = pd.DatetimeIndex(dates)
        start = dates.min()
        years = np.asarray((dates - start).days / 365.25, dtype=float)
        total_years = float(years.max())
        if total_years < min_train_years + 1:
            return {"trained": False, "reason": f"only {total_years:.1f}y of data"}

        boundaries = np.linspace(float(min_train_years), total_years + 1e-6, n_splits + 1)
        oos_predictions: list[float] = []
        oos_labels: list[int] = []
        fold_reports: list[dict[str, Any]] = []
        for fold, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
            train_mask = years < lower
            test_mask = (years >= lower) & (years < upper)
            train = frame.iloc[np.flatnonzero(train_mask)]
            test = frame.iloc[np.flatnonzero(test_mask)]
            if len(train) < 60 or len(test) < 20:
                continue
            X_train = self._fit_scaler(train[FEATURE_COLS].to_numpy(float))
            self._fit(X_train, train["__label__"].to_numpy(float))
            X_test = (test[FEATURE_COLS].to_numpy(float) - self.mu) / self.sd
            predictions = self.predict_proba(X_test)
            labels_test = test["__label__"].to_numpy(int)
            oos_predictions.extend(predictions.tolist())
            oos_labels.extend(labels_test.tolist())
            fold_reports.append(
                {
                    "fold": fold,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "test_start": str(dates[test_mask].min().date()),
                    "test_end": str(dates[test_mask].max().date()),
                }
            )
        if not oos_predictions:
            return {"trained": False, "reason": "walk-forward produced no test set"}

        predictions = np.asarray(oos_predictions)
        outcomes = np.asarray(oos_labels)
        baseline = float(outcomes.mean())
        predicts_win = predictions >= 0.5
        accuracy = float((predicts_win == outcomes).mean())
        positives = int(predicts_win.sum())
        true_positives = int((predicts_win & (outcomes == 1)).sum())
        precision = true_positives / positives if positives else float("nan")
        brier = float(np.mean((predictions - outcomes) ** 2))
        self.oos_report = {
            "trained": True,
            "n_oos": int(len(outcomes)),
            "baseline_win_rate": baseline * 100,
            "oos_win_rate": float(outcomes.mean() * 100),
            "accuracy": accuracy * 100,
            "precision_when_win": precision * 100 if np.isfinite(precision) else float("nan"),
            "lift": precision / baseline if baseline > 0 and np.isfinite(precision) else float("nan"),
            "brier_score": brier,
            "folds": fold_reports,
        }
        # Fit the deployable model on all observations only after OOS scoring.
        self.fit(frame[FEATURE_COLS], frame["__label__"])
        return self.oos_report

    def gate(self, p_win: float, baseline: float) -> bool:
        """Allow only forecasts at or above the strategy's OOS base rate."""
        if not np.isfinite(p_win):
            return False
        return p_win >= float(baseline)

    def to_dict(self, baseline: float | None = None) -> dict:
        if self.w is None or self.mu is None or self.sd is None:
            raise RuntimeError("model is not fitted")
        return {
            "feature_cols": FEATURE_COLS,
            "weights": self.w.tolist(),
            "bias": self.b,
            "mean": self.mu.tolist(),
            "std": self.sd.tolist(),
            "baseline": baseline,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WinProbModel":
        if payload.get("feature_cols") != FEATURE_COLS:
            raise ValueError("saved model feature schema does not match")
        model = cls()
        model.w = np.asarray(payload["weights"], dtype=float)
        model.b = float(payload["bias"])
        model.mu = np.asarray(payload["mean"], dtype=float)
        model.sd = np.asarray(payload["std"], dtype=float)
        return model


def load_model_registry(path: str = MODEL_REGISTRY_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError):
        return {}


def predict_latest(
    strategy_name: str,
    df: pd.DataFrame,
    signals: pd.DataFrame,
    breadth: pd.Series | None = None,
    registry: dict | None = None,
) -> dict:
    """Return the deployable model's probability for the latest signal bar."""
    registry = registry if registry is not None else load_model_registry()
    payload = registry.get(strategy_name)
    if not payload:
        return {"available": False, "reason": "no persisted model"}
    try:
        features = build_features(df, signals, breadth).iloc[[-1]]
        if features[FEATURE_COLS].isna().any(axis=None):
            return {"available": False, "reason": "latest feature row is incomplete"}
        model = WinProbModel.from_dict(payload)
        probability = float(model.predict_frame(features)[0])
        baseline = float(payload.get("baseline") or 0.5)
        return {
            "available": True,
            "p_win": probability,
            "baseline": baseline,
            "allow": model.gate(probability, baseline),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
