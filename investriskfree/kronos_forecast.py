"""Optional Kronos candlestick foundation-model integration.

Kronos (https://github.com/shiyu-coder/Kronos) forecasts future OHLCV candles.
This adapter converts this project's NSE frames to the upstream input format,
lazily loads the open-source model, and turns a forecast into a transparent
*confluence gate*.  A Kronos forecast never bypasses regime, stop-loss, sizing,
or confidence rules.

The upstream runtime is vendored under ``third_party/kronos`` at commit
67b630e67f6a18c9e9be918d9b4337c960db1e9a (MIT license).  Model weights are
fetched from Hugging Face at runtime and are deliberately not committed.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd

DEFAULT_MODEL = "NeoQuasar/Kronos-mini"
DEFAULT_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-2k"


@dataclass
class KronosForecast:
    symbol: str
    generated_at: str
    history_last: float
    forecast: pd.DataFrame
    expected_return_pct: float
    forecast_low_pct: float
    forecast_high_pct: float
    direction: str
    model_name: str

    def summary(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "generated_at": self.generated_at,
            "history_last": self.history_last,
            "expected_return_pct": self.expected_return_pct,
            "forecast_low_pct": self.forecast_low_pct,
            "forecast_high_pct": self.forecast_high_pct,
            "direction": self.direction,
            "model_name": self.model_name,
        }


def kronos_dependencies_available() -> tuple[bool, str]:
    missing = [
        module
        for module in ("torch", "einops", "huggingface_hub", "safetensors", "tqdm")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        return False, "Missing optional packages: " + ", ".join(missing)
    return True, "Kronos runtime is available"


def prepare_kline_frame(df: pd.DataFrame, lookback: int = 512) -> pd.DataFrame:
    """Validate/convert an InvestRiskFreeAI OHLCV frame for Kronos."""
    required = ["Open", "High", "Low", "Close"]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"missing OHLC columns: {', '.join(missing)}")
    if len(df) < 64:
        raise ValueError("Kronos needs at least 64 historical candles")
    out = df.tail(int(lookback)).copy()
    out = out.rename(columns={column: column.lower() for column in out.columns})
    if "volume" not in out:
        out["volume"] = 0.0
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["amount"] = out["volume"] * out[["open", "high", "low", "close"]].mean(axis=1)
    out = out[["open", "high", "low", "close", "volume", "amount"]].dropna()
    if len(out) < 64:
        raise ValueError("not enough clean candles after validation")
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(None)
    return out


def future_business_timestamps(last_timestamp, horizon: int) -> pd.DatetimeIndex:
    """Daily NSE forecast timestamps (weekends excluded)."""
    horizon = int(horizon)
    if not 1 <= horizon <= 60:
        raise ValueError("forecast horizon must be between 1 and 60 bars")
    last = pd.Timestamp(last_timestamp).tz_localize(None)
    return pd.bdate_range(last + pd.offsets.BDay(1), periods=horizon)


class KronosForecastService:
    """Lazy singleton-style wrapper around the official Kronos predictor."""

    _loaded: ClassVar[dict[tuple[str, str, str], Any]] = {}

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        tokenizer_name: str = DEFAULT_TOKENIZER,
        device: str | None = None,
        max_context: int = 512,
    ):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.device = device
        self.max_context = max_context

    def _predictor(self):
        ok, message = kronos_dependencies_available()
        if not ok:
            raise RuntimeError(
                f"{message}. Install with: pip install -r requirements-kronos.txt"
            )
        import torch
        from third_party.kronos.model import Kronos, KronosPredictor, KronosTokenizer

        device = self.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        key = (self.model_name, self.tokenizer_name, device)
        if key not in self._loaded:
            tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
            model = Kronos.from_pretrained(self.model_name)
            tokenizer.eval()
            model.eval()
            self._loaded[key] = KronosPredictor(
                model, tokenizer, device=device, max_context=self.max_context
            )
        return self._loaded[key]

    def forecast(
        self,
        symbol: str,
        df: pd.DataFrame,
        horizon: int = 10,
        *,
        lookback: int = 256,
        sample_count: int = 3,
    ) -> KronosForecast:
        prepared = prepare_kline_frame(df, min(lookback, self.max_context))
        future = future_business_timestamps(prepared.index[-1], horizon)
        predictor = self._predictor()
        prediction = predictor.predict(
            df=prepared,
            x_timestamp=pd.Series(prepared.index),
            y_timestamp=pd.Series(future),
            pred_len=int(horizon),
            T=1.0,
            top_k=0,
            top_p=0.9,
            sample_count=max(1, int(sample_count)),
            verbose=False,
        )
        prediction.index = future
        last = float(prepared["close"].iloc[-1])
        expected = (float(prediction["close"].iloc[-1]) / last - 1) * 100
        forecast_low = (float(prediction["low"].min()) / last - 1) * 100
        forecast_high = (float(prediction["high"].max()) / last - 1) * 100
        direction = "BULLISH" if expected > 0.25 else "BEARISH" if expected < -0.25 else "NEUTRAL"
        return KronosForecast(
            symbol=symbol,
            generated_at=pd.Timestamp.now(tz="UTC").isoformat(),
            history_last=last,
            forecast=prediction,
            expected_return_pct=float(expected),
            forecast_low_pct=float(forecast_low),
            forecast_high_pct=float(forecast_high),
            direction=direction,
            model_name=self.model_name,
        )

    def long_signal_gate(
        self,
        signal: dict,
        df: pd.DataFrame,
        *,
        horizon: int = 5,
        min_return_pct: float = 0.25,
    ) -> dict:
        """Require a positive Kronos forecast as one additional long gate."""
        result = self.forecast(signal["symbol"], df, horizon=horizon)
        allow = result.expected_return_pct >= float(min_return_pct)
        return {
            "allow": allow,
            "expected_return_pct": round(result.expected_return_pct, 3),
            "direction": result.direction,
            "horizon_bars": horizon,
            "model": result.model_name,
            "reason": (
                "forecast confirms long signal"
                if allow
                else f"forecast return below {min_return_pct:.2f}% gate"
            ),
        }
