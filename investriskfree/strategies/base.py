"""Strategy base class.

Signal contract (columns produced by .signals(df)):
  entry   : 1.0 on the bar whose CLOSE triggers a long entry (executed next open)
  exit    : 1.0 on the bar whose CLOSE triggers an indicator/time exit (next open)
  sl      : stop-loss price for that entry (checked intrabar)
  target  : target price for that entry (checked intrabar)
  reason  : human-readable entry reason
  rr      : reward/risk ratio of the entry
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Strategy:
    name: str
    style: str          # 'swing' | 'intraday' | 'invest'
    timeframe: str      # 'daily' | '5m'
    params: dict = field(default_factory=dict)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df + entry/exit/sl/target/reason/rr columns."""
        raise NotImplementedError

    # ----- helpers -------------------------------------------------
    @staticmethod
    def blank(df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        out = pd.DataFrame(
            {
                "entry": np.zeros(n, dtype=float),
                "exit": np.zeros(n, dtype=float),
                "sl": np.full(n, np.nan),
                "target": np.full(n, np.nan),
                "reason": [""] * n,
                "rr": np.zeros(n, dtype=float),
            },
            index=df.index,
        )
        return out

    @staticmethod
    def fill_sl_target(
        out: pd.DataFrame, entry_mask: pd.Series, entry_px: pd.Series,
        atr: pd.Series, sl_mult: float, target_mult: float,
    ) -> pd.DataFrame:
        """Populate sl/target/rr for entry bars."""
        idx = entry_mask[entry_mask > 0].index
        e = entry_px.loc[idx]
        a = atr.loc[idx]
        out.loc[idx, "sl"] = e - sl_mult * a
        out.loc[idx, "target"] = e + target_mult * a
        out.loc[idx, "rr"] = target_mult / sl_mult
        return out
