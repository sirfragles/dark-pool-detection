"""VPIN — Volume-Synchronized Probability of Informed Trading.

Implements Easley, López de Prado, O'Hara (2011):
"Volume-Synchronized Probability of Informed Trading"
"""

import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import get_logger

logger = get_logger(__name__)


class VPINCalculator:
    """Calculate VPIN from tick or bar data."""

    def __init__(
        self, volume_bucket_size: int = 50, n_buckets: int = 50,
        use_bulk_classification: bool = True,
    ):
        self.volume_bucket_size = volume_bucket_size
        self.n_buckets = n_buckets
        self.use_bulk_classification = use_bulk_classification
        self._bucket_buffer: list[dict] = []
        self._vpin_history: list[float] = []

    def compute(
        self, df: pd.DataFrame, price_col: str = "close",
        volume_col: str = "volume", time_col: str | None = None,
    ) -> pd.DataFrame:
        if len(df) < 2:
            return pd.DataFrame()
        if self.use_bulk_classification:
            df = self._bulk_volume_classification(df, price_col, volume_col)
        else:
            df = self._tick_rule_classification(df, price_col)
        buckets = self._build_buckets(df, volume_col, time_col)
        return self._compute_rolling_vpin(buckets)

    def update(self, price: float, volume: float, timestamp: float | None = None) -> float | None:
        self._bucket_buffer.append({"price": price, "volume": volume, "time": timestamp or 0})
        if len(self._bucket_buffer) >= self.volume_bucket_size:
            bucket = self._bucket_buffer[:self.volume_bucket_size]
            self._bucket_buffer = self._bucket_buffer[self.volume_bucket_size:]
            vpin = self._compute_bucket_vpin(bucket)
            self._vpin_history.append(vpin)
            if len(self._vpin_history) > self.n_buckets:
                self._vpin_history.pop(0)
            return vpin
        return None

    def current_vpin(self) -> float:
        if not self._vpin_history:
            return 0.0
        return float(np.mean(self._vpin_history[-self.n_buckets:]))

    def is_toxic(self, threshold: float = 0.8) -> bool:
        return self.current_vpin() > threshold

    def _bulk_volume_classification(self, df, price_col, volume_col):
        df = df.copy()
        df["price_change"] = df[price_col].diff()
        sigma = float(df["price_change"].std())
        if sigma and sigma > 1e-10 and not np.isnan(sigma):
            z_score = df["price_change"].fillna(0) / sigma
            df["buy_prob"] = 0.5 + 0.5 * np.tanh(z_score / np.sqrt(2))
        else:
            df["buy_prob"] = 0.5
            df.loc[df["price_change"] > 0, "buy_prob"] = 0.7
            df.loc[df["price_change"] < 0, "buy_prob"] = 0.3
        df["buy_volume"] = df[volume_col] * df["buy_prob"]
        df["sell_volume"] = df[volume_col] * (1 - df["buy_prob"])
        df["direction"] = 0
        df.loc[df["price_change"] > 0, "direction"] = 1
        df.loc[df["price_change"] < 0, "direction"] = -1
        return df

    def _tick_rule_classification(self, df, price_col):
        df = df.copy()
        df["price_diff"] = df[price_col].diff()
        df["direction"] = 0
        df.loc[df["price_diff"] > 0, "direction"] = 1
        df.loc[df["price_diff"] < 0, "direction"] = -1
        df.loc[df["price_diff"] == 0, "direction"] = df["direction"].shift(1, fill_value=0)
        return df

    def _build_buckets(self, df, volume_col, time_col):
        buckets = []
        cumvol = 0.0
        buy_vol = 0.0
        sell_vol = 0.0
        start_idx = 0
        for i, row in df.iterrows():
            cumvol += row[volume_col]
            buy_vol += row.get("buy_volume", row[volume_col] if row.get("direction", 0) > 0 else 0)
            sell_vol += row.get("sell_volume", row[volume_col] if row.get("direction", 0) < 0 else 0)
            if cumvol >= self.volume_bucket_size:
                bucket_time = row[time_col] if time_col else i
                buckets.append({
                    "time": bucket_time, "volume": cumvol,
                    "buy_volume": buy_vol, "sell_volume": sell_vol,
                    "n_ticks": i - start_idx + 1,
                })
                cumvol = 0.0
                buy_vol = 0.0
                sell_vol = 0.0
                start_idx = i + 1
        return buckets

    def _compute_rolling_vpin(self, buckets):
        if not buckets:
            return pd.DataFrame()
        records = []
        vpin_values = []
        for b in buckets:
            imbalance = abs(b["buy_volume"] - b["sell_volume"])
            bucket_vpin = imbalance / max(b["volume"], 1e-10)
            vpin_values.append(bucket_vpin)
            if len(vpin_values) > self.n_buckets:
                vpin_values.pop(0)
            vpin = float(np.mean(vpin_values))
            records.append({
                "time": b["time"], "vpin": vpin, "bucket_vpin": bucket_vpin,
                "volume": b["volume"], "buy_volume": b["buy_volume"],
                "sell_volume": b["sell_volume"],
                "imbalance_ratio": imbalance / max(b["volume"], 1e-10),
                "n_ticks": b["n_ticks"],
            })
        return pd.DataFrame(records)

    def _compute_bucket_vpin(self, bucket):
        total_vol = sum(b["volume"] for b in bucket)
        if total_vol == 0:
            return 0.0
        buy_vol = sell_vol = 0.0
        prev_price = None
        for tick in bucket:
            p = tick["price"]
            v = tick["volume"]
            if prev_price is None:
                buy_vol += v * 0.5
                sell_vol += v * 0.5
            elif p > prev_price:
                buy_vol += v
            elif p < prev_price:
                sell_vol += v
            else:
                buy_vol += v * 0.5
                sell_vol += v * 0.5
            prev_price = p
        return abs(buy_vol - sell_vol) / total_vol

    def summary(self) -> dict:
        if not self._vpin_history:
            return {"current_vpin": 0.0, "mean": 0.0, "std": 0.0, "toxic_pct": 0.0}
        arr = np.array(self._vpin_history)
        return {
            "current_vpin": float(arr[-1]), "mean": float(np.mean(arr)),
            "std": float(np.std(arr)), "max": float(np.max(arr)),
            "min": float(np.min(arr)), "toxic_pct": float(np.mean(arr > 0.8) * 100),
            "n_buckets": len(arr),
        }
