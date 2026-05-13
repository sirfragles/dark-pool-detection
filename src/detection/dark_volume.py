"""Dark Volume Reconstruction — Faza 4.

Reconstructs dark pool trading volume from public data.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class DarkVolumeReport:
    timestamp: float
    ticker: str
    lit_volume: float
    dark_volume_est: float
    total_volume: float
    dark_share: float
    anomaly_score: float
    anomaly_reasons: list[str] = field(default_factory=list)


class DarkVolumeReconstructor:
    """Reconstruct dark pool trading activity from public data."""

    def __init__(
        self, anomaly_threshold_std: float = 3.0, min_dark_share: float = 0.05,
        lookback_window_bars: int = 20, venue_efficiency: float = 0.85,
    ):
        self.anomaly_threshold = anomaly_threshold_std
        self.min_dark_share = min_dark_share
        self.lookback = lookback_window_bars
        self.venue_efficiency = venue_efficiency
        self._lit_baseline: dict[str, float] = {}
        self._reports: list[DarkVolumeReport] = []
        self._finra_ats_data: pd.DataFrame | None = None

    def reconstruct(
        self, lit_data: pd.DataFrame, ticker: str = "UNKNOWN",
        total_market_volume: Optional[float] = None,
    ) -> list[DarkVolumeReport]:
        if lit_data.empty:
            return []
        vol_col = next((c for c in ["volume", "Volume"] if c in lit_data.columns), None)
        if vol_col is None:
            return []
        lit_data = lit_data.copy()
        self._update_baseline(ticker, lit_data[vol_col])
        lit_data["volume_ma"] = lit_data[vol_col].rolling(self.lookback, min_periods=1).mean()
        lit_data["volume_std"] = lit_data[vol_col].rolling(self.lookback, min_periods=1).std()
        lit_data["volume_zscore"] = (
            (lit_data[vol_col] - lit_data["volume_ma"]) / lit_data["volume_std"].replace(0, np.nan)
        ).fillna(0)

        reports = []
        time_col = next((c for c in ["timestamp", "datetime", "date"] if c in lit_data.columns), None)
        for idx, row in lit_data.iterrows():
            lit_vol = float(row[vol_col])
            zscore = float(row.get("volume_zscore", 0))
            ts = float(row.get(time_col, idx)) if time_col else float(idx)
            dark_est, reasons = self._estimate_dark_for_bar(ticker, lit_vol, zscore, ts)
            anomaly_score = min(1.0, abs(zscore) / self.anomaly_threshold)
            reports.append(DarkVolumeReport(
                timestamp=ts, ticker=ticker, lit_volume=lit_vol,
                dark_volume_est=dark_est, total_volume=lit_vol + dark_est,
                dark_share=dark_est / max(lit_vol + dark_est, 1),
                anomaly_score=anomaly_score, anomaly_reasons=reasons,
            ))
        self._reports.extend(reports)
        return reports

    def reconstruct_batch(
        self, lit_data: dict[str, pd.DataFrame],
        total_market_volume: Optional[dict[str, float]] = None,
    ) -> pd.DataFrame:
        all_rows = []
        for ticker, df in lit_data.items():
            reports = self.reconstruct(df, ticker=ticker)
            for r in reports:
                all_rows.append({
                    "timestamp": r.timestamp, "ticker": r.ticker,
                    "lit_volume": r.lit_volume, "dark_volume_est": r.dark_volume_est,
                    "total_volume": r.total_volume, "dark_share": r.dark_share,
                    "anomaly_score": r.anomaly_score,
                    "anomaly_reasons": ";".join(r.anomaly_reasons),
                })
        return pd.DataFrame(all_rows).sort_values("anomaly_score", ascending=False)

    def detect_anomaly_periods(
        self, reports: list[DarkVolumeReport], min_anomaly_score: float = 0.7,
    ) -> pd.DataFrame:
        anomalies = [r for r in reports if r.anomaly_score >= min_anomaly_score]
        if not anomalies:
            return pd.DataFrame()
        return pd.DataFrame([{
            "timestamp": r.timestamp, "ticker": r.ticker,
            "dark_volume": r.dark_volume_est, "lit_volume": r.lit_volume,
            "dark_share": r.dark_share, "anomaly_score": r.anomaly_score,
            "reasons": ", ".join(r.anomaly_reasons),
        } for r in anomalies])

    def hourly_pattern(self, reports: list[DarkVolumeReport]) -> pd.DataFrame:
        if not reports:
            return pd.DataFrame()
        df = pd.DataFrame([{
            "hour": int((r.timestamp % 86_400_000) // 3_600_000),
            "dark_vol": r.dark_volume_est, "lit_vol": r.lit_volume,
            "dark_share": r.dark_share,
        } for r in reports])
        hourly = df.groupby("hour").agg(
            dark_volume=("dark_vol", "sum"), lit_volume=("lit_vol", "sum"),
            dark_share_avg=("dark_share", "mean"), n_bars=("dark_vol", "count"),
        ).reset_index()
        hourly["dark_pct"] = hourly["dark_volume"] / (hourly["dark_volume"] + hourly["lit_volume"]) * 100
        return hourly.sort_values("dark_pct", ascending=False)

    def _update_baseline(self, ticker: str, volume_series: pd.Series) -> None:
        avg = float(volume_series.mean()) if len(volume_series) > 0 else 0.0
        if ticker in self._lit_baseline:
            alpha = 0.3
            self._lit_baseline[ticker] = alpha * avg + (1 - alpha) * self._lit_baseline[ticker]
        else:
            self._lit_baseline[ticker] = avg

    def _estimate_dark_for_bar(
        self, ticker: str, lit_vol: float, zscore: float, timestamp: float,
    ) -> tuple[float, list[str]]:
        reasons = []
        base_dark = lit_vol * (0.35 / 0.65)
        anomaly_mult = 1.0
        if abs(zscore) > self.anomaly_threshold:
            anomaly_mult = min(3.0, 1.0 + (abs(zscore) - self.anomaly_threshold) * 0.5)
            reasons.append(f"volume_spike_z={zscore:.1f}")
        hour = (timestamp % 86_400_000) // 3_600_000
        tod_mult = 1.0
        if hour in (9, 10, 15, 16):
            tod_mult = 1.3
            reasons.append(f"tod_hour={int(hour)}")
        dark_est = base_dark * anomaly_mult * tod_mult
        dark_est = max(0.0, dark_est)
        if dark_est / max(lit_vol + dark_est, 1) < self.min_dark_share:
            dark_est = lit_vol * self.min_dark_share
            reasons.append("min_share_floor")
        return dark_est, reasons

    @property
    def summary(self) -> dict:
        if not self._reports:
            return {"n_reports": 0}
        dark_vol = sum(r.dark_volume_est for r in self._reports)
        lit_vol = sum(r.lit_volume for r in self._reports)
        total = dark_vol + lit_vol
        return {
            "n_reports": len(self._reports),
            "total_dark_volume": dark_vol, "total_lit_volume": lit_vol,
            "dark_share_pct": dark_vol / max(total, 1) * 100,
            "n_anomalies": sum(1 for r in self._reports if r.anomaly_score > 0.7),
            "avg_anomaly_score": float(np.mean([r.anomaly_score for r in self._reports])),
        }

    def reset(self) -> None:
        self._reports.clear()
        self._lit_baseline.clear()
