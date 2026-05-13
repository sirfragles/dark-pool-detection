"""Dark Pool Detection — Alert Engine."""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import get_logger

logger = get_logger(__name__)


class AlertLevel(str, Enum):
    INFO = "info"
    WATCH = "watch"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    timestamp: float
    level: AlertLevel
    module: str
    message: str
    value: float
    threshold: float
    ticker: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def iso_time(self) -> str:
        dt = datetime.fromtimestamp(self.timestamp / 1000, tz=timezone.utc)
        return dt.isoformat()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["level"] = self.level.value
        d["iso_time"] = self.iso_time
        return d


class AlertEngine:
    DEFAULT_THRESHOLDS = {
        "vpin": {"toxic": 0.8, "elevated": 0.6},
        "iceberg": {"min_confidence": 0.7, "min_hidden_volume": 5000},
        "dark_volume": {"anomaly_zscore": 3.0, "dark_share_pct": 40.0},
        "ml": {"dark_trade_prob": 0.7, "iceberg_fill_prob": 0.6},
    }

    def __init__(self, output_dir="output", thresholds=None, max_history=10000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self.max_history = max_history
        self._alerts: list[Alert] = []
        self._alert_log_path = self.output_dir / "alerts.jsonl"
        self._handlers: list[Callable] = []

    def register_handler(self, handler):
        self._handlers.append(handler)

    def check_vpin(self, vpin_summary, timestamp=None):
        ts = timestamp or time.time() * 1000
        alerts = []
        current = vpin_summary.get("current_vpin", 0)
        toxic_pct = vpin_summary.get("toxic_pct", 0)
        t = self.thresholds["vpin"]
        if current >= t["toxic"]:
            alerts.append(Alert(ts, AlertLevel.CRITICAL, "vpin", f"VPIN toxic: {current:.3f}", current, t["toxic"], metadata=vpin_summary))
        elif current >= t["elevated"]:
            alerts.append(Alert(ts, AlertLevel.WARNING, "vpin", f"VPIN elevated: {current:.3f}", current, t["elevated"], metadata=vpin_summary))
        if toxic_pct > 30:
            alerts.append(Alert(ts, AlertLevel.WARNING, "vpin", f"High toxic: {toxic_pct:.1f}%", toxic_pct, 30.0, metadata=vpin_summary))
        for a in alerts:
            self._emit(a)
        return alerts

    def check_iceberg(self, iceberg_summary, timestamp=None):
        ts = timestamp or time.time() * 1000
        alerts = []
        t = self.thresholds["iceberg"]
        n_active = iceberg_summary.get("n_active", 0)
        avg_conf = iceberg_summary.get("avg_confidence", 0)
        total_hidden = iceberg_summary.get("total_estimated_hidden", 0)
        if avg_conf >= t["min_confidence"] and n_active > 0:
            alerts.append(Alert(ts, AlertLevel.INFO, "iceberg", f"{n_active} icebergs (conf: {avg_conf:.2%})", avg_conf, t["min_confidence"], metadata=iceberg_summary))
        if total_hidden >= t["min_hidden_volume"]:
            alerts.append(Alert(ts, AlertLevel.WATCH, "iceberg", f"Large hidden vol: {total_hidden:,.0f}", total_hidden, t["min_hidden_volume"], metadata=iceberg_summary))
        for a in alerts:
            self._emit(a)
        return alerts

    def check_dark_volume(self, dark_summary, ticker=None, timestamp=None):
        ts = timestamp or time.time() * 1000
        alerts = []
        t = self.thresholds["dark_volume"]
        dark_share = dark_summary.get("dark_share_pct", 0)
        n_anomalies = dark_summary.get("n_anomalies", 0)
        avg_score = dark_summary.get("avg_anomaly_score", 0)
        if dark_share >= t["dark_share_pct"]:
            alerts.append(Alert(ts, AlertLevel.WARNING, "dark_volume", f"High dark share: {dark_share:.1f}%", dark_share, t["dark_share_pct"], ticker=ticker, metadata=dark_summary))
        if n_anomalies > 0 and avg_score > 0.5:
            alerts.append(Alert(ts, AlertLevel.WATCH, "dark_volume", f"{n_anomalies} anomalies", avg_score, 0.5, ticker=ticker, metadata=dark_summary))
        for a in alerts:
            self._emit(a)
        return alerts

    def run_checks(self, pipeline_results, ticker=None, timestamp=None):
        ts = timestamp or time.time() * 1000
        all_alerts = []
        if "vpin" in pipeline_results:
            all_alerts.extend(self.check_vpin(pipeline_results["vpin"], ts))
        if "iceberg" in pipeline_results:
            all_alerts.extend(self.check_iceberg(pipeline_results["iceberg"], ts))
        if "dark_analysis" in pipeline_results:
            all_alerts.extend(self.check_dark_volume(pipeline_results["dark_analysis"], ticker, ts))
        return all_alerts

    def _emit(self, alert):
        self._alerts.append(alert)
        if len(self._alerts) > self.max_history:
            self._alerts = self._alerts[-self.max_history // 2:]
        try:
            with open(self._alert_log_path, "a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
        except Exception:
            pass
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception:
                pass

    def get_recent(self, minutes=60, min_level=AlertLevel.INFO):
        now = time.time() * 1000
        window = now - minutes * 60_000
        lo = {AlertLevel.INFO: 0, AlertLevel.WATCH: 1, AlertLevel.WARNING: 2, AlertLevel.CRITICAL: 3}
        return [a for a in self._alerts if a.timestamp >= window and lo.get(a.level, 0) >= lo.get(min_level, 0)]

    def summary(self):
        if not self._alerts:
            return {"n_alerts": 0}
        df = pd.DataFrame([a.to_dict() for a in self._alerts])
        return {"n_alerts": len(self._alerts), "by_level": df["level"].value_counts().to_dict(),
                "by_module": df["module"].value_counts().to_dict(),
                "critical_count": sum(1 for a in self._alerts if a.level == AlertLevel.CRITICAL)}

    def export_csv(self, path=None):
        path = path or str(self.output_dir / "alerts_export.csv")
        if self._alerts:
            pd.DataFrame([a.to_dict() for a in self._alerts]).to_csv(path, index=False)
        return path

    def reset(self):
        self._alerts.clear()
