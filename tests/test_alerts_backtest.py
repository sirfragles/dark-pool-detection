"""Unit tests for Alert Engine and Backtest modules."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.alerts.engine import Alert, AlertEngine, AlertLevel
from src.backtest.engine import BacktestEngine


class TestAlert:
    def test_create_alert(self):
        a = Alert(1000000, AlertLevel.WARNING, "vpin", "VPIN elevated", 0.75, 0.6)
        assert a.module == "vpin" and a.level == AlertLevel.WARNING

    def test_to_dict(self):
        a = Alert(1000000, AlertLevel.CRITICAL, "ml", "Test", 0.9, 0.8, ticker="SPY")
        assert a.to_dict()["level"] == "critical" and a.to_dict()["ticker"] == "SPY"


class TestAlertEngine:
    def test_init_defaults(self, tmp_path):
        e = AlertEngine(output_dir=str(tmp_path))
        assert "vpin" in e.thresholds

    def test_check_vpin_toxic(self, tmp_path):
        e = AlertEngine(output_dir=str(tmp_path),
                       thresholds={"vpin": {"toxic": 0.8, "elevated": 0.6}})
        alerts = e.check_vpin({"current_vpin": 0.85, "toxic_pct": 50.0})
        assert any(a.level == AlertLevel.CRITICAL for a in alerts)

    def test_check_vpin_normal(self, tmp_path):
        e = AlertEngine(output_dir=str(tmp_path),
                       thresholds={"vpin": {"toxic": 0.9, "elevated": 0.7}})
        assert len(e.check_vpin({"current_vpin": 0.3, "toxic_pct": 0.0})) == 0

    def test_check_iceberg(self, tmp_path):
        e = AlertEngine(output_dir=str(tmp_path))
        alerts = e.check_iceberg({"n_active": 10, "avg_confidence": 0.85, "total_estimated_hidden": 100_000})
        assert len(alerts) >= 1

    def test_run_checks_integrates(self, pipeline_results, tmp_path):
        e = AlertEngine(output_dir=str(tmp_path))
        alerts = e.run_checks(pipeline_results, timestamp=1000.0)
        assert isinstance(alerts, list)

    def test_summary_after_alerts(self, pipeline_results, tmp_path):
        e = AlertEngine(output_dir=str(tmp_path))
        e.run_checks(pipeline_results)
        assert e.summary()["n_alerts"] > 0

    def test_reset(self, pipeline_results, tmp_path):
        e = AlertEngine(output_dir=str(tmp_path))
        e.run_checks(pipeline_results)
        assert e.summary()["n_alerts"] > 0
        e.reset()
        assert e.summary()["n_alerts"] == 0


class TestBacktestEngine:
    @pytest.fixture
    def price_data(self):
        n = 200
        return pd.DataFrame({
            "timestamp": np.arange(n * 1000, step=1000, dtype=float),
            "close": 100 + np.cumsum(np.random.randn(n) * 0.5),
            "price": 100 + np.cumsum(np.random.randn(n) * 0.5),
        }).set_index("timestamp")

    def test_init_defaults(self):
        e = BacktestEngine()
        assert e.initial_capital == 1_000_000.0

    def test_backtest_vpin_fade(self, price_data):
        sig = pd.Series(np.random.uniform(0.1, 0.9, len(price_data)), index=price_data.index)
        result = BacktestEngine().backtest_vpin_fade(price_data, sig, threshold=0.7)
        assert result["strategy"] == "vpin_fade" and "sharpe_ratio" in result

    def test_backtest_iceberg_frontrun(self, price_data):
        sig = pd.Series(np.random.choice([0, 1], len(price_data)), index=price_data.index)
        result = BacktestEngine().backtest_iceberg_frontrun(price_data, sig)
        assert result["strategy"] == "iceberg_frontrun"

    def test_insufficient_data_handled(self):
        tiny = pd.DataFrame({"close": [100.0]}).set_index(pd.Index([0.0]))
        sig = pd.Series([0.5], index=tiny.index)
        assert "error" in BacktestEngine().backtest_vpin_fade(tiny, sig)

    def test_to_dataframe(self, price_data):
        e = BacktestEngine()
        sig = pd.Series(np.random.uniform(0.1, 0.9, len(price_data)), index=price_data.index)
        e.backtest_vpin_fade(price_data, sig)
        e.backtest_iceberg_frontrun(price_data, sig)
        assert isinstance(e.to_dataframe(), pd.DataFrame)

    def test_empty_results_handled(self):
        assert "No backtest" in BacktestEngine().summary()
