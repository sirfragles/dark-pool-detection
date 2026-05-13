"""End-to-end integration tests."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.simulator import OrderBookSimulator
from src.detection.iceberg import IcebergDetector
from src.detection.vpin import VPINCalculator
from src.detection.dark_volume import DarkVolumeReconstructor
from src.detection.trader_type import TraderTypeClassifier
from src.pipeline import DarkPoolPipeline
from src.alerts.engine import AlertEngine
from src.backtest.engine import BacktestEngine
from src.utils import load_config


class TestPipelineIntegration:
    def test_pipeline_initialization(self):
        p = DarkPoolPipeline()
        assert p.config is not None and p.vpin is not None and p.iceberg is not None

    def test_run_simulation_completes(self, pipeline):
        results = pipeline.run_simulation(n_ticks=390, n_tickers=3, seed=42)
        for key in ["simulation", "vpin", "iceberg", "dark_analysis", "detection_score"]:
            assert key in results

    def test_detection_score_in_range(self, pipeline_results):
        score = pipeline_results["detection_score"]
        assert 0 <= score["overall"] <= 100

    def test_report_generated(self, pipeline):
        pipeline.run_simulation(n_ticks=390, n_tickers=3, seed=42)
        r = pipeline.report()
        assert "DARK POOL DETECTION" in r and "VPIN" in r

    def test_pipeline_reproducibility(self):
        r1 = DarkPoolPipeline().run_simulation(n_ticks=200, n_tickers=2, seed=42)
        r2 = DarkPoolPipeline().run_simulation(n_ticks=200, n_tickers=2, seed=42)
        assert r1["simulation"]["total_trades"] == r2["simulation"]["total_trades"]

    def test_vpin_values_in_range(self, pipeline_results):
        vpin = pipeline_results["vpin"]
        assert 0 <= vpin["current_vpin"] <= 1 and 0 <= vpin["mean"] <= 1

    def test_iceberg_detection_active(self, pipeline_results):
        ice = pipeline_results["iceberg"]
        assert ice["n_active"] >= 0 and "n_native" in ice


class TestFullWorkflow:
    def test_complete_workflow(self):
        t0 = time.time()
        sim = OrderBookSimulator(n_tickers=3, n_ticks=500, seed=42)
        df = sim.generate()
        trades = df[df["volume"] > 0]
        assert len(trades) > 0

        pipeline = DarkPoolPipeline()
        results = pipeline.run_simulation(n_ticks=500, n_tickers=3, seed=42)
        assert results["detection_score"]["overall"] >= 0

        tc = TraderTypeClassifier(n_clusters=3)
        features = tc.extract_features(trades)
        if len(features) >= 3:
            tc.fit(features)
            assert len(tc.predict(features)) > 0

        dv = DarkVolumeReconstructor()
        lit = df.rename(columns={"price": "close"})
        reports = dv.reconstruct(lit[lit["ticker"] == "STOCK_A"], ticker="STOCK_A")
        assert len(reports) > 0

        ae = AlertEngine()
        alerts = ae.run_checks(results)
        assert isinstance(alerts, list)

        elapsed = time.time() - t0
        assert elapsed < 60, f"Workflow too slow: {elapsed:.1f}s"


class TestConfig:
    def test_load_config_returns_dict(self):
        cfg = load_config()
        assert isinstance(cfg, dict) and "system" in cfg and "detection" in cfg

    def test_config_has_required_sections(self):
        cfg = load_config()
        for section in ["system", "data_sources", "detection", "ml", "simulation"]:
            assert section in cfg


class TestCrossModule:
    def test_simulator_output_compatible_with_detection(self):
        sim = OrderBookSimulator(n_tickers=2, n_ticks=200, seed=42)
        df = sim.generate()
        trades = df[df["volume"] > 0]
        vpin = VPINCalculator()
        result = vpin.compute(trades, price_col="price", volume_col="volume")
        assert isinstance(result, pd.DataFrame)

    def test_trader_type_reads_simulator_output(self):
        sim = OrderBookSimulator(n_tickers=3, n_ticks=300, seed=42)
        trades = sim.generate()[sim.generate()["volume"] > 0]
        features = TraderTypeClassifier().extract_features(trades)
        assert len(features.columns) > 0

    def test_dark_volume_reads_lit_data(self):
        sim = OrderBookSimulator(n_tickers=2, n_ticks=100, seed=42)
        lit = sim.generate().rename(columns={"price": "close"})
        for ticker in lit["ticker"].unique():
            reports = DarkVolumeReconstructor().reconstruct(lit[lit["ticker"] == ticker], ticker=ticker)
            assert len(reports) > 0


class TestPerformance:
    def test_simulation_speed_small(self):
        t0 = time.time()
        OrderBookSimulator(n_tickers=3, n_ticks=500, seed=42).generate()
        assert time.time() - t0 < 5

    def test_full_pipeline_speed(self):
        t0 = time.time()
        DarkPoolPipeline().run_simulation(n_ticks=500, n_tickers=3, seed=42)
        assert time.time() - t0 < 10


class TestErrorHandling:
    def test_empty_dataframe_everywhere(self):
        empty = pd.DataFrame()
        assert VPINCalculator().compute(empty, "price", "volume").empty
        assert len(DarkVolumeReconstructor().reconstruct(empty)) == 0
        assert TraderTypeClassifier().extract_features(empty).empty

    def test_negative_prices_handled(self):
        df = pd.DataFrame({"price": [100.0, -50.0, 90.0], "volume": [10.0, 20.0, 30.0]})
        VPINCalculator().compute(df, "price", "volume")
