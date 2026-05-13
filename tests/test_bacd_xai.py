"""Unit tests for Faza 9-10: BACD + XAI Explainability."""

import time

import numpy as np
import pandas as pd
import pytest

from src.detection.bacd import BACDAnalyzer, BACDProfile, BurstProfile


class TestBACDAnalyzer:
    def test_init_defaults(self):
        a = BACDAnalyzer()
        assert a.burst_threshold_ms == 100
        assert a.burst_min_trades == 3

    def test_insufficient_data_returns_none(self):
        tiny = pd.DataFrame({"timestamp": [0, 1000, 2000], "volume": [10, 20, 30], "price": [100.0, 101.0, 102.0]})
        assert BACDAnalyzer().analyze_session(tiny) is None

    def test_analyze_session_returns_profile(self, sample_trades_df):
        a = BACDAnalyzer(min_trades=5, burst_threshold_ms=500)
        profile = a.analyze_session(sample_trades_df.head(20))
        if profile is not None:
            assert isinstance(profile, BACDProfile)
            assert profile.interval_mean > 0
            assert profile.weibull_shape > 0

    def test_analyze_session_with_bursts(self):
        a = BACDAnalyzer(burst_threshold_ms=50, burst_min_trades=3, min_trades=5)
        timestamps = np.array([0, 30, 60, 5000, 5030, 5060, 10000], dtype=float)
        df = pd.DataFrame({"timestamp": timestamps, "volume": [100]*7, "price": [100.0]*7})
        profile = a.analyze_session(df)
        assert profile is not None
        assert profile.n_bursts >= 1
        assert profile.burst_ratio > 0

    def test_to_feature_vector(self, sample_trades_df):
        a = BACDAnalyzer(min_trades=5)
        profile = a.analyze_session(sample_trades_df.head(20))
        vec = a.to_feature_vector(profile)
        assert isinstance(vec, dict)
        for name in a.FEATURE_NAMES:
            assert name in vec

    def test_to_feature_vector_none(self):
        vec = BACDAnalyzer().to_feature_vector(None)
        assert all(v == 0.0 for v in vec.values())

    def test_weibull_fit_goodness(self):
        a = BACDAnalyzer(min_trades=5)
        np.random.seed(42)
        intervals = np.random.exponential(1000, 100)
        times = np.cumsum(intervals)
        df = pd.DataFrame({"timestamp": times, "volume": [100]*100, "price": 100 + np.random.randn(100)*2})
        profile = a.analyze_session(df)
        assert profile is not None
        assert profile.weibull_r2 > 0.5

    def test_diurnal_pattern(self):
        a = BACDAnalyzer(min_trades=5)
        morning_ms = 10 * 3_600_000
        times = np.arange(0, 10000, 1000) + morning_ms
        df = pd.DataFrame({"timestamp": times, "volume": [100]*len(times), "price": [100.0]*len(times)})
        profile = a.analyze_session(df)
        assert profile is not None
        assert profile.morning_activity > 0.9

    def test_burstiness_index(self):
        a = BACDAnalyzer(min_trades=5)
        regular = pd.DataFrame({"timestamp": np.arange(0, 50000, 1000, dtype=float), "volume": [100]*50, "price": [100.0]*50})
        profile = a.analyze_session(regular)
        assert profile is not None
        assert profile.burstiness_index < 0

    def test_performance(self):
        a = BACDAnalyzer(min_trades=5)
        df = pd.DataFrame({"timestamp": np.cumsum(np.random.exponential(1000, 1000)),
                          "volume": np.random.randint(1, 1000, 1000).astype(float),
                          "price": 100 + np.random.randn(1000)*2})
        t0 = time.time()
        profile = a.analyze_session(df)
        elapsed = time.time() - t0
        assert profile is not None
        assert elapsed < 1.0, f"BACD too slow: {elapsed:.3f}s"


class TestExplainability:
    def test_decompose_detection_score(self, pipeline_results):
        from src.ml.explainability import ModelExplainer
        decomp = ModelExplainer().decompose_detection_score(pipeline_results)
        assert "overall_score" in decomp
        assert len(decomp["module_contributions"]) == 3

    def test_quick_explainer_feature_importance(self, large_simulated_data):
        from src.ml.explainability import QuickExplainer
        from src.ml.iceberg_predictor import IcebergPredictor
        df = large_simulated_data["full"]
        trades = large_simulated_data["trades"]
        iceberg_rows = df[df["iceberg_active"]].drop_duplicates(subset=["ticker", "timestamp"])
        if len(iceberg_rows) >= 50:
            ip = IcebergPredictor(prediction_horizon=10)
            X = ip.extract_features(iceberg_rows)
            y = ip.build_labels(iceberg_rows, trades)
            ip.train(X, y, save=False)
            fi = QuickExplainer().feature_importance_xgboost(ip.model)
            assert isinstance(fi, dict) and len(fi) > 0

    def test_api_explain_endpoint(self):
        from src.web.app import app
        with app.test_client() as c:
            c.post("/run", data={"n_tickers": "2", "n_ticks": "200", "seed": "42"})
            resp = c.get("/api/explain")
            assert resp.status_code == 200
            assert "score_decomposition" in resp.get_json()

    def test_api_duration_endpoint(self):
        from src.web.app import app
        with app.test_client() as c:
            resp = c.get("/api/duration?ticks=500&seed=42")
            assert resp.status_code == 200
            assert len(resp.get_json()) > 0


class TestBACDIntegration:
    def test_trader_type_uses_bacd_features(self, simulated_data):
        from src.detection.trader_type import TraderTypeClassifier
        tc = TraderTypeClassifier()
        features = tc.extract_features(simulated_data["trades"], use_bacd=True)
        assert isinstance(features, pd.DataFrame) and len(features) > 0
        bacd_cols = [c for c in features.columns if c.startswith("bacd_")]
        assert len(bacd_cols) > 0

    def test_trader_type_works_without_bacd(self, simulated_data):
        from src.detection.trader_type import TraderTypeClassifier
        features = TraderTypeClassifier().extract_features(simulated_data["trades"], use_bacd=False)
        assert isinstance(features, pd.DataFrame) and len(features) > 0
