"""Unit tests for ML Prediction Layer (Faza 6)."""

import numpy as np
import pandas as pd
import pytest

from src.ml.dark_trade_predictor import DarkTradePredictor
from src.ml.iceberg_predictor import IcebergPredictor
from src.ml.trader_fingerprint import TraderFingerprint


class TestIcebergPredictor:
    def test_init_defaults(self, tmp_path):
        p = IcebergPredictor(model_dir=str(tmp_path / "models"))
        assert p.n_estimators == 100 and p.prediction_horizon == 10 and not p.is_trained

    def test_extract_features_returns_dataframe(self, simulated_data):
        p = IcebergPredictor()
        iceberg_rows = simulated_data["full"][simulated_data["full"]["iceberg_active"]]\
            .drop_duplicates(subset=["ticker", "timestamp"])
        X = p.extract_features(iceberg_rows)
        assert isinstance(X, pd.DataFrame) and len(X) > 0

    def test_build_labels_returns_binary(self, simulated_data):
        p = IcebergPredictor()
        iceberg_rows = simulated_data["full"][simulated_data["full"]["iceberg_active"]]\
            .drop_duplicates(subset=["ticker", "timestamp"])
        y = p.build_labels(iceberg_rows, simulated_data["trades"], horizon=10)
        assert y.isin([0, 1]).all()

    def test_train_and_predict(self, large_simulated_data):
        p = IcebergPredictor(prediction_horizon=10)
        df = large_simulated_data["full"]
        trades = large_simulated_data["trades"]
        iceberg_rows = df[df["iceberg_active"]].drop_duplicates(subset=["ticker", "timestamp"])
        if len(iceberg_rows) >= 50:
            X = p.extract_features(iceberg_rows)
            y = p.build_labels(iceberg_rows, trades)
            metrics = p.train(X, y, save=False)
            assert p.is_trained and "cv_accuracy_mean" in metrics
            preds = p.predict(X.head(10))
            assert len(preds) == min(10, len(X))

    def test_untrained_predict_returns_zeros(self):
        p = IcebergPredictor()
        X = pd.DataFrame({"displayed_size": [100, 200], "estimated_total": [1000, 2000]})
        assert (p.predict(X) == 0).all()


class TestDarkTradePredictor:
    def test_init_defaults(self):
        p = DarkTradePredictor()
        assert p.sequence_length == 50 and not p.is_trained

    def test_build_sequences_returns_arrays(self, large_simulated_data):
        p = DarkTradePredictor(sequence_length=10)
        X, y, ts = p.build_sequences(large_simulated_data["trades"], label_col="is_dark")
        assert isinstance(X, np.ndarray) and X.ndim == 3 and X.shape[1] == 10

    def test_train_fallback(self, large_simulated_data):
        p = DarkTradePredictor(sequence_length=10)
        X, y, _ = p.build_sequences(large_simulated_data["trades"], label_col="is_dark")
        if len(X) >= 50:
            metrics = p.train(X, y, epochs=5, save=False)
            assert p.is_trained
            assert "final_accuracy" in metrics or "validation_accuracy" in metrics

    def test_predict_after_training(self, large_simulated_data):
        p = DarkTradePredictor(sequence_length=10)
        X, y, _ = p.build_sequences(large_simulated_data["trades"], label_col="is_dark")
        if len(X) >= 50:
            p.train(X, y, epochs=5, save=False)
            probs = p.predict(X[:5])
            assert len(probs) == 5 and all(0 <= prob <= 1 for prob in probs)

    def test_untrained_predict_returns_zeros(self):
        p = DarkTradePredictor(sequence_length=10)
        assert (p.predict(np.zeros((3, 10, 8), dtype=np.float32)) == 0).all()


class TestTraderFingerprint:
    def test_init_kmeans(self):
        fp = TraderFingerprint(n_clusters=4, method="kmeans")
        assert fp.n_clusters == 4

    def test_extract_features(self, simulated_data):
        features = TraderFingerprint().extract_features(simulated_data["trades"])
        assert isinstance(features, pd.DataFrame) and len(features) > 0

    def test_fit_kmeans(self, large_simulated_data):
        fp = TraderFingerprint(n_clusters=3, method="kmeans")
        features = fp.extract_features(large_simulated_data["trades"])
        if len(features) >= 3:
            fp.fit(features)
            assert fp._is_fitted and len(fp._fingerprints) > 0

    def test_empty_data_handled(self):
        assert TraderFingerprint().extract_features(pd.DataFrame()).empty

    def test_insufficient_data_for_clusters(self):
        fp = TraderFingerprint(n_clusters=5)
        fp.fit(pd.DataFrame({"trade_size_mean": [1.0, 2.0], "dark_share": [0.1, 0.2]}))
        assert not fp._is_fitted

    def test_reset(self, large_simulated_data):
        fp = TraderFingerprint(n_clusters=3, method="kmeans")
        features = fp.extract_features(large_simulated_data["trades"])
        if len(features) >= 3:
            fp.fit(features)
            assert fp._is_fitted
            fp.reset()
            assert not fp._is_fitted
