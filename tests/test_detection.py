"""Unit tests for Detection Engine modules (Fazy 2-5)."""

import numpy as np
import pandas as pd
import pytest

from src.detection.dark_volume import DarkVolumeReconstructor
from src.detection.iceberg import IcebergDetector, IcebergOrder
from src.detection.trader_type import TraderTypeClassifier, TraderProfile
from src.detection.vpin import VPINCalculator


class TestIcebergDetector:
    def test_init_default_state(self):
        d = IcebergDetector()
        assert len(d.active_icebergs) == 0
        assert d.summary()["n_active"] == 0

    def test_detect_native_iceberg(self):
        d = IcebergDetector(min_detection_confidence=0.3)
        d.update_order_book(1000, bids=[(100.0, 500)], asks=[(100.0, 5000)])
        detected = d.process_trade(2000, 100.0, 100, "buy")
        assert len(detected) > 0
        assert detected[0].iceberg_type == "native"

    def test_no_false_positive_on_small_discrepancy(self):
        d = IcebergDetector(min_detection_confidence=0.6)
        d.update_order_book(1000, bids=[(100.0, 100)], asks=[(100.0, 105)])
        detected = d.process_trade(2000, 100.0, 100, "buy")
        assert len(detected) == 0

    def test_synthetic_iceberg_detection(self):
        d = IcebergDetector(min_detection_confidence=0.3, synthetic_window_ms=100)
        d.process_trade(1000, 100.0, 500, "buy")
        d.process_trade(1050, 100.0, 500, "buy")
        detected = d.process_trade(1090, 100.0, 500, "buy")
        syn = [ib for ib in detected if ib.iceberg_type == "synthetic"]
        assert len(syn) > 0

    def test_iceberg_order_dataclass(self):
        ib = IcebergOrder(id="test_1", timestamp=1000, price=99.50, displayed_size=100,
                         estimated_total=1500, hidden_size=1400, side="buy",
                         iceberg_type="native", confidence=0.85)
        assert ib.id == "test_1" and ib.hidden_size == 1400 and ib.active

    def test_summary_after_detections(self):
        d = IcebergDetector(min_detection_confidence=0.3)
        d.update_order_book(1000, bids=[(100.0, 5000)], asks=[(100.0, 100)])
        d.process_trade(2000, 100.0, 50, "sell")
        assert d.summary()["n_active"] > 0

    def test_reset_clears_state(self):
        d = IcebergDetector(min_detection_confidence=0.3)
        d.update_order_book(1000, bids=[(100.0, 5000)], asks=[(100.0, 100)])
        d.process_trade(2000, 100.0, 50, "sell")
        assert d.summary()["n_active"] > 0
        d.reset()
        assert d.summary()["n_active"] == 0


class TestVPINCalculator:
    def test_compute_returns_dataframe(self, sample_trades_df):
        vpin = VPINCalculator(volume_bucket_size=50)
        result = vpin.compute(sample_trades_df, price_col="price", volume_col="volume")
        assert isinstance(result, pd.DataFrame)
        assert "vpin" in result.columns

    def test_empty_data_handles_gracefully(self):
        result = VPINCalculator().compute(pd.DataFrame(), price_col="close", volume_col="volume")
        assert result.empty

    def test_vpin_in_range(self, simulated_data):
        vpin = VPINCalculator(volume_bucket_size=50)
        trades = simulated_data["trades"].copy()
        result = vpin.compute(trades, price_col="price", volume_col="volume")
        if not result.empty:
            assert (result["vpin"] >= 0).all() and (result["vpin"] <= 1).all()

    def test_is_toxic(self, simulated_data):
        vpin = VPINCalculator(volume_bucket_size=50)
        trades = simulated_data["trades"].copy()
        vpin.compute(trades, price_col="price", volume_col="volume")
        assert isinstance(vpin.is_toxic(threshold=0.8), bool)

    def test_streaming_update_returns_vpin(self):
        vpin = VPINCalculator(volume_bucket_size=5)
        results = []
        for i in range(10):
            v = vpin.update(price=100.0 + np.sin(i * 0.5) * 2, volume=10.0, timestamp=float(i * 1000))
            if v is not None:
                results.append(v)
        assert len(results) > 0

    def test_current_vpin_default_zero(self):
        assert VPINCalculator().current_vpin() == 0.0

    def test_summary_after_compute(self, simulated_data):
        vpin = VPINCalculator(volume_bucket_size=50)
        trades = simulated_data["trades"].copy()
        vpin.compute(trades, price_col="price", volume_col="volume")
        s = vpin.summary()
        assert "current_vpin" in s and "mean" in s


class TestDarkVolumeReconstructor:
    def test_init_defaults(self):
        r = DarkVolumeReconstructor()
        assert r.anomaly_threshold == 3.0 and r.min_dark_share == 0.05

    def test_reconstruct_returns_reports(self, simulated_data):
        r = DarkVolumeReconstructor()
        lit = simulated_data["full"].rename(columns={"price": "close"})
        reports = r.reconstruct(lit[lit["ticker"] == "STOCK_A"], ticker="STOCK_A")
        assert len(reports) > 0
        assert reports[0].ticker == "STOCK_A" and reports[0].dark_volume_est >= 0

    def test_reconstruct_batch_returns_dataframe(self, simulated_data):
        r = DarkVolumeReconstructor()
        lit = simulated_data["full"].rename(columns={"price": "close"})
        lit_data = {t: lit[lit["ticker"] == t] for t in lit["ticker"].unique()}
        result = r.reconstruct_batch(lit_data)
        assert isinstance(result, pd.DataFrame) and "dark_share" in result.columns

    def test_detect_anomaly_periods(self, simulated_data):
        r = DarkVolumeReconstructor(anomaly_threshold_std=0.5)
        lit = simulated_data["full"].rename(columns={"price": "close"})
        reports = r.reconstruct(lit[lit["ticker"] == "STOCK_A"], ticker="STOCK_A")
        anomalies = r.detect_anomaly_periods(reports, min_anomaly_score=0.3)
        assert isinstance(anomalies, pd.DataFrame)

    def test_empty_data_handles_gracefully(self):
        assert len(DarkVolumeReconstructor().reconstruct(pd.DataFrame(), ticker="EMPTY")) == 0

    def test_reset_clears_state(self, simulated_data):
        r = DarkVolumeReconstructor()
        lit = simulated_data["full"].rename(columns={"price": "close"})
        r.reconstruct(lit[lit["ticker"] == "STOCK_A"], ticker="STOCK_A")
        assert r.summary["n_reports"] > 0
        r.reset()
        assert r.summary["n_reports"] == 0


class TestTraderTypeClassifier:
    def test_init_gmm(self):
        c = TraderTypeClassifier(method="gmm", n_clusters=4)
        assert c.method == "gmm" and c.n_clusters == 4 and not c._is_fitted

    def test_extract_features_returns_dataframe(self, sample_trades_df):
        features = TraderTypeClassifier().extract_features(sample_trades_df)
        assert isinstance(features, pd.DataFrame) and len(features) > 0

    def test_fit_and_predict(self, simulated_data):
        c = TraderTypeClassifier(method="gmm", n_clusters=3)
        features = c.extract_features(simulated_data["trades"])
        if len(features) >= 3:
            c.fit(features)
            assert c._is_fitted
            profiles = c.predict(features)
            assert len(profiles) > 0
            assert all(isinstance(p, TraderProfile) for p in profiles)

    def test_empty_data_handled(self):
        assert TraderTypeClassifier().extract_features(pd.DataFrame()).empty

    def test_reset(self, simulated_data):
        c = TraderTypeClassifier(method="gmm", n_clusters=3)
        features = c.extract_features(simulated_data["trades"])
        if len(features) >= 3:
            c.fit(features)
            c.predict(features)
            assert c._is_fitted
            c.reset()
            assert not c._is_fitted
