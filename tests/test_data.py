"""Unit tests for Data Pipeline modules (Faza 1)."""

import numpy as np
import pandas as pd
import pytest

from src.data.simulator import OrderBookSimulator
from src.data.yfinance_feed import YFinanceFeed


class TestOrderBookSimulator:
    def test_generates_correct_shape(self, simulated_data):
        df = simulated_data["full"]
        assert len(df) == simulated_data["n_tickers"] * simulated_data["n_ticks"]

    def test_required_columns_present(self, simulated_data):
        required = ["timestamp", "ticker", "price", "bid", "ask", "bid_size", "ask_size",
                    "volume", "trade_side", "is_dark", "is_iceberg", "trader_type"]
        for col in required:
            assert col in simulated_data["full"].columns

    def test_has_trades(self, simulated_data):
        assert len(simulated_data["trades"]) > 0

    def test_has_dark_trades(self, simulated_data):
        assert simulated_data["trades"]["is_dark"].sum() > 0

    def test_has_iceberg_trades(self, simulated_data):
        assert simulated_data["trades"]["is_iceberg"].sum() > 0

    def test_trader_types_present(self, simulated_data):
        types = simulated_data["trades"]["trader_type"].unique()
        assert len(types) >= 2

    def test_price_nonnegative(self, simulated_data):
        assert (simulated_data["full"]["price"] > 0).all()

    def test_bid_below_ask(self, simulated_data):
        assert (simulated_data["full"]["bid"] <= simulated_data["full"]["ask"]).all()

    def test_reproducibility(self):
        sim1 = OrderBookSimulator(n_tickers=2, n_ticks=100, seed=42)
        sim2 = OrderBookSimulator(n_tickers=2, n_ticks=100, seed=42)
        pd.testing.assert_frame_equal(sim1.generate(), sim2.generate())

    def test_different_seeds_differ(self):
        df1 = OrderBookSimulator(n_tickers=2, n_ticks=100, seed=1).generate()
        df2 = OrderBookSimulator(n_tickers=2, n_ticks=100, seed=999).generate()
        assert not df1.equals(df2)

    def test_custom_initial_prices(self):
        sim = OrderBookSimulator(n_tickers=2, n_ticks=50,
                                 initial_prices={"STOCK_A": 50.0, "STOCK_B": 200.0}, seed=42)
        df = sim.generate()
        prices = df.groupby("ticker")["price"].first()
        assert 20 < prices["STOCK_A"] < 150
        assert 100 < prices["STOCK_B"] < 500

    def test_summary_consistent(self, simulated_data):
        trades = simulated_data["trades"]
        summary = simulated_data["summary"]
        assert summary["total_trades"] == len(trades)
        assert summary["dark_trades"] == int(trades["is_dark"].sum())

    def test_single_ticker(self):
        sim = OrderBookSimulator(n_tickers=1, n_ticks=50, seed=42)
        assert sim.generate()["ticker"].nunique() == 1

    def test_many_tickers(self):
        sim = OrderBookSimulator(n_tickers=10, n_ticks=100, seed=42)
        assert sim.generate()["ticker"].nunique() == 10


class TestYFinanceFeed:
    def test_compute_lit_baseline_adds_features(self):
        feed = YFinanceFeed()
        df = pd.DataFrame({
            "timestamp": np.arange(0, 100000, 1000, dtype=float),
            "open": 100 + np.random.randn(100) * 2,
            "high": 102 + np.random.randn(100) * 2,
            "low": 98 + np.random.randn(100) * 2,
            "close": 100 + np.random.randn(100) * 2,
            "volume": np.random.randint(1000, 100000, 100).astype(float),
            "ticker": "TEST",
        })
        result = feed.compute_lit_baseline(df)
        for col in ["volume_ma", "volume_std", "volume_zscore", "returns"]:
            assert col in result.columns

    def test_detect_volume_anomalies(self):
        feed = YFinanceFeed()
        df = pd.DataFrame({
            "timestamp": np.arange(0, 100000, 1000, dtype=float),
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "volume": [1000] * 98 + [100_000, 200_000],
            "ticker": "TEST",
        })
        df = feed.compute_lit_baseline(df)
        anomalies = feed.detect_volume_anomalies(df, zscore_threshold=2.0)
        assert len(anomalies) >= 2

    def test_estimate_dark_volume(self):
        feed = YFinanceFeed()
        df = pd.DataFrame({"volume": [1000.0] * 50})
        result = feed.estimate_dark_volume(df)
        assert result["lit_volume"] == 50000.0
        assert result["dark_share_pct"] > 0

    def test_estimate_dark_volume_with_total(self):
        feed = YFinanceFeed()
        df = pd.DataFrame({"volume": [650.0]})
        result = feed.estimate_dark_volume(df, total_volume_estimate=1000.0)
        assert result["estimated_dark_volume"] == 350.0

    def test_empty_fetch_handles_gracefully(self):
        feed = YFinanceFeed()
        data = feed.fetch_intraday(tickers=["NIEPOPRAWNY_TICKER_XYZ123"], interval="1m", period="1d", use_cache=False)
        assert isinstance(data, dict)

    def test_available_tickers_after_fetch(self):
        feed = YFinanceFeed()
        feed._data["FAKE"] = pd.DataFrame({"volume": [100.0], "close": [100.0], "ticker": "FAKE"})
        assert "FAKE" in feed.available_tickers

    def test_get_data_returns_none_for_missing(self):
        assert YFinanceFeed().get_data("NIEPOPRAWNY") is None
