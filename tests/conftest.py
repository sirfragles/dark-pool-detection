"""Shared fixtures and configuration for Dark Pool Detection tests."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def config():
    from src.utils import load_config
    return load_config()


@pytest.fixture(scope="module")
def simulated_data():
    from src.data.simulator import OrderBookSimulator
    sim = OrderBookSimulator(n_tickers=3, n_ticks=1000, seed=42)
    df = sim.generate()
    trades = df[df["volume"] > 0]
    return {"full": df, "trades": trades, "summary": sim.generate_summary(df),
            "n_tickers": 3, "n_ticks": 1000, "seed": 42}


@pytest.fixture(scope="module")
def large_simulated_data():
    from src.data.simulator import OrderBookSimulator
    sim = OrderBookSimulator(n_tickers=5, n_ticks=2000, seed=42)
    df = sim.generate()
    trades = df[df["volume"] > 0]
    return {"full": df, "trades": trades}


@pytest.fixture
def pipeline():
    from src.pipeline import DarkPoolPipeline
    return DarkPoolPipeline()


@pytest.fixture
def pipeline_results(pipeline):
    return pipeline.run_simulation(n_ticks=390, n_tickers=3, seed=42)


@pytest.fixture
def sample_trades_df():
    return pd.DataFrame({
        "timestamp": np.arange(0, 100000, 1000, dtype=float),
        "ticker": ["STOCK_A"] * 50 + ["STOCK_B"] * 50,
        "price": np.concatenate([
            100 + np.cumsum(np.random.randn(50) * 1.0),
            200 + np.cumsum(np.random.randn(50) * 1.5),
        ]),
        "volume": np.random.randint(1, 1000, 100).astype(float),
        "is_dark": np.random.choice([True, False], 100, p=[0.3, 0.7]),
        "is_iceberg": np.random.choice([True, False], 100, p=[0.15, 0.85]),
        "trade_side": np.random.choice(["buy", "sell"], 100),
        "trader_type": np.random.choice(
            ["retail", "institution", "informed", "noise"], 100,
            p=[0.5, 0.2, 0.15, 0.15]
        ),
    })


@pytest.fixture
def sample_orderbook_df():
    return pd.DataFrame({
        "timestamp": np.arange(0, 50000, 1000, dtype=float),
        "ticker": ["STOCK_A"] * 50,
        "price": 100 + np.cumsum(np.random.randn(50) * 0.5),
        "bid": 99 + np.cumsum(np.random.randn(50) * 0.5),
        "ask": 101 + np.cumsum(np.random.randn(50) * 0.5),
        "bid_size": np.random.randint(100, 1000, 50),
        "ask_size": np.random.randint(100, 1000, 50),
        "volume": np.random.randint(0, 500, 50).astype(float),
        "trade_side": np.random.choice(["buy", "sell", ""], 50),
    })
