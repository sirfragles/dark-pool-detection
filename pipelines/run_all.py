#!/usr/bin/env python3
"""Dark Pool Detection — End-to-End Pipeline Runner.

Usage:
    python pipelines/run_all.py                    # Full pipeline
    python pipelines/run_all.py --live --tickers AAPL MSFT
    python pipelines/run_all.py --sim-only
    python pipelines/run_all.py --backtest-only
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.alerts.engine import AlertEngine
from src.backtest.engine import BacktestEngine
from src.data.simulator import OrderBookSimulator
from src.data.yfinance_feed import YFinanceFeed
from src.detection.dark_volume import DarkVolumeReconstructor
from src.detection.trader_type import TraderTypeClassifier
from src.ml.dark_trade_predictor import DarkTradePredictor
from src.ml.iceberg_predictor import IcebergPredictor
from src.ml.trader_fingerprint import TraderFingerprint
from src.pipeline import DarkPoolPipeline
from src.utils import ensure_dirs, get_logger, load_config

logger = get_logger(__name__)


def run_simulation_pipeline(n_ticks=5000, n_tickers=5, seed=42, save_report=True):
    config = load_config()
    ensure_dirs(config)
    output_dir = Path(config["system"]["output_dir"])
    
    logger.info("Step 1/6: Generating synthetic market data...")
    sim = OrderBookSimulator(n_tickers=n_tickers, n_ticks=n_ticks, seed=seed)
    df = sim.generate()
    trades = df[df["volume"] > 0]
    sim_summary = sim.generate_summary(df)
    
    logger.info("Step 2/6: Running detection modules...")
    pipeline = DarkPoolPipeline()
    pipeline._data = df
    results = pipeline.run_simulation(n_ticks=n_ticks, n_tickers=n_tickers, seed=seed)
    
    logger.info("Step 3/6: ML prediction layer...")
    iceberg_predictor = IcebergPredictor()
    iceberg_rows = df[df["iceberg_active"]].drop_duplicates(subset=["ticker", "timestamp"])
    ml_iceberg_metrics = {}
    if len(iceberg_rows) > 100:
        X_ice = iceberg_predictor.extract_features(iceberg_rows)
        y_ice = iceberg_predictor.build_labels(iceberg_rows, trades)
        ml_iceberg_metrics = iceberg_predictor.train(X_ice, y_ice, save=False)
    
    dark_predictor = DarkTradePredictor()
    X_seq, y_seq, _ = dark_predictor.build_sequences(trades, label_col="is_dark")
    ml_dark_metrics = {}
    if len(X_seq) > 100:
        ml_dark_metrics = dark_predictor.train(X_seq, y_seq, epochs=10, save=False)
    
    fingerprint = TraderFingerprint(n_clusters=4)
    fp_features = fingerprint.extract_features(trades)
    if len(fp_features) >= 4:
        fingerprint.fit(fp_features)
    
    logger.info("Step 4/6: Trader type classification...")
    classifier = TraderTypeClassifier()
    tc_features = classifier.extract_features(trades)
    classifier.fit(tc_features)
    classifier.predict(tc_features)
    
    logger.info("Step 5/6: Dark volume reconstruction...")
    dark_recon = DarkVolumeReconstructor()
    lit_df = df[["timestamp", "ticker", "price", "volume"]].rename(columns={"price": "close"})
    for ticker in df["ticker"].unique():
        dark_recon.reconstruct(lit_df[lit_df["ticker"] == ticker], ticker=ticker)
    
    logger.info("Step 6/6: Generating alerts...")
    alert_engine = AlertEngine()
    alert_engine.run_checks(results)
    
    if save_report:
        json_path = output_dir / "summary.json"
        with open(json_path, "w") as f:
            json.dump(_make_serializable(results), f, indent=2, default=str)
        with open(output_dir / "report.txt", "w") as f:
            f.write(pipeline.report())
    
    return results


def run_live_pipeline(tickers, interval="5m", period="1d"):
    feed = YFinanceFeed()
    lit_data = feed.fetch_intraday(tickers=tickers, interval=interval, period=period)
    if not lit_data:
        return {}
    recon = DarkVolumeReconstructor()
    ticker_results = {}
    for ticker, df in lit_data.items():
        recon.reconstruct(df, ticker=ticker)
        ticker_results[ticker] = recon.summary
    return {"tickers": tickers, "results": ticker_results}


def run_backtest_pipeline(n_ticks=10000, n_tickers=1, seed=42):
    sim = OrderBookSimulator(n_tickers=n_tickers, n_ticks=n_ticks, seed=seed)
    df = sim.generate()
    price = df[["timestamp", "price"]].set_index("timestamp")
    vpin_sig = pd.Series(np.random.uniform(0.2, 0.9, len(df)), index=df.index)
    iceberg_sig = df["is_iceberg"].astype(int)
    dark_z = (df["volume"] - df["volume"].mean()) / df["volume"].std()
    engine = BacktestEngine()
    engine.backtest_vpin_fade(price, vpin_sig, threshold=0.7)
    engine.backtest_iceberg_frontrun(price, iceberg_sig)
    engine.backtest_dark_volume_fade(price, dark_z, threshold=2.0)
    return engine.results


def _make_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(i) for i in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Dark Pool Detection — E2E Pipeline")
    parser.add_argument("--mode", choices=["full", "sim", "live", "backtest", "dashboard"], default="full")
    parser.add_argument("--ticks", type=int, default=5000)
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "AAPL"])
    parser.add_argument("--n-tickers", type=int, default=5)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--period", default="1d")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    
    if args.mode == "dashboard":
        os.system(f"streamlit run {Path(__file__).parent.parent / 'src' / 'dashboard' / 'app.py'}")
        return
    
    t0 = time.time()
    
    if args.mode in ("full", "sim"):
        results = run_simulation_pipeline(n_ticks=args.ticks, n_tickers=args.n_tickers,
                                          seed=args.seed, save_report=not args.no_save)
        print(f"Detection Score: {results['detection_score']['overall']:.1f}/100")
        print(f"Duration: {time.time() - t0:.1f}s")
    
    if args.mode == "live":
        live = run_live_pipeline(tickers=args.tickers, interval=args.interval, period=args.period)
        for ticker, data in live.get("results", {}).items():
            print(f"{ticker}: dark_share={data.get('dark_share_pct', 0):.1f}%")
    
    if args.mode == "backtest":
        bt = run_backtest_pipeline(n_ticks=args.ticks, n_tickers=args.n_tickers, seed=args.seed)
        for name, m in bt.items():
            print(f"{name}: Sharpe={m.get('sharpe_ratio', 0):.3f}")


if __name__ == "__main__":
    main()
