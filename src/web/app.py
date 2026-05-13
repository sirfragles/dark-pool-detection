"""Dark Pool Detection — Flask Web Interface."""

import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import Flask, jsonify, render_template, request, send_file, url_for

from src.alerts.engine import AlertEngine
from src.backtest.engine import BacktestEngine
from src.data.simulator import OrderBookSimulator
from src.data.yfinance_feed import YFinanceFeed
from src.detection.dark_volume import DarkVolumeReconstructor
from src.detection.iceberg import IcebergDetector
from src.detection.trader_type import TraderTypeClassifier
from src.detection.vpin import VPINCalculator
from src.ml.dark_trade_predictor import DarkTradePredictor
from src.ml.iceberg_predictor import IcebergPredictor
from src.ml.trader_fingerprint import TraderFingerprint
from src.pipeline import DarkPoolPipeline
from src.utils import load_config

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dark-pool-dev-key")

_last_results: dict = {}
_last_alerts: list = []
_alert_engine = AlertEngine()


@app.route("/")
def index():
    return render_template("index.html", results=_last_results)


@app.route("/run", methods=["POST"])
def run_pipeline():
    global _last_results, _last_alerts, _alert_engine
    n_tickers = int(request.form.get("n_tickers", 5))
    n_ticks = int(request.form.get("n_ticks", 2000))
    seed = int(request.form.get("seed", 42))
    t0 = time.time()
    try:
        pipeline = DarkPoolPipeline()
        results = pipeline.run_simulation(n_ticks=n_ticks, n_tickers=n_tickers, seed=seed)
        _alert_engine = AlertEngine()
        _last_alerts = _alert_engine.run_checks(results)
        elapsed = time.time() - t0
        results["_meta"] = {"elapsed": round(elapsed, 2), "n_tickers": n_tickers,
                            "n_ticks": n_ticks, "seed": seed,
                            "timestamp": pd.Timestamp.now().isoformat()}
        _last_results = results
        return jsonify({"status": "ok", "score": results["detection_score"]["overall"], "elapsed": elapsed})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/results")
def view_results():
    if not _last_results:
        return render_template("results.html", results=None)
    r = _last_results
    return render_template("results.html", results=r, sim=r.get("simulation", {}),
                          vpin=r.get("vpin", {}), iceberg=r.get("iceberg", {}),
                          dark=r.get("dark_analysis", {}), score=r.get("detection_score", {}),
                          meta=r.get("_meta", {}), trader=r.get("trader_analysis", {}),
                          alerts=_last_alerts)


@app.route("/alerts")
def view_alerts():
    alert_summary = _alert_engine.summary()
    return render_template("alerts.html", alerts=_last_alerts, summary=alert_summary)


@app.route("/live")
def live_page():
    return render_template("live.html")


@app.route("/api/results")
def api_results():
    if not _last_results:
        return jsonify({"error": "No results yet"}), 404
    return jsonify(_make_serializable(_last_results))


@app.route("/api/alerts")
def api_alerts():
    summary = _alert_engine.summary()
    recent = [a.to_dict() for a in _alert_engine.get_recent(minutes=120)]
    return jsonify({"summary": summary, "recent": recent})


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    data = request.get_json(force=True, silent=True) or {}
    n_tickers = data.get("n_tickers", 3)
    n_ticks = data.get("n_ticks", 1000)
    seed = data.get("seed", 42)
    sim = OrderBookSimulator(n_tickers=n_tickers, n_ticks=n_ticks, seed=seed)
    df = sim.generate()
    summary = sim.generate_summary(df)
    return jsonify({"summary": summary, "n_rows": len(df),
                   "sample_rows": df.head(20).to_dict(orient="records")})


@app.route("/api/vpin")
def api_vpin():
    n_ticks = int(request.args.get("ticks", 1000))
    seed = int(request.args.get("seed", 42))
    sim = OrderBookSimulator(n_tickers=1, n_ticks=n_ticks, seed=seed)
    df = sim.generate()
    trades = df[df["volume"] > 0]
    vpin = VPINCalculator(volume_bucket_size=50)
    vpin_df = vpin.compute(trades, price_col="price", volume_col="volume")
    return jsonify({
        "vpin_series": vpin_df[["time", "vpin"]].tail(50).to_dict(orient="records") if not vpin_df.empty else [],
        "summary": vpin.summary(),
    })


@app.route("/api/live", methods=["POST"])
def api_live():
    data = request.get_json(force=True, silent=True) or {}
    tickers = data.get("tickers", ["SPY", "QQQ"])
    interval = data.get("interval", "5m")
    period = data.get("period", "1d")
    feed = YFinanceFeed()
    lit_data = feed.fetch_intraday(tickers=tickers, interval=interval, period=period)
    result = {}
    recon = DarkVolumeReconstructor()
    for ticker, df in lit_data.items():
        reports = recon.reconstruct(df, ticker=ticker)
        anomalies = recon.detect_anomaly_periods(reports)
        result[ticker] = {
            "n_bars": len(df),
            "total_volume": float(df["volume"].sum()) if "volume" in df.columns else 0,
            "dark_volume_est": recon.summary.get("total_dark_volume", 0),
            "dark_share_pct": recon.summary.get("dark_share_pct", 0),
            "n_anomalies": len(anomalies),
        }
    return jsonify(result)


@app.route("/api/health")
def api_health():
    return jsonify({"status": "healthy", "version": "0.1.0",
                   "modules": {"detection": True, "ml": True, "alerts": True, "backtest": True}})


@app.route("/api/report")
def api_report():
    if not _last_results:
        return jsonify({"error": "No report available"}), 404
    pipeline = DarkPoolPipeline()
    pipeline._latest_results = _last_results
    return jsonify({"report": pipeline.report()})


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dark Pool Flask Web Interface")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"🌑 Dark Pool Detection → http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
