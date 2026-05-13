"""Dark Pool Detection — Integration Pipeline.

Orchestrates all detection modules and produces unified output.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from .data.simulator import OrderBookSimulator
from .detection.iceberg import IcebergDetector
from .detection.vpin import VPINCalculator
from .utils import ensure_dirs, get_logger, load_config

logger = get_logger(__name__)


class DarkPoolPipeline:
    """End-to-end dark pool detection pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        ensure_dirs(self.config)

        # Initialize detectors
        vpin_cfg = self.config["detection"]["vpin"]
        self.vpin = VPINCalculator(
            volume_bucket_size=vpin_cfg["volume_bucket_size"],
            n_buckets=vpin_cfg.get("n_buckets", 50),
        )

        iceberg_cfg = self.config["detection"]["iceberg"]
        self.iceberg = IcebergDetector(
            min_detection_confidence=iceberg_cfg["min_confidence"],
            synthetic_window_ms=iceberg_cfg.get("synthetic_window_ms", 50),
        )

        # State
        self._latest_results: dict = {}
        self._data: Optional[pd.DataFrame] = None

    def run_simulation(
        self,
        n_ticks: int = 23400,
        n_tickers: int = 5,
        seed: int = 42,
    ) -> dict:
        """Run full pipeline on simulated data.

        Returns comprehensive results dictionary.
        """
        logger.info(
            f"Starting dark pool detection simulation: "
            f"{n_tickers} tickers, {n_ticks} ticks"
        )

        # Step 1: Generate synthetic data
        sim = OrderBookSimulator(
            n_tickers=n_tickers, n_ticks=n_ticks, seed=seed
        )
        df = sim.generate()
        sim_summary = sim.generate_summary(df)
        self._data = df

        logger.info(
            f"Generated {len(df)} ticks, "
            f"{sim_summary['total_trades']} trades "
            f"({sim_summary['dark_trades']} dark, "
            f"{sim_summary['iceberg_trades']} iceberg)"
        )

        # Step 2: Compute VPIN — fix classification
        trades = df[df["volume"] > 0].copy()
        # Add proper buy/sell classification for VPIN
        trades["buy_volume"] = trades.apply(
            lambda r: r["volume"] if r.get("trade_side") == "buy" else 0, axis=1
        )
        trades["sell_volume"] = trades.apply(
            lambda r: r["volume"] if r.get("trade_side") == "sell" else 0, axis=1
        )
        vpin_df = self.vpin.compute(trades, price_col="price", volume_col="volume")
        vpin_summary = {
            "current_vpin": float(vpin_df["vpin"].iloc[-1]) if len(vpin_df) > 0 else 0.0,
            "mean": float(vpin_df["vpin"].mean()) if len(vpin_df) > 0 else 0.0,
            "max": float(vpin_df["vpin"].max()) if len(vpin_df) > 0 else 0.0,
            "std": float(vpin_df["vpin"].std()) if len(vpin_df) > 0 else 0.0,
            "toxic_pct": float((vpin_df["vpin"] > 0.8).mean() * 100) if len(vpin_df) > 0 else 0.0,
            "n_buckets": len(vpin_df),
        }

        # Step 3: Detect icebergs — with order book sim
        iceberg_results = []
        for _, row in trades.iterrows():
            side = row.get("trade_side", "buy")
            size = int(row["volume"])
            price = row["price"]
            
            # Update order book before processing trade
            spread = price * 0.001
            if side == "buy":
                self.iceberg.update_order_book(
                    row["timestamp"],
                    bids=[(price - spread, row.get("bid_size", 100))],
                    asks=[(price, row.get("ask_size", 100))],
                )
            else:
                self.iceberg.update_order_book(
                    row["timestamp"],
                    bids=[(price, row.get("bid_size", 100))],
                    asks=[(price + spread, row.get("ask_size", 100))],
                )
            
            detected = self.iceberg.process_trade(
                timestamp=row["timestamp"],
                price=price,
                size=size,
                side=side,
            )
            for ib in detected:
                    iceberg_results.append(
                        {
                            "timestamp": ib.timestamp,
                            "iceberg_id": ib.id,
                            "price": ib.price,
                            "displayed": ib.displayed_size,
                            "estimated_total": ib.estimated_total,
                            "hidden": ib.hidden_size,
                            "side": ib.side,
                            "type": ib.iceberg_type,
                            "confidence": ib.confidence,
                        }
                    )

        iceberg_df = pd.DataFrame(iceberg_results)
        iceberg_summary = self.iceberg.summary()

        # Step 4: Analyze dark trades
        dark_trades = trades[trades["is_dark"]]
        dark_analysis = self._analyze_dark_trades(dark_trades, trades)

        # Step 5: Trader type analysis
        trader_analysis = self._analyze_trader_types(trades)

        # Step 6: VPIN vs Dark Activity correlation
        vpin_dark_corr = self._correlate_vpin_with_dark(vpin_df, trades)

        # Compile results
        results = {
            "simulation": sim_summary,
            "vpin": vpin_summary,
            "vpin_history": vpin_df.to_dict("records") if len(vpin_df) < 1000 else [],
            "iceberg": iceberg_summary,
            "iceberg_detected": len(iceberg_df),
            "dark_analysis": dark_analysis,
            "trader_analysis": trader_analysis,
            "vpin_dark_correlation": vpin_dark_corr,
            "detection_score": self._compute_detection_score(
                sim_summary, iceberg_summary, vpin_summary, dark_analysis
            ),
        }

        self._latest_results = results

        # Save to Parquet for later analysis
        output_dir = Path(self.config["system"]["output_dir"])
        trades.to_parquet(output_dir / "simulated_trades.parquet")
        if not iceberg_df.empty:
            iceberg_df.to_parquet(output_dir / "detected_icebergs.parquet")
        if len(vpin_df) > 0:
            vpin_df.to_parquet(output_dir / "vpin_series.parquet")

        logger.info("Pipeline complete.")
        return results

    def _analyze_dark_trades(
        self, dark_trades: pd.DataFrame, all_trades: pd.DataFrame
    ) -> dict:
        """Analyze dark pool trade patterns."""
        if dark_trades.empty:
            return {"n_dark": 0, "dark_volume_pct": 0.0}

        dark_vol = dark_trades["volume"].sum()
        total_vol = all_trades["volume"].sum()

        by_ticker = (
            dark_trades.groupby("ticker")["volume"]
            .sum()
            .to_dict()
        )

        # Dark volume around iceberg activity
        iceberg_mask = all_trades["is_iceberg"]
        if iceberg_mask.any():
            dark_near_iceberg = dark_trades[
                dark_trades["timestamp"].isin(
                    all_trades[iceberg_mask]["timestamp"]
                )
            ]
            dark_with_iceberg_pct = (
                len(dark_near_iceberg) / max(len(dark_trades), 1) * 100
            )
        else:
            dark_with_iceberg_pct = 0.0

        return {
            "n_dark": int(len(dark_trades)),
            "dark_volume": int(dark_vol),
            "total_volume": int(total_vol),
            "dark_volume_pct": float(dark_vol / max(total_vol, 1)) * 100,
            "dark_by_ticker": by_ticker,
            "avg_dark_trade_size": float(dark_trades["volume"].mean()),
            "dark_with_iceberg_pct": dark_with_iceberg_pct,
            "time_of_day_distribution": (
                dark_trades.groupby(dark_trades["timestamp"] // 3600000 * 3600000)["volume"]
                .sum()
                .to_dict()
            ),
        }

    def _analyze_trader_types(self, trades: pd.DataFrame) -> dict:
        """Analyze trader type patterns."""
        if "trader_type" not in trades.columns:
            return {}

        type_stats = trades.groupby("trader_type").agg(
            n_trades=("volume", "count"),
            total_volume=("volume", "sum"),
            avg_size=("volume", "mean"),
            dark_pct=("is_dark", "mean"),
        )

        return type_stats.to_dict()

    def _correlate_vpin_with_dark(
        self, vpin_df: pd.DataFrame, trades: pd.DataFrame
    ) -> float:
        """Measure VPIN vs dark trade activity correlation."""
        if vpin_df.empty or "is_dark" not in trades.columns:
            return 0.0

        # Aggregate dark trades to VPIN time windows
        dark_window = []
        for _, vp_row in vpin_df.iterrows():
            if "time" not in vp_row:
                continue
            t = vp_row["time"]
            dark_count = int(
                (
                    (trades["is_dark"])
                    & (abs(trades["timestamp"] - t) < 5000)
                ).sum()
            )
            dark_window.append(dark_count)

        if len(dark_window) > 1 and len(vpin_df["vpin"].values) > 1:
            corr = pd.Series(dark_window).corr(pd.Series(vpin_df["vpin"].values))
            return float(corr) if not pd.isna(corr) else 0.0
        return 0.0

    def _compute_detection_score(
        self,
        sim: dict,
        iceberg: dict,
        vpin: dict,
        dark: dict,
    ) -> dict:
        """Compute overall detection quality score (0-100)."""
        scores = {}

        # Iceberg detection rate (vs simulated)
        if sim.get("iceberg_trades", 0) > 0:
            scores["iceberg_detection"] = min(
                100,
                iceberg.get("n_active", 0) / max(sim["iceberg_trades"], 1) * 100,
            )
        else:
            scores["iceberg_detection"] = 80  # No icebergs to detect = good

        # VPIN usefulness (should be > 0 when informed traders active)
        if vpin.get("toxic_pct", 0) > 0:
            scores["vpin_signal"] = min(100, vpin.get("toxic_pct", 0) * 100)
        else:
            scores["vpin_signal"] = 50

        # Dark volume detection
        if dark.get("dark_volume_pct", 0) > 0:
            scores["dark_detection"] = min(100, dark["dark_volume_pct"] * 3)
        else:
            scores["dark_detection"] = 60

        scores["overall"] = float(np.mean(list(scores.values())))

        return scores

    def report(self) -> str:
        """Generate human-readable report."""
        if not self._latest_results:
            return "No results available. Run pipeline first."

        r = self._latest_results
        sim = r["simulation"]
        ice = r["iceberg"]
        vp = r["vpin"]
        dark = r["dark_analysis"]
        score = r["detection_score"]

        return f"""╔══════════════════════════════════════════════════════════╗
║         DARK POOL DETECTION — PIPELINE REPORT            ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  📊 SIMULATION                                           ║
║     Trades total:     {sim['total_trades']:>8,d}                       ║
║     Dark pool:        {sim['dark_trades']:>8,d} ({sim['dark_pct']:.1f}%)                ║
║     Iceberg:          {sim['iceberg_trades']:>8,d} ({sim['iceberg_pct']:.1f}%)                ║
║     Informed:         {sim['informed_trades']:>8,d} ({sim['informed_pct']:.1f}%)                ║
║                                                          ║
║  🧊 ICEBERG DETECTION                                    ║
║     Active icebergs:  {ice.get('n_active', 0):>8,d}                       ║
║     Native:           {ice.get('n_native', 0):>8,d}                       ║
║     Synthetic:        {ice.get('n_synthetic', 0):>8,d}                       ║
║     Est. hidden vol:  {ice.get('total_estimated_hidden', 0):>8,d} shares               ║
║     Avg confidence:   {ice.get('avg_confidence', 0):>8.2%}                       ║
║                                                          ║
║  ⚡ VPIN (Order Flow Toxicity)                            ║
║     Current:          {vp.get('current_vpin', 0):>8.3f}                       ║
║     Mean:             {vp.get('mean', 0):>8.3f}                       ║
║     Max:              {vp.get('max', 0):>8.3f}                       ║
║     Toxic %:          {vp.get('toxic_pct', 0):>8.1f}%                       ║
║                                                          ║
║  🌑 DARK VOLUME ANALYSIS                                  ║
║     Dark trades:      {dark.get('n_dark', 0):>8,d}                       ║
║     Dark volume %:    {dark.get('dark_volume_pct', 0):>8.1f}%                       ║
║     Avg dark size:    {dark.get('avg_dark_trade_size', 0):>8.0f}                       ║
║                                                          ║
║  📈 DETECTION QUALITY                                     ║
║     Iceberg:          {score.get('iceberg_detection', 0):>8.1f}/100                    ║
║     VPIN:             {score.get('vpin_signal', 0):>8.1f}/100                    ║
║     Dark:             {score.get('dark_detection', 0):>8.1f}/100                    ║
║     ═══════════════                                    ║
║     OVERALL:          {score.get('overall', 0):>8.1f}/100                    ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""


# Fix missing import
import numpy as np
