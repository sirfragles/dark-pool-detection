"""Dark Pool Detection — Backtesting Engine."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    def __init__(self, initial_capital=1_000_000.0, position_size_pct=0.05, transaction_cost_bps=1.0):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.transaction_cost = transaction_cost_bps / 10000
        self._results: dict[str, dict] = {}

    def backtest_vpin_fade(self, price_data, vpin_signal, threshold=0.8, holding_period_bars=10):
        if len(price_data) < holding_period_bars:
            return {"error": "Insufficient data"}
        df = price_data.copy()
        df["vpin"] = vpin_signal.values if len(vpin_signal) == len(df) else 0
        df["signal"] = 0
        df.loc[df["vpin"] > threshold, "signal"] = -1
        df.loc[df["vpin"] < 0.2, "signal"] = 1
        price_col = "close" if "close" in df.columns else "price"
        df["fwd_return"] = df[price_col].shift(-holding_period_bars) / df[price_col] - 1
        df["strategy_return"] = -df["signal"] * df["fwd_return"] - self.transaction_cost * df["signal"].diff().abs()
        valid = df.dropna(subset=["fwd_return", "strategy_return"])
        return self._compute_metrics(valid, "vpin_fade")

    def backtest_iceberg_frontrun(self, price_data, iceberg_signal, holding_period_bars=5):
        if len(price_data) < holding_period_bars:
            return {"error": "Insufficient data"}
        df = price_data.copy()
        df["iceberg"] = iceberg_signal.values if len(iceberg_signal) == len(df) else 0
        df["signal"] = df["iceberg"].astype(int)
        price_col = "close" if "close" in df.columns else "price"
        df["fwd_return"] = df[price_col].shift(-holding_period_bars) / df[price_col] - 1
        df["strategy_return"] = df["signal"] * df["fwd_return"] - self.transaction_cost * df["signal"].diff().abs()
        return self._compute_metrics(df.dropna(subset=["fwd_return", "strategy_return"]), "iceberg_frontrun")

    def backtest_dark_volume_fade(self, price_data, dark_volume_zscore, threshold=3.0, holding_period_bars=10):
        if len(price_data) < holding_period_bars:
            return {"error": "Insufficient data"}
        df = price_data.copy()
        df["dark_z"] = dark_volume_zscore.values if len(dark_volume_zscore) == len(df) else 0
        df["signal"] = 0
        df.loc[df["dark_z"] > threshold, "signal"] = -1
        df.loc[df["dark_z"] < -threshold, "signal"] = 1
        price_col = "close" if "close" in df.columns else "price"
        df["fwd_return"] = df[price_col].shift(-holding_period_bars) / df[price_col] - 1
        df["strategy_return"] = -df["signal"] * df["fwd_return"] - self.transaction_cost * df["signal"].diff().abs()
        return self._compute_metrics(df.dropna(subset=["fwd_return", "strategy_return"]), "dark_volume_fade")

    def backtest_ensemble(self, price_data, signals, weights=None, holding_period_bars=10):
        if len(price_data) < holding_period_bars:
            return {"error": "Insufficient data"}
        df = price_data.copy()
        n = len(signals)
        combined = pd.Series(0.0, index=df.index)
        for name, sig in signals.items():
            w = weights.get(name, 1.0 / n) if weights else 1.0 / n
            combined += w * sig.reindex(df.index).fillna(0)
        df["signal"] = combined
        price_col = "close" if "close" in df.columns else "price"
        df["fwd_return"] = df[price_col].shift(-holding_period_bars) / df[price_col] - 1
        df["strategy_return"] = df["signal"] * df["fwd_return"] - self.transaction_cost * df["signal"].diff().abs()
        return self._compute_metrics(df.dropna(subset=["fwd_return", "strategy_return"]), "ensemble")

    def _compute_metrics(self, df, strategy_name):
        returns = df["strategy_return"].values
        benchmark = df["fwd_return"].values
        cumret = np.cumprod(1 + returns)
        total_return = float(cumret[-1] - 1) if len(cumret) > 0 else 0.0
        sharpe = self._sharpe_ratio(returns)
        max_dd = self._max_drawdown(cumret)
        hit_rate = float(np.mean(returns > 0))
        avg_return = float(np.mean(returns))
        volatility = float(np.std(returns))
        signal_returns = df.groupby("signal")["fwd_return"].mean().to_dict()
        metrics = {"strategy": strategy_name, "total_return": total_return,
                   "sharpe_ratio": sharpe, "max_drawdown": max_dd,
                   "hit_rate": hit_rate, "avg_return_per_trade": avg_return,
                   "volatility": volatility, "n_periods": len(df),
                   "signal_returns": signal_returns,
                   "benchmark_return": float(np.prod(1 + benchmark) - 1) if len(benchmark) > 0 else 0.0}
        self._results[strategy_name] = metrics
        return metrics

    def walk_forward_test(self, df, strategy_fn, n_splits=5, **kwargs):
        n = len(df)
        fold_size = n // (n_splits + 1)
        fold_results = []
        for fold in range(n_splits):
            train_end = (fold + 1) * fold_size
            test_end = min((fold + 2) * fold_size, n)
            test_df = df.iloc[train_end:test_end]
            if len(test_df) < 10:
                continue
            result = strategy_fn(test_df, **kwargs)
            result["fold"] = fold
            fold_results.append(result)
        if not fold_results:
            return {"error": "No valid folds"}
        sharpe_values = [r.get("sharpe_ratio", 0) for r in fold_results]
        hit_values = [r.get("hit_rate", 0) for r in fold_results]
        return {"n_folds": len(fold_results), "avg_sharpe": float(np.mean(sharpe_values)),
                "avg_hit_rate": float(np.mean(hit_values)), "folds": fold_results}

    @staticmethod
    def _sharpe_ratio(returns, risk_free=0.0):
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free
        mean, std = np.mean(excess), np.std(excess)
        return float(mean / std * np.sqrt(252)) if std != 0 else 0.0

    @staticmethod
    def _max_drawdown(cumret):
        peak = np.maximum.accumulate(cumret)
        drawdown = (cumret - peak) / peak
        return float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    @property
    def results(self):
        return self._results

    def summary(self):
        if not self._results:
            return "No backtest results."
        lines = ["BACKTEST RESULTS"]
        for name, m in self._results.items():
            lines.append(f"{name}: Sharpe={m.get('sharpe_ratio', 0):.3f} Return={m.get('total_return', 0):.2%} Hit={m.get('hit_rate', 0):.1%}")
        return "\n".join(lines)

    def to_dataframe(self):
        rows = []
        for name, m in self._results.items():
            if isinstance(m, dict):
                rows.append({"strategy": name, "sharpe": m.get("sharpe_ratio", 0),
                           "total_return": m.get("total_return", 0),
                           "hit_rate": m.get("hit_rate", 0)})
        return pd.DataFrame(rows).sort_values("sharpe", ascending=False)
