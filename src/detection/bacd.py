"""BACD — Behavioral Autoregressive Conditional Duration Analysis.

Analyzes the temporal distribution of trades to extract behavioral
signatures that distinguish institutional from retail traders.

Core insight: the TIME BETWEEN TRADES is one of the strongest signals
for identifying dark pool activity. Institutions leave characteristic
temporal footprints that differ fundamentally from retail flow.

Features extracted:
- Burst detection: clusters of rapid trades separated by silence
- Inter-trade interval distribution: Weibull fit, autocorrelation
- Diurnal pattern: deviation from U-shape intraday volume curve
- Duration dispersion: regularity vs chaos

Reference: Engle & Russell (1998) "Autoregressive Conditional Duration"
"""

from dataclasses import dataclass
from typing import Optional
from collections import deque

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class BurstProfile:
    """A detected burst of rapid trading activity."""
    start_idx: int
    end_idx: int
    n_trades: int
    total_volume: float
    duration_ms: float
    avg_interval_ms: float
    max_interval_ms: float
    price_impact: float  # price change during burst


@dataclass
class BACDProfile:
    """Complete behavioral duration profile for a trading session."""
    # Burst metrics
    n_bursts: int
    burst_ratio: float          # % trades inside bursts
    burst_size_mean: float      # avg trades per burst
    burst_size_std: float
    burst_interval_mean: float  # avg time between bursts
    burst_interval_std: float

    # Interval distribution
    interval_mean: float
    interval_median: float
    interval_std: float
    interval_skewness: float
    interval_kurtosis: float
    weibull_shape: float        # shape param of Weibull fit
    weibull_scale: float        # scale param of Weibull fit
    weibull_r2: float           # goodness of fit

    # Autocorrelation
    duration_acf_lag1: float
    duration_acf_lag5: float
    duration_acf_lag10: float

    # Diurnal pattern
    morning_activity: float     # 9:30-12:00
    midday_activity: float      # 12:00-14:00
    afternoon_activity: float   # 14:00-16:00
    diurnal_deviation: float    # deviation from U-shape

    # Dispersion
    duration_cv: float          # coefficient of variation
    burstiness_index: float     # (σ - μ) / (σ + μ)

    # Meta
    n_trades: int
    session_duration_ms: float


class BACDAnalyzer:
    """Behavioral Autoregressive Conditional Duration Analyzer.

    Extracts temporal trading signatures from tick-level trade data.
    """

    BURST_THRESHOLD_MS = 100
    BURST_MIN_TRADES = 3
    MIN_TRADES_FOR_ANALYSIS = 10

    def __init__(self, burst_threshold_ms=100, burst_min_trades=3, min_trades=10):
        self.burst_threshold_ms = burst_threshold_ms
        self.burst_min_trades = burst_min_trades
        self.min_trades = min_trades

    def analyze_session(self, trades, timestamp_col="timestamp", volume_col="volume", price_col="price"):
        n = len(trades)
        if n < self.min_trades:
            return None
        df = trades.sort_values(timestamp_col).copy()
        times = df[timestamp_col].values.astype(np.float64)
        volumes = df[volume_col].values.astype(np.float64) if volume_col in df.columns else np.ones(n)
        prices = df[price_col].values.astype(np.float64) if price_col in df.columns else np.ones(n) * 100
        session_duration = float(times[-1] - times[0]) if n > 1 else 0.0

        intervals = np.diff(times)
        intervals = intervals[intervals > 0]
        if len(intervals) == 0:
            return None

        interval_mean = float(np.mean(intervals))
        interval_median = float(np.median(intervals))
        interval_std = float(np.std(intervals))
        interval_skew = float(scipy_stats.skew(intervals)) if len(intervals) > 2 else 0.0
        interval_kurt = float(scipy_stats.kurtosis(intervals)) if len(intervals) > 3 else 0.0
        duration_cv = interval_std / max(interval_mean, 1e-6)
        burstiness = (interval_std - interval_mean) / max(interval_std + interval_mean, 1e-6)

        bursts = self._detect_bursts(times, prices, volumes)
        burst_metrics = self._compute_burst_metrics(bursts, n)
        weibull_shape, weibull_scale, weibull_r2 = self._fit_weibull(intervals)
        acf = self._duration_autocorrelation(intervals, max_lag=10)
        diurnal = self._diurnal_pattern(times)

        return BACDProfile(
            n_bursts=len(bursts), burst_ratio=burst_metrics["burst_ratio"],
            burst_size_mean=burst_metrics["burst_size_mean"], burst_size_std=burst_metrics["burst_size_std"],
            burst_interval_mean=burst_metrics["burst_interval_mean"], burst_interval_std=burst_metrics["burst_interval_std"],
            interval_mean=interval_mean, interval_median=interval_median, interval_std=interval_std,
            interval_skewness=interval_skew, interval_kurtosis=interval_kurt,
            weibull_shape=weibull_shape, weibull_scale=weibull_scale, weibull_r2=weibull_r2,
            duration_acf_lag1=acf.get("lag1", 0.0), duration_acf_lag5=acf.get("lag5", 0.0), duration_acf_lag10=acf.get("lag10", 0.0),
            morning_activity=diurnal["morning"], midday_activity=diurnal["midday"], afternoon_activity=diurnal["afternoon"],
            diurnal_deviation=diurnal["deviation"], duration_cv=duration_cv, burstiness_index=burstiness,
            n_trades=n, session_duration_ms=session_duration,
        )

    FEATURE_NAMES = [
        "bacd_burst_ratio", "bacd_burst_size_mean", "bacd_burst_interval_mean",
        "bacd_interval_mean", "bacd_interval_skewness", "bacd_weibull_shape",
        "bacd_duration_acf_lag1", "bacd_diurnal_deviation", "bacd_duration_cv", "bacd_burstiness_index",
    ]

    def to_feature_vector(self, profile):
        if profile is None:
            return {f"bacd_{k}": 0.0 for k in self.FEATURE_NAMES}
        return {
            "bacd_burst_ratio": profile.burst_ratio,
            "bacd_burst_size_mean": np.log1p(profile.burst_size_mean),
            "bacd_burst_interval_mean": np.log1p(profile.burst_interval_mean),
            "bacd_interval_mean": np.log1p(profile.interval_mean),
            "bacd_interval_skewness": profile.interval_skewness,
            "bacd_weibull_shape": profile.weibull_shape,
            "bacd_duration_acf_lag1": profile.duration_acf_lag1,
            "bacd_diurnal_deviation": profile.diurnal_deviation,
            "bacd_duration_cv": profile.duration_cv,
            "bacd_burstiness_index": profile.burstiness_index,
        }

    def extract_features_df(self, trades, session_col="ticker", timestamp_col="timestamp", volume_col="volume", price_col="price"):
        rows = []
        for sid, session in trades.groupby(session_col):
            profile = self.analyze_session(session, timestamp_col, volume_col, price_col)
            feat = self.to_feature_vector(profile)
            feat["session_id"] = sid
            feat["n_trades"] = len(session)
            rows.append(feat)
        return pd.DataFrame(rows)

    def detect_duration_anomalies(self, trades, timestamp_col="timestamp", window_size=50, n_std=3.0):
        if len(trades) < window_size:
            return pd.DataFrame()
        df = trades.sort_values(timestamp_col).copy()
        intervals = np.diff(df[timestamp_col].values.astype(float))
        intervals = np.maximum(intervals, 1.0)
        log_intervals = np.log(intervals)
        series = pd.Series(log_intervals)
        rolling_mean = series.rolling(window_size, min_periods=window_size // 2).mean()
        rolling_std = series.rolling(window_size, min_periods=window_size // 2).std()
        zscore = (series - rolling_mean) / rolling_std.replace(0, 1.0)
        zscore = zscore.fillna(0)
        anomalies = pd.DataFrame({
            "trade_idx": np.arange(len(intervals)) + 1,
            "interval_ms": intervals,
            "zscore": zscore.values,
            "anomaly_type": np.where(zscore.values < -n_std, "burst", np.where(zscore.values > n_std, "gap", "normal")),
        })
        anomaly_mask = anomalies["anomaly_type"] != "normal"
        anomalies["anomaly_score"] = 0.0
        anomalies.loc[anomaly_mask, "anomaly_score"] = (anomalies.loc[anomaly_mask, "zscore"].abs() / (2 * n_std)).clip(0, 1)
        return anomalies[anomaly_mask].sort_values("anomaly_score", ascending=False)

    def _detect_bursts(self, times, prices, volumes):
        if len(times) < 2:
            return []
        intervals = np.diff(times)
        n = len(times)
        bursts = []
        in_burst = False
        burst_start = 0
        for i in range(len(intervals)):
            is_rapid = intervals[i] <= self.burst_threshold_ms
            if is_rapid and not in_burst:
                in_burst = True
                burst_start = i
            elif not is_rapid and in_burst:
                burst_end = i + 1
                if burst_end - burst_start >= self.burst_min_trades:
                    bursts.append(self._create_burst(burst_start, burst_end, times, prices, volumes))
                in_burst = False
        if in_burst:
            burst_end = n
            if burst_end - burst_start >= self.burst_min_trades:
                bursts.append(self._create_burst(burst_start, burst_end, times, prices, volumes))
        return bursts

    def _create_burst(self, start, end, times, prices, volumes):
        n = end - start
        btimes = times[start:end]
        bvols = volumes[start:end]
        bprices = prices[start:end]
        duration = float(btimes[-1] - btimes[0]) if n > 1 else 0.0
        iv = np.diff(btimes)
        iv = iv[iv > 0]
        price_impact = float(bprices[-1] - bprices[0]) / max(bprices[0], 1e-6)
        return BurstProfile(start_idx=int(start), end_idx=int(end), n_trades=n,
                           total_volume=float(np.sum(bvols)), duration_ms=duration,
                           avg_interval_ms=float(np.mean(iv)) if len(iv) > 0 else 0.0,
                           max_interval_ms=float(np.max(iv)) if len(iv) > 0 else 0.0,
                           price_impact=price_impact)

    def _compute_burst_metrics(self, bursts, n_total):
        if not bursts:
            return {"burst_ratio": 0.0, "burst_size_mean": 0.0, "burst_size_std": 0.0,
                    "burst_interval_mean": 0.0, "burst_interval_std": 0.0}
        total = sum(b.n_trades for b in bursts)
        sizes = np.array([b.n_trades for b in bursts], dtype=float)
        centers = np.array([(b.start_idx + b.end_idx) / 2 for b in bursts])
        bintervals = np.diff(centers) if len(centers) > 1 else np.array([0.0])
        return {"burst_ratio": total / max(n_total, 1),
                "burst_size_mean": float(np.mean(sizes)), "burst_size_std": float(np.std(sizes)),
                "burst_interval_mean": float(np.mean(bintervals)),
                "burst_interval_std": float(np.std(bintervals)) if len(bintervals) > 1 else 0.0}

    @staticmethod
    def _fit_weibull(intervals):
        if len(intervals) < 10:
            return 1.0, 0.0, 0.0
        try:
            clean = intervals[intervals > 0]
            if len(clean) < 10:
                return 1.0, 0.0, 0.0
            shape, loc, scale = scipy_stats.weibull_min.fit(clean, floc=0)
            sorted_i = np.sort(clean)
            emp = np.arange(1, len(sorted_i) + 1) / len(sorted_i)
            fit = scipy_stats.weibull_min.cdf(sorted_i, shape, loc=0, scale=scale)
            ss_res = np.sum((emp - fit) ** 2)
            ss_tot = np.sum((emp - np.mean(emp)) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
            return float(shape), float(scale), float(max(0, min(1, r2)))
        except Exception:
            return 1.0, 0.0, 0.0

    @staticmethod
    def _duration_autocorrelation(intervals, max_lag=10):
        if len(intervals) < max_lag + 5:
            return {"lag1": 0.0, "lag5": 0.0, "lag10": 0.0}
        clean = intervals[intervals > 0]
        if len(clean) < max_lag + 5:
            return {"lag1": 0.0, "lag5": 0.0, "lag10": 0.0}
        log_int = np.log(clean)
        log_int -= np.mean(log_int)
        def acf(data, lag):
            if len(data) <= lag or lag < 0:
                return 0.0
            x, y = data[:-lag] if lag > 0 else data, data[lag:]
            denom = np.dot(x, x) * np.dot(y, y)
            return float(np.dot(x, y) / np.sqrt(denom)) if denom > 0 else 0.0
        return {"lag1": acf(log_int, 1), "lag5": acf(log_int, 5), "lag10": acf(log_int, 10)}

    @staticmethod
    def _diurnal_pattern(times):
        hours = (times % 86_400_000) / 3_600_000
        m = float(np.mean((hours >= 9.5) & (hours < 12)))
        d = float(np.mean((hours >= 12) & (hours < 14)))
        a = float(np.mean((hours >= 14) & (hours <= 16)))
        expected = {"morning": 0.40, "midday": 0.20, "afternoon": 0.40}
        total = m + d + a
        if total > 0:
            dev = float(np.sqrt(((m/total-expected["morning"])/expected["morning"])**2 +
                               ((d/total-expected["midday"])/expected["midday"])**2 +
                               ((a/total-expected["afternoon"])/expected["afternoon"])**2) / 3)
        else:
            dev = 0.0
        return {"morning": m, "midday": d, "afternoon": a, "deviation": dev}
