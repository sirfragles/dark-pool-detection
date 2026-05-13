"""Trader Type Classification — Faza 5.

Classifies market participants by trading signature using temporal
clustering and trade pattern analysis. Identifies:
- Institution: Large blocks, dark pool preference, VWAP execution
- HFT: Ultra-short holding, co-located sub-ms patterns
- Retail: Small odd-lot trades, market orders, round numbers
- Market Maker: Two-sided flow, spread capture, inventory management

Faza 9 (BACD): Extended with behavioral duration features.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class TraderProfile:
    trader_type: str
    confidence: float
    avg_trade_size: float
    trade_interval_ms: float
    dark_share: float
    time_of_day_preference: str
    volume_share: float
    n_trades: int
    signature_features: dict


class TraderTypeClassifier:
    """Classify market participants by behavioral signature.

    Uses unsupervised learning (GMM) to discover natural clusters
    in trading behavior, then maps clusters to known types.
    """

    FEATURES = [
        "trade_size_mean",
        "trade_size_std",
        "trade_interval_mean",
        "trade_interval_std",
        "dark_share",
        "lit_volume_share",
        "morning_preference",
        "midday_preference",
        "afternoon_preference",
        "size_volatility_ratio",
        # BACD features (Faza 9)
        "bacd_burst_ratio",
        "bacd_burst_size_mean",
        "bacd_interval_mean",
        "bacd_interval_skewness",
        "bacd_weibull_shape",
        "bacd_duration_acf_lag1",
        "bacd_diurnal_deviation",
        "bacd_duration_cv",
        "bacd_burstiness_index",
    ]

    PROTOTYPES = {
        "institution": {
            "trade_size_mean": 2.0,
            "trade_interval_mean": 1.0,
            "dark_share": 2.0,
            "morning_preference": 0.0,
            "midday_preference": 1.0,
            "bacd_burst_ratio": 1.0,
            "bacd_interval_skewness": 0.5,
            "bacd_duration_acf_lag1": 0.5,
            "bacd_burstiness_index": -0.3,
        },
        "hft": {
            "trade_size_mean": -1.0,
            "trade_interval_mean": -2.0,
            "dark_share": -1.0,
            "morning_preference": 0.5,
            "midday_preference": 0.0,
            "bacd_burst_ratio": 2.0,
            "bacd_interval_skewness": 1.5,
            "bacd_duration_acf_lag1": 0.8,
            "bacd_burstiness_index": -0.8,
        },
        "retail": {
            "trade_size_mean": -1.5,
            "trade_interval_mean": 0.5,
            "dark_share": -1.5,
            "morning_preference": 0.5,
            "midday_preference": -0.5,
            "bacd_burst_ratio": -1.0,
            "bacd_interval_skewness": -0.3,
            "bacd_duration_acf_lag1": 0.0,
            "bacd_burstiness_index": 0.3,
        },
        "market_maker": {
            "trade_size_mean": 0.5,
            "trade_interval_mean": -1.0,
            "dark_share": 0.0,
            "morning_preference": 1.0,
            "midday_preference": 0.5,
            "bacd_burst_ratio": 0.5,
            "bacd_interval_skewness": 0.2,
            "bacd_duration_acf_lag1": -0.2,
            "bacd_burstiness_index": 0.0,
        },
    }

    def __init__(self, n_clusters: int = 4, method: str = "gmm", min_samples_per_type: int = 50, random_state: int = 42):
        self.n_clusters = n_clusters
        self.method = method
        self.min_samples = min_samples_per_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        if method == "gmm":
            self.model = GaussianMixture(n_components=n_clusters, random_state=random_state, covariance_type="full")
        else:
            self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self._profiles: list[TraderProfile] = []
        self._cluster_labels: dict[int, str] = {}
        self._is_fitted = False

    def extract_features(
        self,
        trades: pd.DataFrame,
        ticker_col: str = "ticker",
        volume_col: str = "volume",
        timestamp_col: str = "timestamp",
        dark_col: str = "is_dark",
        use_bacd: bool = True,
    ) -> pd.DataFrame:
        """Extract behavioral features from tick-level trade data.

        Groups trades into trader-specific sessions using:
        - Temporal proximity (gap < threshold → same trader)
        - Trade size consistency
        - Venue pattern matching
        - BACD duration analysis (Faza 9)
        """
        if trades.empty:
            return pd.DataFrame(columns=self.FEATURES)

        if use_bacd:
            try:
                from src.detection.bacd import BACDAnalyzer
                bacd = BACDAnalyzer(min_trades=5)
            except ImportError:
                bacd = None
                use_bacd = False
        else:
            bacd = None

        features_list = []
        for ticker, group in trades.groupby(ticker_col):
            group = group.sort_values(timestamp_col)
            time_diffs = group[timestamp_col].diff().fillna(0)
            session_breaks = time_diffs > 30_000
            group["session_id"] = session_breaks.cumsum()
            for sid, session in group.groupby("session_id"):
                feat = self._session_features(session, volume_col, timestamp_col, dark_col)
                if feat is not None:
                    if bacd is not None:
                        profile = bacd.analyze_session(session, timestamp_col, volume_col)
                        bacd_feats = bacd.to_feature_vector(profile)
                        feat.update(bacd_feats)
                    feat["ticker"] = ticker
                    feat["session_id"] = f"{ticker}_{sid}"
                    features_list.append(feat)

        if not features_list:
            return pd.DataFrame(columns=self.FEATURES + ["ticker", "session_id"])
        return pd.DataFrame(features_list)

    def _session_features(self, session, volume_col, timestamp_col, dark_col):
        n = len(session)
        if n < 3:
            return None
        vols = session[volume_col].values
        times = session[timestamp_col].values
        size_mean = np.log1p(float(np.mean(vols)))
        size_std = float(np.std(vols)) / max(float(np.mean(vols)), 1)
        intervals = np.diff(times)
        intervals_ms = intervals[intervals > 0]
        interval_mean = np.log1p(float(np.mean(intervals_ms))) if len(intervals_ms) > 0 else 0.0
        interval_std = float(np.std(intervals_ms)) / max(float(np.mean(intervals_ms)), 1) if len(intervals_ms) > 1 else 0.0
        dark_share = float(session[dark_col].mean()) if dark_col in session.columns else 0.0
        hours = (times % 86_400_000) // 3_600_000
        morning = float(np.mean((hours >= 9) & (hours < 12)))
        midday = float(np.mean((hours >= 12) & (hours < 14)))
        afternoon = float(np.mean((hours >= 14) & (hours <= 16)))
        size_vol_ratio = size_std / max(interval_std, 1e-6)
        return {
            "trade_size_mean": size_mean, "trade_size_std": size_std,
            "trade_interval_mean": interval_mean, "trade_interval_std": interval_std,
            "dark_share": dark_share, "lit_volume_share": 1.0 - dark_share,
            "morning_preference": morning, "midday_preference": midday,
            "afternoon_preference": afternoon, "size_volatility_ratio": size_vol_ratio,
            "n_trades": n,
        }

    def fit(self, features: pd.DataFrame):
        if len(features) < self.n_clusters:
            return self
        available = [f for f in self.FEATURES if f in features.columns]
        if not available:
            return self
        X = features[available].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        labels = self.model.predict(X_scaled)
        features["cluster"] = labels
        self._map_clusters_to_types(X_scaled, labels, features)
        self._is_fitted = True
        return self

    def predict(self, features: pd.DataFrame, min_confidence: float = 0.3) -> list[TraderProfile]:
        if not self._is_fitted:
            self.fit(features)
            if not self._is_fitted:
                return []
        available = [f for f in self.FEATURES if f in features.columns]
        if not available:
            return []
        X = features[available].fillna(0).values
        X_scaled = self.scaler.transform(X)
        if self.method == "gmm":
            probs = self.model.predict_proba(X_scaled)
            labels = probs.argmax(axis=1)
            confidences = probs.max(axis=1)
        else:
            labels = self.model.predict(X_scaled)
            distances = self.model.transform(X_scaled)
            confidences = 1.0 / (1.0 + distances.min(axis=1))
        profiles = []
        for i, (_, row) in enumerate(features.iterrows()):
            if confidences[i] < min_confidence:
                continue
            cluster = labels[i]
            trader_type = self._cluster_labels.get(cluster, "unknown")
            tod_pref = "all_day"
            if row.get("morning_preference", 0) > 0.5:
                tod_pref = "open"
            elif row.get("afternoon_preference", 0) > 0.5:
                tod_pref = "close"
            elif row.get("midday_preference", 0) > 0.5:
                tod_pref = "midday"
            profiles.append(TraderProfile(
                trader_type=trader_type, confidence=float(confidences[i]),
                avg_trade_size=float(row.get("trade_size_mean", 0)),
                trade_interval_ms=float(row.get("trade_interval_mean", 0)),
                dark_share=float(row.get("dark_share", 0)),
                time_of_day_preference=tod_pref, volume_share=0.0,
                n_trades=int(row.get("n_trades", 0)),
                signature_features={f: float(row.get(f, 0)) for f in self.FEATURES},
            ))
        total_vol = sum(p.n_trades for p in profiles)
        for p in profiles:
            p.volume_share = p.n_trades / max(total_vol, 1)
        self._profiles = profiles
        return profiles

    def _map_clusters_to_types(self, X_scaled, labels, features_df):
        available = [f for f in self.FEATURES if f in features_df.columns]
        feature_indices = {f: i for i, f in enumerate(available)}
        centroids = {}
        for cid in range(self.n_clusters):
            mask = labels == cid
            if mask.any():
                centroids[cid] = X_scaled[mask].mean(axis=0)
        for cid, centroid in centroids.items():
            best_type = "unknown"
            best_sim = -1.0
            for type_name, proto in self.PROTOTYPES.items():
                proto_vec = np.zeros(len(available))
                for feat, val in proto.items():
                    if feat in feature_indices:
                        proto_vec[feature_indices[feat]] = val
                norm_c = np.linalg.norm(centroid)
                norm_p = np.linalg.norm(proto_vec)
                sim = float(np.dot(centroid, proto_vec) / (norm_c * norm_p)) if norm_c > 0 and norm_p > 0 else 0.0
                if sim > best_sim:
                    best_sim = sim
                    best_type = type_name
            self._cluster_labels[cid] = best_type

    def summary(self) -> dict:
        if not self._profiles:
            return {"n_profiles": 0}
        df = pd.DataFrame([{"type": p.trader_type, "confidence": p.confidence,
                           "n_trades": p.n_trades, "dark_share": p.dark_share} for p in self._profiles])
        type_summary = df.groupby("type").agg(n_sessions=("type", "count"), avg_confidence=("confidence", "mean"),
                                              total_trades=("n_trades", "sum"), avg_dark_share=("dark_share", "mean")).to_dict(orient="index")
        return {"n_profiles": len(self._profiles), "types": list(type_summary.keys()),
                "type_details": type_summary, "cluster_mapping": self._cluster_labels}

    def type_distribution(self) -> dict[str, dict]:
        if not self._profiles:
            return {}
        dist = {}
        for p in self._profiles:
            if p.trader_type not in dist:
                dist[p.trader_type] = {"n_sessions": 0, "total_trades": 0}
            dist[p.trader_type]["n_sessions"] += 1
            dist[p.trader_type]["total_trades"] += p.n_trades
        return dist

    def reset(self) -> None:
        self._profiles.clear()
        self._cluster_labels.clear()
        self._is_fitted = False
