"""Trader Fingerprint — Unsupervised Clustering.

Identifies unique trading signatures through unsupervised clustering.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import get_logger

logger = get_logger(__name__)


class TraderFingerprint:
    FEATURES = ["trade_size_mean", "trade_size_std", "trade_interval_mean",
                "trade_interval_std", "dark_share", "lit_volume_share",
                "morning_preference", "midday_preference", "afternoon_preference",
                "size_volatility_ratio", "burst_ratio", "aggressiveness",
                "iceberg_preference", "spread_cross_pct"]

    def __init__(self, model_dir="data/models", n_clusters=5, method="kmeans", random_state=42):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state
        self.scaler = StandardScaler()
        if method == "dbscan":
            self.model = DBSCAN(eps=0.5, min_samples=5)
        else:
            self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self._fingerprints: dict[int, dict] = {}
        self._tracked_entities: list[dict] = []
        self._is_fitted = False
        self._cluster_centers: Optional[np.ndarray] = None
        self._pca: Optional[PCA] = None

    def extract_features(self, trades, ticker_col="ticker", volume_col="volume",
                         timestamp_col="timestamp", dark_col="is_dark", iceberg_col="is_iceberg"):
        if trades.empty:
            return pd.DataFrame(columns=self.FEATURES)
        features_list = []
        for ticker, group in trades.groupby(ticker_col):
            group = group.sort_values(timestamp_col)
            time_diffs = group[timestamp_col].diff().fillna(0)
            group["entity_id"] = (time_diffs > 60_000).cumsum()
            for eid, session in group.groupby("entity_id"):
                feat = self._entity_features(session, volume_col, timestamp_col, dark_col, iceberg_col)
                if feat is not None:
                    feat["ticker"] = ticker
                    feat["entity_id"] = f"{ticker}_{eid}"
                    feat["n_trades"] = len(session)
                    features_list.append(feat)
        if not features_list:
            return pd.DataFrame(columns=self.FEATURES + ["ticker", "entity_id"])
        return pd.DataFrame(features_list)

    def _entity_features(self, session, volume_col, timestamp_col, dark_col, iceberg_col):
        n = len(session)
        if n < 5:
            return None
        vols = session[volume_col].values.astype(float)
        times = session[timestamp_col].values.astype(float)
        size_mean = float(np.mean(vols))
        size_std = float(np.std(vols))
        intervals = np.diff(times)
        intervals = intervals[intervals > 0]
        interval_mean = float(np.mean(intervals)) if len(intervals) > 0 else 0.0
        interval_std = float(np.std(intervals)) if len(intervals) > 1 else 0.0
        dark_share = float(session[dark_col].mean()) if dark_col in session.columns else 0.0
        iceberg_pref = float(session[iceberg_col].mean()) if iceberg_col in session.columns else 0.0
        hours = (times % 86_400_000) // 3_600_000
        morning = float(np.mean((hours >= 9) & (hours < 12)))
        midday = float(np.mean((hours >= 12) & (hours < 14)))
        afternoon = float(np.mean((hours >= 14) & (hours <= 16)))
        burst_ratio = 0.0
        if len(intervals) > 0:
            med = float(np.median(intervals))
            burst_ratio = float(np.mean(intervals < med * 0.3)) if med > 0 else 0.0
        aggressiveness = 0.0
        spread_cross = 0.5
        size_vol_ratio = size_std / max(size_mean, 1e-6)
        return {
            "trade_size_mean": np.log1p(size_mean), "trade_size_std": np.log1p(size_std),
            "trade_interval_mean": np.log1p(interval_mean), "trade_interval_std": np.log1p(interval_std),
            "dark_share": dark_share, "lit_volume_share": 1.0 - dark_share,
            "morning_preference": morning, "midday_preference": midday,
            "afternoon_preference": afternoon, "size_volatility_ratio": size_vol_ratio,
            "burst_ratio": burst_ratio, "aggressiveness": aggressiveness,
            "iceberg_preference": iceberg_pref, "spread_cross_pct": spread_cross,
        }

    def fit(self, features, use_pca=True):
        if len(features) < self.n_clusters:
            return self
        available = [f for f in self.FEATURES if f in features.columns]
        if not available:
            return self
        X = features[available].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        if use_pca and X_scaled.shape[1] > 3:
            n_components = min(8, X_scaled.shape[1], X_scaled.shape[0])
            self._pca = PCA(n_components=n_components, random_state=self.random_state)
            X_cluster = self._pca.fit_transform(X_scaled)
        else:
            X_cluster = X_scaled
        labels = self.model.fit_predict(X_cluster)
        features["cluster"] = labels
        self._cluster_centers = np.array([X_scaled[labels == c].mean(axis=0) for c in sorted(set(labels)) if c >= 0])
        self._build_profiles(features, labels, available)
        self._is_fitted = True
        n_unique = len(set(l for l in labels if l >= 0))
        valid = labels >= 0
        if n_unique >= 2 and n_unique < valid.sum():
            try:
                score = silhouette_score(X_cluster[valid], labels[valid])
                logger.info(f"Silhouette score: {score:.3f}")
            except ValueError:
                pass
        return self

    def predict(self, features):
        if not self._is_fitted:
            return np.full(len(features), -1)
        available = [f for f in self.FEATURES if f in features.columns]
        X = features[available].fillna(0).values
        X_scaled = self.scaler.transform(X)
        if self._pca is not None:
            X_scaled = self._pca.transform(X_scaled)
        return self.model.fit_predict(X_scaled) if self.method == "dbscan" else self.model.predict(X_scaled)

    def track_entity(self, entity_id, features, timestamp):
        if not self._is_fitted:
            return {"entity_id": entity_id, "fingerprint": -1, "confidence": 0.0}
        cluster = int(self.predict(features)[0]) if len(features) > 0 else -1
        profile = self._fingerprints.get(cluster, {})
        confidence = 0.5
        if self._cluster_centers is not None and cluster >= 0:
            available = [f for f in self.FEATURES if f in features.columns]
            entity_vec = self.scaler.transform(features[available].fillna(0).values)[0]
            center = self._cluster_centers[cluster]
            dist = float(np.linalg.norm(entity_vec - center))
            confidence = 1.0 / (1.0 + dist)
        record = {"entity_id": entity_id, "fingerprint": cluster,
                 "fingerprint_type": profile.get("type", "unknown"),
                 "confidence": confidence, "timestamp": timestamp}
        self._tracked_entities.append(record)
        return record

    def get_entity_history(self, entity_id):
        return [e for e in self._tracked_entities if e["entity_id"] == entity_id]

    def tsne_embedding(self, features):
        available = [f for f in self.FEATURES if f in features.columns]
        X = self.scaler.transform(features[available].fillna(0).values)
        tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=min(30, len(features) - 1))
        return tsne.fit_transform(X)

    def _build_profiles(self, features, labels, feature_names):
        for cid in sorted(set(labels)):
            if cid == -1:
                continue
            mask = labels == cid
            cd = features[mask]
            avg_dark = float(cd["dark_share"].mean()) if "dark_share" in cd.columns else 0.0
            avg_size = float(cd["trade_size_mean"].mean()) if "trade_size_mean" in cd.columns else 0.0
            avg_interval = float(cd["trade_interval_mean"].mean()) if "trade_interval_mean" in cd.columns else 0.0
            avg_iceberg = float(cd.get("iceberg_preference", pd.Series([0])).mean())
            if avg_dark > 0.3 and avg_size > np.percentile(features["trade_size_mean"].dropna(), 66):
                ftype = "dark_institution"
            elif avg_interval < np.percentile(features["trade_interval_mean"].dropna(), 33):
                ftype = "hft"
            elif avg_size < np.percentile(features["trade_size_mean"].dropna(), 33):
                ftype = "retail"
            elif avg_iceberg > 0.1:
                ftype = "iceberg_trader"
            else:
                ftype = "market_maker"
            self._fingerprints[cid] = {
                "type": ftype, "size": int(mask.sum()),
                "avg_dark_share": avg_dark, "avg_trade_size": avg_size,
                "avg_interval": avg_interval, "avg_iceberg_pref": avg_iceberg,
                "top_feature": feature_names[int(np.argmax(np.abs(features[feature_names][mask].mean().values)))] if feature_names else "unknown",
            }

    @property
    def fingerprint_summary(self):
        if not self._fingerprints:
            return {"n_fingerprints": 0}
        return {"n_fingerprints": len(self._fingerprints),
                "fingerprints": {str(k): v for k, v in self._fingerprints.items()},
                "total_tracked": len(self._tracked_entities)}

    def summary(self):
        return self.fingerprint_summary

    def report(self):
        if not self._fingerprints:
            return "No fingerprints discovered."
        lines = ["Trader Fingerprint Analysis", f"Clusters: {len(self._fingerprints)}"]
        for cid, profile in sorted(self._fingerprints.items()):
            lines.append(f"Fingerprint {cid}: {profile['type']} ({profile['size']} entities, {profile['avg_dark_share']:.0%} dark)")
        return "\n".join(lines)

    def reset(self):
        self._fingerprints.clear()
        self._tracked_entities.clear()
        self._is_fitted = False
