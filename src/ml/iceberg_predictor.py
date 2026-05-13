"""Iceberg Order Prediction — XGBoost Model."""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import get_logger

logger = get_logger(__name__)


class IcebergPredictor:
    PREDICTION_HORIZONS = [5, 10, 20, 50]

    def __init__(self, model_dir: str = "data/models", n_estimators: int = 100,
                 max_depth: int = 6, learning_rate: float = 0.05, prediction_horizon: int = 10):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.prediction_horizon = prediction_horizon
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: list[str] = []
        self._metrics: dict = {}

    def extract_features(self, iceberg_data, orderbook_data=None):
        df = iceberg_data.copy()
        features = pd.DataFrame()
        features["displayed_size"] = df.get("displayed_size", 100)
        features["estimated_total"] = df.get("estimated_total", 1000)
        features["hidden_size"] = df.get("hidden_size", 500)
        features["confidence"] = df.get("confidence", 0.5)
        features["hidden_ratio"] = (features["hidden_size"] / features["displayed_size"].replace(0, 1)).clip(0, 50)
        features["fill_ratio"] = (features["displayed_size"] / features["estimated_total"].replace(0, 1)).clip(0, 1)
        if "side" in df.columns:
            features["side_buy"] = (df["side"] == "buy").astype(int)
            features["side_sell"] = (df["side"] == "sell").astype(int)
        if "type" in df.columns:
            features["type_native"] = (df["type"] == "native").astype(int)
            features["type_synthetic"] = (df["type"] == "synthetic").astype(int)
        if "timestamp" in df.columns:
            features["hour"] = (df["timestamp"] // 3_600_000) % 24
            features["minute"] = (df["timestamp"] // 60_000) % 60
        features["n_fills"] = 0
        self.feature_names = list(features.columns)
        return features.fillna(0)

    def build_labels(self, iceberg_data, trade_data, horizon=None):
        horizon = horizon or self.prediction_horizon
        labels = pd.Series(0, index=iceberg_data.index, dtype=int)
        for idx, row in iceberg_data.iterrows():
            if "hidden_size" not in row or row["hidden_size"] <= 0:
                labels[idx] = 1
                continue
            ts = row.get("timestamp", 0)
            price = row.get("price", 0)
            future_trades = trade_data[
                (trade_data["timestamp"] > ts)
                & (trade_data["timestamp"] <= ts + 60_000)
                & (abs(trade_data.get("price", 0) - price) / max(price, 1) < 0.002)
            ].head(horizon)
            total_future_volume = future_trades.get("volume", pd.Series(0)).sum()
            if total_future_volume >= row["hidden_size"] * 0.9:
                labels[idx] = 1
        return labels

    def train(self, X, y, save=True):
        if len(X) < 100:
            logger.warning(f"Small dataset ({len(X)} samples)")
        unique_y = y.unique()
        if len(unique_y) < 2:
            logger.warning(f"Only one class in labels. Adding synthetic negatives.")
            neg_indices = y.sample(len(y) // 2, random_state=42).index
            y = y.copy()
            y.loc[neg_indices] = 0
        tscv = TimeSeriesSplit(n_splits=min(5, len(X) // 20 + 2))
        self.model = xgb.XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                         learning_rate=self.learning_rate, objective="binary:logistic",
                                         eval_metric="logloss", random_state=42, verbosity=0)
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_pred = self.model.predict(X_val)
            cv_scores.append({"fold": fold, "accuracy": accuracy_score(y_val, y_pred),
                            "precision": precision_score(y_val, y_pred, zero_division=0)})
        self.model.fit(X, y, verbose=False)
        y_pred_all = self.model.predict(X)
        self._metrics = {
            "cv_folds": len(cv_scores),
            "cv_accuracy_mean": float(np.mean([s["accuracy"] for s in cv_scores])),
            "cv_precision_mean": float(np.mean([s["precision"] for s in cv_scores])),
            "final_accuracy": accuracy_score(y, y_pred_all),
            "final_precision": precision_score(y, y_pred_all, zero_division=0),
            "final_recall": recall_score(y, y_pred_all, zero_division=0),
            "n_samples": len(X), "n_features": X.shape[1],
            "feature_importance": dict(zip(self.feature_names[:10],
                                           self.model.feature_importances_[:10].tolist()[:10])),
        }
        if save:
            self._save_model()
        return self._metrics

    def predict(self, X, return_proba=False):
        if self.model is None:
            return np.zeros(len(X))
        X_clean = X[self.feature_names].fillna(0) if self.feature_names else X.fillna(0)
        if return_proba:
            return self.model.predict_proba(X_clean)[:, 1]
        return self.model.predict(X_clean)

    def _save_model(self):
        path = self.model_dir / "iceberg_predictor_xgb.joblib"
        joblib.dump({"model": self.model, "features": self.feature_names}, path)

    def load_model(self, path=None):
        path = path or str(self.model_dir / "iceberg_predictor_xgb.joblib")
        if not Path(path).exists():
            return False
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["features"]
        return True

    @property
    def metrics(self):
        return self._metrics

    @property
    def is_trained(self):
        return self.model is not None

    def report(self):
        if not self._metrics:
            return "Model not trained."
        m = self._metrics
        return "\n".join([
            "Iceberg Predictor (XGBoost)",
            f"Samples: {m['n_samples']}", f"CV Accuracy: {m['cv_accuracy_mean']:.1%}",
            f"Final Acc: {m['final_accuracy']:.1%}",
        ])
