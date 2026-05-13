"""Dark Trade Predictor — LSTM Sequence Model.

Predicts whether the next N trades will include dark pool activity.
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import get_logger

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    logger.warning("PyTorch not installed. LSTM will use fallback mode.")

if HAS_TORCH:
    class DarkTradeLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, dropout=dropout, batch_first=True)
            self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(32, 1), nn.Sigmoid())
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
else:
    DarkTradeLSTM = None


class DarkTradePredictor:
    FEATURES = ["trade_size", "trade_direction", "is_dark", "price_change",
                "bid_ask_imbalance", "spread_bps", "volume_momentum", "time_since_last_trade"]

    def __init__(self, model_dir="data/models", sequence_length=50,
                 prediction_horizon_seconds=5, hidden_size=64, num_layers=2):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon_seconds
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self._metrics = {}
        self._is_fitted = False

    def build_sequences(self, trades, label_col=None):
        if len(trades) < self.sequence_length + 1:
            return np.array([]), None, np.array([])
        df = trades.copy()
        if "volume" not in df.columns:
            df["volume"] = 100
        if "price" not in df.columns:
            df["price"] = 100.0
        if "is_dark" not in df.columns:
            df["is_dark"] = 0
        df["price_change"] = df["price"].diff().fillna(0)
        df["trade_direction"] = np.sign(df["price_change"])
        df["trade_size"] = np.log1p(df["volume"])
        df["bid_ask_imbalance"] = df["price_change"].rolling(5, min_periods=1).mean()
        df["spread_bps"] = df["price"].pct_change().abs().rolling(5, min_periods=1).mean() * 10000
        df["volume_momentum"] = df["volume"].diff(5).fillna(0)
        df["time_since_last_trade"] = df.get("timestamp", pd.Series(range(len(df)))).diff().fillna(0)
        feature_cols = [f for f in self.FEATURES if f in df.columns]
        X_raw = df[feature_cols].astype(float).values
        if self.scaler_mean is None:
            self.scaler_mean = np.nanmean(X_raw, axis=0)
            self.scaler_std = np.nanstd(X_raw, axis=0)
            self.scaler_std[self.scaler_std == 0] = 1.0
        X_scaled = (X_raw - self.scaler_mean) / self.scaler_std
        X_scaled = np.nan_to_num(X_scaled, 0.0)
        n_seqs = len(df) - self.sequence_length
        X = np.zeros((n_seqs, self.sequence_length, len(feature_cols)), dtype=np.float32)
        timestamps = np.zeros(n_seqs, dtype=np.float64)
        for i in range(n_seqs):
            X[i] = X_scaled[i:i + self.sequence_length]
            timestamps[i] = df["timestamp"].iloc[i + self.sequence_length] if "timestamp" in df.columns else i
        y = None
        if label_col:
            y = np.zeros(n_seqs, dtype=np.float32)
            for i in range(n_seqs):
                end_idx = i + self.sequence_length
                window = df.iloc[end_idx:end_idx + self.prediction_horizon]
                if not window.empty and window[label_col].any():
                    y[i] = 1.0
        return X, y, timestamps

    def train(self, X, y, epochs=50, batch_size=64, learning_rate=0.001, validation_split=0.2, save=True):
        if not HAS_TORCH:
            return self._train_fallback(X, y, save)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = X.shape[2]
        self.model = DarkTradeLSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers).to(device)
        split_idx = int(len(X) * (1 - validation_split))
        X_train = torch.tensor(X[:split_idx], dtype=torch.float32).to(device)
        y_train = torch.tensor(y[:split_idx], dtype=torch.float32).to(device)
        X_val = torch.tensor(X[split_idx:], dtype=torch.float32).to(device)
        y_val = torch.tensor(y[split_idx:], dtype=torch.float32).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        best_val_loss = float("inf")
        for epoch in range(epochs):
            self.model.train()
            for i in range(0, len(X_train), batch_size):
                bx, by = X_train[i:i+batch_size], y_train[i:i+batch_size]
                optimizer.zero_grad()
                pred = self.model(bx).squeeze()
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val).squeeze()
                val_loss = criterion(val_pred, y_val).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save:
                        self._save_model()
        self._is_fitted = True
        self._metrics = {"n_sequences": len(X), "sequence_length": self.sequence_length, "n_features": input_size}
        return self._metrics

    def _train_fallback(self, X, y, save):
        try:
            import xgboost as xgb
        except ImportError:
            return {"error": "No ML backend available"}
        n_seqs, seq_len, n_feats = X.shape
        flat = np.zeros((n_seqs, n_feats * 3 + 3))
        for i in range(n_feats):
            flat[:, i * 3] = np.mean(X[:, :, i], axis=1)
            flat[:, i * 3 + 1] = np.std(X[:, :, i], axis=1)
            flat[:, i * 3 + 2] = np.max(np.abs(X[:, :, i]), axis=1)
        flat[:, -3] = np.mean(X[:, -5:, :], axis=(1, 2)) - np.mean(X[:, :5, :], axis=(1, 2))
        flat[:, -2] = np.sum(np.abs(np.diff(X, axis=1)), axis=(1, 2))
        flat[:, -1] = np.mean(X[:, :, 2], axis=1)
        split_idx = int(n_seqs * 0.8)
        model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                                   objective="binary:logistic", random_state=42, verbosity=0)
        model.fit(flat[:split_idx], y[:split_idx])
        full_acc = float(model.score(flat, y))
        self._fallback_model = model
        self._is_fitted = True
        self._metrics = {"backend": "xgboost_fallback", "final_accuracy": full_acc,
                        "n_sequences": n_seqs, "sequence_length": seq_len}
        return self._metrics

    def predict(self, X, return_proba=True):
        if not self._is_fitted:
            return np.zeros(len(X))
        if HAS_TORCH and self.model is not None:
            self.model.eval()
            device = next(self.model.parameters()).device
            with torch.no_grad():
                probs = self.model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze().cpu().numpy()
        elif hasattr(self, '_fallback_model'):
            n_seqs, seq_len, n_feats = X.shape
            flat = np.zeros((n_seqs, n_feats * 3 + 3))
            for i in range(n_feats):
                flat[:, i * 3] = np.mean(X[:, :, i], axis=1)
                flat[:, i * 3 + 1] = np.std(X[:, :, i], axis=1)
                flat[:, i * 3 + 2] = np.max(np.abs(X[:, :, i]), axis=1)
            flat[:, -3] = np.mean(X[:, -5:, :], axis=(1, 2)) - np.mean(X[:, :5, :], axis=(1, 2))
            flat[:, -2] = np.sum(np.abs(np.diff(X, axis=1)), axis=(1, 2))
            flat[:, -1] = np.mean(X[:, :, 2], axis=1)
            probs = self._fallback_model.predict_proba(flat)[:, 1]
        else:
            return np.zeros(len(X))
        return probs if return_proba else (probs > 0.5).astype(int)

    def predict_streaming(self, trade_sequence):
        if len(trade_sequence) < self.sequence_length:
            return {"probability": 0.0, "alert": "insufficient_data"}
        recent = trade_sequence.tail(self.sequence_length)
        X, _, _ = self.build_sequences(recent)
        if len(X) == 0:
            return {"probability": 0.0, "alert": "error"}
        prob = float(self.predict(X[-1:])[0])
        alert = "critical" if prob > 0.8 else "warning" if prob > 0.6 else "watch" if prob > 0.4 else "none"
        return {"probability": prob, "alert": alert}

    def _save_model(self):
        if HAS_TORCH and self.model is not None:
            torch.save({"model_state_dict": self.model.state_dict(), "scaler_mean": self.scaler_mean,
                       "scaler_std": self.scaler_std, "metrics": self._metrics},
                      self.model_dir / "dark_trade_lstm.pt")
        elif hasattr(self, '_fallback_model'):
            joblib.dump({"model": self._fallback_model, "scaler_mean": self.scaler_mean,
                        "scaler_std": self.scaler_std, "metrics": self._metrics},
                       self.model_dir / "dark_trade_fallback.joblib")

    def load_model(self, path=None):
        if HAS_TORCH:
            path = path or str(self.model_dir / "dark_trade_lstm.pt")
            if not Path(path).exists():
                fb = self.model_dir / "dark_trade_fallback.joblib"
                if fb.exists():
                    data = joblib.load(fb)
                    self._fallback_model = data["model"]
                    self._is_fitted = True
                    return True
                return False
            ckpt = torch.load(path, map_location="cpu")
            self.model = DarkTradeLSTM(input_size=ckpt.get("n_features", 8))
            self.model.load_state_dict(ckpt["model_state_dict"])
            self._is_fitted = True
            return True
        else:
            path = path or str(self.model_dir / "dark_trade_fallback.joblib")
            if not Path(path).exists():
                return False
            data = joblib.load(path)
            self._fallback_model = data["model"]
            self._is_fitted = True
            return True

    @property
    def metrics(self):
        return self._metrics

    @property
    def is_trained(self):
        return self._is_fitted

    def report(self):
        if not self._metrics:
            return "Model not trained."
        m = self._metrics
        return f"Dark Trade Predictor ({m.get('backend', 'lstm')})\nSequences: {m.get('n_sequences', 0)}\nAccuracy: {m.get('final_accuracy', 0):.1%}"
