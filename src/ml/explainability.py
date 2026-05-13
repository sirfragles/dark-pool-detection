"""XAI — Model Explainability with SHAP.

Provides interpretability for dark pool detection ML models.
Explains WHY a model made a particular prediction.

Methods:
- SHAP TreeExplainer for XGBoost models (exact, fast)
- QuickExplainer: lightweight SHAP-free fallback
- Waterfall-style per-prediction feature contributions
- Detection score decomposition: VPIN vs Iceberg vs Dark
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import get_logger

logger = get_logger(__name__)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not installed. Install with: pip install shap")


class ModelExplainer:
    """Explain ML model predictions using SHAP."""

    def __init__(self, cache_dir="data/models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._explainers = {}
        self._shap_values = {}

    def explain_model(self, model, X_background, X_explain=None, feature_names=None, method="auto", max_background=200, max_explain=100):
        if not HAS_SHAP:
            return {"error": "SHAP not installed"}
        if feature_names is None:
            feature_names = X_background.columns.tolist() if hasattr(X_background, "columns") else [f"feat_{i}" for i in range(X_background.shape[1])]
        bg = X_background[:max_background] if hasattr(X_background, "iloc") else X_background
        ex = (X_explain or bg)[:max_explain] if hasattr((X_explain or bg), "iloc") else (X_explain or bg)
        try:
            if method in ("auto", "tree"):
                explainer = shap.TreeExplainer(model)
                method = "tree"
            else:
                explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.predict(x), bg.values if hasattr(bg, "values") else bg)
            ex_v = ex.values if hasattr(ex, "values") else ex
            sv = explainer.shap_values(ex_v)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            if sv.ndim == 1:
                sv = sv.reshape(1, -1)
            importance = np.abs(sv).mean(axis=0)
            top = sorted(zip(feature_names[:len(importance)], importance), key=lambda x: x[1], reverse=True)
            return {"method": method, "feature_importance": dict(top), "top_features": [(f, float(i)) for f, i in top[:5]], "n_features": len(feature_names)}
        except Exception as e:
            return {"error": str(e)}

    def explain_prediction(self, model, X_single, feature_names=None, n_top_features=10):
        if not HAS_SHAP:
            return {"error": "SHAP not installed"}
        if feature_names is None and hasattr(X_single, "columns"):
            feature_names = X_single.columns.tolist()
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_single)
            if isinstance(sv, list):
                sv = sv[1][0] if len(sv) > 1 else sv[0][0]
            else:
                sv = sv[0]
            base = float(explainer.expected_value) if not isinstance(explainer.expected_value, list) else float(explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0])
            pred = base + float(np.sum(sv))
            contributions = []
            for i in range(min(len(sv), len(feature_names or []))):
                contributions.append({"feature": str(feature_names[i]), "contribution": float(sv[i])})
            contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            return {"base_value": base, "prediction": float(np.clip(pred, 0, 1)), "contributions": contributions[:n_top_features]}
        except Exception as e:
            return {"error": str(e)}

    def decompose_detection_score(self, detection_results):
        score = detection_results.get("detection_score", {})
        contributions = []
        total = 0.0
        ice_score = score.get("iceberg_detection", 0)
        n_ice = detection_results.get("iceberg", {}).get("n_active", 0)
        contributions.append({"module": "iceberg_detection", "score": ice_score, "weight": 0.33, "detail": f"{n_ice} active icebergs"})
        total += ice_score * 0.33
        vpin_score = score.get("vpin_signal", 0)
        vpin_mean = detection_results.get("vpin", {}).get("mean", 0)
        contributions.append({"module": "vpin_signal", "score": vpin_score, "weight": 0.33, "detail": f"mean VPIN = {vpin_mean:.3f}"})
        total += vpin_score * 0.33
        dark_score = score.get("dark_detection", 0)
        dark_pct = detection_results.get("dark_analysis", {}).get("dark_volume_pct", 0)
        contributions.append({"module": "dark_detection", "score": dark_score, "weight": 0.33, "detail": f"{dark_pct:.1f}% dark volume"})
        total += dark_score * 0.33
        return {"overall_score": score.get("overall", 0), "module_contributions": contributions,
                "top_contributor": max(contributions, key=lambda x: x["score"]*x["weight"])["module"],
                "weakest_link": min(contributions, key=lambda x: x["score"]*x["weight"])["module"]}


class QuickExplainer:
    """Lightweight model explainability WITHOUT SHAP dependency."""

    @staticmethod
    def feature_importance_xgboost(model):
        try:
            return {f"feat_{i}": float(imp) for i, imp in enumerate(model.feature_importances_) if imp > 0}
        except Exception:
            return {}

    @staticmethod
    def permutation_importance(model, X, y, feature_names=None, n_repeats=5, metric_name="accuracy"):
        from sklearn.metrics import accuracy_score, precision_score
        metric = accuracy_score if metric_name == "accuracy" else precision_score
        feature_names = feature_names or X.columns.tolist()
        baseline_pred = model.predict(X)
        baseline = metric(y, baseline_pred)
        importances = {}
        for i, feat in enumerate(feature_names):
            scores = []
            for _ in range(n_repeats):
                X_perm = X.copy()
                X_perm.iloc[:, i] = np.random.permutation(X_perm.iloc[:, i].values)
                scores.append(baseline - metric(y, model.predict(X_perm)))
            importances[feat] = float(np.mean(scores))
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    @staticmethod
    def prediction_contributions(model, X_row, feature_names=None):
        feature_names = feature_names or X_row.columns.tolist()
        baseline = float(model.predict_proba(X_row)[:, 1][0]) if hasattr(model, "predict_proba") else float(model.predict(X_row)[0])
        contributions = []
        X_mod = X_row.copy()
        for i, feat in enumerate(feature_names):
            original = X_mod.iloc[0, i]
            X_mod.iloc[0, i] = 0
            perturbed = float(model.predict_proba(X_mod)[:, 1][0]) if hasattr(model, "predict_proba") else float(model.predict(X_mod)[0])
            X_mod.iloc[0, i] = original
            contributions.append({"feature": feat, "value": float(original), "contribution": baseline - perturbed})
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return {"base_prediction": baseline, "contributions": contributions[:10]}
