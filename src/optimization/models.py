"""Model architectures, wrappers, and calibration tools for soccer outcome prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


class TemperatureCalibrator(BaseEstimator, ClassifierMixin):
    """Post-hoc probability temperature scaling for multiclass calibration."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        # Grid search over temperature on validation probabilities
        best_t = 1.0
        best_loss = float("inf")
        for t in np.linspace(0.5, 2.0, 31):
            log_p = np.log(np.clip(probs, 1e-12, 1.0)) / t
            exp_p = np.exp(log_p - np.max(log_p, axis=1, keepdims=True))
            cal_p = exp_p / exp_p.sum(axis=1, keepdims=True)
            loss = -np.mean(np.log(np.clip(cal_p[np.arange(len(y_true)), y_true.astype(int)], 1e-12, 1.0)))
            if loss < best_loss:
                best_loss = loss
                best_t = t
        self.temperature = best_t
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        log_p = np.log(np.clip(probs, 1e-12, 1.0)) / self.temperature
        exp_p = np.exp(log_p - np.max(log_p, axis=1, keepdims=True))
        return exp_p / exp_p.sum(axis=1, keepdims=True)


def build_model_family(
    model_name: str,
    random_state: int = 42,
    params: dict | None = None,
    class_weight: str | dict | None = None,
):
    """Instantiate a configured model from its family name."""
    params = params or {}

    if model_name == "hist_gbdt":
        default_params = {
            "max_iter": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
            "random_state": random_state,
        }
        default_params.update(params)
        if class_weight:
            default_params["class_weight"] = class_weight
        return HistGradientBoostingClassifier(**default_params)

    elif model_name == "lightgbm":
        default_params = {
            "n_estimators": 300,
            "learning_rate": 0.04,
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "verbosity": -1,
            "n_jobs": -1,
        }
        default_params.update(params)
        if class_weight:
            default_params["class_weight"] = class_weight
        return lgb.LGBMClassifier(**default_params)

    elif model_name == "xgboost":
        default_params = {
            "n_estimators": 300,
            "learning_rate": 0.04,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "verbosity": 0,
            "n_jobs": -1,
            "eval_metric": "mlogloss",
        }
        default_params.update(params)
        return xgb.XGBClassifier(**default_params)

    elif model_name == "catboost":
        default_params = {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 4,
            "l2_leaf_reg": 3.0,
            "random_seed": random_state,
            "verbose": 0,
            "thread_count": -1,
        }
        default_params.update(params)
        return CatBoostClassifier(**default_params)

    elif model_name == "random_forest":
        default_params = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
            "random_state": random_state,
            "n_jobs": -1,
        }
        default_params.update(params)
        if class_weight:
            default_params["class_weight"] = class_weight
        return RandomForestClassifier(**default_params)

    elif model_name == "extra_trees":
        default_params = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
            "random_state": random_state,
            "n_jobs": -1,
        }
        default_params.update(params)
        if class_weight:
            default_params["class_weight"] = class_weight
        return ExtraTreesClassifier(**default_params)

    else:
        raise ValueError(f"Unknown model name: {model_name}")
