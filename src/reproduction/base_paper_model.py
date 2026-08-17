"""Base Paper (Berrar, Lopes & Dubitzky, 2024) Model Implementations.

Includes:
1. M0 Gradient Boosted Classifier (Histogram GBDT / LightGBM)
2. M0 Random Forest Classifier
3. M0 Ordinal / Multinomial Logistic Regression Baseline
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_base_paper_model(model_type: str = "gradient_boosting", seed: int = 42):
    """Instantiate the exact learner architectures specified in Berrar et al. (2024)."""
    if model_type == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            l2_regularization=0.5,
            random_state=seed,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            random_state=seed,
            n_jobs=-1,
        )
    elif model_type == "logistic_regression":
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
