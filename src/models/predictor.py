"""Outcome probability predictor (shared across all model variants).

The same learner class and capacity is used for every model variant
(M0 / M0-Elo / M0-Frozen / M3) so that performance differences can be
attributed to the feature/update differences rather than to model
capacity -- a key experimental control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


class OutcomePredictor:
    """Gradient-boosted classifier mapping match features to
    P(home win), P(draw), P(away win)."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        random_state: int = 42,
    ):
        self.clf = HistGradientBoostingClassifier(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            early_stopping=False,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "OutcomePredictor":
        self.clf.fit(X, y)
        self.feature_names_ = list(X.columns)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return (n, 3) probabilities in class order [home, draw, away]."""
        missing = set(self.feature_names_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features at predict time: {sorted(missing)}")
        return self.clf.predict_proba(X[self.feature_names_])

    @property
    def feature_importances(self) -> dict:
        """Histogram-based GBDT has no direct importances; expose null."""
        return {}