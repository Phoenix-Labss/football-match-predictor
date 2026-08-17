"""Probability Ensembling and Weight Optimization for Soccer Match Outcomes.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from src.evaluation.metrics import rps, multiclass_log_loss, accuracy


def optimize_ensemble_weights(
    prob_list: list[np.ndarray],
    y_true: np.ndarray,
    loss_type: str = "rps",
) -> np.ndarray:
    """Find continuous optimal convex ensemble weights on validation predictions."""
    k = len(prob_list)
    initial_weights = np.ones(k) / k
    bounds = [(0.0, 1.0) for _ in range(k)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def loss_fun(w):
        w = np.array(w)
        blended = np.zeros_like(prob_list[0])
        for i in range(k):
            blended += w[i] * prob_list[i]
        blended = np.clip(blended, 1e-12, 1.0)
        blended = blended / blended.sum(axis=1, keepdims=True)

        if loss_type == "rps":
            return rps(y_true, blended)
        else:
            return multiclass_log_loss(y_true, blended)

    res = minimize(
        loss_fun,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    opt_w = np.clip(res.x, 0.0, 1.0)
    return opt_w / np.sum(opt_w)


def blend_probabilities(
    prob_list: list[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Compute convex combination of probability arrays."""
    blended = np.zeros_like(prob_list[0])
    for p, w in zip(prob_list, weights):
        blended += w * p
    blended = np.clip(blended, 1e-12, 1.0)
    return blended / blended.sum(axis=1, keepdims=True)
