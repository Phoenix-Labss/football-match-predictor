"""Accuracy-Oriented Ensemble Weight Optimization for Dynamic Oracle.

Directly maximizes out-of-sample/validation classification accuracy:
    accuracy(argmax(sum_k w_k * P_k))
subject to w_k >= 0 and sum(w_k) == 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from src.evaluation.metrics import rps, multiclass_log_loss, accuracy, expected_calibration_error


def optimize_accuracy_weights(
    prob_list: list[np.ndarray],
    y_true: np.ndarray,
    n_restarts: int = 15,
) -> tuple[np.ndarray, float]:
    """Find continuous ensemble weights that directly maximize validation classification accuracy.

    Uses smooth softmax-margin surrogate optimization combined with Nelder-Mead / Powell
    local direct search and coordinate polishing.
    """
    k = len(prob_list)
    y_int = y_true.astype(int)
    n_samples = len(y_true)

    def calc_acc(w: np.ndarray) -> float:
        w_norm = np.clip(w, 0.0, 1.0)
        s = np.sum(w_norm)
        if s == 0:
            return 0.0
        w_norm /= s
        blended = np.zeros_like(prob_list[0])
        for i in range(k):
            blended += w_norm[i] * prob_list[i]
        preds = np.argmax(blended, axis=1)
        return float(np.mean(preds == y_int))

    def smooth_loss(w: np.ndarray, tau: float = 0.05) -> float:
        w_norm = np.clip(w, 0.0, 1.0)
        s = np.sum(w_norm)
        if s == 0:
            return 10.0
        w_norm /= s
        blended = np.zeros_like(prob_list[0])
        for i in range(k):
            blended += w_norm[i] * prob_list[i]
        
        # Softmax margin score
        scaled = blended / tau
        exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        soft_p = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        true_soft_p = soft_p[np.arange(n_samples), y_int]
        return float(-np.mean(true_soft_p))

    best_w = np.ones(k) / k
    best_acc = calc_acc(best_w)

    # 1. Smooth surrogate optimization with multi-start
    for seed in range(n_restarts):
        np.random.seed(seed * 42 + 7)
        init_w = np.random.dirichlet(np.ones(k))
        bounds = [(0.0, 1.0) for _ in range(k)]
        
        # Solve smooth surrogate
        res = minimize(
            smooth_loss,
            init_w,
            args=(0.04,),
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        )
        cand_w = np.clip(res.x, 0.0, 1.0)
        cand_w /= np.sum(cand_w)
        cand_acc = calc_acc(cand_w)
        if cand_acc > best_acc:
            best_acc = cand_acc
            best_w = cand_w

    # 2. Local coordinate perturbation polishing
    step_sizes = [0.05, 0.02, 0.01, 0.005]
    for step in step_sizes:
        improved = True
        while improved:
            improved = False
            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    test_w = best_w.copy()
                    if test_w[i] + step <= 1.0 and test_w[j] - step >= 0.0:
                        test_w[i] += step
                        test_w[j] -= step
                        test_acc = calc_acc(test_w)
                        if test_acc > best_acc:
                            best_acc = test_acc
                            best_w = test_w
                            improved = True

    best_w = np.clip(best_w, 0.0, 1.0)
    best_w /= np.sum(best_w)
    return best_w, best_acc


def compare_ensemble_objectives(
    prob_list: list[np.ndarray],
    model_names: list[str],
    y_true: np.ndarray,
) -> pd.DataFrame:
    """Compare Equal, LogLoss-optimized, RPS-optimized, and Accuracy-optimized weights."""
    k = len(prob_list)
    y_int = y_true.astype(int)

    # 1. Equal weights
    w_equal = np.ones(k) / k

    # 2. LogLoss-optimized weights
    from src.optimization.ensemble import optimize_ensemble_weights
    w_logloss = optimize_ensemble_weights(prob_list, y_true, loss_type="log_loss")

    # 3. RPS-optimized weights
    w_rps = optimize_ensemble_weights(prob_list, y_true, loss_type="rps")

    # 4. Accuracy-optimized weights
    w_acc, _ = optimize_accuracy_weights(prob_list, y_true)

    strategies = [
        ("Accuracy_Optimized", w_acc),
        ("LogLoss_Optimized", w_logloss),
        ("RPS_Optimized", w_rps),
        ("Equal_Weights", w_equal),
    ]

    # Also include single model performances
    for i, name in enumerate(model_names):
        w_single = np.zeros(k)
        w_single[i] = 1.0
        strategies.append((f"Single_{name}", w_single))

    records = []
    for name, w in strategies:
        blended = np.zeros_like(prob_list[0])
        for i in range(k):
            blended += w[i] * prob_list[i]
        blended = np.clip(blended, 1e-12, 1.0)
        blended /= blended.sum(axis=1, keepdims=True)

        acc = float(accuracy(y_int, blended))
        ll = float(multiclass_log_loss(y_int, blended))
        norm_r = float(rps(y_int, blended) / 2.0)
        ece = float(expected_calibration_error(y_int, blended, n_bins=15))

        w_dict = {model_names[i]: float(np.round(w[i], 4)) for i in range(k)}

        records.append({
            "strategy": name,
            "val_accuracy": acc,
            "val_log_loss": ll,
            "val_norm_rps": norm_r,
            "val_ece": ece,
            "weights": str(w_dict),
        })

    df = pd.DataFrame(records).sort_values("val_accuracy", ascending=False)
    return df
