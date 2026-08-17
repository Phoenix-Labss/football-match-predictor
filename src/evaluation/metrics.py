"""Evaluation metrics for three-outcome soccer match prediction.

Metrics follow the protocol of Berrar et al. (2024) where applicable:

  - Accuracy        : fraction of correct arg-max predictions
  - Log Loss        : proper scoring rule for predicted probabilities
  - Brier Score     : mean squared probability error (multiclass)
  - RPS             : Ranked Probability Score -- the paper's headline
                      metric for the ordered outcome task
                      (home win > draw > away win)
  - ECE             : Expected Calibration Error, for evaluating the
                      confidence-controlled update mechanism

Class order convention (used everywhere in this project):
    0 = home win, 1 = draw, 2 = away win
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, brier_score_loss


def _validate_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"probs must be (n, 3), got shape {probs.shape}")
    # Clip to avoid log(0); renormalise.
    probs = np.clip(probs, 1e-15, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    probs = _validate_probs(probs)
    return float(np.mean(np.argmax(probs, axis=1) == y_true))


def multiclass_log_loss(y_true: np.ndarray, probs: np.ndarray) -> float:
    probs = _validate_probs(probs)
    return float(log_loss(y_true, probs, labels=[0, 1, 2]))


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Multiclass Brier score: mean over samples of sum_k (p_k - y_k)^2."""
    probs = _validate_probs(probs)
    Y = np.eye(3)[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((probs - Y) ** 2, axis=1)))


def rps(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Ranked Probability Score for the ordered 3-outcome task.

    RPS = (1/(K-1)) * sum_{k=1}^{K-1} (CDF_p(k) - CDF_y(k))^2

    Lower is better. This is the headline metric in Berrar et al. (2024).
    """
    probs = _validate_probs(probs)
    y_true = np.asarray(y_true, dtype=int)
    Y = np.eye(3)[y_true]

    cdf_p = np.cumsum(probs, axis=1)[:, :-1]
    cdf_y = np.cumsum(Y, axis=1)[:, :-1]
    return float(np.mean(np.sum((cdf_p - cdf_y) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15
) -> float:
    """Expected Calibration Error over confidence bins.

    For each match, confidence = max predicted probability and the
    prediction = argmax class. Matches are binned by confidence; ECE is
    the sample-weighted mean absolute gap between confidence and
    observed accuracy per bin.
    """
    probs = _validate_probs(probs)
    y_true = np.asarray(y_true, dtype=int)

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


def calibration_curve_data(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15
) -> list[dict]:
    """Per-bin reliability data for plotting reliability diagrams."""
    probs = _validate_probs(probs)
    y_true = np.asarray(y_true, dtype=int)

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        out.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": int(mask.sum()),
                "avg_confidence": float(conf[mask].mean()),
                "accuracy": float(correct[mask].mean()),
            }
        )
    return out


def bootstrap_metric_ci(
    y_true: np.ndarray,
    probs: np.ndarray,
    metric_fn,
    n_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for any metric(y_true, probs)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    n = len(y_true)
    values = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        values[i] = metric_fn(y_true[idx], probs[idx])
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(metric_fn(y_true, probs)),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def evaluate_all(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
    bootstrap_samples: int = 1000,
) -> dict:
    """Compute the full metric suite with bootstrap CIs on primary metrics."""
    y_true = np.asarray(y_true, dtype=int)

    results = {
        "n_matches": int(len(y_true)),
        "accuracy": accuracy(y_true, probs),
        "log_loss": multiclass_log_loss(y_true, probs),
        "brier": multiclass_brier(y_true, probs),
        "rps": rps(y_true, probs),
        "ece": expected_calibration_error(y_true, probs, n_bins),
    }

    results["log_loss_ci"] = bootstrap_metric_ci(
        y_true, probs, multiclass_log_loss, bootstrap_samples
    )
    results["rps_ci"] = bootstrap_metric_ci(y_true, probs, rps, bootstrap_samples)
    results["ece_ci"] = bootstrap_metric_ci(
        y_true, probs, lambda y, p: expected_calibration_error(y, p, n_bins),
        bootstrap_samples,
    )
    return results