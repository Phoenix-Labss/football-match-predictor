"""Validation Error and Confidence Analysis Module for Dynamic Oracle.

Constructs 3x3 confusion matrices, per-class recall/precision, error pair distribution,
and granular confidence/entropy analysis across temporal validation folds.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


CLASS_NAMES = ["Away", "Draw", "Home"]  # Outcome mapping: 0=Away, 1=Draw, 2=Home


def analyze_errors_and_confidence(
    y_true: np.ndarray,
    probs: np.ndarray,
    feature_df: pd.DataFrame,
    fold_ids: np.ndarray | list[int] | None = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Perform thorough error and prediction confidence analysis on validation predictions.

    Args:
        y_true: Ground truth integer labels (0=Away, 1=Draw, 2=Home).
        probs: Predicted probability matrix of shape (N, 3).
        feature_df: Feature DataFrame corresponding to the validation samples.
        fold_ids: Optional array of fold indices for per-fold breakdown.

    Returns:
        error_df: Summary of confusion pairs and error percentages.
        conf_matrices: Nested dictionary containing per-fold and aggregate 3x3 confusion matrices.
        conf_analysis_df: Per-prediction confidence and systematic error feature breakdown.
    """
    y_pred = np.argmax(probs, axis=1)
    n_samples = len(y_true)

    # 1. Compute 3x3 confusion matrix and metrics
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    # cm[i, j]: actual i, predicted j
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    conf_matrices = {
        "aggregate": {
            "matrix_actual_rows_pred_cols": cm.tolist(),
            "labels": ["Away (0)", "Draw (1)", "Home (2)"],
            "total_samples": int(n_samples),
            "total_correct": int(np.sum(y_true == y_pred)),
            "total_errors": int(np.sum(y_true != y_pred)),
            "accuracy": float(np.mean(y_true == y_pred)),
            "per_class": {
                CLASS_NAMES[i]: {
                    "support": int(supports[i]),
                    "recall_accuracy": float(recalls[i]),
                    "precision": float(precisions[i]),
                    "f1": float(f1s[i]),
                }
                for i in range(3)
            },
        },
        "per_fold": {},
    }

    if fold_ids is not None:
        fold_ids = np.array(fold_ids)
        for f in np.unique(fold_ids):
            mask = fold_ids == f
            f_cm = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1, 2])
            f_p, f_r, f_f, f_s = precision_recall_fscore_support(
                y_true[mask], y_pred[mask], labels=[0, 1, 2], zero_division=0
            )
            conf_matrices["per_fold"][f"fold_{f}"] = {
                "matrix_actual_rows_pred_cols": f_cm.tolist(),
                "accuracy": float(np.mean(y_true[mask] == y_pred[mask])),
                "total_samples": int(np.sum(mask)),
                "per_class": {
                    CLASS_NAMES[i]: {
                        "support": int(f_s[i]),
                        "recall": float(f_r[i]),
                        "precision": float(f_p[i]),
                    }
                    for i in range(3)
                },
            }

    # 2. Confusion Pair Error Analysis
    # Confusion pairs: (Predicted, Actual) where Predicted != Actual
    error_records = []
    total_errors = np.sum(y_true != y_pred)

    pairs = [
        ("Home", "Draw", 2, 1),
        ("Home", "Away", 2, 0),
        ("Draw", "Home", 1, 2),
        ("Draw", "Away", 1, 0),
        ("Away", "Home", 0, 2),
        ("Away", "Draw", 0, 1),
    ]

    for pred_name, act_name, pred_code, act_code in pairs:
        count = int(np.sum((y_pred == pred_code) & (y_true == act_code)))
        pct_of_all_errors = float((count / total_errors) * 100.0) if total_errors > 0 else 0.0
        pct_of_total_matches = float((count / n_samples) * 100.0)
        error_records.append({
            "predicted_class": pred_name,
            "actual_class": act_name,
            "error_type": f"Predicted {pred_name} -> Actual {act_name}",
            "error_count": count,
            "pct_of_all_errors": pct_of_all_errors,
            "pct_of_total_samples": pct_of_total_matches,
        })

    error_df = pd.DataFrame(error_records).sort_values("error_count", ascending=False)

    # 3. Granular Confidence and Failure Mode Analysis
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    max_p = sorted_probs[:, 0]
    second_p = sorted_probs[:, 1]
    margin = max_p - second_p
    
    # Entropy: -sum(p * log(p))
    clipped_p = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(clipped_p * np.log(clipped_p), axis=1)

    conf_analysis = pd.DataFrame({
        "actual_code": y_true,
        "actual_class": [CLASS_NAMES[c] for c in y_true],
        "pred_code": y_pred,
        "pred_class": [CLASS_NAMES[c] for c in y_pred],
        "is_correct": (y_true == y_pred).astype(int),
        "prob_away": probs[:, 0],
        "prob_draw": probs[:, 1],
        "prob_home": probs[:, 2],
        "max_prob": max_p,
        "prob_margin": margin,
        "entropy": entropy,
    })

    # Add domain features for error mode characterization
    feature_cols = [
        "elo_home", "elo_away", "elo_diff", "elo_diff_abs", "elo_ratio",
        "expected_home_score", "expected_away_score", "is_neutral", "is_friendly",
        "h2h_matches", "h2h_win_rate_home", "h2h_gd_home",
        "home_rest_days", "away_rest_days", "rest_diff",
        "diff_pts_5", "diff_gd_5", "diff_opp_gd_5",
        "diff_pts_20", "diff_gd_20", "diff_opp_gd_20",
        "home_surprise", "away_surprise", "home_consistency", "away_consistency",
    ]
    for col in feature_cols:
        if col in feature_df.columns:
            conf_analysis[col] = feature_df[col].values

    return error_df, conf_matrices, conf_analysis
