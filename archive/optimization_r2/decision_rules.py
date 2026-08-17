"""Learned Class-Specific Decision Rules and Threshold Search for Dynamic Oracle.

Searches for data-driven decision boundaries (draw threshold, margin tolerance,
Elo-gap draw trigger, class bias multipliers) learned strictly on prior validation
portions and evaluated on out-of-sample validation folds.
"""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from src.evaluation.metrics import accuracy, multiclass_log_loss, rps


def apply_decision_rule(
    probs: np.ndarray,
    rule_type: str,
    params: dict,
    elo_diff: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a parameterized decision rule to predicted probability matrix."""
    n = len(probs)
    preds = np.argmax(probs, axis=1)

    if rule_type == "standard_argmax":
        return preds

    elif rule_type == "draw_threshold":
        # Predict Draw if P(Draw) >= tau_d AND |P(Home) - P(Away)| <= delta_margin
        tau_d = params.get("tau_d", 0.30)
        delta_margin = params.get("delta_margin", 0.15)
        p_margin = np.abs(probs[:, 2] - probs[:, 0])
        draw_mask = (probs[:, 1] >= tau_d) & (p_margin <= delta_margin)
        preds[draw_mask] = 1
        return preds

    elif rule_type == "elo_gap_draw":
        # Predict Draw if |Elo_diff| <= max_elo_gap AND P(Draw) >= tau_d
        max_elo_gap = params.get("max_elo_gap", 40.0)
        tau_d = params.get("tau_d", 0.28)
        if elo_diff is not None:
            gap = np.abs(elo_diff)
            draw_mask = (gap <= max_elo_gap) & (probs[:, 1] >= tau_d)
            preds[draw_mask] = 1
        return preds

    elif rule_type == "class_multipliers":
        # Bias probabilities by class multipliers: y_hat = argmax(P * [b_a, b_d, b_h])
        b_a = params.get("b_a", 1.0)
        b_d = params.get("b_d", 1.0)
        b_h = params.get("b_h", 1.0)
        multiplied = probs * np.array([b_a, b_d, b_h])
        return np.argmax(multiplied, axis=1)

    elif rule_type == "combined_draw_asymmetry":
        # Combined multiplier and draw margin threshold
        b_a = params.get("b_a", 1.0)
        b_d = params.get("b_d", 1.0)
        b_h = params.get("b_h", 1.0)
        tau_d = params.get("tau_d", 0.32)
        multiplied = probs * np.array([b_a, b_d, b_h])
        cand_preds = np.argmax(multiplied, axis=1)
        p_margin = np.abs(probs[:, 2] - probs[:, 0])
        draw_mask = (probs[:, 1] >= tau_d) & (p_margin < 0.12)
        cand_preds[draw_mask] = 1
        return cand_preds

    return preds


def evaluate_decision_rules_on_folds(
    val_probs: list[np.ndarray],           # list of prob arrays per fold
    val_targets: list[np.ndarray],         # list of y arrays per fold
    val_feature_dfs: list[pd.DataFrame],   # list of feature dfs per fold
) -> pd.DataFrame:
    """Optimize decision rule thresholds on prior folds and test on subsequent fold."""
    n_folds = len(val_targets)

    # Candidate rule parameter grids
    grid_draw_threshold = [
        {"tau_d": td, "delta_margin": dm}
        for td in [0.26, 0.28, 0.30, 0.32, 0.34, 0.36]
        for dm in [0.08, 0.10, 0.12, 0.15, 0.20]
    ]

    grid_elo_gap = [
        {"max_elo_gap": gap, "tau_d": td}
        for gap in [25.0, 40.0, 60.0, 80.0]
        for td in [0.26, 0.28, 0.30, 0.32]
    ]

    grid_multipliers = [
        {"b_a": ba, "b_d": bd, "b_h": bh}
        for ba in [0.95, 1.0, 1.05]
        for bd in [0.90, 1.0, 1.10, 1.20, 1.30]
        for bh in [0.95, 1.0, 1.05]
    ]

    grid_combined = [
        {"b_a": 1.0, "b_d": bd, "b_h": 1.0, "tau_d": td}
        for bd in [1.0, 1.10, 1.15, 1.20]
        for td in [0.28, 0.30, 0.32, 0.34]
    ]

    rule_configs = [
        ("Standard_Argmax", "standard_argmax", [{}]),
        ("Draw_Threshold", "draw_threshold", grid_draw_threshold),
        ("Elo_Gap_Draw", "elo_gap_draw", grid_elo_gap),
        ("Class_Multipliers", "class_multipliers", grid_multipliers),
        ("Combined_Draw_Asymmetry", "combined_draw_asymmetry", grid_combined),
    ]

    results = []

    for rule_name, rule_type, grid in rule_configs:
        all_eval_preds = []
        all_eval_y = []
        best_params_per_fold = []

        for k in range(n_folds):
            # 1. Fit best rule parameter strictly on training folds (0..k-1)
            # For fold 0, split fold 0 into first half (tune) and second half (test)
            if k == 0:
                n_half = len(val_targets[0]) // 2
                train_p = val_probs[0][:n_half]
                train_y = val_targets[0][:n_half]
                train_elo = val_feature_dfs[0]["elo_diff"].values[:n_half] if "elo_diff" in val_feature_dfs[0].columns else None

                test_p = val_probs[0][n_half:]
                test_y = val_targets[0][n_half:]
                test_elo = val_feature_dfs[0]["elo_diff"].values[n_half:] if "elo_diff" in val_feature_dfs[0].columns else None
            else:
                train_p = np.vstack([val_probs[i] for i in range(k)])
                train_y = np.concatenate([val_targets[i] for i in range(k)])
                train_elo = np.concatenate([val_feature_dfs[i]["elo_diff"].values for i in range(k)]) if "elo_diff" in val_feature_dfs[0].columns else None

                test_p = val_probs[k]
                test_y = val_targets[k]
                test_elo = val_feature_dfs[k]["elo_diff"].values if "elo_diff" in val_feature_dfs[k].columns else None

            best_param = grid[0]
            best_train_acc = -1.0

            for cand_p in grid:
                cand_preds = apply_decision_rule(train_p, rule_type, cand_p, elo_diff=train_elo)
                cand_acc = float(np.mean(cand_preds == train_y))
                if cand_acc > best_train_acc:
                    best_train_acc = cand_acc
                    best_param = cand_p

            best_params_per_fold.append(best_param)

            # 2. Evaluate selected rule on out-of-sample fold
            fold_preds = apply_decision_rule(test_p, rule_type, best_param, elo_diff=test_elo)
            all_eval_preds.append(fold_preds)
            all_eval_y.append(test_y)

        concat_eval_preds = np.concatenate(all_eval_preds)
        concat_eval_y = np.concatenate(all_eval_y)
        total_acc = float(np.mean(concat_eval_preds == concat_eval_y))

        results.append({
            "rule_system": rule_name,
            "val_accuracy": total_acc,
            "learned_parameters": str(best_params_per_fold[-1]),
            "delta_vs_argmax": float(total_acc - float(np.mean(np.concatenate([np.argmax(val_probs[i], axis=1) for i in range(n_folds)]) == np.concatenate(val_targets)))),
        })

    df = pd.DataFrame(results).sort_values("val_accuracy", ascending=False)
    return df
