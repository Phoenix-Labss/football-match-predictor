"""Meta-Classifier / Stacking Decision Layer for Dynamic Oracle.

Constructs an out-of-fold meta-dataset combining base model probability vectors,
uncertainty metrics (entropy, margin), and domain context features to directly
classify 3-way match outcomes (Home, Draw, Away).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from src.evaluation.metrics import accuracy, multiclass_log_loss, rps, expected_calibration_error


def construct_meta_features(
    prob_dict: dict[str, np.ndarray],
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build rich meta-feature matrix from base model predictions and context."""
    meta_cols = {}

    # 1. Base model probabilities
    for m_name, probs in prob_dict.items():
        meta_cols[f"{m_name}_p_away"] = probs[:, 0]
        meta_cols[f"{m_name}_p_draw"] = probs[:, 1]
        meta_cols[f"{m_name}_p_home"] = probs[:, 2]

        sorted_p = np.sort(probs, axis=1)[:, ::-1]
        meta_cols[f"{m_name}_max_p"] = sorted_p[:, 0]
        meta_cols[f"{m_name}_margin"] = sorted_p[:, 0] - sorted_p[:, 1]

        clipped_p = np.clip(probs, 1e-12, 1.0)
        meta_cols[f"{m_name}_entropy"] = -np.sum(clipped_p * np.log(clipped_p), axis=1)

    # 2. Ensemble average probability summary
    all_p = np.array(list(prob_dict.values()))  # (K, N, 3)
    mean_p = np.mean(all_p, axis=0)  # (N, 3)
    meta_cols["ens_mean_p_away"] = mean_p[:, 0]
    meta_cols["ens_mean_p_draw"] = mean_p[:, 1]
    meta_cols["ens_mean_p_home"] = mean_p[:, 2]
    meta_cols["ens_std_p_draw"] = np.std(all_p[:, :, 1], axis=0)
    meta_cols["ens_std_p_home"] = np.std(all_p[:, :, 2], axis=0)

    # 3. Contextual and football domain features
    domain_candidates = [
        "elo_diff", "elo_diff_abs", "elo_ratio", "expected_home_score",
        "is_neutral", "is_friendly", "is_world_cup",
        "h2h_win_rate_home", "h2h_draw_rate", "h2h_gd_home", "h2h_draw_affinity",
        "diff_pts_5", "diff_gd_5", "diff_opp_gd_5",
        "diff_pts_20", "diff_gd_20", "diff_opp_gd_20",
        "rest_diff", "rest_advantage_flag", "home_surprise", "away_surprise",
        "diff_form_accel", "diff_elo_velocity_5", "diff_clean_sheet_10",
    ]
    for col in domain_candidates:
        if col in feature_df.columns:
            meta_cols[col] = feature_df[col].values

    return pd.DataFrame(meta_cols, index=feature_df.index).fillna(0.0)


def evaluate_meta_models_on_folds(
    base_val_probs: dict[str, list[np.ndarray]],  # m_name -> list of prob arrays per fold
    val_targets: list[np.ndarray],               # list of y arrays per fold
    val_feature_dfs: list[pd.DataFrame],         # list of feature dataframes per fold
) -> pd.DataFrame:
    """Evaluate candidate meta-classifiers strictly out-of-fold across temporal folds."""
    n_folds = len(val_targets)
    
    # Construct fold-by-fold meta datasets
    fold_meta_dfs = []
    for f in range(n_folds):
        f_prob_dict = {m_name: base_val_probs[m_name][f] for m_name in base_val_probs}
        f_meta_df = construct_meta_features(f_prob_dict, val_feature_dfs[f])
        fold_meta_dfs.append(f_meta_df)

    candidate_meta_models = {
        "Meta_LogisticRegression": lambda: LogisticRegression(C=0.5, max_iter=500, random_state=42),
        "Meta_LightGBM": lambda: lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, max_depth=3, num_leaves=7, subsample=0.8, reg_alpha=1.0, reg_lambda=2.0, verbosity=-1, random_state=42, n_jobs=-1),
        "Meta_XGBoost": lambda: xgb.XGBClassifier(n_estimators=100, learning_rate=0.03, max_depth=3, subsample=0.8, reg_alpha=1.0, reg_lambda=2.0, random_state=42, n_jobs=-1, eval_metric="mlogloss"),
        "Meta_CatBoost": lambda: CatBoostClassifier(iterations=120, learning_rate=0.04, depth=3, l2_leaf_reg=3.0, verbose=0, random_state=42),
        "Meta_HistGBDT": lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.03, max_depth=3, l2_regularization=2.0, random_state=42),
    }

    results = []

    for name, model_factory in candidate_meta_models.items():
        all_meta_preds = []
        all_meta_y = []

        # Progressive rolling evaluation across folds:
        # Train meta-model on folds 0..k-1 and test on fold k (for k >= 1)
        # For fold 0 (cold start), fit on half of fold 0 and test on second half
        for k in range(n_folds):
            if k == 0:
                # Split fold 0 temporally in half
                n_half = len(val_targets[0]) // 2
                train_x = fold_meta_dfs[0].iloc[:n_half]
                train_y = val_targets[0][:n_half]
                test_x = fold_meta_dfs[0].iloc[n_half:]
                test_y = val_targets[0][n_half:]
            else:
                train_x = pd.concat([fold_meta_dfs[i] for i in range(k)], ignore_index=True)
                train_y = np.concatenate([val_targets[i] for i in range(k)])
                test_x = fold_meta_dfs[k]
                test_y = val_targets[k]

            clf = model_factory()
            clf.fit(train_x, train_y)
            pred_probs = clf.predict_proba(test_x)

            all_meta_preds.append(pred_probs)
            all_meta_y.append(test_y)

        concat_preds = np.vstack(all_meta_preds)
        concat_y = np.concatenate(all_meta_y)

        acc = float(accuracy(concat_y, concat_preds))
        ll = float(multiclass_log_loss(concat_y, concat_preds))
        norm_r = float(rps(concat_y, concat_preds) / 2.0)
        ece = float(expected_calibration_error(concat_y, concat_preds, n_bins=15))

        results.append({
            "meta_model": name,
            "val_accuracy": acc,
            "val_log_loss": ll,
            "val_norm_rps": norm_r,
            "val_ece": ece,
        })

    df = pd.DataFrame(results).sort_values("val_accuracy", ascending=False)
    return df
