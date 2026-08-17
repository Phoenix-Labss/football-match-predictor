"""Experiment 1: Hyperparameter Tuning for Adaptive Speed-Limit Mechanism.

Tuning is performed strictly on the validation slices of the 4 rolling-origin
temporal folds. The test sets are kept completely untouched until the final
frozen evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_matches, add_outcome_labels
from src.data.split import rolling_origin_folds, assert_no_temporal_leakage
from src.features.strength import UpdaterConfig, run_tracker_over_matches
from src.features.team_form import build_team_form_features, build_match_context_features
from src.models.predictor import OutcomePredictor
from src.evaluation.metrics import evaluate_all, rps, multiclass_log_loss


def run_tuning(
    project_root: str | Path | None = None,
    config_path: str | Path = "config/default.yaml",
    max_evals: int = 80,
    seed: int = 42,
) -> dict:
    root = Path(project_root) if project_root else Path.cwd()
    cfg_file = root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)

    out_dir = root / "results" / "adaptive_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    matches = load_matches(cfg, root)
    matches = add_outcome_labels(matches)
    y = matches["outcome"].to_numpy()

    val_cfg = cfg["validation"]
    folds = rolling_origin_folds(
        matches,
        n_folds=val_cfg["n_folds"],
        test_fraction=val_cfg["test_fractions"][0],
        min_train_matches=val_cfg["min_train_matches"],
    )
    assert_no_temporal_leakage(folds)

    # Precompute static form & context features once outside the search loop
    form_windows = cfg["features"]["form_windows"]
    form_feats = build_team_form_features(matches, form_windows)
    ctx_feats = build_match_context_features(matches)
    base_matrix = pd.concat([form_feats, ctx_feats], axis=1)

    # Search space defined in research plan
    windows = [3, 5, 7, 10, 15]
    wc_list = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    ws_list = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]
    base_caps = [0.005, 0.010, 0.015, 0.020]
    max_caps = [0.020, 0.030, 0.050, 0.075, 0.100]

    all_combos = []
    for w in windows:
        for wc in wc_list:
            for ws in ws_list:
                for bc in base_caps:
                    for mc in max_caps:
                        if bc < mc:
                            all_combos.append({
                                "evidence_window": w,
                                "consistency_weight": wc,
                                "surprise_weight": ws,
                                "base_cap": bc,
                                "max_cap": mc,
                            })

    rng = np.random.default_rng(seed)
    default_combo = {
        "evidence_window": 7,
        "consistency_weight": 3.0,
        "surprise_weight": 1.0,
        "base_cap": 0.01,
        "max_cap": 0.05,
    }
    
    selected_combos = [default_combo]
    other_combos = [c for c in all_combos if c != default_combo]
    rng.shuffle(other_combos)
    selected_combos.extend(other_combos[: max_evals - 1])

    print(f"[tuning] Evaluating {len(selected_combos)} candidate configurations on validation folds...")

    model_cfg = cfg["model"]["gradient_boosting"]
    tuning_results = []
    best_val_rps = float("inf")
    best_params = None

    for i, params in enumerate(selected_combos):
        updater_cfg = UpdaterConfig(
            mode="adaptive",
            k=cfg["updater"]["k"],
            base_cap=params["base_cap"],
            max_cap=params["max_cap"],
            consistency_weight=params["consistency_weight"],
            surprise_weight=params["surprise_weight"],
            evidence_window=params["evidence_window"],
            home_advantage=cfg["features"]["elo"]["home_advantage"],
            initial_rating=cfg["features"]["elo"]["initial_rating"],
        )

        strength_feats, _ = run_tracker_over_matches(matches, updater_cfg)
        X = pd.concat([strength_feats, base_matrix], axis=1)
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

        val_rps_list = []
        val_loss_list = []

        for fold in folds:
            clf = OutcomePredictor(
                n_estimators=100,  # Efficient during validation grid search
                learning_rate=model_cfg["learning_rate"],
                max_depth=model_cfg["max_depth"],
                random_state=cfg["model"]["random_state"],
            )
            clf.fit(X.iloc[fold.train_idx], y[fold.train_idx])
            val_preds = clf.predict_proba(X.iloc[fold.val_idx])
            val_y = y[fold.val_idx]

            val_rps_list.append(rps(val_y, val_preds))
            val_loss_list.append(multiclass_log_loss(val_y, val_preds))

        mean_val_rps = float(np.mean(val_rps_list))
        mean_val_loss = float(np.mean(val_loss_list))

        record = {
            **params,
            "val_rps_unnorm": mean_val_rps,
            "val_rps_norm": mean_val_rps / 2.0,
            "val_log_loss": mean_val_loss,
        }
        tuning_results.append(record)

        if mean_val_rps < best_val_rps:
            best_val_rps = mean_val_rps
            best_params = params

        if (i + 1) % 20 == 0 or i == 0:
            print(f"[tuning] Step {i+1}/{len(selected_combos)}: Best Val RPS (norm) = {best_val_rps/2.0:.4f}")

    tuning_df = pd.DataFrame(tuning_results).sort_values("val_rps_norm")
    tuning_df.to_csv(out_dir / "tuning_grid_validation.csv", index=False)

    with open(out_dir / "best_hyperparameters.json", "w") as f:
        json.dump({
            "best_params": best_params,
            "best_val_rps_norm": best_val_rps / 2.0,
            "n_evaluated": len(selected_combos),
        }, f, indent=2)

    print(f"\n[tuning] Optimal Validation Configuration: {best_params}")
    print(f"[tuning] Best Validation Normalized RPS: {best_val_rps/2.0:.4f}")

    # Final frozen test evaluation on untouched test folds with full 300 estimators
    print("[tuning] Evaluating frozen optimal configuration on untouched test folds (n_estimators=300)...")
    best_updater = UpdaterConfig(
        mode="adaptive",
        k=cfg["updater"]["k"],
        base_cap=best_params["base_cap"],
        max_cap=best_params["max_cap"],
        consistency_weight=best_params["consistency_weight"],
        surprise_weight=best_params["surprise_weight"],
        evidence_window=best_params["evidence_window"],
        home_advantage=cfg["features"]["elo"]["home_advantage"],
        initial_rating=cfg["features"]["elo"]["initial_rating"],
    )
    best_strength_feats, _ = run_tracker_over_matches(matches, best_updater)
    X_best = pd.concat([best_strength_feats, base_matrix], axis=1)
    X_best = X_best.fillna(X_best.median(numeric_only=True)).fillna(0.0)

    test_preds = np.full((len(matches), 3), np.nan)
    for fold in folds:
        clf = OutcomePredictor(
            n_estimators=model_cfg["n_estimators"],
            learning_rate=model_cfg["learning_rate"],
            max_depth=model_cfg["max_depth"],
            random_state=cfg["model"]["random_state"],
        )
        clf.fit(X_best.iloc[fold.train_idx], y[fold.train_idx])
        test_preds[fold.test_idx] = clf.predict_proba(X_best.iloc[fold.test_idx])

    mask = ~np.isnan(test_preds[:, 0])
    test_eval = evaluate_all(y[mask], test_preds[mask], n_bins=15)
    test_eval["normalized_rps"] = test_eval["rps"] / 2.0
    test_eval["best_params"] = best_params

    with open(out_dir / "test_evaluation_frozen.json", "w") as f:
        json.dump(test_eval, f, indent=2, default=str)

    print(f"[tuning] Frozen Test Results: Acc={test_eval['accuracy']:.4f}, "
          f"LogLoss={test_eval['log_loss']:.4f}, NormRPS={test_eval['normalized_rps']:.4f}, ECE={test_eval['ece']:.4f}")

    return {
        "best_params": best_params,
        "val_results": tuning_df,
        "test_eval": test_eval,
    }


if __name__ == "__main__":
    run_tuning()
