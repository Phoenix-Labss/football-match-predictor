"""Track 1 experiment: confidence-controlled strength updating (RQ2).

Compares, under identical conditions (same learner, same capacity, same
rolling-origin temporal splits):

    M0-Frozen : no strength updating at all (control)
    M0-Elo    : classic fixed-K Elo updating (control)
    M0-Cap-X  : bounded updating with fixed caps X in {1%, 3%, 5%, 10%}
    M3        : adaptive confidence-controlled updating (proposed)

Every variant uses the same team-form and context features; only the
strength-update rule differs. This isolates the contribution of the
update mechanism.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_matches, add_outcome_labels
from src.data.split import rolling_origin_folds, assert_no_temporal_leakage
from src.features.strength import UpdaterConfig, run_tracker_over_matches
from src.features.team_form import build_m0_feature_matrix
from src.models.predictor import OutcomePredictor
from src.evaluation.metrics import evaluate_all


VARIANTS = {
    "M0-Frozen": {"mode": "frozen"},
    "M0-Elo": {"mode": "fixed_k"},
    "M0-Cap-1": {"mode": "bounded", "cap": 0.01},
    "M0-Cap-3": {"mode": "bounded", "cap": 0.03},
    "M0-Cap-5": {"mode": "bounded", "cap": 0.05},
    "M0-Cap-10": {"mode": "bounded", "cap": 0.10},
    "M3-Adaptive": {"mode": "adaptive"},
}


def _build_variant_features(
    matches: pd.DataFrame, updater_cfg: UpdaterConfig, form_windows: list[int]
) -> pd.DataFrame:
    strength_feats, _ = run_tracker_over_matches(matches, updater_cfg)
    return build_m0_feature_matrix(matches, strength_feats, form_windows)


def run_track1(
    config_path: str | Path = "config/default.yaml",
    project_root: str | Path | None = None,
    variants: dict | None = None,
) -> pd.DataFrame:
    """Run the full Track 1 comparison and return a results table."""
    root = Path(project_root) if project_root else Path.cwd()
    with open(root / config_path if not Path(config_path).is_absolute() else config_path) as f:
        cfg = yaml.safe_load(f)

    variants = variants or VARIANTS
    form_windows = cfg["features"]["form_windows"]
    val_cfg = cfg["validation"]
    eval_cfg = cfg["evaluation"]
    model_cfg = cfg["model"]["gradient_boosting"]

    # ------------------------------------------------------------------ #
    # data
    # ------------------------------------------------------------------ #
    matches = load_matches(cfg, root)
    matches = add_outcome_labels(matches)
    y = matches["outcome"].to_numpy()

    folds = rolling_origin_folds(
        matches,
        n_folds=val_cfg["n_folds"],
        test_fraction=val_cfg["test_fractions"][0],
        min_train_matches=val_cfg["min_train_matches"],
    )
    assert_no_temporal_leakage(folds)

    # ------------------------------------------------------------------ #
    # per-variant evaluation
    # ------------------------------------------------------------------ #
    all_results = []

    for name, overrides in variants.items():
        base_updater = dict(
            mode=overrides.get("mode", "adaptive"),
            k=cfg["updater"]["k"],
            cap=overrides.get("cap", cfg["updater"]["cap"]),
            base_cap=cfg["updater"]["adaptive"]["base_cap"],
            consistency_weight=cfg["updater"]["adaptive"]["consistency_weight"],
            surprise_weight=cfg["updater"]["adaptive"]["surprise_weight"],
            evidence_window=cfg["updater"]["adaptive"]["evidence_window"],
            home_advantage=cfg["features"]["elo"]["home_advantage"],
            initial_rating=cfg["features"]["elo"]["initial_rating"],
        )
        updater_cfg = UpdaterConfig(**base_updater)

        # Features are computed over the FULL chronological table with
        # pre-match snapshots only, so no future information is used.
        X = _build_variant_features(matches, updater_cfg, form_windows)

        # Collect out-of-fold test predictions.
        test_preds = np.full((len(matches), 3), np.nan)
        for fold in folds:
            model = OutcomePredictor(
                n_estimators=model_cfg["n_estimators"],
                learning_rate=model_cfg["learning_rate"],
                max_depth=model_cfg["max_depth"],
                random_state=cfg["model"]["random_state"],
            )
            model.fit(X.iloc[fold.train_idx], y[fold.train_idx])
            test_preds[fold.test_idx] = model.predict_proba(X.iloc[fold.test_idx])

        mask = ~np.isnan(test_preds[:, 0])
        res = evaluate_all(
            y[mask],
            test_preds[mask],
            n_bins=eval_cfg["calibration_bins"],
            bootstrap_samples=eval_cfg["bootstrap_samples"],
        )
        res["variant"] = name
        all_results.append(res)

        print(
            f"[track1] {name:<12} log_loss={res['log_loss']:.4f} "
            f"rps={res['rps']:.4f} ece={res['ece']:.4f} "
            f"acc={res['accuracy']:.4f}"
        )

    results_df = pd.DataFrame(all_results).set_index("variant")

    # ------------------------------------------------------------------ #
    # persist
    # ------------------------------------------------------------------ #
    out_dir = root / cfg["output"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "track1_results.csv")
    with open(out_dir / "track1_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return results_df.reset_index()


if __name__ == "__main__":
    run_track1()