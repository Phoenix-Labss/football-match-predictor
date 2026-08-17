"""Track 2 experiment: player-aware team representations (RQ1).

Compares, under identical conditions (same learner, capacity, splits):

    M0 : team-level features only (baseline)
    M1 : M0 + player ability & form channels
    M2 : M1 + availability & chemistry (interactions)

This isolates the contribution of the player-aware representation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_matches, add_outcome_labels
from src.data.split import rolling_origin_folds, assert_no_temporal_leakage
from src.data.synthetic_players import generate_synthetic_player_matches
from src.features.strength import UpdaterConfig, run_tracker_over_matches
from src.features.team_form import build_m0_feature_matrix
from src.features.player import build_player_team_features
from src.models.predictor import OutcomePredictor
from src.evaluation.metrics import evaluate_all

PLAYER_FEATURE_GROUPS = {
    "ability": ["att_ability", "mid_ability", "def_ability", "gk_ability", "avg_ability"],
    "form": ["att_form", "mid_form", "def_form", "gk_form"],
    "availability": ["att_expected", "mid_expected", "def_expected", "gk_expected"],
    "chemistry": ["chemistry_mean", "squad_depth"],
}


def _load_player_matches(cfg: dict, root: Path, matches: pd.DataFrame) -> pd.DataFrame:
    """Load real player data if available, else synthetic."""
    sqlite_path = root / cfg["data"]["european_soccer_sqlite"]
    if sqlite_path.exists():
        from src.data.european_soccer import load_player_matches  # future module

        return load_player_matches(sqlite_path)
    print(f"[track2] '{sqlite_path}' not found -- using synthetic player data.")
    return generate_synthetic_player_matches(matches, seed=cfg["data"]["synthetic"]["seed"])


def run_track2(
    config_path: str | Path = "config/default.yaml",
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    root = Path(project_root) if project_root else Path.cwd()
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

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

    player_matches = _load_player_matches(cfg, root, matches)

    folds = rolling_origin_folds(
        matches,
        n_folds=val_cfg["n_folds"],
        test_fraction=val_cfg["test_fractions"][0],
        min_train_matches=val_cfg["min_train_matches"],
    )
    assert_no_temporal_leakage(folds)

    # ------------------------------------------------------------------ #
    # base (M0) features: strength + form + context
    # ------------------------------------------------------------------ #
    updater_cfg = UpdaterConfig.from_config(cfg["updater"])
    strength_feats, _ = run_tracker_over_matches(matches, updater_cfg)
    X_m0 = build_m0_feature_matrix(matches, strength_feats, form_windows)

    # ------------------------------------------------------------------ #
    # player features, joined onto matches (pre-match snapshots)
    # ------------------------------------------------------------------ #
    team_player_feats = build_player_team_features(player_matches)

    def join_player_feats(X: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
        cols = sum((PLAYER_FEATURE_GROUPS[g] for g in groups), [])
        pf = team_player_feats[["date", "team"] + cols].copy()

        home_pf = pf.copy()
        home_pf.columns = ["date", "home_team"] + [f"home_{c}" for c in cols]
        away_pf = pf.copy()
        away_pf.columns = ["date", "away_team"] + [f"away_{c}" for c in cols]

        out = matches[["date", "home_team", "away_team"]].copy()
        out = out.merge(home_pf, on=["date", "home_team"], how="left")
        out = out.merge(away_pf, on=["date", "away_team"], how="left")
        out = out.drop(columns=["date", "home_team", "away_team"])
        out.index = X.index
        combined = pd.concat([X, out], axis=1)
        combined = combined.fillna(combined.median(numeric_only=True)).fillna(0.0)
        return combined

    variants = {
        "M0": [],
        "M1": ["ability", "form"],
        "M2": ["ability", "form", "availability", "chemistry"],
    }

    all_results = []
    for name, groups in variants.items():
        X = X_m0 if not groups else join_player_feats(X_m0, groups)

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
            f"[track2] {name:<4} log_loss={res['log_loss']:.4f} "
            f"rps={res['rps']:.4f} ece={res['ece']:.4f} acc={res['accuracy']:.4f}"
        )

    results_df = pd.DataFrame(all_results).set_index("variant")
    out_dir = root / cfg["output"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "track2_results.csv")
    with open(out_dir / "track2_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return results_df.reset_index()


if __name__ == "__main__":
    run_track2()