"""Reliability (calibration) diagrams for model comparison.

Reads the saved Track 1 / Track 2 results and produces reliability
diagrams comparing variants. Run after the experiments:

    python run_plots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data.loader import load_matches, add_outcome_labels
from src.data.split import rolling_origin_folds
from src.features.strength import UpdaterConfig, run_tracker_over_matches
from src.features.team_form import build_m0_feature_matrix
from src.models.predictor import OutcomePredictor
from src.evaluation.metrics import calibration_curve_data


def plot_reliability_track1(
    config_path: str | Path = "config/default.yaml",
    project_root: str | Path | None = None,
) -> Path:
    """Reliability diagrams for the Track 1 update-mechanism variants."""
    root = Path(project_root) if project_root else Path.cwd()
    with open(root / config_path) as f:
        cfg = yaml.safe_load(f)

    matches = add_outcome_labels(load_matches(cfg, root))
    y = matches["outcome"].to_numpy()
    folds = rolling_origin_folds(
        matches,
        n_folds=cfg["validation"]["n_folds"],
        test_fraction=cfg["validation"]["test_fractions"][0],
        min_train_matches=cfg["validation"]["min_train_matches"],
    )

    from src.experiments.track1 import VARIANTS

    model_cfg = cfg["model"]["gradient_boosting"]
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, overrides in VARIANTS.items():
        updater_cfg = UpdaterConfig(
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
        strength_feats, _ = run_tracker_over_matches(matches, updater_cfg)
        X = build_m0_feature_matrix(matches, strength_feats, cfg["features"]["form_windows"])

        preds = np.full((len(matches), 3), np.nan)
        for fold in folds:
            model = OutcomePredictor(
                n_estimators=model_cfg["n_estimators"],
                learning_rate=model_cfg["learning_rate"],
                max_depth=model_cfg["max_depth"],
                random_state=cfg["model"]["random_state"],
            )
            model.fit(X.iloc[fold.train_idx], y[fold.train_idx])
            preds[fold.test_idx] = model.predict_proba(X.iloc[fold.test_idx])

        mask = ~np.isnan(preds[:, 0])
        bins = calibration_curve_data(
            y[mask], preds[mask], n_bins=cfg["evaluation"]["calibration_bins"]
        )
        conf = [b["avg_confidence"] for b in bins]
        acc = [b["accuracy"] for b in bins]
        ax.plot(conf, acc, marker="o", label=name)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Track 1: reliability by update mechanism")
    ax.legend(fontsize=8)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)

    out_dir = root / cfg["output"]["figures_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "track1_reliability.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plots] saved {out}")
    return out


if __name__ == "__main__":
    plot_reliability_track1()