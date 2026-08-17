"""Experiment 6: Statistical Significance Testing for Probabilistic Forecasts.

Implements the Diebold-Mariano test with Harvey-Leybourne-Newbold (HLN)
finite-sample correction and Newey-West HAC variance estimator on per-match
Ranked Probability Score (RPS) and Log Loss differentials, alongside
1,000-sample bootstrap confidence intervals.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import yaml

from src.data.loader import load_matches, add_outcome_labels
from src.data.split import rolling_origin_folds, assert_no_temporal_leakage
from src.features.strength import UpdaterConfig, run_tracker_over_matches
from src.features.team_form import build_m0_feature_matrix
from src.models.predictor import OutcomePredictor
from src.evaluation.metrics import rps, multiclass_log_loss


def _per_match_rps(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Compute per-match normalized RPS (shape: (n,))."""
    probs = np.clip(probs, 1e-15, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    Y = np.eye(3)[y_true.astype(int)]
    cdf_p = np.cumsum(probs, axis=1)[:, :-1]
    cdf_y = np.cumsum(Y, axis=1)[:, :-1]
    # Normalized by (K-1) = 2
    return np.sum((cdf_p - cdf_y) ** 2, axis=1) / 2.0


def _per_match_log_loss(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Compute per-match Log Loss (shape: (n,))."""
    probs = np.clip(probs, 1e-15, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    idx = np.arange(len(y_true))
    return -np.log(probs[idx, y_true.astype(int)])


def diebold_mariano_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    h: int = 1,
) -> dict:
    """Diebold-Mariano test on loss differential d_t = loss_a - loss_b.

    Negative d_bar means Model A has lower loss (better).
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    T = len(d)
    d_bar = float(np.mean(d))

    # Sample autocovariances for Newey-West HAC estimator
    gamma_0 = float(np.var(d, ddof=0))
    gamma_sum = 0.0
    for lag in range(1, h):
        weight = 1.0 - (lag / h)
        cov = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        gamma_sum += 2.0 * weight * cov

    lr_var = max(1e-12, gamma_0 + gamma_sum)
    dm_stat = d_bar / np.sqrt(lr_var / T)

    # Harvey-Leybourne-Newbold (HLN) small-sample modification
    hln_correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm_stat * hln_correction

    # Two-tailed p-value from Student's t with T-1 degrees of freedom
    p_val = 2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=T - 1))

    return {
        "mean_diff": float(d_bar),
        "dm_stat": float(dm_stat),
        "dm_hln": float(dm_hln),
        "p_value": float(p_val),
        "is_significant_05": bool(p_val < 0.05),
        "is_significant_01": bool(p_val < 0.01),
    }


def paired_bootstrap_ci(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Paired bootstrap confidence interval for mean(loss_a - loss_b)."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    T = len(d)
    rng = np.random.default_rng(seed)

    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, T, size=T)
        boot_diffs[i] = np.mean(d[idx])

    ci_low, ci_high = np.percentile(boot_diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean_diff": float(np.mean(d)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def run_statistical_significance_tests(
    project_root: str | Path | None = None,
    config_path: str | Path = "config/default.yaml",
) -> dict:
    root = Path(project_root) if project_root else Path.cwd()
    cfg_file = root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)

    out_dir = root / "results" / "statistical_tests"
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

    form_windows = cfg["features"]["form_windows"]
    model_cfg = cfg["model"]["gradient_boosting"]

    # Load tuned adaptive parameters if available
    best_params_path = root / "results" / "adaptive_tuning" / "best_hyperparameters.json"
    if best_params_path.exists():
        with open(best_params_path) as f:
            best_p = json.load(f).get("best_params", {})
    else:
        best_p = {
            "base_cap": cfg["updater"]["adaptive"]["base_cap"],
            "max_cap": cfg["updater"].get("max_cap", 0.05),
            "consistency_weight": cfg["updater"]["adaptive"]["consistency_weight"],
            "surprise_weight": cfg["updater"]["adaptive"]["surprise_weight"],
            "evidence_window": cfg["updater"]["adaptive"]["evidence_window"],
        }

    updaters = {
        "Classic_Elo": UpdaterConfig(mode="fixed_k", k=cfg["updater"]["k"]),
        "Fixed_5pct": UpdaterConfig(mode="bounded", k=cfg["updater"]["k"], cap=0.05),
        "Adaptive_Elo": UpdaterConfig(
            mode="adaptive",
            k=cfg["updater"]["k"],
            base_cap=best_p.get("base_cap", 0.01),
            max_cap=best_p.get("max_cap", 0.05),
            consistency_weight=best_p.get("consistency_weight", 3.0),
            surprise_weight=best_p.get("surprise_weight", 1.0),
            evidence_window=best_p.get("evidence_window", 7),
        ),
    }

    test_predictions = {}

    print("[significance] Generating out-of-sample test predictions for statistical comparison...")

    for name, u_cfg in updaters.items():
        s_feats, _ = run_tracker_over_matches(matches, u_cfg)
        X = build_m0_feature_matrix(matches, s_feats, form_windows)

        preds = np.full((len(matches), 3), np.nan)
        for fold in folds:
            clf = OutcomePredictor(
                n_estimators=model_cfg["n_estimators"],
                learning_rate=model_cfg["learning_rate"],
                max_depth=model_cfg["max_depth"],
                random_state=cfg["model"]["random_state"],
            )
            clf.fit(X.iloc[fold.train_idx], y[fold.train_idx])
            preds[fold.test_idx] = clf.predict_proba(X.iloc[fold.test_idx])

        test_predictions[name] = preds

    mask = ~np.isnan(test_predictions["Classic_Elo"][:, 0])
    y_test = y[mask]

    rps_losses = {m: _per_match_rps(y_test, test_predictions[m][mask]) for m in updaters}
    log_losses = {m: _per_match_log_loss(y_test, test_predictions[m][mask]) for m in updaters}

    comparisons = [
        ("Classic_Elo", "Fixed_5pct"),
        ("Classic_Elo", "Adaptive_Elo"),
        ("Fixed_5pct", "Adaptive_Elo"),
    ]

    dm_results = []
    boot_results = []

    print("\n=== Experiment 6: Diebold-Mariano & Bootstrap Significance Results ===")

    for model_a, model_b in comparisons:
        # RPS Test
        dm_rps = diebold_mariano_test(rps_losses[model_a], rps_losses[model_b])
        boot_rps = paired_bootstrap_ci(rps_losses[model_a], rps_losses[model_b])

        # Log Loss Test
        dm_ll = diebold_mariano_test(log_losses[model_a], log_losses[model_b])
        boot_ll = paired_bootstrap_ci(log_losses[model_a], log_losses[model_b])

        dm_results.append({
            "comparison": f"{model_a} vs {model_b}",
            "metric": "Ranked Probability Score (Norm. RPS)",
            "mean_diff_A_minus_B": round(dm_rps["mean_diff"], 6),
            "dm_stat": round(dm_rps["dm_hln"], 4),
            "p_value": round(dm_rps["p_value"], 5),
            "statistically_significant_p05": dm_rps["is_significant_05"],
            "bootstrap_ci_95": f"[{boot_rps['ci_low']:.6f}, {boot_rps['ci_high']:.6f}]",
        })

        dm_results.append({
            "comparison": f"{model_a} vs {model_b}",
            "metric": "Multiclass Log Loss",
            "mean_diff_A_minus_B": round(dm_ll["mean_diff"], 6),
            "dm_stat": round(dm_ll["dm_hln"], 4),
            "p_value": round(dm_ll["p_value"], 5),
            "statistically_significant_p05": dm_ll["is_significant_05"],
            "bootstrap_ci_95": f"[{boot_ll['ci_low']:.6f}, {boot_ll['ci_high']:.6f}]",
        })

        print(f"[{model_a} vs {model_b}]")
        print(f"  RPS Diff: {dm_rps['mean_diff']:+.6f} | DM-HLN: {dm_rps['dm_hln']:+.3f} | p-val: {dm_rps['p_value']:.4f} | 95% CI: [{boot_rps['ci_low']:.6f}, {boot_rps['ci_high']:.6f}]")
        print(f"  LL  Diff: {dm_ll['mean_diff']:+.6f} | DM-HLN: {dm_ll['dm_hln']:+.3f} | p-val: {dm_ll['p_value']:.4f} | 95% CI: [{boot_ll['ci_low']:.6f}, {boot_ll['ci_high']:.6f}]")

    dm_df = pd.DataFrame(dm_results)
    dm_df.to_csv(out_dir / "diebold_mariano_results.csv", index=False)

    with open(out_dir / "bootstrap_ci_results.json", "w") as f:
        json.dump(dm_results, f, indent=2)

    return {
        "results": dm_results,
        "summary_table": dm_df,
    }


if __name__ == "__main__":
    run_statistical_significance_tests()
