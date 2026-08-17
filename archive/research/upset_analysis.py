"""Experiments 2 & 3: Major Upset Analysis and Upset Recovery Trajectories.

Systematically identifies high-confidence favorites (P(win) >= threshold) that suffer
unexpected defeats, and analyzes the rating shock and recovery dynamics of
Classic Elo vs. Fixed 5% vs. Adaptive Elo.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_matches, add_outcome_labels
from src.features.strength import UpdaterConfig, StrengthTracker
from src.evaluation.metrics import rps, multiclass_log_loss, expected_calibration_error


def run_upset_and_recovery_analysis(
    project_root: str | Path | None = None,
    config_path: str | Path = "config/default.yaml",
    thresholds: list[float] = (0.75, 0.80, 0.85, 0.90),
) -> dict:
    root = Path(project_root) if project_root else Path.cwd()
    cfg_file = root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)

    out_upset_dir = root / "results" / "upset_analysis"
    out_rec_dir = root / "results" / "upset_recovery"
    out_upset_dir.mkdir(parents=True, exist_ok=True)
    out_rec_dir.mkdir(parents=True, exist_ok=True)

    matches = load_matches(cfg, root)
    matches = add_outcome_labels(matches)

    # Initialize three parallel trackers
    tracker_classic = StrengthTracker(UpdaterConfig(mode="fixed_k", k=cfg["updater"]["k"]))
    tracker_fixed5 = StrengthTracker(UpdaterConfig(mode="bounded", k=cfg["updater"]["k"], cap=0.05))
    
    # Load tuned adaptive params if available, else default
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

    tracker_adaptive = StrengthTracker(UpdaterConfig(
        mode="adaptive",
        k=cfg["updater"]["k"],
        base_cap=best_p.get("base_cap", 0.01),
        max_cap=best_p.get("max_cap", 0.05),
        consistency_weight=best_p.get("consistency_weight", 3.0),
        surprise_weight=best_p.get("surprise_weight", 1.0),
        evidence_window=best_p.get("evidence_window", 7),
    ))

    records = []
    team_match_indices: dict[str, list[int]] = {}

    print(f"[upset] Running sequential simulation of {len(matches):,} matches across 3 update mechanisms...")

    for idx, row in enumerate(matches.itertuples(index=False)):
        home = row.home_team
        away = row.away_team
        neutral = bool(row.neutral)
        hg = int(row.home_goals)
        ag = int(row.away_goals)
        date = row.date

        # Record team match appearances
        team_match_indices.setdefault(home, []).append(idx)
        team_match_indices.setdefault(away, []).append(idx)

        # Pre-match states
        eh_classic = tracker_classic.rating(home)
        ea_classic = tracker_classic.rating(away)
        p_home_classic = tracker_classic.expected_score(home, away, neutral)

        eh_fixed5 = tracker_fixed5.rating(home)
        ea_fixed5 = tracker_fixed5.rating(away)
        p_home_fixed5 = tracker_fixed5.expected_score(home, away, neutral)

        eh_adaptive = tracker_adaptive.rating(home)
        ea_adaptive = tracker_adaptive.rating(away)
        p_home_adaptive = tracker_adaptive.expected_score(home, away, neutral)

        # Pre-match adaptive signals
        c_home = tracker_adaptive._consistency(home)
        s_home = tracker_adaptive._surprise(home)
        cap_pts_home = tracker_adaptive._adaptive_cap_points(home)

        c_away = tracker_adaptive._consistency(away)
        s_away = tracker_adaptive._surprise(away)
        cap_pts_away = tracker_adaptive._adaptive_cap_points(away)

        # Apply updates
        u_classic = tracker_classic.update(home, away, hg, ag, neutral)
        u_fixed5 = tracker_fixed5.update(home, away, hg, ag, neutral)
        u_adaptive = tracker_adaptive.update(home, away, hg, ag, neutral)

        # Check for favorite status based on pre-match expected probability
        # Scenario 1: Home is heavy favorite
        if p_home_classic >= 0.5:
            fav, und = home, away
            p_fav = p_home_classic
            fav_won = (hg > ag)
            fav_lost = (hg < ag)
            fav_d_classic = u_classic["delta_home"]
            fav_d_fixed5 = u_fixed5["delta_home"]
            fav_d_adaptive = u_adaptive["delta_home"]
            fav_elo_pre = eh_classic
            und_elo_pre = ea_classic
            fav_c = c_home
            fav_s = s_home
            fav_cap = cap_pts_home
            fav_elo_post_classic = tracker_classic.rating(home)
            fav_elo_post_fixed5 = tracker_fixed5.rating(home)
            fav_elo_post_adaptive = tracker_adaptive.rating(home)
        else:
            fav, und = away, home
            p_fav = 1.0 - p_home_classic
            fav_won = (ag > hg)
            fav_lost = (ag < hg)
            fav_d_classic = u_classic["delta_away"]
            fav_d_fixed5 = u_fixed5["delta_away"]
            fav_d_adaptive = u_adaptive["delta_away"]
            fav_elo_pre = ea_classic
            und_elo_pre = eh_classic
            fav_c = c_away
            fav_s = s_away
            fav_cap = cap_pts_away
            fav_elo_post_classic = tracker_classic.rating(away)
            fav_elo_post_fixed5 = tracker_fixed5.rating(away)
            fav_elo_post_adaptive = tracker_adaptive.rating(away)

        records.append({
            "match_idx": idx,
            "date": str(date)[:10],
            "tournament": row.tournament,
            "favorite": fav,
            "underdog": und,
            "score": f"{hg}-{ag}",
            "p_fav_win": p_fav,
            "fav_won": fav_won,
            "fav_lost": fav_lost,
            "elo_fav_pre": fav_elo_pre,
            "elo_und_pre": und_elo_pre,
            "elo_diff": fav_elo_pre - und_elo_pre,
            "delta_classic": fav_d_classic,
            "delta_fixed5": fav_d_fixed5,
            "delta_adaptive": fav_d_adaptive,
            "consistency": fav_c,
            "surprise": fav_s,
            "adaptive_cap": fav_cap,
            "post_elo_classic": fav_elo_post_classic,
            "post_elo_fixed5": fav_elo_post_fixed5,
            "post_elo_adaptive": fav_elo_post_adaptive,
        })

    df = pd.DataFrame(records)

    # ------------------------------------------------------------------ #
    # Experiment 2: Upset Summaries by Threshold
    # ------------------------------------------------------------------ #
    summary_by_thresh = []
    for th in thresholds:
        upsets = df[(df["p_fav_win"] >= th) & (df["fav_lost"])].copy()
        n_upsets = len(upsets)
        if n_upsets == 0:
            continue

        mean_drop_classic = float(upsets["delta_classic"].abs().mean())
        mean_drop_fixed5 = float(upsets["delta_fixed5"].abs().mean())
        mean_drop_adaptive = float(upsets["delta_adaptive"].abs().mean())

        med_drop_classic = float(upsets["delta_classic"].abs().median())
        med_drop_fixed5 = float(upsets["delta_fixed5"].abs().median())
        med_drop_adaptive = float(upsets["delta_adaptive"].abs().median())

        max_drop_classic = float(upsets["delta_classic"].abs().max())
        max_drop_fixed5 = float(upsets["delta_fixed5"].abs().max())
        max_drop_adaptive = float(upsets["delta_adaptive"].abs().max())

        pct_smaller_classic = float((upsets["delta_adaptive"].abs() < upsets["delta_classic"].abs()).mean() * 100.0)
        pct_smaller_fixed5 = float((upsets["delta_adaptive"].abs() < upsets["delta_fixed5"].abs()).mean() * 100.0)

        summary_by_thresh.append({
            "threshold": th,
            "n_upsets": n_upsets,
            "mean_drop_classic": round(mean_drop_classic, 2),
            "mean_drop_fixed5": round(mean_drop_fixed5, 2),
            "mean_drop_adaptive": round(mean_drop_adaptive, 2),
            "median_drop_classic": round(med_drop_classic, 2),
            "median_drop_fixed5": round(med_drop_fixed5, 2),
            "median_drop_adaptive": round(med_drop_adaptive, 2),
            "max_drop_classic": round(max_drop_classic, 2),
            "max_drop_fixed5": round(max_drop_fixed5, 2),
            "max_drop_adaptive": round(max_drop_adaptive, 2),
            "pct_adaptive_smaller_than_classic": round(pct_smaller_classic, 1),
            "pct_adaptive_smaller_than_fixed5": round(pct_smaller_fixed5, 1),
        })

    summary_df = pd.DataFrame(summary_by_thresh)
    summary_df.to_csv(out_upset_dir / "upset_summary_by_threshold.csv", index=False)

    upsets_80 = df[(df["p_fav_win"] >= 0.80) & (df["fav_lost"])].copy()
    upsets_80.to_csv(out_upset_dir / "major_upsets_threshold_80.csv", index=False)

    with open(out_upset_dir / "upset_summary.json", "w") as f:
        json.dump(summary_by_thresh, f, indent=2)

    print(f"\n=== Experiment 2: Major Upset Summary (P(Fav) >= 0.80: {len(upsets_80)} upsets) ===")
    print(f"Mean Rating Drop: Classic = {upsets_80['delta_classic'].abs().mean():.2f} pts | "
          f"Fixed 5% = {upsets_80['delta_fixed5'].abs().mean():.2f} pts | "
          f"Adaptive = {upsets_80['delta_adaptive'].abs().mean():.2f} pts")
    print(f"Adaptive smaller than Classic in {(upsets_80['delta_adaptive'].abs() < upsets_80['delta_classic'].abs()).mean()*100:.1f}% of major upsets.")

    # ------------------------------------------------------------------ #
    # Experiment 3: Upset Recovery Trajectories (k in {1, 3, 5})
    # ------------------------------------------------------------------ #
    recovery_trajectories = []
    post_upset_rps_classic = []
    post_upset_rps_fixed5 = []
    post_upset_rps_adaptive = []

    for _, up in upsets_80.iterrows():
        fav = up["favorite"]
        u_idx = int(up["match_idx"])
        all_fav_matches = team_match_indices.get(fav, [])
        subsequent = [m for m in all_fav_matches if m > u_idx]

        for step in (1, 3, 5):
            if len(subsequent) >= step:
                target_m_idx = subsequent[step - 1]
                t_row = matches.iloc[target_m_idx]
                
                # Check performance in subsequent match
                is_home = (t_row["home_team"] == fav)
                opp = t_row["away_team"] if is_home else t_row["home_team"]
                fav_won_sub = (t_row["home_goals"] > t_row["away_goals"]) if is_home else (t_row["away_goals"] > t_row["home_goals"])
                
                recovery_trajectories.append({
                    "upset_match_idx": u_idx,
                    "favorite": fav,
                    "upset_date": up["date"],
                    "recovery_step": step,
                    "subsequent_match_idx": target_m_idx,
                    "opponent": opp,
                    "fav_won_subsequent": fav_won_sub,
                })

    rec_df = pd.DataFrame(recovery_trajectories)
    rec_df.to_csv(out_rec_dir / "recovery_trajectories.csv", index=False)

    rec_metrics = {
        "n_upsets_evaluated": len(upsets_80),
        "n_trajectories_tracked": len(rec_df),
        "mean_recovery_steps_available": float(rec_df.groupby("upset_match_idx")["recovery_step"].max().mean()) if len(rec_df) else 0.0,
    }
    with open(out_rec_dir / "recovery_metrics.json", "w") as f:
        json.dump(rec_metrics, f, indent=2)

    return {
        "upset_summary": summary_by_thresh,
        "upsets_80_count": len(upsets_80),
        "recovery_count": len(rec_df),
    }


if __name__ == "__main__":
    run_upset_and_recovery_analysis()
