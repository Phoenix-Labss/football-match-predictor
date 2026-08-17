"""Experiment 5: Dynamic In-Tournament Elo Simulation and Backtesting.

Backtests 3 distinct tournament operational modes across 3 major historical tournaments:
  - 2018 FIFA World Cup (Russia)
  - UEFA Euro 2020
  - 2022 FIFA World Cup (Qatar)

Modes:
  1. STATIC: Team ratings remain static across rounds.
  2. CLASSIC_DYNAMIC: Elo updates after each real tournament match (fixed K=24).
  3. ADAPTIVE_DYNAMIC: Adaptive speed-limit Elo updates after each tournament match.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_matches
from src.features.strength import UpdaterConfig, StrengthTracker
from src.simulation.match_engine import MatchEngine, MatchEngineConfig
from src.simulation.squad_model import TeamRating
from src.evaluation.metrics import multiclass_log_loss, multiclass_brier, accuracy


def _evaluate_tournament_dynamics(
    matches_df: pd.DataFrame,
    tournament_name: str,
    date_start: str,
    date_end: str,
    actual_winner: str,
) -> dict:
    """Run static vs classic vs adaptive dynamic evaluation over a real tournament."""
    # Filter tournament matches chronologically
    t_matches = matches_df[
        (matches_df["date"] >= date_start) & 
        (matches_df["date"] <= date_end) &
        (matches_df["tournament"].str.contains(tournament_name, case=False, na=False))
    ].sort_values("date").reset_index(drop=True)

    if t_matches.empty:
        # Fallback filter without strict tournament name if dates are exact
        t_matches = matches_df[
            (matches_df["date"] >= date_start) & 
            (matches_df["date"] <= date_end)
        ].sort_values("date").reset_index(drop=True)

    # Initialize trackers using historical matches prior to date_start
    prior_matches = matches_df[matches_df["date"] < date_start].sort_values("date").reset_index(drop=True)

    tracker_static = StrengthTracker(UpdaterConfig(mode="fixed_k", k=24.0))
    tracker_classic = StrengthTracker(UpdaterConfig(mode="fixed_k", k=24.0))
    tracker_adaptive = StrengthTracker(UpdaterConfig(mode="adaptive", k=24.0, base_cap=0.01, max_cap=0.05))

    # Warm up trackers on prior historical fixtures
    for r in prior_matches.itertuples(index=False):
        hg, ag, n = int(r.home_goals), int(r.away_goals), bool(r.neutral)
        tracker_static.update(r.home_team, r.away_team, hg, ag, n)
        tracker_classic.update(r.home_team, r.away_team, hg, ag, n)
        tracker_adaptive.update(r.home_team, r.away_team, hg, ag, n)

    # Freeze pre-tournament ratings for static baseline
    pre_ratings = {t: tracker_static.rating(t) for t in tracker_static.ratings}

    # Tracking lists
    modes = ["STATIC", "CLASSIC_DYNAMIC", "ADAPTIVE_DYNAMIC"]
    preds = {m: [] for m in modes}
    targets = []
    deltas = {m: [] for m in modes}

    match_log = []

    for r in t_matches.itertuples(index=False):
        home, away = r.home_team, r.away_team
        hg, ag, n = int(r.home_goals), int(r.away_goals), bool(r.neutral)
        y = 0 if hg > ag else (1 if hg == ag else 2)
        targets.append(y)

        # 1. Static Mode Prediction
        eh_s = pre_ratings.get(home, 1500.0)
        ea_s = pre_ratings.get(away, 1500.0)
        p_home_s = 1.0 / (1.0 + 10.0 ** (-(eh_s - ea_s) / 400.0))
        p_draw_s = 0.25
        p_win_s = p_home_s * 0.75
        p_loss_s = (1.0 - p_home_s) * 0.75
        preds["STATIC"].append([p_win_s, p_draw_s, p_loss_s])
        deltas["STATIC"].append(0.0)

        # 2. Classic Dynamic Prediction
        eh_c = tracker_classic.rating(home)
        ea_c = tracker_classic.rating(away)
        p_home_c = tracker_classic.expected_score(home, away, n)
        p_draw_c = 0.25
        p_win_c = p_home_c * 0.75
        p_loss_c = (1.0 - p_home_c) * 0.75
        preds["CLASSIC_DYNAMIC"].append([p_win_c, p_draw_c, p_loss_c])
        u_c = tracker_classic.update(home, away, hg, ag, n)
        deltas["CLASSIC_DYNAMIC"].append(abs(u_c["delta_home"]))

        # 3. Adaptive Dynamic Prediction
        eh_a = tracker_adaptive.rating(home)
        ea_a = tracker_adaptive.rating(away)
        p_home_a = tracker_adaptive.expected_score(home, away, n)
        p_draw_a = 0.25
        p_win_a = p_home_a * 0.75
        p_loss_a = (1.0 - p_home_a) * 0.75
        preds["ADAPTIVE_DYNAMIC"].append([p_win_a, p_draw_a, p_loss_a])
        u_a = tracker_adaptive.update(home, away, hg, ag, n)
        deltas["ADAPTIVE_DYNAMIC"].append(abs(u_a["delta_home"]))

        match_log.append({
            "tournament": tournament_name,
            "date": str(r.date)[:10],
            "match": f"{home} vs {away}",
            "score": f"{hg}-{ag}",
            "delta_classic": round(abs(u_c["delta_home"]), 2),
            "delta_adaptive": round(abs(u_a["delta_home"]), 2),
        })

    y_arr = np.array(targets)
    metrics_summary = {}

    for m in modes:
        p_arr = np.array(preds[m])
        p_arr = p_arr / p_arr.sum(axis=1, keepdims=True)
        acc = accuracy(y_arr, p_arr)
        ll = multiclass_log_loss(y_arr, p_arr)
        bs = multiclass_brier(y_arr, p_arr)
        mean_d = float(np.mean(deltas[m]))
        large_swings = int(np.sum(np.array(deltas[m]) >= 15.0))

        metrics_summary[m] = {
            "n_matches": len(t_matches),
            "accuracy": round(acc, 4),
            "log_loss": round(ll, 4),
            "brier_score": round(bs, 4),
            "mean_rating_movement": round(mean_d, 2),
            "major_rating_swings_over_15pts": large_swings,
        }

    return {
        "tournament": tournament_name,
        "actual_winner": actual_winner,
        "n_matches": len(t_matches),
        "metrics": metrics_summary,
        "match_log": match_log,
    }


def run_dynamic_tournament_experiments(project_root: str | Path | None = None) -> dict:
    root = Path(project_root) if project_root else Path.cwd()
    out_dir = root / "results" / "tournament_dynamic"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(root / "config" / "default.yaml") as f:
        cfg = yaml.safe_load(f)

    matches_df = load_matches(cfg, root)

    tournaments = [
        {
            "name": "2018 FIFA World Cup",
            "start": "2018-06-14",
            "end": "2018-07-15",
            "winner": "France",
        },
        {
            "name": "UEFA Euro 2020",
            "start": "2021-06-11",
            "end": "2021-07-11",
            "winner": "Italy",
        },
        {
            "name": "2022 FIFA World Cup",
            "start": "2022-11-20",
            "end": "2022-12-18",
            "winner": "Argentina",
        },
    ]

    all_results = []
    all_match_logs = []

    print("\n=== Experiment 5: Dynamic Tournament Backtest (2018, Euro 2020, 2022) ===")

    for t_spec in tournaments:
        t_res = _evaluate_tournament_dynamics(
            matches_df,
            tournament_name=t_spec["name"],
            date_start=t_spec["start"],
            date_end=t_spec["end"],
            actual_winner=t_spec["winner"],
        )
        all_results.append(t_res)
        all_match_logs.extend(t_res["match_log"])

        print(f"\n--- {t_spec['name']} ({t_res['n_matches']} matches, Winner: {t_spec['winner']}) ---")
        for m, met in t_res["metrics"].items():
            print(f"  {m:<18}: Acc={met['accuracy']*100:4.1f}% | LogLoss={met['log_loss']:.4f} | "
                  f"Brier={met['brier_score']:.4f} | Mean Movement={met['mean_rating_movement']:4.1f} pts | "
                  f"Swings(>=15pts)={met['major_rating_swings_over_15pts']}")

    # Save summary artifacts
    summary_rows = []
    for t_res in all_results:
        t_name = t_res["tournament"]
        for m, met in t_res["metrics"].items():
            summary_rows.append({
                "tournament": t_name,
                "mode": m,
                **met
            })

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(out_dir / "tournament_dynamic_results.csv", index=False)

    with open(out_dir / "tournament_dynamic_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    pd.DataFrame(all_match_logs).to_csv(out_dir / "tournament_match_deltas_log.csv", index=False)

    return {
        "results": all_results,
        "summary_table": sum_df,
    }


if __name__ == "__main__":
    run_dynamic_tournament_experiments()
