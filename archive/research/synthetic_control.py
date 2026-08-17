"""Experiment 4: Synthetic Controlled Test of the Evidence-Bounded Update Hypothesis.

Constructs 4 controlled match sequences for Team A (Elo = 2000) vs Team B (Elo = 1600):
  Scenario A: 10 consistent expected wins (Baseline control).
  Scenario B: 1 isolated unexpected loss at t=5, surrounded by wins.
  Scenario C: 3 consecutive unexpected losses at t=3, 4, 5.
  Scenario D: 5 consecutive unexpected losses (sustained structural decline).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.features.strength import UpdaterConfig, StrengthTracker


def run_synthetic_control(project_root: str | Path | None = None) -> dict:
    root = Path(project_root) if project_root else Path.cwd()
    out_dir = root / "results" / "synthetic_control"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define scenarios: list of (home_goals, away_goals) where Team A is home (neutral venue)
    # 2-0 is a normal expected win, 0-1 is an unexpected loss
    scenarios = {
        "Scenario_A_Baseline": [(2, 0)] * 10,
        "Scenario_B_Isolated_Shock": [(2, 0)] * 4 + [(0, 1)] + [(2, 0)] * 5,
        "Scenario_C_Transient_Slump": [(2, 0)] * 2 + [(0, 1)] * 3 + [(2, 0)] * 5,
        "Scenario_D_Structural_Decline": [(0, 1)] * 5 + [(2, 0)] * 5,
    }

    all_rows = []
    summary = {}

    for scen_name, match_sequence in scenarios.items():
        # Instantiate fresh trackers for each scenario
        t_classic = StrengthTracker(UpdaterConfig(mode="fixed_k", k=24.0, initial_rating=2000.0, home_advantage=0.0))
        t_fixed5 = StrengthTracker(UpdaterConfig(mode="bounded", k=24.0, cap=0.05, initial_rating=2000.0, home_advantage=0.0))
        t_adaptive = StrengthTracker(UpdaterConfig(
            mode="adaptive", k=24.0, base_cap=0.01, max_cap=0.05,
            consistency_weight=3.0, surprise_weight=1.0, evidence_window=7,
            initial_rating=2000.0, home_advantage=0.0
        ))

        # Set opponent rating to 1600.0
        t_classic.ratings["Team_B"] = 1600.0
        t_fixed5.ratings["Team_B"] = 1600.0
        t_adaptive.ratings["Team_B"] = 1600.0

        # Record initial t=0 state
        all_rows.append({
            "scenario": scen_name,
            "match_step": 0,
            "score": "Initial",
            "p_win_classic": t_classic.expected_score("Team_A", "Team_B", neutral=True),
            "p_win_fixed5": t_fixed5.expected_score("Team_A", "Team_B", neutral=True),
            "p_win_adaptive": t_adaptive.expected_score("Team_A", "Team_B", neutral=True),
            "rating_classic": 2000.0,
            "rating_fixed5": 2000.0,
            "rating_adaptive": 2000.0,
            "delta_classic": 0.0,
            "delta_fixed5": 0.0,
            "delta_adaptive": 0.0,
            "consistency": 0.0,
            "surprise": 0.0,
            "adaptive_cap": 4.0,
        })

        for step, (hg, ag) in enumerate(match_sequence, start=1):
            # Pre-match metrics
            p_win_c = t_classic.expected_score("Team_A", "Team_B", neutral=True)
            p_win_f = t_fixed5.expected_score("Team_A", "Team_B", neutral=True)
            p_win_a = t_adaptive.expected_score("Team_A", "Team_B", neutral=True)

            cons = t_adaptive._consistency("Team_A")
            surp = t_adaptive._surprise("Team_A")
            cap_val = t_adaptive._adaptive_cap_points("Team_A")

            # Apply updates
            u_c = t_classic.update("Team_A", "Team_B", hg, ag, neutral=True)
            u_f = t_fixed5.update("Team_A", "Team_B", hg, ag, neutral=True)
            u_a = t_adaptive.update("Team_A", "Team_B", hg, ag, neutral=True)

            all_rows.append({
                "scenario": scen_name,
                "match_step": step,
                "score": f"{hg}-{ag}",
                "p_win_classic": p_win_c,
                "p_win_fixed5": p_win_f,
                "p_win_adaptive": p_win_a,
                "rating_classic": t_classic.rating("Team_A"),
                "rating_fixed5": t_fixed5.rating("Team_A"),
                "rating_adaptive": t_adaptive.rating("Team_A"),
                "delta_classic": u_c["delta_home"],
                "delta_fixed5": u_f["delta_home"],
                "delta_adaptive": u_a["delta_home"],
                "consistency": cons,
                "surprise": surp,
                "adaptive_cap": cap_val,
            })

        # Summary for scenario
        scen_df = pd.DataFrame([r for r in all_rows if r["scenario"] == scen_name])
        summary[scen_name] = {
            "min_rating_classic": float(scen_df["rating_classic"].min()),
            "min_rating_fixed5": float(scen_df["rating_fixed5"].min()),
            "min_rating_adaptive": float(scen_df["rating_adaptive"].min()),
            "max_drop_classic": float(2000.0 - scen_df["rating_classic"].min()),
            "max_drop_fixed5": float(2000.0 - scen_df["rating_fixed5"].min()),
            "max_drop_adaptive": float(2000.0 - scen_df["rating_adaptive"].min()),
            "final_rating_classic": float(scen_df["rating_classic"].iloc[-1]),
            "final_rating_fixed5": float(scen_df["rating_fixed5"].iloc[-1]),
            "final_rating_adaptive": float(scen_df["rating_adaptive"].iloc[-1]),
        }

    res_df = pd.DataFrame(all_rows)
    res_df.to_csv(out_dir / "synthetic_scenarios_trajectories.csv", index=False)

    with open(out_dir / "synthetic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Experiment 4: Synthetic Control Summary ===")
    for sc, v in summary.items():
        print(f"{sc:<30} Max Drop: Classic = {v['max_drop_classic']:.1f} pts | "
              f"Fixed 5% = {v['max_drop_fixed5']:.1f} pts | Adaptive = {v['max_drop_adaptive']:.1f} pts")

    return {
        "summary": summary,
        "trajectories": res_df,
    }


if __name__ == "__main__":
    run_synthetic_control()
