"""Track 3 experiment: FIFA-style match engine + World Cup Monte Carlo simulator.

Runs the full simulation pipeline end-to-end:

    1. Load or synthesize the FIFA player dataset.
    2. Build PlayerModel states for a target year.
    3. Select starting XIs and aggregate TeamRatings (SquadModel + ChemistryModel).
    4. Validate the match engine against real/synthetic scorelines.
    5. Run the World Cup Monte Carlo simulator (n_simulations) and save
       probability tables + a summary report.

With no Kaggle data present, the pipeline falls back to a realistic
synthetic FIFA player dataset (same pattern as Tracks 1 and 2).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.fifa_players import load_or_synthesize_fifa
from src.data.worldcup2026 import (
    load_worldcup2026_players,
    load_worldcup2026_teams,
    load_worldcup2026_matches,
)
from src.simulation.player_model import PlayerModel
from src.simulation.squad_model import SquadModel, ManagerStyle, TeamRating
from src.simulation.chemistry import ChemistryModel, ChemistryConfig
from src.simulation.match_engine import MatchEngine, MatchEngineConfig
from src.simulation.tournament import WorldCupSimulator


def _validate_match_engine(
    engine: MatchEngine,
    ratings: dict[str, TeamRating],
    teams: list[str],
    n_matches: int = 2000,
    seed: int = 42,
) -> dict:
    """Validate the match engine against expected football statistics.

    Returns summary stats: mean goals, draw rate, home advantage effect,
    and the scoreline distribution (top scorelines).
    """
    rng = np.random.default_rng(seed)
    scorelines = []
    for _ in range(n_matches):
        a = teams[rng.integers(len(teams))]
        b = teams[rng.integers(len(teams))]
        if a == b:
            continue
        # ~15% of matches have a home advantage (not neutral).
        neutral = rng.random() < 0.85
        gh, ga = engine.sample_scoreline(ratings[a], ratings[b], neutral=neutral)
        scorelines.append((gh, ga, neutral))

    df = pd.DataFrame(scorelines, columns=["hg", "ag", "neutral"])
    home = df[~df["neutral"]]
    away = df[df["neutral"]]

    total_goals = df["hg"].sum() + df["ag"].sum()
    mean_goals = total_goals / len(df)
    draw_rate = ((df["hg"] == df["ag"]).mean())
    home_win_rate = (df["hg"] > df["ag"]).mean()
    away_win_rate = (df["hg"] < df["ag"]).mean()

    # Home advantage: home teams score more than away teams in non-neutral.
    home_adv_goals = float(
        home["hg"].mean() - home["ag"].mean()
    ) if len(home) else 0.0

    # Top scorelines.
    counts = (
        df.groupby(["hg", "ag"]).size().sort_values(ascending=False).head(6)
    )
    top_scorelines = [
        {"scoreline": f"{h}-{a}", "count": int(c)}
        for (h, a), c in counts.items()
    ]

    return {
        "n_matches": len(df),
        "mean_goals_per_match": float(mean_goals),
        "draw_rate": float(draw_rate),
        "home_win_rate": float(home_win_rate),
        "away_win_rate": float(away_win_rate),
        "home_advantage_goals": float(home_adv_goals),
        "real_mean_goals_per_match": 2.69,  # World Cup historical average
        "real_draw_rate": 0.23,             # World Cup historical ≈0.23
        "top_scorelines": top_scorelines,
    }


def _validate_against_real_xg(
    engine: MatchEngine,
    ratings: dict[str, TeamRating],
    real_matches: pd.DataFrame,
) -> dict:
    """Compare model xG to real recorded xG for the 2026 World Cup."""
    df = real_matches.dropna(subset=["home_xg", "away_xg"])
    df = df[df["home_team"].isin(ratings) & df["away_team"].isin(ratings)]
    if df.empty:
        return {"n_matches": 0, "note": "no overlapping teams"}

    model_xg, real_xg = [], []
    for r in df.itertuples(index=False):
        lh, la = engine.expected_goals(
            ratings[r.home_team], ratings[r.away_team], neutral=True
        )
        model_xg.extend([lh, la])
        real_xg.extend([float(r.home_xg), float(r.away_xg)])

    model_arr = np.array(model_xg)
    real_arr = np.array(real_xg)
    return {
        "n_matches": int(len(df)),
        "model_mean_xg": float(model_arr.mean()),
        "real_mean_xg": float(real_arr.mean()),
        "xg_mae": float(np.mean(np.abs(model_arr - real_arr))),
        "xg_correlation": float(np.corrcoef(model_arr, real_arr)[0, 1]) if len(model_arr) > 1 else 0.0,
    }


def run_track3(
    config_path: str | Path = "config/default.yaml",
    project_root: str | Path | None = None,
    n_simulations: int | None = None,
) -> dict:
    """Run the full Track 3 simulation pipeline.

    Returns a dict with ratings breakdown, validation stats, and tournament
    results (also written to results/track3_results.csv + .json).
    """
    root = Path(project_root) if project_root else Path.cwd()
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    track3_cfg = cfg.get("track3", {})
    n_sim = n_simulations or track3_cfg.get("n_simulations", 1000)
    fifa_dir = root / track3_cfg.get("fifa_data_dir", "data/raw/fifa")
    target_year = track3_cfg.get("target_year", 2022)

    # ------------------------------------------------------------------ #
    # 1. FIFA player data
    # ------------------------------------------------------------------ #
    wc2026_squads = fifa_dir / "squads_and_players.csv"
    team_priors: dict[str, dict] = {}
    real_matches = None
    if wc2026_squads.exists():
        print("[track3] real World Cup 2026 dataset found -- using real data.")
        players = load_worldcup2026_players(fifa_dir, seed=cfg["data"]["synthetic"]["seed"])
        target_year = 2026
        teams_df = load_worldcup2026_teams(fifa_dir)
        team_priors = {
            r.team_name: {
                "elo": float(r.elo_rating),
                "fifa_rank": float(r.fifa_ranking_pre_tournament),
                "manager": r.manager_name,
                "group": r.group_letter,
            }
            for r in teams_df.itertuples(index=False)
        }
        real_matches = load_worldcup2026_matches(fifa_dir)
        print(f"[track3] real player table: {len(players):,} players, "
              f"{players['nationality'].nunique()} nations")
    else:
        players = load_or_synthesize_fifa(
            fifa_dir, years=list(range(2018, target_year + 1)), seed=cfg["data"]["synthetic"]["seed"]
        )
        print(f"[track3] FIFA player table: {len(players):,} rows "
              f"({players['year'].nunique()} years, "
              f"{players['nationality'].nunique()} nations)")

    # ------------------------------------------------------------------ #
    # 2. Player states
    # ------------------------------------------------------------------ #
    pm_cfg = track3_cfg.get("player_model", {})
    player_model = PlayerModel(
        form_volatility=pm_cfg.get("form_volatility", 0.02),
        availability_floor=pm_cfg.get("availability_floor", 0.5),
        seed=cfg["data"]["synthetic"]["seed"],
    )
    states = player_model.build_states(players, year=target_year)
    nations = sorted({s.nationality for s in states.values()})
    teams = track3_cfg.get("teams") or nations[:32]
    print(f"[track3] {len(states):,} player states; {len(nations)} nations")

    # ------------------------------------------------------------------ #
    # 3. Squad + chemistry models
    # ------------------------------------------------------------------ #
    sq_cfg = track3_cfg.get("squad_model", {})
    # Manager styles: learnable team-level tactical effect (proxy).
    manager_styles = {}
    field_mean = (
        float(np.mean([p["elo"] for p in team_priors.values()]))
        if team_priors else 0.0
    )
    for t in teams:
        if team_priors and t in team_priors:
            bias = float(np.clip((team_priors[t]["elo"] - field_mean) / 1000.0, -0.1, 0.1))
        else:
            bias = sq_cfg.get("manager_attack_bias", 0.0)
        manager_styles[t] = ManagerStyle(
            attack_bias=bias,
            defence_bias=sq_cfg.get("manager_defence_bias", 0.0),
            aggression=sq_cfg.get("manager_aggression", 0.0),
        )
    squad_model = SquadModel(
        formation=sq_cfg.get("formation", "4-3-3"),
        manager_styles=manager_styles,
    )
    chem_cfg = track3_cfg.get("chemistry", {})
    chemistry_model = ChemistryModel(
        ChemistryConfig(
            club_weight=chem_cfg.get("club_weight", 0.5),
            minutes_weight=chem_cfg.get("minutes_weight", 0.3),
            position_weight=chem_cfg.get("position_weight", 0.2),
        )
    )
    me_cfg = track3_cfg.get("match_engine", {})
    match_engine = MatchEngine(
        MatchEngineConfig(
            home_advantage=me_cfg.get("home_advantage", 0.30),
            chemistry_attack_weight=me_cfg.get("chemistry_attack_weight", 0.15),
            chemistry_defence_weight=me_cfg.get("chemistry_defence_weight", 0.10),
            form_noise=me_cfg.get("form_noise", 0.05),
            rating_scale=me_cfg.get("rating_scale", 0.02),
            baseline_goals=me_cfg.get("baseline_goals", 0.30),
        ),
        seed=cfg["data"]["synthetic"]["seed"],
    )

    # ------------------------------------------------------------------ #
    # 4. Build team ratings + validate the match engine
    # ------------------------------------------------------------------ #
    sim = WorldCupSimulator(
        player_model=player_model,
        squad_model=squad_model,
        chemistry_model=chemistry_model,
        match_engine=match_engine,
    )
    ratings = sim.build_ratings(states, teams)
    validation = _validate_match_engine(match_engine, ratings, teams[:16])

    # ------------------------------------------------------------------ #
    # 5. Monte Carlo tournament simulation
    # ------------------------------------------------------------------ #
    print(f"[track3] running {n_sim:,} World Cup simulations...")
    result = sim.simulate_tournament(
        states=states,
        teams=teams[:32],
        n_simulations=n_sim,
        seed=cfg["data"]["synthetic"]["seed"],
    )
    frame = result.to_frame()

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    out_dir = root / cfg["output"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "track3_results.csv", index=False)

    payload = {
        "n_simulations": result.n_simulations,
        "target_year": target_year,
        "teams": teams[:32],
        "most_likely_winner": result.most_likely_winner,
        "most_likely_final": result.most_likely_final,
        "top_10_winner_prob": frame.head(10).to_dict("records"),
        "validation": validation,
    }
    with open(out_dir / "track3_results.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    # Console summary.
    print(f"\n=== Track 3 summary ({n_sim:,} simulations) ===")
    print(f"Most likely winner : {result.most_likely_winner}")
    print(f"Most likely final  : {result.most_likely_final}")
    print("\nTop 10 by P(win World Cup):")
    for _, row in frame.head(10).iterrows():
        print(f"  {row['team']:<25} {row['p_winner']*100:5.1f}%  "
              f"final {row['p_finalist']*100:4.1f}%  "
              f"semi {row['p_semi']*100:4.1f}%  "
              f"exit-grp {row['p_group_exit']*100:4.1f}%")
    print("\nMatch engine validation vs real World Cup averages:")
    print(f"  mean goals: {validation['mean_goals_per_match']:.2f} "
          f"(real ~{validation['real_mean_goals_per_match']})")
    print(f"  draw rate : {validation['draw_rate']:.2%} "
          f"(real ~{validation['real_draw_rate']:.0%})")
    print(f"  top scorelines: {validation['top_scorelines'][:3]}")

    return payload


if __name__ == "__main__":
    run_track3()