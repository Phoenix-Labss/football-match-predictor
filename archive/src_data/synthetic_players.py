"""Synthetic player-match data generator for Track 2 development.

Produces one row per player per match with the generic schema expected
by src.features.player.build_player_team_features:

    date, team, player, position, rating, minutes, goals, assists, available

Player ratings correlate with team strength so the player-aware features
carry genuine signal. Falls back gracefully until the real European
Soccer Database is available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

POSITIONS = ["GK"] + ["CB", "LB", "RB"] * 2 + ["CM", "DM", "AM", "LM", "RM"] + ["ST", "CF", "LW", "RW"]


def generate_synthetic_player_matches(
    matches: pd.DataFrame,
    seed: int = 42,
    squad_size: int = 18,
) -> pd.DataFrame:
    """Attach a synthetic squad and ratings to each match in `matches`."""
    rng = np.random.default_rng(seed)

    # Persistent player pool per team.
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    squads: dict[str, list[str]] = {}
    player_base: dict[str, float] = {}
    team_quality: dict[str, float] = {}

    for t in teams:
        squad = [f"{t}_P{i:02d}" for i in range(squad_size)]
        squads[t] = squad
        tq = float(rng.normal(6.5, 0.6))
        team_quality[t] = tq
        for p in squad:
            player_base[p] = float(np.clip(tq + rng.normal(0, 0.5), 3.0, 10.0))

    rows = []
    for m in matches.itertuples(index=False):
        for team in (m.home_team, m.away_team):
            # Slight team-quality drift over time.
            team_quality[team] = float(
                np.clip(team_quality[team] + rng.normal(0, 0.01), 4.0, 9.0)
            )
            starters = rng.choice(squads[team], size=11, replace=False)
            for p in starters:
                available = bool(rng.random() > 0.06)  # ~6% unavailable
                # Rating: base ability + form noise + team-level shift.
                rating = float(
                    np.clip(
                        player_base[p]
                        + (team_quality[team] - 6.5) * 0.3
                        + rng.normal(0, 0.7),
                        1.0,
                        10.0,
                    )
                )
                rows.append(
                    {
                        "date": m.date,
                        "team": team,
                        "player": p,
                        "position": POSITIONS[int(rng.integers(len(POSITIONS)))],
                        "rating": rating,
                        "minutes": 90 if available else 0,
                        "goals": int(rng.poisson(0.12)) if available else 0,
                        "assists": int(rng.poisson(0.08)) if available else 0,
                        "available": available,
                    }
                )

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df