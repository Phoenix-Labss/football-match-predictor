"""Synthetic match generator for pipeline development and testing.

Generates a realistic league of matches where team strengths evolve slowly
over time (with occasional abrupt regime changes) and outcomes are drawn
from a Poisson goal model. This gives the confidence-controlled update
mechanism something meaningful to track, and lets the whole pipeline run
end-to-end before the real Kaggle data is available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOURNAMENTS = ["Friendly", "World Cup", "Continental Cup", "Nations League"]


def generate_synthetic_matches(
    n_teams: int = 40,
    n_matches: int = 12000,
    start_date: str = "2000-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a chronological synthetic match table.

    Teams have latent attack/defence strengths that drift slowly and
    occasionally jump (regime change). Goals are Poisson-distributed.
    """
    rng = np.random.default_rng(seed)

    teams = [f"Team_{i:02d}" for i in range(n_teams)]
    n_teams = len(teams)

    # Latent strengths: attack and defence multipliers around 1.0.
    attack = np.clip(rng.normal(1.0, 0.15, n_teams), 0.5, 2.0)
    defence = np.clip(rng.normal(1.0, 0.15, n_teams), 0.5, 2.0)

    base_goals = 1.35  # expected goals for evenly matched teams

    rows = []
    start = pd.Timestamp(start_date)
    # Roughly 4 matches per day keeps dates realistic.
    dates = start + pd.to_timedelta(np.arange(n_matches) // 4, unit="D")

    for i in range(n_matches):
        # Slow drift + rare regime change in latent strengths.
        attack += rng.normal(0.0, 0.004, n_teams)
        defence += rng.normal(0.0, 0.004, n_teams)
        jump = rng.random(n_teams) < 0.0008
        attack[jump] += rng.normal(0.0, 0.12, jump.sum())
        defence[jump] += rng.normal(0.0, 0.12, jump.sum())
        attack = np.clip(attack, 0.5, 2.0)
        defence = np.clip(defence, 0.5, 2.0)

        home_idx, away_idx = rng.choice(n_teams, size=2, replace=False)
        home, away = teams[home_idx], teams[away_idx]

        neutral = bool(rng.random() < 0.25)
        home_adv = 1.15 if not neutral else 1.0

        lam_home = np.clip(base_goals * attack[home_idx] * defence[away_idx] * home_adv, 0.1, 6.0)
        lam_away = np.clip(base_goals * attack[away_idx] * defence[home_idx], 0.1, 6.0)

        hg = int(rng.poisson(lam_home))
        ag = int(rng.poisson(lam_away))

        rows.append(
            {
                "date": dates[i],
                "home_team": home,
                "away_team": away,
                "home_goals": hg,
                "away_goals": ag,
                "tournament": TOURNAMENTS[int(rng.integers(len(TOURNAMENTS)))],
                "city": f"City_{int(rng.integers(50)):02d}",
                "country": f"Country_{int(rng.integers(30)):02d}",
                "neutral": neutral,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df