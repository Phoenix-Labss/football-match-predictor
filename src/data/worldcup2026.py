"""World Cup 2026 dataset loader (Track 3, real data).

Loads the FIFA World Cup 2026 dataset (mominullptr/FIFA-World-Cup-2026-Dataset,
mirrored on GitHub). This is real tournament data with:

    squads_and_players.csv : player_id, team_id, player_name, position,
                             club_team, market_value_eur, caps, date_of_birth,
                             height_cm, goals
    player_stats.csv       : per-tournament performance (minutes, goals,
                             assists, shots, cards, saves, ...)
    teams.csv              : team_id, team_name, fifa_code, group_letter,
                             confederation, fifa_ranking_pre_tournament,
                             elo_rating, manager_name
    match_lineups.csv      : match_id, player_id, is_starting_xi,
                             tactical_position, minutes_played
    matches.csv            : scores, xG, stages, venues, penalties
    venues.csv             : stadium, city, capacity, elevation_meters

Player ability is estimated from `market_value_eur` (a strong, leakage-free
pre-tournament proxy for quality) log-scaled to the FIFA 0-99 range, plus a
small adjustment for international experience (`caps`). Sub-attributes are
derived from the estimated overall and the player's position group, exactly
as in the synthetic generator, so the same PlayerModel / SquadModel /
MatchEngine pipeline runs unchanged on real data.

The team table (Elo, FIFA ranking, manager) is exposed separately for use
as strength priors and manager effects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Canonical player schema (matches src/data/fifa_players.py CANONICAL_COLUMNS).
CANONICAL_COLUMNS = [
    "year", "sofifa_id", "name", "age", "nationality", "club", "league",
    "overall", "potential", "preferred_foot", "positions", "work_rate",
    "pace", "shooting", "passing", "dribbling", "defending", "physical",
    "attacking_finishing", "attacking_heading", "skill_long_passing",
    "skill_ball_control", "movement_sprint_speed", "movement_agility",
    "power_shot_power", "power_stamina", "power_strength",
    "mentality_vision", "mentality_composure", "mentality_interceptions",
    "defending_marking", "defending_standing_tackle",
    "defending_sliding_tackle", "goalkeeping_diving",
    "goalkeeping_handling", "goalkeeping_kicking",
    "goalkeeping_reflexes", "goalkeeping_positioning",
]

# Map the dataset's coarse positions to our fine position labels.
POSITION_MAP = {
    "GK": "GK",
    "DEF": "CB",
    "MID": "CM",
    "FWD": "ST",
    "ATT": "ST",
}

TOURNAMENT_YEAR = 2026
TOURNAMENT_DATE = pd.Timestamp("2026-06-11")  # opening match


def _market_value_to_overall(mv: float, caps: float) -> float:
    """Estimate a FIFA-style overall rating from market value + caps.

    Market value is the strongest public signal of player quality and is
    leakage-free (it's a pre-tournament valuation). We log-scale it to
    roughly 52-93 and add a small experience bonus from caps.
    """
    mv = max(float(mv), 1.0)
    log_mv = np.log10(mv)
    # log10 range in the data: ~4.4 (25k) to ~8.3 (200M).
    base = 52.0 + (log_mv - 4.4) / (8.3 - 4.4) * (93.0 - 52.0)
    # Experience bonus: up to +3 for very capped players.
    exp_bonus = 3.0 * min(float(caps), 100.0) / 100.0
    return float(np.clip(base + exp_bonus, 45.0, 94.0))


def _position_group(pos: str) -> str:
    pos = str(pos).strip().upper()
    if pos == "GK":
        return "GK"
    if pos == "DEF":
        return "DEF"
    if pos == "MID":
        return "MID"
    return "ATT"  # FWD / ATT


def _derive_attributes(overall: float, group: str, rng: np.random.Generator) -> dict:
    """Derive sub-attributes from overall + position group.

    Mirrors the synthetic generator's position biases so the match engine
    receives the same channels on real data.
    """
    def attr(base: float, bias: float, spread: float = 4.0) -> float:
        return float(np.clip(base + bias + rng.normal(0, spread), 1, 99))

    if group == "GK":
        return {
            "pace": overall, "shooting": overall, "passing": overall,
            "dribbling": overall, "defending": overall, "physical": overall,
            "attacking_finishing": attr(overall, -30),
            "attacking_heading": attr(overall, -20),
            "skill_long_passing": attr(overall, -10),
            "skill_ball_control": attr(overall, -10),
            "movement_sprint_speed": attr(overall, -25),
            "movement_agility": attr(overall, -15),
            "power_shot_power": attr(overall, -20),
            "power_stamina": attr(overall, -10),
            "power_strength": attr(overall, 0),
            "mentality_vision": attr(overall, -10),
            "mentality_composure": attr(overall, 5),
            "mentality_interceptions": attr(overall, -10),
            "defending_marking": attr(overall, -10),
            "defending_standing_tackle": attr(overall, -10),
            "defending_sliding_tackle": attr(overall, -15),
            "goalkeeping_diving": attr(overall, 0),
            "goalkeeping_handling": attr(overall, 0),
            "goalkeeping_kicking": attr(overall, 0),
            "goalkeeping_reflexes": attr(overall, 2),
            "goalkeeping_positioning": attr(overall, 0),
        }

    if group == "DEF":
        p, s, pa, d, de, ph = (overall-5, overall-25, overall-10, overall-15, overall+3, overall+2)
    elif group == "MID":
        p, s, pa, d, de, ph = (overall-3, overall-12, overall+3, overall-2, overall-5, overall-3)
    else:  # ATT
        p, s, pa, d, de, ph = (overall+2, overall+5, overall-5, overall-3, overall-20, overall-5)

    return {
        "pace": float(p), "shooting": float(s), "passing": float(pa),
        "dribbling": float(d), "defending": float(de), "physical": float(ph),
        "attacking_finishing": attr(s, 5 if group == "ATT" else -10),
        "attacking_heading": attr(ph, 3 if group == "ATT" else -5),
        "skill_long_passing": attr(pa, 2),
        "skill_ball_control": attr(d, 3),
        "movement_sprint_speed": attr(p, 0),
        "movement_agility": attr(d, 2),
        "power_shot_power": attr(s, 2),
        "power_stamina": attr(ph, 5),
        "power_strength": attr(ph, 0),
        "mentality_vision": attr(pa, 3),
        "mentality_composure": attr(overall, 2),
        "mentality_interceptions": attr(de, 3),
        "defending_marking": attr(de, 2),
        "defending_standing_tackle": attr(de, 3),
        "defending_sliding_tackle": attr(de, 0),
        "goalkeeping_diving": np.nan, "goalkeeping_handling": np.nan,
        "goalkeeping_kicking": np.nan, "goalkeeping_reflexes": np.nan,
        "goalkeeping_positioning": np.nan,
    }


def load_worldcup2026_players(
    data_dir: str | Path,
    seed: int = 42,
) -> pd.DataFrame:
    """Load squads + player stats and return the canonical player schema.

    Player ability is estimated from market value + caps (leakage-free).
    """
    data_dir = Path(data_dir)
    squads = pd.read_csv(data_dir / "squads_and_players.csv")
    teams = pd.read_csv(data_dir / "teams.csv")

    team_names = dict(zip(teams["team_id"], teams["team_name"]))
    rng = np.random.default_rng(seed)

    rows = []
    for r in squads.itertuples(index=False):
        group = _position_group(r.position)
        overall = _market_value_to_overall(r.market_value_eur, r.caps)
        # Age from date of birth at the tournament date.
        try:
            dob = pd.to_datetime(r.date_of_birth, errors="coerce")
            age = float((TOURNAMENT_DATE - dob).days / 365.25) if pd.notna(dob) else 27.0
        except Exception:
            age = 27.0

        attrs = _derive_attributes(overall, group, rng)
        rows.append({
            "year": TOURNAMENT_YEAR,
            "sofifa_id": int(r.player_id),
            "name": r.player_name,
            "age": age,
            "nationality": team_names.get(r.team_id, "Unknown"),
            "club": r.club_team if pd.notna(r.club_team) else "Free Agent",
            "league": "Unknown",
            "overall": int(round(overall)),
            "potential": int(round(overall)),
            "preferred_foot": "Right",
            "positions": POSITION_MAP.get(group, "CM"),
            "work_rate": "Medium/Medium",
            **attrs,
        })

    df = pd.DataFrame(rows)
    for col in CANONICAL_COLUMNS[12:]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df["overall"])
    return df[CANONICAL_COLUMNS]


def load_worldcup2026_teams(data_dir: str | Path) -> pd.DataFrame:
    """Load the team table: Elo, FIFA ranking, manager, group, confederation."""
    data_dir = Path(data_dir)
    teams = pd.read_csv(data_dir / "teams.csv")
    return teams


def load_worldcup2026_matches(data_dir: str | Path) -> pd.DataFrame:
    """Load real match results with xG for validation."""
    data_dir = Path(data_dir)
    matches = pd.read_csv(data_dir / "matches.csv")
    teams = pd.read_csv(data_dir / "teams.csv")
    names = dict(zip(teams["team_id"], teams["team_name"]))
    matches["home_team"] = matches["home_team_id"].map(names)
    matches["away_team"] = matches["away_team_id"].map(names)
    return matches


def load_worldcup2026_venues(data_dir: str | Path) -> pd.DataFrame:
    """Load venues (elevation, capacity) for ground-condition effects."""
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "venues.csv")