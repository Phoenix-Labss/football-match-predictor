"""FIFA/EA player dataset loader (Track 3).

Loads the public "FIFA complete player dataset" (FIFA 15-23, scraped from
sofifa.com and shared on Kaggle). Each yearly file contains ~18,000 players
with ~60 attributes: overall, potential, pace, shooting, passing, dribbling,
defending, physical, plus sub-attributes, age, club, league, nationality,
preferred foot, work rates, player_positions.

For the World Cup simulator we map players -> national teams via the
`nationality` column (the dataset is club football, but nationality is
present, so this is doable).

When the real Kaggle files are absent, a realistic synthetic generator
produces FIFA-style player tables so the full Track 3 pipeline runs
end-to-end (same fallback pattern as Tracks 1 and 2).

Canonical player schema (one row per player per year):

    year            : int        (FIFA version year, e.g. 22 for FIFA 22)
    sofifa_id       : int
    name            : str
    age             : int
    nationality     : str        (maps to national teams)
    club            : str
    league          : str
    overall         : int        (0-99 overall rating)
    potential       : int
    preferred_foot  : str        (Left / Right)
    positions       : str        (e.g. "ST,CF,LW" -- comma-separated)
    work_rate       : str        (e.g. "High/Medium")
    pace            : float
    shooting        : float
    passing         : float
    dribbling       : float
    defending       : float
    physical        : float
    attacking_finishing      : float
    attacking_heading         : float
    skill_long_passing       : float
    skill_ball_control       : float
    movement_sprint_speed    : float
    movement_agility         : float
    power_shot_power         : float
    power_stamina            : float
    power_strength           : float
    mentality_vision         : float
    mentality_composure      : float
    mentality_interceptions  : float
    defending_marking        : float
    defending_standing_tackle: float
    defending_sliding_tackle : float
    goalkeeping_diving       : float
    goalkeeping_handling     : float
    goalkeeping_kicking      : float
    goalkeeping_reflexes     : float
    goalkeeping_positioning  : float
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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

# Kaggle dataset ships one CSV per year: players_15.csv ... players_23.csv
KAGGLE_YEAR_FILES = {y: f"players_{y - 2000}.csv" for y in range(2015, 2024)}


def load_fifa_players(
    data_dir: str | Path,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Load the FIFA complete player dataset from Kaggle CSVs.

    Expects a directory containing `players_15.csv` ... `players_23.csv`
    (the standard Kaggle layout). Returns the canonical schema above.
    """
    data_dir = Path(data_dir)
    years = years or list(range(2015, 2024))
    frames = []
    for year in years:
        fname = KAGGLE_YEAR_FILES.get(year)
        if fname is None:
            continue
        path = data_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["year"] = year
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No FIFA player CSVs found in '{data_dir}'. Download the "
            "'FIFA complete player dataset' from Kaggle "
            "(stefanoleone99/fifa-all-time-football-database) or use the "
            "synthetic fallback."
        )

    raw = pd.concat(frames, ignore_index=True)
    return _normalise_fifa(raw)


def _normalise_fifa(raw: pd.DataFrame) -> pd.DataFrame:
    """Map the raw Kaggle schema to the canonical player schema."""
    rename = {
        "sofifa_id": "sofifa_id",
        "short_name": "name",
        "player_positions": "positions",
        "player_face": "real_face",
    }
    df = raw.rename(columns=rename)

    # Keep only the columns we use; missing ones become NaN.
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[CANONICAL_COLUMNS].copy()
    df["sofifa_id"] = df["sofifa_id"].fillna(-1).astype(int)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
    df["potential"] = pd.to_numeric(df["potential"], errors="coerce")
    for col in CANONICAL_COLUMNS[12:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["nationality"] = df["nationality"].fillna("Unknown").astype(str)
    df["positions"] = df["positions"].fillna("").astype(str)
    df["preferred_foot"] = df["preferred_foot"].fillna("Right").astype(str)
    df["work_rate"] = df["work_rate"].fillna("Medium/Medium").astype(str)
    df["club"] = df["club"].fillna("Free Agent").astype(str)
    df["league"] = df["league"].fillna("Unknown").astype(str)
    return df.dropna(subset=["overall"]).reset_index(drop=True)


# --------------------------------------------------------------------- #
# Synthetic fallback -- realistic FIFA-style player generation
# --------------------------------------------------------------------- #
NATIONAL_TEAMS = [
    "Argentina", "Brazil", "France", "England", "Spain", "Germany",
    "Portugal", "Netherlands", "Belgium", "Italy", "Croatia", "Uruguay",
    "Mexico", "USA", "Switzerland", "Senegal", "Poland", "Denmark",
    "Japan", "Morocco", "Colombia", "Korea Republic", "Serbia", "Wales",
    "Ecuador", "Qatar", "Canada", "Cameroon", "Ghana", "Tunisia",
    "Saudi Arabia", "Australia",
]

CLUBS = [
    "Manchester City", "Real Madrid", "Bayern Munich", "Liverpool",
    "Barcelona", "Paris Saint-Germain", "Chelsea", "Arsenal",
    "Juventus", "Atletico Madrid", "Inter", "AC Milan", "Napoli",
    "Borussia Dortmund", "Tottenham Hotspur", "Atletico Madrid",
]

POSITION_POOL = {
    "GK": ["GK"],
    "DEF": ["CB", "LB", "RB", "LWB", "RWB"],
    "MID": ["CM", "CDM", "CAM", "LM", "RM"],
    "ATT": ["ST", "CF", "LW", "RW"],
}


def _age_curve(age: int) -> float:
    """Multiplier on ability from age: peak ~27-29, decline after."""
    if age <= 29:
        return 1.0 + 0.4 * max(0.0, (age - 17)) / 12.0
    return max(0.7, 1.4 - 0.04 * (age - 29))


def generate_synthetic_fifa_players(
    years: list[int] | None = None,
    players_per_nation: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic FIFA-style player dataset.

    Each national team gets `players_per_nation` players spread across
    positions, with overall ratings drawn from a realistic distribution
    (mean ~68, std ~8, capped 45-93) and sub-attributes derived from the
    overall plus position-appropriate offsets. Players persist across years
    with an age curve so ability evolves.
    """
    rng = np.random.default_rng(seed)
    years = years or list(range(2018, 2024))
    rows = []

    # Build a stable pool of players per nation.
    pid = 0
    for nation in NATIONAL_TEAMS:
        # Position mix: 3 GK, 18 DEF, 22 MID, 17 ATT
        pos_assign = (
            ["GK"] * 3 + ["DEF"] * 18 + ["MID"] * 22 + ["ATT"] * 17
        )[:players_per_nation]
        for group in pos_assign:
            pid += 1
            base_overall = int(np.clip(rng.normal(68, 8), 45, 93))
            base_age = int(np.clip(rng.normal(26, 4), 17, 38))
            foot = rng.choice(["Left", "Right"], p=[0.25, 0.75])
            club = rng.choice(CLUBS)
            league = "English Premier League" if "Manchester" in club or club in (
                "Liverpool", "Chelsea", "Arsenal", "Tottenham Hotspur"
            ) else "Other"
            work = rng.choice(
                ["High/High", "High/Medium", "Medium/Medium",
                 "Medium/Low", "Low/Medium"],
                p=[0.1, 0.3, 0.35, 0.15, 0.1],
            )
            pool = POSITION_POOL[group]
            n_pos = int(rng.integers(1, min(3, len(pool) + 1)))
            pos_str = ",".join(rng.choice(pool, size=n_pos, replace=False))

            for year in years:
                age = base_age + (year - years[0])
                if age > 40 or age < 16:
                    continue
                mult = _age_curve(age)
                overall = int(np.clip(base_overall * mult + rng.normal(0, 1.5), 40, 95))
                potential = int(np.clip(overall + rng.integers(0, 8), overall, 99))

                # Sub-attributes derived from overall + position bias.
                def attr(base: float, bias: float, spread: float = 4.0) -> float:
                    return float(np.clip(base + bias + rng.normal(0, spread), 1, 99))

                if group == "GK":
                    gk_base = overall
                    rows.append({
                        "year": year, "sofifa_id": pid,
                        "name": f"Player_{pid}", "age": age,
                        "nationality": nation, "club": club, "league": league,
                        "overall": overall, "potential": potential,
                        "preferred_foot": foot, "positions": pos_str,
                        "work_rate": work,
                        "pace": np.nan, "shooting": np.nan, "passing": np.nan,
                        "dribbling": np.nan, "defending": np.nan,
                        "physical": np.nan,
                        "attacking_finishing": attr(gk_base, -30),
                        "attacking_heading": attr(gk_base, -20),
                        "skill_long_passing": attr(gk_base, -10),
                        "skill_ball_control": attr(gk_base, -10),
                        "movement_sprint_speed": attr(gk_base, -25),
                        "movement_agility": attr(gk_base, -15),
                        "power_shot_power": attr(gk_base, -20),
                        "power_stamina": attr(gk_base, -10),
                        "power_strength": attr(gk_base, 0),
                        "mentality_vision": attr(gk_base, -10),
                        "mentality_composure": attr(gk_base, 5),
                        "mentality_interceptions": attr(gk_base, -10),
                        "defending_marking": attr(gk_base, -10),
                        "defending_standing_tackle": attr(gk_base, -10),
                        "defending_sliding_tackle": attr(gk_base, -15),
                        "goalkeeping_diving": attr(gk_base, 0),
                        "goalkeeping_handling": attr(gk_base, 0),
                        "goalkeeping_kicking": attr(gk_base, 0),
                        "goalkeeping_reflexes": attr(gk_base, 2),
                        "goalkeeping_positioning": attr(gk_base, 0),
                    })
                else:
                    # Outfielder: pace/shooting/passing/dribbling/defending/physical
                    if group == "DEF":
                        p, s, pa, d, de, ph = (overall-5, overall-25, overall-10, overall-15, overall+3, overall+2)
                    elif group == "MID":
                        p, s, pa, d, de, ph = (overall-3, overall-12, overall+3, overall-2, overall-5, overall-3)
                    else:  # ATT
                        p, s, pa, d, de, ph = (overall+2, overall+5, overall-5, overall-3, overall-20, overall-5)
                    rows.append({
                        "year": year, "sofifa_id": pid,
                        "name": f"Player_{pid}", "age": age,
                        "nationality": nation, "club": club, "league": league,
                        "overall": overall, "potential": potential,
                        "preferred_foot": foot, "positions": pos_str,
                        "work_rate": work,
                        "pace": float(p), "shooting": float(s), "passing": float(pa),
                        "dribbling": float(d), "defending": float(de),
                        "physical": float(ph),
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
                        "goalkeeping_diving": np.nan,
                        "goalkeeping_handling": np.nan,
                        "goalkeeping_kicking": np.nan,
                        "goalkeeping_reflexes": np.nan,
                        "goalkeeping_positioning": np.nan,
                    })

    df = pd.DataFrame(rows)
    # Fill any remaining NaN sub-attributes with overall (sensible default).
    for col in CANONICAL_COLUMNS[12:]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df["overall"])
    return df[CANONICAL_COLUMNS]


def load_or_synthesize_fifa(
    data_dir: str | Path | None,
    years: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load real FIFA data if present, else synthesize."""
    if data_dir is not None:
        data_dir = Path(data_dir)
        if (data_dir / KAGGLE_YEAR_FILES.get(years[0] if years else 2018, "")).exists() if years else any((data_dir / f).exists() for f in KAGGLE_YEAR_FILES.values()):
            try:
                return load_fifa_players(data_dir, years)
            except Exception as exc:  # pragma: no cover
                print(f"[fifa] real load failed ({exc}); using synthetic.")
    print("[fifa] FIFA player CSVs not found -- using synthetic player data.")
    return generate_synthetic_fifa_players(years=years, seed=seed)