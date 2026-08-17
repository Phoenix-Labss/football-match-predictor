"""Multi-year FIFA player dataset loader (FIFA 15 to 22 + World Cup 2026 era).

Loads real per-year FIFA player attributes:
    FIFA 15: players_15.csv (2015)
    FIFA 16: players_16.csv (2016)
    FIFA 17: players_17.csv (2017)
    FIFA 18: players_18.csv / CompleteDataset.csv (2018)
    FIFA 19: players_19.csv / 2019/data.csv (2019)
    FIFA 20: players_20.csv (2020)
    FIFA 21: players_21.csv / 2021/data.csv (2021)
    FIFA 22: players_22.csv (2022)

Normalises all column-naming conventions into the canonical player schema
used by PlayerModel / SquadModel / MatchEngine.
"""

from __future__ import annotations

import re
from pathlib import Path
import unicodedata
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

DERIVED_ATTRS = [
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


def clean_text(val: any) -> str:
    if val is None or pd.isna(val):
        return ""
    text = str(val).strip()
    return text


def load_fifa_multiyear(
    data_dir: str | Path,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Load all available multi-year FIFA player CSVs into the canonical schema."""
    data_dir = Path(data_dir)
    frames = []

    # Map file patterns to years
    year_map: dict[int, Path] = {}
    
    for p in data_dir.glob("players_*.csv"):
        match = re.search(r"players_(\d+)\.csv", p.name)
        if match:
            yy = int(match.group(1))
            yr = 2000 + yy if yy < 100 else yy
            year_map[yr] = p

    if 2018 not in year_map and (data_dir / "CompleteDataset.csv").exists():
        year_map[2018] = data_dir / "CompleteDataset.csv"
    if 2019 not in year_map and (data_dir / "2019" / "data.csv").exists():
        year_map[2019] = data_dir / "2019" / "data.csv"
    if 2021 not in year_map and (data_dir / "2021" / "data.csv").exists():
        year_map[2021] = data_dir / "2021" / "data.csv"

    for year, path in sorted(year_map.items()):
        if years and year not in years:
            continue
        if not path.exists():
            continue
        try:
            # Try utf-8 first, fallback to latin-1
            try:
                raw = pd.read_csv(path, low_memory=False, encoding="utf-8")
            except UnicodeDecodeError:
                raw = pd.read_csv(path, low_memory=False, encoding="latin-1")
            
            df = _normalise_edition(raw, year)
            frames.append(df)
        except Exception as e:
            print(f"[fifa_multiyear] Warning: failed to load {path}: {e}")

    if not frames:
        raise FileNotFoundError(
            f"No multi-year FIFA CSVs found under '{data_dir}'."
        )

    out = pd.concat(frames, ignore_index=True)
    return out[CANONICAL_COLUMNS]


def _normalise_edition(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """Map one edition's columns to the canonical schema."""
    df = pd.DataFrame(index=raw.index)
    df["year"] = int(year)

    # 1. Identity
    id_col = next((c for c in ["sofifa_id", "ID", "id"] if c in raw.columns), None)
    df["sofifa_id"] = pd.to_numeric(raw[id_col], errors="coerce").fillna(-1).astype(int) if id_col else np.arange(len(raw))

    name_col = next((c for c in ["short_name", "Name", "name", "long_name"] if c in raw.columns), None)
    df["name"] = raw[name_col].astype(str).map(clean_text) if name_col else "Unknown"

    age_col = next((c for c in ["age", "Age"] if c in raw.columns), None)
    df["age"] = pd.to_numeric(raw[age_col], errors="coerce").fillna(26.0) if age_col else 26.0

    nat_col = next((c for c in ["nationality_name", "Nationality", "nationality"] if c in raw.columns), None)
    df["nationality"] = raw[nat_col].astype(str).map(clean_text) if nat_col else "Unknown"

    club_col = next((c for c in ["club_name", "Club", "club"] if c in raw.columns), None)
    df["club"] = raw[club_col].astype(str).map(clean_text) if club_col else "Free Agent"

    league_col = next((c for c in ["league_name", "League", "league"] if c in raw.columns), None)
    df["league"] = raw[league_col].astype(str).map(clean_text) if league_col else "Unknown"

    # 2. Overall & Potential
    ovr_col = next((c for c in ["overall", "Overall"] if c in raw.columns), None)
    df["overall"] = pd.to_numeric(raw[ovr_col], errors="coerce").fillna(65.0) if ovr_col else 65.0

    pot_col = next((c for c in ["potential", "Potential"] if c in raw.columns), None)
    df["potential"] = pd.to_numeric(raw[pot_col], errors="coerce").fillna(df["overall"]) if pot_col else df["overall"]

    # 3. Preferred Foot & Work Rate
    foot_col = next((c for c in ["preferred_foot", "Preferred Foot"] if c in raw.columns), None)
    if foot_col:
        df["preferred_foot"] = raw[foot_col].astype(str).map(lambda x: "Left" if "Left" in str(x) else "Right")
    else:
        df["preferred_foot"] = "Right"

    wr_col = next((c for c in ["work_rate", "Work Rate"] if c in raw.columns), None)
    df["work_rate"] = raw[wr_col].astype(str).fillna("Medium/Medium") if wr_col else "Medium/Medium"

    # 4. Positions
    pos_col = next((c for c in ["player_positions", "Preferred Positions", "Position", "club_position", "positions"] if c in raw.columns), None)
    if pos_col:
        df["positions"] = raw[pos_col].astype(str).fillna("").map(clean_text)
    else:
        df["positions"] = ""

    # 5. Core attributes
    attr_map = {
        "pace": ["pace", "Acceleration", "SprintSpeed", "Sprint Speed"],
        "shooting": ["shooting", "Finishing", "shot_power", "attacking_finishing"],
        "passing": ["passing", "ShortPassing", "Short Passing", "LongPassing"],
        "dribbling": ["dribbling", "Dribbling", "BallControl", "Ball Control"],
        "defending": ["defending", "StandingTackle", "Standing Tackle", "Marking"],
        "physical": ["physic", "physical", "Strength", "power_strength"],
        "attacking_finishing": ["attacking_finishing", "Finishing"],
        "attacking_heading": ["attacking_heading", "HeadingAccuracy", "Heading accuracy"],
        "skill_long_passing": ["skill_long_passing", "LongPassing", "Long Passing"],
        "skill_ball_control": ["skill_ball_control", "BallControl", "Ball Control"],
        "movement_sprint_speed": ["movement_sprint_speed", "SprintSpeed", "Sprint Speed"],
        "movement_agility": ["movement_agility", "Agility"],
        "power_shot_power": ["power_shot_power", "ShotPower", "Shot power"],
        "power_stamina": ["power_stamina", "Stamina"],
        "power_strength": ["power_strength", "Strength"],
        "mentality_vision": ["mentality_vision", "Vision"],
        "mentality_composure": ["mentality_composure", "Composure"],
        "mentality_interceptions": ["mentality_interceptions", "Interceptions"],
        "defending_marking": ["defending_marking", "Marking"],
        "defending_standing_tackle": ["defending_standing_tackle", "StandingTackle", "Standing Tackle"],
        "defending_sliding_tackle": ["defending_sliding_tackle", "SlidingTackle", "Sliding Tackle"],
        "goalkeeping_diving": ["goalkeeping_diving", "GKDiving", "GK diving", "GK Diving"],
        "goalkeeping_handling": ["goalkeeping_handling", "GKHandling", "GK handling", "GK Handling"],
        "goalkeeping_kicking": ["goalkeeping_kicking", "GKKicking", "GK kicking", "GK Kicking"],
        "goalkeeping_reflexes": ["goalkeeping_reflexes", "GKReflexes", "GK reflexes", "GK Reflexes"],
        "goalkeeping_positioning": ["goalkeeping_positioning", "GKPositioning", "GK positioning", "GK Positioning"],
    }

    for canonical, candidates in attr_map.items():
        found = False
        for cand in candidates:
            if cand in raw.columns:
                df[canonical] = pd.to_numeric(raw[cand], errors="coerce")
                found = True
                break
        if not found:
            df[canonical] = np.nan

    # Fill NaNs with overall or sane defaults
    fallback = df["overall"].fillna(65.0)
    for c in DERIVED_ATTRS:
        df[c] = df[c].fillna(fallback)

    return df[CANONICAL_COLUMNS]