"""Data loading for the soccer prediction project.

Supports:
  1. Kaggle Dataset A -- International Football Results (results.csv)
     https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
  2. A synthetic generator (fallback) so the full pipeline can be run
     end-to-end before the real datasets are downloaded.

The canonical match table schema used throughout the project:

    date          : datetime64  (match date)
    home_team     : str
    away_team     : str
    home_goals    : int
    away_goals    : int
    tournament    : str
    city          : str
    country       : str
    neutral       : bool        (True if played at a neutral venue)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "tournament",
    "city",
    "country",
    "neutral",
]


def load_international_results(csv_path: str | Path) -> pd.DataFrame:
    """Load and normalise the Kaggle international results dataset.

    Expected raw columns (Kaggle 'results.csv'):
        date, home_team, away_team, home_score, away_score,
        tournament, city, country, neutral
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"International results CSV not found at '{path}'. "
            "Download it from Kaggle (martj42/international-football-results-"
            "from-1872-to-2017) or remove the path to use synthetic data."
        )

    df = pd.read_csv(path)

    rename = {
        "home_score": "home_goals",
        "away_score": "away_goals",
    }
    df = df.rename(columns=rename)

    missing = set(CANONICAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset at '{path}' is missing expected columns: {sorted(missing)}"
        )

    df = df[CANONICAL_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["neutral"] = df["neutral"].astype(bool)
    df = df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    # Sort chronologically -- mandatory for leakage-free feature generation.
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_matches(config: dict, project_root: str | Path | None = None) -> pd.DataFrame:
    """Load the match table according to the data config.

    Falls back to the synthetic generator when the configured CSV is absent.
    """
    root = Path(project_root) if project_root else Path.cwd()
    csv_path = root / config["data"]["international_results_csv"]

    if csv_path.exists():
        return load_international_results(csv_path)

    # Fallback: synthetic data so the pipeline is runnable out of the box.
    from src.data.synthetic import generate_synthetic_matches

    print(f"[data] '{csv_path}' not found -- using synthetic data instead.")
    return generate_synthetic_matches(**config["data"]["synthetic"])


def add_outcome_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three-class outcome label from the home team's perspective.

    outcome: 0 = home win, 1 = draw, 2 = away win
    """
    df = df.copy()
    conditions = [
        df["home_goals"] > df["away_goals"],
        df["home_goals"] == df["away_goals"],
        df["home_goals"] < df["away_goals"],
    ]
    choices = [0, 1, 2]
    df["outcome"] = np.select(conditions, choices, default=-1)
    return df