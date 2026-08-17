"""Player model: ability, form, age curve, and availability (Track 3).

Turns the raw FIFA player table (one row per player per year) into a
per-player state used by the squad model. The key quantities:

    ability      : long-term quality (the FIFA `overall`, age-adjusted)
    form         : short-term fluctuation around ability (we lack
                   match-level ratings for internationals, so form is
                   modelled as a small mean-reverting noise around ability)
    age_factor   : multiplicative adjustment from the age curve (players
                   peak ~27-29, decline after)
    availability : probability the player is in the match-day squad, a
                   function of their quality relative to the national pool

The model is deliberately simple and interpretable: the downstream squad
and match models learn how these channels combine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def age_factor(age: float) -> float:
    """Multiplicative ability factor from age.

    Players develop until ~27-29, hold until ~31, then decline.
    Calibrated to typical FIFA career arcs.
    """
    if age <= 24:
        return 0.85 + 0.15 * max(0.0, (age - 17)) / 7.0
    if age <= 31:
        return 1.0
    if age <= 35:
        return 1.0 - 0.03 * (age - 31)
    return max(0.6, 0.88 - 0.05 * (age - 35))


@dataclass
class PlayerState:
    """Snapshot of a single player's modelled state for a given year."""

    sofifa_id: int
    name: str
    nationality: str
    club: str
    league: str
    age: float
    positions: str
    preferred_foot: str
    work_rate: str
    overall: float
    potential: float
    ability: float          # age-adjusted long-term quality
    form: float             # short-term fluctuation
    availability: float     # P(in squad)
    # Raw attribute channels used by the match engine.
    pace: float
    shooting: float
    passing: float
    dribbling: float
    defending: float
    physical: float
    gk_ability: float
    finishing: float
    composure: float
    vision: float
    interceptions: float
    tackling: float
    stamina: float


class PlayerModel:
    """Builds per-player states from the FIFA player table for a given year."""

    def __init__(
        self,
        form_volatility: float = 0.02,
        availability_floor: float = 0.5,
        seed: int = 42,
    ):
        self.form_volatility = form_volatility
        self.availability_floor = availability_floor
        self.rng = np.random.default_rng(seed)

    def build_states(
        self, players: pd.DataFrame, year: int
    ) -> dict[int, PlayerState]:
        """Return {sofifa_id: PlayerState} for all players in the given year."""
        yr = players[players["year"] == year].copy()
        if yr.empty:
            avail = players[players["year"] <= year]
            if avail.empty:
                avail = players
            year = avail["year"].max()
            yr = players[players["year"] == year].copy()

        # Availability: relative quality within the national pool.
        yr["pool_rank"] = yr.groupby("nationality")["overall"].rank(
            method="first", ascending=False
        )
        pool_size = yr.groupby("nationality")["overall"].transform("count")
        yr["availability"] = np.where(
            yr["pool_rank"] <= 23,
            self.availability_floor + 0.5 * (1.0 - yr["pool_rank"] / 23.0),
            np.maximum(self.availability_floor - 0.3, 0.1)
            * (1.0 - (yr["pool_rank"] - 23) / np.maximum(pool_size - 23, 1)),
        )
        yr["availability"] = yr["availability"].clip(0.05, 0.99)

        states: dict[int, PlayerState] = {}
        for r in yr.itertuples(index=False):
            af = age_factor(r.age)
            ability = r.overall * af
            form = ability * (1.0 + self.rng.normal(0, self.form_volatility))
            gk_cols = [
                getattr(r, "goalkeeping_diving", np.nan),
                getattr(r, "goalkeeping_handling", np.nan),
                getattr(r, "goalkeeping_reflexes", np.nan),
                getattr(r, "goalkeeping_positioning", np.nan),
            ]
            gk_vals = [g for g in gk_cols if not np.isnan(g)]
            gk = float(np.mean(gk_vals)) if gk_vals else 0.0

            states[r.sofifa_id] = PlayerState(
                sofifa_id=r.sofifa_id,
                name=r.name,
                nationality=r.nationality,
                club=r.club,
                league=r.league,
                age=float(r.age),
                positions=r.positions,
                preferred_foot=r.preferred_foot,
                work_rate=r.work_rate,
                overall=float(r.overall),
                potential=float(r.potential),
                ability=float(ability),
                form=float(form),
                availability=float(r.availability),
                pace=float(getattr(r, "pace", 0.0) or 0.0),
                shooting=float(getattr(r, "shooting", 0.0) or 0.0),
                passing=float(getattr(r, "passing", 0.0) or 0.0),
                dribbling=float(getattr(r, "dribbling", 0.0) or 0.0),
                defending=float(getattr(r, "defending", 0.0) or 0.0),
                physical=float(getattr(r, "physical", 0.0) or 0.0),
                gk_ability=float(gk),
                finishing=float(getattr(r, "attacking_finishing", 0.0) or 0.0),
                composure=float(getattr(r, "mentality_composure", 0.0) or 0.0),
                vision=float(getattr(r, "mentality_vision", 0.0) or 0.0),
                interceptions=float(getattr(r, "mentality_interceptions", 0.0) or 0.0),
                tackling=float(getattr(r, "defending_standing_tackle", 0.0) or 0.0),
                stamina=float(getattr(r, "power_stamina", 0.0) or 0.0),
            )
        return states

    def national_pool(
        self, states: dict[int, PlayerState], nation: str
    ) -> list[PlayerState]:
        """All modelled players for a given nationality, sorted by ability."""
        pool = [s for s in states.values() if s.nationality == nation]
        return sorted(pool, key=lambda s: s.ability, reverse=True)