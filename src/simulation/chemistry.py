"""Chemistry layer: club teammates, minutes shared, positional compatibility.

Football chemistry is the idea that players who know each other well
perform better together. We model three concrete, data-driven signals:

    1. Club-teammate bonus : players from the same club have shared
       training and match minutes, so they gel faster.
    2. Minutes shared       : how often a pair has played together for the
       national team (approximated from club co-membership).
    3. Positional compatibility : a midfield pairing is more natural than
       a striker paired with a centre-back.

The output is a single `chemistry_score` per team (0-1) that the squad
model folds into the team rating.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .player_model import PlayerState


@dataclass
class ChemistryConfig:
    """Weights for the chemistry components."""

    club_weight: float = 0.5
    minutes_weight: float = 0.3
    position_weight: float = 0.2


class ChemistryModel:
    """Computes a team chemistry score from a starting XI."""

    def __init__(
        self,
        config: ChemistryConfig | None = None,
        shared_minutes: dict[tuple[str, int, int], float] | None = None,
    ):
        self.config = config or ChemistryConfig()
        self.shared_minutes = shared_minutes or {}

    def _pair_chemistry(
        self,
        team: str,
        a: PlayerState,
        b: PlayerState,
    ) -> float:
        """Chemistry for a single player pair, in [0, 1]."""
        cfg = self.config

        # 1. Club-teammate bonus.
        club = 1.0 if (a.club and b.club and a.club == b.club) else 0.0

        # 2. Minutes shared (real data if available, else club proxy).
        key = (team, a.sofifa_id, b.sofifa_id)
        minutes = self.shared_minutes.get(key, 0.0)
        if minutes <= 0 and club:
            minutes = 1800.0  # approx shared club minutes per season
        minutes_norm = min(1.0, minutes / 5000.0)

        # 3. Positional compatibility.
        pos_compat = _pair_position_compat(
            _primary_group(a.positions), _primary_group(b.positions)
        )

        return (
            cfg.club_weight * club
            + cfg.minutes_weight * minutes_norm
            + cfg.position_weight * pos_compat
        )

    def team_chemistry(
        self,
        team: str,
        lineup: list[PlayerState],
    ) -> float:
        """Mean pairwise chemistry over the starting XI, in [0, 1]."""
        if len(lineup) < 2:
            return 0.0
        scores = []
        for i in range(len(lineup)):
            for j in range(i + 1, len(lineup)):
                scores.append(self._pair_chemistry(team, lineup[i], lineup[j]))
        return float(np.mean(scores))


def _primary_group(positions: str) -> str:
    """Return the primary positional group of a player's listed positions."""
    if not positions:
        return "MID"
    first = positions.split(",")[0].strip()
    if first in ("CB", "LB", "RB", "LWB", "RWB"):
        return "DEF"
    if first in ("CM", "CDM", "CAM", "LM", "RM"):
        return "MID"
    if first in ("ST", "CF", "LW", "RW"):
        return "ATT"
    return "MID"


def _pair_position_compat(g1: str, g2: str) -> float:
    """How natural is a pairing between two positional groups, in [0, 1]."""
    if g1 == g2:
        return 1.0
    if g1 == "GK" or g2 == "GK":
        return 0.5
    if {g1, g2} == {"ATT", "DEF"}:
        return 0.3
    return 0.8