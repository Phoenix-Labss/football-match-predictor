"""Player-aware team representation features (Track 2, RQ1).

Implements the player-level feature groups from the research plan:

  Phase 3 -- per-position aggregation of player quality into team-level
             attack / midfield / defence / GK strength (learned by the
             downstream model, not hand-averaged).
  Phase 4 -- separation of long-term ability vs. recent form, with the
             mixing weight learned by the model.
  Phase 5 -- availability-adjusted expected contribution:
             E[contribution] = P(available) * quality.
  Phase 6 -- pairwise player interaction features (co-appearance,
             assists between pairs, positional compatibility).

This module operates on a generic player-match table so it can be fed
from the European Soccer Database (Dataset B) or any provider with the
same schema:

    date, team, player, position, rating, minutes, goals, assists,
    available (bool)

The aggregation is deliberately simple (per-position weighted means)
because the LEARNER is expected to learn how the aggregates combine --
the point is to give it separate, interpretable channels rather than a
single blended number.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

POSITION_GROUPS = {
    "GK": ["GK"],
    "DEF": ["CB", "LB", "RB", "LWB", "RWB", "DF"],
    "MID": ["CM", "DM", "AM", "LM", "RM", "MF"],
    "ATT": ["ST", "CF", "LW", "RW", "FW"],
}


def normalise_position(pos: str) -> str:
    """Map a raw position string to one of GK / DEF / MID / ATT."""
    pos = str(pos).strip().upper()
    for group, members in POSITION_GROUPS.items():
        if pos in members:
            return group
    # Fallback heuristics for messy data.
    if "K" in pos:
        return "GK"
    if "B" in pos or "D" in pos:
        return "DEF"
    if "W" in pos or "S" in pos or "F" in pos:
        return "ATT"
    return "MID"


class PlayerState:
    """Tracks per-player long-term ability and recent form over time.

    - ability : exponentially weighted mean of match ratings (slow)
    - form    : mean of the last `form_window` match ratings (fast)
    """

    def __init__(self, ability_halflife: int = 60, form_window: int = 6):
        self.ability_halflife = ability_halflife
        self.form_window = form_window
        self._ability: dict[str, float] = {}
        self._form: dict[str, deque] = {}

    def ability(self, player: str) -> float:
        return self._ability.get(player, np.nan)

    def form(self, player: str) -> float:
        f = self._form.get(player)
        if not f:
            return np.nan
        return float(np.mean(f))

    def update(self, player: str, rating: float) -> None:
        a = self._ability.get(player)
        alpha = 1.0 - 0.5 ** (1.0 / self.ability_halflife)
        if a is None or np.isnan(a):
            self._ability[player] = rating
        else:
            self._ability[player] = a + alpha * (rating - a)
        if player not in self._form:
            self._form[player] = deque(maxlen=self.form_window)
        self._form[player].append(rating)


class PairInteractionState:
    """Tracks pairwise player interaction statistics over time.

    Records co-appearance counts and (optionally) assist links between
    player pairs within a team. Used to build chemistry features.
    """

    def __init__(self):
        self._coappear: dict[tuple[str, str, str], int] = defaultdict(int)
        self._assists: dict[tuple[str, str, str], int] = defaultdict(int)

    @staticmethod
    def _key(team: str, a: str, b: str) -> tuple[str, str, str]:
        return (team,) + tuple(sorted((a, b)))

    def coappearances(self, team: str, a: str, b: str) -> int:
        return self._coappear[self._key(team, a, b)]

    def assist_links(self, team: str, a: str, b: str) -> int:
        return self._assists[self._key(team, a, b)]

    def record_lineup(
        self,
        team: str,
        players: list[str],
        assist_pairs: list[tuple[str, str]] | None = None,
    ) -> None:
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                self._coappear[self._key(team, players[i], players[j])] += 1
        if assist_pairs:
            for a, b in assist_pairs:
                self._assists[self._key(team, a, b)] += 1


def build_player_team_features(
    player_matches: pd.DataFrame,
    form_window: int = 6,
    ability_halflife: int = 60,
) -> pd.DataFrame:
    """Aggregate player-level data into per-(date, team) strength channels.

    Input schema (one row per player per match):
        date, team, player, position, rating, minutes, goals, assists,
        available

    Output: one row per (date, team) with columns:
        att_ability, mid_ability, def_ability, gk_ability,
        att_form, mid_form, def_form, gk_form,
        att_expected, mid_expected, def_expected, gk_expected,
        avg_ability, avg_form, squad_depth, chemistry_mean
    """
    ps = PlayerState(ability_halflife=ability_halflife, form_window=form_window)
    pair_state = PairInteractionState()

    out_rows = []
    # Process chronologically, grouped by (date, team).
    player_matches = player_matches.sort_values("date").reset_index(drop=True)

    for (date, team), grp in player_matches.groupby(["date", "team"], sort=True):
        feats: dict = {"date": date, "team": team}

        for group in ("ATT", "MID", "DEF", "GK"):
            grp_pos = grp[grp["position"].map(normalise_position) == group]
            abilities, forms, expected = [], [], []
            for r in grp_pos.itertuples(index=False):
                ab = ps.ability(r.player)
                fm = ps.form(r.player)
                avail = float(getattr(r, "available", True))
                abilities.append(ab)
                forms.append(fm)
                # Availability-adjusted expected contribution (Phase 5).
                expected.append(
                    np.nan if np.isnan(ab) else avail * ab
                )
            feats[f"{group.lower()}_ability"] = (
                float(np.nanmean(abilities)) if abilities else np.nan
            )
            feats[f"{group.lower()}_form"] = (
                float(np.nanmean(forms)) if forms else np.nan
            )
            feats[f"{group.lower()}_expected"] = (
                float(np.nanmean(expected)) if expected else np.nan
            )

        all_ab = [ps.ability(r.player) for r in grp.itertuples(index=False)]
        feats["avg_ability"] = float(np.nanmean(all_ab)) if all_ab else np.nan
        feats["squad_depth"] = float(len(grp))

        # Chemistry: mean co-appearance count over lineup pairs (Phase 6).
        lineup = list(grp["player"])
        if len(lineup) > 1:
            co = [
                pair_state.coappearances(team, lineup[i], lineup[j])
                for i in range(len(lineup))
                for j in range(i + 1, len(lineup))
            ]
            feats["chemistry_mean"] = float(np.mean(co))
        else:
            feats["chemistry_mean"] = 0.0

        out_rows.append(feats)

        # Update states AFTER snapshotting (no leakage).
        for r in grp.itertuples(index=False):
            if getattr(r, "available", True) and getattr(r, "minutes", 90) > 0:
                ps.update(r.player, float(r.rating))
        pair_state.record_lineup(team, lineup)

    return pd.DataFrame(out_rows)