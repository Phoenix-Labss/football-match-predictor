"""Squad model: starting XI selection, formations, positional fit, manager effect.

Given a national player pool (from PlayerModel), select a starting XI for a
chosen formation, applying:

    - positional fit: a player fielded out of position is penalised
      (e.g. a ST played at CB). The penalty is a multiplicative factor on
      the player's effective contribution, derived from how "close" their
      listed positions are to the slot.
    - manager effect: a learnable team-level multiplier on attack vs defence
      orientation (since real manager identities are sparse in public data,
      we model the *effect* as a tactical-style parameter per team).
    - availability: each selected player's contribution is weighted by
      P(available) so missing key players degrades the XI.

The output is a TeamRating: attack, midfield, defence, gk strength channels
plus a chemistry score, fed into the match engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .player_model import PlayerState


# --------------------------------------------------------------------- #
# Formations
# --------------------------------------------------------------------- #
# Each formation is a list of positional slots. A slot is (group, label)
# where group is one of GK/DEF/MID/ATT and label is a specific position.
# The number of slots is always 11.
FORMATIONS: dict[str, list[tuple[str, str]]] = {
    "4-3-3": [
        ("GK", "GK"),
        ("DEF", "LB"), ("DEF", "CB"), ("DEF", "CB"), ("DEF", "RB"),
        ("MID", "CM"), ("MID", "CM"), ("MID", "CM"),
        ("ATT", "LW"), ("ATT", "ST"), ("ATT", "RW"),
    ],
    "4-2-3-1": [
        ("GK", "GK"),
        ("DEF", "LB"), ("DEF", "CB"), ("DEF", "CB"), ("DEF", "RB"),
        ("MID", "CDM"), ("MID", "CDM"),
        ("MID", "LM"), ("MID", "CAM"), ("MID", "RM"),
        ("ATT", "ST"),
    ],
    "3-5-2": [
        ("GK", "GK"),
        ("DEF", "CB"), ("DEF", "CB"), ("DEF", "CB"),
        ("MID", "LWB"), ("MID", "CM"), ("MID", "CM"), ("MID", "CM"), ("MID", "RWB"),
        ("ATT", "ST"), ("ATT", "ST"),
    ],
    "4-4-2": [
        ("GK", "GK"),
        ("DEF", "LB"), ("DEF", "CB"), ("DEF", "CB"), ("DEF", "RB"),
        ("MID", "LM"), ("MID", "CM"), ("MID", "CM"), ("MID", "RM"),
        ("ATT", "ST"), ("ATT", "ST"),
    ],
    "5-3-2": [
        ("GK", "GK"),
        ("DEF", "LWB"), ("DEF", "CB"), ("DEF", "CB"), ("DEF", "CB"), ("DEF", "RWB"),
        ("MID", "CM"), ("MID", "CM"), ("MID", "CM"),
        ("ATT", "ST"), ("ATT", "ST"),
    ],
}

# Positional compatibility matrix: how well a player whose listed group is
# `row` fits a slot of group `col`. 1.0 = ideal, <1.0 = penalty.
GROUP_COMPAT = {
    "GK":  {"GK": 1.0, "DEF": 0.0, "MID": 0.0, "ATT": 0.0},
    "DEF": {"GK": 0.0, "DEF": 1.0, "MID": 0.6, "ATT": 0.2},
    "MID": {"GK": 0.0, "DEF": 0.6, "MID": 1.0, "ATT": 0.7},
    "ATT": {"GK": 0.0, "DEF": 0.2, "MID": 0.7, "ATT": 1.0},
}

# Within-group fine position compatibility (specific labels).
FINE_COMPAT = {
    ("CB", "CB"): 1.0, ("CB", "LB"): 0.75, ("CB", "RB"): 0.75,
    ("CB", "LWB"): 0.55, ("CB", "RWB"): 0.55,
    ("LB", "LB"): 1.0, ("LB", "LWB"): 0.9, ("LB", "CB"): 0.7, ("LB", "RB"): 0.4,
    ("RB", "RB"): 1.0, ("RB", "RWB"): 0.9, ("RB", "CB"): 0.7, ("RB", "LB"): 0.4,
    ("LWB", "LWB"): 1.0, ("LWB", "LB"): 0.9, ("LWB", "LM"): 0.6,
    ("RWB", "RWB"): 1.0, ("RWB", "RB"): 0.9, ("RWB", "RM"): 0.6,
    ("CM", "CM"): 1.0, ("CM", "CDM"): 0.85, ("CM", "CAM"): 0.85,
    ("CM", "LM"): 0.7, ("CM", "RM"): 0.7,
    ("CDM", "CDM"): 1.0, ("CDM", "CM"): 0.9, ("CDM", "CB"): 0.5,
    ("CAM", "CAM"): 1.0, ("CAM", "CM"): 0.85, ("CAM", "LM"): 0.65, ("CAM", "RM"): 0.65,
    ("LM", "LM"): 1.0, ("LM", "RM"): 0.7, ("LM", "CM"): 0.7, ("LM", "LW"): 0.8,
    ("RM", "RM"): 1.0, ("RM", "LM"): 0.7, ("RM", "CM"): 0.7, ("RM", "RW"): 0.8,
    ("ST", "ST"): 1.0, ("ST", "CF"): 0.95, ("ST", "LW"): 0.6, ("ST", "RW"): 0.6,
    ("CF", "CF"): 1.0, ("CF", "ST"): 0.95, ("CF", "CAM"): 0.7,
    ("LW", "LW"): 1.0, ("LW", "RW"): 0.75, ("LW", "ST"): 0.7, ("LW", "LM"): 0.8,
    ("RW", "RW"): 1.0, ("RW", "LW"): 0.75, ("RW", "ST"): 0.7, ("RW", "RM"): 0.8,
    ("GK", "GK"): 1.0,
}


def positional_fit(player_positions: str, slot_group: str, slot_label: str) -> float:
    """Return a 0-1 fit factor for a player in a slot."""
    if not player_positions:
        return 0.5 * GROUP_COMPAT.get("MID", {}).get(slot_group, 0.5)

    listed = [p.strip() for p in player_positions.split(",") if p.strip()]
    best = 0.0
    for pos in listed:
        pgroup = "GK"
        if pos in ("CB", "LB", "RB", "LWB", "RWB"):
            pgroup = "DEF"
        elif pos in ("CM", "CDM", "CAM", "LM", "RM"):
            pgroup = "MID"
        elif pos in ("ST", "CF", "LW", "RW"):
            pgroup = "ATT"
        g_compat = GROUP_COMPAT.get(pgroup, {}).get(slot_group, 0.3)
        f_compat = FINE_COMPAT.get((pos, slot_label), g_compat)
        if pgroup == "GK" and slot_group != "GK":
            score = 0.0
        else:
            score = g_compat * f_compat
        best = max(best, score)
    return best


@dataclass
class TeamRating:
    """Aggregated team strength channels for the match engine."""

    team: str
    formation: str
    attack: float
    midfield: float
    defence: float
    gk: float
    chemistry: float
    manager_attack: float
    manager_defence: float
    lineup: list[int] = field(default_factory=list)


@dataclass
class ManagerStyle:
    """Learnable team-level tactical style (manager effect proxy)."""

    attack_bias: float = 0.0
    defence_bias: float = 0.0
    aggression: float = 0.0


class SquadModel:
    """Selects a starting XI and aggregates team strength channels."""

    def __init__(
        self,
        formation: str = "4-3-3",
        manager_styles: dict[str, ManagerStyle] | None = None,
        off_position_floor: float = 0.3,
    ):
        if formation not in FORMATIONS:
            raise ValueError(f"Unknown formation '{formation}'")
        self.formation = formation
        self.manager_styles = manager_styles or {}
        self.off_position_floor = off_position_floor

    def select_lineup(
        self, pool: list[PlayerState]
    ) -> list[tuple[PlayerState, float]]:
        """Greedy best-fit selection aligned to formation slots."""
        slots = FORMATIONS[self.formation]
        used: set[int] = set()
        # Order slots by scarcity (GK first, then DEF, ATT, MID).
        slot_order = sorted(
            range(len(slots)),
            key=lambda i: {"GK": 0, "DEF": 1, "ATT": 2, "MID": 3}[slots[i][0]],
        )
        picks: dict[int, tuple[PlayerState, float]] = {}
        for i in slot_order:
            group, label = slots[i]
            best_player = None
            best_score = -1.0
            best_fit = 0.0
            for p in pool:
                if p.sofifa_id in used:
                    continue
                fit = positional_fit(p.positions, group, label)
                fit = max(fit, self.off_position_floor if p.ability > 0 else 0.0)
                score = fit * p.form * p.availability
                if score > best_score:
                    best_score = score
                    best_player = p
                    best_fit = fit
            if best_player is not None:
                used.add(best_player.sofifa_id)
                picks[i] = (best_player, best_fit)
        return [picks[i] for i in range(len(slots)) if i in picks]

    def aggregate(
        self,
        team: str,
        pool: list[PlayerState],
        chemistry_score: float = 0.0,
    ) -> TeamRating:
        """Select the XI and aggregate into a TeamRating."""
        lineup = self.select_lineup(pool)
        slots = FORMATIONS[self.formation]

        att_vals, mid_vals, def_vals, gk_vals = [], [], [], []
        lineup_ids = []
        for (p, fit), (group, _label) in zip(lineup, slots):
            lineup_ids.append(p.sofifa_id)
            if group == "GK":
                gk_vals.append((p.gk_ability * fit) if p.gk_ability else (p.form * 0.8))
            elif group == "DEF":
                def_vals.append(
                    (0.6 * p.defending + 0.25 * p.physical + 0.15 * p.ability) * fit
                )
            elif group == "MID":
                mid_vals.append(
                    (0.35 * p.passing + 0.25 * p.dribbling
                     + 0.2 * p.defending + 0.2 * p.ability) * fit
                )
            else:  # ATT
                att_vals.append(
                    (0.4 * p.shooting + 0.25 * p.dribbling
                     + 0.2 * p.pace + 0.15 * p.ability) * fit
                )

        all_ability = [p.ability for p, _ in lineup] or [50.0]
        fallback = float(np.mean(all_ability))

        style = self.manager_styles.get(team, ManagerStyle())
        atk = (np.mean(att_vals) if att_vals else fallback) * (1.0 + style.attack_bias)
        mid = np.mean(mid_vals) if mid_vals else fallback
        dfn = (np.mean(def_vals) if def_vals else fallback) * (1.0 + style.defence_bias)
        gk = np.mean(gk_vals) if gk_vals else fallback * 0.8

        return TeamRating(
            team=team,
            formation=self.formation,
            attack=float(atk),
            midfield=float(mid),
            defence=float(dfn),
            gk=float(gk),
            chemistry=float(chemistry_score),
            manager_attack=float(style.attack_bias),
            manager_defence=float(style.defence_bias),
            lineup=lineup_ids,
        )