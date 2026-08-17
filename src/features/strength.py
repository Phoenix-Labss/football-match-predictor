"""Team strength tracking with confidence-controlled updates.

This module implements the core contribution of the project: treating
team-strength updating as a *confidence-bounded learning process* rather
than a fixed-rule (classic Elo) update.

Update rule
-----------
Let S_t be a team's strength (Elo-style rating) before a match and let
Shat_t be the strength implied by the observed result:

    standard:  S_{t+1} = S_t + K_eff * (W - E)          [classic Elo]
    bounded:   S_{t+1} = S_t + clip(K_eff * (W - E), -delta, +delta)
    adaptive:  S_{t+1} = S_t + clip(K_eff * (W - E), -delta_t, +delta_t)

where
    K_eff = K * goal_diff_multiplier   (World Football Elo style)
    W     = observed result score (1 / 0.5 / 0)
    E     = expected result score from the current ratings
    delta   = cap_fraction * rating_scale          (fixed cap, in points)
    delta_t = base_cap * rating_scale * (1 + w_c * consistency_t
                                          + w_s * surprise_t)   [adaptive]

Adaptive evidence signals (computed per team from the last
`evidence_window` matches):
    consistency_t = |mean(residuals)| / (mean(|residuals|) + eps)
        -- a signal-to-noise ratio in [0, 1]: near 1 when recent results
           consistently deviate in the SAME direction (sustained decline
           or improvement), near 0 when deviations are random noise.
    surprise_t = |residual_last| clipped to [0, 1]
        -- magnitude of the most recent unexpected result.

Interpretation: one shocking loss has low consistency (noise-like) and
gets a small allowed update; five losses in seven matches has high
consistency and permits a larger update. This is the "speed limit".
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RATING_SCALE = 400.0  # Elo scale constant (also used to convert caps to points)


@dataclass
class UpdaterConfig:
    """Configuration for the strength update mechanism."""

    mode: str = "adaptive"          # fixed_k | bounded | adaptive | frozen
    k: float = 24.0                 # base learning rate
    cap: float = 0.05               # bounded-mode cap (fraction of rating scale)
    base_cap: float = 0.01          # adaptive-mode base cap (fraction)
    max_cap: float = 0.05           # adaptive-mode maximum cap ceiling (fraction)
    consistency_weight: float = 3.0
    surprise_weight: float = 1.0
    evidence_window: int = 7
    home_advantage: float = 65.0    # Elo points added to home team
    initial_rating: float = 1500.0
    goal_diff_scale: float = 1.0

    def __post_init__(self) -> None:
        valid = {"fixed_k", "bounded", "adaptive", "frozen"}
        if self.mode not in valid:
            raise ValueError(f"mode must be one of {valid}, got '{self.mode}'")

    @classmethod
    def from_config(cls, cfg: dict) -> "UpdaterConfig":
        """Build from the parsed `updater` + `features.elo` config sections."""
        adaptive = cfg.get("adaptive", {})
        return cls(
            mode=cfg.get("mode", "adaptive"),
            k=cfg.get("k", 24.0),
            cap=cfg.get("cap", 0.05),
            base_cap=adaptive.get("base_cap", 0.01),
            max_cap=adaptive.get("max_cap", 0.05),
            consistency_weight=adaptive.get("consistency_weight", 3.0),
            surprise_weight=adaptive.get("surprise_weight", 1.0),
            evidence_window=adaptive.get("evidence_window", 7),
        )


def _goal_diff_multiplier(goal_diff: int, scale: float = 1.0) -> float:
    """World-Football-Elo-style K multiplier based on goal difference."""
    ad = abs(goal_diff)
    if ad <= 1:
        mult = 1.0
    elif ad == 2:
        mult = 1.5
    else:
        mult = (11.0 + ad) / 8.0
    return mult * scale


class StrengthTracker:
    """Chronological team-strength tracker with configurable update rules.

    Usage:
        tracker = StrengthTracker(UpdaterConfig(mode="adaptive"))
        for each match (in date order):
            pre = tracker.pre_match_state(home, away, neutral)
            ... use pre.elo_home / pre.elo_away as features ...
            tracker.update(home, away, home_goals, away_goals, neutral)
    """

    def __init__(self, config: UpdaterConfig):
        self.cfg = config
        self.ratings: dict[str, float] = {}
        # Per-team history of result residuals (W - E) for evidence signals.
        self._residuals: dict[str, deque] = {}

    # ------------------------------------------------------------------ #
    # rating access
    # ------------------------------------------------------------------ #
    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.cfg.initial_rating)

    # ------------------------------------------------------------------ #
    # pre-match quantities
    # ------------------------------------------------------------------ #
    def expected_score(self, home: str, away: str, neutral: bool) -> float:
        """Expected result score for the home team (Elo formula)."""
        ha = 0.0 if neutral else self.cfg.home_advantage
        diff = (self.rating(home) + ha) - self.rating(away)
        return 1.0 / (1.0 + 10.0 ** (-diff / RATING_SCALE))

    def pre_match_state(self, home: str, away: str, neutral: bool) -> dict:
        """Pre-match strength snapshot used as model features."""
        eh = self.rating(home)
        ea = self.rating(away)
        return {
            "elo_home": eh,
            "elo_away": ea,
            "elo_diff": eh - ea,
            "expected_home_score": self.expected_score(home, away, neutral),
        }

    # ------------------------------------------------------------------ #
    # evidence signals (adaptive mode)
    # ------------------------------------------------------------------ #
    def _consistency(self, team: str) -> float:
        """Signal-to-noise ratio of recent residuals, in [0, 1]."""
        res = self._residuals.get(team)
        if not res:
            return 0.0
        s = 0.0
        s_abs = 0.0
        for x in res:
            s += x
            s_abs += abs(x)
        if s_abs < 1e-9:
            return 0.0
        return min(1.0, max(0.0, abs(s) / s_abs))

    def _surprise(self, team: str) -> float:
        """Magnitude of the most recent residual, clipped to [0, 1]."""
        res = self._residuals.get(team)
        if not res:
            return 0.0
        return min(1.0, max(0.0, abs(res[-1])))

    def _adaptive_cap_points(self, team: str) -> float:
        """delta_t in rating points for the adaptive mode."""
        c = self.cfg
        base = c.base_cap * RATING_SCALE
        growth = 1.0 + c.consistency_weight * self._consistency(
            team
        ) + c.surprise_weight * self._surprise(team)
        pts = base * growth
        if c.max_cap is not None:
            pts = min(pts, c.max_cap * RATING_SCALE)
        return float(pts)

    # ------------------------------------------------------------------ #
    # update
    # ------------------------------------------------------------------ #
    def update(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        neutral: bool,
    ) -> dict:
        """Apply the configured update rule after observing a result."""
        c = self.cfg
        if c.mode == "frozen":
            # Control condition: record residuals but never move ratings.
            self._record_residuals(home, away, home_goals, away_goals, neutral)
            return {"delta_home": 0.0, "delta_away": 0.0, "cap_home": 0.0, "cap_away": 0.0}

        expected_home = self.expected_score(home, away, neutral)
        observed_home = 1.0 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0.0)
        residual_home = observed_home - expected_home
        residual_away = -residual_home

        k_eff = c.k * _goal_diff_multiplier(home_goals - away_goals, c.goal_diff_scale)
        raw_home = k_eff * residual_home
        raw_away = k_eff * residual_away

        if c.mode == "fixed_k":
            cap_points = np.inf
        elif c.mode == "bounded":
            cap_points = c.cap * RATING_SCALE
        else:  # adaptive
            cap_points = None  # per-team

        delta_home = raw_home
        delta_away = raw_away
        cap_h = cap_a = float(cap_points) if cap_points is not None else 0.0

        if c.mode == "adaptive":
            cap_h = self._adaptive_cap_points(home)
            cap_a = self._adaptive_cap_points(away)
            delta_home = float(np.clip(raw_home, -cap_h, cap_h))
            delta_away = float(np.clip(raw_away, -cap_a, cap_a))
        elif cap_points is not np.inf:
            delta_home = float(np.clip(raw_home, -cap_points, cap_points))
            delta_away = float(np.clip(raw_away, -cap_points, cap_points))

        self.ratings[home] = self.rating(home) + delta_home
        self.ratings[away] = self.rating(away) + delta_away

        self._record_residuals(home, away, home_goals, away_goals, neutral)

        return {
            "delta_home": delta_home,
            "delta_away": delta_away,
            "cap_home": cap_h,
            "cap_away": cap_a,
        }

    def _record_residuals(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        neutral: bool,
    ) -> None:
        expected_home = self.expected_score(home, away, neutral)
        observed_home = (
            1.0 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0.0)
        )
        residual = observed_home - expected_home
        w = self.cfg.evidence_window
        for team, res in ((home, residual), (away, -residual)):
            if team not in self._residuals:
                self._residuals[team] = deque(maxlen=w)
            self._residuals[team].append(res)


def run_tracker_over_matches(
    matches: pd.DataFrame, config: UpdaterConfig
) -> tuple[pd.DataFrame, StrengthTracker]:
    """Run the tracker over a chronological match table.

    Returns a DataFrame (aligned with `matches`) of PRE-match strength
    features -- safe to use as model inputs without leakage -- plus the
    fitted tracker for inspection.
    """
    tracker = StrengthTracker(config)
    pre_rows = []
    for row in matches.itertuples(index=False):
        pre = tracker.pre_match_state(row.home_team, row.away_team, bool(row.neutral))
        pre_rows.append(pre)
        tracker.update(
            row.home_team,
            row.away_team,
            int(row.home_goals),
            int(row.away_goals),
            bool(row.neutral),
        )
    return pd.DataFrame(pre_rows, index=matches.index), tracker