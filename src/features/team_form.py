"""Springer-style team-level historical performance features (M0 baseline).

Following Berrar, Lopes & Dubitzky (2024), predictive features are derived
from the historical performance of the competing teams. For each match,
features are computed strictly from PREVIOUS matches (no leakage):

    - rolling win rate, goals scored/conceded, goal difference
    - rolling recent form over configurable windows (5 / 10 / 20 matches)
    - rest days since the team's previous match
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


class TeamHistoryBuffer:
    """Per-team rolling history of recent match results."""

    def __init__(self, max_len: int = 20):
        self.max_len = max_len
        self._history: dict[str, deque] = {}
        self._last_date: dict[str, pd.Timestamp] = {}

    def stats(self, team: str, window: int) -> dict | None:
        """Rolling stats over the last `window` matches (None if insufficient)."""
        hist = self._history.get(team)
        if hist is None or len(hist) < window:
            return None
        recent = list(hist)[-window:]
        gf = np.mean([m["gf"] for m in recent])
        ga = np.mean([m["ga"] for m in recent])
        wins = np.mean([1.0 if m["res"] == 1 else 0.0 for m in recent])
        draws = np.mean([1.0 if m["res"] == 0.5 else 0.0 for m in recent])
        losses = np.mean([1.0 if m["res"] == 0.0 else 0.0 for m in recent])
        return {
            "gf": gf,
            "ga": ga,
            "gd": gf - ga,
            "win_rate": wins,
            "draw_rate": draws,
            "loss_rate": losses,
        }

    def rest_days(self, team: str, date: pd.Timestamp) -> float:
        last = self._last_date.get(team)
        if last is None:
            return np.nan  # no previous match; imputed downstream
        return float((date - last).days)

    def record(
        self,
        team: str,
        date: pd.Timestamp,
        gf: int,
        ga: int,
        result: float,
    ) -> None:
        if team not in self._history:
            self._history[team] = deque(maxlen=self.max_len)
        self._history[team].append({"gf": gf, "ga": ga, "res": result})
        self._last_date[team] = date


def build_team_form_features(
    matches: pd.DataFrame, form_windows: list[int] = (5, 10, 20)
) -> pd.DataFrame:
    """Compute pre-match team form features for a chronological match table.

    Returns a DataFrame aligned with `matches`; columns prefixed
    `home_` / `away_` for each window, e.g. home_form5_gd.
    """
    max_window = max(form_windows)
    buf = TeamHistoryBuffer(max_len=max_window)
    rows = []

    for row in matches.itertuples(index=False):
        feats: dict = {}
        res_home = 1.0 if row.home_goals > row.away_goals else (
            0.5 if row.home_goals == row.away_goals else 0.0
        )
        res_away = 1.0 - res_home

        for team, gf, ga, res, prefix in (
            (row.home_team, row.home_goals, row.away_goals, res_home, "home"),
            (row.away_team, row.away_goals, row.home_goals, res_away, "away"),
        ):
            for w in form_windows:
                s = buf.stats(team, w)
                if s is None:
                    feats[f"{prefix}_form{w}_gf"] = np.nan
                    feats[f"{prefix}_form{w}_ga"] = np.nan
                    feats[f"{prefix}_form{w}_gd"] = np.nan
                    feats[f"{prefix}_form{w}_win"] = np.nan
                else:
                    feats[f"{prefix}_form{w}_gf"] = s["gf"]
                    feats[f"{prefix}_form{w}_ga"] = s["ga"]
                    feats[f"{prefix}_form{w}_gd"] = s["gd"]
                    feats[f"{prefix}_form{w}_win"] = s["win_rate"]
            feats[f"{prefix}_rest_days"] = buf.rest_days(team, row.date)

        rows.append(feats)

        buf.record(row.home_team, row.date, row.home_goals, row.away_goals, res_home)
        buf.record(row.away_team, row.date, row.away_goals, row.home_goals, res_away)

    return pd.DataFrame(rows, index=matches.index)


def build_match_context_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Simple match-context features (Phase 7 subset available from Dataset A).

    NOTE: only pre-match information is used here. Goals are deliberately
    excluded (they are the target, not context).
    """
    ctx = pd.DataFrame(index=matches.index)
    ctx["is_neutral"] = matches["neutral"].astype(float)
    ctx["is_friendly"] = (matches["tournament"] == "Friendly").astype(float)
    return ctx


def build_m0_feature_matrix(
    matches: pd.DataFrame,
    strength_features: pd.DataFrame,
    form_windows: list[int] = (5, 10, 20),
) -> pd.DataFrame:
    """Assemble the full M0 feature matrix: strength + form + context."""
    form = build_team_form_features(matches, form_windows)
    ctx = build_match_context_features(matches)
    X = pd.concat([strength_features, form, ctx], axis=1)

    # Impute missing form values (early matches) with the column median.
    X = X.fillna(X.median(numeric_only=True))
    # Any all-NaN columns (degenerate) -> 0.
    X = X.fillna(0.0)
    return X