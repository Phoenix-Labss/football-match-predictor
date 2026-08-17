"""Base Paper (Berrar, Lopes & Dubitzky, 2024) M0 Feature Pipeline.

Implements the exact domain-knowledge and historical-window feature set from the
Springer 2024 baseline paper:
- Rolling win rate, draw rate, loss rate over historical windows W in {5, 10, 20}
- Rolling mean goals scored (GF) and goals conceded (GA)
- Rolling goal difference (GD = GF - GA)
- Rest days since previous competitive match
- Standard World-Football-Elo rating prior to kickoff
- Pre-match home advantage flag

Strict Zero Temporal Leakage Guarantee:
All features for match t are computed using strictly matches < t.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd

from src.features.strength import UpdaterConfig, StrengthTracker, RATING_SCALE


class BasePaperHistoryBuffer:
    """Historical match buffer tracking rolling window performance per team."""

    def __init__(self, windows: list[int] = (5, 10, 20)):
        self.windows = sorted(windows)
        self.max_len = max(windows)
        self._history: dict[str, deque] = {}
        self._last_date: dict[str, pd.Timestamp] = {}

    def get_stats(self, team: str, w: int) -> dict:
        hist = self._history.get(team)
        if hist is None or len(hist) < 1:
            return {
                f"win_rate_{w}": 0.333,
                f"draw_rate_{w}": 0.333,
                f"loss_rate_{w}": 0.333,
                f"gf_mean_{w}": 1.25,
                f"ga_mean_{w}": 1.25,
                f"gd_mean_{w}": 0.0,
            }
        recent = list(hist)[-w:]
        k = len(recent)
        gf = np.mean([m["gf"] for m in recent])
        ga = np.mean([m["ga"] for m in recent])
        wins = np.mean([1.0 if m["res"] == 1.0 else 0.0 for m in recent])
        draws = np.mean([1.0 if m["res"] == 0.5 else 0.0 for m in recent])
        losses = np.mean([1.0 if m["res"] == 0.0 else 0.0 for m in recent])
        return {
            f"win_rate_{w}": float(wins),
            f"draw_rate_{w}": float(draws),
            f"loss_rate_{w}": float(losses),
            f"gf_mean_{w}": float(gf),
            f"ga_mean_{w}": float(ga),
            f"gd_mean_{w}": float(gf - ga),
        }

    def get_rest_days(self, team: str, match_date: pd.Timestamp) -> float:
        last = self._last_date.get(team)
        if last is None:
            return 30.0  # Median rest day imputation
        return min(float((match_date - last).days), 180.0)

    def record(self, team: str, match_date: pd.Timestamp, gf: int, ga: int, res: float):
        if team not in self._history:
            self._history[team] = deque(maxlen=self.max_len)
        self._history[team].append({"gf": gf, "ga": ga, "res": res})
        self._last_date[team] = match_date


def build_base_paper_m0_features(
    matches: pd.DataFrame,
    form_windows: list[int] = (5, 10, 20),
) -> pd.DataFrame:
    """Build the canonical M0 feature matrix from Berrar et al. (2024)."""
    # Track standard Elo
    elo_cfg = UpdaterConfig(mode="fixed_k", k=24.0, initial_rating=1500.0, home_advantage=65.0)
    tracker = StrengthTracker(elo_cfg)

    buf = BasePaperHistoryBuffer(windows=form_windows)
    records = []

    for idx, row in matches.iterrows():
        date = pd.to_datetime(row["date"])
        h = str(row["home_team"])
        a = str(row["away_team"])
        neutral = bool(row["neutral"])

        # 1. Pre-match Elo features (before match t is processed)
        pre = tracker.pre_match_state(h, a, neutral)
        e_h = pre["expected_home_score"]
        e_a = 1.0 - e_h
        elo_diff = pre["elo_diff"] + (0.0 if neutral else elo_cfg.home_advantage)

        row_feats = {
            "elo_home": pre["elo_home"] / RATING_SCALE,
            "elo_away": pre["elo_away"] / RATING_SCALE,
            "elo_diff": elo_diff / RATING_SCALE,
            "elo_exp_home": e_h,
            "elo_exp_away": e_a,
            "is_neutral": 1.0 if neutral else 0.0,
            "rest_days_home": buf.get_rest_days(h, date),
            "rest_days_away": buf.get_rest_days(a, date),
            "rest_days_diff": buf.get_rest_days(h, date) - buf.get_rest_days(a, date),
        }

        # 2. Form features over windows (5, 10, 20)
        for w in form_windows:
            h_stats = buf.get_stats(h, w)
            a_stats = buf.get_stats(a, w)
            for k_stat in ["win_rate", "draw_rate", "loss_rate", "gf_mean", "ga_mean", "gd_mean"]:
                row_feats[f"home_{k_stat}_{w}"] = h_stats[f"{k_stat}_{w}"]
                row_feats[f"away_{k_stat}_{w}"] = a_stats[f"{k_stat}_{w}"]
                row_feats[f"diff_{k_stat}_{w}"] = h_stats[f"{k_stat}_{w}"] - a_stats[f"{k_stat}_{w}"]

        records.append(row_feats)

        # 3. Post-match state update for future matches
        gh = int(row["home_goals"] if "home_goals" in row else row["home_score"])
        ga = int(row["away_goals"] if "away_goals" in row else row["away_score"])
        tracker.update(h, a, gh, ga, neutral=neutral)

        h_res = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        a_res = 1.0 - h_res
        buf.record(h, date, gh, ga, h_res)
        buf.record(a, date, ga, gh, a_res)

    return pd.DataFrame(records)
