"""Chronological Sequence Builder for Temporal Team State Modeling.

Constructs 3D tensors of historical match sequences per team with zero temporal leakage.
Guarantees that for every match at index t, all sequence timesteps strictly precede t.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from src.features.strength import StrengthTracker, UpdaterConfig


# Feature definitions per historical timestep
TIMESTEP_FEATURE_NAMES = [
    # 1. Elo & Strength
    "team_elo",
    "opp_elo",
    "elo_diff",
    "adaptive_volatility",
    "expected_score",
    # 2. Performance
    "goals_for",
    "goals_against",
    "goal_diff",
    "result_numeric",
    "points",
    "clean_sheet",
    "failed_to_score",
    "xg_for",
    "xg_against",
    "shots",
    "shots_on_target",
    "possession",
    "pass_accuracy",
    # 3. Match Context
    "rest_days_norm",
    "tournament_tier",
    "is_competitive",
    "is_home_venue",
    "opp_strength_ratio",
    # 4. Squad & FIFA Ratings
    "squad_top5_ovr",
    "squad_xi_ovr",
    "squad_depth_ovr",
    "squad_avg_age",
    "lineup_continuity",
]


@dataclass
class HistoricalMatchRecord:
    """Historical match record stored in team chronological history buffer."""
    match_idx: int
    date: pd.Timestamp
    team: str
    opp: str
    is_home: bool
    is_competitive: bool
    tournament_tier: int
    gf: float
    ga: float
    team_elo: float
    opp_elo: float
    volatility: float
    rest_days: float
    fifa_stats: dict[str, float]


class ChronologicalSequenceBuilder:
    """Builds historical match sequence tensors for all teams over chronological matches."""

    def __init__(
        self,
        max_seq_len: int = 20,
        updater_cfg: UpdaterConfig | None = None,
        fifa_lookup: dict | None = None,
    ):
        self.max_seq_len = max_seq_len
        self.updater_cfg = updater_cfg or UpdaterConfig(
            mode="adaptive", k=24.0, base_cap=0.02, max_cap=0.075,
            consistency_weight=3.0, surprise_weight=1.0, evidence_window=3
        )
        self.fifa_lookup = fifa_lookup or {}

        # team -> deque of HistoricalMatchRecord
        self._history: dict[str, deque[HistoricalMatchRecord]] = {}
        self._last_match_date: dict[str, pd.Timestamp] = {}
        self.feature_dim = len(TIMESTEP_FEATURE_NAMES)

    def _convert_record_to_vector(self, rec: HistoricalMatchRecord) -> np.ndarray:
        """Convert a historical match record to a normalized numerical vector."""
        # 1. Elo & Strength
        team_elo_norm = (rec.team_elo - 1500.0) / 400.0
        opp_elo_norm = (rec.opp_elo - 1500.0) / 400.0
        elo_diff = (rec.team_elo - rec.opp_elo) / 400.0
        expected_score = 1.0 / (1.0 + np.exp(-elo_diff))
        volatility = rec.volatility / 0.05

        # 2. Performance
        gf = rec.gf
        ga = rec.ga
        gd = gf - ga
        res = 1.0 if gf > ga else (0.5 if gf == ga else 0.0)
        pts = 3.0 if res == 1.0 else (1.0 if res == 0.5 else 0.0)
        cs = 1.0 if ga == 0 else 0.0
        fts = 1.0 if gf == 0 else 0.0

        # Estimated xG and advanced stats
        xg_f = 0.5 + 0.3 * gf + 0.4 * expected_score
        xg_a = 0.5 + 0.3 * ga + 0.4 * (1.0 - expected_score)
        shots = 3.0 + 3.0 * gf + 5.0 * expected_score
        sot = 1.0 + 1.5 * gf + 2.5 * expected_score
        poss = 0.5 + 0.15 * (expected_score - 0.5) * 2.0
        pass_acc = 0.75 + 0.10 * (expected_score - 0.5) * 2.0

        # 3. Context
        rest_norm = np.log1p(min(rec.rest_days, 180.0)) / np.log1p(180.0)
        t_tier = float(rec.tournament_tier) / 4.0
        is_comp = 1.0 if rec.is_competitive else 0.0
        is_home = 1.0 if rec.is_home else 0.0
        opp_ratio = rec.opp_elo / max(500.0, rec.team_elo)

        # 4. Squad / FIFA
        f = rec.fifa_stats
        top5 = (f.get("top5_ovr", 75.0) - 75.0) / 15.0
        xi = (f.get("xi_ovr", 75.0) - 75.0) / 15.0
        depth = (f.get("depth_ovr", 70.0) - 70.0) / 15.0
        age = (f.get("age_mean", 26.5) - 26.5) / 5.0
        continuity = 0.85 if is_comp > 0.5 else 0.65

        vec = np.array([
            team_elo_norm, opp_elo_norm, elo_diff, volatility, expected_score,
            gf, ga, gd, res, pts, cs, fts, xg_f, xg_a, shots, sot, poss, pass_acc,
            rest_norm, t_tier, is_comp, is_home, opp_ratio,
            top5, xi, depth, age, continuity
        ], dtype=np.float32)

        return vec

    def get_sequence(self, team: str, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Get chronological sequence matrix of shape (seq_len, feature_dim) and mask (seq_len)."""
        hist = self._history.get(team)
        seq = np.zeros((seq_len, self.feature_dim), dtype=np.float32)
        mask = np.zeros(seq_len, dtype=np.float32)

        if hist is None or len(hist) == 0:
            return seq, mask

        recent = list(hist)[-min(len(hist), seq_len):]
        n = len(recent)
        
        # Right-aligned padding (recent items at the end of the sequence)
        start_idx = seq_len - n
        for i, rec in enumerate(recent):
            seq[start_idx + i] = self._convert_record_to_vector(rec)
            mask[start_idx + i] = 1.0

        return seq, mask

    def build_all_sequences(
        self,
        matches: pd.DataFrame,
        seq_len: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process all matches chronologically and generate sequence tensors.

        Returns:
            seq_home: (N, seq_len, feature_dim)
            mask_home: (N, seq_len)
            seq_away: (N, seq_len, feature_dim)
            mask_away: (N, seq_len)
        """
        n_matches = len(matches)
        seq_home = np.zeros((n_matches, seq_len, self.feature_dim), dtype=np.float32)
        mask_home = np.zeros((n_matches, seq_len), dtype=np.float32)
        seq_away = np.zeros((n_matches, seq_len, self.feature_dim), dtype=np.float32)
        mask_away = np.zeros((n_matches, seq_len), dtype=np.float32)

        # Reset history
        self._history.clear()
        self._last_match_date.clear()
        tracker = StrengthTracker(self.updater_cfg)

        matches_dt = pd.to_datetime(matches["date"])
        home_teams = matches["home_team"].to_numpy()
        away_teams = matches["away_team"].to_numpy()
        home_col = "home_goals" if "home_goals" in matches.columns else "home_score"
        away_col = "away_goals" if "away_goals" in matches.columns else "away_score"
        home_scores = matches[home_col].to_numpy()
        away_scores = matches[away_col].to_numpy()
        neutrals = matches["neutral"].to_numpy()
        tournaments = matches["tournament"].to_numpy()

        is_comp_arr = (~matches["tournament"].str.lower().str.contains("friendly")).to_numpy()
        tier_arr = np.where(
            matches["tournament"].str.contains("FIFA World Cup", na=False), 1,
            np.where(matches["tournament"].str.contains("UEFA Euro|Copa América", na=False), 2,
            np.where(is_comp_arr, 3, 4))
        )
        years = matches_dt.dt.year.to_numpy()

        for i in range(n_matches):
            h_team = home_teams[i]
            a_team = away_teams[i]
            cur_date = matches_dt.iloc[i]
            is_neutral = bool(neutrals[i])
            is_comp = bool(is_comp_arr[i])
            tier = int(tier_arr[i])
            yr = int(years[i])

            # 1. PRE-MATCH RETRIEVAL (Zero Temporal Leakage Guarantee)
            sh, mh = self.get_sequence(h_team, seq_len)
            sa, ma = self.get_sequence(a_team, seq_len)

            seq_home[i] = sh
            mask_home[i] = mh
            seq_away[i] = sa
            mask_away[i] = ma

            # 2. POST-MATCH UPDATE
            h_score = float(home_scores[i])
            a_score = float(away_scores[i])

            h_elo_before = tracker.rating(h_team)
            a_elo_before = tracker.rating(a_team)
            h_volatility = tracker._adaptive_cap_points(h_team) / 400.0
            a_volatility = tracker._adaptive_cap_points(a_team) / 400.0

            # Rest days
            last_h_date = self._last_match_date.get(h_team, cur_date - pd.Timedelta(days=180))
            last_a_date = self._last_match_date.get(a_team, cur_date - pd.Timedelta(days=180))
            h_rest = max(1.0, (cur_date - last_h_date).days)
            a_rest = max(1.0, (cur_date - last_a_date).days)

            # FIFA lookups
            f_h = self.fifa_lookup.get((h_team, yr), {})
            f_a = self.fifa_lookup.get((a_team, yr), {})

            # Record for home team
            rec_h = HistoricalMatchRecord(
                match_idx=i,
                date=cur_date,
                team=h_team,
                opp=a_team,
                is_home=not is_neutral,
                is_competitive=is_comp,
                tournament_tier=tier,
                gf=h_score,
                ga=a_score,
                team_elo=h_elo_before,
                opp_elo=a_elo_before,
                volatility=h_volatility,
                rest_days=h_rest,
                fifa_stats=f_h,
            )

            # Record for away team
            rec_a = HistoricalMatchRecord(
                match_idx=i,
                date=cur_date,
                team=a_team,
                opp=h_team,
                is_home=False,
                is_competitive=is_comp,
                tournament_tier=tier,
                gf=a_score,
                ga=h_score,
                team_elo=a_elo_before,
                opp_elo=h_elo_before,
                volatility=a_volatility,
                rest_days=a_rest,
                fifa_stats=f_a,
            )

            if h_team not in self._history:
                self._history[h_team] = deque(maxlen=self.max_seq_len)
            self._history[h_team].append(rec_h)
            self._last_match_date[h_team] = cur_date

            if a_team not in self._history:
                self._history[a_team] = deque(maxlen=self.max_seq_len)
            self._history[a_team].append(rec_a)
            self._last_match_date[a_team] = cur_date

            # Update strength tracker
            tracker.update(h_team, a_team, int(h_score), int(a_score), neutral=is_neutral)

        return seq_home, mask_home, seq_away, mask_away
