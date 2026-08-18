"""Continuous Tactical Identity & Matchup Compatibility Engine.

Computes multi-dimensional continuous tactical style representations for soccer teams
across historical rolling windows and EWMA filters with zero temporal leakage.
Calculates dynamic tactical matchup features and style interactions.
"""

from __future__ import annotations

from collections import deque
import numpy as np
import pandas as pd


# 11 Continuous Numerical Tactical Dimensions
TACTICAL_DIMENSIONS = [
    "possession_control",
    "pressing_intensity",
    "directness",
    "progression_rate",
    "passing_accuracy",
    "crossing_frequency",
    "centrality",
    "transition_speed",
    "shot_creation",
    "defensive_activity_height",
    "build_up_tempo",
]


class ContinuousTacticalTracker:
    """Chronological tracker computing continuous tactical profiles with zero lookahead."""

    def __init__(
        self,
        windows: list[int] = (5, 10, 20),
        alphas: list[float] = (0.1, 0.2, 0.3),
    ):
        self.windows = sorted(windows)
        self.alphas = sorted(alphas)
        self.max_len = max(windows)

        # team -> deque of historical per-match tactical profiles
        self._history: dict[str, deque] = {}
        # team -> dict of EWMA states for each alpha
        self._ewma: dict[str, dict[float, np.ndarray]] = {}

    def _estimate_match_tactical_vector(
        self,
        gf: float,
        ga: float,
        is_home: bool,
        is_competitive: bool,
        tournament_tier: int,
        opp_elo: float,
        team_elo: float,
    ) -> np.ndarray:
        """Estimate continuous tactical vector for a single completed match.

        Values are continuous in [0.0, 1.0].
        """
        elo_diff = team_elo - opp_elo
        elo_factor = 1.0 / (1.0 + np.exp(-elo_diff / 400.0))  # expected dominance

        # 1. Possession control
        possession = 0.50 + 0.15 * (elo_factor - 0.5) * 2.0 + (0.05 if is_home else -0.05)
        possession = float(np.clip(possession + 0.04 * (gf - ga), 0.25, 0.75))

        # 2. Pressing intensity
        pressing = 0.50 + 0.12 * (1.0 - elo_factor) + (0.08 if is_competitive else -0.05)
        pressing = float(np.clip(pressing + 0.05 * (ga - 1.0), 0.20, 0.85))

        # 3. Directness (higher when underdogs or away)
        directness = 0.50 - 0.15 * (possession - 0.5) * 2.0 + (0.06 if not is_home else -0.04)
        directness = float(np.clip(directness, 0.20, 0.85))

        # 4. Progression rate
        progression = 0.50 + 0.10 * (elo_factor - 0.5) * 2.0 + 0.06 * gf
        progression = float(np.clip(progression, 0.20, 0.85))

        # 5. Passing accuracy
        pass_acc = 0.75 + 0.10 * (elo_factor - 0.5) * 2.0 + (0.03 if is_home else -0.03)
        pass_acc = float(np.clip(pass_acc, 0.55, 0.92))

        # 6. Crossing frequency
        crossing = 0.50 + 0.08 * (1.0 - directness) + 0.05 * (1 if is_home else 0)
        crossing = float(np.clip(crossing, 0.20, 0.80))

        # 7. Centrality
        centrality = 0.50 + 0.10 * (possession - 0.5) * 2.0 - 0.08 * crossing
        centrality = float(np.clip(centrality, 0.20, 0.80))

        # 8. Transition speed
        transition = 0.50 + 0.15 * directness - 0.10 * (possession - 0.5) * 2.0
        transition = float(np.clip(transition, 0.20, 0.85))

        # 9. Shot creation
        shots = 0.50 + 0.15 * (elo_factor - 0.5) * 2.0 + 0.08 * gf
        shots = float(np.clip(shots, 0.15, 0.90))

        # 10. Defensive activity height
        def_height = 0.50 + 0.18 * (possession - 0.5) * 2.0 + (0.06 if is_home else -0.06)
        def_height = float(np.clip(def_height, 0.20, 0.85))

        # 11. Build-up tempo
        tempo = 0.50 + 0.10 * (1.0 - directness) + (0.05 if is_competitive else -0.05)
        tempo = float(np.clip(tempo, 0.20, 0.80))

        return np.array([
            possession, pressing, directness, progression, pass_acc,
            crossing, centrality, transition, shots, def_height, tempo
        ], dtype=np.float32)

    def get_tactical_profile(self, team: str) -> dict[str, float]:
        """Retrieve pre-match tactical profile over rolling windows & EWMA filters."""
        hist = self._history.get(team)
        feats = {}

        d_dim = len(TACTICAL_DIMENSIONS)
        default_vec = np.full(d_dim, 0.5, dtype=np.float32)

        # 1. Rolling window profiles
        for w in self.windows:
            if hist is None or len(hist) < 2:
                mean_vec = default_vec
            else:
                recent = list(hist)[-min(len(hist), w):]
                mean_vec = np.mean(recent, axis=0)

            for d_idx, d_name in enumerate(TACTICAL_DIMENSIONS):
                feats[f"tactical_{d_name}_w{w}"] = float(mean_vec[d_idx])

        # 2. EWMA profiles
        for a in self.alphas:
            ewma_vec = self._ewma.get(team, {}).get(a, default_vec)
            for d_idx, d_name in enumerate(TACTICAL_DIMENSIONS):
                feats[f"tactical_{d_name}_ewma_{int(a*100)}"] = float(ewma_vec[d_idx])

        return feats

    def get_tactical_vector(self, team: str, window: int = 10) -> np.ndarray:
        """Get summary tactical vector of length 11 for matchup computations."""
        hist = self._history.get(team)
        if hist is None or len(hist) < 2:
            return np.full(len(TACTICAL_DIMENSIONS), 0.5, dtype=np.float32)
        recent = list(hist)[-min(len(hist), window):]
        return np.mean(recent, axis=0).astype(np.float32)

    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: float,
        away_score: float,
        is_competitive: bool,
        tournament_tier: int,
        home_elo: float,
        away_elo: float,
        is_neutral: bool = False,
    ):
        """Update historical tactical states after match completion."""
        h_vec = self._estimate_match_tactical_vector(
            gf=home_score,
            ga=away_score,
            is_home=not is_neutral,
            is_competitive=is_competitive,
            tournament_tier=tournament_tier,
            opp_elo=away_elo,
            team_elo=home_elo,
        )

        a_vec = self._estimate_match_tactical_vector(
            gf=away_score,
            ga=home_score,
            is_home=False,
            is_competitive=is_competitive,
            tournament_tier=tournament_tier,
            opp_elo=home_elo,
            team_elo=away_elo,
        )

        # Update deque history
        for t, vec in [(home_team, h_vec), (away_team, a_vec)]:
            if t not in self._history:
                self._history[t] = deque(maxlen=self.max_len)
            self._history[t].append(vec)

            # Update EWMA
            if t not in self._ewma:
                self._ewma[t] = {}
            for a in self.alphas:
                if a not in self._ewma[t]:
                    self._ewma[t][a] = vec.copy()
                else:
                    self._ewma[t][a] = a * vec + (1.0 - a) * self._ewma[t][a]


def compute_tactical_matchup_features(
    t_home: np.ndarray,
    t_away: np.ndarray,
) -> dict[str, float]:
    """Compute rich continuous tactical matchup interactions between two teams."""
    diff = t_home - t_away
    dist = float(np.linalg.norm(diff))
    
    # Cosine similarity
    norm_h = float(np.linalg.norm(t_home)) + 1e-8
    norm_a = float(np.linalg.norm(t_away)) + 1e-8
    cos_sim = float(np.dot(t_home, t_away) / (norm_h * norm_a))

    # Tactical dimensions indices:
    # 0: possession_control, 1: pressing_intensity, 2: directness, 3: progression_rate,
    # 4: passing_accuracy, 5: crossing_frequency, 6: centrality, 7: transition_speed,
    # 8: shot_creation, 9: defensive_activity_height, 10: build_up_tempo

    # Specific tactical compatibilities:
    # 1. Attack vs Defence compatibility
    home_att_vs_away_def = float(t_home[8] * (1.0 - t_away[9]))
    away_att_vs_home_def = float(t_away[8] * (1.0 - t_home[9]))
    att_def_ratio = float(home_att_vs_away_def - away_att_vs_home_def)

    # 2. Pressing vs Build-up compatibility (High press vs slow build-up exploit)
    home_press_vs_away_buildup = float(t_home[1] * t_away[10])
    away_press_vs_home_buildup = float(t_away[1] * t_home[10])
    press_buildup_mismatch = float(home_press_vs_away_buildup - away_press_vs_home_buildup)

    # 3. Possession vs Transition compatibility (Possession vs Counter-attack risk)
    home_poss_vs_away_counter = float(t_home[0] * t_away[7])
    away_poss_vs_home_counter = float(t_away[0] * t_home[7])
    poss_counter_dynamic = float(home_poss_vs_away_counter - away_poss_vs_home_counter)

    # 4. Style mismatch magnitude
    style_mismatch = float(abs(t_home[0] - t_away[0]) + abs(t_home[2] - t_away[2]) + abs(t_home[1] - t_away[1]))

    feats = {
        "tactical_euclidean_distance": dist,
        "tactical_cosine_similarity": cos_sim,
        "tactical_style_mismatch": style_mismatch,
        "home_att_vs_away_def": home_att_vs_away_def,
        "away_att_vs_home_def": away_att_vs_home_def,
        "att_def_ratio": att_def_ratio,
        "home_press_vs_away_buildup": home_press_vs_away_buildup,
        "away_press_vs_home_buildup": away_press_vs_home_buildup,
        "press_buildup_mismatch": press_buildup_mismatch,
        "home_poss_vs_away_counter": home_poss_vs_away_counter,
        "away_poss_vs_home_counter": away_poss_vs_home_counter,
        "poss_counter_dynamic": poss_counter_dynamic,
    }

    # Add directional differences per dimension
    for d_idx, d_name in enumerate(TACTICAL_DIMENSIONS):
        feats[f"tactical_diff_{d_name}"] = float(diff[d_idx])
        feats[f"tactical_abs_diff_{d_name}"] = float(abs(diff[d_idx]))

    return feats
