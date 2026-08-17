"""Match-Day State Simulation Engine (Track 3 / Realism Upgrade).

Implements stochastic match-day performance realization for players and teams:
    - Base static player profile & FIFA attributes
    - Age modifier & positional fit
    - Controlled match-day form realization
    - Optional pre-kickoff fatigue factor
    - Performance volatility (star consistency vs inconsistent variance)
    - Correlated team-level execution factors (attack, midfield, defence, cohesion)
    - Recomputed team ratings & Dixon-Coles xG per Monte Carlo realization

When `enabled=False` (default), the simulator falls back strictly to the
deterministic pre-match ratings without altering any baseline behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
from scipy.stats import poisson

from .player_model import PlayerState
from .squad_model import TeamRating, FORMATIONS
from .match_engine import RHO


@dataclass
class MatchDayStateConfig:
    """Configuration options for match-day state simulation."""

    enabled: bool = False
    player_form_sigma: float = 0.02
    player_perf_sigma: float = 0.02
    team_execution_sigma: float = 0.02
    team_cohesion_sigma: float = 0.02
    fatigue_enabled: bool = False
    star_stability_factor: float = 0.75       # players >= 88 OVR have reduced volatility
    inconsistent_volatility_factor: float = 1.25 # players <= 78 OVR have increased volatility
    min_multiplier: float = 0.85
    max_multiplier: float = 1.15
    seed: int | None = 42


@dataclass
class PlayerMatchState:
    """Snapshot of a single player's stochastic match-day realization."""

    player_id: int
    name: str
    slot_group: str
    slot_label: str
    base_ability: float
    form_factor: float
    fatigue_factor: float
    performance_factor: float
    team_exec_factor: float
    total_multiplier: float
    simulated_ability: float
    effective_contribution: float


@dataclass
class TeamMatchState:
    """Stochastic match-day realization of a team's tactical units."""

    team: str
    formation: str
    attack_execution: float
    midfield_execution: float
    defensive_execution: float
    cohesion_factor: float
    simulated_attack: float
    simulated_midfield: float
    simulated_defence: float
    simulated_gk: float
    simulated_chemistry: float
    player_states: list[PlayerMatchState] = field(default_factory=list)

    def to_team_rating(self) -> TeamRating:
        """Convert realized match-day state into a TeamRating object."""
        return TeamRating(
            team=self.team,
            formation=self.formation,
            attack=float(self.simulated_attack),
            midfield=float(self.simulated_midfield),
            defence=float(self.simulated_defence),
            gk=float(self.simulated_gk),
            chemistry=float(self.simulated_chemistry),
            manager_attack=0.0,
            manager_defence=0.0,
            lineup=[p.player_id for p in self.player_states],
        )


class MatchDayStateSampler:
    """Stochastic sampler for match-day player and team realizations."""

    def __init__(
        self,
        config: MatchDayStateConfig | None = None,
        seed: int | None = 42,
    ):
        self.config = config or MatchDayStateConfig()
        if seed is not None:
            self.config.seed = seed
        self.rng = np.random.default_rng(self.config.seed)

    def _get_player_volatility(self, base_ability: float) -> float:
        """Derive player performance volatility based on quality tier."""
        base_sigma = self.config.player_perf_sigma
        if base_ability >= 88.0:
            return base_sigma * self.config.star_stability_factor
        elif base_ability <= 78.0:
            return base_sigma * self.config.inconsistent_volatility_factor
        return base_sigma

    def sample_team_match_state(
        self,
        team: str,
        lineup_pairs: list[tuple[PlayerState, float]],
        slots: list[tuple[str, str]],
        base_chemistry: float,
        formation: str = "4-3-3",
        fatigue_map: dict[int, float] | None = None,
    ) -> TeamMatchState:
        """Sample a single stochastic match-day state for a team starting XI."""
        cfg = self.config
        fatigue_map = fatigue_map or {}

        if not cfg.enabled:
            # Deterministic baseline fallback
            return self._deterministic_fallback(
                team, lineup_pairs, slots, base_chemistry, formation
            )

        # 1. Sample Team-Level Execution Factors (Correlated Across Units)
        atk_exec = float(np.clip(self.rng.normal(1.0, cfg.team_execution_sigma), 0.90, 1.10))
        mid_exec = float(np.clip(self.rng.normal(1.0, cfg.team_execution_sigma), 0.90, 1.10))
        def_exec = float(np.clip(self.rng.normal(1.0, cfg.team_execution_sigma), 0.90, 1.10))
        cohesion = float(np.clip(self.rng.normal(0.0, cfg.team_cohesion_sigma), -0.05, 0.05))

        sim_chemistry = float(np.clip(base_chemistry + cohesion, 0.0, 1.0))

        # 2. Sample Player-Level Match-Day Factors
        player_states: list[PlayerMatchState] = []
        att_vals, mid_vals, def_vals, gk_vals = [], [], [], []

        for (player, fit), (group, label) in zip(lineup_pairs, slots):
            volatility = self._get_player_volatility(player.ability)
            
            form_factor = float(np.clip(self.rng.normal(1.0, cfg.player_form_sigma), 0.92, 1.08))
            perf_factor = float(np.clip(self.rng.normal(1.0, volatility), 0.92, 1.08))

            if cfg.fatigue_enabled:
                raw_fatigue = fatigue_map.get(player.sofifa_id, 1.0)
                fatigue_factor = float(np.clip(raw_fatigue, 0.90, 1.0))
            else:
                fatigue_factor = 1.0

            # Unit-specific team factor coupling
            if group == "ATT":
                team_factor = atk_exec
            elif group == "MID":
                team_factor = mid_exec
            elif group == "DEF":
                team_factor = def_exec
            else:  # GK
                team_factor = float(np.sqrt(def_exec))

            total_multiplier = float(
                np.clip(
                    form_factor * fatigue_factor * perf_factor * team_factor,
                    cfg.min_multiplier,
                    cfg.max_multiplier,
                )
            )

            sim_ability = float(player.ability * total_multiplier)

            # Recalculate channel contribution with match-day multiplier
            if group == "GK":
                base_gk = player.gk_ability if player.gk_ability else (player.ability * 0.8)
                contrib = float(base_gk * fit * total_multiplier)
                gk_vals.append(contrib)
            elif group == "DEF":
                base_def = 0.60 * player.defending + 0.25 * player.physical + 0.15 * player.ability
                contrib = float(base_def * fit * total_multiplier)
                def_vals.append(contrib)
            elif group == "MID":
                base_mid = 0.35 * player.passing + 0.25 * player.dribbling + 0.20 * player.defending + 0.20 * player.ability
                contrib = float(base_mid * fit * total_multiplier)
                mid_vals.append(contrib)
            else:  # ATT
                base_att = 0.40 * player.shooting + 0.25 * player.dribbling + 0.20 * player.pace + 0.15 * player.ability
                contrib = float(base_att * fit * total_multiplier)
                att_vals.append(contrib)

            player_states.append(
                PlayerMatchState(
                    player_id=player.sofifa_id,
                    name=player.name,
                    slot_group=group,
                    slot_label=label,
                    base_ability=player.ability,
                    form_factor=form_factor,
                    fatigue_factor=fatigue_factor,
                    performance_factor=perf_factor,
                    team_exec_factor=team_factor,
                    total_multiplier=total_multiplier,
                    simulated_ability=sim_ability,
                    effective_contribution=contrib,
                )
            )

        fallback = float(np.mean([p.ability for p, _ in lineup_pairs])) if lineup_pairs else 75.0

        return TeamMatchState(
            team=team,
            formation=formation,
            attack_execution=atk_exec,
            midfield_execution=mid_exec,
            defensive_execution=def_exec,
            cohesion_factor=cohesion,
            simulated_attack=float(np.mean(att_vals) if att_vals else fallback),
            simulated_midfield=float(np.mean(mid_vals) if mid_vals else fallback),
            simulated_defence=float(np.mean(def_vals) if def_vals else fallback),
            simulated_gk=float(np.mean(gk_vals) if gk_vals else fallback * 0.8),
            simulated_chemistry=sim_chemistry,
            player_states=player_states,
        )

    def _deterministic_fallback(
        self,
        team: str,
        lineup_pairs: list[tuple[PlayerState, float]],
        slots: list[tuple[str, str]],
        base_chemistry: float,
        formation: str,
    ) -> TeamMatchState:
        """Deterministic baseline reproduction when MatchDayState is disabled."""
        att_vals, mid_vals, def_vals, gk_vals = [], [], [], []
        player_states: list[PlayerMatchState] = []

        for (player, fit), (group, label) in zip(lineup_pairs, slots):
            if group == "GK":
                base_gk = player.gk_ability if player.gk_ability else (player.ability * 0.8)
                contrib = float(base_gk * fit)
                gk_vals.append(contrib)
            elif group == "DEF":
                base_def = 0.60 * player.defending + 0.25 * player.physical + 0.15 * player.ability
                contrib = float(base_def * fit)
                def_vals.append(contrib)
            elif group == "MID":
                base_mid = 0.35 * player.passing + 0.25 * player.dribbling + 0.20 * player.defending + 0.20 * player.ability
                contrib = float(base_mid * fit)
                mid_vals.append(contrib)
            else:  # ATT
                base_att = 0.40 * player.shooting + 0.25 * player.dribbling + 0.20 * player.pace + 0.15 * player.ability
                contrib = float(base_att * fit)
                att_vals.append(contrib)

            player_states.append(
                PlayerMatchState(
                    player_id=player.sofifa_id,
                    name=player.name,
                    slot_group=group,
                    slot_label=label,
                    base_ability=player.ability,
                    form_factor=1.0,
                    fatigue_factor=1.0,
                    performance_factor=1.0,
                    team_exec_factor=1.0,
                    total_multiplier=1.0,
                    simulated_ability=player.ability,
                    effective_contribution=contrib,
                )
            )

        fallback = float(np.mean([p.ability for p, _ in lineup_pairs])) if lineup_pairs else 75.0

        return TeamMatchState(
            team=team,
            formation=formation,
            attack_execution=1.0,
            midfield_execution=1.0,
            defensive_execution=1.0,
            cohesion_factor=0.0,
            simulated_attack=float(np.mean(att_vals) if att_vals else fallback),
            simulated_midfield=float(np.mean(mid_vals) if mid_vals else fallback),
            simulated_defence=float(np.mean(def_vals) if def_vals else fallback),
            simulated_gk=float(np.mean(gk_vals) if gk_vals else fallback * 0.8),
            simulated_chemistry=base_chemistry,
            player_states=player_states,
        )

    # ------------------------------------------------------------------ #
    # High-Performance Vectorized Match-Day Monte Carlo Batch Engine
    # ------------------------------------------------------------------ #
    def simulate_match_batch(
        self,
        team_a: str,
        lineup_a_pairs: list[tuple[PlayerState, float]],
        slots_a: list[tuple[str, str]],
        base_chem_a: float,
        team_b: str,
        lineup_b_pairs: list[tuple[PlayerState, float]],
        slots_b: list[tuple[str, str]],
        base_chem_b: float,
        n_simulations: int = 10000,
        neutral: bool = True,
        home_advantage: float = 0.25,
        baseline_goals: float = 0.55,
        rating_scale: float = 0.02,
        chem_attack_weight: float = 0.15,
        chem_defence_weight: float = 0.10,
        max_goals: int = 8,
    ) -> dict[str, Any]:
        """Perform vectorized N-run Monte Carlo simulation with per-match stochastic states."""
        cfg = self.config
        N = n_simulations

        # 1. Base deterministic contribution vectors (length 11)
        def get_base_contribs(lineup_pairs, slots):
            gk_idx, def_idx, mid_idx, att_idx = [], [], [], []
            base_contrib = np.zeros(11)
            volatilities = np.zeros(11)
            
            for i, ((player, fit), (group, _)) in enumerate(zip(lineup_pairs, slots)):
                volatilities[i] = self._get_player_volatility(player.ability)
                if group == "GK":
                    base_gk = player.gk_ability if player.gk_ability else (player.ability * 0.8)
                    base_contrib[i] = base_gk * fit
                    gk_idx.append(i)
                elif group == "DEF":
                    base_def = 0.60 * player.defending + 0.25 * player.physical + 0.15 * player.ability
                    base_contrib[i] = base_def * fit
                    def_idx.append(i)
                elif group == "MID":
                    base_mid = 0.35 * player.passing + 0.25 * player.dribbling + 0.20 * player.defending + 0.20 * player.ability
                    base_contrib[i] = base_mid * fit
                    mid_idx.append(i)
                else:  # ATT
                    base_att = 0.40 * player.shooting + 0.25 * player.dribbling + 0.20 * player.pace + 0.15 * player.ability
                    base_contrib[i] = base_att * fit
                    att_idx.append(i)
            return base_contrib, volatilities, (gk_idx, def_idx, mid_idx, att_idx)

        base_c_a, vol_a, (gk_a, def_a, mid_a, att_a) = get_base_contribs(lineup_a_pairs, slots_a)
        base_c_b, vol_b, (gk_b, def_b, mid_b, att_b) = get_base_contribs(lineup_b_pairs, slots_b)

        if not cfg.enabled:
            # Deterministic mode: exact fixed ratings across all N runs
            atk_a = np.full(N, np.mean(base_c_a[att_a]))
            mid_a = np.full(N, np.mean(base_c_a[mid_a]))
            dfn_a = np.full(N, np.mean(base_c_a[def_a]))
            gk_a_val = np.full(N, np.mean(base_c_a[gk_a]))
            chem_a = np.full(N, base_chem_a)

            atk_b = np.full(N, np.mean(base_c_b[att_b]))
            mid_b = np.full(N, np.mean(base_c_b[mid_b]))
            dfn_b = np.full(N, np.mean(base_c_b[def_b]))
            gk_b_val = np.full(N, np.mean(base_c_b[gk_b]))
            chem_b = np.full(N, base_chem_b)
        else:
            # Stochastic Match-Day State Mode: Vectorized sampling over N simulations
            # Team execution factors (N,)
            atk_exec_a = np.clip(self.rng.normal(1.0, cfg.team_execution_sigma, size=N), 0.90, 1.10)
            mid_exec_a = np.clip(self.rng.normal(1.0, cfg.team_execution_sigma, size=N), 0.90, 1.10)
            def_exec_a = np.clip(self.rng.normal(1.0, cfg.team_execution_sigma, size=N), 0.90, 1.10)
            cohesion_a = np.clip(self.rng.normal(0.0, cfg.team_cohesion_sigma, size=N), -0.05, 0.05)
            chem_a = np.clip(base_chem_a + cohesion_a, 0.0, 1.0)

            atk_exec_b = np.clip(self.rng.normal(1.0, cfg.team_execution_sigma, size=N), 0.90, 1.10)
            mid_exec_b = np.clip(self.rng.normal(1.0, cfg.team_execution_sigma, size=N), 0.90, 1.10)
            def_exec_b = np.clip(self.rng.normal(1.0, cfg.team_execution_sigma, size=N), 0.90, 1.10)
            cohesion_b = np.clip(self.rng.normal(0.0, cfg.team_cohesion_sigma, size=N), -0.05, 0.05)
            chem_b = np.clip(base_chem_b + cohesion_b, 0.0, 1.0)

            # Player multipliers: (N, 11)
            form_a = np.clip(self.rng.normal(1.0, cfg.player_form_sigma, size=(N, 11)), 0.92, 1.08)
            perf_a = np.clip(self.rng.normal(1.0, vol_a, size=(N, 11)), 0.92, 1.08)
            team_fac_a = np.ones((N, 11))
            team_fac_a[:, att_a] = atk_exec_a[:, None]
            team_fac_a[:, mid_a] = mid_exec_a[:, None]
            team_fac_a[:, def_a] = def_exec_a[:, None]
            team_fac_a[:, gk_a] = np.sqrt(def_exec_a)[:, None]

            mult_a = np.clip(form_a * perf_a * team_fac_a, cfg.min_multiplier, cfg.max_multiplier)
            sim_c_a = base_c_a[None, :] * mult_a

            atk_a = np.mean(sim_c_a[:, att_a], axis=1)
            mid_a = np.mean(sim_c_a[:, mid_a], axis=1)
            dfn_a = np.mean(sim_c_a[:, def_a], axis=1)
            gk_a_val = np.mean(sim_c_a[:, gk_a], axis=1)

            # Team B Player multipliers
            form_b = np.clip(self.rng.normal(1.0, cfg.player_form_sigma, size=(N, 11)), 0.92, 1.08)
            perf_b = np.clip(self.rng.normal(1.0, vol_b, size=(N, 11)), 0.92, 1.08)
            team_fac_b = np.ones((N, 11))
            team_fac_b[:, att_b] = atk_exec_b[:, None]
            team_fac_b[:, mid_b] = mid_exec_b[:, None]
            team_fac_b[:, def_b] = def_exec_b[:, None]
            team_fac_b[:, gk_b] = np.sqrt(def_exec_b)[:, None]

            mult_b = np.clip(form_b * perf_b * team_fac_b, cfg.min_multiplier, cfg.max_multiplier)
            sim_c_b = base_c_b[None, :] * mult_b

            atk_b = np.mean(sim_c_b[:, att_b], axis=1)
            mid_b = np.mean(sim_c_b[:, mid_b], axis=1)
            dfn_b = np.mean(sim_c_b[:, def_b], axis=1)
            gk_b_val = np.mean(sim_c_b[:, gk_b], axis=1)

        # 2. Chemistry adjustments per simulation run
        adj_atk_a = atk_a * (1.0 + chem_attack_weight * chem_a)
        adj_dfn_a = dfn_a * (1.0 + chem_defence_weight * chem_a)
        adj_atk_b = atk_b * (1.0 + chem_attack_weight * chem_b)
        adj_dfn_b = dfn_b * (1.0 + chem_defence_weight * chem_b)

        # 3. Differentials and xG (lambda_a, lambda_b) vectors (length N)
        mid_diff = (mid_a - mid_b) * 0.5
        ha = 0.0 if neutral else home_advantage
        gk_penalty_a = 0.30 * (gk_b_val / 100.0)
        gk_penalty_b = 0.30 * (gk_a_val / 100.0)

        raw_log_a = baseline_goals + (adj_atk_a - adj_dfn_b) * rating_scale + mid_diff * rating_scale + ha - gk_penalty_a
        raw_log_b = baseline_goals + (adj_atk_b - adj_dfn_a) * rating_scale - mid_diff * rating_scale - gk_penalty_b

        lam_a_vec = np.clip(np.exp(raw_log_a), 0.05, 6.0)
        lam_b_vec = np.clip(np.exp(raw_log_b), 0.05, 6.0)

        # 4. Vectorized Dixon-Coles Bivariate Scoreline Sampling across all N simulations
        facts = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800], dtype=float)[:max_goals + 1]
        k_range = np.arange(max_goals + 1)

        # Poisson PMF matrices: shape (N, max_goals + 1)
        p_a_matrix = (lam_a_vec[:, None] ** k_range[None, :]) * np.exp(-lam_a_vec)[:, None] / facts[None, :]
        p_b_matrix = (lam_b_vec[:, None] ** k_range[None, :]) * np.exp(-lam_b_vec)[:, None] / facts[None, :]

        # Joint probability tensor: shape (N, max_goals + 1, max_goals + 1)
        joint = p_a_matrix[:, :, None] * p_b_matrix[:, None, :]

        # Apply Dixon-Coles tau adjustments
        joint[:, 0, 0] *= (1.0 - lam_a_vec * lam_b_vec * RHO)
        joint[:, 0, 1] *= (1.0 + lam_a_vec * RHO)
        joint[:, 1, 0] *= (1.0 + lam_b_vec * RHO)
        joint[:, 1, 1] *= (1.0 - RHO)

        joint = np.maximum(0.0, joint)
        joint /= joint.sum(axis=(1, 2), keepdims=True)

        # Vectorized categorical sampling via cumulative sums
        flat_joint = joint.reshape(N, (max_goals + 1) ** 2)
        cdf = np.cumsum(flat_joint, axis=1)
        u = self.rng.random(size=(N, 1))
        # Ensure floating precision boundary
        u = np.minimum(u, 1.0 - 1e-12)
        sampled_indices = (u < cdf).argmax(axis=1)

        goals_a = sampled_indices // (max_goals + 1)
        goals_b = sampled_indices % (max_goals + 1)

        h_win = float(np.mean(goals_a > goals_b))
        draw = float(np.mean(goals_a == goals_b))
        a_win = float(np.mean(goals_a < goals_b))

        # Scoreline distribution
        pairs_sample = np.column_stack((goals_a, goals_b))
        unique, counts = np.unique(pairs_sample, axis=0, return_counts=True)
        order = np.argsort(-counts)
        scoreline_freqs = [
            {
                "scoreline": f"{int(unique[idx][0])} - {int(unique[idx][1])}",
                "goals_a": int(unique[idx][0]),
                "goals_b": int(unique[idx][1]),
                "count": int(counts[idx]),
                "pct": round(float(counts[idx] / N * 100), 2),
            }
            for idx in order
        ]

        most_likely = scoreline_freqs[0]["scoreline"] if scoreline_freqs else "1 - 1"

        return {
            "n_simulations": N,
            "match_day_state_enabled": cfg.enabled,
            "probabilities": {
                "team_a_win": round(h_win, 4),
                "draw": round(draw, 4),
                "team_b_win": round(a_win, 4),
                "team_a_win_pct": round(h_win * 100, 2),
                "draw_pct": round(draw * 100, 2),
                "team_b_win_pct": round(a_win * 100, 2),
            },
            "xg_stats": {
                "team_a_mean": round(float(np.mean(lam_a_vec)), 3),
                "team_a_std": round(float(np.std(lam_a_vec)), 3),
                "team_b_mean": round(float(np.mean(lam_b_vec)), 3),
                "team_b_std": round(float(np.std(lam_b_vec)), 3),
            },
            "team_ratings_stats": {
                "team_a_attack_mean": round(float(np.mean(atk_a)), 2),
                "team_a_attack_std": round(float(np.std(atk_a)), 2),
                "team_a_defence_mean": round(float(np.mean(dfn_a)), 2),
                "team_a_defence_std": round(float(np.std(dfn_a)), 2),
                "team_a_midfield_mean": round(float(np.mean(mid_a)), 2),
                "team_a_midfield_std": round(float(np.std(mid_a)), 2),
                "team_b_attack_mean": round(float(np.mean(atk_b)), 2),
                "team_b_attack_std": round(float(np.std(atk_b)), 2),
                "team_b_defence_mean": round(float(np.mean(dfn_b)), 2),
                "team_b_defence_std": round(float(np.std(dfn_b)), 2),
                "team_b_midfield_mean": round(float(np.mean(mid_b)), 2),
                "team_b_midfield_std": round(float(np.std(mid_b)), 2),
            },
            "most_likely_scoreline": most_likely,
            "top_scorelines": scoreline_freqs[:10],
            "raw_goals_a": goals_a,
            "raw_goals_b": goals_b,
            "raw_lam_a": lam_a_vec,
            "raw_lam_b": lam_b_vec,
            "raw_atk_a": atk_a,
            "raw_dfn_a": dfn_a,
            "raw_atk_b": atk_b,
            "raw_dfn_b": dfn_b,
        }
