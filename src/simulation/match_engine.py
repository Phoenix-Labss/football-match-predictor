"""Match engine: team ratings -> expected goals -> Poisson scoreline sampling.

This is the heart of the simulator. Given two TeamRatings (from the squad
model), we compute expected goals (xG) for each side, then sample a
scoreline.

The xG model is a **Dixon-Coles**-style bivariate Poisson model, which is
the standard in the academic literature and fixes the plain-Poisson
under-prediction of draws and low-scoring games:

    lambda_home = exp(attack_home - defence_away + midfield_diff + home_adv)
    lambda_away = exp(attack_away - defence_home - midfield_diff)

with a Dixon-Coles correction term tau(x, y) that adjusts the probability
of the low-scoring cells (0-0, 1-0, 0-1, 1-1) to match reality.

The model also folds in:
    - home advantage (team-specific, learnable)
    - chemistry (a small multiplier on attack)
    - manager style (attack/defence bias already baked into TeamRating)
    - a small random "form noise" so identical teams don't always draw
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson

from .squad_model import TeamRating

# Dixon-Coles low-score correction parameter (standard values).
RHO = -0.1  # negative: low scores are more likely than independent Poisson


@dataclass
class MatchEngineConfig:
    """Configuration for the match engine."""

    home_advantage: float = 0.30        # log-scale home boost
    chemistry_attack_weight: float = 0.15
    chemistry_defence_weight: float = 0.10
    form_noise: float = 0.05            # std of log-noise on xG
    max_goals: int = 10                 # cap on sampled goals per team
    # Scale from FIFA-style ratings (0-99) to log-xG.
    rating_scale: float = 0.02
    # Baseline log-goal intercept: equal teams produce ~exp(0.30) = 1.35
    # expected goals each (calibrates mean goals toward real ~2.7/game).
    baseline_goals: float = 0.30
    # Goal distribution model: "negbin" (default production) or "poisson" (legacy)
    goal_model: str = "negbin"
    # Pre-tournament frozen overdispersion parameter for Negative Binomial: Var(X) = mu + alpha*mu^2
    dispersion_alpha: float = 0.1262


class MatchEngine:
    """Samples match scorelines from team ratings."""

    def __init__(
        self,
        config: MatchEngineConfig | None = None,
        home_advantages: dict[str, float] | None = None,
        seed: int | None = None,
    ):
        self.config = config or MatchEngineConfig()
        self.home_advantages = home_advantages or {}
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # xG computation
    # ------------------------------------------------------------------ #
    def expected_goals(
        self,
        home: TeamRating,
        away: TeamRating,
        neutral: bool = False,
    ) -> tuple[float, float]:
        """Return (lambda_home, lambda_away) expected goals."""
        cfg = self.config

        # Attack vs defence differentials (FIFA-style ratings, 0-100).
        home_attack = home.attack * (1.0 + cfg.chemistry_attack_weight * home.chemistry)
        away_attack = away.attack * (1.0 + cfg.chemistry_attack_weight * away.chemistry)
        home_defence = home.defence * (1.0 + cfg.chemistry_defence_weight * home.chemistry)
        away_defence = away.defence * (1.0 + cfg.chemistry_defence_weight * away.chemistry)

        # Midfield control shifts xG slightly.
        mid_diff = (home.midfield - away.midfield) * 0.5

        # Home advantage (team-specific if provided, else global).
        ha = 0.0 if neutral else self.home_advantages.get(
            home.team, cfg.home_advantage
        )

        # GK quality reduces opponent xG.
        gk_home = home.gk / 100.0
        gk_away = away.gk / 100.0

        raw_home = (
            cfg.baseline_goals
            + (home_attack - away_defence) * cfg.rating_scale
            + mid_diff * cfg.rating_scale
            + ha
            - 0.3 * gk_away
        )
        raw_away = (
            cfg.baseline_goals
            + (away_attack - home_defence) * cfg.rating_scale
            - mid_diff * cfg.rating_scale
            - 0.3 * gk_home
        )

        # Form noise: small random fluctuation so identical teams vary.
        noise_h = self.rng.normal(0, cfg.form_noise)
        noise_a = self.rng.normal(0, cfg.form_noise)

        lam_home = float(np.exp(raw_home + noise_h))
        lam_away = float(np.exp(raw_away + noise_a))

        # Clamp to a sane range (avoid degenerate blowouts).
        lam_home = float(np.clip(lam_home, 0.05, 6.0))
        lam_away = float(np.clip(lam_away, 0.05, 6.0))
        return lam_home, lam_away

    # ------------------------------------------------------------------ #
    # Dixon-Coles correction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tau(x: int, y: int, lam_h: float, lam_a: float) -> float:
        """Dixon-Coles correction factor for low-scoring cells."""
        if x == 0 and y == 0:
            return 1.0 - lam_h * lam_a * RHO
        if x == 0 and y == 1:
            return 1.0 + lam_h * RHO
        if x == 1 and y == 0:
            return 1.0 + lam_a * RHO
        if x == 1 and y == 1:
            return 1.0 - RHO
        return 1.0

    def _goal_pmf(self, k: int, mu: float) -> float:
        """Evaluate univariate goal PMF based on config.goal_model ('negbin' or 'poisson')."""
        if self.config.goal_model == "poisson" or self.config.dispersion_alpha <= 1e-6:
            return float(poisson.pmf(k, mu))
        # Negative Binomial parameterization: mean = mu, Var = mu + alpha*mu^2
        alpha = self.config.dispersion_alpha
        r = 1.0 / alpha
        log_p = (
            gammaln(k + r)
            - gammaln(k + 1.0)
            - gammaln(r)
            + k * np.log(alpha * mu / (1.0 + alpha * mu))
            - r * np.log(1.0 + alpha * mu)
        )
        return float(np.exp(log_p))

    # ------------------------------------------------------------------ #
    # Scoreline sampling
    # ------------------------------------------------------------------ #
    def sample_scoreline(
        self,
        home: TeamRating,
        away: TeamRating,
        neutral: bool = False,
    ) -> tuple[int, int]:
        """Sample a single (home_goals, away_goals) scoreline."""
        lam_h, lam_a = self.expected_goals(home, away, neutral)

        max_g = self.config.max_goals
        probs = np.zeros((max_g + 1, max_g + 1))
        for i in range(max_g + 1):
            p_i = self._goal_pmf(i, lam_h)
            for j in range(max_g + 1):
                p = (
                    p_i
                    * self._goal_pmf(j, lam_a)
                    * self._tau(i, j, lam_h, lam_a)
                )
                probs[i, j] = max(p, 0.0)
        probs /= probs.sum()

        flat = probs.ravel()
        idx = self.rng.choice(len(flat), p=flat)
        home_goals, away_goals = divmod(idx, max_g + 1)
        return int(home_goals), int(away_goals)

    def match_probabilities(
        self,
        home: TeamRating,
        away: TeamRating,
        neutral: bool = False,
    ) -> tuple[float, float, float]:
        """Return P(home win), P(draw), P(away win) from the xG model."""
        lam_h, lam_a = self.expected_goals(home, away, neutral)
        max_g = self.config.max_goals
        p_home = p_draw = p_away = 0.0
        for i in range(max_g + 1):
            p_i = self._goal_pmf(i, lam_h)
            for j in range(max_g + 1):
                p = (
                    p_i
                    * self._goal_pmf(j, lam_a)
                    * self._tau(i, j, lam_h, lam_a)
                )
                if p < 0:
                    p = 0.0
                if i > j:
                    p_home += p
                elif i == j:
                    p_draw += p
                else:
                    p_away += p
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total