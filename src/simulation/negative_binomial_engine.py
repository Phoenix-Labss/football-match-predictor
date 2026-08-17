"""Negative Binomial match engine for football score distributions.

Implements a Negative Binomial goal model with mean mu = lambda and overdispersion
parameter alpha > 0, where:
    Var[X] = mu + alpha * mu^2  (VMR = 1 + alpha * mu > 1)

Includes:
- Negative Binomial independent goal model
- Bivariate joint distribution with Dixon-Coles-style low-score correction
- Exact joint PMF and Poisson-Gamma mixture sampling
- Full compatibility with TeamRating and MatchDayState architectures
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

from .squad_model import TeamRating

# Standard Dixon-Coles low-score correction parameter
RHO = -0.10


@dataclass
class NegativeBinomialConfig:
    """Configuration for Negative Binomial match engine."""

    dispersion_alpha: float = 0.15       # Global overdispersion alpha: Var(X) = mu + alpha*mu^2
    dispersion_alpha_home: float | None = None
    dispersion_alpha_away: float | None = None
    use_dixon_coles_correction: bool = True
    rho: float = -0.10
    home_advantage: float = 0.30         # log-scale home boost
    chemistry_attack_weight: float = 0.15
    chemistry_defence_weight: float = 0.10
    form_noise: float = 0.05             # std of log-noise on xG
    max_goals: int = 10                  # max goal grid dimension
    rating_scale: float = 0.02           # scale from ratings to log-xG
    baseline_goals: float = 0.30         # baseline log-goal intercept


class NegativeBinomialEngine:
    """Samples match scorelines and computes probabilities using Negative Binomial distributions."""

    def __init__(
        self,
        config: NegativeBinomialConfig | None = None,
        home_advantages: dict[str, float] | None = None,
        seed: int | None = None,
    ):
        self.config = config or NegativeBinomialConfig()
        self.home_advantages = home_advantages or {}
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # xG computation (Identical underlying tactical formulation)
    # ------------------------------------------------------------------ #
    def expected_goals(
        self,
        home: TeamRating,
        away: TeamRating,
        neutral: bool = False,
    ) -> tuple[float, float]:
        """Return (lambda_home, lambda_away) expected goals."""
        cfg = self.config

        home_attack = home.attack * (1.0 + cfg.chemistry_attack_weight * home.chemistry)
        away_attack = away.attack * (1.0 + cfg.chemistry_attack_weight * away.chemistry)
        home_defence = home.defence * (1.0 + cfg.chemistry_defence_weight * home.chemistry)
        away_defence = away.defence * (1.0 + cfg.chemistry_defence_weight * away.chemistry)

        mid_diff = (home.midfield - away.midfield) * 0.5

        ha = 0.0 if neutral else self.home_advantages.get(home.team, cfg.home_advantage)

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

        noise_h = self.rng.normal(0, cfg.form_noise)
        noise_a = self.rng.normal(0, cfg.form_noise)

        lam_home = float(np.exp(raw_home + noise_h))
        lam_away = float(np.exp(raw_away + noise_a))

        lam_home = float(np.clip(lam_home, 0.05, 6.0))
        lam_away = float(np.clip(lam_away, 0.05, 6.0))
        return lam_home, lam_away

    # ------------------------------------------------------------------ #
    # Negative Binomial Probability Density
    # ------------------------------------------------------------------ #
    @staticmethod
    def nb_pmf(k_arr: np.ndarray, mu: float, alpha: float) -> np.ndarray:
        """Evaluate Negative Binomial PMF for array k with mean mu and dispersion alpha.

        When alpha -> 0, converges to Poisson(mu).
        """
        if alpha <= 1e-6:
            # Poisson fallback
            k_fact = np.array([math.factorial(int(k)) for k in k_arr], dtype=float)
            return (mu ** k_arr) * np.exp(-mu) / k_fact

        r = 1.0 / alpha
        # P(X = k) = exp( gammaln(k + r) - gammaln(k + 1) - gammaln(r) + k * ln(alpha*mu / (1 + alpha*mu)) - r * ln(1 + alpha*mu) )
        log_prob = (
            gammaln(k_arr + r)
            - gammaln(k_arr + 1.0)
            - gammaln(r)
            + k_arr * np.log(alpha * mu / (1.0 + alpha * mu))
            - r * np.log(1.0 + alpha * mu)
        )
        return np.exp(log_prob)

    # ------------------------------------------------------------------ #
    # Joint Scoreline Probability Matrix
    # ------------------------------------------------------------------ #
    def joint_pmf(
        self,
        lam_h: float,
        lam_a: float,
        alpha_h: float | None = None,
        alpha_a: float | None = None,
        use_dc_correction: bool | None = None,
    ) -> np.ndarray:
        """Compute joint (max_goals+1, max_goals+1) probability matrix."""
        cfg = self.config
        a_h = alpha_h if alpha_h is not None else (cfg.dispersion_alpha_home or cfg.dispersion_alpha)
        a_a = alpha_a if alpha_a is not None else (cfg.dispersion_alpha_away or cfg.dispersion_alpha)
        use_dc = use_dc_correction if use_dc_correction is not None else cfg.use_dixon_coles_correction

        k_range = np.arange(cfg.max_goals + 1)
        p_h = self.nb_pmf(k_range, lam_h, a_h)
        p_a = self.nb_pmf(k_range, lam_a, a_a)

        joint = np.outer(p_h, p_a)

        if use_dc:
            rho = cfg.rho
            joint[0, 0] *= (1.0 - lam_h * lam_a * rho)
            joint[0, 1] *= (1.0 + lam_h * rho)
            joint[1, 0] *= (1.0 + lam_a * rho)
            joint[1, 1] *= (1.0 - rho)
            joint = np.maximum(0.0, joint)

        joint_sum = joint.sum()
        if joint_sum > 0:
            joint /= joint_sum
        return joint

    def match_probabilities(
        self,
        lam_h: float,
        lam_a: float,
        alpha_h: float | None = None,
        alpha_a: float | None = None,
        use_dc_correction: bool | None = None,
    ) -> tuple[float, float, float, str, float]:
        """Return (P(Home), P(Draw), P(Away), mode_scoreline, mode_prob)."""
        joint = self.joint_pmf(lam_h, lam_a, alpha_h, alpha_a, use_dc_correction)
        p_home = float(np.sum(np.tril(joint, -1)))
        p_draw = float(np.sum(np.diag(joint)))
        p_away = float(np.sum(np.triu(joint, 1)))

        tot = p_home + p_draw + p_away
        if tot > 0:
            p_home /= tot
            p_draw /= tot
            p_away /= tot

        idx_mode = int(np.argmax(joint))
        ga_mode, gb_mode = divmod(idx_mode, self.config.max_goals + 1)
        mode_str = f"{ga_mode} - {gb_mode}"
        mode_prob = float(joint[ga_mode, gb_mode])

        return p_home, p_draw, p_away, mode_str, mode_prob

    # ------------------------------------------------------------------ #
    # Scoreline sampling
    # ------------------------------------------------------------------ #
    def sample_scoreline(
        self,
        home: TeamRating,
        away: TeamRating,
        neutral: bool = False,
    ) -> tuple[int, int]:
        """Sample a scoreline (goals_home, goals_away) via direct joint PMF sampling."""
        lam_h, lam_a = self.expected_goals(home, away, neutral=neutral)
        joint = self.joint_pmf(lam_h, lam_a)
        flat_joint = joint.ravel()
        idx = int(self.rng.choice(len(flat_joint), p=flat_joint))
        gh, ga = divmod(idx, self.config.max_goals + 1)
        return gh, ga
