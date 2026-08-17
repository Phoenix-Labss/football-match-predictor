"""World Cup tournament simulator (Track 3).

Simulates the FIFA World Cup format:

    - 32 teams, 8 groups of 4
    - round-robin group stage (3 matches each)
    - top 2 of each group advance
    - knockout bracket: R16 -> QF -> SF -> Final
    - group tie-breakers: points, goal difference, goals for

The simulator runs the tournament `n_simulations` times. Each run samples
different match outcomes, so we get full probability distributions:

    P(win World Cup), P(reach final), P(reach semi), P(reach QF),
    P(exit group stage)

as well as the most likely finalists and champion.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import poisson

from .chemistry import ChemistryModel
from .match_engine import MatchEngine
from .player_model import PlayerModel, PlayerState
from .squad_model import SquadModel, TeamRating


# --------------------------------------------------------------------- #
# Draw helpers
# --------------------------------------------------------------------- #
def _default_pots(teams: list[str]) -> list[list[str]]:
    """Divide 32 teams into 4 pots of 8 (teams pre-sorted by strength)."""
    return [teams[i : i + 8] for i in range(0, 32, 8)]


def _snake_draw(pots: list[list[str]]) -> list[list[str]]:
    """Deterministic S-curve draw: 8 groups of 4 from 4 seeded pots."""
    groups: list[list[str]] = [[] for _ in range(8)]
    for pot_idx, pot in enumerate(pots):
        ordered = reversed(pot) if pot_idx % 2 == 1 else pot
        for gi, team in enumerate(ordered):
            groups[gi % 8].append(team)
    return groups


@dataclass
class TournamentResult:
    """Aggregated Monte Carlo tournament results."""

    n_simulations: int
    teams: list[str]
    champion_prob: dict[str, float]
    finalist_prob: dict[str, float]
    semi_prob: dict[str, float]
    qf_prob: dict[str, float]
    group_exit_prob: dict[str, float]
    most_likely_semis: list[tuple[str, str]] = field(default_factory=list)
    most_likely_final: tuple[str, str] | None = None
    most_likely_winner: str | None = None

    def to_frame(self):
        """Return a tidy results frame with one row per team."""
        import pandas as pd

        return pd.DataFrame(
            {
                "team": list(self.teams),
                "p_winner": [self.champion_prob.get(t, 0.0) for t in self.teams],
                "p_finalist": [self.finalist_prob.get(t, 0.0) for t in self.teams],
                "p_semi": [self.semi_prob.get(t, 0.0) for t in self.teams],
                "p_qf": [self.qf_prob.get(t, 0.0) for t in self.teams],
                "p_group_exit": [self.group_exit_prob.get(t, 0.0) for t in self.teams],
            }
        ).sort_values("p_winner", ascending=False)

    def to_csv(self, path: str) -> None:
        self.to_frame().to_csv(path, index=False)


class SimTournamentRun:
    """Bookkeeping for a single tournament simulation."""

    __slots__ = (
        "champion", "finalists", "sf_pairs", "qf_teams", "group_winners",
        "group_runners_up", "group_finishers",
    )

    def __init__(self):
        self.champion: str | None = None
        self.finalists: list[str] = []
        self.sf_pairs: list[tuple[str, str]] = []
        self.qf_teams: list[str] = []
        self.group_winners: list[str] = []
        self.group_runners_up: list[str] = []
        self.group_finishers: dict[str, int] = {}  # team -> 1..4 in group


class WorldCupSimulator:
    """Monte Carlo World Cup simulator.

    Parameters
    ----------
    player_model : PlayerModel (builds per-year player states).
    squad_model : SquadModel (selects XI + aggregates team ratings).
    chemistry_model : ChemistryModel (team chemistry score).
    match_engine : MatchEngine (samples scorelines).
    formations : optional {team: formation_name} per team.
    """

    def __init__(
        self,
        player_model: PlayerModel,
        squad_model: SquadModel,
        chemistry_model: ChemistryModel,
        match_engine: MatchEngine,
        formations: dict[str, str] | None = None,
    ):
        self.player_model = player_model
        self.squad_model = squad_model
        self.chemistry_model = chemistry_model
        self.match_engine = match_engine
        self.formations = formations or {}

    # ------------------------------------------------------------------ #
    # Team rating cache
    # ------------------------------------------------------------------ #
    def _team_rating(
        self,
        states: dict[int, PlayerState],
        team: str,
    ) -> TeamRating:
        """Build the current TeamRating for a national team."""
        pool = self.player_model.national_pool(states, team)
        if not pool:
            raise ValueError(f"No players found for national team '{team}'")

        # Per-team formation override.
        formation = self.formations.get(team, self.squad_model.formation)
        original = self.squad_model.formation
        self.squad_model.formation = formation
        try:
            lineup = self.squad_model.select_lineup(pool)
            chem = self.chemistry_model.team_chemistry(
                team, [p for p, _ in lineup]
            )
            return self.squad_model.aggregate(team, pool, chemistry_score=chem)
        finally:
            self.squad_model.formation = original

    def build_ratings(
        self, states: dict[int, PlayerState], teams: list[str]
    ) -> dict[str, TeamRating]:
        """Pre-compute TeamRatings for all participating teams."""
        return {t: self._team_rating(states, t) for t in teams}

    # ------------------------------------------------------------------ #
    # Group stage
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sample_match_fast(
        a: str,
        b: str,
        ratings: dict[str, TeamRating],
        engine: MatchEngine,
        rng: np.random.Generator,
        match_cdfs: dict[tuple[str, str], tuple[np.ndarray, int]] | None = None,
    ) -> tuple[int, int]:
        if match_cdfs and (a, b) in match_cdfs:
            cdf, stride = match_cdfs[(a, b)]
            u = rng.random()
            idx = int(np.searchsorted(cdf, u))
            return divmod(idx, stride)
        return engine.sample_scoreline(ratings[a], ratings[b], neutral=True)

    @classmethod
    def _simulate_group(
        cls,
        group: list[str],
        ratings: dict[str, TeamRating],
        engine: MatchEngine,
        rng: np.random.Generator,
        match_cdfs: dict[tuple[str, str], tuple[np.ndarray, int]] | None = None,
    ) -> list[tuple[str, int, int, int]]:
        """Play round-robin, return standings [(team, pts, gd, gf)...] sorted."""
        standings: dict[str, list[int]] = {t: [0, 0, 0] for t in group}
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                gh, ga = cls._sample_match_fast(a, b, ratings, engine, rng, match_cdfs)
                if gh > ga:
                    standings[a][0] += 3
                elif gh < ga:
                    standings[b][0] += 3
                else:
                    standings[a][0] += 1
                    standings[b][0] += 1
                standings[a][1] += gh - ga
                standings[b][1] += ga - gh
                standings[a][2] += gh
                standings[b][2] += ga
        order = sorted(
            group,
            key=lambda t: (standings[t][0], standings[t][1], standings[t][2]),
            reverse=True,
        )
        return [(t, *standings[t]) for t in order]

    # ------------------------------------------------------------------ #
    # Knockout
    # ------------------------------------------------------------------ #
    @classmethod
    def _play_knockout_match(
        cls,
        rating_a: TeamRating,
        rating_b: TeamRating,
        engine: MatchEngine,
        rng: np.random.Generator,
        match_cdfs: dict[tuple[str, str], tuple[np.ndarray, int]] | None = None,
    ) -> str:
        """Play a knockout match, resolving draws (ET + penalties).
        Returns the winner's team name."""
        # Normal time
        gh, ga = cls._sample_match_fast(rating_a.team, rating_b.team, {rating_a.team: rating_a, rating_b.team: rating_b}, engine, rng, match_cdfs)
        if gh != ga:
            return rating_a.team if gh > ga else rating_b.team
        # Extra time: resample.
        gh2, ga2 = cls._sample_match_fast(rating_a.team, rating_b.team, {rating_a.team: rating_a, rating_b.team: rating_b}, engine, rng, match_cdfs)
        if gh2 != ga2:
            return rating_a.team if gh2 > ga2 else rating_b.team
        # Penalties: coin flip weighted by a small ability edge.
        edge = (rating_a.attack - rating_b.attack) / 100.0
        p_a = float(np.clip(0.5 + edge * 0.3, 0.2, 0.8))
        return rating_a.team if rng.random() < p_a else rating_b.team

    # ------------------------------------------------------------------ #
    # Single tournament
    # ------------------------------------------------------------------ #
    def simulate_one(
        self,
        ratings: dict[str, TeamRating],
        groups: list[list[str]] | None = None,
        rng: np.random.Generator | None = None,
        match_cdfs: dict[tuple[str, str], tuple[np.ndarray, int]] | None = None,
    ) -> SimTournamentRun | None:
        """Simulate one full World Cup, returning a SimTournamentRun."""
        run = SimTournamentRun()
        rng = rng or np.random.default_rng()
        engine = self.match_engine

        if groups is None:
            teams = sorted(ratings.keys(), key=lambda t: -ratings[t].attack)
            pots = _default_pots(teams[:32])
            groups = _snake_draw(pots)

        # Group stage.
        group_winners: list[str] = []
        group_runners_up: list[str] = []
        for grp in groups:
            standings = self._simulate_group(grp, ratings, engine, rng, match_cdfs)
            for rank, (t, *_rest) in enumerate(standings, start=1):
                run.group_finishers[t] = rank
            group_winners.append(standings[0][0])
            group_runners_up.append(standings[1][0])
        run.group_winners = group_winners
        run.group_runners_up = group_runners_up

        # Round of 16 (standard 2022 bracket layout).
        r16_pairs = [
            (group_winners[0], group_runners_up[1]),  # 1A vs 2B
            (group_winners[2], group_runners_up[3]),  # 1C vs 2D
            (group_winners[4], group_runners_up[5]),  # 1E vs 2F
            (group_winners[6], group_runners_up[7]),  # 1G vs 2H
            (group_winners[1], group_runners_up[0]),  # 1B vs 2A
            (group_winners[3], group_runners_up[2]),  # 1D vs 2C
            (group_winners[5], group_runners_up[4]),  # 1F vs 2E
            (group_winners[7], group_runners_up[6]),  # 1H vs 2G
        ]
        r16_winners = [
            self._play_knockout_match(ratings[a], ratings[b], engine, rng, match_cdfs)
            for a, b in r16_pairs
        ]

        # Quarter-finals.
        qf_pairs = [
            (r16_winners[0], r16_winners[1]),
            (r16_winners[2], r16_winners[3]),
            (r16_winners[4], r16_winners[5]),
            (r16_winners[6], r16_winners[7]),
        ]
        run.qf_teams = [t for pair in qf_pairs for t in pair]
        qf_winners = [
            self._play_knockout_match(ratings[a], ratings[b], engine, rng, match_cdfs)
            for a, b in qf_pairs
        ]

        # Semi-finals.
        sf_pairs = [
            (qf_winners[0], qf_winners[1]),
            (qf_winners[2], qf_winners[3]),
        ]
        finalists = [
            self._play_knockout_match(ratings[a], ratings[b], engine, rng, match_cdfs)
            for a, b in sf_pairs
        ]
        run.sf_pairs = sf_pairs
        run.finalists = finalists

        # Final.
        run.champion = self._play_knockout_match(
            ratings[finalists[0]], ratings[finalists[1]], engine, rng, match_cdfs
        )
        return run

    # ------------------------------------------------------------------ #
    # Monte Carlo loop
    # ------------------------------------------------------------------ #
    def simulate_tournament(
        self,
        states: dict[int, PlayerState],
        teams: list[str],
        n_simulations: int = 1000,
        groups: list[list[str]] | None = None,
        seed: int | None = None,
    ) -> TournamentResult:
        """Run `n_simulations` tournaments and aggregate probabilities."""
        ratings = self.build_ratings(states, teams)
        valid_teams = [t for t in teams if t in ratings]

        # Fast precomputation of match CDFs for all pairs of teams
        match_cdfs: dict[tuple[str, str], tuple[np.ndarray, int]] = {}
        for a in valid_teams:
            for b in valid_teams:
                if a == b:
                    continue
                lam_h, lam_a = self.match_engine.expected_goals(ratings[a], ratings[b], neutral=True)
                max_g = self.match_engine.config.max_goals
                grid = np.zeros((max_g + 1, max_g + 1))
                for i in range(max_g + 1):
                    for j in range(max_g + 1):
                        p = (
                            poisson.pmf(i, lam_h)
                            * poisson.pmf(j, lam_a)
                            * self.match_engine._tau(i, j, lam_h, lam_a)
                        )
                        grid[i, j] = max(p, 0.0)
                grid /= grid.sum()
                match_cdfs[(a, b)] = (np.cumsum(grid.ravel()), max_g + 1)

        champion_counts: Counter = Counter()
        finalist_counts: Counter = Counter()
        semi_counts: Counter = Counter()
        qf_counts: Counter = Counter()
        group_exit_counts: Counter = Counter()
        finals_seen: Counter = Counter()
        semis_seen: Counter = Counter()

        rng = np.random.default_rng(seed)
        for _ in range(n_simulations):
            run = self.simulate_one(ratings, groups, rng, match_cdfs=match_cdfs)
            if run.champion is None:
                continue

            champion_counts[run.champion] += 1
            for t in run.finalists:
                finalist_counts[t] += 1
            for pair in run.sf_pairs:
                for t in pair:
                    semi_counts[t] += 1
            for t in run.qf_teams:
                qf_counts[t] += 1
            # Group exit = finished 3rd or 4th in group.
            for t in valid_teams:
                if run.group_finishers.get(t, 0) >= 3:
                    group_exit_counts[t] += 1

            finals_seen[tuple(sorted(run.finalists))] += 1
            for pair in run.sf_pairs:
                semis_seen[tuple(sorted(pair))] += 1

        n = n_simulations
        return TournamentResult(
            n_simulations=n,
            teams=valid_teams,
            champion_prob={t: champion_counts[t] / n for t in valid_teams},
            finalist_prob={t: finalist_counts[t] / n for t in valid_teams},
            semi_prob={t: semi_counts[t] / n for t in valid_teams},
            qf_prob={t: qf_counts[t] / n for t in valid_teams},
            group_exit_prob={t: group_exit_counts[t] / n for t in valid_teams},
            most_likely_semis=[],
            most_likely_final=(
                finals_seen.most_common(1)[0][0] if finals_seen else None
            ),
            most_likely_winner=(
                champion_counts.most_common(1)[0][0] if champion_counts else None
            ),
        )