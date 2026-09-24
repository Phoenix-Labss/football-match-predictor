"""Tournament simulation service for the Dynamic Oracle web app.

Wraps the DynamicOracle player-aware rating pipeline with a fast Monte Carlo
tournament engine:

    - Group stage: round-robin, 3-1-0 points, GD/GF tie-breakers.
    - Knockout stage: seeded single-elimination bracket; drawn matches are
      decided by a rating-weighted penalty shootout.
    - Aggregated distributions: P(champion), P(runner-up), P(final),
      P(semi), P(quarter/round-of-16), P(group win), P(group advance),
      expected group points and expected goals.
    - A fully detailed "showcase run" (every match scoreline, group tables
      and bracket rounds) so the frontend can render one concrete
      simulated tournament, not just probabilities.

Scoreline probability grids are cached per unique pairing, so hundreds of
tournament runs complete in a few seconds.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np

from src.simulation.squad_model import SquadModel

MAX_GOALS = 10
GRID_W = MAX_GOALS + 1

ROUND_NAMES = {
    2: "Final",
    4: "Semi-Finals",
    8: "Quarter-Finals",
    16: "Round of 16",
    32: "Round of 32",
}


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return round(v, 4)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


class TournamentService:
    """Monte Carlo tournament engine on top of the DynamicOracle ratings."""

    def __init__(self, oracle):
        self.oracle = oracle
        self.rng = np.random.default_rng()
        self._rating_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._grid_cache: dict[tuple, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Team ratings
    # ------------------------------------------------------------------ #
    def _team_label(self, name: str, year: int) -> str:
        return f"{name} ({year})"

    def get_rating(self, name: str, year: int) -> dict[str, Any]:
        """Build (or fetch cached) TeamRating + display meta for a team."""
        key = (name, int(year))
        if key in self._rating_cache:
            return self._rating_cache[key]

        oracle = self.oracle
        pool, resolved = oracle._get_player_pool(name, int(year))
        squad_model = SquadModel(formation="4-3-3")
        lineup_pairs = squad_model.select_lineup(pool)
        chem = oracle.chemistry_model.team_chemistry(name, [p for p, _ in lineup_pairs])
        rating = squad_model.aggregate(name, pool, chemistry_score=chem)
        rating = oracle._apply_priors(rating, name)

        overall = float(np.mean([p.overall for p, _ in lineup_pairs]))
        bundle = {
            "name": name,
            "year": int(year),
            "year_resolved": int(resolved),
            "label": self._team_label(name, int(year)),
            "rating": rating,
            "overall": round(overall, 1),
            "attack": round(float(rating.attack), 1),
            "defence": round(float(rating.defence), 1),
            "midfield": round(float(rating.midfield), 1),
            "chemistry": round(float(rating.chemistry), 2),
            "power": round(
                0.4 * float(rating.attack)
                + 0.3 * float(rating.defence)
                + 0.3 * float(rating.midfield),
                2,
            ),
        }
        self._rating_cache[key] = bundle
        return bundle

    # ------------------------------------------------------------------ #
    # Fast scoreline sampling (cached Dixon-Coles / NegBinomial grids)
    # ------------------------------------------------------------------ #
    def _pair_grid(self, a: dict, b: dict) -> dict[str, Any]:
        key = (a["name"], a["year"], b["name"], b["year"])
        grid = self._grid_cache.get(key)
        if grid is not None:
            return grid

        engine = self.oracle.match_engine
        lam_a, lam_b = engine.expected_goals(a["rating"], b["rating"], neutral=True)
        lam_a = float(np.clip(lam_a, 0.05, 6.0))
        lam_b = float(np.clip(lam_b, 0.05, 6.0))

        ks = np.arange(GRID_W)
        pmf_a = np.array([engine._goal_pmf(int(k), lam_a) for k in ks])
        pmf_b = np.array([engine._goal_pmf(int(k), lam_b) for k in ks])
        probs = np.outer(pmf_a, pmf_b)
        # Dixon-Coles low-score correction
        tau = np.ones_like(probs)
        tau[0, 0] = 1.0 - lam_a * lam_b * (-0.1)
        tau[0, 1] = 1.0 + lam_a * (-0.1)
        tau[1, 0] = 1.0 + lam_b * (-0.1)
        tau[1, 1] = 1.0 - (-0.1)
        probs = np.maximum(probs * tau, 0.0)
        probs /= probs.sum()

        grid = {
            "cum": np.cumsum(probs.ravel()),
            "lam_a": lam_a,
            "lam_b": lam_b,
        }
        self._grid_cache[key] = grid
        return grid

    def _sample_score(self, a: dict, b: dict) -> tuple[int, int]:
        grid = self._pair_grid(a, b)
        idx = int(np.searchsorted(grid["cum"], self.rng.random()))
        ga, gb = divmod(idx, GRID_W)
        return int(ga), int(gb)

    def _penalty_winner(self, a: dict, b: dict) -> dict:
        edge = (a["power"] - b["power"]) / 100.0
        p_a = float(np.clip(0.5 + edge, 0.32, 0.68))
        return a if self.rng.random() < p_a else b
    # ------------------------------------------------------------------ #
    # Group stage
    # ------------------------------------------------------------------ #
    def _simulate_group(
        self, teams: list[dict], collect: bool = False
    ) -> tuple[list[dict], list[dict]]:
        """Round-robin group. Returns (ranked standings, played matches)."""
        table = {
            t["label"]: {
                "team": t["label"], "name": t["name"], "year": t["year"],
                "p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "gd": 0, "pts": 0,
            }
            for t in teams
        }
        matches: list[dict] = []
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                a, b = teams[i], teams[j]
                ga, gb = self._sample_score(a, b)
                ta, tb = table[a["label"]], table[b["label"]]
                ta["p"] += 1; tb["p"] += 1
                ta["gf"] += ga; ta["ga"] += gb
                tb["gf"] += gb; tb["ga"] += ga
                if ga > gb:
                    ta["w"] += 1; ta["pts"] += 3; tb["l"] += 1
                elif ga < gb:
                    tb["w"] += 1; tb["pts"] += 3; ta["l"] += 1
                else:
                    ta["d"] += 1; tb["d"] += 1; ta["pts"] += 1; tb["pts"] += 1
                if collect:
                    matches.append({
                        "home": a["label"], "away": b["label"],
                        "home_goals": ga, "away_goals": gb,
                    })
        standings = list(table.values())
        for row in standings:
            row["gd"] = row["gf"] - row["ga"]
        # Tie-break: points, GD, GF, then a fair coin flip
        noise = {row["team"]: self.rng.random() * 1e-6 for row in standings}
        standings.sort(
            key=lambda r: (r["pts"], r["gd"], r["gf"], noise[r["team"]]),
            reverse=True,
        )
        return standings, matches

    # ------------------------------------------------------------------ #
    # Knockout bracket
    # ------------------------------------------------------------------ #
    def _play_knockout_match(
        self, a: dict, b: dict, collect: bool = False
    ) -> tuple[dict, dict | None]:
        ga, gb = self._sample_score(a, b)
        pens = None
        if ga > gb:
            winner = a
        elif gb > ga:
            winner = b
        else:
            winner = self._penalty_winner(a, b)
            pens = {"winner": winner["label"], "score": "5-4" if winner is a else "4-5"}
        record = None
        if collect:
            record = {
                "home": a["label"], "away": b["label"],
                "home_goals": ga, "away_goals": gb,
                "winner": winner["label"],
                "penalties": pens,
            }
        return winner, record

    def _simulate_bracket(
        self, seeded: list[dict], collect: bool = False
    ) -> dict[str, Any]:
        """Single elimination from a seeded list (seed 1 plays weakest seed)."""
        n = len(seeded)
        size = 1
        while size < n:
            size *= 2

        # Pair strongest with weakest; missing slots are byes for top seeds.
        padded: list[dict | None] = list(seeded) + [None] * (size - n)
        first_round: list[tuple[dict | None, dict | None]] = []
        top, bottom = 0, size - 1
        while top < bottom:
            first_round.append((padded[top], padded[bottom]))
            top += 1
            bottom -= 1

        active: list[dict] = []
        prelim_matches: list[dict] = []
        placements: dict[str, str] = {}
        for a, b in first_round:
            if a is None and b is None:
                continue
            if a is None:
                active.append(b)  # type: ignore[arg-type]
                continue
            if b is None:
                active.append(a)
                continue
            winner, record = self._play_knockout_match(a, b, collect=collect)
            loser = b if winner is a else a
            placements[loser["label"]] = ROUND_NAMES.get(size, f"Round of {size}")
            active.append(winner)
            if collect and record is not None:
                prelim_matches.append(record)

        rounds: list[dict] = []
        if collect and prelim_matches:
            rounds.append({
                "name": ROUND_NAMES.get(size, f"Round of {size}"),
                "matches": prelim_matches,
            })

        while len(active) > 1:
            round_name = ROUND_NAMES.get(len(active), f"Round of {len(active)}")
            next_round: list[dict] = []
            round_matches: list[dict] = []
            for k in range(0, len(active), 2):
                a, b = active[k], active[k + 1]
                winner, record = self._play_knockout_match(a, b, collect=collect)
                loser = b if winner is a else a
                placements[loser["label"]] = round_name
                next_round.append(winner)
                if collect and record is not None:
                    round_matches.append(record)
            if collect:
                rounds.append({"name": round_name, "matches": round_matches})
            active = next_round

        return {
            "champion": active[0],
            "rounds": rounds,
            "placements": placements,
            "bracket_size": size,
        }
    # ------------------------------------------------------------------ #
    # Presets
    # ------------------------------------------------------------------ #
    def get_presets(self) -> list[dict[str, Any]]:
        oracle = self.oracle
        presets: list[dict[str, Any]] = []

        if oracle.wc2026_teams is not None:
            presets.append({
                "id": "wc2026",
                "title": "FIFA World Cup 2026",
                "description": "All 48 qualified nations in their official groups A-L. "
                               "Top 2 + 8 best third-placed teams advance to the Round of 32.",
                "format": "groups",
                "team_count": 48,
                "fixed": True,
                "badge": "OFFICIAL GROUPS",
            })

        for year in oracle.available_years():
            if year == 2026:
                continue
            clubs = [t for t in oracle.teams_index.get(year, []) if t["type"] == "club"]
            nats = [t for t in oracle.teams_index.get(year, []) if t["type"] == "national"]
            if len(clubs) >= 32:
                presets.append({
                    "id": f"club32_{year}",
                    "title": f"Club World Championship {year}",
                    "description": f"The 32 highest-rated club squads from FIFA {year} "
                                   "in 8 groups of 4, knockout from the Round of 16.",
                    "format": "groups",
                    "team_count": 32,
                    "year": year,
                    "badge": "CLUBS",
                })
            if len(nats) >= 16:
                presets.append({
                    "id": f"intl16_{year}",
                    "title": f"International Elite Cup {year}",
                    "description": f"Top 16 national teams of FIFA {year}, 4 groups, "
                                   "quarter-final knockout stage.",
                    "format": "groups",
                    "team_count": 16,
                    "year": year,
                    "badge": "NATIONS",
                })

        presets.append({
            "id": "legends16",
            "title": "Legends All-Time Knockout",
            "description": "Sixteen iconic cross-era squads seeded straight into a "
                           "single-elimination bracket: prime MSN Barcelona, three-peat "
                           "Real Madrid, Pep's City, Messi's Argentina and more.",
            "format": "knockout",
            "team_count": 16,
            "fixed": True,
            "badge": "CROSS-ERA",
        })
        return presets

    def build_preset_teams(
        self, preset_id: str
    ) -> tuple[list[dict], list[list[int]] | None]:
        """Resolve a preset to (team_specs, groups_of_indices or None)."""
        oracle = self.oracle

        if preset_id == "wc2026":
            if oracle.wc2026_teams is None:
                raise ValueError("World Cup 2026 dataset not loaded.")
            specs: list[dict] = []
            groups_map: dict[str, list[int]] = defaultdict(list)
            for row in oracle.wc2026_teams.itertuples(index=False):
                groups_map[str(row.group_letter)].append(len(specs))
                specs.append({"name": str(row.team_name), "year": 2026})
            groups = [groups_map[g] for g in sorted(groups_map)]
            return specs, groups

        if preset_id == "legends16":
            legend_specs = [
                ("FC Barcelona", 2015), ("Real Madrid", 2017),
                ("Manchester City", 2022), ("FC Bayern München", 2020),
                ("Liverpool", 2019), ("Paris Saint-Germain", 2022),
                ("Argentina", 2022), ("France", 2022),
                ("Spain", 2015), ("Germany", 2015),
                ("Brazil", 2022), ("Morocco", 2026),
                ("Juventus", 2017), ("Chelsea", 2021),
                ("Manchester United", 2015), ("Inter", 2021),
            ]
            specs = []
            for name, year in legend_specs:
                self.get_rating(name, year)  # validates existence
                specs.append({"name": name, "year": year})
            return specs, None

        if preset_id.startswith("club32_") or preset_id.startswith("intl16_"):
            kind, year_str = preset_id.rsplit("_", 1)
            year = int(year_str)
            team_type = "club" if kind == "club32" else "national"
            count = 32 if kind == "club32" else 16
            pool = [
                t for t in oracle.teams_index.get(year, [])
                if t["type"] == team_type
            ]
            pool = sorted(pool, key=lambda t: -t["avg_rating"])[:count]
            if len(pool) < count:
                raise ValueError(f"Not enough {team_type} teams for {year}.")
            specs = [{"name": t["name"], "year": year} for t in pool]
            return specs, self._snake_groups(specs, group_size=4)

        raise ValueError(f"Unknown preset '{preset_id}'.")

    def _snake_groups(self, specs: list[dict], group_size: int = 4) -> list[list[int]]:
        """Balanced snake draw of team indices into groups (seeded by rating)."""
        ordered = sorted(
            range(len(specs)),
            key=lambda i: -self.get_rating(specs[i]["name"], specs[i]["year"])["power"],
        )
        n_groups = max(1, len(specs) // group_size)
        groups: list[list[int]] = [[] for _ in range(n_groups)]
        for pos, idx in enumerate(ordered):
            row = pos // n_groups
            col = pos % n_groups
            if row % 2 == 1:
                col = n_groups - 1 - col
            groups[col].append(idx)
        return groups
    # ------------------------------------------------------------------ #
    # Main Monte Carlo entry point
    # ------------------------------------------------------------------ #
    def simulate(
        self,
        team_specs: list[dict],
        format: str = "groups",
        n_simulations: int = 200,
        groups: list[list[int]] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Run a full Monte Carlo tournament simulation.

        team_specs : [{"name": str, "year": int}, ...]
        format     : "groups" (group stage + knockout) or "knockout".
        groups     : optional explicit group assignment as index lists.
        """
        t0 = time.time()
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        n = len(team_specs)
        if n < 4:
            raise ValueError("A tournament needs at least 4 teams.")
        labels = [self._team_label(s["name"], int(s["year"])) for s in team_specs]
        if len(set(labels)) != n:
            raise ValueError("Each team+year combination may only be added once.")

        # Resolve all ratings up-front (cached across runs)
        bundles = [self.get_rating(s["name"], int(s["year"])) for s in team_specs]

        if format == "groups":
            if n % 4 != 0 or n < 8:
                raise ValueError(
                    "Group format requires 8, 12, 16, 20, 24, 32 or 48 teams "
                    "(divisible by 4)."
                )
            if groups is None:
                groups = self._snake_groups(team_specs, group_size=4)
        elif format != "knockout":
            raise ValueError("format must be 'groups' or 'knockout'.")

        stats: dict[str, dict[str, float]] = {
            b["label"]: defaultdict(float) for b in bundles
        }
        final_pairs: dict[tuple[str, str], int] = defaultdict(int)
        showcase: dict[str, Any] | None = None

        for run in range(int(n_simulations)):
            collect = run == int(n_simulations) - 1  # showcase = final run
            group_tables = None

            if format == "groups":
                qualifiers: list[tuple[int, dict, dict]] = []  # (tier, bundle, row)
                group_tables = []
                for gi, gidx in enumerate(groups or []):
                    gteams = [bundles[i] for i in gidx]
                    standings, gmatches = self._simulate_group(gteams, collect=collect)
                    if collect:
                        group_tables.append({
                            "name": f"Group {chr(65 + gi)}",
                            "standings": standings,
                            "matches": gmatches,
                        })
                    for rank, row in enumerate(standings):
                        b = next(b for b in gteams if b["label"] == row["team"])
                        stats[b["label"]]["pts_sum"] += row["pts"]
                        stats[b["label"]]["gf_sum"] += row["gf"]
                        if rank == 0:
                            stats[b["label"]]["group_win"] += 1
                        qualifiers.append((rank, b, row))

                n_groups = len(groups or [])
                base = 2 * n_groups
                target = 4
                while target < base:
                    target *= 2
                thirds_needed = min(target - base, n_groups)

                auto = [q for q in qualifiers if q[0] <= 1]
                thirds = sorted(
                    [q for q in qualifiers if q[0] == 2],
                    key=lambda q: (q[2]["pts"], q[2]["gd"], q[2]["gf"]),
                    reverse=True,
                )[:thirds_needed]
                going_through = auto + [(2, b, r) for _, b, r in thirds]
                # Seed: winners, then runners-up, then thirds — each by record
                going_through.sort(
                    key=lambda q: (q[0], -q[2]["pts"], -q[2]["gd"], -q[2]["gf"])
                )
                seeded = [b for _, b, _ in going_through]
                for _, b, _ in going_through:
                    stats[b["label"]]["advance"] += 1
            else:
                seeded = sorted(bundles, key=lambda b: -b["power"])
            bracket = self._simulate_bracket(seeded, collect=collect)
            champ = bracket["champion"]
            placements = bracket["placements"]

            stats[champ["label"]]["champion"] += 1
            stats[champ["label"]]["final"] += 1
            stats[champ["label"]]["semi"] += 1
            runner_up = None
            for lbl, stage in placements.items():
                if stage == "Final":
                    runner_up = lbl
                    stats[lbl]["runner_up"] += 1
                    stats[lbl]["final"] += 1
                    stats[lbl]["semi"] += 1
                elif stage == "Semi-Finals":
                    stats[lbl]["semi"] += 1
                elif stage == "Quarter-Finals":
                    stats[lbl]["qf"] += 1
                elif stage == "Round of 16":
                    stats[lbl]["r16"] += 1
                elif stage == "Round of 32":
                    stats[lbl]["r32"] += 1
            if runner_up is not None:
                pair = tuple(sorted([champ["label"], runner_up]))
                final_pairs[pair] += 1

            if collect:
                showcase = {
                    "groups": group_tables,
                    "rounds": bracket["rounds"],
                    "champion": champ["label"],
                }

        ns = float(n_simulations)
        teams_out = []
        for b in bundles:
            s = stats[b["label"]]
            sf = s["semi"]
            qf = s["qf"] + sf
            r16 = s["r16"] + qf
            r32 = s["r32"] + r16
            teams_out.append({
                "name": b["name"],
                "year": b["year"],
                "label": b["label"],
                "overall": b["overall"],
                "attack": b["attack"],
                "defence": b["defence"],
                "champion_pct": round(s["champion"] / ns * 100, 2),
                "runner_up_pct": round(s["runner_up"] / ns * 100, 2),
                "final_pct": round(s["final"] / ns * 100, 2),
                "semi_pct": round(sf / ns * 100, 2),
                "qf_pct": round(qf / ns * 100, 2),
                "r16_pct": round(r16 / ns * 100, 2),
                "r32_pct": round(r32 / ns * 100, 2),
                "advance_pct": round(s["advance"] / ns * 100, 2) if format == "groups" else None,
                "group_win_pct": round(s["group_win"] / ns * 100, 2) if format == "groups" else None,
                "exp_points": round(s["pts_sum"] / ns, 2) if format == "groups" else None,
                "exp_goals": round(s["gf_sum"] / ns, 2) if format == "groups" else None,
            })
        teams_out.sort(key=lambda t: (-t["champion_pct"], -t["final_pct"], -t["semi_pct"]))

        most_likely_final = None
        if final_pairs:
            pair, cnt = max(final_pairs.items(), key=lambda kv: kv[1])
            most_likely_final = {"teams": list(pair), "pct": round(cnt / ns * 100, 2)}

        result = {
            "meta": {
                "n_teams": n,
                "format": format,
                "n_simulations": int(n_simulations),
                "n_groups": len(groups) if (format == "groups" and groups) else 0,
                "duration_ms": int((time.time() - t0) * 1000),
                "engine": "Dixon-Coles / Negative-Binomial xG Monte Carlo",
            },
            "teams": teams_out,
            "podium": {
                "champion": teams_out[0]["label"],
                "champion_pct": teams_out[0]["champion_pct"],
                "runner_up": teams_out[1]["label"] if n > 1 else None,
                "runner_up_pct": teams_out[1]["champion_pct"] if n > 1 else None,
                "third": teams_out[2]["label"] if n > 2 else None,
                "third_pct": teams_out[2]["champion_pct"] if n > 2 else None,
            },
            "most_likely_final": most_likely_final,
            "showcase": showcase,
        }
        return _sanitize(result)
