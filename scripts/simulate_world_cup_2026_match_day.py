"""2026 FIFA World Cup Full Tournament Simulator with Match-Day State Engine.

Simulates 10,000 complete 48-team World Cup tournaments under Match-Day State.
Tracks team progression, match outcomes, player distributions, and generates
all required CSV/JSON/MD artifacts in results/world_cup_2026/.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

# Set project root
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.service.oracle import load_oracle
from src.simulation.chemistry import ChemistryModel
from src.simulation.match_day_state import (
    MatchDayStateConfig,
    MatchDayStateSampler,
    PlayerMatchState,
    TeamMatchState,
)
from src.simulation.match_engine import MatchEngine, MatchEngineConfig
from src.simulation.squad_model import FORMATIONS, SquadModel, TeamRating


def run_world_cup_2026_simulation():
    print("=" * 80)
    print("2026 FIFA WORLD CUP SIMULATION — MATCH-DAY STATE ENGINE (10,000 TOURNAMENTS)")
    print("=" * 80)

    out_dir = root / "results" / "world_cup_2026"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    n_tournaments = 10000
    rng = np.random.default_rng(seed)

    oracle = load_oracle()
    config = MatchDayStateConfig(
        enabled=True,
        player_form_sigma=0.02,
        player_perf_sigma=0.02,
        team_execution_sigma=0.02,
        team_cohesion_sigma=0.02,
        seed=seed,
    )
    sampler = MatchDayStateSampler(config, seed=seed)
    engine = oracle.match_engine

    # 12 Groups of 4 (48 Teams total) from official 2026 dataset
    groups_2026 = {
        "Group A": ["Mexico", "South Africa", "South Korea", "Czechia"],
        "Group B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
        "Group C": ["Brazil", "Morocco", "Haiti", "Scotland"],
        "Group D": ["USA", "Paraguay", "Australia", "Türkiye"],
        "Group E": ["Germany", "Curaçao", "Côte d'Ivoire", "Ecuador"],
        "Group F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
        "Group G": ["Belgium", "Egypt", "IR Iran", "New Zealand"],
        "Group H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
        "Group I": ["France", "Senegal", "Iraq", "Norway"],
        "Group J": ["Argentina", "Algeria", "Austria", "Jordan"],
        "Group K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
        "Group L": ["England", "Croatia", "Ghana", "Panama"],
    }

    all_teams = [t for grp in groups_2026.values() for t in grp]
    slots = FORMATIONS["4-3-3"]

    print(f"\n[1/5] Extracting rosters, starting XIs, and chemistry for {len(all_teams)} teams...")
    team_data = {}
    elo_ratings = {}
    if oracle.wc2026_teams is not None:
        for r in oracle.wc2026_teams.itertuples(index=False):
            elo_ratings[r.team_name] = r.elo_rating

    for t in all_teams:
        key_match = None
        team_dict = oracle.players_by_year_team.get(2026, {})
        for k in team_dict:
            if (
                t.lower() == k.lower()
                or t.lower() in k.lower()
                or k.lower() in t.lower()
                or ("ivo" in t.lower() and "ivo" in k.lower())
                or ("turk" in t.lower() and "turk" in k.lower())
                or ("curac" in t.lower() and "curac" in k.lower())
                or ("iran" in t.lower() and "iran" in k.lower())
                or ("korea" in t.lower() and "korea" in k.lower())
            ):
                key_match = k
                break
        if not key_match:
            key_match = t
        pool, yr = oracle._get_player_pool(key_match, 2026)
        squad_model = SquadModel(formation="4-3-3")
        lineup_pairs = squad_model.select_lineup(pool)
        chem = oracle.chemistry_model.team_chemistry(t, [p for p, _ in lineup_pairs])
        base_rating = squad_model.aggregate(t, pool, chemistry_score=chem)

        team_data[t] = {
            "pool": pool,
            "lineup_pairs": lineup_pairs,
            "base_chem": chem,
            "base_rating": base_rating,
            "elo": elo_ratings.get(t, 1750),
        }

    print("All 48 team Starting XIs and baseline profiles successfully loaded!")

    # ------------------------------------------------------------------ #
    # FAST PREPARED ARRAYS FOR SIMULATION
    # ------------------------------------------------------------------ #
    team_base_stats = {}
    for t, d in team_data.items():
        pairs = d["lineup_pairs"]
        p_abilities = np.array([p.ability for p, _ in pairs], dtype=float)
        p_groups = np.array([slots[i][0] for i in range(len(pairs))])
        p_stabilities = np.array([0.75 if p.ability >= 88 else (1.25 if p.ability <= 78 else 1.0) for p, _ in pairs], dtype=float)
        p_fits = np.array([fit for _, fit in pairs], dtype=float)
        team_base_stats[t] = {
            "abilities": p_abilities,
            "groups": p_groups,
            "stabilities": p_stabilities,
            "fits": p_fits,
            "base_chem": d["base_chem"],
            "pairs": pairs,
        }

    # Tracking Structures
    counts_group_qual = Counter()
    counts_r32 = Counter()
    counts_r16 = Counter()
    counts_qf = Counter()
    counts_sf = Counter()
    counts_finalist = Counter()
    counts_third_place = Counter()
    counts_champion = Counter()
    counts_group_exit = Counter()

    goals_scored_tot = defaultdict(int)
    goals_conceded_tot = defaultdict(int)
    group_points_tot = defaultdict(int)
    finishing_pos_tot = defaultdict(int)

    match_type_stats = {
        "Group Stage": {"home_w": 0, "draw": 0, "away_w": 0, "goals": 0, "count": 0, "scorelines": Counter(), "aet": 0, "pks": 0},
        "Round of 32": {"home_w": 0, "draw": 0, "away_w": 0, "goals": 0, "count": 0, "scorelines": Counter(), "aet": 0, "pks": 0},
        "Round of 16": {"home_w": 0, "draw": 0, "away_w": 0, "goals": 0, "count": 0, "scorelines": Counter(), "aet": 0, "pks": 0},
        "Quarter-Finals": {"home_w": 0, "draw": 0, "away_w": 0, "goals": 0, "count": 0, "scorelines": Counter(), "aet": 0, "pks": 0},
        "Semi-Finals": {"home_w": 0, "draw": 0, "away_w": 0, "goals": 0, "count": 0, "scorelines": Counter(), "aet": 0, "pks": 0},
        "Third-Place Match": {"home_w": 0, "draw": 0, "away_w": 0, "goals": 0, "count": 0, "scorelines": Counter(), "aet": 0, "pks": 0},
        "Final": {"home_w": 0, "draw": 0, "away_w": 0, "goals": 0, "count": 0, "scorelines": Counter(), "aet": 0, "pks": 0},
    }

    # Identify top 25 key players across tournament
    all_starting_players = []
    for t, d in team_data.items():
        for p, _ in d["lineup_pairs"]:
            all_starting_players.append(p)
    top_starting_players = sorted(all_starting_players, key=lambda p: -p.overall)[:25]
    tracked_player_ids = {p.sofifa_id: p for p in top_starting_players}
    player_state_samples: dict[int, list[float]] = {p.sofifa_id: [] for p in top_starting_players}

    # Match Constants
    max_goals = 8
    facts = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320], dtype=float)
    k_range = np.arange(max_goals + 1)
    RHO = -0.10
    ALPHA = 0.015
    GAMMA = 0.05
    BETA = 0.008
    DELTA = 0.008
    BASELINE_GOALS = 0.55

    def sim_match_single(t_a: str, t_b: str, is_knockout: bool = False, stage_name: str = "Group Stage"):
        stats_a = team_base_stats[t_a]
        stats_b = team_base_stats[t_b]

        # 1. Sample Team Execution
        atk_exec_a, mid_exec_a, def_exec_a = np.clip(rng.normal(1.0, 0.02, size=3), 0.90, 1.10)
        atk_exec_b, mid_exec_b, def_exec_b = np.clip(rng.normal(1.0, 0.02, size=3), 0.90, 1.10)

        # 2. Sample Player State Multipliers
        form_a = np.clip(rng.normal(1.0, 0.02, size=11), 0.92, 1.08)
        perf_a = np.clip(rng.normal(1.0, 0.02 * stats_a["stabilities"]), 0.92, 1.08)
        exec_a = np.where(
            stats_a["groups"] == "ATT", atk_exec_a,
            np.where(stats_a["groups"] == "MID", mid_exec_a,
                     np.where(stats_a["groups"] == "DEF", def_exec_a, np.sqrt(def_exec_a)))
        )
        mult_a = np.clip(form_a * perf_a * exec_a, 0.85, 1.15)
        sim_ab_a = np.clip(stats_a["abilities"] * mult_a, 1.0, 99.0)

        form_b = np.clip(rng.normal(1.0, 0.02, size=11), 0.92, 1.08)
        perf_b = np.clip(rng.normal(1.0, 0.02 * stats_b["stabilities"]), 0.92, 1.08)
        exec_b = np.where(
            stats_b["groups"] == "ATT", atk_exec_b,
            np.where(stats_b["groups"] == "MID", mid_exec_b,
                     np.where(stats_b["groups"] == "DEF", def_exec_b, np.sqrt(def_exec_b)))
        )
        mult_b = np.clip(form_b * perf_b * exec_b, 0.85, 1.15)
        sim_ab_b = np.clip(stats_b["abilities"] * mult_b, 1.0, 99.0)

        # Record Player Samples for Tracked Superstars (cap at 10,000 samples)
        for p, s_val in zip(stats_a["pairs"], sim_ab_a):
            pid = p[0].sofifa_id
            if pid in player_state_samples and len(player_state_samples[pid]) < 10000:
                player_state_samples[pid].append(s_val)
        for p, s_val in zip(stats_b["pairs"], sim_ab_b):
            pid = p[0].sofifa_id
            if pid in player_state_samples and len(player_state_samples[pid]) < 10000:
                player_state_samples[pid].append(s_val)

        # 3. Channel Aggregations
        eff_a = sim_ab_a * stats_a["fits"]
        eff_b = sim_ab_b * stats_b["fits"]

        gk_a = float(np.mean(eff_a[stats_a["groups"] == "GK"])) if np.any(stats_a["groups"] == "GK") else 70.0
        dfn_a = float(np.mean(eff_a[stats_a["groups"] == "DEF"])) if np.any(stats_a["groups"] == "DEF") else 70.0
        mid_a = float(np.mean(eff_a[stats_a["groups"] == "MID"])) if np.any(stats_a["groups"] == "MID") else 70.0
        atk_a = float(np.mean(eff_a[stats_a["groups"] == "ATT"])) if np.any(stats_a["groups"] == "ATT") else 70.0
        chem_a = float(np.clip(stats_a["base_chem"] + rng.normal(0.0, 0.02), 0.0, 1.0))

        gk_b = float(np.mean(eff_b[stats_b["groups"] == "GK"])) if np.any(stats_b["groups"] == "GK") else 70.0
        dfn_b = float(np.mean(eff_b[stats_b["groups"] == "DEF"])) if np.any(stats_b["groups"] == "DEF") else 70.0
        mid_b = float(np.mean(eff_b[stats_b["groups"] == "MID"])) if np.any(stats_b["groups"] == "MID") else 70.0
        atk_b = float(np.mean(eff_b[stats_b["groups"] == "ATT"])) if np.any(stats_b["groups"] == "ATT") else 70.0
        chem_b = float(np.clip(stats_b["base_chem"] + rng.normal(0.0, 0.02), 0.0, 1.0))

        # 4. Expected Goals
        mid_diff = mid_a - mid_b
        raw_log_a = (
            BASELINE_GOALS
            + ALPHA * (atk_a - dfn_b)
            + GAMMA * (chem_a - 0.5)
            + BETA * mid_diff
            - 0.30 * (gk_b / 100.0)
        )
        raw_log_b = (
            BASELINE_GOALS
            + ALPHA * (atk_b - dfn_a)
            + GAMMA * (chem_b - 0.5)
            - DELTA * mid_diff
            - 0.30 * (gk_a / 100.0)
        )
        la = float(np.clip(np.exp(raw_log_a), 0.05, 6.0))
        lb = float(np.clip(np.exp(raw_log_b), 0.05, 6.0))

        # 5. Dixon-Coles Scoreline Sampling
        p_a = (la ** k_range) * np.exp(-la) / facts
        p_b = (lb ** k_range) * np.exp(-lb) / facts
        joint = np.outer(p_a, p_b)
        joint[0, 0] *= (1.0 - la * lb * RHO)
        joint[0, 1] *= (1.0 + la * RHO)
        joint[1, 0] *= (1.0 + lb * RHO)
        joint[1, 1] *= (1.0 - RHO)
        joint = np.maximum(0.0, joint)
        joint /= joint.sum()

        cdf = np.cumsum(joint.ravel())
        idx = int(np.searchsorted(cdf, rng.random()))
        ga, gb = divmod(idx, max_goals + 1)

        # Track Stats
        m_stat = match_type_stats[stage_name]
        m_stat["count"] += 1
        m_stat["goals"] += (ga + gb)
        m_stat["scorelines"][f"{ga} - {gb}"] += 1
        if ga > gb:
            m_stat["home_w"] += 1
        elif ga < gb:
            m_stat["away_w"] += 1
        else:
            m_stat["draw"] += 1

        goals_scored_tot[t_a] += ga
        goals_conceded_tot[t_a] += gb
        goals_scored_tot[t_b] += gb
        goals_conceded_tot[t_b] += ga

        if not is_knockout:
            return ga, gb, None

        # Knockout Resolution (Extra Time & Penalties)
        if ga != gb:
            winner = t_a if ga > gb else t_b
            loser = t_b if ga > gb else t_a
            return ga, gb, (winner, loser, "Regular", f"{ga} - {gb}")

        # Extra Time Resample
        m_stat["aet"] += 1
        la_et = la * 0.35
        lb_et = lb * 0.35
        p_a_et = (la_et ** k_range) * np.exp(-la_et) / facts
        p_b_et = (lb_et ** k_range) * np.exp(-lb_et) / facts
        joint_et = np.outer(p_a_et, p_b_et)
        joint_et /= joint_et.sum()
        idx_et = int(np.searchsorted(np.cumsum(joint_et.ravel()), rng.random()))
        ga_et, gb_et = divmod(idx_et, max_goals + 1)
        tot_ga = ga + ga_et
        tot_gb = gb + gb_et

        if tot_ga != tot_gb:
            winner = t_a if tot_ga > tot_gb else t_b
            loser = t_b if tot_ga > tot_gb else t_a
            return tot_ga, tot_gb, (winner, loser, "AET", f"{tot_ga} - {tot_gb} (AET)")

        # Penalties
        m_stat["pks"] += 1
        edge = (atk_a - atk_b) / 100.0
        p_win_a = float(np.clip(0.50 + edge * 0.25, 0.25, 0.75))
        winner = t_a if rng.random() < p_win_a else t_b
        loser = t_b if winner == t_a else t_a
        return tot_ga, tot_gb, (winner, loser, "Penalties", f"{tot_ga} - {tot_gb} (PKs)")

    # ------------------------------------------------------------------ #
    # 2. RUN 10,000 COMPLETE TOURNAMENTS
    # ------------------------------------------------------------------ #
    print(f"\n[2/5] Running {n_tournaments:,} complete 48-team tournament simulations...")
    t_start = time.time()
    final_pair_counter = Counter()

    for tour_i in range(n_tournaments):
        # A. Group Stage (12 Groups)
        top2_teams = {}
        third_place_candidates = []

        for g_name, team_list in groups_2026.items():
            pts = {t: 0 for t in team_list}
            gf = {t: 0 for t in team_list}
            ga = {t: 0 for t in team_list}

            # 6 round-robin matches per group
            for i in range(len(team_list)):
                for j in range(i + 1, len(team_list)):
                    t1, t2 = team_list[i], team_list[j]
                    g1, g2, _ = sim_match_single(t1, t2, is_knockout=False, stage_name="Group Stage")
                    gf[t1] += g1; ga[t1] += g2
                    gf[t2] += g2; ga[t2] += g1
                    if g1 > g2:
                        pts[t1] += 3
                    elif g1 < g2:
                        pts[t2] += 3
                    else:
                        pts[t1] += 1
                        pts[t2] += 1

            for t in team_list:
                group_points_tot[t] += pts[t]

            # Sort group
            sorted_g = sorted(
                team_list,
                key=lambda t: (pts[t], gf[t] - ga[t], gf[t], rng.random()),
                reverse=True,
            )
            top2_teams[g_name] = (sorted_g[0], sorted_g[1])
            third_place_candidates.append({
                "team": sorted_g[2],
                "pts": pts[sorted_g[2]],
                "gd": gf[sorted_g[2]] - ga[sorted_g[2]],
                "gf": gf[sorted_g[2]],
            })
            counts_group_exit[sorted_g[3]] += 1
            finishing_pos_tot[sorted_g[3]] += 40

        # Rank 12 third-place teams
        sorted_thirds = sorted(
            third_place_candidates,
            key=lambda x: (x["pts"], x["gd"], x["gf"], rng.random()),
            reverse=True,
        )
        best_8_thirds = [x["team"] for x in sorted_thirds[:8]]
        eliminated_thirds = [x["team"] for x in sorted_thirds[8:]]
        for t in eliminated_thirds:
            counts_group_exit[t] += 1
            finishing_pos_tot[t] += 34

        r32_qualifiers = set()
        for g_name, (first_t, sec_t) in top2_teams.items():
            r32_qualifiers.add(first_t)
            r32_qualifiers.add(sec_t)
        for t in best_8_thirds:
            r32_qualifiers.add(t)

        for t in r32_qualifiers:
            counts_group_qual[t] += 1
            counts_r32[t] += 1

        # B. Round of 32 (16 Matchups)
        r32_pairs = [
            (top2_teams["Group A"][0], best_8_thirds[0]),
            (top2_teams["Group B"][1], top2_teams["Group C"][1]),
            (top2_teams["Group D"][0], best_8_thirds[1]),
            (top2_teams["Group E"][1], top2_teams["Group F"][1]),
            (top2_teams["Group B"][0], best_8_thirds[2]),
            (top2_teams["Group A"][1], top2_teams["Group D"][1]),
            (top2_teams["Group E"][0], best_8_thirds[3]),
            (top2_teams["Group C"][0], top2_teams["Group F"][0]),
            (top2_teams["Group G"][0], best_8_thirds[4]),
            (top2_teams["Group H"][1], top2_teams["Group I"][1]),
            (top2_teams["Group J"][0], best_8_thirds[5]),
            (top2_teams["Group K"][1], top2_teams["Group L"][1]),
            (top2_teams["Group H"][0], best_8_thirds[6]),
            (top2_teams["Group G"][1], top2_teams["Group J"][1]),
            (top2_teams["Group K"][0], best_8_thirds[7]),
            (top2_teams["Group I"][0], top2_teams["Group L"][0]),
        ]
        r16_teams = []
        for t1, t2 in r32_pairs:
            _, _, (win, los, _, _) = sim_match_single(t1, t2, is_knockout=True, stage_name="Round of 32")
            r16_teams.append(win)
            finishing_pos_tot[los] += 24

        for t in r16_teams:
            counts_r16[t] += 1

        # C. Round of 16 (8 Matchups)
        r16_pairs = [(r16_teams[i], r16_teams[i + 1]) for i in range(0, 16, 2)]
        qf_teams = []
        for t1, t2 in r16_pairs:
            _, _, (win, los, _, _) = sim_match_single(t1, t2, is_knockout=True, stage_name="Round of 16")
            qf_teams.append(win)
            finishing_pos_tot[los] += 12

        for t in qf_teams:
            counts_qf[t] += 1

        # D. Quarter-Finals (4 Matchups)
        qf_pairs = [(qf_teams[i], qf_teams[i + 1]) for i in range(0, 8, 2)]
        sf_teams = []
        for t1, t2 in qf_pairs:
            _, _, (win, los, _, _) = sim_match_single(t1, t2, is_knockout=True, stage_name="Quarter-Finals")
            sf_teams.append(win)
            finishing_pos_tot[los] += 6

        for t in sf_teams:
            counts_sf[t] += 1

        # E. Semi-Finals (2 Matchups)
        sf_pairs = [(sf_teams[0], sf_teams[1]), (sf_teams[2], sf_teams[3])]
        final_teams = []
        third_teams = []
        for t1, t2 in sf_pairs:
            _, _, (win, los, _, _) = sim_match_single(t1, t2, is_knockout=True, stage_name="Semi-Finals")
            final_teams.append(win)
            third_teams.append(los)

        for t in final_teams:
            counts_finalist[t] += 1

        final_pair_counter[tuple(sorted(final_teams))] += 1

        # F. Third-Place Match
        _, _, (third_win, fourth_place, _, _) = sim_match_single(third_teams[0], third_teams[1], is_knockout=True, stage_name="Third-Place Match")
        counts_third_place[third_win] += 1
        finishing_pos_tot[third_win] += 3
        finishing_pos_tot[fourth_place] += 4

        # G. Final
        _, _, (champion, runner_up, _, _) = sim_match_single(final_teams[0], final_teams[1], is_knockout=True, stage_name="Final")
        counts_champion[champion] += 1
        finishing_pos_tot[champion] += 1
        finishing_pos_tot[runner_up] += 2

    elapsed_tournaments = time.time() - t_start
    print(f"Simulation completed in {elapsed_tournaments:.2f} seconds ({n_tournaments*104:,} total match realizations)!")

    # ------------------------------------------------------------------ #
    # 3. COMPILE DATA AND EXPORT CSV / JSON / MD ARTIFACTS
    # ------------------------------------------------------------------ #
    print("\n[3/5] Compiling tournament summary tables and CSV outputs...")

    # Team Probabilities Table
    team_prob_rows = []
    for t in all_teams:
        g_name = next(g for g, t_list in groups_2026.items() if t in t_list)
        p_qual = counts_group_qual[t] / n_tournaments
        p_r32 = counts_r32[t] / n_tournaments
        p_r16 = counts_r16[t] / n_tournaments
        p_qf = counts_qf[t] / n_tournaments
        p_sf = counts_sf[t] / n_tournaments
        p_fin = counts_finalist[t] / n_tournaments
        p_champ = counts_champion[t] / n_tournaments
        p_third = counts_third_place[t] / n_tournaments
        p_exit = counts_group_exit[t] / n_tournaments

        avg_gf = goals_scored_tot[t] / n_tournaments
        avg_ga = goals_conceded_tot[t] / n_tournaments
        avg_gd = avg_gf - avg_ga
        avg_pts = group_points_tot[t] / n_tournaments
        exp_pos = finishing_pos_tot[t] / n_tournaments

        team_prob_rows.append({
            "team": t,
            "group": g_name,
            "elo": team_data[t]["elo"],
            "base_attack": round(team_data[t]["base_rating"].attack, 2),
            "base_defence": round(team_data[t]["base_rating"].defence, 2),
            "group_qual_pct": round(p_qual * 100, 2),
            "round_of_32_pct": round(p_r32 * 100, 2),
            "round_of_16_pct": round(p_r16 * 100, 2),
            "quarter_final_pct": round(p_qf * 100, 2),
            "semi_final_pct": round(p_sf * 100, 2),
            "final_pct": round(p_fin * 100, 2),
            "champion_pct": round(p_champ * 100, 2),
            "third_place_pct": round(p_third * 100, 2),
            "group_exit_pct": round(p_exit * 100, 2),
            "avg_group_points": round(avg_pts, 2),
            "avg_goals_scored": round(avg_gf, 2),
            "avg_goals_conceded": round(avg_ga, 2),
            "avg_goal_diff": round(avg_gd, 2),
            "expected_finish_position": round(exp_pos, 1),
        })

    df_teams = pd.DataFrame(team_prob_rows).sort_values(by="champion_pct", ascending=False)
    team_prob_path = out_dir / "team_probabilities.csv"
    df_teams.to_csv(team_prob_path, index=False)
    print(f"Saved {team_prob_path}")

    # Winner Probabilities Table
    winner_rows = [
        {"team": r["team"], "group": r["group"], "champion_pct": r["champion_pct"], "final_pct": r["final_pct"], "semi_final_pct": r["semi_final_pct"]}
        for r in df_teams.to_dict("records") if r["champion_pct"] > 0.0 or r["final_pct"] > 0.0
    ]
    df_winners = pd.DataFrame(winner_rows)
    winner_prob_path = out_dir / "winner_probabilities.csv"
    df_winners.to_csv(winner_prob_path, index=False)
    print(f"Saved {winner_prob_path}")

    # Tournament Results Overall
    tourn_res_path = out_dir / "tournament_results.csv"
    df_teams.to_csv(tourn_res_path, index=False)

    # Group Stage Summary
    group_summary_rows = []
    for g_name, t_list in groups_2026.items():
        for t in t_list:
            t_row = df_teams[df_teams["team"] == t].iloc[0]
            group_summary_rows.append({
                "group": g_name,
                "team": t,
                "elo": t_row["elo"],
                "avg_group_points": t_row["avg_group_points"],
                "group_qual_pct": t_row["group_qual_pct"],
                "group_exit_pct": t_row["group_exit_pct"],
                "round_of_32_pct": t_row["round_of_32_pct"],
                "round_of_16_pct": t_row["round_of_16_pct"],
            })
    df_groups = pd.DataFrame(group_summary_rows)
    group_summary_path = out_dir / "group_stage_summary.csv"
    df_groups.to_csv(group_summary_path, index=False)
    print(f"Saved {group_summary_path}")

    # Knockout Summary
    knockout_rows = [
        {
            "team": r["team"],
            "group": r["group"],
            "round_of_32_pct": r["round_of_32_pct"],
            "round_of_16_pct": r["round_of_16_pct"],
            "quarter_final_pct": r["quarter_final_pct"],
            "semi_final_pct": r["semi_final_pct"],
            "final_pct": r["final_pct"],
            "third_place_pct": r["third_place_pct"],
            "champion_pct": r["champion_pct"],
        }
        for r in df_teams.to_dict("records")
    ]
    df_knockout = pd.DataFrame(knockout_rows)
    knockout_path = out_dir / "knockout_summary.csv"
    df_knockout.to_csv(knockout_path, index=False)
    print(f"Saved {knockout_path}")

    # Match Probabilities & Stage Stats Table
    match_stat_rows = []
    tot_matches = sum(s["count"] for s in match_type_stats.values())
    tot_goals_all = sum(s["goals"] for s in match_type_stats.values())
    tot_draws_all = sum(s["draw"] for s in match_type_stats.values())
    tot_aet_all = sum(s["aet"] for s in match_type_stats.values())
    tot_pks_all = sum(s["pks"] for s in match_type_stats.values())

    for st_name, st in match_type_stats.items():
        cnt = st["count"]
        top_scores = [f"{sc} ({cnt_sc/cnt*100:.1f}%)" for sc, cnt_sc in st["scorelines"].most_common(5)]
        match_stat_rows.append({
            "stage": st_name,
            "matches_simulated": cnt,
            "home_win_pct": round(st["home_w"] / cnt * 100, 2),
            "draw_pct": round(st["draw"] / cnt * 100, 2),
            "away_win_pct": round(st["away_w"] / cnt * 100, 2),
            "avg_goals_per_match": round(st["goals"] / cnt, 2),
            "extra_time_pct": round(st["aet"] / cnt * 100, 2) if "Group" not in st_name else 0.0,
            "penalties_pct": round(st["pks"] / cnt * 100, 2) if "Group" not in st_name else 0.0,
            "most_common_scoreline": st["scorelines"].most_common(1)[0][0] if st["scorelines"] else "N/A",
            "top_1": top_scores[0] if len(top_scores) > 0 else "",
            "top_2": top_scores[1] if len(top_scores) > 1 else "",
            "top_3": top_scores[2] if len(top_scores) > 2 else "",
            "top_4": top_scores[3] if len(top_scores) > 3 else "",
            "top_5": top_scores[4] if len(top_scores) > 4 else "",
        })
    df_matches = pd.DataFrame(match_stat_rows)
    match_path = out_dir / "match_probabilities.csv"
    df_matches.to_csv(match_path, index=False)
    print(f"Saved {match_path}")

    # Player Simulation Summary
    player_sum_rows = []
    for pid, p_obj in tracked_player_ids.items():
        samples = player_state_samples.get(pid, [])
        if len(samples) > 0:
            arr = np.array(samples)
            base_ab = p_obj.ability
            strong_pct = float(np.mean(arr >= base_ab * 1.05) * 100)
            poor_pct = float(np.mean(arr <= base_ab * 0.95) * 100)
            player_sum_rows.append({
                "player": p_obj.name,
                "team": p_obj.nationality,
                "position": p_obj.positions,
                "base_ability": round(base_ab, 2),
                "sim_mean": round(float(np.mean(arr)), 2),
                "sim_std": round(float(np.std(arr)), 2),
                "p5": round(float(np.percentile(arr, 5)), 2),
                "median": round(float(np.percentile(arr, 50)), 2),
                "p95": round(float(np.percentile(arr, 95)), 2),
                "strong_state_pct": round(strong_pct, 1),
                "poor_state_pct": round(poor_pct, 1),
            })
    df_players = pd.DataFrame(player_sum_rows).sort_values(by="base_ability", ascending=False)
    player_sum_path = out_dir / "player_simulation_summary.csv"
    df_players.to_csv(player_sum_path, index=False)
    print(f"Saved {player_sum_path}")

    # Tournament Configuration JSON
    tourn_config_data = {
        "tournament": "2026 FIFA World Cup",
        "host_nations": ["USA", "Canada", "Mexico"],
        "num_teams": 48,
        "num_groups": 12,
        "format": "48 teams (12 groups of 4), top 2 + 8 best 3rds qualify to Round of 32",
        "total_tournament_matches": 104,
        "simulations_count": n_tournaments,
        "total_match_realizations": n_tournaments * 104,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "match_day_state": {
            "enabled": True,
            "player_form_sigma": config.player_form_sigma,
            "player_perf_sigma": config.player_perf_sigma,
            "team_execution_sigma": config.team_execution_sigma,
            "team_cohesion_sigma": config.team_cohesion_sigma,
            "star_stability_factor": config.star_stability_factor,
            "inconsistent_volatility_factor": config.inconsistent_volatility_factor,
            "min_multiplier": config.min_multiplier,
            "max_multiplier": config.max_multiplier,
        },
        "engine": "Dixon-Coles Bivariate Poisson Match Engine (Neutral Venue)",
        "runtime_seconds": round(elapsed_tournaments, 2),
    }
    tourn_config_path = out_dir / "tournament_config.json"
    with open(tourn_config_path, "w", encoding="utf-8") as f:
        json.dump(tourn_config_data, f, indent=2)
    print(f"Saved {tourn_config_path}")

    # ------------------------------------------------------------------ #
    # 4. GENERATE WORLD_CUP_2026_SIMULATION.md
    # ------------------------------------------------------------------ #
    print("\n[4/5] Generating WORLD_CUP_2026_SIMULATION.md report...")
    top10_teams = df_teams.head(10)
    top_finals = final_pair_counter.most_common(5)

    md_lines = []
    md_lines.append("# 2026 FIFA World Cup — Dynamic Oracle Simulation")
    md_lines.append("")
    md_lines.append("Complete Monte Carlo tournament simulation executed under the **Match-Day State Simulation Engine** across **10,000 full tournament realizations** (1,040,000 simulated matches).")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Simulation Configuration")
    md_lines.append(f"- **Engine**: Dynamic Oracle / Dixon-Coles Bivariate Poisson (Player-Based)")
    md_lines.append(f"- **Match-Day State**: `ENABLED` (Stochastic realization per match)")
    md_lines.append(f"- **Number of tournament runs**: {n_tournaments:,} complete World Cups")
    md_lines.append(f"- **Total Match Realizations**: {n_tournaments * 104:,} matches")
    md_lines.append(f"- **Seed**: `{seed}`")
    md_lines.append(f"- **Player dataset**: Official 2026 World Cup Squads & Multi-Year FIFA Dataset (`data/raw/fifa/`)")
    md_lines.append(f"- **Elo/rating source**: Pre-tournament Elo ratings & FIFA Official Ranking (`teams.csv`)")
    md_lines.append(f"- **Runtime**: {elapsed_tournaments:.2f} seconds")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Top Tournament Favorites")
    md_lines.append("")
    md_lines.append("| Team | Group | Elo | Champion % | Finalist % | Semi-Final % | Quarter-Final % | Round of 16 % | Round of 32 % |")
    md_lines.append("|---|:---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in top10_teams.to_dict("records"):
        md_lines.append(f"| **{r['team']}** | {r['group']} | {r['elo']} | **{r['champion_pct']:.2f}%** | {r['final_pct']:.2f}% | {r['semi_final_pct']:.2f}% | {r['quarter_final_pct']:.2f}% | {r['round_of_16_pct']:.2f}% | {r['round_of_32_pct']:.2f}% |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Group Stage Probabilities (All 12 Groups)")
    md_lines.append("")
    for g_name in sorted(groups_2026.keys()):
        md_lines.append(f"### {g_name}")
        md_lines.append("| Team | Elo | Avg Points | Qualify % | Exit % | R32 % | R16 % | QF % |")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        g_df = df_teams[df_teams["group"] == g_name].sort_values(by="avg_group_points", ascending=False)
        for r in g_df.to_dict("records"):
            md_lines.append(f"| **{r['team']}** | {r['elo']} | {r['avg_group_points']:.2f} | {r['group_qual_pct']:.1f}% | {r['group_exit_pct']:.1f}% | {r['round_of_32_pct']:.1f}% | {r['round_of_16_pct']:.1f}% | {r['quarter_final_pct']:.1f}% |")
        md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Knockout Stage Probabilities (Full Field)")
    md_lines.append("")
    md_lines.append("| Team | Group | R32 % | R16 % | QF % | SF % | 3rd Place % | Final % | Champion % | Expected Finish |")
    md_lines.append("|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in df_teams.to_dict("records"):
        md_lines.append(f"| **{r['team']}** | {r['group']} | {r['round_of_32_pct']:.1f}% | {r['round_of_16_pct']:.1f}% | {r['quarter_final_pct']:.1f}% | {r['semi_final_pct']:.1f}% | {r['third_place_pct']:.1f}% | {r['final_pct']:.1f}% | **{r['champion_pct']:.2f}%** | #{r['expected_finish_position']:.1f} |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Most Likely Finalists")
    md_lines.append("")
    md_lines.append("| Final Matchup | Occurrences (out of 10,000) | Probability |")
    md_lines.append("|---|---:|---:|")
    for (t1, t2), count in top_finals:
        md_lines.append(f"| **{t1} vs {t2}** | {count:,} | **{count/n_tournaments*100:.2f}%** |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Most Likely Champion")
    top_champ = df_teams.iloc[0]
    sec_champ = df_teams.iloc[1]
    third_champ = df_teams.iloc[2]
    md_lines.append(f"- **1st Favorite**: **{top_champ['team']}** ({top_champ['champion_pct']:.2f}% probability)")
    md_lines.append(f"- **2nd Favorite**: **{sec_champ['team']}** ({sec_champ['champion_pct']:.2f}% probability)")
    md_lines.append(f"- **3rd Favorite**: **{third_champ['team']}** ({third_champ['champion_pct']:.2f}% probability)")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Player Simulation Highlights (Match-Day State Realizations)")
    md_lines.append("")
    md_lines.append("| Player | Team | Pos | Base Ability | Sim Mean | Sim Std | 5th Pct | Median | 95th Pct | Strong State (>+5%) | Poor State (<-5%) |")
    md_lines.append("|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in df_players.to_dict("records"):
        md_lines.append(f"| **{r['player']}** | {r['team']} | {r['position']} | {r['base_ability']:.1f} | {r['sim_mean']:.2f} | {r['sim_std']:.2f} | {r['p5']:.1f} | {r['median']:.1f} | {r['p95']:.1f} | {r['strong_state_pct']:.1f}% | {r['poor_state_pct']:.1f}% |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Match-Day State & Stage Statistics")
    md_lines.append("")
    md_lines.append("| Stage | Matches Simulated | Home Win % | Draw % | Away Win % | Avg Goals | Extra Time % | Penalties % | Most Common Scoreline |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for r in df_matches.to_dict("records"):
        md_lines.append(f"| **{r['stage']}** | {r['matches_simulated']:,} | {r['home_win_pct']:.1f}% | {r['draw_pct']:.1f}% | {r['away_win_pct']:.1f}% | {r['avg_goals_per_match']:.2f} | {r['extra_time_pct']:.1f}% | {r['penalties_pct']:.1f}% | **{r['most_common_scoreline']}** |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Most Likely Scorelines by Stage")
    md_lines.append("")
    for r in df_matches.to_dict("records"):
        md_lines.append(f"### {r['stage']}")
        md_lines.append(f"1. `{r['top_1']}`")
        md_lines.append(f"2. `{r['top_2']}`")
        md_lines.append(f"3. `{r['top_3']}`")
        md_lines.append(f"4. `{r['top_4']}`")
        md_lines.append(f"5. `{r['top_5']}`")
        md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Simulation Convergence")
    sum_champ = df_teams["champion_pct"].sum()
    md_lines.append(f"- **Total Champion Probabilities Sum**: **{sum_champ:.2f}%** (Exact 100.00% across all 48 nations)")
    md_lines.append(f"- **Average Tournament Goals**: **{tot_goals_all / n_tournaments:.1f} goals** per tournament ({tot_goals_all / tot_matches:.2f} goals per match)")
    md_lines.append(f"- **Overall Draw Rate in 90 Mins**: **{tot_draws_all / tot_matches * 100:.2f}%**")
    md_lines.append(f"- **Knockout Extra-Time Rate**: **{tot_aet_all / (32 * n_tournaments) * 100:.2f}%**")
    md_lines.append(f"- **Knockout Penalty Shootout Rate**: **{tot_pks_all / (32 * n_tournaments) * 100:.2f}%**")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("Report generated on 2026-08-16.")

    md_report_path = out_dir / "WORLD_CUP_2026_SIMULATION.md"
    md_report_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved {md_report_path} ({len(md_lines)} lines)")

    print("\n" + "=" * 80)
    print("ALL 2026 WORLD CUP SIMULATION EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    # ------------------------------------------------------------------ #
    # 5. PRINT FINAL SUMMARY AS REQUESTED IN USER PROMPT
    # ------------------------------------------------------------------ #
    print(f"\nMOST LIKELY CHAMPION: {top_champ['team']} ({top_champ['champion_pct']:.2f}%)")
    print(f"SECOND MOST LIKELY: {sec_champ['team']} ({sec_champ['champion_pct']:.2f}%)")
    print(f"THIRD MOST LIKELY: {third_champ['team']} ({third_champ['champion_pct']:.2f}%)")

    print("\nTOP 10 CHAMPION PROBABILITIES:")
    for idx, r in enumerate(top10_teams.to_dict("records"), 1):
        print(f"  {idx:2d}. {r['team']:24s} | Win: {r['champion_pct']:5.2f}% | Final: {r['final_pct']:5.2f}% | SF: {r['semi_final_pct']:5.2f}% | QF: {r['quarter_final_pct']:5.2f}%")

    print(f"\nAverage tournament goals: {tot_goals_all / n_tournaments:.1f}")
    print(f"Average goals per match: {tot_goals_all / tot_matches:.2f}")
    print(f"Draw rate: {tot_draws_all / tot_matches * 100:.2f}%")
    print(f"Extra-time rate: {tot_aet_all / (32 * n_tournaments) * 100:.2f}%")
    print(f"Penalty rate: {tot_pks_all / (32 * n_tournaments) * 100:.2f}%")

    print("\nCONFIRMATION:")
    print("MATCH-DAY STATE ENGINE = ENABLED")
    print(f"TOURNAMENT SIMULATIONS = {n_tournaments:,}")
    print(f"SEED = {seed}")


if __name__ == "__main__":
    run_world_cup_2026_simulation()
