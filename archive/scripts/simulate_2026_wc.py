"""Fast Monte Carlo Simulator for 2026 FIFA World Cup (USA/Canada/Mexico 48-team format) using our model."""

import sys
from collections import Counter
import numpy as np
import pandas as pd

from src.service.oracle import load_oracle
from src.simulation.squad_model import SquadModel, TeamRating
from src.simulation.match_engine import MatchEngine, MatchEngineConfig

def run_2026_world_cup_simulation():
    oracle = load_oracle()
    engine = MatchEngine(MatchEngineConfig(home_advantage=0.0, baseline_goals=0.55), seed=2026)

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

    # Build TeamRatings for all 48 teams
    ratings = {}
    for g_name, team_list in groups_2026.items():
        for t in team_list:
            key_match = None
            team_dict = oracle.players_by_year_team.get(2026, oracle.players_by_year_team[2022])
            for k in team_dict:
                if t.lower() in k.lower() or k.lower() in t.lower() or ("ivo" in t.lower() and "ivo" in k.lower()) or ("turk" in t.lower() and "turk" in k.lower()) or ("curac" in t.lower() and "curac" in k.lower()):
                    key_match = k
                    break
            if not key_match:
                key_match = t
            try:
                pool, yr = oracle._get_player_pool(key_match, 2026)
            except Exception:
                pool, yr = oracle._get_player_pool(key_match, 2022)
            squad_model = SquadModel(formation="4-3-3")
            lineup = squad_model.select_lineup(pool)
            chem = oracle.chemistry_model.team_chemistry(t, [p for p, _ in lineup])
            rating = squad_model.aggregate(t, pool, chemistry_score=chem)
            ratings[t] = rating

    # Precompute pairwise match probabilities across all 48 teams
    print("[Simulator] Precomputing match probabilities across all 48 teams...", flush=True)
    all_teams = [t for group in groups_2026.values() for t in group]
    pair_probs = {}

    for i, t1 in enumerate(all_teams):
        for j, t2 in enumerate(all_teams):
            if i != j:
                p_win, p_draw, p_loss = engine.match_probabilities(ratings[t1], ratings[t2], neutral=True)
                pair_probs[(t1, t2)] = (p_win, p_draw, p_loss)

    # 1. Single Tournament Realization (Seed 2026)
    group_results = {}
    top2_per_group = {}
    third_placed = []

    for g_name, team_list in groups_2026.items():
        table = {t: {"pts": 0, "gd": 0, "gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0} for t in team_list}
        for i in range(len(team_list)):
            for j in range(i + 1, len(team_list)):
                t1, t2 = team_list[i], team_list[j]
                g1, g2 = engine.sample_scoreline(ratings[t1], ratings[t2], neutral=True)
                table[t1]["gf"] += g1; table[t1]["ga"] += g2; table[t1]["gd"] += (g1 - g2)
                table[t2]["gf"] += g2; table[t2]["ga"] += g1; table[t2]["gd"] += (g2 - g1)
                if g1 > g2: table[t1]["pts"] += 3; table[t1]["w"] += 1; table[t2]["l"] += 1
                elif g1 < g2: table[t2]["pts"] += 3; table[t2]["w"] += 1; table[t1]["l"] += 1
                else: table[t1]["pts"] += 1; table[t1]["d"] += 1; table[t2]["pts"] += 1; table[t2]["d"] += 1
        
        sorted_teams = sorted(
            team_list,
            key=lambda t: (table[t]["pts"], table[t]["gd"], table[t]["gf"]),
            reverse=True
        )
        group_results[g_name] = [(t, table[t]) for t in sorted_teams]
        top2_per_group[g_name] = (sorted_teams[0], sorted_teams[1])
        third_placed.append((sorted_teams[2], table[sorted_teams[2]]))

    # Best 8 third-placed teams qualify to Round of 32 (24 + 8 = 32 teams)
    sorted_thirds = sorted(third_placed, key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"]), reverse=True)
    best_8_thirds = [t for t, _ in sorted_thirds[:8]]

    def play_ko(t1, t2):
        g1, g2 = engine.sample_scoreline(ratings[t1], ratings[t2], neutral=True)
        if g1 > g2:
            return t1, f"{g1} - {g2}"
        elif g1 < g2:
            return t2, f"{g1} - {g2}"
        else:
            g1_et, g2_et = engine.sample_scoreline(ratings[t1], ratings[t2], neutral=True)
            if g1_et > g2_et:
                return t1, f"{g1+1} - {g2} (AET)"
            elif g1_et < g2_et:
                return t2, f"{g1} - {g2+1} (AET)"
            else:
                winner = t1 if ratings[t1].attack >= ratings[t2].attack else t2
                return winner, f"{g1} - {g2} (PKs)"

    # Round of 32 (16 Matchups)
    # Structure pairing 1st places, 2nd places, and the 8 best 3rds
    r32_pairs = [
        (top2_per_group["Group A"][0], best_8_thirds[0]),
        (top2_per_group["Group B"][1], top2_per_group["Group C"][1]),
        (top2_per_group["Group D"][0], best_8_thirds[1]),
        (top2_per_group["Group E"][1], top2_per_group["Group F"][1]),
        (top2_per_group["Group B"][0], best_8_thirds[2]),
        (top2_per_group["Group A"][1], top2_per_group["Group D"][1]),
        (top2_per_group["Group E"][0], best_8_thirds[3]),
        (top2_per_group["Group C"][0], top2_per_group["Group F"][0]),
        (top2_per_group["Group G"][0], best_8_thirds[4]),
        (top2_per_group["Group H"][1], top2_per_group["Group I"][1]),
        (top2_per_group["Group J"][0], best_8_thirds[5]),
        (top2_per_group["Group K"][1], top2_per_group["Group L"][1]),
        (top2_per_group["Group H"][0], best_8_thirds[6]),
        (top2_per_group["Group G"][1], top2_per_group["Group J"][1]),
        (top2_per_group["Group K"][0], best_8_thirds[7]),
        (top2_per_group["Group I"][0], top2_per_group["Group L"][0]),
    ]
    r32_results = []
    r16_teams = []
    for t1, t2 in r32_pairs:
        win, score = play_ko(t1, t2)
        r32_results.append({"match": f"{t1} vs {t2}", "winner": win, "score": score})
        r16_teams.append(win)

    # Round of 16 (8 Matchups)
    r16_pairs = [(r16_teams[i], r16_teams[i+1]) for i in range(0, 16, 2)]
    r16_results = []
    qf_teams = []
    for t1, t2 in r16_pairs:
        win, score = play_ko(t1, t2)
        r16_results.append({"match": f"{t1} vs {t2}", "winner": win, "score": score})
        qf_teams.append(win)

    # Quarter-Finals (4 Matchups)
    qf_pairs = [(qf_teams[i], qf_teams[i+1]) for i in range(0, 8, 2)]
    qf_results = []
    sf_teams = []
    for t1, t2 in qf_pairs:
        win, score = play_ko(t1, t2)
        qf_results.append({"match": f"{t1} vs {t2}", "winner": win, "score": score})
        sf_teams.append(win)

    # Semi-Finals (2 Matchups)
    sf_pairs = [(sf_teams[0], sf_teams[1]), (sf_teams[2], sf_teams[3])]
    sf_results = []
    final_teams = []
    third_place_teams = []
    for t1, t2 in sf_pairs:
        win, score = play_ko(t1, t2)
        loser = t2 if win == t1 else t1
        sf_results.append({"match": f"{t1} vs {t2}", "winner": win, "score": score})
        final_teams.append(win)
        third_place_teams.append(loser)

    # 3rd Place Match
    third_winner, third_score = play_ko(third_place_teams[0], third_place_teams[1])
    fourth_place = third_place_teams[1] if third_winner == third_place_teams[0] else third_place_teams[0]

    # Final
    champion, final_score = play_ko(final_teams[0], final_teams[1])
    runner_up = final_teams[1] if champion == final_teams[0] else final_teams[0]

    # 2. Monte Carlo 5,000 runs
    print("[Simulator] Running 5,000 Monte Carlo tournament iterations for 48 teams...", flush=True)
    champ_counts = Counter()
    finalist_counts = Counter()
    semi_counts = Counter()
    qf_counts = Counter()
    r16_counts = Counter()
    r32_counts = Counter()
    exit_counts = Counter()

    rng = np.random.default_rng(2026)
    n_sims = 5000

    for _ in range(n_sims):
        mc_top2 = {}
        mc_3rds = []
        for g_name, t_list in groups_2026.items():
            pts = {t: 0 for t in t_list}
            for i in range(len(t_list)):
                for j in range(i + 1, len(t_list)):
                    t1, t2 = t_list[i], t_list[j]
                    pw, pd, pl = pair_probs[(t1, t2)]
                    r = rng.random()
                    if r < pw: pts[t1] += 3
                    elif r < pw + pd: pts[t1] += 1; pts[t2] += 1
                    else: pts[t2] += 3
            ranked = sorted(t_list, key=lambda t: pts[t] + rng.random()*0.1, reverse=True)
            mc_top2[g_name] = (ranked[0], ranked[1])
            mc_3rds.append(ranked[2])
            exit_counts[ranked[3]] += 1
            r32_counts[ranked[0]] += 1
            r32_counts[ranked[1]] += 1

        best_8 = mc_3rds[:8]
        for t in best_8:
            r32_counts[t] += 1
        for t in mc_3rds[8:]:
            exit_counts[t] += 1

        mc_r32_pairs = [
            (mc_top2["Group A"][0], best_8[0]),
            (mc_top2["Group B"][1], mc_top2["Group C"][1]),
            (mc_top2["Group D"][0], best_8[1]),
            (mc_top2["Group E"][1], mc_top2["Group F"][1]),
            (mc_top2["Group B"][0], best_8[2]),
            (mc_top2["Group A"][1], mc_top2["Group D"][1]),
            (mc_top2["Group E"][0], best_8[3]),
            (mc_top2["Group C"][0], mc_top2["Group F"][0]),
            (mc_top2["Group G"][0], best_8[4]),
            (mc_top2["Group H"][1], mc_top2["Group I"][1]),
            (mc_top2["Group J"][0], best_8[5]),
            (mc_top2["Group K"][1], mc_top2["Group L"][1]),
            (mc_top2["Group H"][0], best_8[6]),
            (mc_top2["Group G"][1], mc_top2["Group J"][1]),
            (mc_top2["Group K"][0], best_8[7]),
            (mc_top2["Group I"][0], mc_top2["Group L"][0]),
        ]
        mc_r16_w = []
        for t1, t2 in mc_r32_pairs:
            pw, pd, pl = pair_probs[(t1, t2)]
            p_t1_adv = pw + pd * (0.5 + (ratings[t1].attack - ratings[t2].attack)/200.0)
            winner = t1 if rng.random() < p_t1_adv else t2
            mc_r16_w.append(winner)
            r16_counts[winner] += 1

        mc_qf_w = []
        for i in range(0, 16, 2):
            t1, t2 = mc_r16_w[i], mc_r16_w[i+1]
            pw, pd, pl = pair_probs[(t1, t2)]
            p_t1_adv = pw + pd * (0.5 + (ratings[t1].attack - ratings[t2].attack)/200.0)
            winner = t1 if rng.random() < p_t1_adv else t2
            mc_qf_w.append(winner)
            qf_counts[winner] += 1

        mc_sf_w = []
        for i in range(0, 8, 2):
            t1, t2 = mc_qf_w[i], mc_qf_w[i+1]
            pw, pd, pl = pair_probs[(t1, t2)]
            p_t1_adv = pw + pd * (0.5 + (ratings[t1].attack - ratings[t2].attack)/200.0)
            winner = t1 if rng.random() < p_t1_adv else t2
            mc_sf_w.append(winner)
            semi_counts[winner] += 1

        f_w = []
        for i in range(0, 4, 2):
            t1, t2 = mc_sf_w[i], mc_sf_w[i+1]
            pw, pd, pl = pair_probs[(t1, t2)]
            p_t1_adv = pw + pd * (0.5 + (ratings[t1].attack - ratings[t2].attack)/200.0)
            winner = t1 if rng.random() < p_t1_adv else t2
            f_w.append(winner)
            finalist_counts[winner] += 1

        t1, t2 = f_w[0], f_w[1]
        pw, pd, pl = pair_probs[(t1, t2)]
        p_t1_adv = pw + pd * (0.5 + (ratings[t1].attack - ratings[t2].attack)/200.0)
        champ = t1 if rng.random() < p_t1_adv else t2
        champ_counts[champ] += 1

    return {
        "groups": group_results,
        "r32": r32_results,
        "r16": r16_results,
        "qf": qf_results,
        "sf": sf_results,
        "third_place": {"match": f"{third_place_teams[0]} vs {third_place_teams[1]}", "winner": third_winner, "fourth": fourth_place, "score": third_score},
        "final": {"match": f"{final_teams[0]} vs {final_teams[1]}", "champion": champion, "runner_up": runner_up, "score": final_score},
        "mc_champ": champ_counts,
        "mc_final": finalist_counts,
        "mc_semi": semi_counts,
        "mc_qf": qf_counts,
        "mc_r16": r16_counts,
        "mc_r32": r32_counts,
        "mc_exit": exit_counts,
        "n_sims": n_sims,
    }

if __name__ == "__main__":
    res = run_2026_world_cup_simulation()
    
    print("\n" + "="*80)
    print("[2026 FIFA WORLD CUP (USA / CANADA / MEXICO) - 48 TEAMS SIMULATION RESULTS]")
    print("="*80)
    
    for g_name, table in res["groups"].items():
        print(f"\n--- {g_name} Standings ---")
        print(f"{'Pos':<4}{'Team':<26}{'P':<4}{'W':<4}{'D':<4}{'L':<4}{'GF':<4}{'GA':<4}{'GD':<4}{'Pts':<4}")
        for pos, (t, stats) in enumerate(table, start=1):
            print(f"{pos:<4}{t:<26}{stats['w']+stats['d']+stats['l']:<4}{stats['w']:<4}{stats['d']:<4}{stats['l']:<4}{stats['gf']:<4}{stats['ga']:<4}{stats['gd']:<4}{stats['pts']:<4}")

    print("\n" + "="*80)
    print("[KNOCKOUT BRACKET PREDICTIONS (32-TEAM BRACKET)]")
    print("="*80)
    print("\n--- Round of 32 (16 Matches) ---")
    for r in res["r32"]:
        print(f"  {r['match']:<42} -> Winner: {r['winner']:<16} ({r['score']})")

    print("\n--- Round of 16 (8 Matches) ---")
    for r in res["r16"]:
        print(f"  {r['match']:<42} -> Winner: {r['winner']:<16} ({r['score']})")

    print("\n--- Quarter-Finals (4 Matches) ---")
    for r in res["qf"]:
        print(f"  {r['match']:<42} -> Winner: {r['winner']:<16} ({r['score']})")

    print("\n--- Semi-Finals (2 Matches) ---")
    for r in res["sf"]:
        print(f"  {r['match']:<42} -> Winner: {r['winner']:<16} ({r['score']})")

    print(f"\n--- 3rd Place Play-off (Miami / Hard Rock Stadium) ---")
    print(f"  {res['third_place']['match']:<42} -> 3rd Place: {res['third_place']['winner']:<12} ({res['third_place']['score']})")

    print(f"\n--- [WORLD CUP FINAL (MetLife Stadium, New Jersey)] ---")
    print(f"  {res['final']['match']:<42} -> CHAMPION: {res['final']['champion']} ({res['final']['score']})")
    print(f"                                             Runner-up: {res['final']['runner_up']}")
    print(f"                                             3rd Place : {res['third_place']['winner']}")
    print(f"                                             4th Place : {res['third_place']['fourth']}")

    print("\n" + "="*80)
    print("[5,000 MONTE CARLO TOURNAMENT PROBABILITIES (TOP 20 NATIONS)]")
    print("="*80)
    print(f"{'Team':<24}{'P(Winner)':<12}{'P(Final)':<12}{'P(Semi)':<12}{'P(QF)':<12}{'P(R16)':<12}{'P(Exit)':<12}")
    for t, c in res["mc_champ"].most_common(20):
        pw = (c / res['n_sims']) * 100
        pf = (res["mc_final"][t] / res['n_sims']) * 100
        ps = (res["mc_semi"][t] / res['n_sims']) * 100
        pq = (res["mc_qf"][t] / res['n_sims']) * 100
        pr = (res["mc_r16"][t] / res['n_sims']) * 100
        pe = (res["mc_exit"][t] / res['n_sims']) * 100
        print(f"{t:<24}{pw:>7.1f}%{pf:>11.1f}%{ps:>11.1f}%{pq:>11.1f}%{pr:>11.1f}%{pe:>11.1f}%")
