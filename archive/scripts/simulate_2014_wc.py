"""Fast Monte Carlo Simulator for 2014 FIFA World Cup using our model."""

import sys
from collections import Counter
import numpy as np
import pandas as pd

from src.service.oracle import load_oracle
from src.simulation.squad_model import SquadModel, TeamRating
from src.simulation.match_engine import MatchEngine, MatchEngineConfig

def run_2014_world_cup_simulation():
    oracle = load_oracle()
    engine = MatchEngine(MatchEngineConfig(home_advantage=0.0, baseline_goals=0.55), seed=2014)

    groups_2014 = {
        "Group A": ["Brazil", "Croatia", "Mexico", "Cameroon"],
        "Group B": ["Spain", "Netherlands", "Chile", "Australia"],
        "Group C": ["Colombia", "Greece", "Côte d'Ivoire", "Japan"],
        "Group D": ["Uruguay", "Costa Rica", "England", "Italy"],
        "Group E": ["Switzerland", "Ecuador", "France", "Honduras"],
        "Group F": ["Argentina", "Bosnia and Herzegovina", "Iran", "Nigeria"],
        "Group G": ["Germany", "Portugal", "Ghana", "United States"],
        "Group H": ["Belgium", "Algeria", "Russia", "Korea Republic"]
    }

    # Build TeamRatings for all 32 teams using 2015 edition (corresponds to 2014 post-WC season)
    ratings = {}
    for g_name, team_list in groups_2014.items():
        for t in team_list:
            key_match = None
            for k in oracle.players_by_year_team[2015]:
                if t.lower() in k.lower() or k.lower() in t.lower() or ("ivo" in t.lower() and "ivo" in k.lower()) or ("bosnia" in t.lower() and "bosnia" in k.lower()):
                    key_match = k
                    break
            if not key_match:
                key_match = t
            pool, yr = oracle._get_player_pool(key_match, 2015)
            squad_model = SquadModel(formation="4-3-3")
            lineup = squad_model.select_lineup(pool)
            chem = oracle.chemistry_model.team_chemistry(t, [p for p, _ in lineup])
            rating = squad_model.aggregate(t, pool, chemistry_score=chem)
            ratings[t] = rating

    # Precompute pairwise match probabilities
    print("[Simulator] Precomputing match probabilities across all 32 teams...", flush=True)
    all_teams = [t for group in groups_2014.values() for t in group]
    pair_probs = {}

    for i, t1 in enumerate(all_teams):
        for j, t2 in enumerate(all_teams):
            if i != j:
                p_win, p_draw, p_loss = engine.match_probabilities(ratings[t1], ratings[t2], neutral=True)
                pair_probs[(t1, t2)] = (p_win, p_draw, p_loss)

    # 1. Single Tournament Realization (Seed 2014)
    group_results = {}
    group_qualifiers = {}

    for g_name, team_list in groups_2014.items():
        table = {t: {"pts": 0, "gd": 0, "gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0} for t in team_list}
        for i in range(len(team_list)):
            for j in range(i + 1, len(team_list)):
                t1, t2 = team_list[i], team_list[j]
                g1, g2 = engine.sample_scoreline(ratings[t1], ratings[t2], neutral=True)
                table[t1]["gf"] += g1
                table[t1]["ga"] += g2
                table[t1]["gd"] += (g1 - g2)
                table[t2]["gf"] += g2
                table[t2]["ga"] += g1
                table[t2]["gd"] += (g2 - g1)
                if g1 > g2:
                    table[t1]["pts"] += 3; table[t1]["w"] += 1; table[t2]["l"] += 1
                elif g1 < g2:
                    table[t2]["pts"] += 3; table[t2]["w"] += 1; table[t1]["l"] += 1
                else:
                    table[t1]["pts"] += 1; table[t1]["d"] += 1; table[t2]["pts"] += 1; table[t2]["d"] += 1
        
        sorted_teams = sorted(
            team_list,
            key=lambda t: (table[t]["pts"], table[t]["gd"], table[t]["gf"]),
            reverse=True
        )
        group_results[g_name] = [(t, table[t]) for t in sorted_teams]
        group_qualifiers[g_name] = (sorted_teams[0], sorted_teams[1])

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

    # Round of 16 (2014 bracket)
    r16_matchups = [
        ("1A vs 2B", group_qualifiers["Group A"][0], group_qualifiers["Group B"][1]),
        ("1C vs 2D", group_qualifiers["Group C"][0], group_qualifiers["Group D"][1]),
        ("1E vs 2F", group_qualifiers["Group E"][0], group_qualifiers["Group F"][1]),
        ("1G vs 2H", group_qualifiers["Group G"][0], group_qualifiers["Group H"][1]),
        ("1B vs 2A", group_qualifiers["Group B"][0], group_qualifiers["Group A"][1]),
        ("1D vs 2C", group_qualifiers["Group D"][0], group_qualifiers["Group C"][1]),
        ("1F vs 2E", group_qualifiers["Group F"][0], group_qualifiers["Group E"][1]),
        ("1H vs 2G", group_qualifiers["Group H"][0], group_qualifiers["Group G"][1]),
    ]
    r16_results = []
    qf_teams = []
    for label, t1, t2 in r16_matchups:
        win, score = play_ko(t1, t2)
        r16_results.append({"match": f"{t1} vs {t2}", "winner": win, "score": score})
        qf_teams.append(win)

    # Quarter-Finals
    qf_matchups = [
        (qf_teams[0], qf_teams[1]),
        (qf_teams[2], qf_teams[3]),
        (qf_teams[4], qf_teams[5]),
        (qf_teams[6], qf_teams[7]),
    ]
    qf_results = []
    sf_teams = []
    for t1, t2 in qf_matchups:
        win, score = play_ko(t1, t2)
        qf_results.append({"match": f"{t1} vs {t2}", "winner": win, "score": score})
        sf_teams.append(win)

    # Semi-Finals
    sf_matchups = [
        (sf_teams[0], sf_teams[1]),
        (sf_teams[2], sf_teams[3]),
    ]
    sf_results = []
    final_teams = []
    third_place_teams = []
    for t1, t2 in sf_matchups:
        win, score = play_ko(t1, t2)
        loser = t2 if win == t1 else t1
        sf_results.append({"match": f"{t1} vs {t2}", "winner": win, "score": score})
        final_teams.append(win)
        third_place_teams.append(loser)

    # 3rd Place
    third_winner, third_score = play_ko(third_place_teams[0], third_place_teams[1])
    fourth_place = third_place_teams[1] if third_winner == third_place_teams[0] else third_place_teams[0]

    # Final
    champion, final_score = play_ko(final_teams[0], final_teams[1])
    runner_up = final_teams[1] if champion == final_teams[0] else final_teams[0]

    # 2. Monte Carlo 5,000 runs
    print("[Simulator] Running 5,000 Monte Carlo tournament iterations...", flush=True)
    champ_counts = Counter()
    finalist_counts = Counter()
    semi_counts = Counter()
    qf_counts = Counter()
    r16_counts = Counter()
    group_exit_counts = Counter()

    rng = np.random.default_rng(2014)
    n_sims = 5000

    for _ in range(n_sims):
        mc_group_qual = {}
        for g_name, team_list in groups_2014.items():
            pts = {t: 0 for t in team_list}
            for i in range(len(team_list)):
                for j in range(i + 1, len(team_list)):
                    t1, t2 = team_list[i], team_list[j]
                    pw, pd, pl = pair_probs[(t1, t2)]
                    r = rng.random()
                    if r < pw: pts[t1] += 3
                    elif r < pw + pd: pts[t1] += 1; pts[t2] += 1
                    else: pts[t2] += 3
            ranked = sorted(team_list, key=lambda t: pts[t] + rng.random()*0.1, reverse=True)
            mc_group_qual[g_name] = (ranked[0], ranked[1])
            r16_counts[ranked[0]] += 1
            r16_counts[ranked[1]] += 1
            group_exit_counts[ranked[2]] += 1
            group_exit_counts[ranked[3]] += 1

        # R16
        r16_pairs = [
            (mc_group_qual["Group A"][0], mc_group_qual["Group B"][1]),
            (mc_group_qual["Group C"][0], mc_group_qual["Group D"][1]),
            (mc_group_qual["Group E"][0], mc_group_qual["Group F"][1]),
            (mc_group_qual["Group G"][0], mc_group_qual["Group H"][1]),
            (mc_group_qual["Group B"][0], mc_group_qual["Group A"][1]),
            (mc_group_qual["Group D"][0], mc_group_qual["Group C"][1]),
            (mc_group_qual["Group F"][0], mc_group_qual["Group E"][1]),
            (mc_group_qual["Group H"][0], mc_group_qual["Group G"][1]),
        ]
        qf_w = []
        for t1, t2 in r16_pairs:
            pw, pd, pl = pair_probs[(t1, t2)]
            p_t1_adv = pw + pd * (0.5 + (ratings[t1].attack - ratings[t2].attack)/200.0)
            winner = t1 if rng.random() < p_t1_adv else t2
            qf_w.append(winner)
            qf_counts[winner] += 1

        # QF
        sf_w = []
        for i in range(0, 8, 2):
            t1, t2 = qf_w[i], qf_w[i+1]
            pw, pd, pl = pair_probs[(t1, t2)]
            p_t1_adv = pw + pd * (0.5 + (ratings[t1].attack - ratings[t2].attack)/200.0)
            winner = t1 if rng.random() < p_t1_adv else t2
            sf_w.append(winner)
            semi_counts[winner] += 1

        # SF
        f_w = []
        for i in range(0, 4, 2):
            t1, t2 = sf_w[i], sf_w[i+1]
            pw, pd, pl = pair_probs[(t1, t2)]
            p_t1_adv = pw + pd * (0.5 + (ratings[t1].attack - ratings[t2].attack)/200.0)
            winner = t1 if rng.random() < p_t1_adv else t2
            f_w.append(winner)
            finalist_counts[winner] += 1

        # Final
        t1, t2 = f_w[0], f_w[1]
        pw, pd, pl = pair_probs[(t1, t2)]
        p_t1_adv = pw + pd * (0.5 + (ratings[t1].attack - ratings[t2].attack)/200.0)
        champ = t1 if rng.random() < p_t1_adv else t2
        champ_counts[champ] += 1

    return {
        "groups": group_results,
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
        "mc_exit": group_exit_counts,
        "n_sims": n_sims,
    }

if __name__ == "__main__":
    res = run_2014_world_cup_simulation()
    
    print("\n" + "="*75)
    print("[2014 FIFA WORLD CUP - MODEL SIMULATION RESULTS]")
    print("="*75)
    
    for g_name, table in res["groups"].items():
        print(f"\n--- {g_name} Predicted Standings ---")
        print(f"{'Pos':<4}{'Team':<24}{'P':<4}{'W':<4}{'D':<4}{'L':<4}{'GF':<4}{'GA':<4}{'GD':<4}{'Pts':<4}")
        for pos, (t, stats) in enumerate(table, start=1):
            print(f"{pos:<4}{t:<24}{stats['w']+stats['d']+stats['l']:<4}{stats['w']:<4}{stats['d']:<4}{stats['l']:<4}{stats['gf']:<4}{stats['ga']:<4}{stats['gd']:<4}{stats['pts']:<4}")

    print("\n" + "="*75)
    print("[KNOCKOUT BRACKET PREDICTIONS]")
    print("="*75)
    print("\n--- Round of 16 ---")
    for r in res["r16"]:
        print(f"  {r['match']:<40} -> Winner: {r['winner']:<15} ({r['score']})")

    print("\n--- Quarter-Finals ---")
    for r in res["qf"]:
        print(f"  {r['match']:<40} -> Winner: {r['winner']:<15} ({r['score']})")

    print("\n--- Semi-Finals ---")
    for r in res["sf"]:
        print(f"  {r['match']:<40} -> Winner: {r['winner']:<15} ({r['score']})")

    print(f"\n--- 3rd Place Play-off ---")
    print(f"  {res['third_place']['match']:<40} -> 3rd Place: {res['third_place']['winner']:<12} ({res['third_place']['score']})")

    print(f"\n--- WORLD CUP FINAL ---")
    print(f"  {res['final']['match']:<40} -> CHAMPION: {res['final']['champion']} ({res['final']['score']})")
    print(f"                                           Runner-up: {res['final']['runner_up']}")
    print(f"                                           3rd Place : {res['third_place']['winner']}")
    print(f"                                           4th Place : {res['third_place']['fourth']}")

    print("\n" + "="*75)
    print("[5,000 MONTE CARLO TOURNAMENT PROBABILITIES]")
    print("="*75)
    print(f"{'Team':<24}{'P(Winner)':<12}{'P(Final)':<12}{'P(Semi)':<12}{'P(QF)':<12}{'P(Exit Grp)':<12}")
    for t, c in res["mc_champ"].most_common(16):
        pw = (c / res['n_sims']) * 100
        pf = (res["mc_final"][t] / res['n_sims']) * 100
        ps = (res["mc_semi"][t] / res['n_sims']) * 100
        pq = (res["mc_qf"][t] / res['n_sims']) * 100
        pe = (res["mc_exit"][t] / res['n_sims']) * 100
        print(f"{t:<24}{pw:>7.1f}%{pf:>11.1f}%{ps:>11.1f}%{pq:>11.1f}%{pe:>11.1f}%")
