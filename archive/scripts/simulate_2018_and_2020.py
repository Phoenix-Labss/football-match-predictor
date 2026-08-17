"""Simulate 2018 FIFA World Cup and Euro 2020 using Dynamic Oracle."""

import sys
from collections import Counter
import numpy as np
import pandas as pd

from src.service.oracle import load_oracle
from src.simulation.squad_model import SquadModel, TeamRating
from src.simulation.match_engine import MatchEngine, MatchEngineConfig

def build_team_ratings(oracle, year, teams):
    ratings = {}
    for t in teams:
        key_match = None
        team_dict = oracle.players_by_year_team.get(year, oracle.players_by_year_team[2018])
        for k in team_dict:
            if t.lower() in k.lower() or k.lower() in t.lower() or ("ivo" in t.lower() and "ivo" in k.lower()):
                key_match = k
                break
        if not key_match:
            key_match = t
        try:
            pool, yr = oracle._get_player_pool(key_match, year)
        except Exception:
            pool, yr = oracle._get_player_pool(key_match, 2018)
        squad_model = SquadModel(formation="4-3-3")
        lineup = squad_model.select_lineup(pool)
        chem = oracle.chemistry_model.team_chemistry(t, [p for p, _ in lineup])
        rating = squad_model.aggregate(t, pool, chemistry_score=chem)
        ratings[t] = rating
    return ratings

def precompute_probs(engine, ratings, teams):
    pair_probs = {}
    for i, t1 in enumerate(teams):
        for j, t2 in enumerate(teams):
            if i != j:
                pw, pd, pl = engine.match_probabilities(ratings[t1], ratings[t2], neutral=True)
                pair_probs[(t1, t2)] = (pw, pd, pl)
    return pair_probs

# ------------------------------------------------------------------------- #
# 2018 World Cup
# ------------------------------------------------------------------------- #
def run_2018_world_cup(oracle, engine):
    groups_2018 = {
        "Group A": ["Uruguay", "Russia", "Saudi Arabia", "Egypt"],
        "Group B": ["Spain", "Portugal", "Iran", "Morocco"],
        "Group C": ["France", "Denmark", "Peru", "Australia"],
        "Group D": ["Croatia", "Argentina", "Nigeria", "Iceland"],
        "Group E": ["Brazil", "Switzerland", "Serbia", "Costa Rica"],
        "Group F": ["Sweden", "Mexico", "Korea Republic", "Germany"],
        "Group G": ["Belgium", "England", "Tunisia", "Panama"],
        "Group H": ["Colombia", "Japan", "Senegal", "Poland"],
    }
    all_teams = [t for grp in groups_2018.values() for t in grp]
    ratings = build_team_ratings(oracle, 2018, all_teams)
    pair_probs = precompute_probs(engine, ratings, all_teams)

    # Single tournament
    group_results = {}
    group_qualifiers = {}
    for g_name, team_list in groups_2018.items():
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
        sorted_teams = sorted(team_list, key=lambda t: (table[t]["pts"], table[t]["gd"], table[t]["gf"]), reverse=True)
        group_results[g_name] = [(t, table[t]) for t in sorted_teams]
        group_qualifiers[g_name] = (sorted_teams[0], sorted_teams[1])

    def play_ko(t1, t2):
        g1, g2 = engine.sample_scoreline(ratings[t1], ratings[t2], neutral=True)
        if g1 > g2: return t1, f"{g1} - {g2}"
        elif g1 < g2: return t2, f"{g1} - {g2}"
        else:
            g1_et, g2_et = engine.sample_scoreline(ratings[t1], ratings[t2], neutral=True)
            if g1_et > g2_et: return t1, f"{g1+1} - {g2} (AET)"
            elif g1_et < g2_et: return t2, f"{g1} - {g2+1} (AET)"
            else:
                winner = t1 if ratings[t1].attack >= ratings[t2].attack else t2
                return winner, f"{g1} - {g2} (PKs)"

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
    r16_res = [play_ko(t1, t2) for _, t1, t2 in r16_matchups]
    qf_teams = [w for w, _ in r16_res]

    qf_pairs = [(qf_teams[0], qf_teams[1]), (qf_teams[2], qf_teams[3]), (qf_teams[4], qf_teams[5]), (qf_teams[6], qf_teams[7])]
    qf_res = [play_ko(t1, t2) for t1, t2 in qf_pairs]
    sf_teams = [w for w, _ in qf_res]

    sf_pairs = [(sf_teams[0], sf_teams[1]), (sf_teams[2], sf_teams[3])]
    sf_res = [play_ko(t1, t2) for t1, t2 in sf_pairs]
    final_teams = [w for w, _ in sf_res]
    third_teams = [sf_pairs[0][1] if sf_res[0][0] == sf_pairs[0][0] else sf_pairs[0][0],
                   sf_pairs[1][1] if sf_res[1][0] == sf_pairs[1][0] else sf_pairs[1][0]]

    third_w, third_sc = play_ko(third_teams[0], third_teams[1])
    fourth = third_teams[1] if third_w == third_teams[0] else third_teams[0]

    champ, final_sc = play_ko(final_teams[0], final_teams[1])
    runner_up = final_teams[1] if champ == final_teams[0] else final_teams[0]

    # Monte Carlo 5,000 runs
    champ_counts = Counter()
    finalist_counts = Counter()
    semi_counts = Counter()
    qf_counts = Counter()
    exit_counts = Counter()
    rng = np.random.default_rng(2018)
    n_sims = 5000

    for _ in range(n_sims):
        mc_qual = {}
        for g_name, t_list in groups_2018.items():
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
            mc_qual[g_name] = (ranked[0], ranked[1])
            exit_counts[ranked[2]] += 1
            exit_counts[ranked[3]] += 1

        r16_p = [
            (mc_qual["Group A"][0], mc_qual["Group B"][1]),
            (mc_qual["Group C"][0], mc_qual["Group D"][1]),
            (mc_qual["Group E"][0], mc_qual["Group F"][1]),
            (mc_qual["Group G"][0], mc_qual["Group H"][1]),
            (mc_qual["Group B"][0], mc_qual["Group A"][1]),
            (mc_qual["Group D"][0], mc_qual["Group C"][1]),
            (mc_qual["Group F"][0], mc_qual["Group E"][1]),
            (mc_qual["Group H"][0], mc_qual["Group G"][1]),
        ]
        qf_w = []
        for t1, t2 in r16_p:
            pw, pd, pl = pair_probs[(t1, t2)]
            p_t1_adv = pw + pd * (0.5 + (ratings[t1].attack - ratings[t2].attack)/200.0)
            winner = t1 if rng.random() < p_t1_adv else t2
            qf_w.append(winner)
            qf_counts[winner] += 1

        sf_w = []
        for i in range(0, 8, 2):
            t1, t2 = qf_w[i], qf_w[i+1]
            pw, pd, pl = pair_probs[(t1, t2)]
            p_t1_adv = pw + pd * (0.5 + (ratings[t1].attack - ratings[t2].attack)/200.0)
            winner = t1 if rng.random() < p_t1_adv else t2
            sf_w.append(winner)
            semi_counts[winner] += 1

        f_w = []
        for i in range(0, 4, 2):
            t1, t2 = sf_w[i], sf_w[i+1]
            pw, pd, pl = pair_probs[(t1, t2)]
            p_t1_adv = pw + pd * (0.5 + (ratings[t1].attack - ratings[t2].attack)/200.0)
            winner = t1 if rng.random() < p_t1_adv else t2
            f_w.append(winner)
            finalist_counts[winner] += 1

        t1, t2 = f_w[0], f_w[1]
        pw, pd, pl = pair_probs[(t1, t2)]
        p_t1_adv = pw + pd * (0.5 + (ratings[t1].attack - ratings[t2].attack)/200.0)
        c = t1 if rng.random() < p_t1_adv else t2
        champ_counts[c] += 1

    return {
        "groups": group_results,
        "r16": [(r16_matchups[i][1], r16_matchups[i][2], r16_res[i][0], r16_res[i][1]) for i in range(8)],
        "qf": [(qf_pairs[i][0], qf_pairs[i][1], qf_res[i][0], qf_res[i][1]) for i in range(4)],
        "sf": [(sf_pairs[i][0], sf_pairs[i][1], sf_res[i][0], sf_res[i][1]) for i in range(2)],
        "final": (final_teams[0], final_teams[1], champ, runner_up, final_sc),
        "third": (third_teams[0], third_teams[1], third_w, fourth, third_sc),
        "mc_champ": champ_counts,
        "mc_final": finalist_counts,
        "mc_semi": semi_counts,
        "mc_qf": qf_counts,
        "mc_exit": exit_counts,
        "n_sims": n_sims
    }

# ------------------------------------------------------------------------- #
# Euro 2020 (24-team format)
# ------------------------------------------------------------------------- #
def run_euro_2020(oracle, engine):
    groups_euro = {
        "Group A": ["Italy", "Switzerland", "Turkey", "Wales"],
        "Group B": ["Belgium", "Denmark", "Finland", "Russia"],
        "Group C": ["Netherlands", "Austria", "Ukraine", "North Macedonia"],
        "Group D": ["England", "Croatia", "Czech Republic", "Scotland"],
        "Group E": ["Spain", "Sweden", "Poland", "Slovakia"],
        "Group F": ["France", "Germany", "Portugal", "Hungary"],
    }
    all_teams = [t for grp in groups_euro.values() for t in grp]
    ratings = build_team_ratings(oracle, 2021, all_teams)
    pair_probs = precompute_probs(engine, ratings, all_teams)

    # Group Stage
    group_results = {}
    top2_per_group = {}
    third_placed = []

    for g_name, team_list in groups_euro.items():
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
        sorted_teams = sorted(team_list, key=lambda t: (table[t]["pts"], table[t]["gd"], table[t]["gf"]), reverse=True)
        group_results[g_name] = [(t, table[t]) for t in sorted_teams]
        top2_per_group[g_name] = (sorted_teams[0], sorted_teams[1])
        third_placed.append((sorted_teams[2], table[sorted_teams[2]]))

    # Best 4 third-placed teams qualify
    sorted_thirds = sorted(third_placed, key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"]), reverse=True)
    best_4_thirds = [t for t, _ in sorted_thirds[:4]]

    def play_ko(t1, t2):
        g1, g2 = engine.sample_scoreline(ratings[t1], ratings[t2], neutral=True)
        if g1 > g2: return t1, f"{g1} - {g2}"
        elif g1 < g2: return t2, f"{g1} - {g2}"
        else:
            g1_et, g2_et = engine.sample_scoreline(ratings[t1], ratings[t2], neutral=True)
            if g1_et > g2_et: return t1, f"{g1+1} - {g2} (AET)"
            elif g1_et < g2_et: return t2, f"{g1} - {g2+1} (AET)"
            else:
                winner = t1 if ratings[t1].attack >= ratings[t2].attack else t2
                return winner, f"{g1} - {g2} (PKs)"

    # Euro 2020 Round of 16 Bracket
    r16_matchups = [
        ("2A vs 2B", top2_per_group["Group A"][1], top2_per_group["Group B"][1]),
        ("1A vs 2C", top2_per_group["Group A"][0], top2_per_group["Group C"][1]),
        ("1C vs 3D/E/F", top2_per_group["Group C"][0], best_4_thirds[0]),
        ("1B vs 3A/D/E/F", top2_per_group["Group B"][0], best_4_thirds[1]),
        ("2D vs 2E", top2_per_group["Group D"][1], top2_per_group["Group E"][1]),
        ("1F vs 3A/B/C", top2_per_group["Group F"][0], best_4_thirds[2]),
        ("1D vs 2F", top2_per_group["Group D"][0], top2_per_group["Group F"][1]),
        ("1E vs 3A/B/C/D", top2_per_group["Group E"][0], best_4_thirds[3]),
    ]
    r16_res = [play_ko(t1, t2) for _, t1, t2 in r16_matchups]
    qf_teams = [w for w, _ in r16_res]

    qf_pairs = [(qf_teams[0], qf_teams[1]), (qf_teams[2], qf_teams[3]), (qf_teams[4], qf_teams[5]), (qf_teams[6], qf_teams[7])]
    qf_res = [play_ko(t1, t2) for t1, t2 in qf_pairs]
    sf_teams = [w for w, _ in qf_res]

    sf_pairs = [(sf_teams[0], sf_teams[1]), (sf_teams[2], sf_teams[3])]
    sf_res = [play_ko(t1, t2) for t1, t2 in sf_pairs]
    final_teams = [w for w, _ in sf_res]

    champ, final_sc = play_ko(final_teams[0], final_teams[1])
    runner_up = final_teams[1] if champ == final_teams[0] else final_teams[0]

    # Monte Carlo 5,000 runs
    champ_counts = Counter()
    finalist_counts = Counter()
    semi_counts = Counter()
    qf_counts = Counter()
    exit_counts = Counter()
    rng = np.random.default_rng(2020)
    n_sims = 5000

    for _ in range(n_sims):
        mc_top2 = {}
        mc_3rds = []
        for g_name, t_list in groups_euro.items():
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
        
        # 4 best thirds advance, 2 worst exit
        best_4 = mc_3rds[:4]
        exit_counts[mc_3rds[4]] += 1
        exit_counts[mc_3rds[5]] += 1

        r16_p = [
            (mc_top2["Group A"][1], mc_top2["Group B"][1]),
            (mc_top2["Group A"][0], mc_top2["Group C"][1]),
            (mc_top2["Group C"][0], best_4[0]),
            (mc_top2["Group B"][0], best_4[1]),
            (mc_top2["Group D"][1], mc_top2["Group E"][1]),
            (mc_top2["Group F"][0], best_4[2]),
            (mc_top2["Group D"][0], mc_top2["Group F"][1]),
            (mc_top2["Group E"][0], best_4[3]),
        ]
        qf_w = []
        for t1, t2 in r16_p:
            pw, pd, pl = pair_probs[(t1, t2)]
            p_t1_adv = pw + pd * (0.5 + (ratings[t1].attack - ratings[t2].attack)/200.0)
            winner = t1 if rng.random() < p_t1_adv else t2
            qf_w.append(winner)
            qf_counts[winner] += 1

        sf_w = []
        for i in range(0, 8, 2):
            t1, t2 = qf_w[i], qf_w[i+1]
            pw, pd, pl = pair_probs[(t1, t2)]
            p_t1_adv = pw + pd * (0.5 + (ratings[t1].attack - ratings[t2].attack)/200.0)
            winner = t1 if rng.random() < p_t1_adv else t2
            sf_w.append(winner)
            semi_counts[winner] += 1

        f_w = []
        for i in range(0, 4, 2):
            t1, t2 = sf_w[i], sf_w[i+1]
            pw, pd, pl = pair_probs[(t1, t2)]
            p_t1_adv = pw + pd * (0.5 + (ratings[t1].attack - ratings[t2].attack)/200.0)
            winner = t1 if rng.random() < p_t1_adv else t2
            f_w.append(winner)
            finalist_counts[winner] += 1

        t1, t2 = f_w[0], f_w[1]
        pw, pd, pl = pair_probs[(t1, t2)]
        p_t1_adv = pw + pd * (0.5 + (ratings[t1].attack - ratings[t2].attack)/200.0)
        c = t1 if rng.random() < p_t1_adv else t2
        champ_counts[c] += 1

    return {
        "groups": group_results,
        "r16": [(r16_matchups[i][1], r16_matchups[i][2], r16_res[i][0], r16_res[i][1]) for i in range(8)],
        "qf": [(qf_pairs[i][0], qf_pairs[i][1], qf_res[i][0], qf_res[i][1]) for i in range(4)],
        "sf": [(sf_pairs[i][0], sf_pairs[i][1], sf_res[i][0], sf_res[i][1]) for i in range(2)],
        "final": (final_teams[0], final_teams[1], champ, runner_up, final_sc),
        "mc_champ": champ_counts,
        "mc_final": finalist_counts,
        "mc_semi": semi_counts,
        "mc_qf": qf_counts,
        "mc_exit": exit_counts,
        "n_sims": n_sims
    }

if __name__ == "__main__":
    oracle = load_oracle()
    engine = MatchEngine(MatchEngineConfig(home_advantage=0.0, baseline_goals=0.55), seed=42)

    print("\n" + "="*80)
    print("[1. SIMULATING 2018 FIFA WORLD CUP (RUSSIA)]")
    print("="*80)
    res_2018 = run_2018_world_cup(oracle, engine)

    for g_name, table in res_2018["groups"].items():
        print(f"\n--- {g_name} Standings ---")
        print(f"{'Pos':<4}{'Team':<20}{'P':<4}{'W':<4}{'D':<4}{'L':<4}{'GF':<4}{'GA':<4}{'GD':<4}{'Pts':<4}")
        for pos, (t, s) in enumerate(table, start=1):
            print(f"{pos:<4}{t:<20}{s['w']+s['d']+s['l']:<4}{s['w']:<4}{s['d']:<4}{s['l']:<4}{s['gf']:<4}{s['ga']:<4}{s['gd']:<4}{s['pts']:<4}")

    print("\n--- 2018 Knockout Bracket ---")
    for t1, t2, w, sc in res_2018["r16"]:
        print(f"  R16: {t1} vs {t2:<16} -> Winner: {w:<15} ({sc})")
    for t1, t2, w, sc in res_2018["qf"]:
        print(f"  QF : {t1} vs {t2:<16} -> Winner: {w:<15} ({sc})")
    for t1, t2, w, sc in res_2018["sf"]:
        print(f"  SF : {t1} vs {t2:<16} -> Winner: {w:<15} ({sc})")
    print(f"  3RD: {res_2018['third'][0]} vs {res_2018['third'][1]} -> 3rd Place: {res_2018['third'][2]} ({res_2018['third'][4]})")
    print(f"  FIN: {res_2018['final'][0]} vs {res_2018['final'][1]} -> CHAMPION: {res_2018['final'][2]} ({res_2018['final'][4]}) | Runner-up: {res_2018['final'][3]}")

    print("\n--- 2018 5,000 Monte Carlo Probabilities ---")
    print(f"{'Team':<20}{'P(Winner)':<12}{'P(Final)':<12}{'P(Semi)':<12}{'P(QF)':<12}{'P(Exit)':<12}")
    for t, c in res_2018["mc_champ"].most_common(12):
        pw = (c / res_2018['n_sims']) * 100
        pf = (res_2018["mc_final"][t] / res_2018['n_sims']) * 100
        ps = (res_2018["mc_semi"][t] / res_2018['n_sims']) * 100
        pq = (res_2018["mc_qf"][t] / res_2018['n_sims']) * 100
        pe = (res_2018["mc_exit"][t] / res_2018['n_sims']) * 100
        print(f"{t:<20}{pw:>7.1f}%{pf:>11.1f}%{ps:>11.1f}%{pq:>11.1f}%{pe:>11.1f}%")

    print("\n" + "="*80)
    print("[2. SIMULATING UEFA EURO 2020]")
    print("="*80)
    res_euro = run_euro_2020(oracle, engine)

    for g_name, table in res_euro["groups"].items():
        print(f"\n--- {g_name} Standings ---")
        print(f"{'Pos':<4}{'Team':<20}{'P':<4}{'W':<4}{'D':<4}{'L':<4}{'GF':<4}{'GA':<4}{'GD':<4}{'Pts':<4}")
        for pos, (t, s) in enumerate(table, start=1):
            print(f"{pos:<4}{t:<20}{s['w']+s['d']+s['l']:<4}{s['w']:<4}{s['d']:<4}{s['l']:<4}{s['gf']:<4}{s['ga']:<4}{s['gd']:<4}{s['pts']:<4}")

    print("\n--- Euro 2020 Knockout Bracket ---")
    for t1, t2, w, sc in res_euro["r16"]:
        print(f"  R16: {t1} vs {t2:<16} -> Winner: {w:<15} ({sc})")
    for t1, t2, w, sc in res_euro["qf"]:
        print(f"  QF : {t1} vs {t2:<16} -> Winner: {w:<15} ({sc})")
    for t1, t2, w, sc in res_euro["sf"]:
        print(f"  SF : {t1} vs {t2:<16} -> Winner: {w:<15} ({sc})")
    print(f"  FIN: {res_euro['final'][0]} vs {res_euro['final'][1]} -> CHAMPION: {res_euro['final'][2]} ({res_euro['final'][4]}) | Runner-up: {res_euro['final'][3]}")

    print("\n--- Euro 2020 5,000 Monte Carlo Probabilities ---")
    print(f"{'Team':<20}{'P(Winner)':<12}{'P(Final)':<12}{'P(Semi)':<12}{'P(QF)':<12}{'P(Exit)':<12}")
    for t, c in res_euro["mc_champ"].most_common(12):
        pw = (c / res_euro['n_sims']) * 100
        pf = (res_euro["mc_final"][t] / res_euro['n_sims']) * 100
        ps = (res_euro["mc_semi"][t] / res_euro['n_sims']) * 100
        pq = (res_euro["mc_qf"][t] / res_euro['n_sims']) * 100
        pe = (res_euro["mc_exit"][t] / res_euro['n_sims']) * 100
        print(f"{t:<20}{pw:>7.1f}%{pf:>11.1f}%{ps:>11.1f}%{pq:>11.1f}%{pe:>11.1f}%")
