"""Comprehensive Match Simulation Pipeline Audit Script.
Traces Brazil 2022 vs France 2022 end-to-end, performs 55-pair chemistry breakdown,
evaluates Dixon-Coles xG, tests Monte Carlo convergence across [1k, 2.5k, 5k, 10k, 25k, 50k],
generates convergence.csv and convergence_plot.png, and prepares data for SIMULATION_AUDIT.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt

# Ensure root is in path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.service.oracle import load_oracle
from src.simulation.squad_model import SquadModel, FORMATIONS, positional_fit
from src.simulation.chemistry import ChemistryModel, ChemistryConfig, _pair_position_compat, _primary_group
from src.simulation.player_model import age_factor, PlayerModel
from src.simulation.match_engine import MatchEngine, MatchEngineConfig, RHO


def run_audit():
    print("=" * 80)
    print("STARTING DYNAMIC ORACLE SIMULATION PIPELINE AUDIT")
    print("=" * 80)

    results_dir = root / "results" / "simulation_validation"
    results_dir.mkdir(parents=True, exist_ok=True)

    oracle = load_oracle()
    
    # ---------------------------------------------------------
    # 1. TRACE ONE REAL MATCH: Brazil 2022 vs France 2022
    # ---------------------------------------------------------
    team_a_name = "Brazil"
    year_a = 2022
    team_b_name = "France"
    year_b = 2022

    pool_a, res_year_a = oracle._get_player_pool(team_a_name, year_a)
    pool_b, res_year_b = oracle._get_player_pool(team_b_name, year_b)

    squad_model_a = SquadModel(formation="4-3-3")
    squad_model_b = SquadModel(formation="4-3-3")

    lineup_a_pairs = squad_model_a.select_lineup(pool_a)
    lineup_b_pairs = squad_model_b.select_lineup(pool_b)

    slots = FORMATIONS["4-3-3"]

    print("\n--- TEAM A: BRAZIL 2022 STARTING XI ---")
    team_a_rows = []
    for (p, fit), (grp, slot_label) in zip(lineup_a_pairs, slots):
        af = age_factor(p.age)
        ability = p.overall * af
        # final player contribution formula in SquadModel.aggregate:
        if grp == "GK":
            contrib = (p.gk_ability * fit) if p.gk_ability else (p.form * 0.8)
        elif grp == "DEF":
            contrib = (0.6 * p.defending + 0.25 * p.physical + 0.15 * p.ability) * fit
        elif grp == "MID":
            contrib = (0.35 * p.passing + 0.25 * p.dribbling + 0.2 * p.defending + 0.2 * p.ability) * fit
        else: # ATT
            contrib = (0.4 * p.shooting + 0.25 * p.dribbling + 0.2 * p.pace + 0.15 * p.ability) * fit

        team_a_rows.append({
            "slot": f"{grp} ({slot_label})",
            "name": p.name,
            "positions": p.positions,
            "club": p.club,
            "age": p.age,
            "overall": p.overall,
            "pac": p.pace,
            "sho": p.shooting,
            "pas": p.passing,
            "dri": p.dribbling,
            "def": p.defending,
            "phy": p.physical,
            "gk": p.gk_ability,
            "af": af,
            "ability": ability,
            "form": p.form,
            "availability": p.availability,
            "fit": fit,
            "contrib": contrib,
        })
    df_a = pd.DataFrame(team_a_rows)
    print(df_a.to_string(index=False))

    print("\n--- TEAM B: FRANCE 2022 STARTING XI ---")
    team_b_rows = []
    for (p, fit), (grp, slot_label) in zip(lineup_b_pairs, slots):
        af = age_factor(p.age)
        ability = p.overall * af
        if grp == "GK":
            contrib = (p.gk_ability * fit) if p.gk_ability else (p.form * 0.8)
        elif grp == "DEF":
            contrib = (0.6 * p.defending + 0.25 * p.physical + 0.15 * p.ability) * fit
        elif grp == "MID":
            contrib = (0.35 * p.passing + 0.25 * p.dribbling + 0.2 * p.defending + 0.2 * p.ability) * fit
        else: # ATT
            contrib = (0.4 * p.shooting + 0.25 * p.dribbling + 0.2 * p.pace + 0.15 * p.ability) * fit

        team_b_rows.append({
            "slot": f"{grp} ({slot_label})",
            "name": p.name,
            "positions": p.positions,
            "club": p.club,
            "age": p.age,
            "overall": p.overall,
            "pac": p.pace,
            "sho": p.shooting,
            "pas": p.passing,
            "dri": p.dribbling,
            "def": p.defending,
            "phy": p.physical,
            "gk": p.gk_ability,
            "af": af,
            "ability": ability,
            "form": p.form,
            "availability": p.availability,
            "fit": fit,
            "contrib": contrib,
        })
    df_b = pd.DataFrame(team_b_rows)
    print(df_b.to_string(index=False))

    # ---------------------------------------------------------
    # 7. VERIFY CHEMISTRY (All 55 pairs for Team A and Team B)
    # ---------------------------------------------------------
    chem_model = ChemistryModel(ChemistryConfig(club_weight=0.5, minutes_weight=0.3, position_weight=0.2))
    
    print("\n--- TEAM A (BRAZIL) 55 CHEMISTRY PAIRS ---")
    lineup_a_players = [p for p, _ in lineup_a_pairs]
    chem_pairs_a = []
    for i in range(len(lineup_a_players)):
        for j in range(i + 1, len(lineup_a_players)):
            p1, p2 = lineup_a_players[i], lineup_a_players[j]
            same_club = 1.0 if (p1.club and p2.club and p1.club == p2.club) else 0.0
            minutes = 1800.0 if same_club else 0.0
            minutes_norm = min(1.0, minutes / 5000.0)
            g1, g2 = _primary_group(p1.positions), _primary_group(p2.positions)
            pos_compat = _pair_position_compat(g1, g2)
            pair_c = 0.5 * same_club + 0.3 * minutes_norm + 0.2 * pos_compat
            chem_pairs_a.append({
                "pair_idx": len(chem_pairs_a) + 1,
                "player_1": p1.name,
                "club_1": p1.club,
                "group_1": g1,
                "player_2": p2.name,
                "club_2": p2.club,
                "group_2": g2,
                "same_club": same_club,
                "shared_min": minutes,
                "min_norm": round(minutes_norm, 3),
                "pos_compat": round(pos_compat, 2),
                "pair_chem": round(pair_c, 4),
            })
    df_chem_a = pd.DataFrame(chem_pairs_a)
    print(f"Total pairs: {len(df_chem_a)}")
    print(f"Team A Chem Average: {df_chem_a['pair_chem'].mean():.4f}")

    print("\n--- TEAM B (FRANCE) 55 CHEMISTRY PAIRS ---")
    lineup_b_players = [p for p, _ in lineup_b_pairs]
    chem_pairs_b = []
    for i in range(len(lineup_b_players)):
        for j in range(i + 1, len(lineup_b_players)):
            p1, p2 = lineup_b_players[i], lineup_b_players[j]
            same_club = 1.0 if (p1.club and p2.club and p1.club == p2.club) else 0.0
            minutes = 1800.0 if same_club else 0.0
            minutes_norm = min(1.0, minutes / 5000.0)
            g1, g2 = _primary_group(p1.positions), _primary_group(p2.positions)
            pos_compat = _pair_position_compat(g1, g2)
            pair_c = 0.5 * same_club + 0.3 * minutes_norm + 0.2 * pos_compat
            chem_pairs_b.append({
                "pair_idx": len(chem_pairs_b) + 1,
                "player_1": p1.name,
                "club_1": p1.club,
                "group_1": g1,
                "player_2": p2.name,
                "club_2": p2.club,
                "group_2": g2,
                "same_club": same_club,
                "shared_min": minutes,
                "min_norm": round(minutes_norm, 3),
                "pos_compat": round(pos_compat, 2),
                "pair_chem": round(pair_c, 4),
            })
    df_chem_b = pd.DataFrame(chem_pairs_b)
    print(f"Total pairs: {len(df_chem_b)}")
    print(f"Team B Chem Average: {df_chem_b['pair_chem'].mean():.4f}")

    # ---------------------------------------------------------
    # 8. TEAM RATINGS & MATCHUP INTERACTIONS
    # ---------------------------------------------------------
    chem_a_score = chem_model.team_chemistry(team_a_name, lineup_a_players)
    chem_b_score = chem_model.team_chemistry(team_b_name, lineup_b_players)

    rating_a = squad_model_a.aggregate(team_a_name, pool_a, chemistry_score=chem_a_score)
    rating_b = squad_model_b.aggregate(team_b_name, pool_b, chemistry_score=chem_b_score)

    print("\n--- AGGREGATED TEAM RATINGS ---")
    print(f"Team A (Brazil): Attack={rating_a.attack:.2f}, Midfield={rating_a.midfield:.2f}, Defence={rating_a.defence:.2f}, GK={rating_a.gk:.2f}, Chemistry={rating_a.chemistry:.4f}")
    print(f"Team B (France): Attack={rating_b.attack:.2f}, Midfield={rating_b.midfield:.2f}, Defence={rating_b.defence:.2f}, GK={rating_b.gk:.2f}, Chemistry={rating_b.chemistry:.4f}")

    match_engine = MatchEngine(MatchEngineConfig(home_advantage=0.25, baseline_goals=0.55), seed=42)
    cfg = match_engine.config

    # Manual step-by-step match engine interaction
    home_attack_adj = rating_a.attack * (1.0 + cfg.chemistry_attack_weight * rating_a.chemistry)
    away_attack_adj = rating_b.attack * (1.0 + cfg.chemistry_attack_weight * rating_b.chemistry)
    home_defence_adj = rating_a.defence * (1.0 + cfg.chemistry_defence_weight * rating_a.chemistry)
    away_defence_adj = rating_b.defence * (1.0 + cfg.chemistry_defence_weight * rating_b.chemistry)

    mid_diff = (rating_a.midfield - rating_b.midfield) * 0.5
    gk_home = rating_a.gk / 100.0
    gk_away = rating_b.gk / 100.0

    raw_home = (
        cfg.baseline_goals
        + (home_attack_adj - away_defence_adj) * cfg.rating_scale
        + mid_diff * cfg.rating_scale
        + 0.0 # Neutral venue
        - 0.3 * gk_away
    )
    raw_away = (
        cfg.baseline_goals
        + (away_attack_adj - home_defence_adj) * cfg.rating_scale
        - mid_diff * cfg.rating_scale
        - 0.3 * gk_home
    )

    lam_h_det = float(np.exp(raw_home))
    lam_a_det = float(np.exp(raw_away))

    print(f"\n--- EXPECTED GOALS DERIVATION (NEUTRAL VENUE) ---")
    print(f"Home Attack (adj): {home_attack_adj:.3f} vs Away Defence (adj): {away_defence_adj:.3f} -> Diff: {home_attack_adj - away_defence_adj:+.3f}")
    print(f"Away Attack (adj): {away_attack_adj:.3f} vs Home Defence (adj): {home_defence_adj:.3f} -> Diff: {away_attack_adj - home_defence_adj:+.3f}")
    print(f"Midfield Diff: {mid_diff:+.3f}")
    print(f"GK Home factor: {0.3*gk_home:.3f}, GK Away factor: {0.3*gk_away:.3f}")
    print(f"Raw Log-Goals Home: {raw_home:.4f} -> lambda_home = exp({raw_home:.4f}) = {lam_h_det:.4f}")
    print(f"Raw Log-Goals Away: {raw_away:.4f} -> lambda_away = exp({raw_away:.4f}) = {lam_a_det:.4f}")

    # ---------------------------------------------------------
    # 9. DIXON-COLES PROBABILITIES & TOP 10 SCORELINES
    # ---------------------------------------------------------
    max_g = 8
    probs = np.zeros((max_g + 1, max_g + 1))
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            tau = 1.0
            if i == 0 and j == 0:
                tau = 1.0 - lam_h_det * lam_a_det * RHO
            elif i == 0 and j == 1:
                tau = 1.0 + lam_h_det * RHO
            elif i == 1 and j == 0:
                tau = 1.0 + lam_a_det * RHO
            elif i == 1 and j == 1:
                tau = 1.0 - RHO
            probs[i, j] = max(0.0, poisson.pmf(i, lam_h_det) * poisson.pmf(j, lam_a_det) * tau)
    
    raw_sum = probs.sum()
    probs /= raw_sum

    p_home = float(np.sum(np.tril(probs, -1)))
    p_draw = float(np.sum(np.diag(probs)))
    p_away = float(np.sum(np.triu(probs, 1)))

    print(f"\n--- DIXON-COLES EXACT ANALYTIC PROBABILITIES ---")
    print(f"P(Home Win): {p_home*100:.2f}%")
    print(f"P(Draw):     {p_draw*100:.2f}%")
    print(f"P(Away Win): {p_away*100:.2f}%")

    scoreline_list = []
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            scoreline_list.append((f"{i} - {j}", i, j, probs[i, j]))
    scoreline_list.sort(key=lambda x: -x[3])

    print("\n--- TOP 10 MOST LIKELY SCORELINES ---")
    for rank, (sc, i, j, p) in enumerate(scoreline_list[:10], 1):
        print(f"{rank:2d}. {sc:5s}: {p*100:.2f}%")

    # ---------------------------------------------------------
    # 10. EXTRA TIME AND PENALTIES
    # ---------------------------------------------------------
    pen_a_win, pen_b_win = oracle._simulate_penalties(lineup_a_pairs, lineup_b_pairs)
    print(f"\n--- PENALTY SHOOTOUT PROBABILITIES ---")
    print(f"Team A (Brazil) Shootout Win %: {pen_a_win*100:.2f}%")
    print(f"Team B (France) Shootout Win %: {pen_b_win*100:.2f}%")

    # ---------------------------------------------------------
    # 11. MONTE CARLO CONVERGENCE TEST
    # ---------------------------------------------------------
    sim_counts = [1000, 2500, 5000, 10000, 25000, 50000]
    convergence_rows = []
    flat_p = probs.ravel()
    rng = np.random.default_rng(42)

    for n in sim_counts:
        sampled_indices = rng.choice(len(flat_p), size=n, p=flat_p)
        ga, gb = np.divmod(sampled_indices, max_g + 1)
        h_win = float(np.mean(ga > gb))
        draw = float(np.mean(ga == gb))
        a_win = float(np.mean(ga < gb))

        # Top scorelines in sample
        pairs_sample = np.column_stack((ga, gb))
        unique, counts = np.unique(pairs_sample, axis=0, return_counts=True)
        order = np.argsort(-counts)
        top4 = [f"{unique[idx][0]}-{unique[idx][1]} ({counts[idx]/n*100:.1f}%)" for idx in order[:4]]
        most_likely = f"{unique[order[0]][0]}-{unique[order[0]][1]}"

        convergence_rows.append({
            "simulations": n,
            "home_win_pct": round(h_win * 100, 3),
            "draw_pct": round(draw * 100, 3),
            "away_win_pct": round(a_win * 100, 3),
            "most_likely_scoreline": most_likely,
            "top_1_scoreline": top4[0],
            "top_2_scoreline": top4[1] if len(top4) > 1 else "",
            "top_3_scoreline": top4[2] if len(top4) > 2 else "",
            "top_4_scoreline": top4[3] if len(top4) > 3 else "",
        })

    df_conv = pd.DataFrame(convergence_rows)
    conv_csv_path = results_dir / "convergence.csv"
    df_conv.to_csv(conv_csv_path, index=False)
    print(f"\n--- CONVERGENCE TABLE SAVED TO {conv_csv_path} ---")
    print(df_conv.to_string(index=False))

    # Calculate step-by-step deltas
    print("\n--- PROBABILITY DELTAS BETWEEN SIMULATION STEPS ---")
    deltas = []
    for i in range(1, len(convergence_rows)):
        prev = convergence_rows[i-1]
        curr = convergence_rows[i]
        d_h = abs(curr["home_win_pct"] - prev["home_win_pct"])
        d_d = abs(curr["draw_pct"] - prev["draw_pct"])
        d_a = abs(curr["away_win_pct"] - prev["away_win_pct"])
        max_d = max(d_h, d_d, d_a)
        deltas.append({
            "step": f"{prev['simulations']} -> {curr['simulations']}",
            "delta_home_pct": round(d_h, 3),
            "delta_draw_pct": round(d_d, 3),
            "delta_away_pct": round(d_a, 3),
            "max_delta_pct": round(max_d, 3),
        })
        print(f"{prev['simulations']} -> {curr['simulations']}: Delta Home={d_h:.3f}%, Draw={d_d:.3f}%, Away={d_a:.3f}% (Max={max_d:.3f}%)")

    # ---------------------------------------------------------
    # GENERATE CONVERGENCE PLOT
    # ---------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    sims = [r["simulations"] for r in convergence_rows]
    hw = [r["home_win_pct"] for r in convergence_rows]
    dr = [r["draw_pct"] for r in convergence_rows]
    aw = [r["away_win_pct"] for r in convergence_rows]

    # Subplot 1: Probability Convergence
    ax1.plot(sims, hw, 'o-', color='#1f77b4', linewidth=2, label=f'{team_a_name} Win % (True: {p_home*100:.2f}%)')
    ax1.plot(sims, dr, 's-', color='#7f7f7f', linewidth=2, label=f'Draw % (True: {p_draw*100:.2f}%)')
    ax1.plot(sims, aw, '^-', color='#d62728', linewidth=2, label=f'{team_b_name} Win % (True: {p_away*100:.2f}%)')
    
    # Ground truth horizontal lines
    ax1.axhline(p_home * 100, color='#1f77b4', linestyle='--', alpha=0.5)
    ax1.axhline(p_draw * 100, color='#7f7f7f', linestyle='--', alpha=0.5)
    ax1.axhline(p_away * 100, color='#d62728', linestyle='--', alpha=0.5)

    ax1.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Monte Carlo Convergence Analysis: {team_a_name} vs {team_b_name} (2022)', fontsize=14, fontweight='bold')
    ax1.legend(loc='center right', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Subplot 2: Absolute Error from Analytic Ground Truth
    err_h = [abs(h - p_home * 100) for h in hw]
    err_d = [abs(d - p_draw * 100) for d in dr]
    err_a = [abs(a - p_away * 100) for a in aw]

    ax2.plot(sims, err_h, 'o--', color='#1f77b4', label=f'{team_a_name} Win Error')
    ax2.plot(sims, err_d, 's--', color='#7f7f7f', label='Draw Error')
    ax2.plot(sims, err_a, '^--', color='#d62728', label=f'{team_b_name} Win Error')
    ax2.axhline(0.5, color='orange', linestyle=':', label='±0.5% Tolerance Band')
    ax2.axhline(0.2, color='green', linestyle=':', label='±0.2% High Precision Band')

    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Monte Carlo Iterations (log scale)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Absolute Error (%)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plot_path = results_dir / "convergence_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n--- CONVERGENCE PLOT SAVED TO {plot_path} ---")

    return {
        "df_a": df_a,
        "df_b": df_b,
        "df_chem_a": df_chem_a,
        "df_chem_b": df_chem_b,
        "rating_a": rating_a,
        "rating_b": rating_b,
        "lam_h": lam_h_det,
        "lam_a": lam_a_det,
        "probs": (p_home, p_draw, p_away),
        "scorelines": scoreline_list,
        "pen_win": (pen_a_win, pen_b_win),
        "conv_df": df_conv,
        "deltas": deltas,
    }


if __name__ == "__main__":
    run_audit()
