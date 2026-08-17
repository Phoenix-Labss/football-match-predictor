# 2026 FIFA World Cup — Production Negative Binomial Simulation

Production simulation of the 2026 FIFA World Cup (48 teams, 12 groups, 10,000 tournaments) using **Engine C: Match-Day State + Negative Binomial + Dixon-Coles**.

---

## 1. Poisson vs Negative Binomial Engine Comparison (2026 World Cup)

| Metric | Legacy Poisson Engine | Engine C (MDS + NegBin + DC) | Physical / Statistical Advantage |
|:---|---:|---:|:---|
| **Top Champion Pick** | **Spain (7.70%)** | **Spain (8.08%)** | Consistent Primary Favorite |
| **Goal Variance-to-Mean Ratio (VMR)** | `1.04` (Thin Tail) | **`1.22` (Calibrated)** | Fixes blowout scoreline underestimation |
| **4+ Total Goals Match Rate** | `26.4%` | **`27.0%`** | Matches real tournament blowout density |
| **5+ Total Goals Match Rate** | `12.3%` | **`13.7%`** | Higher likelihood on high-scoring thrillers |
| **Draw Rate (90 mins)** | `28.8%` | **`29.2%`** | Exact Dixon-Coles calibration |
| **Most Likely Scoreline** | `1 - 1 (11.2%)` | **`1 - 1 (11.0%)`** | Robust mode preservation |
| **Simulation Runtime (10k Runs)** | `~4.8s` | **`7.36s`** | Real-time vectorized performance |

---

## 2. Top 10 Champion Contenders Comparison

| Rank | Poisson Champion (Prob %) | NegBin Champion (Prob %) | Final Prob (NB) | Semi Prob (NB) |
|:---:|:---|:---|---:|---:|
| **#1** | Spain (7.70%) | **Spain (8.08%)** | 13.62% | 23.84% |
| **#2** | France (5.86%) | **Netherlands (5.57%)** | 10.11% | 18.59% |
| **#3** | Argentina (5.43%) | **Brazil (5.32%)** | 10.07% | 18.46% |
| **#4** | England (5.26%) | **Germany (5.21%)** | 9.72% | 18.00% |
| **#5** | Portugal (5.17%) | **Belgium (4.45%)** | 8.80% | 16.30% |
| **#6** | Germany (5.00%) | **USA (4.00%)** | 8.00% | 14.92% |
| **#7** | Brazil (4.61%) | **Türkiye (3.90%)** | 8.03% | 14.89% |
| **#8** | Netherlands (4.39%) | **South Korea (3.84%)** | 7.19% | 13.35% |
| **#9** | USA (3.21%) | **Sweden (3.72%)** | 7.29% | 14.05% |
| **#10** | Belgium (3.13%) | **Switzerland (3.71%)** | 7.30% | 14.32% |

---

## 3. Production Configuration
```json
{
  "tournament": "2026 FIFA World Cup",
  "simulation_engine": "Match-Day State + Negative Binomial + Dixon-Coles (Engine C)",
  "simulation_goal_model": "negbin",
  "match_day_state": true,
  "dispersion_alpha": 0.1262,
  "dixon_coles_rho": -0.1,
  "n_tournaments": 10000,
  "seed": 42,
  "runtime_seconds": 7.36,
  "top_champion": "Spain",
  "top_champion_prob": 8.08
}
```