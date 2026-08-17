# Dynamic Oracle — Experiment History

This document chronicles all major research tracks, milestones, and optimization iterations conducted on the Dynamic Oracle codebase.

---

## Summary of Milestones

| Milestone / Experiment | Objective | Best Accuracy | Key Result / Metric | Status | Primary Artifacts |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **Track 1: Dynamic Strength Tracking** | Confidence-controlled Elo updates vs fixed-K baselines | ~59.5% | Adaptive updates outperformed fixed Elo and frozen controls on temporal folds. | ✅ Completed | `results/archive/tracks/track1_results.*` |
| **Track 2: European Club Integration** | Cross-league player & club strength transfer | ~59.4% | Validated synthetic & FIFA cross-domain feature pipelines. | ✅ Completed | `results/archive/tracks/track2_results.*` |
| **Track 3: FIFA Simulation Engine** | Tournament Monte Carlo & Dixon-Coles xG engine | N/A | Full bivariate Poisson match engine with squad chemistry & player models. | ✅ Completed (Production) | `src/simulation/`, `src/service/` |
| **Accuracy Optimization Round 1** | Direct 3-way match accuracy optimization on 9,904 test matches | **60.14%** (5,956 / 9,904) | 5-Model LogLoss-weighted ensemble (LightGBM + XGBoost + CatBoost + HistGBDT + Dixon-Coles) with 217 engineered features. | 🏆 **CURRENT CHAMPION** | `results/champion/final_test_results.json`, `results/champion/champion_config.json` |
| **Accuracy Optimization Round 2** | Continuous accuracy search, extended feature matrix (241 feats), meta-classifier stacking | **60.12%** (Ensemble) / 59.76% (Meta) | Meta-classifier overfit validation folds (+0.73% val -> -0.27% test). Accuracy-weighted ensemble reached 60.12% but did not beat Round 1. Round 1 retained. | 📦 Archived | `results/archive/accuracy_optimization_r2/`, `archive/optimization_r2/` |

---

## Detailed Track Summaries

### 1. Track 1: Dynamic Strength Tracking (Springer Baseline Replication)
- **Objective**: Implement confidence-controlled strength updates where Elo volatility is bounded based on consistency of recent performances and surprise magnitude.
- **Dataset**: 49,520 international matches (`results.csv`).
- **Validation**: 4 rolling-origin temporal folds (never random shuffle).
- **Finding**: Adaptive Elo update mode provided higher calibration and lower RPS compared to classical static Elo or frozen baselines.

### 2. Track 2: European Club & Player Representation
- **Objective**: Extract player and squad attributes to assess whether club-level synergies enhance international match prediction.
- **Finding**: Multi-year FIFA player datasets (2015-2022) provide useful micro-level chemistry attributes for head-to-head simulations.

### 3. Track 3: FIFA Match Simulation & Web Service
- **Objective**: Build an interactive Dixon-Coles Monte Carlo tournament simulation engine capable of simulating full World Cup brackets (2010 through 2026).
- **Finding**: Bivariate Poisson scoreline sampling with manager tactical bias and positional chemistry provides realistic match timelines and penalty shootouts. Powering the active FastAPI backend and frontend UI.

### 4. Accuracy Optimization Round 1 (Reigning Champion)
- **Baseline**: 59.73% (Single HistGBDT baseline).
- **Feature Set**: 217 features ($F_1$) incorporating multi-scale form windows (3, 5, 10, 20), Dixon-Coles pre-match goal intensities, Bayesian H2H draws, venue advantage, and tournament stakes.
- **Models**: Gradient-boosted tree family (HistGBDT, LightGBM, XGBoost, CatBoost) + Dixon-Coles Poisson model.
- **Ensemble Result**: **60.14% (5,956 / 9,904 correct)** on the untouched 9,904 out-of-sample test set.

### 5. Accuracy Optimization Round 2
- **Exploration**: Added 24 new features (Elo velocity, form acceleration, clean-sheet ratios), tested direct 0-1 accuracy objective ensembling, decision threshold shifting, and out-of-fold stacking meta-classifiers.
- **Outcome**: Stacking meta-classifier achieved 60.29% on validation folds but degraded to 59.76% on test due to distribution shift. The continuous ensemble reached 60.12% (5,953/9,904).
- **Decision**: Strict preservation of Round 1 champion (60.14%). Round 2 code archived in `archive/optimization_r2/`.

---

## Next Research Directions

1. **Hierarchical 2-Stage Classifier**: Stage 1 predicts Result vs Draw (targeting the ~66% error share from draws); Stage 2 predicts Home vs Away conditional on a decisive match.
2. **Dynamic Live In-Match Bayesian Updating**: Extending the pre-match predictor to dynamic in-game live score updates.
