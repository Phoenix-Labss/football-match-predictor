# Dynamic Oracle — Simulation Architecture Specification

Comprehensive architecture documentation detailing the separation of concerns between the **Supervised Outcome Predictor** and the **Match-Day State Simulation Engine**.

---

## 1. System Overview: Dual-Engine Architecture

Dynamic Oracle separates **Categorical Outcome Prediction (1X2)** from **Match & Tournament Scoreline Simulation**:

```text
                             DYNAMIC ORACLE
                                   │
                 ┌─────────────────┴─────────────────┐
                 ↓                                   ↓
          OUTCOME PREDICTOR                  SIMULATION ENGINE
                 │                                   │
       Supervised Ensemble                 FIFA Player Attributes
       (60.14% OOS Champion)                         ↓
       - Logistic Regression               Age Curves & Formation XI
       - LightGBM GBDT                               ↓
       - HistGradientBoosting              Club & League Chemistry
       - Strength & Form Trackers                    ↓
                                              Match-Day State
                                              - Player Form Shocks
                                              - Performance Noise
                                              - Team Execution Shocks
                                                     ↓
                                           Expected Goals (xG)
                                                     ↓
                                           Negative Binomial (alpha)
                                                     ↓
                                           Dixon-Coles Low-Score tau
                                                     ↓
                                           Tournament Monte Carlo (10k)
```

---

## 2. Why Two Different Engines?

1. **Supervised Outcome Predictor (60.14% Accuracy)**:
   - Optimized strictly for categorical 1X2 win/draw/loss accuracy and log loss across 9,904 out-of-sample matches.
   - Uses tabular strength features, Elo trajectories, and calibrated classifier ensembles.

2. **Simulation Engine (Match-Day State + Negative Binomial + Dixon-Coles)**:
   - Solves the physical problem of realistic goal generation, blowout distributions (4+, 5+, 6+ goals), match-day form variance, and tournament bracket Monte Carlo sampling.
   - Corrects Poisson underdispersion ($\text{VMR} = 1.22$ vs observed $1.37$–$1.81$), giving realistic heavy-tail probabilities on shock blowouts.

---

## 3. Goal Generation Modes

Configured via `MatchEngineConfig(goal_model='negbin' | 'poisson', dispersion_alpha=0.1262)`:

### Negative Binomial Mode (Default Production)
- **Mean**: $\mathbb{E}[X] = \mu = \lambda$
- **Variance**: $\text{Var}(X) = \mu + \alpha \mu^2$
- **Dispersion Parameter**: $\alpha = 0.1262$ (frozen strictly from pre-tournament historical data)
- **Low-Score Coupling**: Dixon-Coles adjustment $\tau(x, y)$ on $(0,0), (1,0), (0,1), (1,1)$ with $\rho = -0.10$.

### Poisson Mode (Legacy / Reproducibility)
- Standard Dixon-Coles bivariate Poisson with $\alpha = 0.0$.
- Available anytime for backward compatibility and research benchmarking.

---

## 4. Extra Time & Penalty Shootout Modeling

1. **Normal Time (90 Minutes)**: Scoreline sampled from 90-minute joint PMF.
2. **Extra Time (30 Minutes)**: Evaluated with extra-time rate scaling, resolving overtime goals.
3. **Penalty Shootouts**: Modeled using individual player Finishing, Composure, and Goalkeeper diving abilities.

---

## 5. Recommended Monte Carlo Simulation Count
- **Production Standard**: **10,000 complete tournaments** ($\text{SE} \le \pm 0.22\%$, runtime $\approx 5.1\text{s}$).
- **High-Precision Convergence**: **25,000 complete tournaments** ($\text{SE} \le \pm 0.14\%$, runtime $\approx 12.2\text{s}$).