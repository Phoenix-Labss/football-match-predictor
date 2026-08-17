# Dynamic Oracle — Current Champion Feature Inventory

Complete inventory of all **217 pre-match features** currently utilized by the 60.14% Supervised Champion Ensemble.

| Feature Category | Count | Primary Data Source | Temporal Constraint | Leakage Audit |
|:---|---:|:---|:---|:---|
| **Rolling Form & EWMA** | 184 | Match Results History (1872–Present) | Pre-kickoff only | Verified Clean |
| **Elo & Team Strength** | 15 | Adaptive Strength Tracker | Pre-kickoff state | Verified Clean |
| **Poisson & Dixon-Coles** | 7 | Bivariate Poisson Intensities | Pre-kickoff ratings | Verified Clean |
| **Match Context & Rest** | 7 | Match Metadata | Fixed schedule | Verified Clean |
| **Head-to-Head (H2H)** | 4 | Pairwise Historic Encounters | Prior meetings only | Verified Clean |
| **Player & Squad Attributes** | 0 | FIFA Male Players Database | Static edition lookup | Verified Clean |

**Total Features in Production Champion**: `217`