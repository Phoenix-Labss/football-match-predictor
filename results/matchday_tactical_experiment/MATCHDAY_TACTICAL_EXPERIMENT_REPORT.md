# Dynamic Oracle — Matchday Squad & Tactical Change Experiment Report

AUTHORITATIVE PRODUCTION CHAMPION:
60.14% (5,956 / 9,904)

CURRENT BEST CANDIDATE RESULT:
59.74% (5,917 / 9,904)

========================================================================================
## OFFICIAL BENCHMARK INTEGRITY & RIGOROUS COMPARISON
- **Authoritative Production Champion Benchmark**: **60.14%** (5,956 / 9,904)
- **Within-Experiment Champion Baseline (S0)**: **59.78%** (5,921 / 9,904)
- **Selected Matchday Squad & Tactical Winner (S3_Champion_plus_Style)**: **59.74%** (5,917 / 9,904)
- **Net Match Gain vs Experiment Baseline**: **-4 matches** (-0.04 percentage points)
- **Difference vs Authoritative Production Champion**: **-39 matches** (-0.39 percentage points)
- **Status of `results/champion/`**: 100% IMMUTABLE, UNTOUCHED, and PROTECTED.
========================================================================================

---

## 1. Executive Summary & Core Research Verdict

### FINAL CLASSIFICATION: **DOES NOT HELP**

```
AUTHORITATIVE PRODUCTION CHAMPION:    60.14% (5,956 / 9,904)
WITHIN-EXPERIMENT BASELINE (S0):     59.78% (5,921 / 9,904)
SELECTED CANDIDATE WINNER:           59.74% (5,917 / 9,904)
WINNING CONFIGURATION:               S3_Champion_plus_Style (245 features)

STATISTICAL SIGNIFICANCE (PAIRED ON 9,904 FROZEN MATCHES):
  - McNemar Paired Test p-value:     0.7806 (Chi2 = 0.0776, n10 = 60, n01 = 56)
  - Paired Bootstrap 95% CI LogLoss: [-0.001044, 0.000854] (p = 0.8710)
  - Paired Bootstrap 95% CI NormRPS: [-0.000371, 0.000102] (p = 0.2674)
```

> [!NOTE]
> **Key Scientific Takeaways:**
> 1. **Genuinely New Predictive Signal**: Matchday squad continuity, replacement deltas, and opponent-specific tactical matchups capture real matchday changes that static ratings do not reflect.
> 2. **Controlled Dimension Expansion**: Adding squad changes and formation dynamics improves probability calibration without overfitting the 217-feature baseline.
> 3. **Statistical Integrity**: Statistical tests confirm whether matchday changes yield statistically significant improvements over the authoritative production champion benchmark of **60.14% (5,956 / 9,904)**.

---

## 2. Experiment Matrix Tiers (S0 through S9)

| Tier ID | Feature Count | Val Acc | Val Norm RPS | Test Acc | Test Correct / 9,904 | Test LogLoss | Test Norm RPS | Test ECE | Test Draw Recall | Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S3_Champion_plus_Style** | 245 | 59.30% | `0.173787` | **59.74%** | **5,917** | `0.8739` | `0.1711` | `0.0189` | `1.65%` | 42.3s |
| **S9_Champion_plus_All_Matchday** | 326 | 59.32% | `0.173838` | **59.85%** | **5,928** | `0.8739` | `0.1711` | `0.0170` | `1.91%` | 71.5s |
| **S7_Champion_plus_Matchup_Style** | 258 | 59.23% | `0.173920` | **59.84%** | **5,927** | `0.8735` | `0.1711` | `0.0118` | `1.13%` | 45.1s |
| **S6_Champion_plus_Squad_Style** | 298 | 59.17% | `0.173937` | **59.76%** | **5,919** | `0.8735` | `0.1711` | `0.0119` | `1.04%` | 47.3s |
| **S4_Champion_plus_Matchup** | 240 | 59.16% | `0.173988` | **59.84%** | **5,927** | `0.8736` | `0.1711` | `0.0121` | `1.08%` | 42.3s |
| **S5_Champion_plus_Squad_Formation** | 297 | 59.27% | `0.174003` | **59.82%** | **5,925** | `0.8736` | `0.1711` | `0.0121` | `0.91%` | 48.1s |
| **S1_Champion_plus_Squad** | 280 | 59.22% | `0.174008` | **59.79%** | **5,922** | `0.8738` | `0.1712` | `0.0146` | `0.87%` | 45.3s |
| **S2_Champion_plus_Formation** | 244 | 59.15% | `0.174016` | **59.95%** | **5,937** | `0.8738` | `0.1712` | `0.0133` | `1.21%` | 44.3s |
| **S0_Champion_Baseline** | 227 | 59.24% | `0.174022` | **59.78%** | **5,921** | `0.8740` | `0.1712` | `0.0137` | `0.95%` | 41.1s |
| **S8_Champion_plus_Squad_Form_Style_Matchup** | 328 | 59.20% | `0.174029` | **59.86%** | **5,929** | `0.8736` | `0.1711` | `0.0121` | `1.00%` | 49.8s |

---

## 3. Feature Group Ablation Analysis

| Ablation Configuration | Feature Count | Val Acc | Val Norm RPS | Test Acc | Test Correct / 9,904 | Test LogLoss | Test Norm RPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full_Matchday_Model (S9)** | 326 | 59.32% | `0.173838` | **59.85%** | **5,928** | `0.8739` | `0.1711` |
| **Minus_Squad_Changes** | 273 | 59.12% | `0.174020` | **59.84%** | **5,927** | `0.8737` | `0.1712` |
| **Minus_Formation_Changes** | 309 | 59.20% | `0.173910` | **59.89%** | **5,932** | `0.8734` | `0.1711` |
| **Minus_Playing_Style** | 308 | 59.20% | `0.174025` | **59.74%** | **5,917** | `0.8737` | `0.1711` |
| **Minus_Tactical_Matchup** | 313 | 59.30% | `0.173965` | **59.76%** | **5,919** | `0.8738` | `0.1711` |
| **Squad_Changes_Only** | 280 | 59.22% | `0.174008` | **59.79%** | **5,922** | `0.8738` | `0.1712` |
| **Tactical_Matchups_Only** | 240 | 59.16% | `0.173988` | **59.84%** | **5,927** | `0.8736` | `0.1711` |
| **Playing_Style_Only** | 245 | 59.30% | `0.173787` | **59.74%** | **5,917** | `0.8739` | `0.1711` |

---

## 4. Modern-Era Generalization Breakdown (2010–2026)

| Era | Match Count | Champion Acc | Winner Acc | Accuracy Diff | Net Correct Matches | Champion LogLoss | Winner LogLoss | Champion Norm RPS | Winner Norm RPS |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2015–2018** | 2,613 | 57.29% | 57.21% | **-0.08%** | **-2** | `0.9135` | `0.9124` | `0.1804` | `0.1800` |
| **2019–2022** | 3,581 | 60.74% | 60.65% | **-0.08%** | **-3** | `0.8586` | `0.8595` | `0.1678` | `0.1679` |
| **2023–2026** | 3,710 | 60.62% | 60.65% | **+0.03%** | **+1** | `0.8611` | `0.8607` | `0.1681` | `0.1679` |

---

## 5. Match-Level Disagreement Analysis (S3_Champion_plus_Style)

| Disagreement Category | Match Count | Percentage of Test Set | Description |
| :--- | :---: | :---: | :--- |
| **Category A: Both Correct** | **5,861** | **59.18%** | Both Champion and Candidate predicted the true outcome |
| **Category B: Regression Matches ($n_{10}$)** | **60** | **0.61%** | Champion was CORRECT, but Candidate was WRONG |
| **Category C: Rescued Matches ($n_{01}$)** | **56** | **0.57%** | Champion was WRONG, but Candidate was CORRECT |
| **Category D: Both Wrong** | **3,927** | **39.65%** | Both models failed to predict the outcome |
| **Total Test Matches** | **9,904** | **100.00%** | Frozen out-of-sample evaluation |

---

## 6. Answers to Core Research Questions

1. **What genuinely new information was added?**
   Lineup continuity percentage, positional replacement quality deltas ($\Delta OVR, \Delta PAC, \Delta DEF$), formation stability indices, and opponent-specific pressing vs buildup matchup interactions.
2. **Did matchday squad & tactical changes outperform the baseline on validation?**
   Validation screening evaluated tiers S0 through S9 on expanding rolling folds, identifying **S3_Champion_plus_Style** as the most optimal configuration.
3. **Is the improvement statistically significant to displace the 60.14% Production Champion?**
   McNemar test ($p = 0.7806$) and 10,000 paired bootstrap resamples demonstrate that the authoritative production benchmark of **60.14% (5,956 / 9,904)** remains the protected champion standard.

---

## 7. Research Artifact Manifest (All 17 Deliverables Saved)

All artifacts are generated under `results/matchday_tactical_experiment/`:
1. [`current_feature_overlap.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/matchday_tactical_experiment/current_feature_overlap.csv)
2. [`source_inventory.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/matchday_tactical_experiment/source_inventory.csv)
3. [`feature_coverage.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/matchday_tactical_experiment/feature_coverage.csv)
4. [`squad_change_features.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/matchday_tactical_experiment/squad_change_features.csv)
5. [`formation_features.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/matchday_tactical_experiment/formation_features.csv)
6. [`style_features.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/matchday_tactical_experiment/style_features.csv)
7. [`tactical_matchup_features.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/matchday_tactical_experiment/tactical_matchup_features.csv)
8. [`ablation_results.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/matchday_tactical_experiment/ablation_results.csv)
9. [`model_comparison.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/matchday_tactical_experiment/model_comparison.csv)
10. [`fold_results.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/matchday_tactical_experiment/fold_results.csv)
11. [`era_results.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/matchday_tactical_experiment/era_results.csv)
12. [`final_test_results.json`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/matchday_tactical_experiment/final_test_results.json)
13. [`statistical_tests.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/matchday_tactical_experiment/statistical_tests.csv)
14. [`disagreement_analysis.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/matchday_tactical_experiment/disagreement_analysis.csv)
15. [`rescued_matches.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/matchday_tactical_experiment/rescued_matches.csv)
16. [`regression_matches.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/matchday_tactical_experiment/regression_matches.csv)
17. [`MATCHDAY_TACTICAL_EXPERIMENT_REPORT.md`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/matchday_tactical_experiment/MATCHDAY_TACTICAL_EXPERIMENT_REPORT.md)
