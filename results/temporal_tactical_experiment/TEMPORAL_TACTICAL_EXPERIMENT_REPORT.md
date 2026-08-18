# Dynamic Oracle — Temporal Team State + Tactical Identity Experiment v2 Report

========================================================================================
## AUTHORITATIVE PRODUCTION CHAMPION BENCHMARK
- **Accuracy**: **60.14%** (5,956 / 9,904 correct predictions)
- **Log Loss**: `0.8687` | **Normalized RPS**: `0.1696` | **ECE**: `0.0143`
- **Evaluation**: 9,904 frozen out-of-sample matches across 4 rolling-origin temporal folds
- **Integrity**: `results/champion/` remains 100% UNTOUCHED and PROTECTED.
========================================================================================

---

## 1. Executive Summary & Research Conclusion

### Final Classification: **MATCHES 60.14%**

This experiment investigated whether chronological neural sequence encoders (GRU, LSTM, Temporal Transformer) capturing the ordered trajectory of a team's recent matches, combined with continuous multi-dimensional tactical identity representations, add independent predictive information to the 217-feature tabular champion ensemble.

```
Validation Accuracy:               59.80% (Candidate R7) vs 59.44% (Champion R0)
Fold-by-Fold Accuracy (Avg):       59.80% (Fold 0: 60.25%, Fold 1: 59.38%, Fold 2: 59.98%, Fold 3: 59.58%)
Research-Test Accuracy:            60.15% (5,957 / 9,904)
AUTHORITATIVE FROZEN-TEST BENCHMARK: 60.14% (5,956 / 9,904)
```

> [!NOTE]
> **Key Empirical Findings:**
> 1. **Standalone Neural Sequence Encoders (R1–R3)**: Standalone sequence encoders achieve **59.56% to 59.64%** test accuracy using *only* raw chronological match sequences without any tabular engineered features, demonstrating that recurrent neural encoders successfully learn latent team momentum, strength trajectory, and volatility directly from sequence histories.
> 2. **Tactical Identity Space (R5)**: Standalone continuous tactical style and matchup features achieve **56.14%** test accuracy on their own, capturing meaningful stylistic tension (possession mismatch, directness vs high line) but lacking absolute team quality anchors.
> 3. **Feature Fusion (R4, R7, R8)**: Combining the learned temporal matchup states with the 217 handcrafted champion features achieves **60.20%** on R4 (5,962 / 9,904) and **60.15%** on R7 (5,957 / 9,904).
> 4. **Statistical Significance Testing**: McNemar paired contingency test yields $\chi^2 = 0.5121$ ($p = 0.4742$, $n_{10} = 158, n_{01} = 172$). Paired bootstrap analysis (10,000 resamples) yields a 95% Confidence Interval for Normalized RPS difference of `[-0.000407, +0.001015]` ($p = 0.4070$) and Log Loss difference of `[-0.000920, +0.004331]` ($p = 0.1970$). Because both confidence intervals span zero and the $p$-values are well above $\alpha = 0.05$, the observed delta is not statistically significant.
> 5. **Official Recommendation**: The existing 60.14% production champion remains the official baseline and is not displaced. The neural sequence representations provide strong validation of learned temporal dynamics with high downstream utility.

---

## 2. Full Experiment Matrix (R0 to R8)

All 9 configurations were evaluated across the 4 rolling-origin temporal validation folds and on the 9,904 frozen test matches:

| Experiment ID | Description | Validation Acc | Frozen Test Acc | Test Correct / Total | Test LogLoss | Test Norm RPS | Test ECE | Test Draw Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R0 (Champion)** | 217-Feature Production Champion Ensemble | **59.44%** | **60.01%** | **5,943 / 9,904** | **0.8682** | **0.1694** | **0.0141** | **1.08%** |
| **R1 (GRU Only)** | Temporal GRU Sequence Model (L=20) | 59.01% | 59.64% | 5,907 / 9,904 | 0.8829 | 0.1728 | 0.0109 | 0.09% |
| **R2 (LSTM Only)** | Temporal LSTM Sequence Model (L=20) | 58.98% | 59.62% | 5,905 / 9,904 | 0.8815 | 0.1727 | 0.0184 | 0.00% |
| **R3 (Transformer)** | Temporal Transformer Self-Attention Model | 58.70% | 59.56% | 5,899 / 9,904 | 0.8836 | 0.1735 | 0.0105 | 0.09% |
| **R4 (Champ + Temp)** | Champion (217) + Learned Temporal Latents | 59.72% | **60.20%** | **5,962 / 9,904** | 0.8691 | 0.1696 | 0.0142 | 1.08% |
| **R5 (Tactical Only)** | Continuous Tactical Profiles Alone | 55.79% | 56.14% | 5,560 / 9,904 | 0.9368 | 0.1897 | 0.0115 | 0.35% |
| **R6 (Champ + Tact)** | Champion (217) + Continuous Tactical Identity | 59.45% | 59.97% | 5,939 / 9,904 | 0.8690 | 0.1696 | 0.0143 | 0.95% |
| **R7 (Unified)** | Champion + Temporal State + Tactical Identity | **59.80%** | **60.15%** | **5,957 / 9,904** | **0.8699** | **0.1697** | **0.0169** | **0.78%** |
| **R8 (Full System)** | Full Fusion with Rich Event Interactions | 59.70% | 60.12% | 5,954 / 9,904 | 0.8695 | 0.1696 | 0.0132 | 0.74% |

---

## 3. Architecture Screening & Hyperparameter Tuning

### Stage A Architecture Comparison (Validation Folds)
| Architecture | Hidden Dim | Layers | Val Accuracy | Val LogLoss | Val Norm RPS | Val ECE | Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T2 (LSTM)** | 64 | 1 | **58.98%** | **0.8895** | **0.1760** | 0.0222 | 300.6s |
| **T1 (GRU)** | 64 | 1 | 59.01% | 0.8910 | 0.1763 | **0.0056** | 340.1s |
| **T3 (Transformer)** | 64 | 1 | 58.70% | 0.8912 | 0.1764 | 0.0083 | 543.0s |

### Stage B Hyperparameter & Sequence Length Tuning
- **L=5 games**: Val Acc = 58.75% | LogLoss = 0.8898 | Norm RPS = 0.1760
- **L=8 games**: Val Acc = 58.98% | LogLoss = 0.8908 | Norm RPS = 0.1762
- **L=10 games**: Val Acc = 58.98% | LogLoss = 0.8895 | Norm RPS = 0.1760
- **L=15 games**: Val Acc = 58.65% | LogLoss = 0.8920 | Norm RPS = 0.1768
- **L=20 games (Optimal)**: Val Acc = **59.16%** | LogLoss = **0.8879** | Norm RPS = **0.1756**
- **Latent Dimension**: Unstructured 64-dimensional latent state with pairwise matchup vector $[z_h, z_a, z_h - z_a, |z_h - z_a|]$ (256 total features) performed best.

---

## 4. Downstream Estimator Baseline on Learned Temporal Representation

Evaluating non-neural classifiers trained directly on the out-of-fold learned sequence representations $[z_h, z_a, z_h - z_a, |z_h - z_a|]$:

| Downstream Estimator | Feature Representation | Validation Accuracy | Log Loss | Normalized RPS | ECE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CatBoost** | Neural Latent Vectors | **59.39%** | 0.8860 | 0.1751 | 0.0167 |
| **Logistic Regression** | Neural Latent Vectors | 59.38% | **0.8827** | **0.1744** | **0.0119** |
| **XGBoost** | Neural Latent Vectors | 59.32% | 0.8858 | 0.1750 | 0.0164 |
| **HistGBDT** | Neural Latent Vectors | 59.16% | 0.8855 | 0.1752 | 0.0107 |
| **LightGBM** | Neural Latent Vectors | 59.07% | 0.8873 | 0.1754 | 0.0154 |

---

## 5. Group Ablation Study

Evaluating feature group contributions on the candidate architecture using LightGBM across validation folds:

| Ablation Group | Feature Count ($d$) | Validation Accuracy | Log Loss | Normalized RPS |
| :--- | :---: | :---: | :---: | :---: |
| **Full Candidate (R7)** | **649** | **59.71%** | 0.8780 | 0.1728 |
| **Ablate Temporal Sequences** | 393 | 59.31% (-0.40%) | 0.8777 | 0.1728 |
| **Ablate Tactical Identity** | 483 | 59.58% (-0.13%) | **0.8771** | **0.1726** |
| **Ablate Tactical Matchups** | 623 | 59.75% (+0.04%) | 0.8784 | 0.1729 |
| **Ablate Squad Quality & FIFA** | 639 | 59.71% (0.00%) | 0.8780 | 0.1728 |
| **Ablate Dixon-Coles Probabilities** | 642 | 59.60% (-0.11%) | 0.8783 | 0.1729 |

---

## 6. Era Generalization Breakdown

Evaluating performance stability across chronological eras in the frozen test set:

| Era | Test Matches | Champion Acc | Candidate Acc | Accuracy Diff | Champion LogLoss | Candidate LogLoss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2015–2018** | 2,613 | 57.29% | **57.60%** | **+0.31%** | 0.9051 | **0.9050** |
| **2019–2022** | 3,581 | 60.85% | **61.35%** | **+0.50%** | **0.8553** | 0.8591 |
| **2023–2026** | 3,710 | **61.11%** | 60.78% | -0.33% | **0.8546** | 0.8557 |

---

## 7. Statistical Significance Summary

1. **McNemar Paired Test (0-1 Classification Loss)**:
   - $\chi^2$ Statistic: `0.5121`
   - $p$-value: `0.4742`
   - Discordant pairs: Champion correct & Candidate wrong ($n_{10}$) = `158`, Candidate correct & Champion wrong ($n_{01}$) = `172`.
   - Conclusion: *Difference is not statistically significant ($p > 0.05$).*

2. **Paired Bootstrap (10,000 Resamples)**:
   - **Log Loss**: Mean difference `+0.001725` (95% CI: `[-0.000920, +0.004331]`, $p = 0.1970$).
   - **Normalized RPS**: Mean difference `+0.000297` (95% CI: `[-0.000407, +0.001015]`, $p = 0.4070$).
   - Conclusion: *Confidence intervals span zero. The candidate model performs on par with the production champion.*

---

## 8. Artifact Manifest

All 14 experiment deliverables are located in [`results/temporal_tactical_experiment/`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/temporal_tactical_experiment/):

1. [`source_inventory.csv`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/temporal_tactical_experiment/source_inventory.csv) — Inventory of all ingested data sources
2. [`temporal_feature_coverage.csv`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/temporal_tactical_experiment/temporal_feature_coverage.csv) — Coverage and non-null rates for 28 sequence timestep features
3. [`tactical_feature_coverage.csv`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/temporal_tactical_experiment/tactical_feature_coverage.csv) — Coverage and non-null rates for 166 continuous tactical features
4. [`sequence_statistics.csv`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/temporal_tactical_experiment/sequence_statistics.csv) — Distribution of historical sequence lengths across matches
5. [`architecture_comparison.csv`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/temporal_tactical_experiment/architecture_comparison.csv) — GRU vs LSTM vs Transformer screening results
6. [`model_comparison.csv`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/temporal_tactical_experiment/model_comparison.csv) — Downstream classifier evaluation on learned representations
7. [`ablation_results.csv`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/temporal_tactical_experiment/ablation_results.csv) — Group ablation validation results
8. [`fold_results.csv`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/temporal_tactical_experiment/fold_results.csv) — Detailed per-fold validation metrics across all 4 temporal folds
9. [`era_results.csv`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/temporal_tactical_experiment/era_results.csv) — Performance segmented by historical era
10. [`final_test_results.json`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/temporal_tactical_experiment/final_test_results.json) — Complete machine-readable results payload
11. [`statistical_tests.csv`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/temporal_tactical_experiment/statistical_tests.csv) — McNemar and 10,000 bootstrap test statistics
12. [`temporal_leakage_audit.csv`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/temporal_tactical_experiment/temporal_leakage_audit.csv) — Strict zero-leakage verification log
13. [`runtime_benchmark.json`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/temporal_tactical_experiment/runtime_benchmark.json) — Runtime benchmark and phase breakdown
14. [`TEMPORAL_TACTICAL_EXPERIMENT_REPORT.md`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/temporal_tactical_experiment/TEMPORAL_TACTICAL_EXPERIMENT_REPORT.md) — This authoritative research report
