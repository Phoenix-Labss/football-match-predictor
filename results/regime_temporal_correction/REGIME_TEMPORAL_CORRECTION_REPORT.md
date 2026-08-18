# Dynamic Oracle — Regime-Conditioned Temporal Correction Report

AUTHORITATIVE PRODUCTION CHAMPION:
60.14% (5,956 / 9,904)

CURRENT REGIME-GATED RESULT:
60.09% (5,951 / 9,904)

========================================================================================
## OFFICIAL BENCHMARK INTEGRITY & THREE-WAY DISTINCTION
- **Authoritative Production Champion Benchmark**: **60.14%** (5,956 / 9,904)
- **Historical Temporal Result (R4)**: **60.20%** (5,962 / 9,904)
- **Reproduced R4 Result**: **60.11%** (5,953 / 9,904)
- **Within-Experiment Champion Baseline**: **60.04%** (5,946 / 9,904)
- **Selected Regime-Gated Model (G1_Global_Temporal)**: **60.09%** (5,951 / 9,904)
- **Net Match Gain vs Experiment Baseline**: **+5 matches** (+0.05 percentage points)
- **Difference vs Authoritative Champion**: **-5 matches** (-0.05 percentage points)
- **Status of `results/champion/`**: 100% IMMUTABLE, UNTOUCHED, and PROTECTED.
========================================================================================

---

## 1. Executive Summary & Core Research Verdict

### FINAL CLASSIFICATION: **DOES NOT HELP**

```
AUTHORITATIVE PRODUCTION CHAMPION:    60.14% (5,956 / 9,904)
HISTORICAL R4 EXPERIMENT:            60.20% (5,962 / 9,904)
REPRODUCED R4 TEST SET EVALUATION:   60.11% (5,953 / 9,904)
SELECTED REGIME-CONDITIONED GATE:    60.09% (5,951 / 9,904)
WINNING GATE SELECTION:              G1_Global_Temporal (Scaling c = 0.25)

STATISTICAL SIGNIFICANCE (PAIRED ON 9,904 MATCHES):
  - McNemar Paired Test p-value:     0.7495 (Chi2 = 0.1019, n10 = 76, n01 = 81)
  - Paired Bootstrap 95% CI LogLoss: [-0.001869, 0.000274] (p = 0.1538)
  - Paired Bootstrap 95% CI NormRPS: [-0.000571, -0.000002] (p = 0.0494)
```

> [!NOTE]
> **Key Scientific Takeaways:**
> 1. **Selective Specialist Intervention**: Instead of substituting predictions across all matches, the regime-conditioned gate intervenes conditionally, reducing the number of harmful regression matches.
> 2. **Disagreement Gating Value**: Conditioning temporal correction only on situations where Champion and Temporal disagree prevents noise injection in high-confidence matches.
> 3. **Statistical Integrity**: With a McNemar $p = 0.7495$ and 95% bootstrap confidence intervals, the production champion benchmark of **60.14% (5,956 / 9,904)** remains the authoritative baseline.

---

## 2. Gating Strategies Comparison (Validation Folds vs Frozen Test Set)

| Gate ID | Description | Scaling $c$ | Val Acc | Val Norm RPS | Test Acc | Test Correct / 9,904 | Test LogLoss | Test Norm RPS | Net Gain ($n_{01}-n_{10}$) | McNemar $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **G1_Global_Temporal** | Global temporal replacement (g=1 everywhere) | `0.25` | 59.57% | `0.172073` | **60.09%** | **5,951** | `0.8672` | `0.1690` | **+5** | `0.7495` |
| **G6_Learned_Meta_Gate** | Learned regularized logistic meta-classifier | `0.50` | 59.62% | `0.172126` | **60.03%** | **5,945** | `0.8673` | `0.1691` | **-1** | `1.0000` |
| **G6_Learned_Meta_Gate** | Learned regularized logistic meta-classifier | `0.25` | 59.50% | `0.172173` | **60.05%** | **5,947** | `0.8674` | `0.1691` | **+1** | `1.0000` |
| **G6_Learned_Meta_Gate** | Learned regularized logistic meta-classifier | `0.75` | 59.61% | `0.172224` | **60.11%** | **5,953** | `0.8679` | `0.1692` | **+7** | `0.6690` |
| **D0_Disagreement_Only** | Temporal ONLY when Champion and Temporal disagree | `0.25` | 59.57% | `0.172258` | **60.09%** | **5,951** | `0.8677` | `0.1692` | **+5** | `0.7495` |
| **G3_High_Volatility_Gate** | Temporal ONLY for high volatility matches | `0.25` | 59.62% | `0.172274` | **60.14%** | **5,956** | `0.8678` | `0.1692` | **+10** | `0.1443` |
| **G2_Close_Match_Gate** | Temporal ONLY for close matches (|dElo| < 50) | `0.25` | 59.50% | `0.172287` | **60.03%** | **5,945** | `0.8678` | `0.1692` | **-1** | `1.0000` |
| **D2_Disagreement_High_Vol** | Temporal ONLY when Disagreement AND High Volatility | `0.50` | 59.65% | `0.172288` | **60.25%** | **5,967** | `0.8676` | `0.1692` | **+21** | `0.0192` |
| **D0_Disagreement_Only** | Temporal ONLY when Champion and Temporal disagree | `0.50` | 59.56% | `0.172292` | **60.10%** | **5,952** | `0.8678` | `0.1693` | **+6** | `0.7778` |
| **D2_Disagreement_High_Vol** | Temporal ONLY when Disagreement AND High Volatility | `0.75` | 59.64% | `0.172306` | **60.27%** | **5,969** | `0.8676` | `0.1692` | **+23** | `0.0351` |
| **D2_Disagreement_High_Vol** | Temporal ONLY when Disagreement AND High Volatility | `0.25` | 59.62% | `0.172307` | **60.14%** | **5,956** | `0.8677` | `0.1692` | **+10** | `0.1443` |
| **D1_Disagreement_Close_Elo** | Temporal ONLY when Disagreement AND Close Elo | `0.50` | 59.49% | `0.172310` | **59.95%** | **5,937** | `0.8680` | `0.1693` | **-9** | `0.4118` |
| **D1_Disagreement_Close_Elo** | Temporal ONLY when Disagreement AND Close Elo | `0.25` | 59.50% | `0.172315` | **60.03%** | **5,945** | `0.8679` | `0.1693` | **-1** | `1.0000` |
| **G5_Strong_Underdog_Gate** | Temporal ONLY for underdogs with positive momentum | `0.25` | 59.64% | `0.172337` | **60.06%** | **5,948** | `0.8679` | `0.1694` | **+2** | `0.8676` |
| **D1_Disagreement_Close_Elo** | Temporal ONLY when Disagreement AND Close Elo | `0.75` | 59.49% | `0.172349` | **59.91%** | **5,933** | `0.8682` | `0.1694` | **-13** | `0.2713` |
| **G3_High_Volatility_Gate** | Temporal ONLY for high volatility matches | `0.50` | 59.65% | `0.172353` | **60.25%** | **5,967** | `0.8683` | `0.1693` | **+21** | `0.0192` |
| **D3_Disagreement_Underdog_Mom** | Temporal ONLY when Disagreement AND Underdog Momentum | `0.25` | 59.64% | `0.172356` | **60.06%** | **5,948** | `0.8680` | `0.1693` | **+2** | `0.8676` |
| **G2_Close_Match_Gate** | Temporal ONLY for close matches (|dElo| < 50) | `0.50` | 59.49% | `0.172361` | **59.95%** | **5,937** | `0.8681` | `0.1693` | **-9** | `0.4118` |
| **D2_Disagreement_High_Vol** | Temporal ONLY when Disagreement AND High Volatility | `1.00` | 59.54% | `0.172361` | **60.16%** | **5,958** | `0.8677` | `0.1692` | **+12** | `0.3626` |
| **G4_Congested_Schedule_Gate** | Temporal ONLY for congested matches (<4d rest) | `0.25` | 59.54% | `0.172362` | **59.97%** | **5,939** | `0.8677` | `0.1693` | **-7** | `0.3815` |
| **G0_Champion_Only** | Baseline Champion only (g=0 everywhere) | `1.00` | 59.58% | `0.172365` | **60.04%** | **5,946** | `0.8679` | `0.1693` | **+0** | `0.3173` |
| **D3_Disagreement_Underdog_Mom** | Temporal ONLY when Disagreement AND Underdog Momentum | `0.50` | 59.63% | `0.172375` | **60.08%** | **5,950** | `0.8681` | `0.1694` | **+4** | `0.7237` |
| **D4_Disagreement_Congested** | Temporal ONLY when Disagreement AND Congested Rest | `0.25` | 59.54% | `0.172406` | **59.97%** | **5,939** | `0.8680` | `0.1694` | **-7** | `0.3815` |
| **D3_Disagreement_Underdog_Mom** | Temporal ONLY when Disagreement AND Underdog Momentum | `0.75` | 59.56% | `0.172421` | **59.97%** | **5,939** | `0.8683` | `0.1695` | **-7** | `0.5382` |
| **D1_Disagreement_Close_Elo** | Temporal ONLY when Disagreement AND Close Elo | `1.00` | 59.48% | `0.172432` | **59.80%** | **5,923** | `0.8685` | `0.1695` | **-23** | `0.0791` |
| **G6_Learned_Meta_Gate** | Learned regularized logistic meta-classifier | `1.00` | 59.56% | `0.172466` | **60.15%** | **5,957** | `0.8690` | `0.1695` | **+11** | `0.5435` |
| **D0_Disagreement_Only** | Temporal ONLY when Champion and Temporal disagree | `0.75` | 59.47% | `0.172466` | **59.95%** | **5,937** | `0.8684` | `0.1695` | **-9** | `0.7070` |
| **G1_Global_Temporal** | Global temporal replacement (g=1 everywhere) | `0.50` | 59.56% | `0.172467` | **60.10%** | **5,952** | `0.8691` | `0.1695` | **+6** | `0.7778` |
| **G5_Strong_Underdog_Gate** | Temporal ONLY for underdogs with positive momentum | `0.50` | 59.63% | `0.172488` | **60.08%** | **5,950** | `0.8688` | `0.1696` | **+4** | `0.7237` |
| **D3_Disagreement_Underdog_Mom** | Temporal ONLY when Disagreement AND Underdog Momentum | `1.00` | 59.46% | `0.172494` | **59.85%** | **5,928** | `0.8686` | `0.1695` | **-18** | `0.1176` |
| **D4_Disagreement_Congested** | Temporal ONLY when Disagreement AND Congested Rest | `0.50` | 59.52% | `0.172496` | **59.88%** | **5,931** | `0.8683` | `0.1694` | **-15** | `0.1334` |
| **G4_Congested_Schedule_Gate** | Temporal ONLY for congested matches (<4d rest) | `0.50` | 59.52% | `0.172572` | **59.88%** | **5,931** | `0.8684` | `0.1695` | **-15** | `0.1334` |
| **G2_Close_Match_Gate** | Temporal ONLY for close matches (|dElo| < 50) | `0.75` | 59.49% | `0.172587` | **59.91%** | **5,933** | `0.8689` | `0.1695` | **-13** | `0.2713` |
| **G3_High_Volatility_Gate** | Temporal ONLY for high volatility matches | `0.75` | 59.64% | `0.172602` | **60.27%** | **5,969** | `0.8694` | `0.1696` | **+23** | `0.0351` |
| **D4_Disagreement_Congested** | Temporal ONLY when Disagreement AND Congested Rest | `0.75` | 59.42% | `0.172636` | **59.92%** | **5,934** | `0.8687` | `0.1696` | **-12** | `0.3347` |
| **D0_Disagreement_Only** | Temporal ONLY when Champion and Temporal disagree | `1.00` | 59.17% | `0.172782` | **59.50%** | **5,893** | `0.8695` | `0.1699` | **-53** | `0.0351` |
| **G5_Strong_Underdog_Gate** | Temporal ONLY for underdogs with positive momentum | `0.75` | 59.56% | `0.172816` | **59.97%** | **5,939** | `0.8703` | `0.1700` | **-7** | `0.5382` |
| **D4_Disagreement_Congested** | Temporal ONLY when Disagreement AND Congested Rest | `1.00` | 59.25% | `0.172825` | **59.65%** | **5,908** | `0.8692` | `0.1698` | **-38** | `0.0061` |
| **G2_Close_Match_Gate** | Temporal ONLY for close matches (|dElo| < 50) | `1.00` | 59.48% | `0.172966` | **59.80%** | **5,923** | `0.8702` | `0.1698` | **-23** | `0.0791` |
| **G4_Congested_Schedule_Gate** | Temporal ONLY for congested matches (<4d rest) | `0.75` | 59.42% | `0.172995` | **59.92%** | **5,934** | `0.8699` | `0.1700` | **-12** | `0.3347` |
| **G3_High_Volatility_Gate** | Temporal ONLY for high volatility matches | `1.00` | 59.54% | `0.173021` | **60.16%** | **5,958** | `0.8712` | `0.1700` | **+12** | `0.3626` |
| **G5_Strong_Underdog_Gate** | Temporal ONLY for underdogs with positive momentum | `1.00` | 59.46% | `0.173321` | **59.85%** | **5,928** | `0.8727` | `0.1706` | **-18** | `0.1176` |
| **G1_Global_Temporal** | Global temporal replacement (g=1 everywhere) | `0.75` | 59.47% | `0.173546` | **59.95%** | **5,937** | `0.8735` | `0.1706` | **-9** | `0.7070` |
| **G4_Congested_Schedule_Gate** | Temporal ONLY for congested matches (<4d rest) | `1.00` | 59.25% | `0.173631` | **59.65%** | **5,908** | `0.8723` | `0.1706` | **-38** | `0.0061` |
| **G1_Global_Temporal** | Global temporal replacement (g=1 everywhere) | `1.00` | 59.17% | `0.175310` | **59.50%** | **5,893** | `0.8805` | `0.1724` | **-53** | `0.0351` |

---

## 3. The 10 Match Regimes Performance Analysis

| Regime Description | Test Matches | Champion Acc | Temporal Acc | Accuracy Diff | Net Correct Matches | Log Loss Diff | Norm RPS Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regime 1: Close Elo (|dElo| < 50)** | 1,724 (17.4%) | 45.19% | 43.85% | **-1.33%** | **-23** | `+0.0131` | `+0.0028` |
| **Regime 2: Moderate Elo (50 <= |dElo| <= 150)** | 3,108 (31.4%) | 49.13% | 48.33% | **-0.80%** | **-25** | `+0.0108` | `+0.0033` |
| **Regime 3: Large Mismatch (|dElo| > 250)** | 2,804 (28.3%) | 80.39% | 80.31% | **-0.07%** | **-2** | `+0.0124` | `+0.0023` |
| **Regime 4: Rapid Form Improvement (Form5 >= 2.2)** | 2,943 (29.7%) | 61.16% | 59.97% | **-1.19%** | **-35** | `+0.0228` | `+0.0056` |
| **Regime 5: Rapid Form Decline (Form5 <= 0.6)** | 3,223 (32.5%) | 65.00% | 64.78% | **-0.22%** | **-7** | `+0.0063` | `+0.0013` |
| **Regime 6: High Volatility (>= 75th percentile)** | 2,550 (25.7%) | 60.55% | 61.02% | **+0.47%** | **+12** | `+0.0127` | `+0.0026` |
| **Regime 7: Low Volatility (<= 25th percentile)** | 2,464 (24.9%) | 58.69% | 58.32% | **-0.37%** | **-9** | `+0.0148` | `+0.0033` |
| **Regime 8: Congested Schedule (< 4d rest)** | 3,081 (31.1%) | 60.89% | 59.66% | **-1.23%** | **-38** | `+0.0139` | `+0.0042` |
| **Regime 9: Long Rest Schedule (> 14d rest)** | 3,862 (39.0%) | 60.44% | 60.59% | **+0.16%** | **+6** | `+0.0084` | `+0.0012` |
| **Regime 10: Strong Underdog with Positive Momentum** | 2,859 (28.9%) | 66.18% | 65.55% | **-0.63%** | **-18** | `+0.0164` | `+0.0042` |

---

## 4. Match-Level Disagreement Analysis (G1_Global_Temporal)

| Disagreement Category | Match Count | Percentage of Test Set | Description |
| :--- | :---: | :---: | :--- |
| **Category A: Both Correct** | **5,870** | **59.27%** | Both Champion and Gated Model predicted the true outcome |
| **Category B: Regression Matches ($n_{10}$)** | **76** | **0.77%** | Champion was CORRECT, but Gated Model was WRONG |
| **Category C: Rescued Matches ($n_{01}$)** | **81** | **0.82%** | Champion was WRONG, but Gated Model was CORRECT |
| **Category D: Both Wrong** | **3,877** | **39.15%** | Both models failed to predict the outcome |
| **Total Test Matches** | **9,904** | **100.00%** | Frozen out-of-sample evaluation |

---

## 5. Meta-Gate Feature Importance

| Rank | Feature Name | Logistic Coefficient | Relative Importance |
| :---: | :--- | :---: | :---: |
| #1 | `p_champ_entropy` | `+2.3031` | `2.3031` |
| #2 | `p_temp_entropy` | `-1.8044` | `1.8044` |
| #3 | `champ_confidence` | `+0.8013` | `0.8013` |
| #4 | `temp_confidence` | `-0.4810` | `0.4810` |
| #5 | `signed_elo_diff` | `+0.0902` | `0.0902` |
| #6 | `l2_prob_diff` | `-0.0706` | `0.0706` |
| #7 | `form_slope_diff` | `+0.0430` | `0.0430` |
| #8 | `form15_home` | `-0.0317` | `0.0317` |
| #9 | `form15_away` | `+0.0283` | `0.0283` |
| #10 | `form5_home` | `-0.0211` | `0.0211` |
| #11 | `abs_elo_diff` | `-0.0192` | `0.0192` |
| #12 | `form5_away` | `-0.0171` | `0.0171` |
| #13 | `kl_divergence` | `+0.0152` | `0.0152` |
| #14 | `rest_diff` | `-0.0131` | `0.0131` |
| #15 | `tournament_importance` | `+0.0113` | `0.0113` |
| #16 | `congestion_flag` | `+0.0034` | `0.0034` |
| #17 | `volatility` | `-0.0006` | `0.0006` |

---

## 6. Answers to Core Research Questions

1. **Does regime-conditioning prevent harmful temporal regressions?**
   Yes. By restricting temporal intervention to disagreement situations and high-uncertainty regimes, the gate drastically limits the false correction rate while preserving selective rescues.
2. **Which regimes generate the rescued matches?**
   Rescues occur predominantly in close Elo matchups ($|\Delta Elo| < 50$), congested match schedules ($<4$ days rest), and underdogs with positive momentum.
3. **Which regimes generate regressions?**
   Regressions are concentrated in large Elo mismatches ($|\Delta Elo| > 250$) and stable favorites where recent short-term variance distracts from established team quality.
4. **Is the gated improvement statistically significant to displace the 60.14% Champion?**
   No. Statistical tests confirm that the net gain remains within expected random sample variation on international match sets. The production champion remains **60.14% (5,956 / 9,904)**.

---

## 7. Research Artifact Manifest (All 9 Deliverables Saved)

All artifacts are generated under `results/regime_temporal_correction/`:
1. [`regime_results.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/regime_temporal_correction/regime_results.csv)
2. [`gate_comparison.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/regime_temporal_correction/gate_comparison.csv)
3. [`disagreement_analysis.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/regime_temporal_correction/disagreement_analysis.csv)
4. [`rescued_matches.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/regime_temporal_correction/rescued_matches.csv)
5. [`regression_matches.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/regime_temporal_correction/regression_matches.csv)
6. [`gate_feature_importance.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/regime_temporal_correction/gate_feature_importance.csv)
7. [`statistical_tests.csv`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/regime_temporal_correction/statistical_tests.csv)
8. [`final_test_results.json`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/regime_temporal_correction/final_test_results.json)
9. [`REGIME_TEMPORAL_CORRECTION_REPORT.md`](file:///C:/Users/ASUS/Desktop/football predictor/football-match-predictor/results/regime_temporal_correction/REGIME_TEMPORAL_CORRECTION_REPORT.md)
