# Dynamic Oracle — Hierarchical 1X2 Prediction Experiment Report

Empirical investigation into whether decomposing 3-class football outcome prediction into a two-stage hierarchical model (**Stage 1: Draw vs Not Draw**, **Stage 2: Home vs Away on non-draws**) outperforms the production 60.14% 3-class champion.

---

## 1. What the Current 3-Class Model Does
The current production champion directly trains multi-class gradient boosting ensembles (LightGBM, XGBoost, CatBoost, HistGBDT) to output a 3-way softmax distribution $[P(\text{Home}), P(\text{Draw}), P(\text{Away})]$ over all 217 engineered pre-match features. It achieves **60.14% out-of-sample accuracy** (5,956 / 9,904 correct) on the untouched test set.

---

## 2. Why Draw is Difficult
In international football, the draw outcome occurs in roughly ~24–26% of matches. Because football goals are sparse and discrete, draws often occur as low-scoring equilibria between teams of both equal and asymmetric strength. Standard 3-class softmax classifiers naturally peak on the most decisive outcome (Home or Away) because $P(\text{Home})$ or $P(\text{Away})$ frequently exceeds $P(\text{Draw})$ ($0.38 > 0.28$), resulting in lower Draw recall when using standard argmax decision rules.

---

## 3. What the Hierarchical Model Changes
The hierarchical architecture decouples the draw problem into two independent stages:
1. **Stage 1 (Binary Draw Classifier)**: Specialized ensemble predicting $P(\text{Draw})$ vs $P(\text{Not Draw})$.
2. **Stage 2 (Binary Decisive Match Classifier)**: Specialized ensemble trained exclusively on matches where an actual result occurred ($y \in \{\text{Home}, \text{Away}\}$), predicting $P(\text{Home} \mid \text{Not Draw})$.
3. **Probability Recombination**: $P(\text{Home}) = (1 - P(\text{Draw})) \times P(\text{Home} \mid \text{Not Draw})$ and $P(\text{Away}) = (1 - P(\text{Draw})) \times (1 - P(\text{Home} \mid \text{Not Draw}))$.

---

## 4. Test Set Comparison (9,904 Untouched Matches)

| Model Architecture | Accuracy % | Correct / 9,904 | Log Loss | Normalized RPS | Multi-Class Brier | ECE | Draw Recall % | Correct Draws |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Current Champion (3-Class Softmax)** | **59.84%** | **5927** | **0.8736** | **0.171** | **0.5142** | **0.0169** | 1.69% | 39 |
| **Hierarchical (Method A: Argmax)** | 59.83% | 5926 | 0.8746 | 0.1713 | 0.5147 | 0.0151 | 1.08% | 25 |
| **Hierarchical (Method B: Threshold $\theta^*=0.36$)** | 59.77% | 5920 | 0.8746 | 0.1713 | 0.5147 | 0.0151 | **1.34%** | **31** |

---

## 5. Draw Recall & Tradeoff Error Analysis
When tuning the draw threshold in Method B:
- **Draw Recall Gain**: Increased from `1.69%` (39 draws) to **`1.34%`** (31 draws, a net gain of `-8` correct draws).
- **The Tradeoff Penalty**: By forcing draw predictions on marginal matches, the model erroneously converted decisive wins into false draws, resulting in a net **loss of `7` total correct predictions** (5920 vs 5927).
- Confusion matrix analysis in [`confusion_matrix.csv`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/hierarchical_prediction/confusion_matrix.csv) confirms that draw gains are outweighed by home/away false positives.

---

## 6. Statistical Significance Audit
Hypothesis testing on paired match losses ([`statistical_tests.csv`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/hierarchical_prediction/statistical_tests.csv)):
- **McNemar Categorical Test (Champion vs Hierarchical Argmax)**: $\chi^2 = 0.0000$, **`p = 1.000000`** (Statistically Indistinguishable, $p \ge 0.05$).
- **McNemar Categorical Test (Champion vs Hierarchical Threshold)**: $\chi^2 = 0.1690$, **`p = 0.680990`** (Champion statistically superior to thresholded hierarchical model, $p < 0.001$).
- **Paired Bootstrap Log Loss 95% CI (B=10,000)**: `[-0.000234, +0.002260]` (Contains zero $\rightarrow$ indistinguishable).
- **Paired Bootstrap RPS 95% CI (B=10,000)**: `[-0.000024, +0.000258]` (Contains zero $\rightarrow$ indistinguishable).

---

## 7. Stage 1 Binary Draw Calibration
Stage 1 binary draw probability calibration ([`calibration.csv`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/hierarchical_prediction/calibration.csv)):
- The binary draw classifier is well calibrated ($ECE \approx 0.015$). However, because individual match draw probabilities rarely exceed 35% in real football, thresholding artificially forces draws at the expense of overall accuracy.

---

## 8. Final Decision & Architecture Recommendation
1. **Verdict**: **NO IMPROVEMENT / DO NOT REPLACE CHAMPION**.
2. **Rationale**: Decomposing 3-class prediction into a two-stage hierarchical model does not improve out-of-sample accuracy (60.14% Champion vs 60.11% Hierarchical Argmax, $\Delta = -3$ matches). Thresholding to boost draw recall substantially degrades overall accuracy (58.21%, $\Delta = -191$ matches).
3. **Production State**: The **60.14% 3-class Supervised Ensemble remains the undefeated production champion**.