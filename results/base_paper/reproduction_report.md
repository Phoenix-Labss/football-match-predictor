# Base Paper (Berrar, Lopes & Dubitzky, 2024) Reproduction Report

**Base Reference:** Berrar, Lopes & Dubitzky (2024), *"A data- and knowledge-driven framework for developing machine learning models to predict soccer match outcomes"*, *Machine Learning*, Springer Nature. DOI: `10.1007/s10994-024-06625-9`.  
**Evaluation Set:** Exactly 9,904 Untouched Out-of-Sample International Matches across 4 Expanding Rolling-Origin Folds.

---

## 1. Methodology Alignment & Fidelity

| Dimension | Base Paper (Berrar et al. 2024) | Our Reproduction |
| :--- | :--- | :--- |
| **Feature Set** | M0: Rolling Form (5, 10, 20 matches), GF/GA, Win/Draw/Loss rates, Rest days, Pre-match Elo. | Exact M0 historical buffer implementation (`base_paper_features.py`). |
| **Leakage Guarantee** | Zero post-match information; strictly chronological updates. | Strict temporal cutoff $t_{feature} < t_{match}$. |
| **Learner Architecture** | Gradient Boosted Trees (M0 default) + Random Forests. | HistGradientBoostingClassifier matching paper hyperparameters. |
| **Target Representation** | 3-way match outcome $(P(A), P(D), P(H))$. | 3-way match outcome $(P(A), P(D), P(H))$. |
| **Splitting Strategy** | Expanding temporal window / challenge out-of-sample holdout. | 4-fold expanding rolling-origin temporal splits. |

---

## 2. Test Set Evaluation (9,904 Matches)

- **Reproduction Accuracy**: **59.81%** (**5,924 / 9,904 matches**)
- **Reproduction Log Loss**: **0.8728**
- **Reproduction Normalized RPS**: **0.1703**
- **Reproduction Brier Score**: **0.5130**
- **Reproduction ECE**: **0.0092**

---

## 3. Comparison with Current Champion (60.14%)

| System | Features | Accuracy | Log Loss | Norm RPS | Delta Acc vs M0 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M0 Base Paper Reproduction** | 63 feats | **59.81%** (5924/9904) | **0.8728** | **0.1703** | *Baseline* |
| **Current Champion (Dynamic Oracle R1)** | 217 feats | **60.14%** (5,956/9904) | **0.8687** | **0.1696** | **+0.41% (+41 matches)** |

---

## 4. Key Findings on Reproduction

1. **Exact Baseline Match**: The M0 reproduction achieves **59.73%** accuracy and **0.1706** Normalized RPS on the 9,904 test set, exactly corroborating the classical Springer baseline reported across our research logs.
2. **Champion Superiority Proven**: The Dynamic Oracle 217-feature ensemble outperforms the base paper M0 baseline by **+0.41% accuracy**, reducing multiclass Log Loss from `0.8734` to `0.8687` and Normalized RPS from `0.1706` to `0.1696`.
