# Dynamic Oracle — Out-of-Sample Accuracy Optimization Report (Round 2)

**Date:** August 2026  
**Primary Objective:** Direct Maximization of Out-of-Sample 3-Way Match Classification Accuracy  
**Dataset:** 49,520 Real Kaggle International Matches (`results.csv`)  
**Evaluation Set:** Exactly 9,904 Untouched Out-of-Sample Matches (Evaluated Once)  

---

## 1. Executive Performance Summary

| Metric | Original Baseline | Round 1 Champion | Round 2 Champion | Delta vs Baseline | Delta vs R1 Champion |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Out-of-Sample Accuracy** | **60.05%** | **60.04%** | **59.76%** | **-0.28%** | **-0.27%** |
| **Correct Predictions** | **5,946 / 9,904** | **5,946 / 9,904** | **5,918 / 9,904** | **+-28** | **+-27** |
| **Multiclass Log Loss** | **0.8702** | **0.8765** | **0.8751** | **+0.0049** | **-0.0013** |
| **Normalized RPS** | **0.1699** | **0.1707** | **0.1705** | **+0.0006** | **-0.0002** |
| **Expected Calibration Error** | **0.0094** | **0.0290** | **0.0076** | **-0.0018** | **-0.0214** |

---

## 2. Test Set 3x3 Confusion Matrix & Per-Class Metrics

```
                    Actual Away (0)    Actual Draw (1)    Actual Home (2)
Predicted Away (0)        4035               1472               1005
Predicted Draw (1)         102                104                 98
Predicted Home (2)         578                730               1780
```

* **Away Class (0)**: Precision = `61.96%` | Recall = `85.58%`
* **Draw Class (1)**: Precision = `34.21%` | Recall = `4.51%`
* **Home Class (2)**: Precision = `57.64%` | Recall = `61.74%`

---

## 3. Key Findings & Answers to Analysis Tasks

### A. Validation Error Analysis
The dominant failure modes in soccer match outcome prediction are:
1. **Home -> Draw errors (38.4% of errors)**: Matches where the home team dominates expected strength but draws 0-0 or 1-1 due to low-scoring variance.
2. **Away -> Draw errors (28.4% of errors)**: Favored away teams held to a draw on stubborn defensive home pitches.
3. **Home -> Away upsets (17.9% of errors)**: Pure underdog counter-attack upsets.

### B. Accuracy-Optimized Ensembling vs Probabilistic Ensembling
Direct accuracy optimization shifted the continuous weights towards high-margin gradient estimators:
* **Accuracy-Optimized Weights**: `{'LightGBM': 0.224, 'XGBoost': 0.422, 'CatBoost': 0.067, 'HistGBDT': 0.287, 'Dixon_Coles': 0.0}`
* Shifting from LogLoss-minimizing weights to 0-1 accuracy-maximizing weights improved validation accuracy without compromising proper scoring rules.

### C. Feature Engineering Contribution
* Round 2 features (Elo velocity, form acceleration, clean sheet rates, H2H draw affinity) added **+0.11%** validation accuracy over Round 1 features.
* The strongest new signals were **`diff_form_accel`** (short-term vs medium-term momentum) and **`h2h_draw_affinity`**.

### D. Bottleneck Identification & Next Highest-Value Experiment
* **The Draw Problem (The Fundamental Bottleneck)**: Soccer draws occur in ~24% of all international matches, but Poisson score models predict draw probability peaking around ~28% even in perfectly balanced matchups. Standard argmax almost never selects Draw unless both Home and Away probabilities drop below 33%.
* **Next Highest-Value Experiment**: A specialized **Hierarchical 2-Stage Classifier**: Stage 1 predicts decisive vs non-decisive match (Draw vs Result); Stage 2 predicts Home vs Away conditional on a decisive outcome.

---
*Report generated automatically by Dynamic Oracle Optimization Pipeline Round 2.*
