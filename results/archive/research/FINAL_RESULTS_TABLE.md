# Dynamic Oracle — Final Standardized Results Table

**Dataset:** Kaggle International Football Results (`results.csv`, 49,520 matches)  
**Evaluation:** Strict 4-Fold Rolling-Origin Temporal Split (9,904 Out-of-Sample Test Matches)  
**Learner:** Histogram Gradient Boosted Decision Tree (GBDT, 300 estimators, depth 3, LR 0.05)  
**Verification:** 100% Paired Apples-to-Apples Evaluation  

---

## 1. Global Predictive Performance (9,904 Out-of-Sample Test Matches)

| Metric | Classic Elo ($K=24$) | Fixed 5% Bounded Elo | Adaptive Elo (Default $W=7$) | Adaptive Elo (Tuned $W=3$) | Best Performing |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Classification Accuracy** | 59.81% | 59.76% | 59.47% | **59.94%** | **Adaptive (Tuned)** |
| **Normalized RPS (Proper Loss)** | **0.1710** | **0.1710** | 0.1735 | **0.1710** | **Tied (All Models)** |
| **Unnormalized RPS** | **0.3420** | **0.3420** | 0.3470 | **0.3419** | **Adaptive (Tuned)** |
| **Multiclass Log Loss** | 0.8750 | 0.8747 | 0.8830 | **0.8743** | **Adaptive (Tuned)** |
| **Multiclass Brier Score** | 0.5150 | 0.5149 | 0.5200 | **0.5149** | **Fixed 5% / Adaptive** |
| **Expected Calibration Error (ECE)** | 0.0118 | 0.0127 | **0.0095** | 0.0102 | **Adaptive (Default/Tuned)** |

---

## 2. Major Upset Shock Response ($P(\text{Favorite}) \ge 0.80$, 767 Historical Matches)

| Property / Metric | Classic Elo ($K=24$) | Fixed 5% Bounded Elo | Adaptive Elo (Tuned) | Percentage Advantage |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Rating Drop on Upset** | 23.72 pts | 19.91 pts | **19.25 pts** | **-18.85% vs Classic** |
| **Median Rating Drop on Upset** | 20.74 pts | 20.00 pts | **19.73 pts** | **-4.87% vs Classic** |
| **Maximum Rating Drop (Blowout)** | 61.35 pts | 20.00 pts | **30.00 pts** | **-51.10% vs Classic** |
| **% of Upsets with Smaller Drop** | Reference | 50.6% | **68.58%** | **95% CI: [65.2%, 71.8%]** |

---

## 3. Synthetic Scenario Control Dynamics (Team A 2000 vs Team B 1600)

| Scenario | Classic Elo | Fixed 5% Bounded | Adaptive Elo | Behavior Characterization |
| :--- | :---: | :---: | :---: | :--- |
| **Scenario A (10 Expected Wins)** | Max Drop: 0.0 | Max Drop: 0.0 | Max Drop: 0.0 | Full baseline stability |
| **Scenario B (1 Isolated Shock Loss)** | Max Drop: **9.6 pts** | Max Drop: 7.5 pts | Max Drop: **3.9 pts** | **59.89% shock dampening** |
| **Scenario C (3 Slump Defeats)** | Max Drop: 57.8 pts | Max Drop: 53.6 pts | Max Drop: **43.0 pts** | Smooth progressive adaptation |
| **Scenario D (5 Collapse Defeats)** | Max Drop: 102.6 pts | Max Drop: 99.2 pts | Max Drop: **81.9 pts** | Full structural descent ($1950.5$ rating) |

---

## 4. In-Tournament Dynamic Simulation (2018 WC, Euro 2020, 2022 WC)

| Metric | STATIC Elo | CLASSIC_DYNAMIC ($K=24$) | ADAPTIVE_DYNAMIC |
| :--- | :---: | :---: | :---: |
| **2018 World Cup Log Loss** | 0.9815 | **0.9695** | 0.9760 |
| **UEFA Euro 2020 Log Loss** | 0.8728 | **0.8625** | 0.8705 |
| **2022 World Cup Log Loss** | 1.0254 | 1.0385 | **1.0261** |
| **Mean In-Tournament Movement** | 0.0 pts | 9.55 pts | 7.02 pts |
| **Major Swings ($\ge 15\text{ pts}$)** | 0 swings | **59 swings** | **5 swings (91.5% reduction)** |

---

## 5. Statistical Hypothesis Testing (Diebold-Mariano Paired Test)

| Model Comparison | Loss Metric | Mean Difference ($\bar{d}$) | DM-HLN Statistic | $p$-Value | Conclusion |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Classic vs Adaptive** | Normalized RPS | $+0.000016$ | $+0.0893$ | $0.9289$ | No significant difference ($p \ge 0.05$) |
| **Classic vs Adaptive** | Multiclass Log Loss | $+0.000717$ | $+0.8960$ | $0.3703$ | No significant difference ($p \ge 0.05$) |
| **Fixed 5% vs Adaptive** | Normalized RPS | $+0.000076$ | $+0.4648$ | $0.6421$ | No significant difference ($p \ge 0.05$) |
| **Fixed 5% vs Adaptive** | Multiclass Log Loss | $+0.000375$ | $+0.5320$ | $0.5947$ | No significant difference ($p \ge 0.05$) |
