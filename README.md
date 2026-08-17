# Dynamic Oracle — Soccer Match Outcome Predictor

Dynamic Oracle is a high-performance machine learning system and interactive simulation suite for predicting 3-way international soccer match outcomes (**Home Win / Draw / Away Win**).

---

## Quick Reference (FAQ for Developers & Agents)

### 1. What is Dynamic Oracle?
A multi-model machine learning predictor combining confidence-controlled adaptive Elo ratings, multi-scale team form dynamics, pre-match Dixon-Coles Poisson goal intensities, head-to-head records, and player-level FIFA chemistry.

### 2. What data does it use?
- **International Match Dataset**: 49,520 real historical matches from 1872 to present (`data/raw/results.csv`).
- **FIFA Multi-Year Squads**: Player ratings and positional data across FIFA editions 2015–2022 + 2026 World Cup rosters (`data/raw/fifa/`).

### 3. What is the prediction target?
Multiclass 3-way classification: **Home Win (2)**, **Draw (1)**, **Away Win (0)**, outputting calibrated probability distributions $P(\text{Away}), P(\text{Draw}), P(\text{Home})$.

### 4. What is the current champion?
**Dynamic Oracle Champion Ensemble (Round 1)**: A 5-model LogLoss-optimized convex ensemble combining **LightGBM (~30.4%)**, **XGBoost (~25.0%)**, **HistGBDT (~20.9%)**, **CatBoost (~17.2%)**, and **Dixon-Coles Bivariate Poisson (~6.5%)**.

### 5. What accuracy does it achieve?
- **Accuracy**: **60.14%** (**5,956 / 9,904** matches correct) on the untouched out-of-sample test set across 4 temporal rolling-origin folds.
- **Log Loss**: `0.8687` | **Normalized RPS**: `0.1696` | **ECE**: `0.0143`.

### 6. Where is the feature pipeline?
[`src/optimization/features.py`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/src/optimization/features.py) (`build_advanced_feature_matrix`, 217 engineered features).

### 7. Where is the model?
- **Model Factory**: [`src/optimization/models.py`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/src/optimization/models.py) (`build_model_family`).
- **Ensemble Optimization & Blending**: [`src/optimization/ensemble.py`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/src/optimization/ensemble.py).
- **Service Engine**: [`src/service/oracle.py`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/src/service/oracle.py) (`DynamicOracle`).

### 8. How do I train & evaluate the champion?
```bash
# Run the full pipeline on 4 rolling temporal folds and evaluate the 9,904 test matches:
python scripts/train_and_evaluate.py
# or
python -m src.optimization.pipeline
```

### 9. How do I view champion metadata?
```bash
python scripts/train_and_evaluate.py --info
```

### 10. How do I make interactive predictions or run the web UI?
```bash
# Start the FastAPI web service:
uvicorn src.service.server:app --reload --port 8000
# Then open frontend/index.html in your browser
```

### 11. Where are historical experiments and past track results?
- **Detailed History**: [`docs/EXPERIMENT_HISTORY.md`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/docs/EXPERIMENT_HISTORY.md)
- **Archived Code**: `archive/` (`experiments/`, `research/`, `optimization_r2/`, `scripts/`)
- **Archived Results**: `results/archive/` (`tracks/`, `accuracy_optimization_r2/`, `figures/`, etc.)

### 12. Which files should I NOT modify casually?
- `data/raw/results.csv` — Untouched ground-truth dataset.
- `src/data/split.py` — Strict rolling-origin temporal splits preventing data leakage.
- `results/champion/` — Ground-truth metrics and configuration for the 60.14% champion.
- `src/optimization/features.py` & `src/optimization/pipeline.py` — Core champion code.

### 13. What is the next research direction?
**Hierarchical 2-Stage Classifier**: Stage 1 decomposes matches into Draw vs Decisive outcome (specifically targeting the ~66% error share from misclassified draws); Stage 2 classifies Home vs Away conditional on a decisive match.

---

## Directory Structure

```
soccer-prediction/
│
├── README.md                          # Quick-start & developer reference (this file)
├── requirements.txt                   # Production dependencies
│
├── config/
│   └── default.yaml                   # Global configuration & hyperparameters
│
├── data/
│   └── raw/                           # Raw immutable datasets
│       ├── results.csv                # 49,520 international match records
│       └── fifa/                      # Multi-year FIFA player ratings (2015-2022, 2026)
│
├── src/
│   ├── data/                          # Match loaders & temporal splitters
│   ├── features/                      # Elo strength tracking & M0 baseline features
│   ├── models/                        # Estimator definitions
│   ├── optimization/                  # Champion feature pipeline, models, and ensemble
│   ├── evaluation/                    # Multiclass evaluation metrics (RPS, LogLoss, ECE)
│   ├── simulation/                    # Dixon-Coles match engine & tournament simulator
│   └── service/                       # DynamicOracle backend & FastAPI REST server
│
├── scripts/
│   └── train_and_evaluate.py          # Unified CLI entry point for training & evaluation
│
├── frontend/                          # Football-themed interactive web interface
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── results/
│   ├── champion/                      # Current 60.14% champion verified configuration & metrics
│   └── archive/                       # Historical experiment results & figures
│
├── archive/                           # Archived experimental code & one-off research scripts
│
└── docs/
    ├── PROJECT_ARCHITECTURE.md        # Component architecture & data-flow diagram
    ├── EXPERIMENT_HISTORY.md          # Log of all completed research tracks & milestones
    └── CLEANUP_REPORT.md              # Detailed repository consolidation audit
```
