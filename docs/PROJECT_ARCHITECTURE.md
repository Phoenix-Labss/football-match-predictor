# Project Architecture

## System Overview

Dynamic Oracle is a machine learning system for predicting international football (soccer) match outcomes as 3-way classification: Home Win, Draw, Away Win.

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                 │
│                                                                 │
│  data/raw/results.csv ──► src/data/loader.py                   │
│  (49,520 matches)         load_matches()                        │
│                           add_outcome_labels()                  │
│                                  │                              │
│                                  ▼                              │
│                        src/data/split.py                        │
│                        rolling_origin_folds()                   │
│                        (4 temporal folds)                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                           │
│                                                                 │
│  src/features/strength.py    ◄── Elo/strength rating tracker   │
│    StrengthTracker                (confidence-controlled)       │
│    UpdaterConfig                                                │
│         │                                                       │
│         ▼                                                       │
│  src/optimization/features.py ◄── Champion feature pipeline    │
│    build_advanced_feature_matrix()                              │
│    (217 features: Elo, form, Dixon-Coles probs, H2H, venue)   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING                             │
│                                                                 │
│  src/optimization/models.py  ◄── Model factory                 │
│    build_model_family()          LightGBM, XGBoost,            │
│                                  CatBoost, HistGBDT            │
│         │                                                       │
│         ▼                                                       │
│  src/optimization/ensemble.py ◄── Ensemble optimization        │
│    optimize_ensemble_weights()    (SLSQP on log loss)          │
│    blend_probabilities()                                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EVALUATION                                 │
│                                                                 │
│  src/evaluation/metrics.py                                      │
│    accuracy(), multiclass_log_loss(), rps(),                   │
│    expected_calibration_error(), evaluate_all()                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREDICTION / SERVICE                          │
│                                                                 │
│  src/service/oracle.py     ◄── DynamicOracle class             │
│    (player-aware predictions, Dixon-Coles match engine)        │
│                                                                 │
│  src/service/server.py     ◄── FastAPI REST server             │
│    (serves frontend + prediction API)                          │
│                                                                 │
│  frontend/                 ◄── Web UI                          │
│    index.html, style.css, app.js                               │
└─────────────────────────────────────────────────────────────────┘
```

## Entry Points

| Task | Command | Module |
|:--|:--|:--|
| **Train & evaluate champion** | `python -m src.optimization.pipeline` | `src/optimization/pipeline.py` |
| **Start web server** | `uvicorn src.service.server:app` | `src/service/server.py` |
| **Quick evaluation script** | `python scripts/train_and_evaluate.py` | `scripts/train_and_evaluate.py` |

## Key Source Files (Champion Pipeline)

### Data Loading
- [`src/data/loader.py`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/src/data/loader.py) — Loads `results.csv`, adds outcome labels
- [`src/data/split.py`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/src/data/split.py) — Rolling-origin temporal folds (4 folds, no leakage)

### Feature Engineering
- [`src/features/strength.py`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/src/features/strength.py) — Elo/strength tracker with adaptive confidence-controlled updates
- [`src/features/team_form.py`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/src/features/team_form.py) — M0 baseline feature matrix
- [`src/optimization/features.py`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/src/optimization/features.py) — **Champion feature pipeline** (217 features)

### Models
- [`src/optimization/models.py`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/src/optimization/models.py) — Model family factory (LightGBM, XGBoost, CatBoost, HistGBDT)
- [`src/optimization/ensemble.py`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/src/optimization/ensemble.py) — Ensemble weight optimization

### Champion Pipeline
- [`src/optimization/pipeline.py`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/src/optimization/pipeline.py) — **End-to-end champion training, validation, and test evaluation**

### Evaluation
- [`src/evaluation/metrics.py`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/src/evaluation/metrics.py) — All scoring metrics

### Web Service
- [`src/service/oracle.py`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/src/service/oracle.py) — DynamicOracle prediction engine
- [`src/service/server.py`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/src/service/server.py) — FastAPI REST server

### Simulation (Track 3 / Frontend)
- [`src/simulation/`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/src/simulation/) — Match engine, player model, squad model, chemistry, tournament simulator

## Current Champion

- **Accuracy**: 60.14% (5,956 / 9,904)
- **Method**: 5-model ensemble (LightGBM + XGBoost + CatBoost + HistGBDT + Dixon-Coles)
- **Features**: 217 advanced features (Elo, form windows, Dixon-Coles probs, H2H, venue, competition)
- **Config**: [`results/champion/champion_config.json`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/champion/champion_config.json)
- **Test results**: [`results/champion/final_test_results.json`](file:///c:/Users/bisme/OneDrive/Desktop/fifa%20predictor/soccer-prediction/results/champion/final_test_results.json)
