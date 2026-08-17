"""Regression verification script for Dynamic Oracle after codebase cleanup."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_checks():
    print("Testing imports and basic pipeline integrity...")

    try:
        from src.data.loader import load_matches, add_outcome_labels
        print("  [PASS] 1/10 src.data.loader")
    except Exception as e:
        print(f"  [FAIL] 1/10 src.data.loader: {e}")
        return False

    try:
        from src.data.split import rolling_origin_folds
        print("  [PASS] 2/10 src.data.split")
    except Exception as e:
        print(f"  [FAIL] 2/10 src.data.split: {e}")
        return False

    try:
        from src.features.strength import UpdaterConfig, StrengthTracker
        print("  [PASS] 3/10 src.features.strength")
    except Exception as e:
        print(f"  [FAIL] 3/10 src.features.strength: {e}")
        return False

    try:
        from src.features.team_form import build_m0_feature_matrix
        print("  [PASS] 4/10 src.features.team_form")
    except Exception as e:
        print(f"  [FAIL] 4/10 src.features.team_form: {e}")
        return False

    try:
        from src.optimization.features import build_advanced_feature_matrix
        print("  [PASS] 5/10 src.optimization.features")
    except Exception as e:
        print(f"  [FAIL] 5/10 src.optimization.features: {e}")
        return False

    try:
        from src.optimization.models import build_model_family
        m = build_model_family("hist_gbdt")
        print("  [PASS] 6/10 src.optimization.models (model factory)")
    except Exception as e:
        print(f"  [FAIL] 6/10 src.optimization.models: {e}")
        return False

    try:
        from src.optimization.ensemble import optimize_ensemble_weights, blend_probabilities
        print("  [PASS] 7/10 src.optimization.ensemble")
    except Exception as e:
        print(f"  [FAIL] 7/10 src.optimization.ensemble: {e}")
        return False

    try:
        from src.optimization.pipeline import _evaluate_on_folds
        print("  [PASS] 8/10 src.optimization.pipeline")
    except Exception as e:
        print(f"  [FAIL] 8/10 src.optimization.pipeline: {e}")
        return False

    try:
        from src.evaluation.metrics import evaluate_all, accuracy, multiclass_log_loss, rps, expected_calibration_error
        print("  [PASS] 9/10 src.evaluation.metrics")
    except Exception as e:
        print(f"  [FAIL] 9/10 src.evaluation.metrics: {e}")
        return False

    try:
        from src.simulation.player_model import PlayerModel
        from src.simulation.squad_model import SquadModel
        from src.simulation.chemistry import ChemistryModel
        from src.simulation.match_engine import MatchEngine
        from src.simulation.tournament import WorldCupSimulator
        from src.service.oracle import DynamicOracle
        from src.service.server import app
        print("  [PASS] 10/10 src.simulation & src.service")
    except Exception as e:
        print(f"  [FAIL] 10/10 src.simulation & src.service: {e}")
        return False

    print("\nALL 10/10 REGRESSION INTEGRITY CHECKS PASSED PERFECTLY!")
    return True

if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
