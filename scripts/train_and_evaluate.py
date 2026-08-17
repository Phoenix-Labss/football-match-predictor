"""Canonical Training and Evaluation Entry Point for Dynamic Oracle.

Usage:
    # Run full champion training and evaluation on the 9,904-match test set:
    python scripts/train_and_evaluate.py

    # View champion metadata:
    python scripts/train_and_evaluate.py --info
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def show_champion_info():
    config_path = PROJECT_ROOT / "results" / "champion" / "champion_config.json"
    results_path = PROJECT_ROOT / "results" / "champion" / "final_test_results.json"

    print("\n" + "=" * 70)
    print(" DYNAMIC ORACLE — CURRENT VERIFIED CHAMPION")
    print("=" * 70)

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f" Champion Name       : {cfg.get('champion_name')}")
        print(f" Verified Accuracy   : {cfg.get('champion_accuracy', 0) * 100:.2f}% ({cfg.get('champion_correct')} / {cfg.get('test_set_size')})")
        print(f" Feature Matrix      : {cfg.get('feature_pipeline', {}).get('n_features')} features ({cfg.get('feature_pipeline', {}).get('module')})")
        print(f" Base Models         : {list(cfg.get('base_models', {}).keys())}")
        print(f" Ensemble Method     : {cfg.get('ensemble', {}).get('method')}")
        print(f" Test Set Size       : {cfg.get('test_set_size')} untouched matches")

    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            res = json.load(f)
        champ = res.get("champion_optimized_model", {})
        print("\n Verified Test Set Metrics:")
        print(f"   Accuracy          : {champ.get('accuracy', 0) * 100:.2f}%")
        print(f"   Log Loss          : {champ.get('log_loss', 0):.4f}")
        print(f"   Normalized RPS    : {champ.get('normalized_rps', 0):.4f}")
        print(f"   Brier Score       : {champ.get('brier_score', 0):.4f}")
        print(f"   ECE               : {champ.get('ece', 0):.4f}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Dynamic Oracle Champion Training & Evaluation")
    parser.add_argument("--info", action="store_true", help="Display verified champion metadata and exit")
    args = parser.parse_args()

    if args.info:
        show_champion_info()
        return

    print("[Dynamic Oracle] Launching Champion Training & Evaluation Pipeline...")
    from src.optimization.pipeline import run_full_optimization_suite
    run_full_optimization_suite(project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()
