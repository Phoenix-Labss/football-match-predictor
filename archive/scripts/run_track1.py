"""Entry point: run the Track 1 experiment from the project root.

Usage (from the soccer-prediction directory):
    python run_track1.py
"""

from src.experiments.track1 import run_track1

if __name__ == "__main__":
    results = run_track1(config_path="config/default.yaml")
    print("\n=== Track 1 Results ===")
    cols = ["variant", "log_loss", "rps", "ece", "accuracy", "brier"]
    print(results[cols].to_string(index=False))