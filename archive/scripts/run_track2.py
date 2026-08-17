"""Entry point: run the Track 2 experiment (player-aware features).

Usage (from the soccer-prediction directory):
    python run_track2.py
"""

from src.experiments.track2 import run_track2

if __name__ == "__main__":
    results = run_track2(config_path="config/default.yaml")
    print("\n=== Track 2 Results ===")
    cols = ["variant", "log_loss", "rps", "ece", "accuracy", "brier"]
    print(results[cols].to_string(index=False))