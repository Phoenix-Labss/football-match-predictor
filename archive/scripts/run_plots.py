"""Entry point: generate reliability diagrams for Track 1.

Usage (from the soccer-prediction directory):
    python run_plots.py
"""

from src.evaluation.plots import plot_reliability_track1

if __name__ == "__main__":
    plot_reliability_track1()