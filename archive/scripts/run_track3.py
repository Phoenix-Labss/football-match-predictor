"""Track 3 entry point: FIFA-style match engine + World Cup Monte Carlo simulator."""

import argparse
from pathlib import Path

from src.experiments.track3 import run_track3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track 3: FIFA-style match engine + World Cup Monte Carlo simulator"
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="path to the YAML config (default: config/default.yaml)",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=None,
        help="override the number of Monte Carlo tournament simulations",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    run_track3(
        config_path=args.config,
        project_root=project_root,
        n_simulations=args.n_simulations,
    )


if __name__ == "__main__":
    main()