"""Run Optuna hyperparameter search for Bandit-STOR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bandit_stor.training.hpo import run_optuna_hpo  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize Bandit-STOR hyperparameters with Optuna")
    parser.add_argument("overrides", nargs="*", help="Hydra-style overrides, e.g. data=open_bandit")
    parser.add_argument("--hpo-config", default=None, help="Path to Optuna HPO YAML config")
    parser.add_argument("--output-dir", default=None, help="Directory for HPO artifacts")
    parser.add_argument("--n-trials", type=int, default=None, help="Override number of trials")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data = "open_bandit"
    for override in args.overrides:
        if override.startswith("data="):
            data = override.split("=", 1)[1]
    run_dir = run_optuna_hpo(
        data=data,
        hpo_config_path=args.hpo_config,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
    )
    print(f"Bandit-STOR Optuna HPO complete: {run_dir}")


if __name__ == "__main__":
    main()
