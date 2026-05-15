"""Compatibility CLI for reward-lift Optuna actor tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bandit_stor.training.hpo import run_optuna_hpo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Bandit-STOR actor with Optuna")
    parser.add_argument("--data", default="open_bandit")
    parser.add_argument("--behavior_policy", default=None, help="Documentary override; config controls loading")
    parser.add_argument("--mode", default=None, help="reward_aggressive or deployment_conservative")
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--validation_size", type=int, default=None)
    parser.add_argument("--hpo-config", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    # Mode/validation_size are currently configured in YAML; keep CLI for requested run shape.
    run_dir = run_optuna_hpo(
        data=args.data,
        hpo_config_path=args.hpo_config,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        mode=args.mode,
        validation_size=args.validation_size,
    )
    print(f"Bandit-STOR actor tuning complete: {run_dir}")


if __name__ == "__main__":
    main()
