"""Run phase-1 reward-model calibration sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bandit_stor.training.reward_hpo import run_reward_hpo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Bandit-STOR reward model calibration")
    parser.add_argument("overrides", nargs="*", help="Hydra-style overrides, e.g. data=open_bandit")
    parser.add_argument("--data", default=None)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--max-train-rows", type=int, default=100000, help="Use 0 or negative for all training rows")
    parser.add_argument("--max-valid-rows", type=int, default=50000, help="Use 0 or negative for all validation rows")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    data = args.data or "open_bandit"
    for override in args.overrides:
        if override.startswith("data="):
            data = override.split("=", 1)[1]
    max_train_rows = None if args.max_train_rows is not None and args.max_train_rows <= 0 else args.max_train_rows
    max_valid_rows = None if args.max_valid_rows is not None and args.max_valid_rows <= 0 else args.max_valid_rows
    run_dir = run_reward_hpo(
        data=data,
        n_trials=args.n_trials,
        max_train_rows=max_train_rows,
        max_valid_rows=max_valid_rows,
        output_dir=args.output_dir,
    )
    print(f"Bandit-STOR reward-model tuning complete: {run_dir}")


if __name__ == "__main__":
    main()
