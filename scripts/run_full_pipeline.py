"""Run the Bandit-STOR MVP full pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bandit_stor.training.full_pipeline import run_full_pipeline  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate Bandit-STOR")
    parser.add_argument("overrides", nargs="*", help="Hydra-style overrides, e.g. data=tiny_fixture")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data = "open_bandit"
    for override in args.overrides:
        if override.startswith("data="):
            data = override.split("=", 1)[1]
    run_dir = run_full_pipeline(data=data, output_dir=args.output_dir)
    print(f"Bandit-STOR pipeline complete: {run_dir}")


if __name__ == "__main__":
    main()
