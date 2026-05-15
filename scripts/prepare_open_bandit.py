"""Validate Open Bandit prepared data for Bandit-STOR.

This helper intentionally does not generate synthetic data. It checks whether local
Open Bandit-style files are present and loadable by the adapter.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bandit_stor.data.open_bandit import load_open_bandit  # noqa: E402
from bandit_stor.utils import load_yaml  # noqa: E402


def main() -> None:
    cfg = load_yaml(ROOT / "configs/data/open_bandit.yaml")
    dataset, splits = load_open_bandit(cfg)
    print(
        "Open Bandit data ready: "
        f"n={len(dataset)} train={len(splits.train)} valid={len(splits.valid)} test={len(splits.test)} "
        f"context_dim={dataset.context_dim} action_dim={dataset.action_dim} n_actions={dataset.n_actions}"
    )


if __name__ == "__main__":
    main()
