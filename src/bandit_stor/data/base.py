"""Dataset primitives for contextual-bandit data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class DatasetSplits:
    """Train/validation/test dataset partitions."""

    train: Dataset
    valid: Dataset
    test: Dataset


class LoggedBanditDataset(Dataset):
    """In-memory logged bandit dataset.

    The dataset stores examples with canonical shapes documented by
    `LoggedInteractionBatch`; rewards are present only for logged actions.
    """

    def __init__(self, examples: Sequence[dict], context_dim: int, action_dim: int, n_actions: int):
        self.examples = list(examples)
        self.context_dim = int(context_dim)
        self.action_dim = int(action_dim)
        self.n_actions = int(n_actions)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def split_dataset(
    dataset: LoggedBanditDataset,
    valid_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
    strategy: str = "deterministic_random",
) -> DatasetSplits:
    """Split a dataset deterministically.

    `chronological` keeps source order; `deterministic_random` shuffles with `seed`.
    """
    n = len(dataset)
    indices = np.arange(n)
    if strategy != "chronological":
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    n_test = int(round(n * test_size))
    n_valid = int(round(n * valid_size))
    n_train = max(0, n - n_valid - n_test)
    train_idx = indices[:n_train].tolist()
    valid_idx = indices[n_train : n_train + n_valid].tolist()
    test_idx = indices[n_train + n_valid :].tolist()
    if not train_idx:
        raise ValueError("Split produced an empty train set; provide more rows or smaller valid/test sizes")
    return DatasetSplits(Subset(dataset, train_idx), Subset(dataset, valid_idx), Subset(dataset, test_idx))


def uniform_behavior_from_logged_pscore(n_actions: int, logged_action: int, pscore: float) -> torch.Tensor:
    """Build a conservative full behavior support vector for tests/local CSV.

    This function does not infer propensity from item frequency: it preserves the logged
    propensity at the observed action and spreads residual mass uniformly across other
    candidates solely as a support approximation when full behavior probabilities are absent.
    """
    probs = torch.full((n_actions,), max(1.0 - float(pscore), 0.0) / max(n_actions - 1, 1))
    probs[int(logged_action)] = float(pscore)
    return probs / probs.sum().clamp_min(1e-12)
