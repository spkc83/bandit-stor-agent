"""Tsallis entropy objectives."""

from __future__ import annotations

import torch


def tsallis_q2_entropy(policy_probs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute Tsallis q=2 entropy per row.

    Args:
        policy_probs: Tensor `[B, K]`.
        mask: Optional bool tensor `[B, K]`.

    Returns:
        Tensor `[B]` with `1 - sum_a p(a)^2` over eligible candidates.
    """
    probs = policy_probs if mask is None else policy_probs.masked_fill(~mask, 0.0)
    return 1.0 - probs.square().sum(dim=-1)
