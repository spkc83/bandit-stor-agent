"""Doubly robust contextual-bandit objective."""

from __future__ import annotations

import torch

from bandit_stor.data.schema import gather_logged_values
from bandit_stor.objectives.ips import importance_weights


def doubly_robust_values(
    policy_probs: torch.Tensor,
    q_hat: torch.Tensor,
    logged_action_index: torch.Tensor,
    reward: torch.Tensor,
    pscore: torch.Tensor,
    *,
    clip: float | None = None,
) -> torch.Tensor:
    """Return per-row DR values.

    Args:
        policy_probs: Target policy probabilities `[B, K]`.
        q_hat: Reward model predictions `[B, K]`.
        logged_action_index: Logged candidate index `[B]`.
        reward: Observed reward for logged action `[B]`.
        pscore: Logged propensity for logged action `[B]`.
        clip: Optional training-time cap for importance weights.

    Returns:
        Tensor `[B]` with direct-method plus residual correction terms.
    """
    direct = (policy_probs * q_hat).sum(dim=-1)
    pi_logged = gather_logged_values(policy_probs, logged_action_index)
    q_logged = gather_logged_values(q_hat, logged_action_index)
    weights = importance_weights(pi_logged, pscore, clip=clip)
    return direct + weights * (reward - q_logged)


def doubly_robust_value(*args, **kwargs) -> torch.Tensor:
    """Mean DR policy value scalar."""
    return doubly_robust_values(*args, **kwargs).mean()
