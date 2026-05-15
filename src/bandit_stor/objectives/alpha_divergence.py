"""Alpha-divergence support constraint."""

from __future__ import annotations

import torch


def alpha_divergence(
    policy_probs: torch.Tensor,
    behavior_probs: torch.Tensor,
    *,
    alpha: float = 1.5,
    eps: float = 1e-8,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute `D_alpha(policy || behavior)` per row.

    Args:
        policy_probs: Tensor `[B, K]`.
        behavior_probs: Tensor `[B, K]`.
        alpha: Divergence order, must be positive and not one.
        eps: Clamp value for numerical stability.
        mask: Optional bool tensor `[B, K]`.

    Returns:
        Tensor `[B]` containing alpha divergence values.
    """
    if alpha <= 0 or abs(alpha - 1.0) < eps:
        raise ValueError("alpha must be positive and not equal to 1")
    p = policy_probs.clamp_min(eps)
    q = behavior_probs.clamp_min(eps)
    if mask is not None:
        p = p.masked_fill(~mask, eps)
        q = q.masked_fill(~mask, eps)
    term = (p.pow(alpha) * q.pow(1.0 - alpha)).sum(dim=-1)
    return (term - 1.0) / (alpha * (alpha - 1.0))


def unsupported_action_mass(policy_probs: torch.Tensor, behavior_probs: torch.Tensor, *, mu_min: float = 1e-4) -> torch.Tensor:
    """Return per-row target-policy mass where behavior support is below `mu_min`.

    Shapes: both tensors `[B, K]`; return `[B]`.
    """
    return policy_probs.masked_fill(behavior_probs >= mu_min, 0.0).sum(dim=-1)
