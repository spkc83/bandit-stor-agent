"""IPS/SNIPS estimators and weighting diagnostics."""

from __future__ import annotations

import torch


def importance_weights(
    logged_policy_probs: torch.Tensor,
    pscore: torch.Tensor,
    *,
    eps: float = 1e-8,
    clip: float | None = None,
) -> torch.Tensor:
    """Return importance weights `[B] = pi(a_i|x_i) / mu(a_i|x_i)`.

    `pscore` is the logged propensity source of truth for the logged action.
    """
    weights = logged_policy_probs / pscore.clamp_min(eps)
    if clip is not None:
        weights = weights.clamp_max(float(clip))
    return weights


def ips_value(weights: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
    """IPS estimate scalar from weights `[B]` and rewards `[B]`."""
    return (weights * reward).mean()


def snips_value(weights: torch.Tensor, reward: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Self-normalized IPS estimate scalar."""
    return (weights * reward).sum() / weights.sum().clamp_min(eps)


def effective_sample_size(weights: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """ESS scalar `(sum w)^2 / sum w^2` for weights `[B]`."""
    return weights.sum().square() / weights.square().sum().clamp_min(eps)
