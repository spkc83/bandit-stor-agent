"""Actor loss for Sparse Tsallis offline policy optimization."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from bandit_stor.objectives.alpha_divergence import alpha_divergence, unsupported_action_mass
from bandit_stor.objectives.doubly_robust import doubly_robust_value
from bandit_stor.objectives.tsallis import tsallis_q2_entropy


@dataclass(frozen=True)
class ActorLossOutput:
    """Actor objective output with scalar loss and logged components."""

    loss: torch.Tensor
    components: dict[str, float]


def sparse_tsallis_actor_loss(
    policy_probs: torch.Tensor,
    behavior_probs: torch.Tensor,
    q_hat: torch.Tensor,
    logged_action_index: torch.Tensor,
    reward: torch.Tensor,
    pscore: torch.Tensor,
    *,
    alpha: float = 1.5,
    beta_alpha: float = 0.1,
    lambda_tsallis: float = 0.01,
    lambda_support: float = 1.0,
    mu_min: float = 1e-4,
    importance_clip: float | None = 20.0,
    reward_value_scale: float = 1.0,
    mask: torch.Tensor | None = None,
) -> ActorLossOutput:
    """Compute the Bandit-STOR actor loss.

    Shapes: policy/behavior/q_hat `[B, K]`; logged_action_index/reward/pscore `[B]`.
    The residual correction uses logged rewards only and never creates labels for
    unobserved candidate actions.
    """
    dr = doubly_robust_value(
        policy_probs,
        q_hat,
        logged_action_index,
        reward,
        pscore,
        clip=importance_clip,
    )
    div = alpha_divergence(policy_probs, behavior_probs, alpha=alpha, mask=mask).mean()
    entropy = tsallis_q2_entropy(policy_probs, mask=mask).mean()
    unsupported = unsupported_action_mass(policy_probs, behavior_probs, mu_min=mu_min).mean()
    scaled_dr = reward_value_scale * dr
    loss = -scaled_dr + beta_alpha * div - lambda_tsallis * entropy + lambda_support * unsupported
    if not torch.isfinite(loss):
        raise FloatingPointError("Sparse Tsallis actor loss became non-finite")
    return ActorLossOutput(
        loss=loss,
        components={
            "loss": float(loss.detach().cpu()),
            "dr_value": float(dr.detach().cpu()),
            "scaled_dr_value": float(scaled_dr.detach().cpu()),
            "reward_value_scale": float(reward_value_scale),
            "alpha_divergence": float(div.detach().cpu()),
            "tsallis_entropy": float(entropy.detach().cpu()),
            "unsupported_action_mass": float(unsupported.detach().cpu()),
        },
    )
