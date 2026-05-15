"""Policy distribution helpers."""

from __future__ import annotations

from typing import Any

import torch

from bandit_stor.models.behavior_policy import BehaviorPolicyModel


def get_behavior_policy_distribution(
    batch,
    dataset_config: dict[str, Any],
    behavior_model: BehaviorPolicyModel | None = None,
) -> torch.Tensor:
    """Return full behavior distribution μ(.|x) over candidates `[B, K]`.

    For Open Bandit random policy, μ is known exactly: uniform over all actions. Logged
    `pscore` remains the source of truth for μ(a_i|x_i) in OPE residual/weights.
    """
    if (
        str(dataset_config.get("name")) == "open_bandit"
        and str(dataset_config.get("behavior_policy")) == "random"
    ):
        b, k = batch.candidate_actions.shape
        return torch.full((b, k), 1.0 / float(k), device=batch.context.device, dtype=torch.float32)
    if batch.behavior_policy_probs is not None:
        return batch.behavior_policy_probs.clamp_min(1e-8)
    if behavior_model is None:
        raise ValueError("behavior_model is required when full behavior probabilities are unavailable")
    return behavior_model(batch.context, batch.action_context, batch.mask).clamp_min(1e-8)
