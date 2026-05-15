"""Canonical logged contextual-bandit schema.

Shape conventions:
- B: batch size
- K: number of candidate actions
- D_x: context feature dimension
- D_a: action feature dimension
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LoggedInteractionBatch:
    """A mini-batch of logged bandit interactions.

    Attributes:
        context: Float tensor `[B, D_x]` with observed context features.
        candidate_actions: Long tensor `[B, K]` with candidate-set-local action ids.
        action_context: Float tensor `[B, K, D_a]` with per-candidate action features.
        logged_action_index: Long tensor `[B]` indexing the logged action within `K`.
        reward: Float tensor `[B]` with observed reward for the logged action only.
        pscore: Float tensor `[B]` with logged propensity for the logged action.
        position: Long tensor `[B]` with slate position or zero if unavailable.
        mask: Bool tensor `[B, K]` where true means candidate is eligible.
        behavior_policy_probs: Optional float tensor `[B, K]` with behavior probabilities.
    """

    context: torch.Tensor
    candidate_actions: torch.Tensor
    action_context: torch.Tensor
    logged_action_index: torch.Tensor
    reward: torch.Tensor
    pscore: torch.Tensor
    position: torch.Tensor
    mask: torch.Tensor
    behavior_policy_probs: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "LoggedInteractionBatch":
        """Move every tensor field to `device`, preserving shapes."""
        return LoggedInteractionBatch(
            context=self.context.to(device),
            candidate_actions=self.candidate_actions.to(device),
            action_context=self.action_context.to(device),
            logged_action_index=self.logged_action_index.to(device),
            reward=self.reward.to(device),
            pscore=self.pscore.to(device),
            position=self.position.to(device),
            mask=self.mask.to(device),
            behavior_policy_probs=None
            if self.behavior_policy_probs is None
            else self.behavior_policy_probs.to(device),
        )

    @property
    def batch_size(self) -> int:
        return int(self.context.shape[0])

    @property
    def num_candidates(self) -> int:
        return int(self.candidate_actions.shape[1])


def gather_logged_values(values: torch.Tensor, logged_action_index: torch.Tensor) -> torch.Tensor:
    """Gather values at logged actions.

    Args:
        values: Tensor `[B, K]`.
        logged_action_index: Long tensor `[B]`.

    Returns:
        Tensor `[B]` containing `values[i, logged_action_index[i]]`.
    """
    return values.gather(1, logged_action_index.view(-1, 1)).squeeze(1)
