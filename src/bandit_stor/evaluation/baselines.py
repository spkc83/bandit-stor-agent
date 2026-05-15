"""Policy baselines for reward-lift reports."""

from __future__ import annotations

import torch
from torch import nn

from bandit_stor.models.reward_model import RewardModel
from bandit_stor.models.sparsemax import sparsemax
from bandit_stor.policy_utils import get_behavior_policy_distribution


class _PolicyWrapper(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, context, action_context, mask=None):
        return self.fn(context, action_context, mask)


def make_behavior_policy(dataset_config: dict, behavior_model) -> nn.Module:
    """Policy wrapper returning μ(.|x)."""

    def fn(context, action_context, mask=None):
        batch = type(
            "BatchView",
            (),
            {"context": context, "action_context": action_context, "mask": mask, "candidate_actions": torch.empty(context.shape[0], action_context.shape[1], device=context.device, dtype=torch.long), "behavior_policy_probs": None},
        )
        return get_behavior_policy_distribution(batch, dataset_config, behavior_model)

    return _PolicyWrapper(fn)


def make_uniform_policy() -> nn.Module:
    """Uniform policy over eligible candidates."""

    def fn(context, action_context, mask=None):
        b, k, _ = action_context.shape
        if mask is None:
            return torch.full((b, k), 1.0 / k, device=context.device)
        probs = mask.float()
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    return _PolicyWrapper(fn)


def make_reward_greedy_policy(reward_model: RewardModel) -> nn.Module:
    """Top-1 deterministic policy induced by reward model predictions."""

    def fn(context, action_context, mask=None):
        q = reward_model(context, action_context)
        if mask is not None:
            q = q.masked_fill(~mask, -1e30)
        idx = q.argmax(dim=-1, keepdim=True)
        return torch.zeros_like(q).scatter(1, idx, 1.0)

    return _PolicyWrapper(fn)


def make_reward_topk_sparse_policy(reward_model: RewardModel, top_k: int = 20) -> nn.Module:
    """Sparse reward-model top-k policy using Sparsemax over reward predictions."""

    def fn(context, action_context, mask=None):
        q = reward_model(context, action_context)
        masked_q = q if mask is None else q.masked_fill(~mask, -1e30)
        k = min(int(top_k), q.shape[-1])
        idx = torch.topk(masked_q, k=k, dim=-1).indices
        topk_mask = torch.zeros_like(q, dtype=torch.bool).scatter(1, idx, True)
        effective_mask = topk_mask if mask is None else (mask & topk_mask)
        return sparsemax(q, mask=effective_mask)

    return _PolicyWrapper(fn)
