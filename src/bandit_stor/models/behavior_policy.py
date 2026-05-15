"""Behavior support model for full candidate-set diagnostics."""

from __future__ import annotations

import torch
from torch import nn

from bandit_stor.models.common import make_mlp, pair_features


class BehaviorPolicyModel(nn.Module):
    """Estimate behavior probabilities over candidates.

    Input shapes: context `[B, D_x]`, action_context `[B, K, D_a]`, mask `[B, K]`.
    Output shape: probabilities `[B, K]`.
    """

    def __init__(self, context_dim: int, action_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.net = make_mlp(context_dim + action_dim, hidden_dim, 1, num_layers, dropout)

    def logits(self, context: torch.Tensor, action_context: torch.Tensor) -> torch.Tensor:
        features = pair_features(context, action_context)
        return self.net(features).squeeze(-1)

    def forward(self, context: torch.Tensor, action_context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.logits(context, action_context)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e30)
        return torch.softmax(logits, dim=-1)
