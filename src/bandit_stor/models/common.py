"""Common neural modules."""

from __future__ import annotations

import torch
from torch import nn


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
    """Build a small MLP for tabular contextual-bandit models."""
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(max(num_layers - 1, 0)):
        layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


def pair_features(context: torch.Tensor, action_context: torch.Tensor) -> torch.Tensor:
    """Concatenate context and action features for every candidate.

    Args:
        context: Tensor `[B, D_x]`.
        action_context: Tensor `[B, K, D_a]`.

    Returns:
        Tensor `[B, K, D_x + D_a]`.
    """
    b, k, _ = action_context.shape
    context_expanded = context.unsqueeze(1).expand(b, k, context.shape[-1])
    return torch.cat([context_expanded, action_context], dim=-1)
