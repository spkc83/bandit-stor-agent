"""Reward model q_hat(x, a)."""

from __future__ import annotations

import torch
from torch import nn

from bandit_stor.models.common import make_mlp, pair_features


class RewardModel(nn.Module):
    """Predict expected reward for each candidate.

    Input shapes: context `[B, D_x]`, action_context `[B, K, D_a]`.
    Output shape: reward probabilities/values `[B, K]`.
    """

    def __init__(self, context_dim: int, action_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.net = make_mlp(context_dim + action_dim, hidden_dim, 1, num_layers, dropout)
        self.calibration_method = "none"
        self.calibration_a = 1.0
        self.calibration_b = 0.0

    def logits(self, context: torch.Tensor, action_context: torch.Tensor) -> torch.Tensor:
        return self.net(pair_features(context, action_context)).squeeze(-1)

    def set_calibration(self, *, method: str, a: float = 1.0, b: float = 0.0) -> None:
        """Set scalar logit calibration used for q_hat probabilities."""
        self.calibration_method = method
        self.calibration_a = float(a)
        self.calibration_b = float(b)

    def calibrated_logits(self, context: torch.Tensor, action_context: torch.Tensor) -> torch.Tensor:
        """Return calibrated logits `[B, K]` for reward probabilities."""
        return self.calibration_a * self.logits(context, action_context) + self.calibration_b

    def forward(self, context: torch.Tensor, action_context: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.calibrated_logits(context, action_context))
