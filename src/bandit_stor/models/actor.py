"""Sparse Tsallis actor."""

from __future__ import annotations

import torch
from torch import nn

from bandit_stor.models.common import make_mlp, pair_features
from bandit_stor.models.sparsemax import Sparsemax


class SparseTsallisActor(nn.Module):
    """Actor producing sparse candidate probabilities via Sparsemax.

    Input shapes: context `[B, D_x]`, action_context `[B, K, D_a]`, mask `[B, K]`.
    Output shape: sparse probabilities `[B, K]`.
    """

    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        temperature: float = 1.0,
        top_k_before_sparsemax: int | None = None,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.top_k_before_sparsemax = top_k_before_sparsemax
        self.net = make_mlp(context_dim + action_dim, hidden_dim, 1, num_layers, dropout)
        self.policy_head = Sparsemax()

    def logits(self, context: torch.Tensor, action_context: torch.Tensor) -> torch.Tensor:
        return self.net(pair_features(context, action_context)).squeeze(-1)

    def forward(self, context: torch.Tensor, action_context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return sparse policy probabilities `[B, K]`.

        If `top_k_before_sparsemax` is set, candidates outside the per-row top-k logits
        are masked before Sparsemax. This recommender-oriented ablation guarantees an
        upper bound on sparse support size while retaining differentiability among the
        retained logits.
        """
        logits = self.logits(context, action_context) / max(self.temperature, 1e-6)
        effective_mask = mask
        if self.top_k_before_sparsemax is not None and self.top_k_before_sparsemax > 0:
            k = min(int(self.top_k_before_sparsemax), logits.shape[-1])
            masked_logits = logits if mask is None else logits.masked_fill(~mask, -1e30)
            topk_idx = torch.topk(masked_logits, k=k, dim=-1).indices
            topk_mask = torch.zeros_like(logits, dtype=torch.bool).scatter(1, topk_idx, True)
            effective_mask = topk_mask if mask is None else (mask & topk_mask)
        return self.policy_head(logits, mask=effective_mask)
