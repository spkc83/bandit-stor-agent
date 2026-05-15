"""Sparsemax transformation for sparse categorical policies."""

from __future__ import annotations

import torch
from torch import nn


class Sparsemax(nn.Module):
    """Sparsemax over the last dimension.

    Input shape: logits `[B, K]`. Optional mask shape: `[B, K]`.
    Output shape: probabilities `[B, K]` with exact zeros and rows summing to one.
    """

    def forward(self, logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return sparsemax(logits, mask=mask, dim=-1)


def sparsemax(
    logits: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1, eps: float = 1e-12
) -> torch.Tensor:
    """Apply sparsemax with optional eligibility mask.

    Masked entries receive probability zero. Rows with all candidates masked raise a
    `ValueError` because a valid policy cannot be formed.
    """
    if dim != -1:
        logits = logits.transpose(dim, -1)
    z = logits
    if mask is not None:
        if mask.shape != z.shape:
            raise ValueError(f"mask shape {mask.shape} must match logits shape {z.shape}")
        if (~mask).all(dim=-1).any():
            raise ValueError("Sparsemax received a row with no eligible candidates")
        z = z.masked_fill(~mask, -1e30)
    z = z - z.max(dim=-1, keepdim=True).values
    z_sorted = torch.sort(z, descending=True, dim=-1).values
    k = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype).view(
        *([1] * (z.ndim - 1)), -1
    )
    z_cumsum = z_sorted.cumsum(dim=-1)
    support = 1 + k * z_sorted > z_cumsum
    k_z = support.sum(dim=-1, keepdim=True).clamp_min(1)
    tau = (z_cumsum.gather(-1, k_z - 1) - 1) / k_z.to(z.dtype)
    probs = torch.clamp(z - tau, min=0.0)
    if mask is not None:
        probs = probs.masked_fill(~mask, 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    if dim != -1:
        probs = probs.transpose(dim, -1)
    return probs
