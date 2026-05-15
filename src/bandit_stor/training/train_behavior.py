"""Behavior policy training."""

from __future__ import annotations

import logging
import torch
from torch.utils.data import DataLoader

from bandit_stor.data.collate import collate_logged_interactions
from bandit_stor.models.behavior_policy import BehaviorPolicyModel


logger = logging.getLogger(__name__)

def train_behavior_model(
    model: BehaviorPolicyModel,
    dataset,
    *,
    epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Train behavior model by logged action cross-entropy.

    Batch shapes follow `LoggedInteractionBatch`; targets are logged action indices `[B]`.
    """
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_logged_interactions)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    last = 0.0
    logger.info(
        "Behavior training start: epochs=%s batch_size=%s batches_per_epoch=%s",
        epochs,
        batch_size,
        len(loader),
    )
    for epoch in range(max(int(epochs), 0)):
        logger.debug("Behavior epoch %s/%s start", epoch + 1, epochs)
        for batch in loader:
            batch = batch.to(device)
            logits = model.logits(batch.context, batch.action_context).masked_fill(~batch.mask, -1e30)
            loss = torch.nn.functional.cross_entropy(logits, batch.logged_action_index)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.detach().cpu())
        logger.info("Behavior epoch %s/%s complete: loss=%.6f", epoch + 1, epochs, last)
    return {"behavior_loss": last}
