"""Reward model training."""

from __future__ import annotations

import logging
import torch
from torch.utils.data import DataLoader

from bandit_stor.data.collate import collate_logged_interactions
from bandit_stor.data.schema import gather_logged_values
from bandit_stor.models.reward_model import RewardModel


logger = logging.getLogger(__name__)

def train_reward_model(
    model: RewardModel,
    dataset,
    *,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    device: torch.device | str = "cpu",
    loss_type: str = "weighted_bce",
    pos_weight: float | str | None = "auto",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    negative_downsample_ratio: float | None = None,
) -> dict[str, float]:
    """Train reward model on logged actions only.

    The model emits `[B, K]`, but the supervised BCE is computed only on
    `q_hat[i, logged_action_index[i]]`; unobserved actions are not assigned zero labels.
    """
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_logged_interactions)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    resolved_pos_weight: torch.Tensor | None = None
    if pos_weight == "auto":
        positives = 0.0
        total = 0.0
        scan_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_logged_interactions
        )
        for scan_batch in scan_loader:
            positives += float(scan_batch.reward.sum())
            total += float(scan_batch.reward.numel())
        negatives = max(total - positives, 0.0)
        if positives > 0:
            resolved_pos_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    elif pos_weight is not None:
        resolved_pos_weight = torch.tensor(float(pos_weight), dtype=torch.float32, device=device)
    last = 0.0
    logger.info(
        "Reward training start: epochs=%s batch_size=%s batches_per_epoch=%s",
        epochs,
        batch_size,
        len(loader),
    )
    for epoch in range(max(int(epochs), 0)):
        logger.debug("Reward epoch %s/%s start", epoch + 1, epochs)
        for batch in loader:
            batch = batch.to(device)
            logits = model.logits(batch.context, batch.action_context)
            logged_logits = gather_logged_values(logits, batch.logged_action_index)
            train_mask = torch.ones_like(batch.reward, dtype=torch.bool)
            if negative_downsample_ratio is not None and float(negative_downsample_ratio) > 0.0:
                negative_mask = batch.reward <= 0.0
                keep_negative = torch.rand_like(batch.reward) < float(negative_downsample_ratio)
                train_mask = (batch.reward > 0.0) | (negative_mask & keep_negative)
                if not bool(train_mask.any()):
                    train_mask = torch.ones_like(batch.reward, dtype=torch.bool)
            selected_logits = logged_logits[train_mask]
            selected_rewards = batch.reward[train_mask]
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                selected_logits,
                selected_rewards,
                pos_weight=resolved_pos_weight,
                reduction="none",
            )
            if loss_type == "focal":
                prob = torch.sigmoid(selected_logits)
                p_t = prob * selected_rewards + (1.0 - prob) * (1.0 - selected_rewards)
                alpha_t = float(focal_alpha) * selected_rewards + (1.0 - float(focal_alpha)) * (1.0 - selected_rewards)
                loss = (alpha_t * (1.0 - p_t).pow(float(focal_gamma)) * bce).mean()
            elif loss_type == "weighted_bce":
                loss = bce.mean()
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(selected_logits, selected_rewards)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.detach().cpu())
        logger.info("Reward epoch %s/%s complete: loss=%.6f", epoch + 1, epochs, last)
    return {
        "reward_loss": last,
        "reward_loss_type": loss_type,
        "reward_pos_weight": None if resolved_pos_weight is None else float(resolved_pos_weight.detach().cpu()),
        "reward_negative_downsample_ratio": negative_downsample_ratio,
    }
