"""Sparse Tsallis actor training."""

from __future__ import annotations

import logging
import torch
from torch.utils.data import DataLoader

from bandit_stor.data.collate import collate_logged_interactions
from bandit_stor.models.actor import SparseTsallisActor
from bandit_stor.models.behavior_policy import BehaviorPolicyModel
from bandit_stor.models.reward_model import RewardModel
from bandit_stor.objectives.losses import sparse_tsallis_actor_loss
from bandit_stor.policy_utils import get_behavior_policy_distribution


logger = logging.getLogger(__name__)

def train_actor(
    actor: SparseTsallisActor,
    reward_model: RewardModel,
    behavior_model: BehaviorPolicyModel,
    dataset,
    *,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    gradient_clip_norm: float = 1.0,
    device: torch.device | str = "cpu",
    dataset_config: dict | None = None,
    **loss_kwargs,
) -> dict[str, float]:
    """Train actor against the DR/support-constrained objective.

    Actor probabilities, behavior probabilities, and q_hat all have shape `[B, K]`.
    """
    actor.to(device)
    reward_model.to(device).eval()
    behavior_model.to(device).eval()
    for module in (reward_model, behavior_model):
        for param in module.parameters():
            param.requires_grad_(False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_logged_interactions)
    opt = torch.optim.AdamW(actor.parameters(), lr=learning_rate, weight_decay=weight_decay)
    components: dict[str, float] = {}
    logger.info(
        "Actor training start: epochs=%s batch_size=%s batches_per_epoch=%s",
        epochs,
        batch_size,
        len(loader),
    )
    for epoch in range(max(int(epochs), 0)):
        logger.debug("Actor epoch %s/%s start", epoch + 1, epochs)
        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad():
                q_hat = reward_model(batch.context, batch.action_context)
                behavior_probs = get_behavior_policy_distribution(
                    batch, dataset_config or {}, behavior_model
                )
            policy_probs = actor(batch.context, batch.action_context, batch.mask)
            out = sparse_tsallis_actor_loss(
                policy_probs,
                behavior_probs,
                q_hat,
                batch.logged_action_index,
                batch.reward,
                batch.pscore,
                mask=batch.mask,
                **loss_kwargs,
            )
            opt.zero_grad()
            out.loss.backward()
            if gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), gradient_clip_norm)
            opt.step()
            components = {f"actor_{k}": v for k, v in out.components.items()}
        logger.info("Actor epoch %s/%s complete: %s", epoch + 1, epochs, components)
    return components
