"""Cross-fitted reward-model OPE utilities."""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

from bandit_stor.evaluation.ope import evaluate_policy
from bandit_stor.evaluation.reward import fit_reward_calibration
from bandit_stor.models.actor import SparseTsallisActor
from bandit_stor.models.behavior_policy import BehaviorPolicyModel
from bandit_stor.models.reward_model import RewardModel
from bandit_stor.training.train_reward import train_reward_model

logger = logging.getLogger(__name__)


def _weighted_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    """Weighted mean of numeric fold metrics by fold size."""
    total = sum(int(row.get("n", 0)) for row in rows)
    if total <= 0:
        return None
    vals = [(int(row.get("n", 0)), row.get(key)) for row in rows]
    vals = [(n, float(v)) for n, v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not vals:
        return None
    return float(sum(n * v for n, v in vals) / total)


def evaluate_policy_crossfit_reward(
    actor: SparseTsallisActor,
    reward_template: RewardModel,
    behavior_model: BehaviorPolicyModel,
    dataset,
    *,
    dataset_context_dim: int,
    dataset_action_dim: int,
    reward_model_config: dict[str, Any],
    reward_train_config: dict[str, Any],
    dataset_config: dict[str, Any] | None = None,
    folds: int = 2,
    seed: int = 42,
    batch_size: int = 512,
    device: torch.device | str = "cpu",
    alpha: float = 1.5,
    mu_min: float = 1e-4,
    switch_tau: float = 25.0,
    dros_lambda: float = 1.0,
    dros_lambdas: list[float] | tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Train q_hat on fold-complements and evaluate DR on held-out folds.

    The actor and behavior policy are fixed.  Each fold trains a fresh reward model on
    A = all rows except the fold and evaluates OPE on B = held-out fold, then rotates.
    Returned metrics are fold-size-weighted summaries plus per-fold evidence.
    """
    n = len(dataset)
    k = max(2, min(int(folds), n))
    rng = np.random.default_rng(int(seed))
    indices = np.arange(n)
    rng.shuffle(indices)
    fold_indices = [fold.tolist() for fold in np.array_split(indices, k) if len(fold)]
    fold_metrics: list[dict[str, Any]] = []
    for fold_id, heldout in enumerate(fold_indices):
        heldout_set = set(heldout)
        train_idx = [int(i) for i in indices.tolist() if int(i) not in heldout_set]
        if not train_idx:
            continue
        logger.info("Cross-fit reward fold %s/%s: train=%s heldout=%s", fold_id + 1, k, len(train_idx), len(heldout))
        model = RewardModel(
            dataset_context_dim,
            dataset_action_dim,
            hidden_dim=int(reward_model_config.get("hidden_dim", 128)),
            num_layers=int(reward_model_config.get("num_layers", 3)),
            dropout=float(reward_model_config.get("dropout", 0.1)),
        )
        # Preserve any calibration defaults from the template if no fold calibration is possible.
        model.set_calibration(
            method=getattr(reward_template, "calibration_method", "none"),
            a=float(getattr(reward_template, "calibration_a", 1.0)),
            b=float(getattr(reward_template, "calibration_b", 0.0)),
        )
        train_subset = Subset(dataset, train_idx)
        heldout_subset = Subset(dataset, heldout)
        train_metrics = train_reward_model(
            model,
            train_subset,
            batch_size=batch_size,
            device=device,
            **copy.deepcopy(reward_train_config),
        )
        fit_reward_calibration(
            model,
            train_subset,
            batch_size=batch_size,
            device=device,
            method=str(reward_model_config.get("calibration", "auto_logit_correction")),
            pos_weight=train_metrics.get("reward_pos_weight"),
        )
        metrics = evaluate_policy(
            actor,
            model,
            behavior_model,
            heldout_subset,
            batch_size=batch_size,
            device=device,
            alpha=alpha,
            mu_min=mu_min,
            bootstrap_samples=0,
            dataset_config=dataset_config,
            switch_tau=switch_tau,
            dros_lambda=dros_lambda,
            dros_lambdas=dros_lambdas,
        )
        metrics["fold_id"] = fold_id
        fold_metrics.append(metrics)
    summary: dict[str, Any] = {"crossfit_enabled": True, "crossfit_folds": len(fold_metrics), "crossfit_fold_metrics": fold_metrics}
    for key in [
        "behavior_policy_value",
        "direct_method",
        "ips_lift_over_behavior",
        "snips_lift_over_behavior",
        "dr_lift_over_behavior",
        "sndr_lift_over_behavior",
        "switch_dr_lift_over_behavior",
        "dros_lift_over_behavior",
        "ess_ratio",
        "max_importance_weight",
        "unsupported_action_mass",
        "max_lift_disagreement",
    ]:
        value = _weighted_mean(fold_metrics, key)
        if value is not None:
            summary[f"crossfit_{key}"] = value
    summary["crossfit_dros_agrees_with_dr"] = bool(
        summary.get("crossfit_dr_lift_over_behavior", 0.0) * summary.get("crossfit_dros_lift_over_behavior", 0.0) >= 0.0
    )
    return summary
