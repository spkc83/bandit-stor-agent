"""Off-policy evaluation and diagnostics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from bandit_stor.evaluation.bootstrap import bootstrap_ope_ci
from bandit_stor.evaluation.ope_estimators import estimate_all
from bandit_stor.data.collate import collate_logged_interactions
from bandit_stor.data.schema import gather_logged_values
from bandit_stor.models.actor import SparseTsallisActor
from bandit_stor.models.behavior_policy import BehaviorPolicyModel
from bandit_stor.models.reward_model import RewardModel
from bandit_stor.objectives.alpha_divergence import alpha_divergence, unsupported_action_mass
from bandit_stor.objectives.doubly_robust import doubly_robust_values
from bandit_stor.objectives.ips import effective_sample_size, importance_weights
from bandit_stor.objectives.tsallis import tsallis_q2_entropy
from bandit_stor.policy_utils import get_behavior_policy_distribution


logger = logging.getLogger(__name__)


def _bootstrap_ci(
    values: np.ndarray,
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float | None, float | None]:
    """Bootstrap a mean confidence interval for row-level lift values."""
    if samples <= 0 or values.size < 2:
        return None, None
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    n = values.size
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    alpha = (1.0 - confidence_level) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def evaluate_policy(
    actor: SparseTsallisActor,
    reward_model: RewardModel,
    behavior_model: BehaviorPolicyModel,
    dataset,
    *,
    batch_size: int = 512,
    device: torch.device | str = "cpu",
    alpha: float = 1.5,
    mu_min: float = 1e-4,
    bootstrap_samples: int = 0,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
    dataset_config: dict | None = None,
    switch_tau: float = 25.0,
    dros_lambda: float = 1.0,
    dros_lambdas: list[float] | tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Evaluate a target actor with IPS, SNIPS, DR, and reliability diagnostics.

    Batch shapes: policy/behavior/q_hat `[B, K]`; logged fields `[B]`.
    `logged_action_coverage` is diagnostic only for sparse policies, not a hard gate.
    """
    actor.to(device).eval()
    reward_model.to(device).eval()
    behavior_model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_logged_interactions)
    logger.info("OPE evaluation start: batch_size=%s", batch_size)
    weights_all: list[torch.Tensor] = []
    rewards_all: list[torch.Tensor] = []
    dr_all: list[torch.Tensor] = []
    logged_action_coverage: list[torch.Tensor] = []
    unsupported: list[torch.Tensor] = []
    divergences: list[torch.Tensor] = []
    support_sizes: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    pi_logged_all: list[torch.Tensor] = []
    q_logged_all: list[torch.Tensor] = []
    direct_all: list[torch.Tensor] = []
    pscore_all: list[torch.Tensor] = []
    num_candidates: int | None = None
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pi = actor(batch.context, batch.action_context, batch.mask)
            if num_candidates is None:
                num_candidates = int(pi.shape[-1])
            q_hat = reward_model(batch.context, batch.action_context)
            behavior = get_behavior_policy_distribution(batch, dataset_config or {}, behavior_model)
            pi_logged = gather_logged_values(pi, batch.logged_action_index)
            w = importance_weights(pi_logged, batch.pscore)
            pi_logged_all.append(pi_logged.cpu())
            q_logged_all.append(gather_logged_values(q_hat, batch.logged_action_index).cpu())
            direct_all.append((pi * q_hat).sum(dim=-1).cpu())
            pscore_all.append(batch.pscore.cpu())
            weights_all.append(w.cpu())
            rewards_all.append(batch.reward.cpu())
            dr_all.append(
                doubly_robust_values(pi, q_hat, batch.logged_action_index, batch.reward, batch.pscore).cpu()
            )
            mu_logged = gather_logged_values(behavior, batch.logged_action_index)
            logged_action_coverage.append(((pi_logged > 0) & (mu_logged > mu_min)).float().cpu())
            unsupported.append(unsupported_action_mass(pi, behavior, mu_min=mu_min).cpu())
            divergences.append(alpha_divergence(pi, behavior, alpha=alpha, mask=batch.mask).cpu())
            support_sizes.append((pi > 0).sum(dim=-1).float().cpu())
            entropies.append(tsallis_q2_entropy(pi, batch.mask).cpu())
    weights = torch.cat(weights_all) if weights_all else torch.empty(0)
    rewards = torch.cat(rewards_all) if rewards_all else torch.empty(0)
    _ = torch.cat(dr_all) if dr_all else torch.empty(0)
    if weights.numel() == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    n = int(weights.numel())
    ess = float(effective_sample_size(weights))
    behavior_value = float(rewards.mean())
    rows = {
        "pi_logged": torch.cat(pi_logged_all).numpy(),
        "pscore": torch.cat(pscore_all).numpy(),
        "reward": rewards.numpy(),
        "q_logged": torch.cat(q_logged_all).numpy(),
        "direct": torch.cat(direct_all).numpy(),
    }

    estimates = estimate_all(rows, switch_tau=switch_tau, dros_lambda=dros_lambda)
    sensitivity_lambdas = [float(v) for v in (dros_lambdas or [])]
    dros_sensitivity: dict[str, dict[str, float]] = {}
    for lam in sensitivity_lambdas:
        est_lam = estimate_all(rows, switch_tau=switch_tau, dros_lambda=lam)
        value_lam = float(est_lam["dros"])
        dros_sensitivity[str(lam)] = {
            "target_policy_value_dros": value_lam,
            "dros_lift_over_behavior": value_lam - behavior_value,
            "relative_dros_lift": (value_lam - behavior_value) / max(abs(behavior_value), 1e-12),
        }
    ips = estimates["ips"]
    snips = estimates["snips"]
    dr_value = estimates["doubly_robust"]
    dm_value = estimates["direct_method"]
    sndr_value = estimates["self_normalized_doubly_robust"]
    switch_dr_value = estimates["switch_dr"]
    dros_value = estimates["dros"]
    ips_lift = ips - behavior_value
    snips_lift = snips - behavior_value
    dr_lift = dr_value - behavior_value
    dm_lift = dm_value - behavior_value
    sndr_lift = sndr_value - behavior_value
    switch_dr_lift = switch_dr_value - behavior_value
    dros_lift = dros_value - behavior_value

    def _estimate(sample: dict[str, np.ndarray]) -> dict[str, float]:
        est = estimate_all(sample, switch_tau=switch_tau, dros_lambda=dros_lambda)
        behavior_s = float(np.mean(sample["reward"]))
        est["dr_lift"] = est["doubly_robust"] - behavior_s
        est["sndr_lift"] = est["self_normalized_doubly_robust"] - behavior_s
        est["switch_dr_lift"] = est["switch_dr"] - behavior_s
        est["dros_lift"] = est["dros"] - behavior_s
        return est

    ci = bootstrap_ope_ci(
        rows,
        _estimate,
        n_bootstrap=int(bootstrap_samples),
        seed=int(bootstrap_seed),
        confidence=(0.05, 0.95),
    )
    max_entropy = 1.0 - 1.0 / max(float(num_candidates or 1), 1.0)
    entropy = float(torch.cat(entropies).mean())
    estimator_lifts = [ips_lift, snips_lift, dr_lift, sndr_lift, switch_dr_lift, dros_lift]
    lift_signs = [np.sign(v) for v in estimator_lifts if abs(v) > 1e-12]
    direction_agree = len(set(lift_signs)) <= 1
    max_lift_disagreement = max(abs(a - b) for i, a in enumerate(estimator_lifts) for b in estimator_lifts[i + 1 :])
    robust_lifts_positive = sndr_lift > 0.0 and switch_dr_lift > 0.0 and dros_lift > 0.0
    dm_dr_dros_direction_agree = _same_sign = len({np.sign(v) for v in [dm_lift, dr_lift, dros_lift] if abs(v) > 1e-12}) <= 1
    moderate_positive_dros_lambdas = [
        lam for lam, values in dros_sensitivity.items()
        if 0.1 <= float(lam) <= 100.0 and values["dros_lift_over_behavior"] > 0.0
    ]
    alpha_div = float(torch.cat(divergences).mean())
    avg_support = float(torch.cat(support_sizes).mean())
    metrics: dict[str, Any] = {
        "n": n,
        "behavior_policy_value": behavior_value,
        "target_policy_value_ips": ips,
        "target_policy_value_snips": snips,
        "target_policy_value_dm": dm_value,
        "target_policy_value_dr": dr_value,
        "target_policy_value_sndr": sndr_value,
        "target_policy_value_switch_dr": switch_dr_value,
        "target_policy_value_dros": dros_value,
        "direct_method": dm_value,
        "ips": ips,
        "snips": snips,
        "doubly_robust": dr_value,
        "self_normalized_doubly_robust": sndr_value,
        "switch_dr": switch_dr_value,
        "dros": dros_value,
        "dm_lift_over_behavior": dm_lift,
        "ips_lift_over_behavior": ips_lift,
        "snips_lift_over_behavior": snips_lift,
        "dr_lift_over_behavior": dr_lift,
        "sndr_lift_over_behavior": sndr_lift,
        "switch_dr_lift_over_behavior": switch_dr_lift,
        "dros_lift_over_behavior": dros_lift,
        "absolute_dr_lift": dr_lift,
        "relative_dr_lift": dr_lift / max(abs(behavior_value), 1e-12),
        "relative_sndr_lift": sndr_lift / max(abs(behavior_value), 1e-12),
        "relative_switch_dr_lift": switch_dr_lift / max(abs(behavior_value), 1e-12),
        "relative_dros_lift": dros_lift / max(abs(behavior_value), 1e-12),
        "dr_lift_p05": ci.get("dr_lift_p05"),
        "dr_lift_p50": ci.get("dr_lift_p50"),
        "dr_lift_p95": ci.get("dr_lift_p95"),
        "sndr_lift_p05": ci.get("sndr_lift_p05"),
        "sndr_lift_p50": ci.get("sndr_lift_p50"),
        "sndr_lift_p95": ci.get("sndr_lift_p95"),
        "switch_dr_lift_p05": ci.get("switch_dr_lift_p05"),
        "switch_dr_lift_p50": ci.get("switch_dr_lift_p50"),
        "switch_dr_lift_p95": ci.get("switch_dr_lift_p95"),
        "dros_lift_p05": ci.get("dros_lift_p05"),
        "dros_lift_p50": ci.get("dros_lift_p50"),
        "dros_lift_p95": ci.get("dros_lift_p95"),
        "effective_sample_size": ess,
        "ess_ratio": ess / max(float(n), 1.0),
        "max_importance_weight": float(weights.max()),
        "mean_importance_weight": float(weights.mean()),
        "ips_snips_dr_direction_agree": bool(direction_agree),
        "estimator_lifts_direction_agree": bool(direction_agree),
        "robust_lifts_positive": bool(robust_lifts_positive),
        "dm_dr_dros_direction_agree": bool(dm_dr_dros_direction_agree),
        "dros_sensitivity_has_positive_moderate_lambda": bool(moderate_positive_dros_lambdas),
        "dros_negative_may_be_tuning_artifact": bool(dros_lift <= 0.0 and moderate_positive_dros_lambdas),
        "dros_positive_moderate_lambdas": moderate_positive_dros_lambdas,
        "dros_lambda_sensitivity": dros_sensitivity,
        "max_lift_disagreement": float(max_lift_disagreement),
        "switch_tau": float(switch_tau),
        "dros_lambda": float(dros_lambda),
        "logged_action_coverage": float(torch.cat(logged_action_coverage).mean()),
        "unsupported_action_mass": float(torch.cat(unsupported).mean()),
        "alpha_divergence": alpha_div,
        "avg_sparse_support_size": avg_support,
        "support_fraction": avg_support / max(float(num_candidates or 1), 1.0),
        "tsallis_entropy": entropy,
        "max_tsallis_entropy": max_entropy,
        "tsallis_entropy_ratio": entropy / max(max_entropy, 1e-8),
        "is_near_uniform_policy": bool(
            avg_support >= 0.95 * float(num_candidates or 1)
            and entropy / max(max_entropy, 1e-8) >= 0.995
            and alpha_div <= 1e-4
        ),
        "is_deterministic_collapse": bool(avg_support <= 1.01 or entropy / max(max_entropy, 1e-8) <= 0.01),
    }
    # Backward-compatible alias for older reports.  Under known uniform logging, logged-action
    # coverage of a sparse target policy is not a reliability gate; ESS ratio is the primary
    # overlap diagnostic and unsupported action mass is the only support-blocking diagnostic.
    is_uniform_logging = (
        str((dataset_config or {}).get("name")) == "open_bandit"
        and str((dataset_config or {}).get("behavior_policy")) == "random"
    )
    metrics["support_overlap"] = metrics["ess_ratio"] if is_uniform_logging else metrics["logged_action_coverage"]
    logger.info("OPE evaluation complete: %s", metrics)
    return metrics
