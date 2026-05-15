"""Policy selection scoring for reward-lift-driven model choice."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Mapping


def _as_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{k: _as_namespace(v) for k, v in value.items()})
    return value


def _finite_metrics(metrics: dict[str, Any]) -> bool:
    for value in metrics.values():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True


def _cfg(config: Any) -> Any:
    return _as_namespace(config)


def compute_policy_selection_score(metrics: dict, gates: dict, config: dict | Any) -> float:
    """Return a risk-adjusted reward-lift score.

    Primary driver: DR lift over behavior policy. Hard failures return -1e6.
    Soft penalties are reliability/safety controls and should not dominate ordinary
    reward-lift differences unless risk is large.
    """
    cfg = _cfg(config)
    n = int(metrics.get("n", 0))
    dr_lift = float(metrics.get("absolute_dr_lift", metrics["doubly_robust"] - metrics["behavior_policy_value"]))
    snips_lift = float(metrics.get("snips_lift_over_behavior", 0.0))
    ips_lift = float(metrics.get("ips_lift_over_behavior", 0.0))
    sndr_lift = float(metrics.get("sndr_lift_over_behavior", dr_lift))
    switch_dr_lift = float(metrics.get("switch_dr_lift_over_behavior", dr_lift))
    dros_lift = float(metrics.get("dros_lift_over_behavior", dr_lift))
    reward_ok = bool(metrics.get("reward_sanity_passed", False))

    hard = cfg.gates.hard
    if n < int(hard.min_eval_n):
        return -1e6
    if not _finite_metrics(metrics):
        return -1e6
    if float(metrics.get("unsupported_action_mass", 1.0)) > float(hard.unsupported_action_mass_max):
        return -1e6
    if float(metrics.get("max_importance_weight", float("inf"))) > float(
        hard.max_importance_weight_catastrophic
    ):
        return -1e6
    if float(metrics.get("ess_ratio", 0.0)) < float(hard.ess_ratio_catastrophic_min):
        return -1e6
    if bool(metrics.get("is_deterministic_collapse", False)):
        return -1e6
    if float(metrics.get("avg_sparse_support_size", 0.0)) < float(getattr(cfg.gates.sparsity, "min_avg_support_size", 5)):
        return -1e6
    if float(metrics.get("tsallis_entropy_ratio", 1.0)) < float(getattr(cfg.gates.sparsity, "min_entropy_ratio", 0.20)):
        return -1e6
    if dr_lift > 0.0 and (ips_lift <= 0.0 or snips_lift <= 0.0):
        return -1e6
    if dr_lift > 0.0 and (switch_dr_lift <= 0.0 or dros_lift <= 0.0):
        return -1e6
    if bool(metrics.get("estimator_lifts_direction_agree", True)) is False:
        return -1e6
    if float(metrics.get("max_lift_disagreement", 0.0)) > float(getattr(hard, "max_lift_disagreement", 0.002)):
        return -1e6

    preferred = cfg.gates.preferred
    score_cfg = cfg.score
    score = min(dr_lift, sndr_lift, switch_dr_lift, dros_lift) if reward_ok else snips_lift
    score -= float(score_cfg.max_weight_penalty) * max(
        0.0,
        float(metrics["max_importance_weight"]) - float(preferred.max_importance_weight),
    )
    score -= float(score_cfg.ess_penalty) * max(
        0.0,
        float(preferred.ess_ratio) - float(metrics["ess_ratio"]),
    )
    score -= float(score_cfg.alpha_penalty) * max(
        0.0,
        float(metrics["alpha_divergence"]) - float(preferred.alpha_divergence),
    )
    score -= float(score_cfg.entropy_penalty) * max(
        0.0,
        float(metrics["tsallis_entropy_ratio"]) - float(cfg.gates.sparsity.max_entropy_ratio),
    )
    score -= float(score_cfg.support_size_penalty) * max(
        0.0,
        float(metrics["avg_sparse_support_size"]) - float(cfg.gates.sparsity.max_avg_support_size),
    )
    return float(score)
