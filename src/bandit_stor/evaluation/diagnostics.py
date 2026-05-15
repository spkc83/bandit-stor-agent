"""Promotion gate diagnostics."""

from __future__ import annotations

import math
from typing import Any


def _metric(metrics: dict[str, Any], name: str, default: float) -> float:
    value = metrics.get(name, default)
    if value is None:
        return default
    return float(value)


def _all_finite(metrics: dict[str, Any]) -> bool:
    return not any(
        not math.isfinite(float(v)) for v in metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)
    )


def _same_direction(*values: float) -> bool:
    signs = {math.copysign(1.0, v) for v in values if abs(v) > 1e-12}
    return len(signs) <= 1


def evaluate_gates(metrics: dict[str, Any], gate_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate reward-lift readiness gates without hard-failing sparse coverage."""
    cfg = gate_config or {}
    gates = cfg.get("gates", cfg)
    hard = gates.get("hard", {})
    preferred = gates.get("preferred", {})
    sparsity = gates.get("sparsity", {})
    bootstrap = gates.get("bootstrap", {})
    min_eval_n = int(hard.get("min_eval_n", gates.get("min_eval_n", 1)))
    max_weight_pref = float(preferred.get("max_importance_weight", gates.get("max_importance_weight_max", 20.0)))
    ess_pref = float(preferred.get("ess_ratio", gates.get("ess_ratio_min", 0.0)))
    reward_cfg = cfg.get("reward_sanity", {})
    require_significance = bootstrap.get("min_dr_lift_p05", None) is not None
    checks = {
        "min_eval_n": int(metrics.get("n", 0)) >= min_eval_n,
        "no_nan_metrics": _all_finite(metrics),
        "reward_auc_roc_min": _metric(metrics, "reward_auc", 0.0) >= float(reward_cfg.get("auc_roc_min", 0.60)),
        "reward_auc_pr_present": metrics.get("reward_auc_pr") is not None,
        "reward_top_decile_lift_min": _metric(metrics, "reward_top_decile_lift", 0.0) > float(reward_cfg.get("top_decile_lift_min", 1.0)),
        "reward_sanity_passed": bool(metrics.get("reward_sanity_passed", True)),
        "positive_dr_lift": _metric(metrics, "absolute_dr_lift", _metric(metrics, "dr_lift_over_behavior", 0.0)) > 0.0,
        "statistically_significant_dr_lift": (not require_significance) or (metrics.get("dr_lift_p05") is not None and float(metrics["dr_lift_p05"]) >= float(bootstrap.get("min_dr_lift_p05", 0.0))),
        "ips_snips_dr_same_direction": _same_direction(
            _metric(metrics, "ips_lift_over_behavior", 0.0),
            _metric(metrics, "snips_lift_over_behavior", 0.0),
            _metric(metrics, "absolute_dr_lift", _metric(metrics, "dr_lift_over_behavior", 0.0)),
        ),
        "estimator_lifts_same_direction": bool(metrics.get("estimator_lifts_direction_agree", True)),
        "robust_lifts_positive": bool(metrics.get("robust_lifts_positive", True)),
        "dm_dr_dros_direction_agree": bool(metrics.get("dm_dr_dros_direction_agree", True)),
        "switch_dr_positive": _metric(metrics, "switch_dr_lift_over_behavior", 0.0) > 0.0,
        "dros_positive": _metric(metrics, "dros_lift_over_behavior", 0.0) > 0.0,
        "max_lift_disagreement": _metric(metrics, "max_lift_disagreement", 0.0)
        <= float(hard.get("max_lift_disagreement", float("inf"))),
        "max_importance_weight": _metric(metrics, "max_importance_weight", float("inf")) <= max_weight_pref,
        "ess_ratio": _metric(metrics, "ess_ratio", 0.0) >= ess_pref,
        "unsupported_action_mass": _metric(metrics, "unsupported_action_mass", 1.0)
        <= float(hard.get("unsupported_action_mass_max", gates.get("unsupported_action_mass_max", 0.01))),
        "avg_support_size_min": _metric(metrics, "avg_sparse_support_size", 0.0)
        >= float(sparsity.get("min_avg_support_size", gates.get("avg_support_size_min", 1.0))),
        "avg_support_size_max": _metric(metrics, "avg_sparse_support_size", float("inf"))
        <= float(sparsity.get("max_avg_support_size", gates.get("avg_support_size_max", 50.0))),
        "not_uniform_clone": not bool(metrics.get("is_near_uniform_policy", False)),
        "not_deterministic_collapse": not bool(metrics.get("is_deterministic_collapse", False)),
        "hard_max_importance_weight": _metric(metrics, "max_importance_weight", float("inf"))
        <= float(hard.get("max_importance_weight_catastrophic", float("inf"))),
        "hard_ess_ratio": _metric(metrics, "ess_ratio", 0.0)
        >= float(hard.get("ess_ratio_catastrophic_min", 0.0)),
    }
    if bootstrap.get("min_dr_lift_p05") is not None and metrics.get("dr_lift_p05") is not None:
        checks["bootstrap_dr_lift_p05"] = float(metrics["dr_lift_p05"]) >= float(bootstrap["min_dr_lift_p05"])
    return {"passed": bool(all(checks.values())), "checks": checks}
