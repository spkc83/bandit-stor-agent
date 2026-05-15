from bandit_stor.evaluation.diagnostics import evaluate_gates
from bandit_stor.evaluation.reward import compute_reward_sanity
from bandit_stor.evaluation.scoring import compute_policy_selection_score


def _cfg():
    return {
        "score": {
            "max_weight_penalty": 0.00005,
            "ess_penalty": 0.01,
            "alpha_penalty": 0.0001,
            "entropy_penalty": 0.001,
            "support_size_penalty": 0.00001,
        },
        "gates": {
            "hard": {
                "min_eval_n": 100,
                "unsupported_action_mass_max": 0.01,
                "max_importance_weight_catastrophic": 80.0,
                "ess_ratio_catastrophic_min": 0.01,
                "max_lift_disagreement": 0.002,
            },
            "preferred": {
                "max_importance_weight": 25.0,
                "ess_ratio": 0.05,
                "alpha_divergence": 5.0,
            },
            "sparsity": {"min_avg_support_size": 5, "max_entropy_ratio": 0.99, "min_entropy_ratio": 0.20, "max_avg_support_size": 50},
        },
    }


def test_policy_selection_score_is_dr_lift_driven():
    metrics = {
        "n": 1000,
        "doubly_robust": 0.006,
        "behavior_policy_value": 0.004,
        "max_importance_weight": 10.0,
        "ess_ratio": 0.5,
        "alpha_divergence": 1.0,
        "tsallis_entropy_ratio": 0.5,
        "avg_sparse_support_size": 20.0,
        "unsupported_action_mass": 0.0,
        "ips_lift_over_behavior": 0.0015,
        "snips_lift_over_behavior": 0.0018,
        "sndr_lift_over_behavior": 0.0019,
        "switch_dr_lift_over_behavior": 0.0017,
        "dros_lift_over_behavior": 0.0016,
        "max_lift_disagreement": 0.0005,
        "estimator_lifts_direction_agree": True,
        "is_deterministic_collapse": False,
        "reward_sanity_passed": True,
    }
    assert compute_policy_selection_score(metrics, {}, _cfg()) == 0.0016


def test_policy_selection_score_hard_rejects_tiny_eval():
    metrics = {
        "n": 2,
        "doubly_robust": 1.0,
        "behavior_policy_value": 0.0,
        "max_importance_weight": 1.0,
        "ess_ratio": 1.0,
        "alpha_divergence": 0.0,
        "tsallis_entropy_ratio": 0.0,
        "avg_sparse_support_size": 5.0,
        "unsupported_action_mass": 0.0,
        "ips_lift_over_behavior": 1.0,
        "snips_lift_over_behavior": 1.0,
        "sndr_lift_over_behavior": 1.0,
        "switch_dr_lift_over_behavior": 1.0,
        "dros_lift_over_behavior": 1.0,
        "max_lift_disagreement": 0.0,
        "estimator_lifts_direction_agree": True,
        "is_deterministic_collapse": False,
        "reward_sanity_passed": True,
    }
    assert compute_policy_selection_score(metrics, {}, _cfg()) == -1e6


def test_policy_selection_score_hard_rejects_dr_ope_disagreement():
    metrics = {
        "n": 1000,
        "doubly_robust": 0.006,
        "behavior_policy_value": 0.004,
        "max_importance_weight": 10.0,
        "ess_ratio": 0.5,
        "alpha_divergence": 1.0,
        "tsallis_entropy_ratio": 0.5,
        "avg_sparse_support_size": 20.0,
        "unsupported_action_mass": 0.0,
        "ips_lift_over_behavior": -0.001,
        "snips_lift_over_behavior": -0.001,
        "sndr_lift_over_behavior": 0.001,
        "switch_dr_lift_over_behavior": -0.001,
        "dros_lift_over_behavior": -0.001,
        "max_lift_disagreement": 0.003,
        "estimator_lifts_direction_agree": False,
        "is_deterministic_collapse": False,
        "reward_sanity_passed": True,
    }
    assert compute_policy_selection_score(metrics, {}, _cfg()) == -1e6


def test_policy_selection_score_uses_snips_when_reward_sanity_fails():
    metrics = {
        "n": 1000,
        "doubly_robust": 0.006,
        "behavior_policy_value": 0.004,
        "max_importance_weight": 10.0,
        "ess_ratio": 0.5,
        "alpha_divergence": 1.0,
        "tsallis_entropy_ratio": 0.5,
        "avg_sparse_support_size": 20.0,
        "unsupported_action_mass": 0.0,
        "ips_lift_over_behavior": 0.001,
        "snips_lift_over_behavior": 0.0015,
        "sndr_lift_over_behavior": 0.001,
        "switch_dr_lift_over_behavior": 0.001,
        "dros_lift_over_behavior": 0.001,
        "max_lift_disagreement": 0.001,
        "estimator_lifts_direction_agree": True,
        "is_deterministic_collapse": False,
        "reward_sanity_passed": False,
    }
    assert compute_policy_selection_score(metrics, {}, _cfg()) == 0.0015


def test_reward_sanity_requires_auc_and_top_decile_lift():
    metrics = {
        "reward_positive_rate": 0.01,
        "reward_prediction_mean": 0.01,
        "reward_constant_baseline_log_loss": 0.05,
        "reward_log_loss": 0.05,
        "reward_ece_10bin": 0.0,
        "reward_auc_pr": 0.02,
        "reward_auc": 0.59,
        "reward_top_decile_lift": 1.5,
    }
    checks = compute_reward_sanity(metrics, {"auc_roc_min": 0.60, "top_decile_lift_min": 1.0})
    assert checks["auc_pr_lift_min"]
    assert not checks["auc_roc_min"]
    assert not checks["passed"]


def test_gates_do_not_block_on_logged_action_coverage_when_ess_and_support_are_good():
    metrics = {
        "n": 1000,
        "reward_auc": 0.7,
        "reward_auc_pr": 0.02,
        "reward_top_decile_lift": 1.2,
        "reward_sanity_passed": True,
        "absolute_dr_lift": 0.01,
        "ips_lift_over_behavior": 0.01,
        "snips_lift_over_behavior": 0.01,
        "switch_dr_lift_over_behavior": 0.01,
        "dros_lift_over_behavior": 0.01,
        "estimator_lifts_direction_agree": True,
        "robust_lifts_positive": True,
        "dm_dr_dros_direction_agree": True,
        "max_lift_disagreement": 0.0,
        "max_importance_weight": 2.0,
        "ess_ratio": 0.8,
        "unsupported_action_mass": 0.0,
        "logged_action_coverage": 0.0,
        "avg_sparse_support_size": 10.0,
        "is_near_uniform_policy": False,
        "is_deterministic_collapse": False,
    }
    cfg = {
        "reward_sanity": {"auc_roc_min": 0.60, "top_decile_lift_min": 1.0},
        "gates": {
            "hard": {"min_eval_n": 100, "unsupported_action_mass_max": 0.01, "ess_ratio_catastrophic_min": 0.1},
            "preferred": {"max_importance_weight": 3.0, "ess_ratio": 0.2},
            "sparsity": {"min_avg_support_size": 1, "max_avg_support_size": 20},
            "bootstrap": {"min_dr_lift_p05": None},
        },
    }
    gates = evaluate_gates(metrics, cfg)
    assert gates["checks"]["ess_ratio"]
    assert gates["checks"]["unsupported_action_mass"]
    assert "logged_action_coverage" not in gates["checks"]
    assert gates["passed"]
