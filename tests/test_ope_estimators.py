import numpy as np

from bandit_stor.evaluation.ope_estimators import (
    direct_method,
    doubly_robust,
    dros,
    estimate_all,
    ipw,
    self_normalized_doubly_robust,
    snipw,
    switch_dr,
)


def _rows():
    return {
        "direct": np.array([0.2, 0.4]),
        "reward": np.array([1.0, 0.0]),
        "q_logged": np.array([0.1, 0.2]),
        "pi_logged": np.array([0.5, 0.25]),
        "pscore": np.array([0.5, 0.5]),
    }


def test_reference_ope_estimators_match_manual_values():
    rows = _rows()
    weights = np.array([1.0, 0.5])
    residual = rows["reward"] - rows["q_logged"]
    assert direct_method(rows["direct"]) == 0.30000000000000004
    assert ipw(rows["reward"], rows["pi_logged"], rows["pscore"]) == np.mean(weights * rows["reward"])
    assert snipw(rows["reward"], rows["pi_logged"], rows["pscore"]) == np.sum(weights * rows["reward"]) / np.sum(weights)
    assert doubly_robust(**rows) == np.mean(rows["direct"] + weights * residual)
    assert self_normalized_doubly_robust(**rows) == direct_method(rows["direct"]) + np.sum(weights * residual) / np.sum(weights)


def test_switch_dr_and_dros_shrink_unstable_residuals():
    rows = _rows()
    switched = switch_dr(**rows, switch_tau=0.75)
    full_dr = doubly_robust(**rows)
    assert switched != full_dr
    shrunk = dros(**rows, shrinkage_lambda=1.0)
    assert min(direct_method(rows["direct"]), full_dr) <= shrunk <= max(direct_method(rows["direct"]), full_dr)
    all_est = estimate_all(rows, switch_tau=0.75, dros_lambda=1.0)
    assert {"direct_method", "ips", "snips", "doubly_robust", "self_normalized_doubly_robust", "switch_dr", "dros"} <= set(all_est)
