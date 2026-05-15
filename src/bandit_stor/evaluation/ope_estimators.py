"""Reference-style off-policy estimators for contextual-bandit policy values."""

from __future__ import annotations

import numpy as np


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def importance_weight(pi_logged, pscore, *, eps: float = 1e-8) -> np.ndarray:
    """Return importance weights `[N] = pi(a_i|x_i) / mu(a_i|x_i)`."""
    return _as_float_array(pi_logged) / np.maximum(_as_float_array(pscore), eps)


def direct_method(direct) -> float:
    """Direct-method value from row-level target-policy reward predictions `[N]`."""
    return float(_as_float_array(direct).mean())


def ipw(reward, pi_logged, pscore, *, eps: float = 1e-8) -> float:
    """Inverse probability weighting / IPS value."""
    w = importance_weight(pi_logged, pscore, eps=eps)
    return float(np.mean(w * _as_float_array(reward)))


def snipw(reward, pi_logged, pscore, *, eps: float = 1e-8) -> float:
    """Self-normalized inverse probability weighting / SNIPS value."""
    w = importance_weight(pi_logged, pscore, eps=eps)
    return float(np.sum(w * _as_float_array(reward)) / max(np.sum(w), eps))


def doubly_robust(direct, reward, q_logged, pi_logged, pscore, *, eps: float = 1e-8) -> float:
    """Doubly robust value using row-level direct terms and residual correction."""
    w = importance_weight(pi_logged, pscore, eps=eps)
    residual = _as_float_array(reward) - _as_float_array(q_logged)
    return float(np.mean(_as_float_array(direct) + w * residual))


def self_normalized_doubly_robust(
    direct,
    reward,
    q_logged,
    pi_logged,
    pscore,
    *,
    eps: float = 1e-8,
) -> float:
    """Self-normalized DR value: DM plus normalized residual correction."""
    w = importance_weight(pi_logged, pscore, eps=eps)
    residual = _as_float_array(reward) - _as_float_array(q_logged)
    return float(direct_method(direct) + np.sum(w * residual) / max(np.sum(w), eps))


def switch_dr(
    direct,
    reward,
    q_logged,
    pi_logged,
    pscore,
    *,
    switch_tau: float = 25.0,
    eps: float = 1e-8,
) -> float:
    """Switch-DR value: use DR residual only where weights are below `switch_tau`."""
    w = importance_weight(pi_logged, pscore, eps=eps)
    residual = _as_float_array(reward) - _as_float_array(q_logged)
    switched = np.where(w <= float(switch_tau), w * residual, 0.0)
    return float(np.mean(_as_float_array(direct) + switched))


def dros(
    direct,
    reward,
    q_logged,
    pi_logged,
    pscore,
    *,
    shrinkage_lambda: float = 1.0,
    eps: float = 1e-8,
) -> float:
    """Doubly robust with optimistic shrinkage-style residual weights.

    Uses the common DRos shrinkage shape `lambda * w / (w^2 + lambda)` to damp
    high-variance residual corrections while retaining the direct-method term.
    """
    w = importance_weight(pi_logged, pscore, eps=eps)
    lam = max(float(shrinkage_lambda), eps)
    shrink_w = (lam * w) / (np.square(w) + lam)
    residual = _as_float_array(reward) - _as_float_array(q_logged)
    return float(np.mean(_as_float_array(direct) + shrink_w * residual))


def estimate_all(
    rows: dict[str, np.ndarray],
    *,
    switch_tau: float = 25.0,
    dros_lambda: float = 1.0,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute DM/IPW/SNIPW/DR/SNDR/Switch-DR/DRos from row arrays."""
    return {
        "direct_method": direct_method(rows["direct"]),
        "ips": ipw(rows["reward"], rows["pi_logged"], rows["pscore"], eps=eps),
        "snips": snipw(rows["reward"], rows["pi_logged"], rows["pscore"], eps=eps),
        "doubly_robust": doubly_robust(
            rows["direct"], rows["reward"], rows["q_logged"], rows["pi_logged"], rows["pscore"], eps=eps
        ),
        "self_normalized_doubly_robust": self_normalized_doubly_robust(
            rows["direct"], rows["reward"], rows["q_logged"], rows["pi_logged"], rows["pscore"], eps=eps
        ),
        "switch_dr": switch_dr(
            rows["direct"],
            rows["reward"],
            rows["q_logged"],
            rows["pi_logged"],
            rows["pscore"],
            switch_tau=switch_tau,
            eps=eps,
        ),
        "dros": dros(
            rows["direct"],
            rows["reward"],
            rows["q_logged"],
            rows["pi_logged"],
            rows["pscore"],
            shrinkage_lambda=dros_lambda,
            eps=eps,
        ),
    }
