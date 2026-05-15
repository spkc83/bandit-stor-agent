"""Bootstrap confidence intervals for OPE estimates."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def bootstrap_ope_ci(
    rows: dict[str, np.ndarray],
    estimator_fn: Callable[[dict[str, np.ndarray]], dict[str, float]],
    n_bootstrap: int = 500,
    seed: int = 42,
    confidence: tuple[float, float] = (0.05, 0.95),
) -> dict[str, Any]:
    """Resample rows and return percentile intervals for IPS/SNIPS/DR/DR lift.

    Args:
        rows: Equal-length row arrays consumed by `estimator_fn`.
        estimator_fn: Function returning metric dict for a row sample.
        n_bootstrap: Number of bootstrap resamples.
        seed: Deterministic RNG seed.
        confidence: Lower/upper quantiles, default p05/p95.
    """
    if not rows:
        return {}
    n = len(next(iter(rows.values())))
    if n < 2 or n_bootstrap <= 0:
        return {}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        sampled = {k: np.asarray(v)[idx] for k, v in rows.items()}
        metrics = estimator_fn(sampled)
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                samples.setdefault(key, []).append(float(value))
    lo_q, hi_q = confidence
    out: dict[str, Any] = {}
    for key, values in samples.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        out[f"{key}_p05"] = float(np.quantile(arr, lo_q))
        out[f"{key}_p50"] = float(np.quantile(arr, 0.50))
        out[f"{key}_p95"] = float(np.quantile(arr, hi_q))
    return out
