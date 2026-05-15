"""Reward-model calibration and diagnostics for rare-event logged bandit data."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from bandit_stor.data.collate import collate_logged_interactions
from bandit_stor.data.schema import gather_logged_values
from bandit_stor.models.reward_model import RewardModel


logger = logging.getLogger(__name__)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10) -> float:
    """Compute binary ECE from labels `[N]` and probabilities `[N]`."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= left) & (y_prob <= right if right == 1.0 else y_prob < right)
        if mask.any():
            ece += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return ece


def collect_logged_reward_outputs(
    model: RewardModel,
    dataset,
    *,
    batch_size: int = 512,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Collect logged-action raw logits and labels."""
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_logged_interactions)
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            raw_logits = model.logits(batch.context, batch.action_context)
            logits.append(gather_logged_values(raw_logits, batch.logged_action_index).cpu())
            labels.append(batch.reward.cpu())
    return torch.cat(logits).numpy().astype(float), torch.cat(labels).numpy().astype(float)


def fit_reward_calibration(
    model: RewardModel,
    dataset,
    *,
    batch_size: int = 512,
    device: torch.device | str = "cpu",
    method: str = "auto_logit_correction",
    pos_weight: float | None = None,
) -> dict[str, float | str]:
    """Fit or assign calibration used by `RewardModel.forward`.

    Methods:
    - `none`: sigmoid(raw_logit)
    - `auto_logit_correction`: sigmoid(raw_logit - log(pos_weight))
    - `platt`: sigmoid(a * raw_logit + b) fitted on validation labels
    """
    if method == "none":
        model.set_calibration(method="none", a=1.0, b=0.0)
        return {"reward_calibration_method": "none", "reward_calibration_a": 1.0, "reward_calibration_b": 0.0}
    if method == "auto_logit_correction":
        shift = float(np.log(max(float(pos_weight or 1.0), 1e-8)))
        model.set_calibration(method=method, a=1.0, b=-shift)
        return {"reward_calibration_method": method, "reward_calibration_a": 1.0, "reward_calibration_b": -shift}
    if method == "platt":
        logits, labels = collect_logged_reward_outputs(model, dataset, batch_size=batch_size, device=device)
        if len(np.unique(labels)) < 2:
            model.set_calibration(method="none", a=1.0, b=0.0)
            return {"reward_calibration_method": "none_single_class", "reward_calibration_a": 1.0, "reward_calibration_b": 0.0}
        clf = LogisticRegression(solver="lbfgs")
        clf.fit(logits.reshape(-1, 1), labels.astype(int))
        a = float(clf.coef_[0, 0])
        b = float(clf.intercept_[0])
        model.set_calibration(method=method, a=a, b=b)
        return {"reward_calibration_method": method, "reward_calibration_a": a, "reward_calibration_b": b}
    raise ValueError(f"Unsupported reward calibration method: {method}")


def compute_reward_sanity(metrics: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, bool]:
    """Reward-model sanity checks for calibrated probabilities."""
    cfg = config or {}
    base = float(metrics.get("reward_positive_rate", 0.0))
    pred_mean = float(metrics.get("reward_prediction_mean", 0.0))
    baseline_log_loss = float(metrics.get("reward_constant_baseline_log_loss", float("inf")))
    log_loss_value = float(metrics.get("reward_log_loss", float("inf")))
    auc_pr = metrics.get("reward_auc_pr")
    auc_pr_lift = None if auc_pr is None else float(auc_pr) / max(base, 1e-12)
    auc = metrics.get("reward_auc")
    top_decile_lift = float(metrics.get("reward_top_decile_lift", 0.0))
    checks = {
        "prediction_mean_not_too_high": pred_mean <= float(cfg.get("prediction_mean_max_multiple_of_base_rate", 5.0)) * max(base, 1e-12),
        "prediction_mean_not_too_low": pred_mean >= float(cfg.get("prediction_mean_min_multiple_of_base_rate", 0.2)) * max(base, 1e-12),
        "log_loss_close_to_constant": log_loss_value <= baseline_log_loss + float(cfg.get("log_loss_must_not_exceed_constant_by_more_than", 0.005)),
        "ece_below_max": float(metrics.get("reward_ece_10bin", float("inf"))) <= float(cfg.get("ece_max", 0.01)),
        "auc_pr_lift_min": (auc_pr_lift is not None) and auc_pr_lift >= float(cfg.get("auc_pr_lift_min", 1.05)),
        "auc_roc_min": (auc is not None) and float(auc) >= float(cfg.get("auc_roc_min", 0.60)),
        "top_decile_lift_min": top_decile_lift > float(cfg.get("top_decile_lift_min", 1.0)),
    }
    checks["passed"] = bool(all(checks.values()))
    return checks


def evaluate_reward_model(
    model: RewardModel,
    dataset,
    *,
    batch_size: int = 512,
    device: torch.device | str = "cpu",
    threshold: float = 0.5,
    sanity_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate calibrated reward predictions only on logged actions."""
    logits, y_true = collect_logged_reward_outputs(model, dataset, batch_size=batch_size, device=device)
    y_prob = 1.0 / (1.0 + np.exp(-(model.calibration_a * logits + model.calibration_b)))
    y_prob = np.clip(y_prob, 1e-8, 1.0 - 1e-8)
    baseline_prob = float(np.clip(y_true.mean(), 1e-8, 1.0 - 1e-8))
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    base_rate = float(y_true.mean())
    top_decile_n = max(1, int(np.ceil(0.1 * len(y_prob))))
    top_1_n = max(1, int(np.ceil(0.01 * len(y_prob))))
    top_5_n = max(1, int(np.ceil(0.05 * len(y_prob))))
    sorted_idx = np.argsort(y_prob)

    def capture_at(n: int) -> float:
        positives = y_true.sum()
        return 0.0 if positives <= 0 else float(y_true[sorted_idx[-n:]].sum() / positives)

    top_decile_idx = sorted_idx[-top_decile_n:]
    top_decile_rate = float(y_true[top_decile_idx].mean())
    metrics: dict[str, Any] = {
        "reward_positive_rate": base_rate,
        "reward_prediction_mean": float(y_prob.mean()),
        "reward_prediction_std": float(y_prob.std()),
        "reward_prediction_min": float(y_prob.min()),
        "reward_prediction_max": float(y_prob.max()),
        "reward_prediction_p10": float(np.quantile(y_prob, 0.10)),
        "reward_prediction_p50": float(np.quantile(y_prob, 0.50)),
        "reward_prediction_p90": float(np.quantile(y_prob, 0.90)),
        "reward_prediction_top_decile_mean": float(y_prob[top_decile_idx].mean()),
        "reward_top_decile_positive_rate": top_decile_rate,
        "reward_top_decile_lift": top_decile_rate / max(base_rate, 1e-12),
        "reward_top_1pct_capture": capture_at(top_1_n),
        "reward_top_5pct_capture": capture_at(top_5_n),
        "reward_top_10pct_capture": capture_at(top_decile_n),
        "reward_brier_score": float(np.mean((y_prob - y_true) ** 2)),
        "reward_log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "reward_constant_baseline_log_loss": float(log_loss(y_true, np.full_like(y_prob, baseline_prob), labels=[0, 1])),
        "reward_ece_10bin": expected_calibration_error(y_true, y_prob, n_bins=10),
        "reward_precision_at_0_5": float(precision),
        "reward_recall_at_0_5": float(recall),
    }
    try:
        metrics["reward_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["reward_auc"] = None
    try:
        metrics["reward_auc_pr"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["reward_auc_pr"] = None
    metrics["reward_sanity"] = compute_reward_sanity(metrics, sanity_config)
    metrics["reward_sanity_passed"] = bool(metrics["reward_sanity"]["passed"])
    logger.info("Reward diagnostics complete: %s", metrics)
    return metrics
