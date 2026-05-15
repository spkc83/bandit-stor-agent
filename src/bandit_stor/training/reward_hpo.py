"""Reward-model-only calibration search for Bandit-STOR."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna

from bandit_stor.data.open_bandit import load_dataset_from_config
from bandit_stor.evaluation.reward import evaluate_reward_model, fit_reward_calibration
from bandit_stor.logging_utils import setup_run_logging
from bandit_stor.models.reward_model import RewardModel
from bandit_stor.training.full_pipeline import _load_configs, _repo_root
from bandit_stor.training.hpo import _bounded_subset
from bandit_stor.training.train_reward import train_reward_model
from bandit_stor.utils import resolve_device, set_deterministic_seed


logger = logging.getLogger(__name__)


def _reward_score(metrics: dict[str, Any]) -> float:
    """Reward-calibration score for phase-1 reward-only sweeps."""
    base_rate = float(metrics.get("reward_positive_rate", 0.0))
    auc_pr = metrics.get("reward_auc_pr")
    auc_pr_lift = 0.0 if auc_pr is None else float(auc_pr) / max(base_rate, 1e-12)
    top_decile_lift = float(metrics.get("reward_top_decile_lift", 0.0))
    log_loss_excess = max(
        0.0,
        float(metrics.get("reward_log_loss", 1e9))
        - float(metrics.get("reward_constant_baseline_log_loss", 0.0)),
    )
    ece = float(metrics.get("reward_ece_10bin", 1.0))
    pred_mean = float(metrics.get("reward_prediction_mean", 0.0))
    mean_ratio_error = abs(pred_mean / max(base_rate, 1e-12) - 1.0)
    sanity_bonus = 1.0 if metrics.get("reward_sanity_passed") else 0.0
    return float(
        sanity_bonus
        + 0.25 * auc_pr_lift
        + 0.10 * top_decile_lift
        - 20.0 * log_loss_excess
        - 10.0 * ece
        - 0.05 * mean_ratio_error
    )


def run_reward_hpo(
    data: str = "open_bandit",
    *,
    output_dir: str | Path | None = None,
    n_trials: int = 12,
    max_train_rows: int | None = 100000,
    max_valid_rows: int | None = 50000,
) -> Path:
    """Run phase-1 reward-model calibration search and write a ranked report."""
    root = _repo_root()
    cfg = _load_configs(data)
    run_id = f"reward-hpo-{data}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_root = Path(output_dir or root / "outputs" / "reward_hpo")
    if not out_root.is_absolute():
        out_root = root / out_root
    run_dir = out_root / run_id
    setup_run_logging(run_dir)
    set_deterministic_seed(int(cfg["config"].get("seed", 42)))
    device = resolve_device(cfg["config"].get("device", "auto"))
    dataset, splits = load_dataset_from_config(cfg["data"])
    train_split = _bounded_subset(splits.train, max_train_rows)
    valid_split = _bounded_subset(splits.valid if len(splits.valid) else splits.test, max_valid_rows)
    logger.info("Reward HPO data: train=%s valid=%s", len(train_split), len(valid_split))

    model_cfg = cfg["model"].get("reward_model", {})
    train_cfg = cfg["train"]
    batch_size = int(train_cfg.get("batch_size", 512))
    sanity_cfg = cfg["eval"].get("reward_sanity", {})

    def objective(trial: optuna.Trial) -> float:
        set_deterministic_seed(int(cfg["config"].get("seed", 42)) + trial.number)
        loss_type = trial.suggest_categorical("loss_type", ["bce", "weighted_bce", "focal"])
        calibration = trial.suggest_categorical("calibration", ["auto_logit_correction", "platt"])
        neg_ratio = trial.suggest_categorical("negative_downsample_ratio", [None, 0.05, 0.1, 0.2])
        lr = trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True)
        epochs = trial.suggest_categorical("epochs", [5, 10, 15])
        model = RewardModel(
            dataset.context_dim,
            dataset.action_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_layers=int(model_cfg.get("num_layers", 3)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        train_metrics = train_reward_model(
            model,
            train_split,
            batch_size=batch_size,
            device=device,
            epochs=epochs,
            learning_rate=lr,
            weight_decay=float(train_cfg.get("reward", {}).get("weight_decay", 1e-5)),
            loss_type=loss_type,
            pos_weight="auto" if loss_type == "weighted_bce" else None,
            focal_alpha=float(train_cfg.get("reward", {}).get("focal_alpha", 0.25)),
            focal_gamma=float(train_cfg.get("reward", {}).get("focal_gamma", 2.0)),
            negative_downsample_ratio=neg_ratio,
        )
        cal_metrics = fit_reward_calibration(
            model,
            valid_split,
            batch_size=batch_size,
            device=device,
            method=calibration if loss_type == "weighted_bce" else "platt",
            pos_weight=train_metrics.get("reward_pos_weight"),
        )
        eval_metrics = evaluate_reward_model(
            model,
            valid_split,
            batch_size=batch_size,
            device=device,
            sanity_config=sanity_cfg,
        )
        score = _reward_score(eval_metrics)
        trial.set_user_attr("score", score)
        trial.set_user_attr("training_metrics", {**train_metrics, **cal_metrics})
        trial.set_user_attr("reward_diagnostics", eval_metrics)
        return score

    sampler = optuna.samplers.TPESampler(seed=int(cfg["config"].get("seed", 42)))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=int(n_trials), n_jobs=1, gc_after_trial=True)
    trials = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": t.state.name,
            "user_attrs": t.user_attrs,
        }
        for t in study.trials
    ]
    completed = [t for t in trials if t["state"] == "COMPLETE"]
    ranked = sorted(completed, key=lambda t: float(t["value"] or -1e9), reverse=True)
    report = {
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "ranked_trials": ranked,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reward_hpo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run_dir / "reward_hpo_trials.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")
    logger.info("Reward HPO complete: %s", run_dir)
    return run_dir
