"""Optuna hyperparameter search for Bandit-STOR actor training."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
from torch.utils.data import Subset

from bandit_stor.data.open_bandit import load_dataset_from_config
from bandit_stor.evaluation.diagnostics import evaluate_gates
from bandit_stor.evaluation.ope import evaluate_policy
from bandit_stor.evaluation.reward import evaluate_reward_model, fit_reward_calibration
from bandit_stor.evaluation.scoring import compute_policy_selection_score
from bandit_stor.logging_utils import setup_run_logging
from bandit_stor.models.actor import SparseTsallisActor
from bandit_stor.models.behavior_policy import BehaviorPolicyModel
from bandit_stor.models.reward_model import RewardModel
from bandit_stor.training.full_pipeline import _load_configs, _repo_root
from bandit_stor.training.train_actor import train_actor
from bandit_stor.training.train_behavior import train_behavior_model
from bandit_stor.training.train_reward import train_reward_model
from bandit_stor.utils import load_yaml, resolve_device, set_deterministic_seed


logger = logging.getLogger(__name__)


def _bounded_subset(dataset, max_rows: int | None):
    """Return at most `max_rows` deterministic leading rows from a split."""
    if max_rows is None or len(dataset) <= max_rows:
        return dataset
    return Subset(dataset, list(range(int(max_rows))))


def _suggest_float(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> float:
    return trial.suggest_float(
        name,
        float(spec["low"]),
        float(spec["high"]),
        log=bool(spec.get("log", False)),
    )


def _suggest_categorical(trial: optuna.Trial, name: str, spec: dict[str, Any]):
    return trial.suggest_categorical(name, list(spec["choices"]))


def _suggest_from_spec(trial: optuna.Trial, name: str, spec: dict[str, Any]):
    if "choices" in spec:
        return _suggest_categorical(trial, name, spec)
    return _suggest_float(trial, name, spec)


def _trial_params(trial: optuna.Trial, hpo_cfg: dict[str, Any]) -> dict[str, Any]:
    space = hpo_cfg.get("search_space", {})
    return {
        "actor_learning_rate": _suggest_from_spec(trial, "actor_learning_rate", space["actor_learning_rate"]),
        "actor_weight_decay": _suggest_from_spec(trial, "actor_weight_decay", space["actor_weight_decay"]),
        "temperature": _suggest_from_spec(trial, "temperature", space["temperature"]),
        "top_k_before_sparsemax": _suggest_from_spec(
            trial, "top_k_before_sparsemax", space["top_k_before_sparsemax"]
        ),
        "beta_alpha": _suggest_from_spec(trial, "beta_alpha", space["beta_alpha"]),
        "lambda_tsallis": _suggest_from_spec(trial, "lambda_tsallis", space["lambda_tsallis"]),
        "lambda_support": _suggest_from_spec(trial, "lambda_support", space["lambda_support"]),
        "actor_epochs": _suggest_from_spec(trial, "actor_epochs", space["actor_epochs"]),
        "reward_value_scale": _suggest_from_spec(trial, "reward_value_scale", space["reward_value_scale"]),
    }


def optuna_objective_score(metrics: dict[str, float], hpo_cfg: dict[str, Any]) -> float:
    """Risk-adjusted reward-lift score used by Optuna."""
    return compute_policy_selection_score(metrics, {}, hpo_cfg)


def _apply_mode_overrides(hpo_cfg: dict[str, Any], mode: str | None) -> dict[str, Any]:
    """Apply named reward/risk mode overrides to HPO gates."""
    selected = mode or hpo_cfg.get("mode")
    modes = hpo_cfg.get("modes", {})
    if selected in modes:
        gates = hpo_cfg.setdefault("gates", {})
        for section, values in modes[selected].items():
            if isinstance(values, dict):
                gates.setdefault(section, {}).update(values)
        hpo_cfg["mode"] = selected
    return hpo_cfg


def run_optuna_hpo(
    data: str = "open_bandit",
    *,
    hpo_config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    n_trials: int | None = None,
    mode: str | None = None,
    validation_size: int | None = None,
) -> Path:
    """Run Optuna HPO and write best hyperparameters/report artifacts."""
    root = _repo_root()
    cfg = _load_configs(data)
    hpo_cfg = _apply_mode_overrides(load_yaml(hpo_config_path or root / "configs/hpo_optuna.yaml"), mode)
    if validation_size is not None:
        hpo_cfg["max_valid_rows"] = int(validation_size)
    run_id = f"optuna-{data}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_root = Path(output_dir or root / "outputs" / "optuna")
    if not out_root.is_absolute():
        out_root = root / out_root
    run_dir = out_root / run_id
    setup_run_logging(run_dir)
    logger.info("Starting Optuna HPO: data=%s run_dir=%s", data, run_dir)

    set_deterministic_seed(int(cfg["config"].get("seed", 42)))
    device = resolve_device(cfg["config"].get("device", "auto"))
    dataset, splits = load_dataset_from_config(cfg["data"])
    train_split = _bounded_subset(splits.train, hpo_cfg.get("max_train_rows"))
    valid_split = _bounded_subset(splits.valid if len(splits.valid) else splits.test, hpo_cfg.get("max_valid_rows"))
    logger.info(
        "HPO data: total=%s train=%s valid=%s context_dim=%s action_dim=%s n_actions=%s",
        len(dataset),
        len(train_split),
        len(valid_split),
        dataset.context_dim,
        dataset.action_dim,
        dataset.n_actions,
    )

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    batch_size = int(train_cfg.get("batch_size", 512))
    eval_cfg = cfg["eval"]
    reward_cfg = model_cfg.get("reward_model", {})
    behavior_cfg = model_cfg.get("behavior_model", {})

    logger.info("Training fixed offline reward/behavior components once for actor HPO")
    reward_model = RewardModel(
        dataset.context_dim,
        dataset.action_dim,
        hidden_dim=int(reward_cfg.get("hidden_dim", 128)),
        num_layers=int(reward_cfg.get("num_layers", 3)),
        dropout=float(reward_cfg.get("dropout", 0.1)),
    )
    behavior_model = BehaviorPolicyModel(
        dataset.context_dim,
        dataset.action_dim,
        hidden_dim=int(behavior_cfg.get("hidden_dim", 128)),
        num_layers=int(behavior_cfg.get("num_layers", 2)),
        dropout=float(behavior_cfg.get("dropout", 0.1)),
    )
    if cfg["data"].get("name") == "open_bandit" and cfg["data"].get("behavior_policy") == "random":
        behavior_metrics = {"behavior_policy_source": "known_uniform", "behavior_loss": None}
    else:
        behavior_metrics = train_behavior_model(
            behavior_model,
            train_split,
            batch_size=batch_size,
            device=device,
            **train_cfg.get("behavior", {}),
        )
    reward_metrics = train_reward_model(
        reward_model,
        train_split,
        batch_size=batch_size,
        device=device,
        **train_cfg.get("reward", {}),
    )
    reward_calibration_metrics = fit_reward_calibration(
        reward_model,
        valid_split,
        batch_size=batch_size,
        device=device,
        method=str(reward_cfg.get("calibration", "auto_logit_correction")),
        pos_weight=reward_metrics.get("reward_pos_weight"),
    )
    reward_diag = evaluate_reward_model(
        reward_model,
        valid_split,
        batch_size=batch_size,
        device=device,
        sanity_config=eval_cfg.get("reward_sanity", hpo_cfg.get("reward_sanity", {})),
    )
    fixed_training_metrics = {**behavior_metrics, **reward_metrics, **reward_calibration_metrics}
    logger.info("Fixed reward diagnostics for actor HPO: %s", reward_diag)

    def objective(trial: optuna.Trial) -> float:
        trial_seed = int(cfg["config"].get("seed", 42)) + trial.number
        set_deterministic_seed(trial_seed)
        params = _trial_params(trial, hpo_cfg)
        logger.info("Trial %s params: %s", trial.number, params)

        actor_cfg = dict(model_cfg.get("actor", {}))
        actor = SparseTsallisActor(
            dataset.context_dim,
            dataset.action_dim,
            hidden_dim=int(actor_cfg.get("hidden_dim", 128)),
            num_layers=int(actor_cfg.get("num_layers", 3)),
            dropout=float(actor_cfg.get("dropout", 0.1)),
            temperature=float(params["temperature"]),
            top_k_before_sparsemax=int(params["top_k_before_sparsemax"]),
        )
        actor_train = dict(train_cfg.get("actor", {}))
        actor_metrics = train_actor(
            actor,
            reward_model,
            behavior_model,
            train_split,
            batch_size=batch_size,
            device=device,
            epochs=int(params["actor_epochs"]),
            learning_rate=float(params["actor_learning_rate"]),
            weight_decay=float(params["actor_weight_decay"]),
            gradient_clip_norm=float(actor_train.get("gradient_clip_norm", 1.0)),
            alpha=float(actor_train.get("alpha", 1.5)),
            beta_alpha=float(params["beta_alpha"]),
            lambda_tsallis=float(params["lambda_tsallis"]),
            lambda_support=float(params["lambda_support"]),
            mu_min=float(actor_train.get("mu_min", 1e-4)),
            importance_clip=float(actor_train.get("importance_clip", 20.0)),
            reward_value_scale=float(params["reward_value_scale"]),
            dataset_config=cfg["data"],
        )
        ope_metrics = evaluate_policy(
            actor,
            reward_model,
            behavior_model,
            valid_split,
            batch_size=batch_size,
            device=device,
            alpha=float(actor_train.get("alpha", 1.5)),
            mu_min=float(actor_train.get("mu_min", 1e-4)),
            bootstrap_samples=0,
            dataset_config=cfg["data"],
            switch_tau=float(eval_cfg.get("robust_estimators", {}).get("switch_tau", 25.0)),
            dros_lambda=float(eval_cfg.get("robust_estimators", {}).get("dros_lambda", 1.0)),
        )
        gate_metrics = {**ope_metrics, **reward_diag}
        gates = evaluate_gates(gate_metrics, eval_cfg)
        score = optuna_objective_score({**ope_metrics, "reward_sanity_passed": reward_diag.get("reward_sanity_passed", False)}, hpo_cfg)
        trial.set_user_attr("score", score)
        trial.set_user_attr("ope_metrics", ope_metrics)
        trial.set_user_attr("reward_diagnostics", reward_diag)
        trial.set_user_attr("gates", gates)
        trial.set_user_attr("training_metrics", {**fixed_training_metrics, **actor_metrics})
        logger.info("Trial %s score=%s metrics=%s gates=%s", trial.number, score, ope_metrics, gates)
        return score

    storage = hpo_cfg.get("storage")
    if isinstance(storage, str) and storage.startswith("sqlite:///outputs/"):
        (root / "outputs" / "optuna").mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=int(hpo_cfg.get("sampler_seed", 42)))
    base_study_name = str(hpo_cfg.get("study_name", "bandit_stor_actor_hpo"))
    study_name = f"{base_study_name}_{data}_{hpo_cfg.get('mode', 'default')}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=str(hpo_cfg.get("direction", "maximize")),
        sampler=sampler,
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=int(n_trials or hpo_cfg.get("n_trials", 20)),
        timeout=hpo_cfg.get("timeout"),
        n_jobs=1,
        gc_after_trial=True,
    )
    best = study.best_trial
    summary = {
        "best_trial_number": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "n_trials": len(study.trials),
    }
    (run_dir / "optuna_best.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    trial_rows = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": t.state.name,
            "user_attrs": t.user_attrs,
        }
        for t in study.trials
    ]
    (run_dir / "optuna_trials.json").write_text(json.dumps(trial_rows, indent=2), encoding="utf-8")

    completed = [r for r in trial_rows if r["state"] == "COMPLETE" and r["user_attrs"].get("ope_metrics")]

    def raw_lift(row: dict[str, Any]) -> float:
        return float(row["user_attrs"]["ope_metrics"].get("absolute_dr_lift", row["user_attrs"]["ope_metrics"].get("dr_lift_over_behavior", -1e9)))

    def readiness(row: dict[str, Any]) -> tuple[int, float, float]:
        checks = row["user_attrs"].get("gates", {}).get("checks", {})
        passed_count = sum(bool(v) for v in checks.values())
        return (passed_count, float(row.get("value") or -1e9), raw_lift(row))

    rankings = {
        "raw_dr_lift": sorted(completed, key=raw_lift, reverse=True),
        "risk_adjusted_lift": sorted(completed, key=lambda r: float(r.get("value") or -1e9), reverse=True),
        "deployment_conservative_readiness": sorted(completed, key=readiness, reverse=True),
    }
    (run_dir / "rankings.json").write_text(json.dumps(rankings, indent=2), encoding="utf-8")
    logger.info("Optuna HPO complete. Best trial=%s value=%s", best.number, best.value)
    return run_dir
