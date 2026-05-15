"""End-to-end Bandit-STOR training pipeline."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from bandit_stor.data.open_bandit import load_dataset_from_config
from bandit_stor.evaluation.baselines import (
    make_behavior_policy,
    make_reward_greedy_policy,
    make_reward_topk_sparse_policy,
    make_uniform_policy,
)
from bandit_stor.evaluation.crossfit import evaluate_policy_crossfit_reward
from bandit_stor.evaluation.diagnostics import evaluate_gates
from bandit_stor.evaluation.ope import evaluate_policy
from bandit_stor.evaluation.reward import evaluate_reward_model, fit_reward_calibration
from bandit_stor.evaluation.policy_report import write_policy_report
from bandit_stor.logging_utils import setup_run_logging
from bandit_stor.models.actor import SparseTsallisActor
from bandit_stor.models.behavior_policy import BehaviorPolicyModel
from bandit_stor.models.reward_model import RewardModel
from bandit_stor.training.train_actor import train_actor
from bandit_stor.training.train_behavior import train_behavior_model
from bandit_stor.training.train_reward import train_reward_model
from bandit_stor.utils import load_yaml, resolve_device, set_deterministic_seed


logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_configs(data_name: str) -> dict[str, Any]:
    root = _repo_root()
    config = load_yaml(root / "configs/config.yaml")
    data_cfg = load_yaml(root / f"configs/data/{data_name}.yaml")
    model_cfg = load_yaml(root / "configs/model/sparse_tsallis_actor.yaml")
    train_cfg = load_yaml(root / "configs/train/full_pipeline.yaml")
    eval_cfg = load_yaml(root / "configs/eval/ope.yaml")
    return {"root": root, "config": config, "data": data_cfg, "model": model_cfg, "train": train_cfg, "eval": eval_cfg}


def run_full_pipeline(data: str = "open_bandit", *, output_dir: str | Path | None = None) -> Path:
    """Train/evaluate Bandit-STOR and write artifacts.

    Args:
        data: Config name (`open_bandit` or `tiny_fixture`).
        output_dir: Optional output root; defaults to `outputs` from config.

    Returns:
        Path to `outputs/{run_id}`.
    """
    cfg = _load_configs(data)
    root: Path = cfg["root"]
    run_id = f"{cfg['config'].get('run_name', 'bandit_stor_mvp')}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_root = Path(output_dir or cfg["config"].get("output_dir", "outputs"))
    if not out_root.is_absolute():
        out_root = root / out_root
    run_dir = out_root / run_id
    log_path = setup_run_logging(run_dir)
    logger.info("Starting Bandit-STOR full pipeline: data=%s run_id=%s", data, run_id)
    logger.info("Run directory: %s", run_dir)
    logger.info("Log file: %s", log_path)
    set_deterministic_seed(int(cfg["config"].get("seed", 42)))
    device = resolve_device(cfg["config"].get("device", "auto"))
    logger.info("Resolved device=%s", device)
    dataset, splits = load_dataset_from_config(cfg["data"])
    logger.info(
        "Dataset splits: total=%s train=%s valid=%s test=%s",
        len(dataset),
        len(splits.train),
        len(splits.valid),
        len(splits.test),
    )
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    actor_cfg = model_cfg.get("actor", {})
    reward_cfg = model_cfg.get("reward_model", {})
    behavior_cfg = model_cfg.get("behavior_model", {})
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
    actor = SparseTsallisActor(
        dataset.context_dim,
        dataset.action_dim,
        hidden_dim=int(actor_cfg.get("hidden_dim", 128)),
        num_layers=int(actor_cfg.get("num_layers", 3)),
        dropout=float(actor_cfg.get("dropout", 0.1)),
        temperature=float(actor_cfg.get("temperature", 0.7)),
        top_k_before_sparsemax=actor_cfg.get("top_k_before_sparsemax"),
    )
    batch_size = int(train_cfg.get("batch_size", 512))
    if cfg["data"].get("name") == "open_bandit" and cfg["data"].get("behavior_policy") == "random":
        logger.info("Skipping behavior model training: Open Bandit random policy has known uniform μ(.|x)")
        behavior_metrics = {"behavior_policy_source": "known_uniform", "behavior_loss": None}
    else:
        logger.info("Training behavior model")
        behavior_metrics = train_behavior_model(
            behavior_model,
            splits.train,
            batch_size=batch_size,
            device=device,
            **train_cfg.get("behavior", {}),
        )
    logger.info("Training reward model")
    reward_metrics = train_reward_model(
        reward_model,
        splits.train,
        batch_size=batch_size,
        device=device,
        **train_cfg.get("reward", {}),
    )
    actor_train = dict(train_cfg.get("actor", {}))
    reward_calibration_metrics = fit_reward_calibration(
        reward_model,
        splits.valid if len(splits.valid) else splits.train,
        batch_size=batch_size,
        device=device,
        method=str(reward_cfg.get("calibration", "auto_logit_correction")),
        pos_weight=reward_metrics.get("reward_pos_weight"),
    )
    logger.info("Reward calibration: %s", reward_calibration_metrics)
    logger.info("Training sparse Tsallis actor")
    actor_metrics = train_actor(
        actor,
        reward_model,
        behavior_model,
        splits.train,
        batch_size=batch_size,
        device=device,
        epochs=int(actor_train.pop("epochs", 20)),
        learning_rate=float(actor_train.pop("learning_rate", 3e-4)),
        weight_decay=float(actor_train.pop("weight_decay", 1e-5)),
        gradient_clip_norm=float(actor_train.pop("gradient_clip_norm", 1.0)),
        alpha=float(actor_train.pop("alpha", 1.5)),
        beta_alpha=float(actor_train.pop("beta_alpha", 0.1)),
        lambda_tsallis=float(actor_train.pop("lambda_tsallis", 0.01)),
        lambda_support=float(actor_train.pop("lambda_support", 1.0)),
        mu_min=float(actor_train.pop("mu_min", 1e-4)),
        importance_clip=float(actor_train.pop("importance_clip", 20.0)),
        reward_value_scale=float(actor_train.pop("reward_value_scale", 1.0)),
        dataset_config=cfg["data"],
    )
    eval_dataset = splits.test if len(splits.test) else splits.valid if len(splits.valid) else splits.train
    logger.info("Evaluating reward model diagnostics on n=%s", len(eval_dataset))
    reward_diag_metrics = evaluate_reward_model(
        reward_model,
        eval_dataset,
        batch_size=batch_size,
        device=device,
        sanity_config=cfg["eval"].get("reward_sanity", {}),
    )
    logger.info("Evaluating target policy with OPE on n=%s", len(eval_dataset))
    robust_cfg = cfg["eval"].get("robust_estimators", {})
    ope_metrics = evaluate_policy(
        actor,
        reward_model,
        behavior_model,
        eval_dataset,
        batch_size=batch_size,
        device=device,
        alpha=float(train_cfg.get("actor", {}).get("alpha", 1.5)),
        mu_min=float(train_cfg.get("actor", {}).get("mu_min", 1e-4)),
        bootstrap_samples=int(cfg["eval"].get("confidence_intervals", {}).get("bootstrap_samples", 0))
        if cfg["eval"].get("confidence_intervals", {}).get("enabled", False)
        else 0,
        confidence_level=float(cfg["eval"].get("confidence_intervals", {}).get("confidence_level", 0.95)),
        dataset_config=cfg["data"],
        switch_tau=float(robust_cfg.get("switch_tau", 25.0)),
        dros_lambda=float(robust_cfg.get("dros_lambda", 1.0)),
        dros_lambdas=list(robust_cfg.get("dros_lambdas", [])),
    )
    crossfit_cfg = cfg["eval"].get("cross_fitting", {})
    if bool(crossfit_cfg.get("enabled", False)) and len(eval_dataset) >= 2:
        logger.info("Running cross-fitted reward OPE")
        crossfit_metrics = evaluate_policy_crossfit_reward(
            actor,
            reward_model,
            behavior_model,
            eval_dataset,
            dataset_context_dim=dataset.context_dim,
            dataset_action_dim=dataset.action_dim,
            reward_model_config=reward_cfg,
            reward_train_config=train_cfg.get("reward", {}),
            dataset_config=cfg["data"],
            folds=int(crossfit_cfg.get("folds", 2)),
            seed=int(crossfit_cfg.get("seed", cfg["config"].get("seed", 42))),
            batch_size=batch_size,
            device=device,
            alpha=float(train_cfg.get("actor", {}).get("alpha", 1.5)),
            mu_min=float(train_cfg.get("actor", {}).get("mu_min", 1e-4)),
            switch_tau=float(robust_cfg.get("switch_tau", 25.0)),
            dros_lambda=float(robust_cfg.get("dros_lambda", 1.0)),
            dros_lambdas=list(robust_cfg.get("dros_lambdas", [])),
        )
        ope_metrics.update(crossfit_metrics)
    else:
        ope_metrics.update({"crossfit_enabled": False, "crossfit_folds": 0})
    baseline_metrics = {}
    baseline_specs = {
        "behavior": make_behavior_policy(cfg["data"], behavior_model),
        "uniform": make_uniform_policy(),
        "reward_greedy_top1": make_reward_greedy_policy(reward_model),
        "reward_topk_sparse": make_reward_topk_sparse_policy(
            reward_model, top_k=int(actor_cfg.get("top_k_before_sparsemax") or 20)
        ),
        "sparse_tsallis_actor": actor,
    }
    for baseline_name, policy in baseline_specs.items():
        baseline_metrics[baseline_name] = evaluate_policy(
            policy,
            reward_model,
            behavior_model,
            eval_dataset,
            batch_size=batch_size,
            device=device,
            alpha=float(train_cfg.get("actor", {}).get("alpha", 1.5)),
            mu_min=float(train_cfg.get("actor", {}).get("mu_min", 1e-4)),
            bootstrap_samples=0,
            dataset_config=cfg["data"],
            switch_tau=float(cfg["eval"].get("robust_estimators", {}).get("switch_tau", 25.0)),
            dros_lambda=float(cfg["eval"].get("robust_estimators", {}).get("dros_lambda", 1.0)),
            dros_lambdas=list(cfg["eval"].get("robust_estimators", {}).get("dros_lambdas", [])),
        )

    gate_metrics = {**ope_metrics, **reward_diag_metrics}
    gates = evaluate_gates(gate_metrics, cfg["eval"])
    logger.info("Gate evaluation passed=%s", gates.get("passed"))
    run_dir.mkdir(parents=True, exist_ok=True)
    training_metrics = {**behavior_metrics, **reward_metrics, **reward_calibration_metrics, **reward_diag_metrics, **actor_metrics}
    dataset_summary = {
        "data": data,
        "n_total": len(dataset),
        "n_train": len(splits.train),
        "n_valid": len(splits.valid),
        "n_test": len(splits.test),
        "context_dim": dataset.context_dim,
        "action_dim": dataset.action_dim,
        "n_actions": dataset.n_actions,
        "device": str(device),
    }
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "context_dim": dataset.context_dim,
            "action_dim": dataset.action_dim,
            "n_actions": dataset.n_actions,
            "actor_config": actor_cfg,
        },
        run_dir / "actor.pt",
    )
    (run_dir / "metrics.json").write_text(json.dumps({"training": training_metrics, "dataset": dataset_summary}, indent=2), encoding="utf-8")
    (run_dir / "ope_report.json").write_text(json.dumps({"metrics": ope_metrics, "gates": gates, "baselines": baseline_metrics}, indent=2), encoding="utf-8")
    (run_dir / "baseline_report.json").write_text(json.dumps(baseline_metrics, indent=2), encoding="utf-8")
    write_policy_report(
        run_dir / "policy_report.md",
        dataset_summary=dataset_summary,
        training_metrics=training_metrics,
        ope_metrics=ope_metrics,
        gates=gates,
        baseline_metrics=baseline_metrics,
    )
    logger.info("Pipeline complete. Artifacts written to %s", run_dir)
    return run_dir
