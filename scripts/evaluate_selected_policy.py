"""Train/evaluate a frozen Bandit-STOR actor configuration on fixed splits."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from bandit_stor.data.open_bandit import load_dataset_from_config  # noqa: E402
from bandit_stor.evaluation.diagnostics import evaluate_gates  # noqa: E402
from bandit_stor.evaluation.ope import evaluate_policy  # noqa: E402
from bandit_stor.evaluation.reward import evaluate_reward_model, fit_reward_calibration  # noqa: E402
from bandit_stor.logging_utils import setup_run_logging  # noqa: E402
from bandit_stor.models.actor import SparseTsallisActor  # noqa: E402
from bandit_stor.models.behavior_policy import BehaviorPolicyModel  # noqa: E402
from bandit_stor.models.reward_model import RewardModel  # noqa: E402
from bandit_stor.training.full_pipeline import _load_configs  # noqa: E402
from bandit_stor.training.hpo import _bounded_subset  # noqa: E402
from bandit_stor.training.train_actor import train_actor  # noqa: E402
from bandit_stor.training.train_behavior import train_behavior_model  # noqa: E402
from bandit_stor.training.train_reward import train_reward_model  # noqa: E402
from bandit_stor.utils import load_yaml, resolve_device, set_deterministic_seed  # noqa: E402


def _load_champion_params(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data.get("best_params", data))


def _split_map(splits) -> dict[str, Any]:
    return {"train": splits.train, "valid": splits.valid, "test": splits.test}


def evaluate_frozen_policy(
    *,
    data: str,
    optuna_best: Path,
    output_dir: Path | None,
    split: str,
    bootstrap: int,
    use_hpo_row_limits: bool,
) -> Path:
    cfg = _load_configs(data)
    hpo_cfg = load_yaml(ROOT / "configs/hpo_optuna.yaml")
    params = _load_champion_params(optuna_best)
    run_id = f"selected-policy-{data}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = (output_dir or ROOT / "outputs" / "selected_policy") / run_id
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    setup_run_logging(run_dir)

    set_deterministic_seed(int(cfg["config"].get("seed", 42)))
    device = resolve_device(cfg["config"].get("device", "auto"))
    dataset, splits = load_dataset_from_config(cfg["data"])
    train_split = _bounded_subset(splits.train, hpo_cfg.get("max_train_rows") if use_hpo_row_limits else None)
    calibration_split = _bounded_subset(
        splits.valid if len(splits.valid) else splits.test,
        hpo_cfg.get("max_valid_rows") if use_hpo_row_limits else None,
    )
    split_datasets = _split_map(splits)
    if use_hpo_row_limits:
        split_datasets["valid_hpo"] = calibration_split
    selected_splits = list(split_datasets) if split == "all" else [split]

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]
    batch_size = int(train_cfg.get("batch_size", 512))
    reward_cfg = model_cfg.get("reward_model", {})
    behavior_cfg = model_cfg.get("behavior_model", {})
    actor_cfg = dict(model_cfg.get("actor", {}))

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
        temperature=float(params["temperature"]),
        top_k_before_sparsemax=int(params["top_k_before_sparsemax"]),
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
    calibration_metrics = fit_reward_calibration(
        reward_model,
        calibration_split,
        batch_size=batch_size,
        device=device,
        method=str(reward_cfg.get("calibration", "platt")),
        pos_weight=reward_metrics.get("reward_pos_weight"),
    )
    reward_diagnostics = evaluate_reward_model(
        reward_model,
        calibration_split,
        batch_size=batch_size,
        device=device,
        sanity_config=eval_cfg.get("reward_sanity", {}),
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

    evaluations: dict[str, Any] = {}
    for split_name in selected_splits:
        ds = split_datasets[split_name]
        if len(ds) == 0:
            continue
        split_bootstrap = int(bootstrap) if split_name in {"valid", "valid_hpo", "test"} else 0
        metrics = evaluate_policy(
            actor,
            reward_model,
            behavior_model,
            ds,
            batch_size=batch_size,
            device=device,
            alpha=float(actor_train.get("alpha", 1.5)),
            mu_min=float(actor_train.get("mu_min", 1e-4)),
            bootstrap_samples=split_bootstrap,
            dataset_config=cfg["data"],
            switch_tau=float(eval_cfg.get("robust_estimators", {}).get("switch_tau", 25.0)),
            dros_lambda=float(eval_cfg.get("robust_estimators", {}).get("dros_lambda", 1.0)),
        )
        evaluations[split_name] = {"metrics": metrics, "gates": evaluate_gates(metrics, eval_cfg)}

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "context_dim": dataset.context_dim,
            "action_dim": dataset.action_dim,
            "n_actions": dataset.n_actions,
            "best_params": params,
        },
        run_dir / "actor.pt",
    )
    report = {
        "data": data,
        "source_optuna_best": str(optuna_best),
        "use_hpo_row_limits": use_hpo_row_limits,
        "best_params": params,
        "dataset": {
            "n_total": len(dataset),
            "n_train_used": len(train_split),
            "n_calibration_used": len(calibration_split),
            "n_train": len(splits.train),
            "n_valid": len(splits.valid),
            "n_test": len(splits.test),
            "context_dim": dataset.context_dim,
            "action_dim": dataset.action_dim,
            "n_actions": dataset.n_actions,
            "device": str(device),
        },
        "training": {**behavior_metrics, **reward_metrics, **calibration_metrics, **actor_metrics},
        "reward_diagnostics": reward_diagnostics,
        "evaluations": evaluations,
    }
    (run_dir / "selected_policy_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen Bandit-STOR actor config")
    parser.add_argument("--checkpoint", default=None, help="Existing checkpoint path; currently only existence is validated")
    parser.add_argument("--optuna-best", default=None, help="Path to optuna_best.json with frozen actor params")
    parser.add_argument("--data", default="open_bandit")
    parser.add_argument("--split", default="test", choices=["train", "valid", "valid_hpo", "test", "all"])
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--full-train", action="store_true", help="Use full train/validation splits instead of HPO row limits")
    args = parser.parse_args()
    if args.checkpoint and not Path(args.checkpoint).exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.optuna_best:
        if args.checkpoint:
            print("Checkpoint exists. Pass --optuna-best to retrain/evaluate a frozen selected config.")
            return
        raise SystemExit("--optuna-best is required for selected-policy evaluation")
    run_dir = evaluate_frozen_policy(
        data=args.data,
        optuna_best=Path(args.optuna_best),
        output_dir=None if args.output_dir is None else Path(args.output_dir),
        split=args.split,
        bootstrap=args.bootstrap,
        use_hpo_row_limits=not args.full_train,
    )
    print(f"Selected Bandit-STOR policy evaluation complete: {run_dir}")


if __name__ == "__main__":
    main()
