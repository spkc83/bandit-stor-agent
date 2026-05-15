from pathlib import Path

from bandit_stor.training.hpo import run_optuna_hpo


def test_optuna_hpo_tiny_fixture_runs_one_trial(tmp_path: Path):
    cfg = tmp_path / "hpo.yaml"
    cfg.write_text(
        """
study_name: tiny_hpo
storage: null
direction: maximize
n_trials: 1
sampler_seed: 42
max_train_rows: null
max_valid_rows: null
score:
  max_weight_penalty: 0.00005
  ess_penalty: 0.01
  alpha_penalty: 0.0001
  entropy_penalty: 0.001
  support_size_penalty: 0.00001
gates:
  hard:
    min_eval_n: 1
    max_importance_weight_catastrophic: 80.0
    ess_ratio_catastrophic_min: 0.0
    unsupported_action_mass_max: 1.0
    max_lift_disagreement: 100.0
  preferred:
    max_importance_weight: 30.0
    ess_ratio: 0.0
    alpha_divergence: 6.0
  sparsity:
    min_avg_support_size: 1
    max_avg_support_size: 5
    max_entropy_ratio: 1.0
    min_entropy_ratio: 0.0
search_space:
  actor_learning_rate: {low: 0.0003, high: 0.0003, log: false}
  actor_weight_decay: {low: 0.00001, high: 0.00001, log: false}
  temperature: {choices: [0.5]}
  top_k_before_sparsemax: {choices: [3]}
  beta_alpha: {low: 0.01, high: 0.01, log: false}
  lambda_tsallis: {choices: [0.0]}
  lambda_support: {choices: [1.0]}
  actor_epochs: {choices: [1]}
  reward_value_scale: {choices: [10.0]}
reward_sanity:
  prediction_mean_max_multiple_of_base_rate: 1000.0
  prediction_mean_min_multiple_of_base_rate: 0.0
  log_loss_must_not_exceed_constant_by_more_than: 1000.0
  ece_max: 1.0
  auc_pr_lift_min: 0.0
""",
        encoding="utf-8",
    )
    run_dir = run_optuna_hpo(
        data="tiny_fixture",
        hpo_config_path=cfg,
        output_dir=tmp_path / "runs",
        n_trials=1,
    )
    assert (run_dir / "optuna_best.json").exists()
    assert (run_dir / "optuna_trials.json").exists()
    assert (run_dir / "rankings.json").exists()
    assert (run_dir / "pipeline.log").exists()
