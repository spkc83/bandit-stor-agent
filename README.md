# Bandit-STOR Research Prototype

**Bandit-STOR** is a research-oriented Sparse Tsallis Offline Recommender for contextual bandit recommendation.

The project studies offline policy learning from real logged bandit feedback: reward modeling on logged actions, Sparsemax/Tsallis actor training, support constraints against behavior-policy support, and off-policy evaluation before any serving use.

## Research Status

This repository is an experimental research prototype, not a production service or a benchmark leaderboard. The code is intended to make assumptions, diagnostics, and failure modes explicit while keeping the online-serving artifact limited to the trained sparse actor.

The tiny fixture is for unit tests and smoke tests only; it is not evidence for training or benchmark claims. The source code is available under the MIT License; see `LICENSE`.

## Scope of Claims

Bandit-STOR is an offline contextual-bandit recommender. Its IPS, SNIPS, and DR outputs are off-policy estimates under logged-bandit assumptions, not standalone causal-effect claims. Stronger causal interpretations require a separate identification analysis covering the logging policy, overlap, exchangeability, reward stability, and measurement quality.

The package name is `bandit-stor`; the import package is `bandit_stor`. See `docs/ASSUMPTIONS.md` for the required assumptions and `docs/REFERENCES.md` for papers and related repositories.

## Project Scope

Included in the current research prototype:

- Contextual bandit formulation.
- Open Bandit Dataset adapter.
- Reward model \(\hat q(x,a)\).
- Sparsemax actor.
- Tsallis \(q=2\) entropy.
- Alpha-divergence behavior-support constraint.
- Doubly robust actor objective.
- IPS, SNIPS, DR off-policy evaluation.
- Promotion gates and production deployment guidance.

Excluded from the current research prototype:

- PPO.
- Sequential RL.
- Synthetic training data.
- Treating missing rewards as zero.

## Quickstart

Create an isolated Python environment, install the package, then run either the full Open Bandit pipeline or the fixture smoke path.

```bash
python -m venv .venv
source .venv/bin/activate
make install-dev
python scripts/run_full_pipeline.py data=open_bandit
```

For CI and local smoke tests without full dataset download:

```bash
pytest
python scripts/run_full_pipeline.py data=tiny_fixture
```

The tiny fixture path is for tests only, not benchmark training. The default Open Bandit path never falls back to synthetic data: it first uses prepared `data/open_bandit/logged_bandit.csv` / `.parquet` containing logged `pscore`, then local OBP-layout files under `data/open_bandit/{behavior_policy}/{campaign}/`. When `download: true`, the adapter downloads and extracts the official full Open Bandit Dataset zip into `data_path`; it no longer silently uses OBP's packaged small example. OBP-layout CSVs are loaded directly with Polars rather than OBP's pandas loader so full-dataset runs project only the columns required by Bandit-STOR.

## Data and Outputs

Input datasets are read from `data/` by default. Each run writes to `outputs/{run_id}/`:

- `actor.pt` — serving-time Sparse Tsallis actor checkpoint.
- `metrics.json` — dataset and training metrics.
- `ope_report.json` — IPS, SNIPS, DR, ESS, support, alpha-divergence, and sparsity diagnostics.
- `policy_report.md` — promotion-gate summary and assumptions.
- `pipeline.log` — UTC timestamped training/pipeline log.

## Core Actor Loss

```text
L_actor =
  -V_DR(pi_theta)
  + beta_alpha * D_alpha(pi_theta || mu_hat)
  - lambda_tsallis * H_2(pi_theta)
  + lambda_support * unsupported_action_mass
```

## Serving Principle

Train offline with DR and support constraints. Deploy only the actor. Log selected action probabilities perfectly for the next training cycle.

## Hyperparameter Search

Sparse actor hyperparameters are tuned with Optuna on a validation split:

```bash
python scripts/optimize_hyperparameters.py data=open_bandit --n-trials 20
```

The search space is configured in `configs/hpo_optuna.yaml` and includes Sparsemax temperature, `top_k_before_sparsemax`, alpha-divergence weight, Tsallis entropy weight, support penalty weight, actor learning rate, weight decay, and actor epochs. Results are written under `outputs/optuna/{run_id}/` as `optuna_best.json`, `optuna_trials.json`, and `pipeline.log`.

## Reward-lift model selection

Bandit-STOR policy selection uses risk-adjusted DR lift rather than sparsity/support gates alone. The selection score is implemented in `src/bandit_stor/evaluation/scoring.py`; hard rejections cover tiny validation sets, NaNs, catastrophic max importance weights, catastrophic ESS, and unsupported action mass. Soft penalties cover preferred max weight, ESS ratio, alpha divergence, entropy/uniformity, and support size.

For Open Bandit `behavior_policy: random`, the full behavior distribution is known uniform (`1 / n_actions`) and is used directly for full-support regularization/diagnostics; logged `pscore` remains the source of truth for logged-action OPE weights.


## License

Bandit-STOR is released under the MIT License. See `LICENSE` for details. Dataset licenses and terms remain governed by their upstream providers; do not redistribute local Open Bandit downloads or proprietary logs from this repository.
