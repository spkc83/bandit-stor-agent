# Implementation Plan — Bandit-STOR MVP

## 1. Objective

Build an offline-trained contextual bandit recommender that uses real logged bandit data and a sparse, support-constrained actor.

The MVP does not implement sequential RL. It does not use synthetic generated training data. It does not use PPO.

## 2. Architecture Summary

```text
Open-source logged bandit data
  -> data adapter
  -> canonical logged interaction batch
  -> reward model q_hat(x,a)
  -> optional behavior support model mu_hat(.|x)
  -> Sparse Tsallis Actor pi_theta(.|x)
  -> DR actor objective + alpha divergence + Tsallis entropy
  -> OPE and model promotion gates
```

## 3. Milestones

### Milestone 0 — Repo Scaffold

Create:

- `pyproject.toml`
- `Makefile`
- `README.md`
- Config directory
- `src/bandit_stor` package
- Test directory

Acceptance:

- `pip install -e .` works.
- `pytest` discovers tests.
- Package imports cleanly.

### Milestone 1 — Canonical Data Schema

Create:

- `src/bandit_stor/data/schema.py`
- `src/bandit_stor/data/base.py`
- `src/bandit_stor/data/collate.py`

Acceptance:

- `LoggedInteractionBatch` dataclass exists.
- Batch supports context, candidate actions, action features, logged action index, reward, pscore, position, mask.
- Tiny fixture can be batched.

### Milestone 2 — Open Bandit Adapter

Create:

- `src/bandit_stor/data/open_bandit.py`
- `scripts/prepare_open_bandit.py`
- `configs/data/open_bandit.yaml`

Acceptance:

- Loads Open Bandit-style feedback.
- Exposes context, action, reward, pscore, position.
- Does not infer propensity from item frequency.
- Deterministic train/valid/test split.

### Milestone 3 — Math Objectives

Create:

- `models/sparsemax.py`
- `objectives/tsallis.py`
- `objectives/alpha_divergence.py`
- `objectives/ips.py`
- `objectives/doubly_robust.py`

Acceptance:

- Sparsemax outputs valid sparse probabilities.
- Tsallis entropy passes known-value tests.
- Alpha divergence is stable and zero for identical distributions.
- IPS/SNIPS/DR pass toy examples.

### Milestone 4 — Models

Create:

- `models/behavior_policy.py`
- `models/reward_model.py`
- `models/actor.py`

Acceptance:

- All models subclass `torch.nn.Module`.
- Actor uses Sparsemax by default.
- Reward model trains only on logged actions.
- Behavior model can be optional when logged full behavior probabilities exist.

### Milestone 5 — Actor Loss

Create:

- `objectives/losses.py`

Acceptance:

- Implements DR actor loss.
- Logs all loss components.
- Has no-NaN unit tests.
- Does not create labels for unobserved actions.

### Milestone 6 — Training Loops

Create:

- `training/train_behavior.py`
- `training/train_reward.py`
- `training/train_actor.py`
- `training/full_pipeline.py`

Acceptance:

- Each loop runs on tiny fixture.
- Checkpoints save and load.
- Metrics are written as JSON.

### Milestone 7 — Evaluation and Reports

Create:

- `evaluation/ope.py`
- `evaluation/diagnostics.py`
- `evaluation/policy_report.py`

Acceptance:

- Produces `ope_report.json`.
- Produces `policy_report.md`.
- Computes IPS, SNIPS, DR, ESS, max weight, support overlap, unsupported action mass, sparse support size.

### Milestone 8 — Full Pipeline

Create:

- `scripts/run_full_pipeline.py`

Acceptance:

- `python scripts/run_full_pipeline.py data=open_bandit` runs with the real dataset when available.
- `python scripts/run_full_pipeline.py data=tiny_fixture` runs in CI.
- Outputs go to `outputs/{run_id}/`.

## 4. MVP Non-Goals

- Sequential RL.
- PPO.
- Online serving service implementation.
- Full feature store integration.
- Real-time A/B platform integration.
- Synthetic data training benchmarks.

## 5. Completion Criteria

The MVP is complete when:

1. Real Open Bandit training path works.
2. Tiny fixture CI path works.
3. All objectives are tested.
4. OPE report is generated.
5. Policy report includes promotion recommendation.
6. Documentation clearly states assumptions and limitations.
