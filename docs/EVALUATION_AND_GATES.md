# Evaluation and Promotion Gates

## Evaluation Philosophy

Offline loss improvement is insufficient for promotion. Candidate actors are evaluated with off-policy and support-safety gates.

## Required OPE Metrics

- IPS policy value.
- SNIPS policy value.
- Doubly robust policy value.
- Effective sample size.
- Max importance weight.
- Mean importance weight.
- Support overlap / ESS ratio (ESS ratio is primary under uniform logging).
- Unsupported action mass.
- Alpha divergence to behavior support.
- Average sparse support size.
- Tsallis entropy.

## Promotion Gates

Default conservative gates:

```yaml
reward_model:
  validation_auc_roc_min: 0.60
  validation_auc_pr_required: true
  top_decile_lift_min: 1.0
  brier_score_max: 0.25

ope:
  dr_lift_min: 0.00
  statistically_significant_dr_lift: true
  ips_snips_dr_direction_agree: true
  dm_dr_dros_direction_agree: true
  dros_lambda_sensitivity: [0.1, 0.3, 1, 3, 10, 30, 100]

support:
  ess_ratio_min: 0.20
  max_importance_weight_max: 20
  unsupported_action_mass_max: 0.01
  logged_action_coverage: diagnostic_only

sparsity:
  avg_support_size_min: 2
  avg_support_size_max: 50

stability:
  no_nan_metrics: true
  no_segment_regression: true
```

## Baselines

Compare against:

1. Logged behavior policy.
2. Reward-model direct method policy.
3. Softmax actor with Shannon entropy.
4. Sparsemax actor without alpha-divergence.
5. Sparsemax actor without DR residual correction.

## Policy Report

The policy report includes:

- Dataset summary.
- Training configuration.
- Reward model metrics.
- OPE table.
- Support diagnostics.
- Sparsity diagnostics.
- Gate pass/fail summary.
- Recommendation: approve, reject, or needs review.

## Reward-lift selection update

Actor HPO is now reward-lift driven rather than sparsity-gate driven.  The selection
score uses DR lift only when reward calibration sanity checks pass; otherwise it falls
back to risk-adjusted SNIPS lift so an uncalibrated reward model cannot make DR look
artificially high.  Trials are hard-rejected for NaN metrics, tiny validation samples,
unsupported action mass above 1%, catastrophic importance weights, ESS ratio below the
configured catastrophic floor, deterministic collapse, average support below the sparse
minimum, low Tsallis entropy ratio, and material IPS/SNIPS/DR lift disagreement.

`logged_action_coverage` is the fraction of logged actions that fall inside the target
policy's sparse support. It is diagnostic only. For Open Bandit `random` uniform logging,
`support_overlap` is reported as ESS ratio rather than logged-action coverage, because a
sparse target policy is expected to cover only roughly `k / n_actions` of random logged
actions. Promotion blocks on unsupported target-policy mass and low ESS ratio, not on
logged-action coverage.

Reward models trained with weighted BCE are calibrated before their probabilities are
used for actor training, DR evaluation, or reports.  Supported calibration modes are:

- `auto_logit_correction`: `sigmoid(raw_logit - log(pos_weight))`
- `platt`: validation-set Platt scaling, `sigmoid(a * raw_logit + b)`

Reward sanity diagnostics include prediction mean vs. base rate, log loss vs. the constant
baseline, ECE, AUC-PR lift, AUC-ROC >= 0.60, top-decile lift > 1.0, and prediction
distribution statistics. DR-based promotion requires passing reward sanity diagnostics.
Phase-1 reward calibration can be run with:

```bash
python scripts/tune_reward_model.py data=open_bandit --n-trials 12
```

Actor HPO follows interpretable reward calibration, e.g.:

```bash
python scripts/tune_actor.py \
  --data open_bandit \
  --behavior_policy bts \
  --mode reward_aggressive \
  --n_trials 100 \
  --validation_size 50000
```

## Robust OPE estimator suite

Policy reports include OBP-style robust estimators in addition to IPS/SNIPS/DR:

- `direct_method`: mean target-policy reward-model value.
- `self_normalized_doubly_robust` / SNDR: direct method plus a self-normalized residual correction.
- `switch_dr`: DR residual correction only when the logged-action importance weight is below `switch_tau`.
- `dros`: DR with optimistic-shrinkage-style residual weights `lambda * w / (w^2 + lambda)`.

Actor selection does not rely on raw DR alone.  A reward-sane candidate is scored by the
minimum of DR/SNDR/Switch-DR/DRos lift, with hard rejection for estimator direction
disagreement, deterministic collapse, low ESS, high max weights, unsupported action mass,
and material lift disagreement.  If reward sanity fails, selection falls back to SNIPS lift.

The calibrated reward defaults originate from the full Open Bandit reward-HPO result:
trial 2 (`focal`, Platt calibration, 20% negative downsampling, 15 epochs).  Trial 11
(`weighted_bce`, Platt calibration, 20% negative downsampling, 15 epochs) remains a strong
challenger for future reward-candidate actor sweeps.

## Current OBD default

The default Open Bandit policy is `bts`, not `random`. BTS provides context-dependent
logged propensities, making ESS, alpha-divergence, clipping, Switch-DR, and DRos meaningful
variance diagnostics. The `random` policy remains supported for controlled checks; in that
case behavior-model training is skipped, full behavior support is exactly uniform, and the
logged `pscore` remains the OPE denominator for factual residual terms.

## Cross-fitting

Cross-fitted reward OPE is enabled by `eval.cross_fitting`. Each fold trains a fresh reward
model on the fold complement and evaluates DR/SNDR/Switch-DR/DRos on the held-out fold, then
reports fold-size-weighted `crossfit_*` lift diagnostics.
