# Assumptions

## Causal / Off-Policy Assumptions

The doubly robust objective supports exposure-bias correction only under assumptions.

Required assumptions:

1. Consistency: observed reward for the logged action equals the potential reward under that action.
2. Positivity / overlap: actions the target policy may choose have non-zero behavior-policy support.
3. Conditional exchangeability: after conditioning on observed context, action assignment is independent of potential outcomes.
4. Correct or useful propensity: logged pscore or behavior model approximates action assignment probability.
5. Stable reward definition: reward semantics are consistent across train/evaluation windows.

## MVP Scope Assumptions

- One-step contextual bandit.
- Immediate or joinable delayed reward.
- Candidate set known or reconstructable.
- Logged propensity available or estimable.
- Real open-source logged bandit data used for training.

## Non-Assumptions

- We do not assume unobserved confounding is removed.
- We do not assume item frequency is a valid propensity.
- We do not assume offline OPE replaces online A/B testing.

## Implementation Assumptions Added During MVP Build

- When a full candidate-set behavior distribution is missing, the implementation preserves logged `pscore` for the factual action and uses a support approximation over the remaining candidates only for full-support penalties/diagnostics. It is not used as an item-frequency propensity replacement.
- The tiny CSV fixture remains a smoke-test fixture only; it is not valid evidence for benchmark or model-quality claims.
- The Open Bandit command requires either local prepared files with logged `pscore` or the optional OBP dependency plus the real Open Bandit files in OBP layout. There is no synthetic fallback.
- With `download: true`, the Open Bandit adapter may delegate to OBP's built-in real example dataset loader when local `data/open_bandit` files are absent; this remains real OBD data, not synthetic generation.

## Reward calibration and OPE selection assumptions

- Open Bandit `bts` is the default policy for OPE experiments because it has context-dependent
  logged propensities.
- For Open Bandit `random`, the full behavior distribution is assumed known and uniform
  over the candidate set; behavior-model training is skipped and logged `pscore` remains
  the source of truth for the logged-action OPE denominator.
- Weighted-BCE reward logits are not interpreted as calibrated probabilities unless they
  pass through analytic logit correction or validation Platt scaling.
- When reward sanity checks fail, policy selection falls back to IPS/SNIPS evidence over
  DR because DR can inherit reward-model calibration error.
- Low logged-action coverage is expected for sparse policies under uniform random logging
  and is diagnostic rather than a promotion blocker; ESS ratio is the primary overlap
  reliability diagnostic in that regime.
- Cross-fitted OPE reduces reward-model overfit in DR diagnostics, but it does not remove
  the usual logged-bandit identification assumptions.
