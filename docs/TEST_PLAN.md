# Test Plan

## Unit Tests

### Sparsemax

- Output probabilities sum to one.
- Output probabilities are non-negative.
- Some outputs are exactly zero for skewed logits.
- Masked candidates receive zero probability.
- Gradients flow.

### Tsallis Entropy

- Deterministic distribution has entropy zero.
- Uniform distribution over K has `1 - 1/K` for q=2.
- Entropy is non-negative.

### Alpha Divergence

- `D(p || p) ≈ 0`.
- Divergence increases when p puts mass where q is small.
- No NaN when q has tiny values.

### IPS / SNIPS / DR

- IPS matches hand-computed examples.
- SNIPS normalizes weights correctly.
- ESS is high for uniform weights and low for concentrated weights.
- DR equals direct method when residual is zero.
- DR equals IPS when reward model is zero.

## Data Tests

- Tiny fixture loads.
- Required columns exist.
- pscore is positive.
- action indices are valid.
- split is deterministic.

## Model Tests

- Behavior policy outputs valid probability distribution.
- Reward model outputs `[B, K]` values.
- Actor outputs sparse probabilities.
- Actor supports masks.

## Training Smoke Tests

- One training step for behavior model.
- One training step for reward model.
- One training step for actor.
- Full tiny fixture pipeline writes metrics and report.

## Regression Tests

- No unobserved zero labels are created.
- Logged pscore is used when present.
- PPO modules are not required by MVP pipeline.
