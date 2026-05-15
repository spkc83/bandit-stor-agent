# Deployment MVP

## Deployment Principle

Deploy the trained Sparse Tsallis Actor only. The reward model, behavior model, DR objective, and OPE stack remain offline governance components unless explicitly needed for runtime diagnostics.

## Online Serving Flow

```text
request context
  -> candidate generation
  -> feature lookup
  -> hard eligibility filters
  -> SparseTsallisActor inference
  -> sparsemax probabilities
  -> top-k or safe sampling
  -> response
  -> decision log
```

## Runtime Inputs

- User/context features.
- Candidate item IDs.
- Candidate item features.
- Candidate mask.
- Policy version.

## Runtime Outputs

- Recommended item IDs.
- Position.
- Selection probability.
- Sparse support size.
- Tsallis entropy.
- Fallback flag.

## Required Logging

Every served decision log contains:

- request ID.
- policy version.
- context feature version.
- full candidate set or reproducible candidate-set reference.
- selected item.
- selected item probability.
- position.
- support size.
- sparse policy probabilities or sufficient probability reconstruction metadata.
- reward join key.

## Fallbacks

Fallback applies when:

- actor unavailable.
- feature fetch fails.
- candidate set empty.
- sparsemax support empty.
- latency budget exceeded.
- monitoring circuit breaker trips.

Fallback options:

- current production ranker.
- behavior policy.
- business-approved default slate.
- cached recommendations.

## Rollout

1. Offline replay.
2. Shadow mode.
3. Canary at <= 1%.
4. Controlled A/B test.
5. Progressive rollout.

## Monitoring

Monitor:

- latency.
- error rate.
- fallback rate.
- support size.
- unsupported action mass.
- entropy.
- catalog concentration.
- CTR/conversion.
- negative feedback.
- reward delay.
- propensity distribution for future training.
