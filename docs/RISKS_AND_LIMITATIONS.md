# Risks and Limitations

## Causal Limitations

- DR does not eliminate unobserved confounding.
- Propensity correction is only as good as logged pscore or estimated behavior model.
- Weak overlap causes high variance and unsafe extrapolation.

## Dataset Limitations

- Open-source datasets may not match production traffic.
- Candidate sets may be incomplete or reconstructed.
- Reward definitions may be sparse or delayed.

## Algorithmic Risks

- Sparsemax support can collapse to too few items.
- Alpha divergence can over-constrain policy improvement.
- Reward model bias can dominate direct-method term.
- IPS/DR estimates can be high variance under low propensities.

## Production Risks

- Feature skew between training and serving.
- Missing or incorrect probability logging.
- Candidate generator drift.
- Segment-level regressions hidden by aggregate OPE.
- Over-concentration on a small item subset.

## Mitigations

- Support masks.
- Importance weight clipping.
- ESS gates.
- Segment-level evaluation.
- Shadow mode.
- Canary deployment.
- Perfect decision logging.
- Conservative rollout.

## Published Artifact Risks

- Raw logged-bandit data, trained checkpoints, and generated reports may have licensing,
  privacy, or reproducibility constraints.
- The MIT License covers repository source code; upstream dataset licenses and terms govern
  downloaded or prepared data.
- Offline metrics can be misread as product claims unless the dataset version, split,
  propensities, support diagnostics, and assumptions are reported with the result.

Mitigations:

- Publish curated result artifacts with exact configuration, seed, and data provenance.
- Keep causal and deployment claims bounded by `docs/ASSUMPTIONS.md` and promotion gates.
