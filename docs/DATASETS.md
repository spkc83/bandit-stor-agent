# Datasets

## Default: Open Bandit Dataset

Open Bandit Dataset with the BTS behavior policy is the default MVP training dataset because it has logged contextual bandit feedback with actions, rewards, positions, and propensity scores.

## Data Adapter Responsibilities

The adapter returns canonical fields:

```text
context
candidate_actions
action_context
logged_action_index
reward
pscore
position
behavior_policy_probs optional
mask
```

## Propensity Rules

1. If logged `pscore` exists, use it as source of truth for the logged action.
2. `pscore` is not replaced with item frequency.
3. If full behavior distribution over candidates is unavailable, train a behavior model for support diagnostics.
4. Clamp only for numerical stability, and report how often clamping is used.

## Train/Validation/Test Split

Default:

```text
chronological split
train: oldest 70%
valid: next 10%
test: newest 20%
```

Alternative for datasets without timestamp:

```text
deterministic random split with seed
```

## Tiny Fixture

`tests/fixtures/tiny_logged_bandit.csv` is reserved for CI and smoke tests.

The tiny fixture is not valid evidence for model-quality claims.

## Optional Dataset Adapters

### KuaiRand

Future sequential recommendation extensions or contextualized exposure experiments.

### KuaiRec

Dense-evaluation ablations; not default logged-propensity bandit data.

### Criteo Counterfactual/Bandit Data

Later large-scale benchmark candidate.


## Full Open Bandit Download

`configs/data/open_bandit.yaml` controls full dataset preparation:

```yaml
data_path: data/open_bandit
download: true
download_url: https://research.zozo.com/data_release/open_bandit_dataset.zip
```

When local files are absent and `download: true`, the adapter downloads the official full OBD zip and extracts it under `data_path`. The expected OBP layout is either directly under `data_path` or under the archive's `open_bandit_dataset/` top-level directory:

```text
data/open_bandit/{behavior_policy}/{campaign}/{campaign}.csv
data/open_bandit/{behavior_policy}/{campaign}/item_context.csv
data/open_bandit/open_bandit_dataset/{behavior_policy}/{campaign}/{campaign}.csv
data/open_bandit/open_bandit_dataset/{behavior_policy}/{campaign}/item_context.csv
```

The full dataset is real logged bandit feedback with true propensity scores. No synthetic fallback or item-frequency propensity is used. The default `behavior_policy` is `bts`; `random` is retained only for uniform-logging diagnostics and skips behavior-model training in favor of exact uniform `mu_probs`.

## Local Data Layout

Experiment inputs are expected under `data/`. Open Bandit runs may use either prepared
`logged_bandit.csv` / `logged_bandit.parquet` files with logged `pscore` or the extracted
OBD archive layout described above. Published results should identify the upstream dataset
version, preparation command, configuration, seed, and output artifact location.
