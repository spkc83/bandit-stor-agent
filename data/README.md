# Data Directory

Use this directory for local logged-bandit datasets and prepared experiment inputs.

Expected default layout for Open Bandit Dataset runs:

```text
data/open_bandit/
  logged_bandit.csv        # optional prepared file with logged pscore
  logged_bandit.parquet    # optional prepared file with logged pscore
  open_bandit_dataset/     # optional extracted official OBD archive
```

The tiny CSV under `tests/fixtures/` is reserved for tests and smoke runs, not benchmark claims.
