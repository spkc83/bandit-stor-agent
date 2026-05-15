# Outputs Directory

Training, evaluation, hyperparameter search, reports, model checkpoints, logs, and Optuna databases are written here by default.

Expected generated layout:

```text
outputs/{run_id}/
  actor.pt
  metrics.json
  ope_report.json
  policy_report.md
  pipeline.log
```

Published result artifacts should identify the dataset version, configuration, seed, and command used to reproduce them.
