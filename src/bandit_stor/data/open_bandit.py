"""Open Bandit / CSV adapters for Bandit-STOR."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import polars as pl
import torch


from bandit_stor.data.base import (
    LoggedBanditDataset,
    split_dataset,
    uniform_behavior_from_logged_pscore,
)


logger = logging.getLogger(__name__)


DEFAULT_OPEN_BANDIT_FULL_DATA_URL = "https://research.zozo.com/data_release/open_bandit_dataset.zip"


class OpenBanditUnavailableError(RuntimeError):
    """Raised when Open Bandit data/dependency is unavailable."""


def _action_context(candidate_actions: np.ndarray, n_actions: int) -> torch.Tensor:
    """Return deterministic item/action identity features `[K, D_a]`.

    Prepared CSV files often contain only logged actions, not full item metadata.  In
    that case the reward model still needs action/item information beyond context, so
    we expose normalized id features plus a bounded one-hot identity basis.  This is
    a feature representation for q_hat/action policies, not a propensity estimate.
    """
    actions = np.asarray(candidate_actions, dtype=np.int64)
    denom = max(int(n_actions) - 1, 1)
    normalized = actions.astype(np.float32)[:, None] / float(denom)
    angle = 2.0 * np.pi * normalized
    cyclic = np.concatenate([np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)
    if int(n_actions) <= 512:
        identity = np.eye(int(n_actions), dtype=np.float32)[actions]
    else:
        # Hash-bucket identity for very large catalogs keeps memory bounded while
        # still giving the reward model item-specific signal.
        buckets = 512
        identity = np.zeros((len(actions), buckets), dtype=np.float32)
        identity[np.arange(len(actions)), actions % buckets] = 1.0
    return torch.as_tensor(np.concatenate([normalized, cyclic, identity], axis=1), dtype=torch.float32)


def _augment_action_context(action_context: np.ndarray, n_actions: int) -> np.ndarray:
    """Append stable action-id features to OBP item features, returning `[K, D_a+]`."""
    base = np.asarray(action_context, dtype=np.float32)
    if base.ndim == 1:
        base = base[:, None]
    ids = _action_context(np.arange(int(n_actions), dtype=np.int64), int(n_actions)).numpy()
    if base.shape[0] != ids.shape[0]:
        raise ValueError("action_context row count must match n_actions")
    return np.concatenate([base, ids], axis=1).astype(np.float32)


def _obp_raw_file_path(config: dict[str, Any], data_root: Path) -> Path:
    """Return expected OBP raw campaign CSV path under an OBP data root."""
    behavior_policy = str(config.get("behavior_policy", "random"))
    campaign = str(config.get("campaign", "all"))
    return data_root / behavior_policy / campaign / f"{campaign}.csv"


def _find_obp_data_root(config: dict[str, Any], data_path: Path) -> Path | None:
    """Find an OBP root below `data_path`.

    ZOZO's full zip extracts into `open_bandit_dataset/...`, while OBP expects the
    root passed to `OpenBanditDataset` to directly contain `{behavior_policy}/{campaign}`.
    """
    candidates = [data_path, data_path / "open_bandit_dataset"]
    for root in candidates:
        raw = _obp_raw_file_path(config, root)
        if raw.exists() and (raw.parent / "item_context.csv").exists():
            return root
    return None


def _has_local_open_bandit_data(config: dict[str, Any], data_path: Path) -> bool:
    """Return true when canonical or OBP-layout local Open Bandit data exists."""
    if (data_path / "logged_bandit.csv").exists() or (data_path / "logged_bandit.parquet").exists():
        return True
    return _find_obp_data_root(config, data_path) is not None


def download_open_bandit_full_dataset(config: dict[str, Any]) -> Path:
    """Download and extract the official full Open Bandit Dataset zip.

    The full dataset is published by ZOZO Research as a zip file containing OBP-layout
    files such as `{behavior_policy}/{campaign}/{campaign}.csv` and `item_context.csv`.
    This function downloads real OBD data only; it never generates synthetic data.
    """
    data_path = Path(config.get("data_path", "data/open_bandit"))
    url = str(config.get("download_url", DEFAULT_OPEN_BANDIT_FULL_DATA_URL))
    archive_path = Path(config.get("archive_path", data_path / "_downloads" / "open_bandit_dataset.zip"))
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        logger.info("Downloading full Open Bandit Dataset from %s to %s", url, archive_path)
        urlretrieve(url, archive_path)
    else:
        logger.info("Using existing Open Bandit Dataset archive at %s", archive_path)
    logger.info("Extracting Open Bandit Dataset archive to %s", data_path)
    with ZipFile(archive_path) as zf:
        zf.extractall(data_path)
    return data_path


def load_csv_logged_bandit(
    path: str | Path,
    *,
    context_columns: list[str] | None = None,
    action_column: str = "action",
    reward_column: str = "reward",
    pscore_column: str = "pscore",
    position_column: str = "position",
    n_actions: int | None = None,
) -> LoggedBanditDataset:
    """Load an Open Bandit-style CSV/parquet file.

    Each row is converted to candidates `[0..K-1]`; the reward label is kept only for
    the logged action. Logged `pscore` is used as source of truth for that action.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        frame = pl.read_parquet(path)
    else:
        frame = pl.read_csv(path)
    required = {action_column, reward_column, pscore_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required logged bandit columns: {sorted(missing)}")
    if context_columns is None:
        context_columns = [c for c in frame.columns if c.startswith("context_")]
    if not context_columns:
        raise ValueError("No context columns found; pass context_columns or use context_ prefixed columns")
    pscores = frame[pscore_column].cast(pl.Float32).to_numpy()
    if (pscores <= 0).any():
        raise ValueError("Logged pscore must be strictly positive")
    actions = frame[action_column].cast(pl.Int64).to_numpy()
    rewards = frame[reward_column].cast(pl.Float32).to_numpy()
    contexts = frame.select(context_columns).to_numpy().astype(np.float32)
    if position_column in frame.columns:
        positions = frame[position_column].cast(pl.Int64).to_numpy()
    else:
        positions = np.zeros(len(actions), dtype=np.int64)
    inferred_n_actions = int(actions.max()) + 1
    n_actions = int(n_actions or inferred_n_actions)
    if actions.min() < 0 or actions.max() >= n_actions:
        raise ValueError("Logged action outside candidate action range")
    candidates = np.arange(n_actions, dtype=np.int64)
    candidate_action_context = _action_context(candidates, n_actions)
    examples: list[dict[str, Any]] = []
    for index in range(len(actions)):
        action = int(actions[index])
        pscore = float(pscores[index])
        examples.append(
            {
                "context": torch.as_tensor(contexts[index], dtype=torch.float32),
                "candidate_actions": torch.as_tensor(candidates, dtype=torch.long),
                "action_context": candidate_action_context,
                "logged_action_index": action,
                "reward": float(rewards[index]),
                "pscore": pscore,
                "position": int(positions[index]),
                "mask": torch.ones(n_actions, dtype=torch.bool),
                "behavior_policy_probs": uniform_behavior_from_logged_pscore(n_actions, action, pscore),
            }
        )
    return LoggedBanditDataset(examples, len(context_columns), int(candidate_action_context.shape[1]), n_actions)


def load_tiny_fixture(config: dict[str, Any]) -> tuple[LoggedBanditDataset, Any]:
    """Load the handcrafted tiny fixture for tests/smoke only."""
    dataset = load_csv_logged_bandit(
        config["path"],
        context_columns=list(config.get("context_columns", [])) or None,
        action_column=config.get("action_column", "action"),
        reward_column=config.get("reward_column", "reward"),
        pscore_column=config.get("pscore_column", "pscore"),
        position_column=config.get("position_column", "position"),
        n_actions=int(config["n_actions"]),
    )
    split_cfg = config.get("split", {})
    return dataset, split_dataset(
        dataset,
        valid_size=float(split_cfg.get("valid_size", 0.2)),
        test_size=float(split_cfg.get("test_size", 0.2)),
        seed=int(split_cfg.get("seed", 42)),
        strategy=str(split_cfg.get("strategy", "deterministic_random")),
    )


class OBPFeedbackDataset(LoggedBanditDataset):
    """Lazy dataset backed by OBP feedback arrays.

    This avoids materializing one Python dictionary and one `[K, D_a]` action-context
    tensor per row, which is too expensive for the full Open Bandit Dataset.
    """

    def __init__(
        self,
        *,
        contexts: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        pscores: np.ndarray,
        positions: np.ndarray,
        action_context_array: np.ndarray,
        n_actions: int,
    ):
        super().__init__([], int(contexts.shape[1]), int(action_context_array.shape[1]), int(n_actions))
        self.contexts = contexts
        self.actions = actions
        self.rewards = rewards
        self.pscores = pscores
        self.positions = positions
        self.action_context_array = action_context_array
        self.candidates = np.arange(n_actions, dtype=np.int64)
        self._mask = torch.ones(n_actions, dtype=torch.bool)

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        action = int(self.actions[index])
        pscore = float(self.pscores[index])
        return {
            "context": torch.as_tensor(self.contexts[index], dtype=torch.float32),
            "candidate_actions": torch.as_tensor(self.candidates, dtype=torch.long),
            "action_context": torch.as_tensor(self.action_context_array, dtype=torch.float32),
            "logged_action_index": action,
            "reward": float(self.rewards[index]),
            "pscore": pscore,
            "position": int(self.positions[index]),
            "mask": self._mask,
            "behavior_policy_probs": uniform_behavior_from_logged_pscore(
                self.n_actions, action, pscore
            ),
        }


def dataset_from_obp_feedback(feedback: dict[str, Any]) -> LoggedBanditDataset:
    """Convert OBP `obtain_batch_bandit_feedback()` output to canonical examples.

    OBP documents feedback keys including `n_rounds`, `n_actions`, `action`,
    `position`, `reward`, `pscore`, `context`, and `action_context`. Logged `pscore`
    remains the source of truth for the factual action.
    """
    required = {"n_actions", "action", "reward", "pscore", "context"}
    missing = required - set(feedback)
    if missing:
        raise ValueError(f"OBP feedback missing required keys: {sorted(missing)}")
    n_actions = int(feedback["n_actions"])
    actions = np.asarray(feedback["action"], dtype=np.int64)
    rewards = np.asarray(feedback["reward"], dtype=np.float32)
    pscores = np.asarray(feedback["pscore"], dtype=np.float32)
    contexts = np.asarray(feedback["context"], dtype=np.float32)
    positions = np.asarray(feedback.get("position", np.zeros_like(actions)), dtype=np.int64)
    if (pscores <= 0).any():
        raise ValueError("OBP logged pscore must be strictly positive")
    action_context = feedback.get("action_context")
    if action_context is None:
        action_context_array = _action_context(np.arange(n_actions, dtype=np.int64), n_actions).numpy()
    else:
        action_context_array = _augment_action_context(np.asarray(action_context, dtype=np.float32), n_actions)
    return OBPFeedbackDataset(
        contexts=contexts,
        actions=actions,
        rewards=rewards,
        pscores=pscores,
        positions=positions,
        action_context_array=action_context_array,
        n_actions=n_actions,
    )


def _csv_columns(path: Path) -> list[str]:
    """Return CSV header columns without reading data rows."""
    return pl.read_csv(path, n_rows=0).columns


def _encode_polars_categoricals(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Encode string categorical columns as normalized numeric features."""
    if not columns:
        return pl.DataFrame()
    encoded = frame.select(
        [
            pl.col(column)
            .cast(pl.Categorical)
            .to_physical()
            .cast(pl.Float32)
            .alias(column)
            for column in columns
        ]
    )
    max_values = encoded.select(pl.all().max()).row(0)
    normalizers = {
        column: float(max_value)
        for column, max_value in zip(columns, max_values, strict=True)
        if max_value is not None and float(max_value) > 0.0
    }
    if normalizers:
        encoded = encoded.with_columns(
            [(pl.col(column) / normalizer).alias(column) for column, normalizer in normalizers.items()]
        )
    return encoded


def _obp_context_array(frame: pl.DataFrame, user_feature_cols: list[str], encoding: str) -> np.ndarray:
    """Build context features from OBP user feature columns using Polars."""
    if not user_feature_cols:
        raise ValueError("OBP data has no user_feature_* columns for context")
    if encoding == "one_hot":
        context_frame = frame.select(user_feature_cols).to_dummies(
            columns=user_feature_cols, drop_first=True
        )
    elif encoding == "categorical_codes":
        context_frame = _encode_polars_categoricals(frame, user_feature_cols)
    else:
        raise ValueError(
            "Unsupported OBP context_encoding="
            f"{encoding!r}; expected 'categorical_codes' or 'one_hot'"
        )
    return context_frame.to_numpy().astype(np.float32, copy=False)


def _obp_action_context_array(item_context_path: Path) -> np.ndarray:
    """Load and encode OBP item context with Polars."""
    item_frame = pl.read_csv(item_context_path)
    if "" in item_frame.columns:
        item_frame = item_frame.drop("")
    if "item_id" not in item_frame.columns:
        raise ValueError(f"OBP item context missing item_id: {item_context_path}")
    item_frame = item_frame.sort("item_id")
    item_ids = item_frame["item_id"].cast(pl.Int64).to_numpy()
    expected_item_ids = np.arange(len(item_ids), dtype=np.int64)
    if not np.array_equal(item_ids, expected_item_ids):
        raise ValueError("OBP item_id values must be contiguous and zero-based")

    numeric_cols = [
        column
        for column, dtype in zip(item_frame.columns, item_frame.dtypes, strict=True)
        if column != "item_id" and dtype.is_numeric()
    ]
    categorical_cols = [
        column
        for column in item_frame.columns
        if column != "item_id" and column not in set(numeric_cols)
    ]
    parts: list[pl.DataFrame] = []
    if categorical_cols:
        parts.append(_encode_polars_categoricals(item_frame, categorical_cols))
    if numeric_cols:
        parts.append(item_frame.select([pl.col(column).cast(pl.Float32) for column in numeric_cols]))
    if not parts:
        raise ValueError(f"OBP item context has no usable feature columns: {item_context_path}")
    encoded = pl.concat(parts, how="horizontal")
    return encoded.to_numpy().astype(np.float32, copy=False)


def _load_obp_feedback_polars(config: dict[str, Any], obp_root: Path) -> dict[str, Any]:
    """Load OBP-layout Open Bandit data using Polars instead of OBP's pandas loader."""
    raw_path = _obp_raw_file_path(config, obp_root)
    item_context_path = raw_path.parent / "item_context.csv"
    columns = _csv_columns(raw_path)
    user_feature_cols = [column for column in columns if column.startswith("user_feature")]
    required = {"timestamp", "item_id", "position", "click", "propensity_score"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"OBP data missing required columns: {sorted(missing)}")

    selected_columns = ["timestamp", "item_id", "position", "click", "propensity_score"]
    selected_columns.extend(user_feature_cols)
    max_rows = config.get("max_rows")
    n_rows = int(max_rows) if max_rows is not None else None
    logger.info(
        "Reading OBP CSV with Polars: file=%s columns=%s user_feature_cols=%s max_rows=%s",
        raw_path,
        len(selected_columns),
        len(user_feature_cols),
        n_rows,
    )
    frame = pl.read_csv(
        raw_path,
        columns=selected_columns,
        schema_overrides={
            "item_id": pl.Int64,
            "position": pl.Int64,
            "click": pl.Float32,
            "propensity_score": pl.Float32,
            **{column: pl.Utf8 for column in user_feature_cols},
        },
        infer_schema_length=1000,
        n_rows=n_rows,
    )
    sort_setting = config.get("sort_by_timestamp", "auto")
    if isinstance(sort_setting, bool):
        should_sort = sort_setting
    elif str(sort_setting).lower() == "auto":
        should_sort = not frame["timestamp"].is_sorted()
    else:
        should_sort = str(sort_setting).lower() in {"1", "true", "yes", "on"}
    if should_sort:
        logger.info("Sorting OBP rows by timestamp")
        frame = frame.sort("timestamp")
    else:
        logger.info("Keeping OBP source row order for chronological split")
    logger.info("Loaded OBP rows with Polars: n=%s", frame.height)

    context_encoding = str(config.get("context_encoding", "categorical_codes"))
    contexts = _obp_context_array(frame, user_feature_cols, context_encoding)
    action_context = _obp_action_context_array(item_context_path)
    n_actions = int(action_context.shape[0])
    actions = frame["item_id"].cast(pl.Int64).to_numpy()
    rewards = frame["click"].cast(pl.Float32).to_numpy()
    pscores = frame["propensity_score"].cast(pl.Float32).to_numpy()
    if (pscores <= 0).any():
        raise ValueError("OBP logged pscore must be strictly positive")
    if actions.min() < 0 or actions.max() >= n_actions:
        raise ValueError("OBP logged action outside item_context candidate range")
    positions = (
        frame.select((pl.col("position").rank("dense").cast(pl.Int64) - 1).alias("position"))[
            "position"
        ]
        .to_numpy()
        .astype(np.int64, copy=False)
    )
    logger.info(
        "Encoded OBP arrays with Polars: context_shape=%s action_context_shape=%s n_actions=%s",
        contexts.shape,
        action_context.shape,
        n_actions,
    )
    return {
        "n_rounds": int(frame.height),
        "n_actions": n_actions,
        "action": actions,
        "position": positions,
        "reward": rewards,
        "pscore": pscores,
        "context": contexts,
        "action_context": action_context,
    }


def load_open_bandit(config: dict[str, Any]) -> tuple[LoggedBanditDataset, Any]:
    """Load Open Bandit data from local prepared files or OBP.

    This function never substitutes item frequency for propensity. It requires logged
    propensity (`pscore`) from prepared files or the OBP dataset.
    """
    data_path = Path(config.get("data_path", "data/open_bandit"))
    logger.info("Loading Open Bandit data with data_path=%s", data_path)
    if bool(config.get("download", False)) and not _has_local_open_bandit_data(config, data_path):
        download_open_bandit_full_dataset(config)

    candidates = [data_path / "logged_bandit.csv", data_path / "logged_bandit.parquet"]
    for candidate in candidates:
        if candidate.exists():
            logger.info("Loading prepared logged bandit file: %s", candidate)
            dataset = load_csv_logged_bandit(
                candidate,
                context_columns=config.get("context_columns"),
                action_column=config.get("columns", {}).get("action", "action"),
                reward_column=config.get("columns", {}).get("reward", "reward"),
                pscore_column=config.get("columns", {}).get("pscore", "pscore"),
                position_column=config.get("columns", {}).get("position", "position"),
                n_actions=config.get("n_actions"),
            )
            split_cfg = config.get("split", {})
            return dataset, split_dataset(
                dataset,
                valid_size=float(split_cfg.get("valid_size", 0.1)),
                test_size=float(split_cfg.get("test_size", 0.2)),
                seed=int(split_cfg.get("seed", 42)),
                strategy=str(split_cfg.get("strategy", "chronological")),
            )
    try:  # pragma: no cover - requires external Open Bandit files
        obp_root = _find_obp_data_root(config, data_path)
        if obp_root is None:
            raise FileNotFoundError(_obp_raw_file_path(config, data_path))
        logger.info("Loading OBP-layout Open Bandit data from root=%s", obp_root)
        feedback = _load_obp_feedback_polars(config, obp_root)
    except Exception as exc:  # pragma: no cover - requires external Open Bandit files
        raise OpenBanditUnavailableError(
            "Open Bandit adapter was selected, but local/full downloaded data could not "
            "be loaded. With download=true the adapter downloads the official full OBD zip "
            "to data_path before loading. Expected OBP layout: "
            f"{data_path}/<behavior_policy>/<campaign>/ with campaign CSV and "
            "item_context.csv, or prepared logged_bandit.csv/parquet with logged "
            f"pscore under {data_path}. No packaged small-example or synthetic fallback is "
            f"used. Original error: {exc}"
        ) from exc
    dataset = dataset_from_obp_feedback(feedback)
    logger.info(
        "Loaded Open Bandit dataset: n=%s context_dim=%s action_dim=%s n_actions=%s",
        len(dataset),
        dataset.context_dim,
        dataset.action_dim,
        dataset.n_actions,
    )
    split_cfg = config.get("split", {})
    return dataset, split_dataset(
        dataset,
        valid_size=float(split_cfg.get("valid_size", 0.1)),
        test_size=float(split_cfg.get("test_size", 0.2)),
        seed=int(split_cfg.get("seed", 42)),
        strategy=str(split_cfg.get("strategy", "chronological")),
    )


def load_dataset_from_config(config: dict[str, Any]) -> tuple[LoggedBanditDataset, Any]:
    """Dispatch dataset loading by config name."""
    name = str(config.get("name", ""))
    if name == "tiny_fixture":
        return load_tiny_fixture(config)
    if name == "open_bandit":
        return load_open_bandit(config)
    raise ValueError(f"Unsupported data config name: {name}")
