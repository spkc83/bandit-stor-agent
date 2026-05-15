"""Collation helpers for logged interaction examples."""

from __future__ import annotations

from typing import Any

import torch

from bandit_stor.data.schema import LoggedInteractionBatch


def collate_logged_interactions(rows: list[dict[str, Any]]) -> LoggedInteractionBatch:
    """Collate dictionaries into a `LoggedInteractionBatch`.

    Expected per-row shapes before collation: context `[D_x]`, candidate_actions `[K]`,
    action_context `[K, D_a]`, logged_action_index scalar, reward scalar, pscore scalar,
    position scalar, mask `[K]`, optional behavior_policy_probs `[K]`.
    """
    behavior = None
    if rows and rows[0].get("behavior_policy_probs") is not None:
        behavior = torch.stack([torch.as_tensor(r["behavior_policy_probs"], dtype=torch.float32) for r in rows])
    return LoggedInteractionBatch(
        context=torch.stack([torch.as_tensor(r["context"], dtype=torch.float32) for r in rows]),
        candidate_actions=torch.stack(
            [torch.as_tensor(r["candidate_actions"], dtype=torch.long) for r in rows]
        ),
        action_context=torch.stack(
            [torch.as_tensor(r["action_context"], dtype=torch.float32) for r in rows]
        ),
        logged_action_index=torch.as_tensor([r["logged_action_index"] for r in rows], dtype=torch.long),
        reward=torch.as_tensor([r["reward"] for r in rows], dtype=torch.float32),
        pscore=torch.as_tensor([r["pscore"] for r in rows], dtype=torch.float32),
        position=torch.as_tensor([r.get("position", 0) for r in rows], dtype=torch.long),
        mask=torch.stack([torch.as_tensor(r["mask"], dtype=torch.bool) for r in rows]),
        behavior_policy_probs=behavior,
    )
