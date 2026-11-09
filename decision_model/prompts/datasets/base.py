"""Shared helpers for dataset prompts.

Dataset adapters convert environment records into trajectory snapshots using
the shared helpers here. Dataset-specific interpretation stays in each adapter.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from decision_model.prompts.base import BaseDatasetPrompt
from decision_model.types import (
    TaskSpec,
    TrajectorySnapshot,
    snapshot_from_dict,
    snapshot_to_dict,
)

__all__ = [
    "BaseDatasetPrompt",
    "build_snapshot",
    "record_to_snapshot",
    "snapshot_from_record",
    "snapshot_from_dict",
    "snapshot_to_dict",
]


def build_snapshot(
    *,
    trajectory_id: str,
    dataset: str,
    task: str,
    observations: Sequence[str],
    actions: Sequence[str] = (),
    candidates: Sequence[Sequence[str]] = (),
    policy_responses: Sequence[str] = (),
    task_id: Optional[str] = None,
    max_steps: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> TrajectorySnapshot:
    """Assemble a snapshot from the flat fields a dataset record provides.

    This performs no dataset-specific interpretation: the caller decides which
    record field is an observation, an executed action or a pre-action candidate
    snapshot. Candidates must already be captured *before* the action they
    precede.
    """
    task_spec = TaskSpec(
        task_id=task_id if task_id is not None else trajectory_id,
        task=task,
        max_steps=max_steps,
    )
    return TrajectorySnapshot.from_rollout(
        trajectory_id=trajectory_id,
        dataset=dataset,
        task=task_spec,
        observations=tuple(observations),
        actions=tuple(actions),
        candidates=tuple(tuple(c) for c in candidates),
        policy_responses=tuple(policy_responses),
        extra=extra,
    )


def snapshot_from_record(
    record: Mapping[str, Any],
    *,
    dataset: Optional[str] = None,
) -> TrajectorySnapshot:
    """Adapter-facing alias for :func:`snapshot_from_dict`.

    ``dataset`` overrides the dataset name when the raw record does not carry
    one, which is the usual case for files that are homogeneous per dataset.
    """
    data = dict(record)
    if dataset is not None:
        data["dataset"] = dataset
    if "dataset" not in data:
        raise ValueError("record has no 'dataset' field and none was supplied")
    return snapshot_from_dict(data)


def record_to_snapshot(
    record: Mapping[str, Any],
    *,
    dataset: Optional[str] = None,
) -> TrajectorySnapshot:
    """Alias of :func:`snapshot_from_record`; both names are accepted."""
    return snapshot_from_record(record, dataset=dataset)
