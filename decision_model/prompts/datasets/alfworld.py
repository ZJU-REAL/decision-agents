"""Adapt ALFWorld environment records without re-parsing policy prompts."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Optional, Sequence, Tuple

from decision_model.prompts.base import BaseDatasetPrompt
from decision_model.registry import register_dataset_prompt
from decision_model.types import Position, TaskSpec, TrajectorySnapshot


def _strings(value: Any, field: str) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("{} must be a sequence of strings".format(field))
    if any(not isinstance(item, str) for item in value):
        raise ValueError("{} must contain only strings".format(field))
    return tuple(value)


class AlfWorldDatasetPrompt(BaseDatasetPrompt):
    """Adapt the flat environment-record shape captured during environment interaction.

    ``observations`` contains T+1 raw environment observations; ``actions``
    contains T commands actually executed after projection. ``candidates[t]``
    is captured before action t, never from that action's step result. Optional
    ``policy_responses`` are preserved separately and are never used as actions.
    This adapter does not infer missing observations, actions, or candidates.
    """

    dataset = "alfworld"

    def __init__(self, *, max_steps: int = 30) -> None:
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.max_steps = max_steps

    def actor_candidates(self, trajectory: TrajectorySnapshot,
                         position: Optional[Position] = None) -> Tuple[str, ...]:
        """Return only the actor view; raw records and history remain unchanged."""
        from decision_model.prompts.contexts.current import validate_decision_position
        if trajectory.dataset != self.dataset:
            raise ValueError("ALFWorld actor requires an ALFWorld snapshot")
        position = position if position is not None else Position.decision(trajectory.n_steps)
        index = validate_decision_position(trajectory, position)
        try:
            commands = trajectory.candidates_at(index)
        except IndexError as exc:
            raise ValueError("candidate snapshot is absent") from exc
        if any(not isinstance(command, str) or not command.strip() for command in commands):
            raise ValueError("admissible_commands must contain non-empty strings")
        return tuple(command for command in commands if command != "help")

    def adapt(self, record: Mapping[str, Any]) -> TrajectorySnapshot:
        if not isinstance(record, Mapping):
            raise TypeError("ALFWorld record must be a mapping")
        if record.get("dataset", self.dataset) != self.dataset:
            raise ValueError("ALFWorld adapter received another dataset")
        trajectory_id = record.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("trajectory_id must be a non-empty string")

        raw_task = record.get("task")
        if isinstance(raw_task, Mapping):
            task_text = raw_task.get("task")
            task_id = raw_task.get("task_id", record.get("task_id", trajectory_id))
            budget = raw_task.get("max_steps", record.get("max_steps", self.max_steps))
            task_extra = raw_task.get("extra") or {}
        else:
            task_text = raw_task
            task_id = record.get("task_id", trajectory_id)
            budget = record.get("max_steps", self.max_steps)
            task_extra = record.get("task_extra") or {}
        if not isinstance(task_text, str):
            raise ValueError("task must contain the task text")
        if not isinstance(task_id, str):
            raise ValueError("task_id must be a string")
        if budget is None:
            budget = self.max_steps
        if type(budget) is not int or budget < 1:
            raise ValueError("max_steps must be a positive integer")

        observations = _strings(record.get("observations", ()), "observations")
        actions = _strings(record.get("actions", ()), "actions")
        raw_candidates = record.get("candidates", ())
        if isinstance(raw_candidates, (str, bytes)) or not isinstance(raw_candidates, Sequence):
            raise ValueError("candidates must be a sequence of pre-action snapshots")
        candidates = tuple(_strings(group, "candidates") for group in raw_candidates)
        policy_responses = _strings(record.get("policy_responses", ()), "policy_responses")
        if policy_responses and len(policy_responses) != len(actions):
            raise ValueError("policy_responses must be absent or aligned with actions")
        if len(actions) > budget:
            raise ValueError("executed actions exceed max_steps")

        return TrajectorySnapshot(
            trajectory_id=trajectory_id,
            dataset=self.dataset,
            task=TaskSpec(task_id, task_text, budget, copy.deepcopy(task_extra)),
            observations=observations,
            actions=actions,
            candidates=candidates,
            policy_responses=policy_responses,
            extra=copy.deepcopy(record.get("extra") or {}),
        )


register_dataset_prompt("alfworld", AlfWorldDatasetPrompt)
