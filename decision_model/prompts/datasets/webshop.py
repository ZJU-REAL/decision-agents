"""WebShop judge records: separate goal, page observations and executed commands.

Search templates can be preserved as diagnostic candidates, but this adapter
provides no executable WebShop actor or search-query generation.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import re
from typing import Any, Mapping, Sequence, Tuple

from decision_model.prompts.base import BaseDatasetPrompt
from decision_model.registry import register_dataset_prompt
from decision_model.types import TaskSpec, TrajectorySnapshot


TERMINAL_OBSERVATION = "Shopping process has ended."
_SCORE_PAGE = re.compile(r"\byour\s+score\b|\bid\s*=\s*['\"]reward['\"]", re.IGNORECASE)
_PURCHASE = re.compile(r"^click\[\s*buy\s+now\s*\]$", re.IGNORECASE)


def _strings(value: Any, field: str) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("{} must be a sequence of strings".format(field))
    if any(not isinstance(item, str) for item in value):
        raise ValueError("{} must contain only strings".format(field))
    return tuple(value)


class WebShopDatasetPrompt(BaseDatasetPrompt):
    dataset = "webshop"

    def __init__(self, *, max_steps: int = 15) -> None:
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.max_steps = max_steps

    def prepare_for_prompt(self, trajectory: TrajectorySnapshot) -> TrajectorySnapshot:
        """Hide purchase-page outcomes from Judges and Teachers.

        Online collection supplies per-action dones; imported trajectories may
        instead expose the purchase action or the environment's score text.
        A rollout stopped by the action budget retains its last product page.
        Rewards, raw observations and caller-owned snapshots stay unchanged.
        """
        if trajectory.dataset != self.dataset:
            raise ValueError("WebShop prompt received another dataset")
        dones = trajectory.extra.get("dones", ())
        if not isinstance(dones, (list, tuple)):
            dones = ()
        observations = []
        for index, observation in enumerate(trajectory.observations):
            previous = index - 1
            purchased = (0 <= previous < len(trajectory.actions)
                         and _PURCHASE.fullmatch(trajectory.actions[previous].strip()))
            terminated = 0 <= previous < len(dones) and dones[previous] is True
            observations.append(TERMINAL_OBSERVATION if (
                purchased or terminated or _SCORE_PAGE.search(observation)
            ) else observation)
        return replace(trajectory, observations=tuple(observations))

    def adapt(self, record: Mapping[str, Any]) -> TrajectorySnapshot:
        if not isinstance(record, Mapping):
            raise TypeError("WebShop record must be a mapping")
        if record.get("dataset", self.dataset) != self.dataset:
            raise ValueError("WebShop adapter received another dataset")
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
        if not isinstance(task_text, str) or not isinstance(task_id, str):
            raise ValueError("task and task_id must be strings")
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
            trajectory_id=trajectory_id, dataset=self.dataset,
            task=TaskSpec(task_id, task_text, budget, copy.deepcopy(task_extra)),
            observations=observations, actions=actions, candidates=candidates,
            policy_responses=policy_responses, extra=copy.deepcopy(record.get("extra") or {}),
        )


register_dataset_prompt("webshop", WebShopDatasetPrompt)
