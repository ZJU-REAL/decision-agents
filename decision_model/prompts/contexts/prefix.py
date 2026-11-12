"""Prefix context: prior observations and actions plus the target action.

The target action is included, but its feedback and later steps are hidden."""

from __future__ import annotations

from typing import Optional

from decision_model.prompts.base import BaseContextPrompt
from decision_model.prompts.utils import trajectory_target_instructions, validate_action_position
from decision_model.registry import register_context_prompt
from decision_model.types import ContextPayload, Position, TrajectorySnapshot


class PrefixContextPrompt(BaseContextPrompt):
    name = "prefix"

    def __init__(self, *, history_length: Optional[int] = None) -> None:
        if history_length is not None and (type(history_length) is not int or history_length < 0):
            raise ValueError("history_length must be None or a non-negative integer")
        self.history_length = history_length

    def build(self, trajectory: TrajectorySnapshot, position: Position) -> ContextPayload:
        target = validate_action_position(trajectory, position)
        start = 0 if self.history_length is None else max(0, target - self.history_length)
        pairs = [
            {"obs": trajectory.observations[index], "action": trajectory.actions[index]}
            for index in range(start, target + 1)
        ]
        return ContextPayload(text="", materials={
            "state": {"goal": trajectory.task.task, "trajectory": pairs},
            "target_instructions": trajectory_target_instructions(target - start),
        })


register_context_prompt("prefix", PrefixContextPrompt)
