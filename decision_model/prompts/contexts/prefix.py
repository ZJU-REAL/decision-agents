"""Prefix context: prior observations and actions plus the target action.

The target action is included, but its feedback and later steps are hidden."""

from __future__ import annotations

from decision_model.prompts.base import BaseContextPrompt
from decision_model.prompts.contexts.hindsight import validate_action_position
from decision_model.registry import register_context_prompt
from decision_model.types import ContextPayload, Position, TrajectorySnapshot


class PrefixContextPrompt(BaseContextPrompt):
    name = "prefix"

    def __init__(self, *, history_length: int = 30, feedback: bool = True) -> None:
        if type(history_length) is not int or history_length < 0:
            raise ValueError("history_length must be a non-negative integer")
        if type(feedback) is not bool:
            raise ValueError("feedback must be a boolean")
        self.history_length = history_length
        self.feedback = feedback

    def build(self, trajectory: TrajectorySnapshot, position: Position) -> ContextPayload:
        target = validate_action_position(trajectory, position)
        budget = trajectory.task.max_steps
        if type(budget) is not int or budget < 1:
            raise ValueError("prefix requires a positive integer max_steps")
        lines = []
        if trajectory.task.task:
            lines.append("Task: {}".format(trajectory.task.task))
        lines.append("Steps remaining before max_steps: {} (including the current action).".format(budget - target))
        start = max(0, target - self.history_length)
        if start < target:
            lines.append("History:")
            for index in range(start, target):
                action = trajectory.actions[index]
                if self.feedback:
                    lines.append("  obs: {} | action: {}".format(trajectory.observations[index], action))
                else:
                    lines.append("  action: {}".format(action))
        if self.feedback:
            lines.append("Current observation: {}".format(trajectory.observations[target]))
        return ContextPayload(text="", materials={
            "state": {"observation": "\n".join(lines), "action": trajectory.actions[target]},
            "target_instructions": "Evaluate only the action in state.action.",
            "evidence_scope": "pre_action",
        })


register_context_prompt("prefix", PrefixContextPrompt)
