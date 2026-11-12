"""Evidence available before a specified decision, with no selected answer."""

from __future__ import annotations

from decision_model.prompts.base import BaseContextPrompt
from decision_model.prompts.utils import validate_decision_position
from decision_model.registry import register_context_prompt
from decision_model.types import ContextPayload, Position, TrajectorySnapshot


class CurrentContextPrompt(BaseContextPrompt):
    name = "current"

    def build(self, trajectory: TrajectorySnapshot, position: Position) -> ContextPayload:
        index = validate_decision_position(trajectory, position)
        if not isinstance(trajectory.task.task, str) or not trajectory.task.task:
            raise ValueError("task must be a non-empty string")
        observation = trajectory.observations[index]
        if not isinstance(observation, str):
            raise ValueError("current observation must be a string")
        # A completed rollout can also be replayed at decision k. Never read
        # actions[k] or any later observation into the model-visible state.
        history = [{"observation": trajectory.observations[k], "action": trajectory.actions[k]}
                   for k in range(index)]
        return ContextPayload(text="", materials={"state": {
            "task": trajectory.task.task,
            "history": history,
            "current_observation": observation,
            "steps_taken": index,
            "steps_remaining": trajectory.task.max_steps - index,
        }})


register_context_prompt("current", CurrentContextPrompt)
