"""Hindsight context: complete rollout evidence and its target locator."""

from __future__ import annotations

from decision_model.prompts.base import BaseContextPrompt
from decision_model.prompts.utils import trajectory_target_instructions, validate_action_position
from decision_model.registry import register_context_prompt
from decision_model.types import ContextPayload, Position, TrajectorySnapshot


class HindsightContextPrompt(BaseContextPrompt):
    """Expose every observation/action pair, including the final observation."""

    name = "hindsight"

    def build(self, trajectory: TrajectorySnapshot, position: Position) -> ContextPayload:
        target = validate_action_position(trajectory, position)
        if not isinstance(trajectory.task.task, str):
            raise ValueError("task text must be a string")
        if any(not isinstance(obs, str) for obs in trajectory.observations):
            raise ValueError("environment observations must be strings")
        if any(not isinstance(action, str) for action in trajectory.actions):
            raise ValueError("executed actions must be strings")
        pairs = [{"obs": obs, "action": action} for obs, action in trajectory.history]
        pairs.append({"obs": trajectory.final_observation, "action": None})
        return ContextPayload(
            text="",
            materials={
                "state": {"goal": trajectory.task.task, "trajectory": pairs},
                "target_instructions": trajectory_target_instructions(target),
            },
        )


register_context_prompt("hindsight", HindsightContextPrompt)
