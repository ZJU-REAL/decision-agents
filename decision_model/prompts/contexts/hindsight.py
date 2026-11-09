"""Hindsight context: complete rollout evidence and its target locator."""

from __future__ import annotations

from decision_model.prompts.base import BaseContextPrompt
from decision_model.registry import register_context_prompt
from decision_model.types import ContextPayload, Position, TrajectorySnapshot


# The Hindsight target locator describes evidence independently of the question.
HINDSIGHT_TARGET_INSTRUCTIONS = """Evaluate only the action at trajectory[{target_step}].
Trajectory indices start at 0. Each obs is observed before its paired
action; the next obs contains feedback after that action. An entry
with action=null contains observation only."""


def validate_action_position(trajectory: TrajectorySnapshot, position: Position) -> int:
    """Reject an observation boundary or a position outside the action budget."""
    target = position.index
    if (position.kind != "action" or position.end is not None
            or type(target) is not int or not 0 <= target < trajectory.n_steps):
        raise ValueError("position must identify one executed action")
    budget = trajectory.task.max_steps
    if budget is not None and (type(budget) is not int or not 1 <= trajectory.n_steps <= budget):
        raise ValueError("trajectory must fit the positive integer action budget")
    return target


class HindsightContextPrompt(BaseContextPrompt):
    """Expose every observation/action pair, including the final observation."""

    name = "hindsight"

    def __init__(self, *, feedback: bool = True) -> None:
        if type(feedback) is not bool:
            raise ValueError("feedback must be a boolean")
        self.feedback = feedback

    def build(self, trajectory: TrajectorySnapshot, position: Position) -> ContextPayload:
        target = validate_action_position(trajectory, position)
        if not isinstance(trajectory.task.task, str):
            raise ValueError("task text must be a string")
        if any(not isinstance(obs, str) for obs in trajectory.observations):
            raise ValueError("environment observations must be strings")
        if any(not isinstance(action, str) for action in trajectory.actions):
            raise ValueError("executed actions must be strings")
        pairs = [{"obs": obs, "action": action} for obs, action in trajectory.history]
        if self.feedback:
            pairs.append({"obs": trajectory.final_observation, "action": None})
        else:
            pairs = [{"action": action} for action in trajectory.actions]
        return ContextPayload(
            text="",
            materials={
                "state": {"goal": trajectory.task.task, "trajectory": pairs},
                "target_instructions": HINDSIGHT_TARGET_INSTRUCTIONS.format(target_step=target),
                "evidence_scope": "hindsight_trajectory",
            },
        )


register_context_prompt("hindsight", HindsightContextPrompt)
