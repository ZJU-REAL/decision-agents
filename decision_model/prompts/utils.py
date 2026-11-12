"""Shared position validation and trajectory target instructions.

No concrete prompt components are imported or registered here. Each context
chooses its visible range and supplies the target index within that range.
"""

from decision_model.types import Position, TrajectorySnapshot


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


def validate_decision_position(trajectory: TrajectorySnapshot, position: Position) -> int:
    index = position.index
    if (position.kind != "decision" or position.end is not None
            or type(index) is not int or not 0 <= index <= trajectory.n_steps):
        raise ValueError("position must identify an observed decision")
    budget = trajectory.task.max_steps
    if type(budget) is not int or not 0 <= index < budget:
        raise ValueError("no remaining positive integer action budget")
    return index



TRAJECTORY_TARGET_INSTRUCTIONS = """Evaluate only the action at trajectory[{target_step}].
Trajectory indices start at 0. Each obs is observed before its paired
action; the next obs contains feedback after that action. An entry
with action=null contains observation only."""


def trajectory_target_instructions(target_index: int) -> str:
    """Describe how to locate an action in the visible trajectory array."""
    return TRAJECTORY_TARGET_INSTRUCTIONS.format(target_step=target_index)
