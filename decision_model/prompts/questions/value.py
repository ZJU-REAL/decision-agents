"""Value: task completion probability for the context's target action."""

from __future__ import annotations

from typing import Optional

from decision_model.prompts.base import BaseQuestionPrompt
from decision_model.prompts.utils import validate_action_position
from decision_model.registry import register_question_prompt
from decision_model.types import AnswerSpec, ContextPayload, Position, QuestionPayload, TrajectorySnapshot


VALUE_QUESTION = (
    "What is the probability that the task will be completed within the remaining step budget "
    "if this action is executed first and the same agent policy then continues? "
    "Estimate the success probability conditioned on this action, not the probability of choosing "
    "the action. The remaining step budget includes the current action."
)


class ValueQuestion(BaseQuestionPrompt):
    name = "value"

    def build(self, trajectory: TrajectorySnapshot, position: Position,
              context: Optional[ContextPayload] = None) -> QuestionPayload:
        target = validate_action_position(trajectory, position)
        locator = None if context is None else context.materials.get("target_instructions")
        budget = trajectory.task.max_steps
        if not isinstance(locator, str) or not locator:
            raise ValueError("value requires context target_instructions")
        if type(budget) is not int or budget < 1:
            raise ValueError("value requires a positive integer max_steps")
        instructions = (locator + "\nSteps remaining before max_steps: {} "
                        "(including the current action).\n\nQuestion:\n".format(budget - target)
                        + VALUE_QUESTION)
        return QuestionPayload(question=instructions, answer=AnswerSpec(kind="score"),
                               materials={"question_key": "value"})


register_question_prompt("value", ValueQuestion)
