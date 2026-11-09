"""Value: task completion probability under the declared evidence scope."""

from __future__ import annotations

from typing import Optional

from decision_model.prompts.base import BaseQuestionPrompt
from decision_model.prompts.contexts.hindsight import validate_action_position
from decision_model.registry import register_question_prompt
from decision_model.types import AnswerSpec, ContextPayload, Position, QuestionPayload, TrajectorySnapshot


DEFAULT_QUESTION = (
    "Given only the task, the current pre-action observation, the provided past "
    "observation/action history, and the current action, what is the probability "
    "that the task will be completed within the remaining step budget if this action is executed first "
    "and the same agent policy then continues? Estimate the success probability "
    "conditioned on this action, not the probability of choosing the action. "
    "The remaining step budget includes the current action."
)
VALUE_QUESTION = "What is " + DEFAULT_QUESTION.split(", what is ", 1)[1]


class ValueQuestion(BaseQuestionPrompt):
    name = "value"

    def build(self, trajectory: TrajectorySnapshot, position: Position,
              context: Optional[ContextPayload] = None) -> QuestionPayload:
        target = validate_action_position(trajectory, position)
        if context is None:
            raise ValueError("value requires context evidence_scope")
        scope = context.materials.get("evidence_scope")
        if scope == "pre_action":
            instructions = DEFAULT_QUESTION
        elif scope == "hindsight_trajectory":
            locator = context.materials.get("target_instructions")
            budget = trajectory.task.max_steps
            if not isinstance(locator, str) or not locator:
                raise ValueError("value requires context target_instructions")
            if type(budget) is not int or budget < 1:
                raise ValueError("value requires a positive integer max_steps")
            instructions = (locator + "\nSteps remaining before max_steps: {} "
                            "(including the current action).\n\nQuestion:\n".format(budget - target)
                            + VALUE_QUESTION)
        else:
            raise ValueError("unsupported value evidence_scope")
        return QuestionPayload(question=instructions, answer=AnswerSpec(kind="score"),
                               materials={"question_key": "value"})


register_question_prompt("value", ValueQuestion)
