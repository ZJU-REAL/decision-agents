"""Progress: whether an action advances the goal, independent of the dataset."""

from __future__ import annotations

from typing import Optional

from decision_model.prompts.base import BaseQuestionPrompt
from decision_model.registry import register_question_prompt
from decision_model.types import AnswerSpec, ContextPayload, Position, QuestionPayload, TrajectorySnapshot


PROGRESS_QUESTION = """Did the target action make a useful contribution toward completing
the goal?

Later failure does not erase earlier useful progress. Later success
does not make an unnecessary detour useful."""


class ProgressQuestion(BaseQuestionPrompt):
    """Use the context's locator without interpreting a dataset or context name."""

    name = "progress"

    def build(
        self,
        trajectory: TrajectorySnapshot,
        position: Position,
        context: Optional[ContextPayload] = None,
    ) -> QuestionPayload:
        if (position.kind != "action" or position.end is not None
                or type(position.index) is not int or not 0 <= position.index < trajectory.n_steps):
            raise ValueError("progress requires an executed-action position")
        target = None if context is None else context.materials.get("target_instructions")
        if not isinstance(target, str) or not target:
            raise ValueError("progress requires context target_instructions")
        return QuestionPayload(
            question=target + "\n\nQuestion:\n" + PROGRESS_QUESTION,
            answer=AnswerSpec(kind="score", scale=(0.0, 1.0)),
            materials={"question_key": "progress"},
        )


register_question_prompt("progress", ProgressQuestion)
