"""Native candidate selection; dataset filtering precedes this component."""

from __future__ import annotations

from typing import Optional

from decision_model.prompts.base import BaseQuestionPrompt
from decision_model.prompts.utils import validate_decision_position
from decision_model.registry import register_question_prompt
from decision_model.types import AnswerSpec, ChoiceOption, ContextPayload, Position, QuestionPayload, TrajectorySnapshot


CHOICE_QUESTION = "Which available action should be executed next to maximize the chance of completing the task within the remaining action budget, including this action? Use only the task, observed history, current observation, and the available actions."


class ChoiceQuestion(BaseQuestionPrompt):
    name = "choice"

    def build(self, trajectory: TrajectorySnapshot, position: Position,
              context: Optional[ContextPayload] = None) -> QuestionPayload:
        index = validate_decision_position(trajectory, position)
        try:
            candidates = trajectory.candidates_at(index)
        except IndexError as exc:
            raise ValueError("candidate snapshot is absent") from exc
        if any(not isinstance(command, str) or not command.strip() for command in candidates):
            raise ValueError("candidates must contain non-empty strings")
        if not 1 <= len(candidates) <= 255:
            raise ValueError("unsupported_candidate_count")
        # Repeated candidate commands retain independent option IDs.
        options = tuple(ChoiceOption("a{}".format(k), command)
                        for k, command in enumerate(sorted(candidates)))
        return QuestionPayload(question=CHOICE_QUESTION,
                               answer=AnswerSpec(kind="choice", choices=options),
                               materials={"question_key": "next_action"})


register_question_prompt("choice", ChoiceQuestion)
