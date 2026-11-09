"""Combines the prompt components into :class:`DecisionRequest` objects.

The builder is mechanical: for one position it asks the context prompt first,
hands that result to the question prompt, and packages both with the call-side
association metadata. It never enumerates evaluation positions on its own and
never calls a model.
"""

from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Sequence

from decision_model.prompts.base import (
    BaseContextPrompt,
    BaseDatasetPrompt,
    BaseQuestionPrompt,
)
from decision_model.types import (
    AnswerSpec,
    ContextPayload,
    DecisionRequest,
    Position,
    QuestionPayload,
    RequestSource,
    TrajectorySnapshot,
    make_request_id,
)


def _empty_context() -> ContextPayload:
    return ContextPayload(text="")


def _empty_question() -> QuestionPayload:
    return QuestionPayload(question="", answer=AnswerSpec(kind="text"))


class PromptBuilder:
    """Turns ``(trajectory, position)`` pairs into decision requests."""

    def __init__(
        self,
        dataset_prompt: Optional[BaseDatasetPrompt] = None,
        context_prompt: Optional[BaseContextPrompt] = None,
        question_prompt: Optional[BaseQuestionPrompt] = None,
    ) -> None:
        self.dataset_prompt = dataset_prompt
        self.context_prompt = context_prompt
        self.question_prompt = question_prompt

    def build(
        self,
        trajectory: TrajectorySnapshot,
        position: Position,
        *,
        request_id: str,
        config_name: str,
        batch_id: str,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> DecisionRequest:
        """Build one request for ``position``.

        The context prompt runs first and its payload is passed to the question
        prompt, so a question can use or ignore the context but both always see
        the same trajectory and position.
        """
        if self.dataset_prompt is not None:
            trajectory = self.dataset_prompt.prepare_for_prompt(trajectory)

        if self.context_prompt is None:
            context = _empty_context()
        else:
            context = self.context_prompt.build(trajectory, position)

        if self.question_prompt is None:
            question = _empty_question()
        else:
            question = self.question_prompt.build(trajectory, position, context)

        source = RequestSource(
            dataset=trajectory.dataset,
            trajectory_id=trajectory.trajectory_id,
            position=position,
            config_name=config_name,
            batch_id=batch_id,
        )
        return DecisionRequest(
            request_id=request_id,
            context=context,
            question=question,
            source=source,
            meta=dict(meta) if meta else {},
        )

    def build_many(
        self,
        trajectory: TrajectorySnapshot,
        positions: Sequence[Position],
        *,
        request_id_fn: Callable[[Position], str],
        config_name: str,
        batch_id: str,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> List[DecisionRequest]:
        """One request per position, in the order the caller supplied them."""
        return [
            self.build(
                trajectory,
                position,
                request_id=request_id_fn(position),
                config_name=config_name,
                batch_id=batch_id,
                meta=meta,
            )
            for position in positions
        ]


def default_request_id_fn(
    batch_id: str,
    config_name: str,
    trajectory_id: str,
) -> Callable[[Position], str]:
    """Convenience factory for the standard request id rule."""

    def _request_id(position: Position) -> str:
        return make_request_id(batch_id, config_name, trajectory_id, position.key)

    return _request_id
