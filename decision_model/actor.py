"""Candidate actor orchestration, independent of any concrete model transport."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from decision_model.models.base import BaseDecisionModel
from decision_model.prompts.base import BaseContextPrompt, BaseDatasetPrompt, BaseQuestionPrompt
from decision_model.prompts.builder import PromptBuilder
from decision_model.prompts.contexts.current import CurrentContextPrompt
from decision_model.prompts.questions.choice import ChoiceQuestion
from decision_model.types import DecisionRequest, DecisionResult, Position, TrajectorySnapshot, make_request_id


@dataclass(frozen=True)
class ActorDecision:
    status: str
    action: Optional[str] = None
    choice_id: Optional[str] = None
    error: Optional[str] = None
    request: Optional[DecisionRequest] = None
    result: Optional[DecisionResult] = None

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"


class Actor:
    """Return an exact candidate command or an explicit failure, never fallback.

    Any dataset may supply a standard snapshot with executable candidates. An
    optional dataset adapter may expose actor_candidates(snapshot, position) to
    filter that view. Without this hook all supplied candidates remain eligible.
    Context and question receive the same private view; records are not changed.
    """

    def __init__(self, model: BaseDecisionModel,
                 dataset_prompt: Optional[BaseDatasetPrompt] = None,
                 context_prompt: Optional[BaseContextPrompt] = None,
                 question_prompt: Optional[BaseQuestionPrompt] = None,
                 config_name: str = "choice") -> None:
        self.model = model
        self.dataset_prompt = dataset_prompt
        self.context_prompt = context_prompt if context_prompt is not None else CurrentContextPrompt()
        self.question_prompt = question_prompt if question_prompt is not None else ChoiceQuestion()
        if not isinstance(config_name, str) or not config_name:
            raise ValueError("config_name must be a non-empty string")
        self.config_name = config_name
        self.builder = PromptBuilder(self.dataset_prompt, self.context_prompt, self.question_prompt)

    def act(self, snapshot: TrajectorySnapshot, *, position: Optional[Position] = None,
            batch_id: str = "actor", timeout: Optional[float] = None) -> ActorDecision:
        if self.dataset_prompt is not None and self.dataset_prompt.dataset != snapshot.dataset:
            return ActorDecision("error", error="actor_dataset_mismatch")
        position = position if position is not None else Position.decision(snapshot.n_steps)
        try:
            if (position.kind != "decision" or position.end is not None
                    or type(position.index) is not int or not 0 <= position.index <= snapshot.n_steps):
                raise ValueError("actor requires a decision position")
            original = snapshot.candidates_at(position.index)
            select = getattr(self.dataset_prompt, "actor_candidates", None)
            selected = select(snapshot, position) if callable(select) else original
            if isinstance(selected, (str, bytes)):
                raise ValueError("candidates must be a sequence of commands")
            commands = tuple(selected)
            if any(not isinstance(command, str) or not command.strip() or command not in original
                   for command in commands):
                raise ValueError("candidate selection must retain supplied executable commands")
            if not commands:
                return ActorDecision("error", error="no_candidates")
            candidate_view = list(snapshot.candidates)
            candidate_view[position.index] = commands
            view = replace(snapshot, candidates=tuple(candidate_view))
            request = self.builder.build(
                view, position,
                request_id=make_request_id(batch_id, self.config_name, snapshot.trajectory_id, position.key),
                config_name=self.config_name, batch_id=batch_id,
            )
            if request.question.answer.kind != "choice":
                return ActorDecision("error", error="unsupported_actor_answer", request=request)
            options = request.question.answer.choices
            criteria = {option.option_id: option.text for option in options}
            if (not options or len(criteria) != len(options)
                    or any(not isinstance(option.option_id, str) or not option.option_id
                           or option.text not in commands for option in options)):
                return ActorDecision("error", error="invalid_actor_choices", request=request)
        except (ValueError, TypeError, IndexError, AttributeError):
            return ActorDecision("error", error="invalid_actor_request")
        try:
            result = self.model.complete(request, timeout=timeout)
        except Exception:
            # Model exceptions can include credential-bearing transport details.
            return ActorDecision("error", error="model_error", request=request)
        if not isinstance(result, DecisionResult) or result.request_id != request.request_id:
            return ActorDecision("error", error="invalid_model_result", request=request)
        if result.status not in ("ok", "error"):
            return ActorDecision("error", error="invalid_model_status", request=request)
        if not result.is_ok:
            return ActorDecision("error", error=result.error or "model_error", request=request, result=result)
        if not isinstance(result.choice_id, str) or result.choice_id not in criteria:
            return ActorDecision("error", error="invalid_choice_id", request=request, result=result)
        return ActorDecision("ok", action=criteria[result.choice_id], choice_id=result.choice_id,
                             request=request, result=result)
