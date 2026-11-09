"""Decision-guided agents: shared data contracts and component interfaces.

Importing this package does not import the training framework: the public
surface below is pure standard library. Custom backends,
dataset prompts, contexts, questions and advantage implementations plug in
through the registry.
"""

from __future__ import annotations

from decision_model.advantage.base import (
    AdvantageConfig,
    BaseStepAdvantage,
    HistoryData,
    MissingDataError,
    RowData,
    RowRecordRef,
    Score,
    ScoreQuery,
    StepAdvantageResult,
    StepKey,
    StepRecord,
    compute_step_advantage,
)
from decision_model.advantage.view import AdvantageDataSource, InMemoryAdvantageData
from decision_model.config import ComponentConfig, DecisionConfig, EvaluationConfig
from decision_model.judge import (
    EvaluationBatch,
    EvaluationItem,
    Judge,
    expand_positions,
)
from decision_model.models.base import BaseDecisionModel
from decision_model.prompts.base import (
    BaseContextPrompt,
    BaseDatasetPrompt,
    BaseQuestionPrompt,
)
from decision_model.prompts.builder import PromptBuilder
from decision_model.registry import (
    build_advantage,
    build_context_prompt,
    build_dataset_prompt,
    build_model,
    build_question_prompt,
    register_advantage,
    register_context_prompt,
    register_dataset_prompt,
    register_model,
    register_question_prompt,
)
from decision_model.types import (
    AnswerSpec,
    ChoiceOption,
    ContextPayload,
    DecisionRequest,
    DecisionResult,
    Position,
    QuestionPayload,
    RequestSource,
    TaskSpec,
    TrajectorySnapshot,
    make_request_id,
    snapshot_from_dict,
    snapshot_to_dict,
    split_request_id,
)

__all__ = [
    # types
    "TaskSpec",
    "TrajectorySnapshot",
    "Position",
    "AnswerSpec",
    "ChoiceOption",
    "ContextPayload",
    "QuestionPayload",
    "RequestSource",
    "DecisionRequest",
    "DecisionResult",
    "snapshot_from_dict",
    "snapshot_to_dict",
    "make_request_id",
    "split_request_id",
    # model
    "BaseDecisionModel",
    "FakeDecisionModel",
    "DecisionAPIModel",
    "PromptedLLM",
    "ConstrainedLLM",
    # prompts
    "BaseDatasetPrompt",
    "BaseContextPrompt",
    "BaseQuestionPrompt",
    "PromptBuilder",
    # config
    "EvaluationConfig",
    "ComponentConfig",
    "DecisionConfig",
    # evaluation
    "Judge",
    "EvaluationItem",
    "EvaluationBatch",
    "expand_positions",
    # actor and environment execution (loaded on demand)
    "Actor",
    "ActorDecision",
    "ActorEpisode",
    "EnvironmentStep",
    "run_episode",
    # registry
    "register_model",
    "register_dataset_prompt",
    "register_context_prompt",
    "register_question_prompt",
    "register_advantage",
    "build_model",
    "build_dataset_prompt",
    "build_context_prompt",
    "build_question_prompt",
    "build_advantage",
    # advantage
    "StepKey",
    "Score",
    "ScoreQuery",
    "StepRecord",
    "RowRecordRef",
    "RowData",
    "HistoryData",
    "AdvantageConfig",
    "StepAdvantageResult",
    "BaseStepAdvantage",
    "compute_step_advantage",
    "AdvantageDataSource",
    "InMemoryAdvantageData",
    "MissingDataError",
]


def __getattr__(name):
    # Model implementations remain opt-in; no service adapters load at import.
    if name in {"FakeDecisionModel", "DecisionAPIModel", "PromptedLLM", "ConstrainedLLM"}:
        from decision_model import models
        return getattr(models, name)
    if name in {"Actor", "ActorDecision"}:
        from decision_model import actor
        return getattr(actor, name)
    if name in {"ActorEpisode", "EnvironmentStep", "run_episode"}:
        from decision_model import runner
        return getattr(runner, name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
