"""The three prompt interfaces: dataset, context and question.

The split is by responsibility, not by experiment: a dataset prompt adapts a
raw record, a context prompt decides what evidence the model sees for one
position, and a question prompt states what is being asked about that same
position. No base class here contains a dataset-specific or provider-specific
branch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

from decision_model.types import (
    ContextPayload,
    Position,
    QuestionPayload,
    TrajectorySnapshot,
)


class BaseDatasetPrompt(ABC):
    """Adapts one raw dataset record into a :class:`TrajectorySnapshot`."""

    #: Dataset name this prompt adapts, e.g. ``"alfworld"``.
    dataset: str = ""

    @abstractmethod
    def adapt(self, record: Mapping[str, Any]) -> TrajectorySnapshot:
        """Turn a raw dataset record into the shared trajectory shape."""
        raise NotImplementedError

    def prepare_for_prompt(self, trajectory: TrajectorySnapshot) -> TrajectorySnapshot:
        """Return model-visible evidence without modifying the original record.

        Dataset adapters may redact environment labels here. Both judges
        and training use this hook, including for already-created snapshots.
        """
        return trajectory


class BaseContextPrompt(ABC):
    """Builds the context a model sees for one evaluation position."""

    #: Registered component name.
    name: str = ""

    @abstractmethod
    def build(
        self,
        trajectory: TrajectorySnapshot,
        position: Position,
    ) -> ContextPayload:
        """Return the context for ``position`` within ``trajectory``.

        The position is always supplied by the caller; a context prompt never
        enumerates its own evaluation positions.
        """
        raise NotImplementedError


class BaseQuestionPrompt(ABC):
    """Builds the question asked about one evaluation position."""

    #: Registered component name.
    name: str = ""

    @abstractmethod
    def build(
        self,
        trajectory: TrajectorySnapshot,
        position: Position,
        context: Optional[ContextPayload] = None,
    ) -> QuestionPayload:
        """Return the question for ``position``.

        ``context`` is whatever the context prompt produced for the *same*
        ``(trajectory, position)`` pair. A question may use it, ignore it and
        read the trajectory directly, or mix the two.
        """
        raise NotImplementedError
