"""Position expansion, evaluation items and the single-batch judge.

There is exactly one place where evaluation requests are expanded: the
judge, from the rule named in the evaluation config. Context and question
prompts never enumerate requests of their own; they are handed the same
``(trajectory, position)`` pair for each expanded position.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from decision_model.config import EvaluationConfig
from decision_model.prompts.base import (
    BaseContextPrompt,
    BaseDatasetPrompt,
    BaseQuestionPrompt,
)
from decision_model.prompts.builder import PromptBuilder
from decision_model.types import (
    DecisionRequest,
    DecisionResult,
    Position,
    TrajectorySnapshot,
    make_request_id,
)

#: Prefix of an auto-generated batch id.
BATCH_ID_PREFIX = "decision_model"


def new_batch_id() -> str:
    """A fresh unique batch id; pass it twice to keep request IDs reproducible."""
    return "{}-{}".format(BATCH_ID_PREFIX, uuid.uuid4().hex)


def expand_positions(
    rule: str,
    trajectory: TrajectorySnapshot,
    options: Optional[Mapping[str, Any]] = None,
) -> List[Position]:
    """Expand an evaluation rule into explicit positions for one trajectory.

    Supported rules:

    - ``"all_steps"``: one ``action`` position per executed action, so a
      3-action trajectory yields exactly 3 positions.
    - ``"all_boundaries"``: one ``boundary`` position per observation,
      including the final one.
    - ``"single"``: exactly one position, from ``options["kind"]`` (default
      ``"action"``) and ``options["index"]``.
    - ``"trajectory"``: exactly one whole-trajectory position.

    An unknown rule raises ``ValueError``.
    """
    opts: Mapping[str, Any] = options if options is not None else {}

    if rule == "all_steps":
        return [Position.action(k) for k in range(trajectory.n_steps)]

    if rule == "all_boundaries":
        return [Position.boundary(k) for k in range(trajectory.n_observations)]

    if rule == "single":
        kind = opts.get("kind", "action")
        index = opts.get("index")
        end = opts.get("end")
        if index is not None and end is not None:
            return [Position(kind=kind, index=index, end=end)]
        if index is not None:
            return [Position(kind=kind, index=index)]
        return [Position(kind=kind)]

    if rule == "trajectory":
        return [Position.trajectory()]

    raise ValueError(
        "unknown expansion rule {!r}; available rules: all_steps, all_boundaries, "
        "single, trajectory".format(rule)
    )


@dataclass(frozen=True)
class EvaluationItem:
    """One evaluated position: its identity, its score and its status."""

    request_id: str
    config_name: str
    metric: str
    dataset: str
    trajectory_id: str
    position: Position
    batch_id: str
    status: str
    value: Optional[float] = None
    choice_id: Optional[str] = None
    text: Optional[str] = None
    error: Optional[str] = None

    @property
    def position_key(self) -> str:
        return self.position.key

    @property
    def item_key(self) -> str:
        """``"{trajectory_id}:{position_key}"``."""
        return "{}:{}".format(self.trajectory_id, self.position_key)

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "config_name": self.config_name,
            "metric": self.metric,
            "dataset": self.dataset,
            "trajectory_id": self.trajectory_id,
            "position": self.position.to_dict(),
            "position_key": self.position_key,
            "batch_id": self.batch_id,
            "status": self.status,
            "value": self.value,
            "choice_id": self.choice_id,
            "text": self.text,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationItem":
        return cls(
            request_id=str(data["request_id"]),
            config_name=str(data["config_name"]),
            metric=str(data["metric"]),
            dataset=str(data["dataset"]),
            trajectory_id=str(data["trajectory_id"]),
            position=Position.from_dict(data["position"]),
            batch_id=str(data["batch_id"]),
            status=str(data["status"]),
            value=data.get("value"),
            choice_id=data.get("choice_id"),
            text=data.get("text"),
            error=data.get("error"),
        )


@dataclass(frozen=True)
class EvaluationBatch:
    """The items produced by one :meth:`Judge.evaluate` call."""

    config_name: str
    batch_id: str
    items: Tuple[EvaluationItem, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))

    def __len__(self) -> int:
        return len(self.items)

    def by_position_key(self) -> Dict[str, EvaluationItem]:
        """Items keyed by ``"{trajectory_id}:{position_key}"``."""
        return {item.item_key: item for item in self.items}

    def by_item_key(self) -> Dict[str, EvaluationItem]:
        """Items keyed by ``request_id``."""
        return {item.request_id: item for item in self.items}

    def values(self) -> List[Optional[float]]:
        """One entry per item, in item order; ``None`` where the item failed."""
        return [item.value for item in self.items]

    def scores(self, metric: Optional[str] = None) -> List[float]:
        """Successful values only, filtered by metric when one is given."""
        return [
            item.value
            for item in self.items
            if item.is_ok and item.value is not None and (metric is None or item.metric == metric)
        ]

    def ok_items(self) -> Tuple[EvaluationItem, ...]:
        return tuple(item for item in self.items if item.is_ok)

    def failures(self) -> Tuple[EvaluationItem, ...]:
        return tuple(item for item in self.items if not item.is_ok)

    def metrics(self) -> Tuple[str, ...]:
        """Distinct metric names present, in first-seen order."""
        seen: List[str] = []
        for item in self.items:
            if item.metric not in seen:
                seen.append(item.metric)
        return tuple(seen)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_name": self.config_name,
            "batch_id": self.batch_id,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationBatch":
        return cls(
            config_name=str(data["config_name"]),
            batch_id=str(data["batch_id"]),
            items=tuple(EvaluationItem.from_dict(item) for item in data.get("items", ())),
        )


#: ``positions`` accepted by :meth:`Judge.evaluate`.
PositionsArg = Union[
    Mapping[str, Sequence[Position]],
    Sequence[Position],
    Sequence[Sequence[Position]],
]


class Judge:
    """Expands positions, builds requests, and associates results by request id.

    The judge never computes advantage and never imports the training
    framework; it only turns trajectories into a batch of evaluation items.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        *,
        model: Any,
        dataset_prompt: Optional[BaseDatasetPrompt] = None,
        context_prompt: Optional[BaseContextPrompt] = None,
        question_prompt: Optional[BaseQuestionPrompt] = None,
        builder: Optional[PromptBuilder] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.dataset_prompt = dataset_prompt
        self.context_prompt = context_prompt
        self.question_prompt = question_prompt
        self.builder = (
            builder
            if builder is not None
            else PromptBuilder(
                dataset_prompt=dataset_prompt,
                context_prompt=context_prompt,
                question_prompt=question_prompt,
            )
        )
        #: Requests sent in the most recent ``evaluate`` call. Per-call, never
        #: accumulated across calls.
        self.last_requests: List[DecisionRequest] = []
        #: Result objects returned by the model in the most recent call.
        self.last_results: List[DecisionResult] = []

    # ---- position resolution ---------------------------------------------

    def _resolve_positions(
        self,
        trajectory: TrajectorySnapshot,
        positions: Optional[PositionsArg],
        index: int,
        n_trajectories: int,
    ) -> List[Position]:
        """Positions to evaluate for one snapshot.

        Three shapes are honoured when ``positions`` is supplied:

        1. a ``Mapping`` keyed by ``trajectory_id`` (preferred) or by ``dataset``
           - a snapshot matching neither is an explicit error;
        2. a flat ``Sequence[Position]``, legal only for a single snapshot;
        3. a ``Sequence[Sequence[Position]]`` with one inner sequence per
           snapshot, in the same order.
        """
        if positions is None:
            return expand_positions(
                self.config.expansion,
                trajectory,
                self.config.options.get("expansion_options"),
            )
        if isinstance(positions, Mapping):
            if trajectory.trajectory_id in positions:
                return list(positions[trajectory.trajectory_id])
            if trajectory.dataset in positions:
                return list(positions[trajectory.dataset])
            raise ValueError(
                "positions mapping matches neither the trajectory id {!r} nor the "
                "dataset {!r}; available keys: {}".format(
                    trajectory.trajectory_id,
                    trajectory.dataset,
                    ", ".join(repr(key) for key in positions) or "<none>",
                )
            )

        seq = list(positions)
        # An empty sequence carries no inner sequence to inspect, so it is read
        # as the flat shape: legal for one snapshot, an error for several.
        if not seq or isinstance(seq[0], Position):
            if n_trajectories > 1:
                raise ValueError(
                    "a flat position sequence cannot be applied to {} trajectories; "
                    "supply a mapping from trajectory id to positions, or one "
                    "sequence per trajectory".format(n_trajectories)
                )
            return seq
        if len(seq) != n_trajectories:
            raise ValueError(
                "positions has {} entr(ies) but {} trajectories were supplied; supply "
                "exactly one position sequence per trajectory, in the same order".format(
                    len(seq), n_trajectories
                )
            )
        return list(seq[index])

    # ---- evaluation ------------------------------------------------------

    def evaluate(
        self,
        trajectories: Any,
        *,
        batch_id: Optional[str] = None,
        positions: Optional[PositionsArg] = None,
    ) -> EvaluationBatch:
        """Evaluate one snapshot or a sequence of snapshots.

        One request is built per expanded position; the model is called once
        with all of them; each result is then associated back to its item by
        ``request_id``, never by return order. A request with no result, and a
        result with an unknown ``request_id``, both become explicit error items.

        Two items in one call must never share a ``request_id``: a
        ``request_id`` identifies exactly one item, and ``by_item_key()`` must
        never silently collapse rows onto one another. Colliding ids therefore
        raise ``ValueError`` before any model call - in practice this means the
        same ``trajectory_id`` appearing twice in one batch.
        """
        resolved_batch_id = batch_id if batch_id is not None else new_batch_id()
        snapshots = self._as_sequence(trajectories)

        requests: List[DecisionRequest] = []
        identities: List[Tuple[str, str, Position]] = []
        seen_request_ids: Dict[str, str] = {}
        for index, snapshot in enumerate(snapshots):
            for position in self._resolve_positions(
                snapshot, positions, index, len(snapshots)
            ):
                request_id = make_request_id(
                    resolved_batch_id,
                    self.config.name,
                    snapshot.trajectory_id,
                    position.key,
                )
                # Detect the collision before building anything, so a colliding
                # call never reaches the model.
                if request_id in seen_request_ids:
                    raise ValueError(
                        "request id {!r} was produced twice in one evaluate call "
                        "(first by trajectory_id {!r}, then by trajectory_id {!r}); a "
                        "request id identifies exactly one item, so a colliding id "
                        "means the same trajectory_id appears twice in this "
                        "batch".format(
                            request_id,
                            seen_request_ids[request_id],
                            snapshot.trajectory_id,
                        )
                    )
                seen_request_ids[request_id] = snapshot.trajectory_id
                request = self.builder.build(
                    snapshot,
                    position,
                    request_id=request_id,
                    config_name=self.config.name,
                    batch_id=resolved_batch_id,
                )
                requests.append(request)
                identities.append(
                    (snapshot.dataset, snapshot.trajectory_id, position)
                )

        self.last_requests = list(requests)

        if requests:
            results = list(self.model.complete_many(requests))
        else:
            results = []
        self.last_results = list(results)

        by_request_id: Dict[str, DecisionResult] = {}
        ordered_results: List[DecisionResult] = []
        for result in results:
            if not isinstance(result, DecisionResult):
                raise TypeError(
                    "model returned {!r}; expected DecisionResult".format(
                        type(result).__name__
                    )
                )
            ordered_results.append(result)
            by_request_id.setdefault(result.request_id, result)

        known_ids = {request.request_id for request in requests}
        items: List[EvaluationItem] = []

        for request, identity in zip(requests, identities):
            dataset, trajectory_id, position = identity
            result = by_request_id.get(request.request_id)
            items.append(
                self._item(
                    request_id=request.request_id,
                    dataset=dataset,
                    trajectory_id=trajectory_id,
                    position=position,
                    batch_id=resolved_batch_id,
                    result=result,
                )
            )

        for result in ordered_results:
            if result.request_id in known_ids:
                continue
            # An unrecognised id is an error item that keeps the id verbatim; the
            # judge never guesses the components back out of it.
            items.append(
                self._item(
                    request_id=result.request_id,
                    dataset="",
                    trajectory_id="",
                    position=Position(kind="unknown"),
                    batch_id=resolved_batch_id,
                    result=result,
                    unknown=True,
                )
            )

        return EvaluationBatch(
            config_name=self.config.name,
            batch_id=resolved_batch_id,
            items=tuple(items),
        )

    def _item(
        self,
        *,
        request_id: str,
        dataset: str,
        trajectory_id: str,
        position: Position,
        batch_id: str,
        result: Optional[DecisionResult],
        unknown: bool = False,
        config_name: Optional[str] = None,
    ) -> EvaluationItem:
        common = {
            "request_id": request_id,
            "config_name": config_name or self.config.name,
            "metric": self.config.metric,
            "dataset": dataset,
            "trajectory_id": trajectory_id,
            "position": position,
            "batch_id": batch_id,
        }
        if result is None:
            return EvaluationItem(
                status="error",
                value=None,
                error=(
                    "model returned no result for request {}; the result set is "
                    "missing this request id".format(request_id)
                ),
                **common,
            )
        if unknown:
            return EvaluationItem(
                status="error",
                value=None,
                error=(
                    "model returned a result for unknown request id {}; this result "
                    "matches no expanded position".format(request_id)
                ),
                **common,
            )
        if not result.is_ok:
            return EvaluationItem(
                status="error",
                value=None,
                choice_id=result.choice_id,
                text=result.text,
                error=result.error or "decision model reported a failure",
                **common,
            )
        return EvaluationItem(
            status="ok",
            value=result.value,
            choice_id=result.choice_id,
            text=result.text,
            error=None,
            **common,
        )

    @staticmethod
    def _as_sequence(trajectories: Any) -> List[TrajectorySnapshot]:
        if isinstance(trajectories, TrajectorySnapshot):
            return [trajectories]
        snapshots = list(trajectories)
        for snapshot in snapshots:
            if not isinstance(snapshot, TrajectorySnapshot):
                raise TypeError(
                    "evaluate expects TrajectorySnapshot objects, got {!r}".format(
                        type(snapshot).__name__
                    )
                )
        return snapshots
