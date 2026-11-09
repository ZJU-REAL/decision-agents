"""Step-level advantage: the data interface and the dispatch function.

Two index spaces are kept apart here:

- the **row space** is the N rows currently in training, in current order,
  after filtering, duplication and balance reshuffling; row-aligned data is
  indexed by ``0..N-1``;
- the **record space** is the per-trajectory stored steps, final observations
  and scores, addressed by :class:`StepKey`, and it may keep trajectories that
  no longer have any participating row.

One record may be referenced by zero rows, one row or several duplicated rows.
Advantage implementations consume this shared interface and register their
formula separately from the training integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from decision_model.types import RequestSource


class MissingDataError(RuntimeError):
    """Raised when an advantage implementation needs data the caller did not provide.

    The implementation reports the absence; it must never call a model, build a
    prompt or issue a new evaluation to fill the gap.
    """


def _empty_dict() -> Dict[str, Any]:
    return {}


@dataclass(frozen=True)
class StepKey:
    """Stable address of one trajectory step in the record space."""

    trajectory_id: str
    step_index: int

    def to_dict(self) -> Dict[str, Any]:
        return {"trajectory_id": self.trajectory_id, "step_index": self.step_index}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StepKey":
        return cls(
            trajectory_id=str(data["trajectory_id"]),
            step_index=int(data["step_index"]),
        )


@dataclass(frozen=True)
class Score:
    """One evaluation score, always identified with its config and position.

    ``status`` is ``"ok"`` or ``"error"``; a failed score keeps ``value`` as
    ``None`` rather than pretending to be a zero.
    """

    metric: str
    value: Optional[float]
    status: str
    config_name: str
    position_key: str
    source: Optional[RequestSource] = None

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "status": self.status,
            "config_name": self.config_name,
            "position_key": self.position_key,
            "source": self.source.to_dict() if self.source is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Score":
        source = data.get("source")
        return cls(
            metric=str(data["metric"]),
            value=data.get("value"),
            status=str(data["status"]),
            config_name=str(data["config_name"]),
            position_key=str(data["position_key"]),
            source=RequestSource.from_dict(source) if source is not None else None,
        )


@dataclass(frozen=True)
class ScoreQuery:
    """A filter over scores; any field left ``None`` matches anything."""

    metric: Optional[str] = None
    config_name: Optional[str] = None
    position_key: Optional[str] = None

    def matches(self, score: Score) -> bool:
        """True when ``score`` satisfies every constraint that was set."""
        if self.metric is not None and score.metric != self.metric:
            return False
        if self.config_name is not None and score.config_name != self.config_name:
            return False
        if self.position_key is not None and score.position_key != self.position_key:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "config_name": self.config_name,
            "position_key": self.position_key,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScoreQuery":
        return cls(
            metric=data.get("metric"),
            config_name=data.get("config_name"),
            position_key=data.get("position_key"),
        )


@dataclass(frozen=True)
class StepRecord:
    """One stored step of a trajectory, with whatever scores it collected.

    ``action is None`` marks an observation-boundary record, for example the
    final observation: such a record participates in the record space but
    corresponds to no training row.
    """

    key: StepKey
    observation: str
    action: Optional[str] = None
    next_observation: Optional[str] = None
    candidates: Tuple[str, ...] = ()
    scores: Mapping[str, Score] = field(default_factory=_empty_dict)
    extra: Mapping[str, Any] = field(default_factory=_empty_dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))

    def score(self, query: Optional[ScoreQuery] = None) -> Optional[Score]:
        """First stored score matching ``query``, or ``None``."""
        for score in self.scores.values():
            if query is None or query.matches(score):
                return score
        return None

    def scores_matching(self, query: Optional[ScoreQuery] = None) -> Tuple[Score, ...]:
        """Every stored score matching ``query``, in stored order."""
        return tuple(
            score for score in self.scores.values() if query is None or query.matches(score)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "observation": self.observation,
            "action": self.action,
            "next_observation": self.next_observation,
            "candidates": list(self.candidates),
            "scores": {name: score.to_dict() for name, score in self.scores.items()},
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StepRecord":
        return cls(
            key=StepKey.from_dict(data["key"]),
            observation=str(data.get("observation", "")),
            action=data.get("action"),
            next_observation=data.get("next_observation"),
            candidates=tuple(data.get("candidates", ())),
            scores={
                name: Score.from_dict(score)
                for name, score in (data.get("scores") or {}).items()
            },
            extra=dict(data.get("extra") or {}),
        )


@dataclass(frozen=True)
class RowRecordRef:
    """One entry of the row -> record mapping."""

    row_index: int
    key: StepKey

    def to_dict(self) -> Dict[str, Any]:
        return {"row_index": self.row_index, "key": self.key.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RowRecordRef":
        return cls(row_index=int(data["row_index"]), key=StepKey.from_dict(data["key"]))


@dataclass(frozen=True)
class RowData:
    """Row-aligned named training data, indexed ``0..N-1``.

    ``mask`` is opaque to this layer: it may be a nested sequence or any object
    with a per-row shape, and it is passed through rather than interpreted.
    """

    uid: Tuple[str, ...] = ()
    reward: Tuple[float, ...] = ()
    mask: Any = None
    valid: Optional[Tuple[bool, ...]] = None
    values: Mapping[str, Sequence[Any]] = field(default_factory=_empty_dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "uid", tuple(self.uid))
        object.__setattr__(self, "reward", tuple(float(r) for r in self.reward))
        if self.valid is not None:
            object.__setattr__(self, "valid", tuple(bool(v) for v in self.valid))
        # A fresh dict: mutating the caller's mapping afterwards must not leak
        # into this row-aligned view.
        object.__setattr__(self, "values", dict(self.values))

    def get(self, name: str) -> Optional[Sequence[Any]]:
        """Named row-aligned sequence, or ``None`` when the name is unknown."""
        return self.values.get(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": list(self.uid),
            "reward": list(self.reward),
            "valid": None if self.valid is None else list(self.valid),
            "values": {name: list(seq) for name, seq in self.values.items()},
        }


@dataclass(frozen=True)
class HistoryData:
    """Caller-retained history; absent means explicitly unavailable."""

    entries: Sequence[Any] = ()
    metadata: Mapping[str, Any] = field(default_factory=_empty_dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))

    def __len__(self) -> int:
        return len(self.entries)

    def to_dict(self) -> Dict[str, Any]:
        return {"entries": list(self.entries), "metadata": dict(self.metadata)}


@dataclass(frozen=True)
class AdvantageConfig:
    """Which step-advantage implementation to run, and with what options."""

    name: str
    options: Mapping[str, Any] = field(default_factory=_empty_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "options": dict(self.options)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdvantageConfig":
        return cls(name=str(data["name"]), options=dict(data.get("options") or {}))


@dataclass(frozen=True)
class StepAdvantageResult:
    """Exactly N values, aligned to the current row space.

    ``status`` is **required**: it carries one ``"ok"`` or ``"error"`` entry per
    value, so the default ``()`` would fail the ``len(status) == len(values)``
    check for every non-empty result.

    There is deliberately no token matrix here: expanding a per-row value onto
    tokens is the training adapter's job.
    """

    values: Tuple[float, ...]
    status: Tuple[str, ...]
    detail: Mapping[str, Any] = field(default_factory=_empty_dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        object.__setattr__(self, "status", tuple(self.status))
        if len(self.values) != len(self.status):
            raise ValueError(
                "StepAdvantageResult requires len(status) == len(values); got "
                "len(values)={} and len(status)={}".format(
                    len(self.values), len(self.status)
                )
            )

    @property
    def n_rows(self) -> int:
        return len(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": list(self.values),
            "status": list(self.status),
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StepAdvantageResult":
        return cls(
            values=tuple(data.get("values", ())),
            status=tuple(data.get("status", ())),
            detail=dict(data.get("detail") or {}),
        )


class BaseStepAdvantage(ABC):
    """A step-level advantage implementation over the shared data view."""

    #: Registered component name.
    name: str = ""

    @abstractmethod
    def compute(self, data: Any, config: AdvantageConfig) -> StepAdvantageResult:
        """Compute one value per current training row.

        ``data`` is an :class:`~decision_model.advantage.view.AdvantageDataSource`.
        An implementation picks the data it needs, checks that it is present and
        raises :class:`MissingDataError` when a required field is absent.
        """
        raise NotImplementedError


def compute_step_advantage(data: Any, config: AdvantageConfig) -> StepAdvantageResult:
    """Look up ``config.name`` in the advantage registry and run it.

    Construction goes through exactly one path - ``build_advantage(config.name,
    **dict(config.options))``, the same rule as every other component - so
    ``config.options`` always reaches the constructor. There is no "try with no
    arguments first" fallback: that would silently discard the options for any
    implementation whose constructor has defaults, and it would mask a genuine
    ``TypeError`` raised inside a constructor.

    The result is checked for type and for alignment: a result whose
    ``len(values)`` is not ``data.n_rows`` raises ``ValueError``.
    """
    from decision_model.registry import build_advantage

    implementation = build_advantage(config.name, **dict(config.options))
    result = implementation.compute(data, config)
    if not isinstance(result, StepAdvantageResult):
        raise TypeError(
            "advantage {!r} returned {!r}; expected StepAdvantageResult".format(
                config.name, type(result).__name__
            )
        )
    if len(result.values) != data.n_rows:
        raise ValueError(
            "advantage {!r} returned {} value(s) but the row space has {} row(s); "
            "a result must carry exactly one value per row".format(
                config.name, len(result.values), data.n_rows
            )
        )
    return result


def require_row_values(data: Any, name: str) -> Sequence[Any]:
    """Fetch a named row-aligned sequence or raise :class:`MissingDataError`."""
    values = data.optional_row_values(name)
    if values is None:
        raise MissingDataError(
            "required row-aligned data {!r} is unavailable; the caller must supply it".format(
                name
            )
        )
    return values


def require_record(data: Any, key: StepKey) -> StepRecord:
    """Fetch a record or raise :class:`MissingDataError` naming the key."""
    record = data.record_optional(key)
    if record is None:
        raise MissingDataError(
            "required record {} is unavailable in the record space".format(key)
        )
    return record


def require_score(
    data: Any,
    key: StepKey,
    query: Optional[ScoreQuery] = None,
) -> Score:
    """Fetch one score or raise :class:`MissingDataError` naming key and query."""
    score = data.score(key, query)
    if score is None:
        raise MissingDataError(
            "required score for record {} with query {} is unavailable".format(key, query)
        )
    return score


def row_indices_with_uid(uid: Tuple[str, ...], group: str) -> List[int]:
    """Row indices belonging to one training group, in row order."""
    return [index for index, value in enumerate(uid) if value == group]
