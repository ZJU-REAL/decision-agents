"""Shared data contracts for the decision model package.

Everything here is a plain, frozen dataclass with JSON-compatible serialization
helpers. No module in this package may import the training framework; the types
below are the only vocabulary the model, prompt and advantage layers share.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple


def _empty_dict() -> Dict[str, Any]:
    """Fresh empty dict, used as a dataclass default factory."""
    return {}


def _mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize an optional mapping into a plain dict."""
    return dict(value) if value else {}


@dataclass(frozen=True)
class TaskSpec:
    """Dataset-specific episode identity and task text."""

    task_id: str
    task: str
    max_steps: Optional[int] = None
    extra: Mapping[str, Any] = field(default_factory=_empty_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task": self.task,
            "max_steps": self.max_steps,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskSpec":
        return cls(
            task_id=str(data["task_id"]),
            task=str(data["task"]),
            max_steps=data.get("max_steps"),
            extra=_mapping(data.get("extra")),
        )


@dataclass(frozen=True)
class TrajectorySnapshot:
    """One adapted trajectory.

    A single type expresses a complete rollout, an empty history at the first
    decision, and a mid-episode actor snapshot:

    - ``observations[k]`` is what the agent saw before choosing ``actions[k]``;
      ``observations[len(actions)]`` is the final observation.
    - ``candidates[k]`` is the candidate set captured *before* ``actions[k]``
      ran. ``candidates[len(actions)]``, when present, is the pending decision's
      candidate set with no action chosen yet.
    - ``history`` is a tuple of ``(observation, executed action)`` pairs; it
      never contains policy prompts or ``<think>``/``<action>`` responses.
    """

    trajectory_id: str
    dataset: str
    task: TaskSpec
    observations: Tuple[str, ...]
    actions: Tuple[str, ...]
    candidates: Tuple[Tuple[str, ...], ...] = ()
    policy_responses: Tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=_empty_dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "candidates", tuple(tuple(c) for c in self.candidates))
        object.__setattr__(self, "policy_responses", tuple(self.policy_responses))
        # ``extra`` stays a dict, but a fresh one: a snapshot built directly from
        # a caller's dict must not alias it.
        object.__setattr__(self, "extra", dict(self.extra) or {})

        if not self.observations:
            raise ValueError(
                "TrajectorySnapshot requires at least one observation; got an empty "
                "observations tuple for trajectory_id={!r}".format(self.trajectory_id)
            )
        if len(self.actions) != len(self.observations) - 1:
            raise ValueError(
                "TrajectorySnapshot requires len(actions) == len(observations) - 1; "
                "got len(actions)={} and len(observations)={} for trajectory_id={!r}".format(
                    len(self.actions), len(self.observations), self.trajectory_id
                )
            )
        if len(self.candidates) not in (0, len(self.actions), len(self.actions) + 1):
            raise ValueError(
                "TrajectorySnapshot requires len(candidates) to be zero, len(actions) or "
                "len(actions) + 1; got len(candidates)={} and len(actions)={} for "
                "trajectory_id={!r}".format(
                    len(self.candidates), len(self.actions), self.trajectory_id
                )
            )
        # Raw candidate snapshots preserve order, duplicates, and empty entries.
        # Dataset/actor views decide which commands are selectable. An entirely
        # absent sequence is valid for judge-only records.

    # ---- shape -----------------------------------------------------------

    @property
    def n_steps(self) -> int:
        """Number of executed actions."""
        return len(self.actions)

    @property
    def n_observations(self) -> int:
        """Number of observations, including the final one."""
        return len(self.observations)

    @property
    def current_step(self) -> int:
        """Index of the decision that has not been executed yet."""
        return len(self.actions)

    @property
    def is_empty_history(self) -> bool:
        """True when no action has been executed yet."""
        return len(self.actions) == 0

    # ---- data ------------------------------------------------------------

    @property
    def history(self) -> Tuple[Tuple[str, str], ...]:
        """Completed ``(observation, executed action)`` pairs, in order."""
        return tuple(
            (self.observations[k], self.actions[k]) for k in range(len(self.actions))
        )

    @property
    def current_observation(self) -> str:
        """Observation the agent is looking at right now."""
        return self.observations[-1]

    @property
    def final_observation(self) -> str:
        """Last observation of the recorded rollout."""
        return self.observations[-1]

    @property
    def pending_candidates(self) -> Tuple[str, ...]:
        """Candidate set for the pending decision, or ``()`` when none is captured."""
        if len(self.candidates) > len(self.actions):
            return self.candidates[-1]
        return ()

    def candidates_at(self, index: int) -> Tuple[str, ...]:
        """Candidate snapshot captured before ``actions[index]``.

        Negative indices count from the end; an absent snapshot raises
        ``IndexError`` naming the index the caller passed.
        """
        normalised = index + len(self.candidates) if index < 0 else index
        if normalised < 0 or normalised >= len(self.candidates):
            raise IndexError(
                "candidate snapshot at index {} is absent for trajectory_id={!r}; "
                "{} candidate tuple(s) available".format(
                    index, self.trajectory_id, len(self.candidates)
                )
            )
        return self.candidates[normalised]

    # ---- construction ----------------------------------------------------

    def with_action(
        self,
        observation: str,
        action: str,
        *,
        candidates: Tuple[str, ...] = (),
    ) -> "TrajectorySnapshot":
        """Return a new snapshot extended by one executed step.

        A previously pending candidate snapshot becomes the executed action's
        pre-action snapshot. In that case, nonempty ``candidates`` supplies the
        *next* pending snapshot. Without a pending snapshot, ``candidates`` is
        the snapshot from before this newly executed action. Missing historical
        candidates are padded with empty tuples when recording begins later.
        Completely absent candidates stay absent if none are supplied.

        ``policy_responses`` and a **copy** of ``extra`` are carried over, so the
        new snapshot shares no mutable container with this one.
        """
        if candidates:
            previous = self.candidates or ((),) * self.n_steps
            new_candidates = previous + (tuple(candidates),)
        elif self.candidates and len(self.candidates) == self.n_steps:
            new_candidates = self.candidates + ((),)
        else:
            new_candidates = self.candidates
        return TrajectorySnapshot(
            trajectory_id=self.trajectory_id,
            dataset=self.dataset,
            task=self.task,
            observations=self.observations + (observation,),
            actions=self.actions + (action,),
            candidates=new_candidates,
            policy_responses=self.policy_responses,
            extra=dict(self.extra),
        )

    @classmethod
    def from_rollout(
        cls,
        *,
        trajectory_id: str,
        dataset: str,
        task: TaskSpec,
        observations: Tuple[str, ...],
        actions: Tuple[str, ...],
        candidates: Tuple[Tuple[str, ...], ...] = (),
        policy_responses: Tuple[str, ...] = (),
        extra: Optional[Mapping[str, Any]] = None,
    ) -> "TrajectorySnapshot":
        """Validating constructor; the invariants live in ``__post_init__``."""
        return cls(
            trajectory_id=trajectory_id,
            dataset=dataset,
            task=task,
            observations=tuple(observations),
            actions=tuple(actions),
            candidates=tuple(tuple(c) for c in candidates),
            policy_responses=tuple(policy_responses),
            extra=_mapping(extra),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "dataset": self.dataset,
            "task": self.task.to_dict(),
            "observations": list(self.observations),
            "actions": list(self.actions),
            "candidates": [list(c) for c in self.candidates],
            "policy_responses": list(self.policy_responses),
            "extra": dict(self.extra),
        }


def snapshot_from_dict(data: Mapping[str, Any]) -> TrajectorySnapshot:
    """Build a snapshot from the serialized trajectory record.

    ``task`` may be a nested dict or a plain task string; in the latter case a
    sibling ``task_id`` and optional ``max_steps`` describe the task.
    """
    raw_task = data.get("task")
    if isinstance(raw_task, Mapping):
        task = TaskSpec.from_dict(raw_task)
    else:
        task = TaskSpec(
            task_id=str(data.get("task_id", data.get("trajectory_id", ""))),
            task=str(raw_task) if raw_task is not None else "",
            max_steps=data.get("max_steps"),
            extra=_mapping(data.get("task_extra")),
        )
    return TrajectorySnapshot.from_rollout(
        trajectory_id=str(data["trajectory_id"]),
        dataset=str(data["dataset"]),
        task=task,
        observations=tuple(data.get("observations", ())),
        actions=tuple(data.get("actions", ())),
        candidates=tuple(tuple(c) for c in data.get("candidates", ())),
        policy_responses=tuple(data.get("policy_responses", ())),
        extra=data.get("extra"),
    )


def snapshot_to_dict(snapshot: TrajectorySnapshot) -> Dict[str, Any]:
    """JSON-compatible dict for a snapshot; inverse of :func:`snapshot_from_dict`."""
    return snapshot.to_dict()


@dataclass(frozen=True)
class Position:
    """An explicit evaluation position.

    ``kind`` is a free-form label supplied by the expansion rule; there is no
    global enum. With ``kind="trajectory"`` a position names the whole
    trajectory, with ``kind="action"`` a single action, with ``kind="boundary"``
    an observation boundary and with ``kind="decision"`` a pending decision.
    """

    kind: str
    index: Optional[int] = None
    end: Optional[int] = None

    @property
    def key(self) -> str:
        """Canonical stable string; request IDs and score queries build on it."""
        if self.index is None and self.end is None:
            return self.kind
        if self.end is None:
            return "{}[{}]".format(self.kind, self.index)
        return "{}[{}:{}]".format(self.kind, self.index, self.end)

    @classmethod
    def action(cls, index: int) -> "Position":
        return cls(kind="action", index=index)

    @classmethod
    def boundary(cls, index: int) -> "Position":
        return cls(kind="boundary", index=index)

    @classmethod
    def trajectory(cls) -> "Position":
        return cls(kind="trajectory")

    @classmethod
    def decision(cls, index: int) -> "Position":
        return cls(kind="decision", index=index)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "index": self.index, "end": self.end}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Position":
        return cls(
            kind=str(data["kind"]),
            index=data.get("index"),
            end=data.get("end"),
        )


@dataclass(frozen=True)
class ChoiceOption:
    """One selectable answer option."""

    option_id: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"option_id": self.option_id, "text": self.text}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChoiceOption":
        return cls(option_id=str(data["option_id"]), text=str(data["text"]))


@dataclass(frozen=True)
class AnswerSpec:
    """What kind of answer a question expects."""

    kind: str
    scale: Tuple[float, float] = (0.0, 1.0)
    choices: Tuple[ChoiceOption, ...] = ()
    instructions: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", tuple(self.scale))
        object.__setattr__(self, "choices", tuple(self.choices))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "scale": list(self.scale),
            "choices": [c.to_dict() for c in self.choices],
            "instructions": self.instructions,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnswerSpec":
        scale = data.get("scale", (0.0, 1.0))
        return cls(
            kind=str(data["kind"]),
            scale=tuple(scale),
            choices=tuple(ChoiceOption.from_dict(c) for c in data.get("choices", ())),
            instructions=str(data.get("instructions", "")),
        )


@dataclass(frozen=True)
class ContextPayload:
    """What the context prompt hands to the question prompt and the model."""

    text: str
    materials: Mapping[str, Any] = field(default_factory=_empty_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "materials": dict(self.materials)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContextPayload":
        return cls(
            text=str(data.get("text", "")),
            materials=_mapping(data.get("materials")),
        )


@dataclass(frozen=True)
class QuestionPayload:
    """The question text, its answer constraints and optional materials."""

    question: str
    answer: AnswerSpec
    materials: Mapping[str, Any] = field(default_factory=_empty_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer.to_dict(),
            "materials": dict(self.materials),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QuestionPayload":
        return cls(
            question=str(data.get("question", "")),
            answer=AnswerSpec.from_dict(data["answer"]),
            materials=_mapping(data.get("materials")),
        )


@dataclass(frozen=True)
class RequestSource:
    """Call-side association metadata.

    Never contains a training row id: it names the dataset, the source
    trajectory, the evaluated position, the config and the call identity.
    """

    dataset: str
    trajectory_id: str
    position: Position
    config_name: str
    batch_id: str

    @property
    def position_key(self) -> str:
        return self.position.key

    @property
    def item_key(self) -> str:
        return "{}:{}".format(self.trajectory_id, self.position.key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "trajectory_id": self.trajectory_id,
            "position": self.position.to_dict(),
            "position_key": self.position_key,
            "config_name": self.config_name,
            "batch_id": self.batch_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RequestSource":
        return cls(
            dataset=str(data["dataset"]),
            trajectory_id=str(data["trajectory_id"]),
            position=Position.from_dict(data["position"]),
            config_name=str(data["config_name"]),
            batch_id=str(data["batch_id"]),
        )


#: Separator between the four request-id components.
REQUEST_ID_SEPARATOR = ":"


def _escape_request_id_component(value: str) -> str:
    """Percent-escape one request-id component.

    ``%`` is escaped first so the escaping is unambiguous, then ``:``, which is
    the separator. A component containing neither comes out unchanged.
    """
    return str(value).replace("%", "%25").replace(REQUEST_ID_SEPARATOR, "%3A")


def _unescape_request_id_component(value: str) -> str:
    """Exact inverse of :func:`_escape_request_id_component`."""
    return value.replace("%3A", REQUEST_ID_SEPARATOR).replace("%25", "%")


def _validate_escaping(part: str, request_id: str) -> None:
    """Reject a component that is not the escaping of anything.

    Every ``%`` must introduce one of the two escape sequences this module
    emits; anything else could not have been produced by
    :func:`make_request_id`, so the id is not one of ours.
    """
    index = 0
    while index < len(part):
        if part[index] != "%":
            index += 1
            continue
        escape = part[index : index + 3]
        if escape not in ("%3A", "%25"):
            raise ValueError(
                "request id {!r} contains invalid escaping {!r}; expected '%3A' or "
                "'%25' after '%'".format(request_id, escape)
            )
        index += 3


def split_request_id(request_id: str) -> Tuple[str, str, str, str]:
    """Exact inverse of :func:`make_request_id`.

    Returns ``(batch_id, config_name, trajectory_id, position_key)``. Raises
    ``ValueError`` when the id does not have exactly four ``:``-separated
    components, or when any component is not valid escaping.
    """
    parts = str(request_id).split(REQUEST_ID_SEPARATOR)
    if len(parts) != 4:
        raise ValueError(
            "request id {!r} has {} ':'-separated component(s); expected exactly 4 "
            "(batch_id, config_name, trajectory_id, position_key)".format(
                request_id, len(parts)
            )
        )
    for part in parts:
        _validate_escaping(part, request_id)
    return (
        _unescape_request_id_component(parts[0]),
        _unescape_request_id_component(parts[1]),
        _unescape_request_id_component(parts[2]),
        _unescape_request_id_component(parts[3]),
    )


def make_request_id(
    batch_id: str,
    config_name: str,
    trajectory_id: str,
    position_key: str,
) -> str:
    """The single implementation of the request id rule.

    Each of the four components is percent-escaped (``%`` -> ``%25``, then
    ``:`` -> ``%3A``) before the ``":"`` join, which makes the mapping
    **injective**: two different component tuples can never produce the same id.
    A component containing no ``%`` or ``:`` is emitted unchanged, so plain ids
    still read verbatim as
    ``"{batch_id}:{config_name}:{trajectory_id}:{position_key}"``.
    :func:`split_request_id` is the exact inverse.
    """
    return REQUEST_ID_SEPARATOR.join(
        (
            _escape_request_id_component(batch_id),
            _escape_request_id_component(config_name),
            _escape_request_id_component(trajectory_id),
            _escape_request_id_component(position_key),
        )
    )


@dataclass(frozen=True)
class DecisionRequest:
    """One structured request handed to a decision model."""

    request_id: str
    context: ContextPayload
    question: QuestionPayload
    source: RequestSource
    meta: Mapping[str, Any] = field(default_factory=_empty_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "context": self.context.to_dict(),
            "question": self.question.to_dict(),
            "source": self.source.to_dict(),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecisionRequest":
        return cls(
            request_id=str(data["request_id"]),
            context=ContextPayload.from_dict(data["context"]),
            question=QuestionPayload.from_dict(data["question"]),
            source=RequestSource.from_dict(data["source"]),
            meta=_mapping(data.get("meta")),
        )


@dataclass(frozen=True)
class DecisionResult:
    """A structured answer, or an explicit failure.

    ``status`` is ``"ok"`` or ``"error"``. A failed result keeps ``value`` as
    ``None``; a failed model answer is never represented as a valid zero score.
    """

    request_id: str
    status: str
    value: Optional[float] = None
    choice_id: Optional[str] = None
    text: Optional[str] = None
    error: Optional[str] = None
    raw: Mapping[str, Any] = field(default_factory=_empty_dict)

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"

    @classmethod
    def ok(
        cls,
        request_id: str,
        value: Optional[float] = None,
        choice_id: Optional[str] = None,
        text: Optional[str] = None,
        raw: Optional[Mapping[str, Any]] = None,
    ) -> "DecisionResult":
        return cls(
            request_id=request_id,
            status="ok",
            value=value,
            choice_id=choice_id,
            text=text,
            error=None,
            raw=_mapping(raw),
        )

    @classmethod
    def failure(
        cls,
        request_id: str,
        error: str,
        *,
        raw: Optional[Mapping[str, Any]] = None,
    ) -> "DecisionResult":
        message = str(error) if error else "unknown decision model failure"
        return cls(
            request_id=request_id,
            status="error",
            value=None,
            choice_id=None,
            text=None,
            error=message,
            raw=_mapping(raw),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "value": self.value,
            "choice_id": self.choice_id,
            "text": self.text,
            "error": self.error,
            "raw": dict(self.raw),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecisionResult":
        return cls(
            request_id=str(data["request_id"]),
            status=str(data["status"]),
            value=data.get("value"),
            choice_id=data.get("choice_id"),
            text=data.get("text"),
            error=data.get("error"),
            raw=_mapping(data.get("raw")),
        )


def positions_to_keys(positions: Any) -> List[str]:
    """Helper used by callers that need position keys in order."""
    return [p.key for p in positions]
