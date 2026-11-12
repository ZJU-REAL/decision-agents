"""Configuration objects for an evaluation run.

Serialization is plain JSON-compatible dicts (``to_dict`` / ``from_dict``);
there is deliberately no OmegaConf dependency here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

COMPONENT_KINDS: Tuple[str, ...] = (
    "model",
    "dataset",
    "context",
    "question",
    "advantage",
)


def _empty_dict() -> Dict[str, Any]:
    return {}


@dataclass(frozen=True)
class ComponentConfig:
    """One component selection: what kind it is, its name and its options."""

    kind: str
    name: str
    options: Mapping[str, Any] = field(default_factory=_empty_dict)

    def __post_init__(self) -> None:
        if self.kind not in COMPONENT_KINDS:
            raise ValueError(
                "ComponentConfig kind must be one of {}; got {!r}".format(
                    ", ".join(repr(k) for k in COMPONENT_KINDS), self.kind
                )
            )

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "options": dict(self.options)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ComponentConfig":
        return cls(
            kind=str(data["kind"]),
            name=str(data["name"]),
            options=dict(data.get("options") or {}),
        )


@dataclass(frozen=True)
class EvaluationConfig:
    """One evaluation: which components to use and how to expand positions."""

    name: str
    model: str
    dataset: str
    context: str
    question: str
    expansion: str
    metric: str
    options: Mapping[str, Any] = field(default_factory=_empty_dict)

    def expansion_options(self) -> Mapping[str, Any]:
        """Options handed to :func:`decision_model.judge.expand_positions`."""
        return self.options.get("expansion_options") or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "dataset": self.dataset,
            "context": self.context,
            "question": self.question,
            "expansion": self.expansion,
            "metric": self.metric,
            "options": dict(self.options),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationConfig":
        return cls(
            name=str(data["name"]),
            model=str(data["model"]),
            dataset=str(data["dataset"]),
            context=str(data["context"]),
            question=str(data["question"]),
            expansion=str(data["expansion"]),
            metric=str(data["metric"]),
            options=dict(data.get("options") or {}),
        )


def evaluation_to_dict(config: EvaluationConfig) -> Dict[str, Any]:
    """Module-level convenience mirroring ``EvaluationConfig.to_dict``."""
    return config.to_dict()


def evaluation_from_dict(data: Mapping[str, Any]) -> EvaluationConfig:
    """Module-level convenience mirroring ``EvaluationConfig.from_dict``."""
    return EvaluationConfig.from_dict(data)


def make_component_configs(
    selections: Mapping[str, Tuple[str, Optional[Mapping[str, Any]]]],
) -> Dict[str, ComponentConfig]:
    """Build one :class:`ComponentConfig` per kind from ``{kind: (name, options)}``."""
    result: Dict[str, ComponentConfig] = {}
    for kind, selection in selections.items():
        name, options = selection
        result[kind] = ComponentConfig(kind=kind, name=name, options=options or {})
    return result
