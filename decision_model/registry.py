"""Name -> component class registration and construction.

The registry is the only place a component name is turned into a class, so that
model, dataset, context, question and advantage stay independently replaceable.
Package import does not load model implementations. Use models.utils.loader's
get_model(), import a component's module, or register an application class before
building it; component names do not imply a provider, model identity or address.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Type

from decision_model.advantage.base import BaseStepAdvantage
from decision_model.models.base import BaseDecisionModel
from decision_model.prompts.base import (
    BaseContextPrompt,
    BaseDatasetPrompt,
    BaseQuestionPrompt,
)

_MODELS: Dict[str, Type[Any]] = {}
_DATASET_PROMPTS: Dict[str, Type[Any]] = {}
_CONTEXT_PROMPTS: Dict[str, Type[Any]] = {}
_QUESTION_PROMPTS: Dict[str, Type[Any]] = {}
_ADVANTAGES: Dict[str, Type[Any]] = {}

_KIND_TO_REGISTRY: Dict[str, Dict[str, Type[Any]]] = {
    "model": _MODELS,
    "dataset": _DATASET_PROMPTS,
    "context": _CONTEXT_PROMPTS,
    "question": _QUESTION_PROMPTS,
    "advantage": _ADVANTAGES,
}


def _register(
    registry: Dict[str, Type[Any]],
    kind: str,
    name: str,
    cls: Type[Any],
    override: bool = False,
) -> Type[Any]:
    if not isinstance(name, str) or not name:
        raise ValueError("{} name must be a non-empty string".format(kind))
    if not isinstance(cls, type):
        raise TypeError("{} must be registered with a class, got {!r}".format(kind, cls))
    if name in registry and not override:
        raise ValueError(
            "{} name {!r} is already registered as {!r}; pass override=True to replace it".format(
                kind, name, registry[name].__name__
            )
        )
    registry[name] = cls
    return cls


def _names(registry: Dict[str, Type[Any]]) -> List[str]:
    return sorted(registry)


def _lookup(registry: Dict[str, Type[Any]], kind: str, name: str) -> Type[Any]:
    try:
        return registry[name]
    except KeyError:
        available = _names(registry)
        raise KeyError(
            "unknown {} name {!r}; available {}: {}".format(
                kind,
                name,
                kind,
                ", ".join(available) if available else "<none registered>",
            )
        ) from None


def _build(registry: Dict[str, Type[Any]], kind: str, name: str, options: Mapping[str, Any]) -> Any:
    return _lookup(registry, kind, name)(**dict(options))


# ---- registration --------------------------------------------------------


def register_model(name: str, cls: Type[BaseDecisionModel], override: bool = False) -> None:
    """Register a decision model class under ``name``."""
    _register(_MODELS, "model", name, cls, override)


def register_dataset_prompt(
    name: str, cls: Type[BaseDatasetPrompt], override: bool = False
) -> None:
    """Register a dataset prompt class under ``name``."""
    _register(_DATASET_PROMPTS, "dataset prompt", name, cls, override)


def register_context_prompt(
    name: str, cls: Type[BaseContextPrompt], override: bool = False
) -> None:
    """Register a context prompt class under ``name``."""
    _register(_CONTEXT_PROMPTS, "context prompt", name, cls, override)


def register_question_prompt(
    name: str, cls: Type[BaseQuestionPrompt], override: bool = False
) -> None:
    """Register a question prompt class under ``name``."""
    _register(_QUESTION_PROMPTS, "question prompt", name, cls, override)


def register_advantage(
    name: str, cls: Type[BaseStepAdvantage], override: bool = False
) -> None:
    """Register a step-advantage implementation under ``name``."""
    _register(_ADVANTAGES, "advantage", name, cls, override)


# ---- construction --------------------------------------------------------


def build_model(name: str, **options: Any) -> BaseDecisionModel:
    """Construct the registered ``name`` model with ``options`` as keywords."""
    return _build(_MODELS, "model", name, options)


def build_dataset_prompt(name: str, **options: Any) -> BaseDatasetPrompt:
    """Construct the registered ``name`` dataset prompt."""
    return _build(_DATASET_PROMPTS, "dataset prompt", name, options)


def build_context_prompt(name: str, **options: Any) -> BaseContextPrompt:
    """Construct the registered ``name`` context prompt."""
    return _build(_CONTEXT_PROMPTS, "context prompt", name, options)


def build_question_prompt(name: str, **options: Any) -> BaseQuestionPrompt:
    """Construct the registered ``name`` question prompt."""
    return _build(_QUESTION_PROMPTS, "question prompt", name, options)


def build_advantage(name: str, **options: Any) -> BaseStepAdvantage:
    """Construct the registered ``name`` advantage implementation."""
    return _build(_ADVANTAGES, "advantage", name, options)


# ---- introspection -------------------------------------------------------


def available_models() -> List[str]:
    return _names(_MODELS)


def available_dataset_prompts() -> List[str]:
    return _names(_DATASET_PROMPTS)


def available_context_prompts() -> List[str]:
    return _names(_CONTEXT_PROMPTS)


def available_question_prompts() -> List[str]:
    return _names(_QUESTION_PROMPTS)


def available_advantages() -> List[str]:
    return _names(_ADVANTAGES)


def advantage_class(name: str) -> Type[BaseStepAdvantage]:
    """The registered class for ``name``; ``KeyError`` lists the available names."""
    return _lookup(_ADVANTAGES, "advantage", name)


def model_class(name: str) -> Type[BaseDecisionModel]:
    """The registered class for ``name``; ``KeyError`` lists the available names."""
    return _lookup(_MODELS, "model", name)


def dataset_prompt_class(name: str) -> Type[BaseDatasetPrompt]:
    """The registered class for ``name``; ``KeyError`` lists the available names."""
    return _lookup(_DATASET_PROMPTS, "dataset prompt", name)


def context_prompt_class(name: str) -> Type[BaseContextPrompt]:
    """The registered class for ``name``; ``KeyError`` lists the available names."""
    return _lookup(_CONTEXT_PROMPTS, "context prompt", name)


def question_prompt_class(name: str) -> Type[BaseQuestionPrompt]:
    """The registered class for ``name``; ``KeyError`` lists the available names."""
    return _lookup(_QUESTION_PROMPTS, "question prompt", name)


def available_names(kind: str) -> List[str]:
    """Available names for one component kind."""
    try:
        registry = _KIND_TO_REGISTRY[kind]
    except KeyError:
        raise KeyError(
            "unknown component kind {!r}; available kinds: {}".format(
                kind, ", ".join(sorted(_KIND_TO_REGISTRY))
            )
        ) from None
    return _names(registry)
