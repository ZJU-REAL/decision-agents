"""Decision model backends."""

from __future__ import annotations

from decision_model.models.base import BaseDecisionModel

__all__ = ["BaseDecisionModel", "FakeDecisionModel", "DecisionAPIModel",
           "PromptedLLM", "ConstrainedLLM"]


def __getattr__(name):
    if name == "FakeDecisionModel":
        from decision_model.models.fake import FakeDecisionModel
        return FakeDecisionModel
    if name == "ConstrainedLLM":
        from decision_model.models.constrained import ConstrainedLLM
        return ConstrainedLLM
    # Import implementations only when requested, avoiding registry import cycles.
    if name == "PromptedLLM":
        from decision_model.models.prompted import PromptedLLM
        return PromptedLLM
    if name == "DecisionAPIModel":
        from decision_model.models.decisions_api import DecisionAPIModel
        return DecisionAPIModel
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
