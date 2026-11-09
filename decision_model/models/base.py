"""The decision model interface.

The model layer sees only structured requests and results. It must not branch on
dataset, context view or calling role: whether the request came from an actor, an
offline evaluation or training is the caller's business.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from decision_model.types import DecisionRequest, DecisionResult


class BaseDecisionModel(ABC):
    """A backend that answers :class:`DecisionRequest` objects."""

    #: Registered component name.
    name: str = "base"

    #: Served model identity, used for version checks by concrete backends.
    model_name: str = "base/decision"

    @abstractmethod
    def complete(
        self,
        request: DecisionRequest,
        *,
        timeout: Optional[float] = None,
    ) -> DecisionResult:
        """Answer one request, or return an explicit failure."""
        raise NotImplementedError

    def complete_many(
        self,
        requests: Sequence[DecisionRequest],
        *,
        timeout: Optional[float] = None,
    ) -> List[DecisionResult]:
        """Answer several requests, one result per request, in order.

        The default is sequential. Concrete backends and the evaluation runtime
        can provide concurrency, retries, and caching.
        """
        return [self.complete(request, timeout=timeout) for request in requests]

    def close(self) -> None:
        """Release backend resources. No-op by default."""
        return None
