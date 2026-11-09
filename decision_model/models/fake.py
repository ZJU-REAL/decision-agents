"""Deterministic in-process backends and offline protocol subclass examples.

``FakeDecisionModel`` exists so that callers can be checked without a network,
an API key or the training framework: it records what it was asked, returns
configurable answers, can fail on demand, and can deliberately return its
results out of order to prove that callers associate by ``request_id``.

The three protocol subclasses below simulate service responses while retaining
inherited complete(), encode() and parse_response(). They return a fixed score
(default .75) or choose the first encoded option; they never call a service.
Use them as examples for a new backend, register the class at file end, and add
its module to models.utils.loader.MODEL_MODULES.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

from decision_model.models.base import BaseDecisionModel
from decision_model.models.constrained import ConstrainedLLM
from decision_model.models.decisions_api import DecisionAPIModel
from decision_model.models.prompted import PromptedLLM
from decision_model.models.utils.utils import finite_number
from decision_model.registry import register_model
from decision_model.types import DecisionRequest, DecisionResult

#: ``fail_for`` entries: a literal request id or a predicate over the request.
FailPredicate = Union[str, Callable[[DecisionRequest, int], bool]]


class FakeDecisionModel(BaseDecisionModel):
    """A fake backend with deterministic, configurable behaviour.

    ``scorer`` is called as ``scorer(request, index) -> float | DecisionResult``
    and defaults to ``1.0`` for every request. ``chooser`` is called as
    ``chooser(request, index) -> str`` and defaults to the first option's
    ``option_id``, or ``None`` when the request offers no choices. ``index`` is
    the position of the request within the current call. ``fail_for`` is an
    iterable of request ids or predicates; a match produces an explicit
    :meth:`DecisionResult.failure`.
    """

    def __init__(
        self,
        *,
        name: str = "fake",
        model_name: str = "fake/decision-1",
        scorer: Optional[Callable[[DecisionRequest, int], Any]] = None,
        chooser: Optional[Callable[[DecisionRequest, int], Optional[str]]] = None,
        fail_for: Iterable[FailPredicate] = (),
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.model_name = model_name
        self.scorer = scorer
        self.chooser = chooser
        self.fail_for = list(fail_for)
        self.shuffle = shuffle
        self.seed = seed

        #: Every request received, in call order.
        self.requests: List[DecisionRequest] = []
        #: The encoded payload for every request, in the same order.
        self.raw_payloads: List[Dict[str, Any]] = []

    # ---- recording -------------------------------------------------------

    def reset(self) -> None:
        """Forget every recorded request and payload."""
        self.requests = []
        self.raw_payloads = []

    def encode(self, request: DecisionRequest) -> Dict[str, Any]:
        """The single place a request becomes a dictionary.

        This is what an HTTP backend would serialize, and what
        :attr:`raw_payloads` records.
        """
        return {
            "request_id": request.request_id,
            "dataset": request.source.dataset,
            "position_key": request.source.position_key,
            "context": request.context.to_dict(),
            "question": request.question.to_dict(),
            "answer": request.question.answer.to_dict(),
            "source": request.source.to_dict(),
        }

    # ---- answering -------------------------------------------------------

    def _should_fail(self, request: DecisionRequest, index: int) -> bool:
        for entry in self.fail_for:
            if callable(entry):
                if entry(request, index):
                    return True
            elif entry == request.request_id:
                return True
        return False

    def _answer(self, request: DecisionRequest, index: int) -> DecisionResult:
        payload = self.encode(request)
        self.requests.append(request)
        self.raw_payloads.append(payload)

        if self._should_fail(request, index):
            return DecisionResult.failure(
                request.request_id,
                "fake model configured to fail for this request",
                raw=payload,
            )

        if self.scorer is None:
            value: Optional[float] = 1.0
        else:
            scored = self.scorer(request, index)
            if isinstance(scored, DecisionResult):
                return scored
            value = None if scored is None else float(scored)

        choice_id: Optional[str] = None
        if self.chooser is not None:
            chosen = self.chooser(request, index)
            choice_id = None if chosen is None else str(chosen)
        elif request.question.answer.choices:
            choice_id = request.question.answer.choices[0].option_id

        return DecisionResult.ok(
            request.request_id,
            value=value,
            choice_id=choice_id,
            raw=payload,
        )

    def complete(
        self,
        request: DecisionRequest,
        *,
        timeout: Optional[float] = None,
    ) -> DecisionResult:
        return self._answer(request, 0)

    def complete_many(
        self,
        requests: Sequence[DecisionRequest],
        *,
        timeout: Optional[float] = None,
    ) -> List[DecisionResult]:
        """Answer every request, optionally returning them out of order.

        With ``shuffle=True`` the returned list is deterministically permuted
        (seeded by ``seed``) while each result still carries its own
        ``request_id``.
        """
        results = [self._answer(request, index) for index, request in enumerate(requests)]
        if self.shuffle and len(results) > 1:
            random.Random(self.seed).shuffle(results)
        return results

    def close(self) -> None:
        return None


def _register_fake() -> None:
    """Register the fake backend under ``"fake"`` at import time.

    Idempotent: re-importing the module must not raise and must not displace a
    ``"fake"`` registration that already exists.
    """
    from decision_model.registry import available_models, register_model

    if "fake" not in available_models():
        register_model("fake", FakeDecisionModel)


_register_fake()


def _score(value):
    # Keep both Yes and No log probabilities finite in the Constrained example.
    if not finite_number(value) or not 0 < value < 1:
        raise ValueError("fake score must be a finite number strictly between 0 and 1")
    return float(value)


def _chat_question(payload):
    content = payload["messages"][-1]["content"]
    key = content.split("\n", 1)[0][len("Question key: "):]
    marker = "\n\nAvailable options:\n"
    if marker not in content:
        return key, None
    # The constrained parent appends an Answer: suffix after the options JSON.
    options, _ = json.JSONDecoder().raw_decode(content.rsplit(marker, 1)[1])
    return key, [option["id"] for option in options]


def _choice_answer(identifiers):
    return {
        "type": "choice", "choice": identifiers[0], "confidence": 1.0,
        "probabilities": {identifier: float(index == 0)
                          for index, identifier in enumerate(identifiers)},
    }


class FakeDecisionAPIModel(DecisionAPIModel):
    """Native Decisions subclass with an in-process simulated service."""

    name = "fake_decisions_api"

    def __init__(self, *, score=0.75):
        self.score = _score(score)
        super().__init__(model=self.name, endpoint="https://offline.invalid/decisions",
                         transport=self._respond)

    def _respond(self, *, url, headers, payload, timeout):
        answers = {}
        for key, question in payload["questions"].items():
            answers[key] = (_choice_answer(list(question["criteria"]))
                            if question["type"] == "choice"
                            else {"type": "noul", "noul": self.score})
        return {"model": self.model_name, "answers": answers}


class FakePromptedLLM(PromptedLLM):
    """Prompted subclass returning synthetic JSON inside a chat response."""

    name = "fake_prompted"

    def __init__(self, *, score=0.75):
        self.score = _score(score)
        super().__init__(model=self.name, endpoint="https://offline.invalid/chat/completions",
                         transport=self._respond)

    def _respond(self, *, url, headers, payload, timeout):
        key, identifiers = _chat_question(payload)
        answer = (_choice_answer(identifiers) if identifiers is not None
                  else {"type": "score", "value": self.score})
        return {"model": self.model_name, "choices": [{
            "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps({"answers": {key: answer}}),
            },
        }]}


class FakeConstrainedLLM(ConstrainedLLM):
    """Constrained subclass returning synthetic single-token log probabilities."""

    name = "fake_constrained"

    def __init__(self, *, score=0.75):
        self.score = _score(score)
        super().__init__(model=self.name, endpoint="https://offline.invalid/chat/completions",
                         transport=self._respond)

    def _respond(self, *, url, headers, payload, timeout):
        _, identifiers = _chat_question(payload)
        if identifiers is None:
            probabilities = [("Yes", self.score), ("No", 1 - self.score)]
        else:
            count = payload["top_logprobs"]
            tokens = list(identifiers[:count])
            # Include a complete global top-k prefix, including non-option tokens.
            while len(tokens) < count:
                tokens.append("<fake-other-{}>".format(len(tokens)))
            probabilities = [(token, .75 if index == 0 else .25 / (count - 1))
                             for index, token in enumerate(tokens)]
        probabilities.sort(key=lambda item: item[1], reverse=True)
        token = probabilities[0][0]
        return {"model": self.model_name, "choices": [{
            "finish_reason": "length", "message": {"role": "assistant", "content": token},
            "logprobs": {"content": [{
                "token": token,
                "top_logprobs": [{"token": label, "logprob": math.log(probability)}
                                 for label, probability in probabilities],
            }]},
        }]}


register_model("fake_decisions_api", FakeDecisionAPIModel)
register_model("fake_prompted", FakePromptedLLM)
register_model("fake_constrained", FakeConstrainedLLM)
