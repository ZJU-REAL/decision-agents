"""Standard-library Decisions API protocol encoding, transport, and strict response parsing.

This backend never interprets a dataset, context name, or caller role. Prompt
components supply model-visible state, instructions, and the question key.
"""

from __future__ import annotations

import copy
import json
import math
import os
import socket
import time
from dataclasses import replace
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener

from decision_model.models.base import BaseDecisionModel
from decision_model.models.utils.utils import (
    archive_json,
    generation_extra_body,
    merge_extra_body,
    NoRedirect,
    strict_json_loads,
    finite_number,
    http_url as _http_url,
    json_post_request,
    positive_timeout as _positive_timeout,
    question_key as _question_key,
)
from decision_model.types import DecisionRequest, DecisionResult
from decision_model.registry import register_model


def _urllib_transport(
    *, url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float
) -> Mapping[str, Any]:
    """POST a decoded payload; return decoded JSON or raise a transport error."""
    request = json_post_request(url=url, headers=headers, payload=payload)
    proxy = os.environ.get("DECISION_MODEL_PROXY")
    opener = (build_opener(NoRedirect(), ProxyHandler({"http": proxy, "https": proxy}))
              if proxy else build_opener(NoRedirect()))
    with opener.open(request, timeout=timeout) as response:
        return strict_json_loads(response.read().decode("utf-8"))


def _probability(value: Any) -> bool:
    return finite_number(value) and 0 <= value <= 1


def _response_archive(data: Any) -> Dict[str, Any]:
    """Detach JSON data and make malformed non-finite numbers safe to archive."""
    if not isinstance(data, Mapping):
        return {}
    try:
        archived = archive_json(dict(data))
    except (ValueError, TypeError, OverflowError, AttributeError, RecursionError):
        return {}
    return archived if isinstance(archived, dict) else {}


class DecisionAPIModel(BaseDecisionModel):
    """Reusable native Decisions protocol with overridable encode/parse methods.

    endpoint selects the complete address; otherwise base_url + /decisions is
    used. api_key_env=None disables authentication. A configured key is required
    at call time, including with an injected transport. No provider is assumed.

    extra_body adds JSON request fields without overwriting existing fields.
    expected_model optionally checks the response identity. max_retries defaults
    to zero; additional attempts only retry transient transport/HTTP failures.
    complete() calls self.encode() and self.parse_response() so subclass
    overrides are used by the actual request path.
    """

    name = "decisions_api"

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        endpoint: Optional[str] = None,
        api_key_env: Optional[str] = None,
        timeout: float = 60.0,
        extra_body: Optional[Mapping[str, Any]] = None,
        transport: Optional[Callable[..., Mapping[str, Any]]] = None,
        max_retries: int = 0,
        expected_model: Optional[str] = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if api_key_env is not None and (
            not isinstance(api_key_env, str) or not api_key_env.strip()
            or "=" in api_key_env or "\x00" in api_key_env
        ):
            raise ValueError("api_key_env must be a non-empty environment variable name or None")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if expected_model is not None and (not isinstance(expected_model, str) or not expected_model.strip()):
            raise ValueError("expected_model must be a non-empty string or None")
        if transport is not None and not callable(transport):
            raise TypeError("transport must be callable")
        self.model_name = model
        if endpoint is None:
            self.base_url = _http_url(base_url, "base_url").rstrip("/")
            self.endpoint = self.base_url + "/decisions"
        else:
            self.endpoint = _http_url(endpoint)
            self.base_url = base_url.rstrip("/") if isinstance(base_url, str) else base_url
        self.api_key_env = api_key_env
        self.timeout = _positive_timeout(timeout)
        self.extra_body = generation_extra_body(extra_body)
        self.max_retries = max_retries
        self.expected_model = expected_model
        self.transport = transport if transport is not None else _urllib_transport

    def encode(self, request: DecisionRequest) -> Dict[str, Any]:
        """Return the native API fields and JSON extras, detached from input.

        ``materials.state`` is structured model evidence. With no state material,
        a plain context's text is the state. ``target_instructions`` has already
        been composed into the question by the prompt layer and is not encoded
        independently. Source, metadata, and request IDs are never serialized.
        """
        question = request.question
        if question.answer.kind == "score":
            if (tuple(question.answer.scale) != (0.0, 1.0)
                    or not all(finite_number(value) for value in question.answer.scale)
                    or question.answer.choices):
                raise ValueError("Native score answers require the [0, 1] scale and no choices")
            # "noul" is the native API wire type for a scalar score, not a
            # separate paper question: both Value and Progress use this type.
            encoded_question = {"type": "noul"}
        elif question.answer.kind == "choice":
            options = question.answer.choices
            if not 1 <= len(options) <= 255:
                raise ValueError("unsupported_candidate_count")
            if any(not isinstance(option.option_id, str) or not option.option_id
                   or not isinstance(option.text, str) or not option.text.strip() for option in options):
                raise ValueError("choice options require non-empty IDs and text")
            criteria = {option.option_id: option.text for option in options}
            if len(criteria) != len(options):
                raise ValueError("choice option IDs must be unique")
            encoded_question = {"type": "choice", "criteria": criteria}
        else:
            raise ValueError("Native Decisions supports score and choice answers only")
        if not isinstance(question.question, str) or not question.question:
            raise ValueError("question must be a non-empty string")
        if not isinstance(question.answer.instructions, str):
            raise ValueError("answer instructions must be a string")
        instructions = question.question
        if question.answer.instructions:
            instructions += "\n\n" + question.answer.instructions
        if "state" in request.context.materials:
            if request.context.text:
                raise ValueError("supply structured state or context text, not both")
            state = copy.deepcopy(request.context.materials["state"])
        else:
            state = request.context.text
        encoded_question["instructions"] = instructions
        payload = {
            "model": self.model_name,
            "questions": {_question_key(request): encoded_question},
            "state": state,
        }
        payload = merge_extra_body(payload, self.extra_body)
        # Catch invalid evidence before transport; never use default=str because
        # that could silently replace a non-JSON prompt or expose object reprs.
        json.dumps(payload, allow_nan=False)
        return payload

    def parse_response(self, request: DecisionRequest, data: Mapping[str, Any]) -> DecisionResult:
        """Validate typed answers; Choice uses the server ID without an argmax."""
        archived = _response_archive(data)
        raw = {"response": archived, "model": archived.get("model")}
        if not isinstance(data, Mapping):
            return DecisionResult.failure(request.request_id, "invalid_response", raw=raw)
        if self.expected_model is not None and data.get("model") != self.expected_model:
            return DecisionResult.failure(request.request_id, "unexpected_model_version", raw=raw)
        try:
            answers = data.get("answers")
            if not isinstance(answers, Mapping):
                raise ValueError("missing answers object")
            answer = answers.get(_question_key(request))
            if not isinstance(answer, Mapping):
                raise ValueError("missing typed answer")
            if request.question.answer.kind == "score":
                value = answer.get("noul")
                if answer.get("type") != "noul" or not _probability(value):
                    raise ValueError("invalid Noul probability")
                return DecisionResult.ok(request.request_id, value=float(value), raw=raw)
            if request.question.answer.kind == "choice":
                options = request.question.answer.choices
                identifiers = {option.option_id for option in options}
                choice = answer.get("choice")
                probabilities = answer.get("probabilities")
                if (not 1 <= len(options) <= 255 or len(identifiers) != len(options)
                        or answer.get("type") != "choice"
                        or not isinstance(choice, str) or choice not in identifiers):
                    raise ValueError("invalid choice ID")
                if not isinstance(probabilities, Mapping) or set(probabilities) != identifiers:
                    raise ValueError("invalid choice probability coverage")
                if (not all(_probability(value) for value in probabilities.values())
                        or math.fsum(probabilities.values()) <= 0
                        or not _probability(answer.get("confidence"))):
                    raise ValueError("invalid choice probability or confidence")
                return DecisionResult.ok(request.request_id, choice_id=choice, raw=raw)
            raise ValueError("unsupported answer kind")
        except (ValueError, TypeError, OverflowError, AttributeError, RecursionError):
            return DecisionResult.failure(request.request_id, "invalid_response", raw=raw)

    def complete(self, request: DecisionRequest, *, timeout: Optional[float] = None) -> DecisionResult:
        started = time.monotonic()
        attempts = 0

        def finish(result: DecisionResult) -> DecisionResult:
            raw = dict(result.raw)
            raw.update(attempts=attempts, elapsed_s=time.monotonic() - started)
            return replace(result, raw=raw)

        try:
            payload = self.encode(request)
            call_timeout = self.timeout if timeout is None else _positive_timeout(timeout)
        except (ValueError, TypeError, OverflowError, AttributeError, RecursionError):
            return finish(DecisionResult.failure(request.request_id, "invalid_request"))

        api_key = os.environ.get(self.api_key_env, "") if self.api_key_env is not None else ""
        if self.api_key_env is not None and not api_key.strip():
            return finish(DecisionResult.failure(request.request_id, "missing_api_key"))
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        error = "transport_error"
        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            retryable = False
            try:
                data = self.transport(
                    url=self.endpoint,
                    headers=dict(headers),
                    payload=copy.deepcopy(payload),
                    timeout=call_timeout,
                )
                return finish(self.parse_response(request, data))
            except HTTPError as exc:
                code = exc.code
                error = "http_{}".format(code) if type(code) is int else "http_error"
                retryable = code == 429 or (type(code) is int and 500 <= code <= 599)
                try:
                    exc.close()
                except Exception:
                    pass
            except (TimeoutError, socket.timeout):
                error = "timeout"
                retryable = True
            except URLError as exc:
                error = "timeout" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "transport_error"
                retryable = True
            except (ConnectionError, OSError):
                error = "transport_error"
                retryable = True
            except (ValueError, TypeError, OverflowError, AttributeError, RecursionError):
                error = "invalid_response"
            except Exception:
                # A custom transport may raise arbitrary exceptions. Reporting
                # their text could disclose its headers or credential-bearing URL.
                error = "transport_error"
            if not retryable or attempt == self.max_retries:
                break
            time.sleep(min(0.25 * (2 ** attempt), 2.0))
        return finish(DecisionResult.failure(request.request_id, error))


register_model("decisions_api", DecisionAPIModel)
