"""Reusable Constrained client with overridable request/response hooks.

Default templates turn a general LLM into a score/choice decision model. This
client owns its HTTP transport, authentication and failure handling. complete()
calls self.encode() and self.parse_response(), including subclass overrides.
"""
from __future__ import annotations

import copy
import os
import socket
import time
from collections.abc import Mapping
from dataclasses import replace
from string import ascii_uppercase
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener

from decision_model.models.base import BaseDecisionModel
from decision_model.models.utils.prompted_format import validate_score_spec
from decision_model.models.utils.utils import (
    archive_json, merge_extra_body, generation_extra_body, http_url,
    json_post_request, NoRedirect, strict_json_loads,
    positive_timeout as _positive_timeout,
)
from decision_model.models.utils.constrained_format import (
    SCORE_SYSTEM_PROMPT, CHOICE_SYSTEM_PROMPT, ReadoutError,
    encode_score, decode_score, encode_choice, decode_choice,
)
from decision_model.registry import register_model
from decision_model.types import DecisionRequest, DecisionResult


def _urllib_transport(
    *, url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float
) -> Mapping[str, Any]:
    """Send one JSON POST without redirecting or changing global HTTP state."""
    request = json_post_request(url=url, headers=headers, payload=payload)
    proxy = os.environ.get("DECISION_MODEL_PROXY")
    opener = (build_opener(NoRedirect(), ProxyHandler({"http": proxy, "https": proxy}))
              if proxy else build_opener(NoRedirect()))
    with opener.open(request, timeout=timeout) as response:
        return strict_json_loads(response.read().decode("utf-8"))


class ConstrainedLLM(BaseDecisionModel):
    """One configurable HTTP client for both score and choice requests.

    The request's answer.kind selects the default readout. Score normalizes the
    Yes/No token probabilities. Choice maps candidate IDs to labels (A-Z by
    default), selects the most likely legal label, normalizes over returned
    candidate labels, and bounds unreturned mass.
    Custom labels must each be a single token under the served model tokenizer.

    Override encode() to customize request fields/prompts, parse_response() to
    adapt responses, or the system-prompt class attributes to change instructions.
    transport can replace HTTP I/O. No model/provider identity is hard-coded.
    """

    name = "constrained"
    score_system_prompt = SCORE_SYSTEM_PROMPT
    choice_system_prompt = CHOICE_SYSTEM_PROMPT

    def __init__(self, *, model, base_url="http://localhost:8000/v1",
                 endpoint=None, api_key_env=None, timeout=60.0,
                 labels=None, top_logprobs=20, extra_body=None, transport=None):
        if type(top_logprobs) is not int or top_logprobs < 2:
            raise ValueError("top_logprobs must be an integer >= 2")
        if labels is None:
            labels = tuple(ascii_uppercase)
        if isinstance(labels, (str, bytes)):
            raise ValueError("labels must be a sequence of distinct token strings")
        # Accept either token strings or explicit label mappings.
        normalized = tuple(label.get("label") if isinstance(label, Mapping) else label
                           for label in labels)
        if (not normalized or any(not isinstance(label, str) or not label.strip()
                                  for label in normalized)
                or len(set(normalized)) != len(normalized)):
            raise ValueError("labels must be nonempty, distinct token strings")
        self.labels = tuple({"label": label} for label in normalized)
        self.top_logprobs = top_logprobs
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if api_key_env is not None and (
            not isinstance(api_key_env, str) or not api_key_env.strip()
            or "=" in api_key_env or "\x00" in api_key_env
        ):
            raise ValueError("api_key_env must be a non-empty environment variable name or None")
        if transport is not None and not callable(transport):
            raise TypeError("transport must be callable")
        self.model_name = model
        if endpoint is None:
            self.base_url = http_url(base_url, "base_url").rstrip("/")
            self.endpoint = self.base_url + "/chat/completions"
        else:
            self.endpoint = http_url(endpoint)
            self.base_url = base_url.rstrip("/") if isinstance(base_url, str) else base_url
        self.api_key_env = api_key_env
        self.timeout = _positive_timeout(timeout)
        self.temperature = 1.0
        self.max_tokens = 1
        self.extra_body = generation_extra_body(extra_body)
        self.transport = transport if transport is not None else _urllib_transport

    def encode(self, request):
        """Build a one-token readout request from the supplied question."""
        options = {"model": self.model_name, "top_logprobs": self.top_logprobs}
        kind = request.question.answer.kind
        if kind == "score":
            validate_score_spec(request)
            options["system_prompt"] = self.score_system_prompt
            payload = encode_score(request, options)
        elif kind == "choice":
            options["system_prompt"] = self.choice_system_prompt
            payload = encode_choice(request, self.labels, options)
        else:
            raise ValueError("constrained readout supports score and choice")
        return merge_extra_body(payload, self.extra_body)

    def parse_response(self, request, data):
        """Decode probabilities without requiring provider-specific metadata."""
        raw = {"probability_source": "constrained"}
        try:
            if not isinstance(data, Mapping):
                raise ReadoutError("invalid_response_envelope")
            for key in ("model", "provider", "id", "usage"):
                if key in data:
                    raw[key] = archive_json(data[key])
            if data.get("error"):
                raise ReadoutError("api_error_envelope")
            kind = request.question.answer.kind
            if kind == "score":
                validate_score_spec(request)
                result = decode_score(request, data)
            elif kind == "choice":
                result = decode_choice(request, data, self.labels,
                                       top_logprobs=self.top_logprobs)
            else:
                raise ReadoutError("unsupported_answer_kind")
            return replace(result, raw={**raw, **result.raw})
        except (ValueError, TypeError, KeyError, AttributeError, IndexError,
                OverflowError, RecursionError) as exc:
            error = "readout:" + str(exc) if isinstance(exc, ReadoutError) else "invalid_response"
            return DecisionResult.failure(request.request_id, error, raw=raw)


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

        headers = {"Content-Type": "application/json"}
        if self.api_key_env is not None:
            api_key = os.environ.get(self.api_key_env, "")
            if not api_key.strip():
                return finish(DecisionResult.failure(request.request_id, "missing_api_key"))
            headers["Authorization"] = "Bearer " + api_key
        attempts = 1
        try:
            data = self.transport(url=self.endpoint, headers=headers,
                                  payload=copy.deepcopy(payload), timeout=call_timeout)
            return finish(self.parse_response(request, data))
        except HTTPError as exc:
            error = "http_{}".format(exc.code) if type(exc.code) is int else "http_error"
            try:
                exc.close()
            except Exception:
                pass
        except (TimeoutError, socket.timeout):
            error = "timeout"
        except URLError as exc:
            error = "timeout" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "transport_error"
        except (ValueError, TypeError, OverflowError, UnicodeError, RecursionError):
            error = "invalid_response"
        except Exception:
            # Custom transports may include headers or URLs in their exceptions.
            error = "transport_error"
        return finish(DecisionResult.failure(request.request_id, error))


register_model("constrained", ConstrainedLLM)
