"""Plain prompted choice and score decisions through Chat Completions.

Only supplied state, question, instructions and any options enter the prompt.
The model's probabilities are self-reported; Constrained uses token probabilities.
"""

from __future__ import annotations

import copy
import os
import re
import socket
import time
from dataclasses import replace
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener

from decision_model.models.base import BaseDecisionModel
from decision_model.models.utils.utils import (
    NoRedirect as _NoRedirect,
    archive_json as _archive,
    finite_number as _finite_number,
    generation_extra_body as _extra_body,
    http_url as _http_url,
    json_post_request,
    merge_extra_body,
    positive_timeout as _positive_timeout,
    question_key as _question_key,
    strict_json_loads as _strict_json,
    choice_options as _options,
    render_choice_input,
    render_score_input,
    validate_score_spec,
)
from decision_model.models.utils.prompted_format import (
    GENERATED_CHOICE_SYSTEM_PROMPT,
    GENERATED_SCORE_SYSTEM_PROMPT,
    validate_generated_choice,
    validate_generated_score,
)
from decision_model.registry import register_model
from decision_model.types import DecisionRequest, DecisionResult


_JSON_CODE_FENCE = re.compile(
    r"```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```", re.IGNORECASE
)

def _base_url(value: Any) -> str:
    return _http_url(value, "base_url").rstrip("/")


def _urllib_transport(
    *, url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float
) -> Mapping[str, Any]:
    request = json_post_request(url=url, headers=headers, payload=payload)
    # A local opener prevents both credential forwarding and POST-to-GET
    # redirects without altering urllib's process-wide opener or other backends.
    proxy = os.environ.get("DECISION_MODEL_PROXY")
    opener = (build_opener(_NoRedirect(), ProxyHandler({"http": proxy, "https": proxy}))
              if proxy else build_opener(_NoRedirect()))
    with opener.open(request, timeout=timeout) as response:
        return _strict_json(response.read().decode("utf-8"))


class PromptedLLM(BaseDecisionModel):
    """One prompt, one HTTP call, strict validation and no recovery attempts.

    ``transport(*, url, headers, payload, timeout) -> Mapping`` accepts a decoded
    JSON body and returns the decoded Chat Completions envelope. The optional
    ``api_key_env`` is read at call time, including with an injected transport;
    when configured it must contain a non-empty key. Credentials, request
    headers, endpoint URLs and exception messages are never archived. The built-
    in transport rejects redirects as HTTP failures without following them.

    ``endpoint`` is a complete HTTP(S) address and takes precedence over
    ``base_url``. Without it, requests use ``{base_url}/chat/completions``.

    ``extra_body`` passes through finite JSON values for service-specific
    settings such as ``chat_template_kwargs={"enable_thinking": False}``.
    Extra fields cannot overwrite fields already set by encode(). Other JSON
    fields are passed through; subclasses may adapt encoding and parsing. Probability sums are not validated or normalized.
    Confidence equality uses absolute tolerance 1e-6, relative tolerance zero.
    """

    name = "prompted"
    score_system_prompt = GENERATED_SCORE_SYSTEM_PROMPT
    choice_system_prompt = GENERATED_CHOICE_SYSTEM_PROMPT

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        endpoint: Optional[str] = None,
        api_key_env: Optional[str] = None,
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        extra_body: Optional[Mapping[str, Any]] = None,
        transport: Optional[Callable[..., Mapping[str, Any]]] = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if api_key_env is not None and (
            not isinstance(api_key_env, str) or not api_key_env.strip()
            or "=" in api_key_env or "\x00" in api_key_env
        ):
            raise ValueError("api_key_env must be a non-empty environment variable name or None")
        if not _finite_number(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be a finite number between 0 and 2")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if transport is not None and not callable(transport):
            raise TypeError("transport must be callable")
        self.model_name = model
        if endpoint is None:
            self.base_url = _base_url(base_url)
            self.endpoint = self.base_url + "/chat/completions"
        else:
            self.endpoint = _http_url(endpoint)
            self.base_url = base_url.rstrip("/") if isinstance(base_url, str) else base_url
        self.api_key_env = api_key_env
        self.timeout = _positive_timeout(timeout)
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self.extra_body = _extra_body(extra_body)
        self.transport = transport if transport is not None else _urllib_transport

    def encode(self, request: DecisionRequest) -> Dict[str, Any]:
        """Encode the visible question input for its declared answer kind."""
        if request.question.answer.kind == "score":
            content = render_score_input(request)
            system_prompt = self.score_system_prompt
        else:
            content = render_choice_input(request)
            system_prompt = self.choice_system_prompt
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # Revalidate in case a caller has mutated the public extra_body mapping.
        return merge_extra_body(payload, self.extra_body)

    def parse_response(self, request: DecisionRequest, data: Mapping[str, Any]) -> DecisionResult:
        """Validate JSON, optionally wrapped in one complete JSON code fence."""
        raw = {"probability_source": "self_reported"}
        text = None
        try:
            if request.question.answer.kind == "score":
                validate_score_spec(request)
            else:
                _options(request)
            _question_key(request)
            if not isinstance(data, Mapping):
                raise ValueError("invalid response envelope")
            if isinstance(data.get("model"), str):
                raw["model"] = data["model"]
            if isinstance(data.get("usage"), Mapping):
                usage = _archive(dict(data["usage"]))
                if usage is not None:
                    raw["usage"] = usage
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
                raise ValueError("missing response choice")
            completion = choices[0]
            message = completion.get("message")
            if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
                raise ValueError("missing text content")
            text = message["content"]
            # Accept only a fence wrapping the entire reply, never extract JSON
            # from prose or repair its contents. Keep the original text for audit.
            fenced = _JSON_CODE_FENCE.fullmatch(text.strip())
            if fenced is not None:
                raw["output_wrapper"] = "markdown_code_fence"
            response = _strict_json(fenced.group("body") if fenced is not None else text)
            raw["response"] = _archive(response)
            # Preserve parseable truncated answers for audit, but never accept them.
            if (completion.get("finish_reason") != "stop" or message.get("refusal")
                    or message.get("tool_calls") or message.get("function_call")):
                raise ValueError("completion did not finish normally")
            if request.question.answer.kind == "score":
                value = validate_generated_score(request, response)
                return DecisionResult.ok(request.request_id, value=value, text=text, raw=raw)
            choice = validate_generated_choice(request, response)
            return DecisionResult.ok(request.request_id, choice_id=choice, text=text, raw=raw)
        except (ValueError, TypeError, OverflowError, AttributeError, RecursionError):
            return replace(DecisionResult.failure(request.request_id, "invalid_response", raw=raw), text=text)

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


register_model("prompted", PromptedLLM)
