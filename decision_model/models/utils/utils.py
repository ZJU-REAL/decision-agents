"""Shared validation, input rendering and HTTP helpers for decision clients.

Transport, authentication, retries and error mapping belong to each client.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Mapping, Tuple
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request

from decision_model.types import ChoiceOption, DecisionRequest


def generation_extra_body(value: Any) -> Dict[str, Any]:
    """Validate and copy JSON extras without imposing service-specific field rules."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("extra_body must be a mapping or None")
    try:
        # The round trip rejects cycles, arbitrary objects and non-finite floats.
        serialized = json.dumps(dict(value), allow_nan=False)

        def check_keys(item: Any) -> None:
            if isinstance(item, dict):
                if any(not isinstance(key, str) for key in item):
                    raise ValueError("extra_body JSON object keys must be strings")
                for child in item.values():
                    check_keys(child)
            elif isinstance(item, (tuple, list)):
                for child in item:
                    check_keys(child)

        check_keys(dict(value))
        return json.loads(serialized)
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise ValueError("extra_body must contain finite JSON values with string object keys") from None


def merge_extra_body(payload: Mapping[str, Any], extra_body: Any) -> Dict[str, Any]:
    """Add JSON extras without silently replacing fields already in this request."""
    extra = generation_extra_body(extra_body)
    if payload.keys() & extra.keys():
        raise ValueError("extra_body must not overwrite existing request fields")
    return {**payload, **extra}


def http_url(value: Any, field: str = "endpoint") -> str:
    """Validate an HTTP(S) address while preserving its complete path."""
    error = "{} must be an absolute HTTP(S) URL without credentials, query or fragment".format(field)
    if not isinstance(value, str) or not value or any(c.isspace() or ord(c) < 32 for c in value):
        raise ValueError(error)
    try:
        parts = urlsplit(value)
        if (parts.scheme not in ("http", "https") or not parts.hostname
                or parts.username is not None or parts.password is not None
                or parts.query or parts.fragment or "?" in value or "#" in value
                or parts.port == 0):
            raise ValueError(error)
    except ValueError:
        raise ValueError(error) from None
    return value


def finite_number(value: Any) -> bool:
    try:
        return (not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value))
    except OverflowError:
        return False


def positive_timeout(value: Any) -> float:
    if not finite_number(value) or value <= 0:
        raise ValueError("timeout must be a finite positive number")
    return float(value)


def _unique_object(pairs: Any) -> Dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError("non-finite JSON constant")


def strict_json_loads(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def archive_json(value: Any) -> Any:
    """Detach JSON values, including safely recording invalid numeric replies."""
    try:
        return json.loads(json.dumps(value), parse_constant=str)
    except (ValueError, TypeError, OverflowError, RecursionError):
        return None


def question_key(request: DecisionRequest) -> str:
    # The same key associates the supplied question with its returned answer.
    key = request.question.materials.get("question_key")
    if not isinstance(key, str) or not key:
        raise ValueError("question materials must supply a non-empty question_key")
    return key


def json_post_request(
    *, url: str, headers: Mapping[str, str], payload: Mapping[str, Any]
) -> Request:
    """Construct a JSON POST without opening a connection."""
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return Request(url, data=body, headers=dict(headers), method="POST")


class NoRedirect(HTTPRedirectHandler):
    """Keep the POST and its credentials at the explicitly configured endpoint."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        return None


def choice_options(request: DecisionRequest) -> Tuple[ChoiceOption, ...]:
    if request.question.answer.kind != "choice":
        raise ValueError("choice input requires choice answers")
    options = request.question.answer.choices
    if not 1 <= len(options) <= 255:
        raise ValueError("unsupported_candidate_count")
    if any(not isinstance(option, ChoiceOption)
           or not isinstance(option.option_id, str) or not option.option_id.strip()
           or not isinstance(option.text, str) or not option.text.strip() for option in options):
        raise ValueError("choice options require non-empty IDs and text")
    if len({option.option_id for option in options}) != len(options):
        raise ValueError("choice option IDs must be unique")
    return options



def render_choice_input(request: DecisionRequest) -> str:
    """Render only the visible state, question and unchanged ordered options."""
    options = choice_options(request)
    question = request.question
    key = question_key(request)
    if not isinstance(question.question, str) or not question.question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(question.answer.instructions, str):
        raise ValueError("answer instructions must be a string")
    instructions = question.question
    if question.answer.instructions:
        instructions += "\n\n" + question.answer.instructions
    if not isinstance(request.context.text, str):
        raise ValueError("context text must be a string")
    if "state" in request.context.materials:
        if request.context.text:
            raise ValueError("supply structured state or context text, not both")
        state = request.context.materials["state"]
    else:
        state = request.context.text
    content = "Question key: {}\n\nState:\n{}\n\nQuestion:\n{}\n\nAvailable options:\n{}".format(
        key,
        json.dumps(state, ensure_ascii=False, allow_nan=False),
        instructions,
        json.dumps([{"id": option.option_id, "action": option.text} for option in options],
                   ensure_ascii=False, allow_nan=False),
    )
    return content



def validate_score_spec(request: DecisionRequest) -> None:
    """Require a score on the numeric unit interval without choice options."""
    answer = request.question.answer
    if answer.kind != "score":
        raise ValueError("score input requires score answers")
    if (len(answer.scale) != 2 or not all(finite_number(bound) for bound in answer.scale)
            or answer.scale[0] != 0 or answer.scale[1] != 1):
        raise ValueError("score scale must be the finite numeric interval (0, 1)")
    if answer.choices:
        raise ValueError("score answers must not supply choice options")



def render_score_input(request: DecisionRequest) -> str:
    """Render only the visible state, original question and answer instructions."""
    validate_score_spec(request)
    question = request.question
    key = question_key(request)
    if not isinstance(question.question, str) or not question.question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(question.answer.instructions, str):
        raise ValueError("answer instructions must be a string")
    instructions = question.question
    if question.answer.instructions:
        instructions += "\n\n" + question.answer.instructions
    if not isinstance(request.context.text, str):
        raise ValueError("context text must be a string")
    if "state" in request.context.materials:
        if request.context.text:
            raise ValueError("supply structured state or context text, not both")
        state = request.context.materials["state"]
    else:
        state = request.context.text
    return "Question key: {}\n\nState:\n{}\n\nQuestion:\n{}".format(
        key, json.dumps(state, ensure_ascii=False, allow_nan=False), instructions,
    )
