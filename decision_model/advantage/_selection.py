"""Shared score selection for step-level advantage formulas."""

from __future__ import annotations

import math
import re
from numbers import Real
from string import Formatter
from typing import Any, Mapping, Optional, Sequence, Tuple

from decision_model.advantage.base import MissingDataError, ScoreQuery, StepKey
from decision_model.advantage.view import AdvantageDataSource


def nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(name))
    return value


def finite_number(value: Any, name: str) -> float:
    """Reject missing, boolean, nonnumeric and nonfinite numerical inputs."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("{} must be a finite real number; got {!r}".format(name, value))
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError("{} must be a finite real number".format(name)) from None
    if not math.isfinite(number):
        raise ValueError("{} must be finite; got {!r}".format(name, value))
    return number


def failure_policy(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("fail_on_error must be a boolean")
    return value


class ScoreSelector:
    """Normalize one algorithm option without changing the public ScoreQuery.

    A mapping may additionally use ``position_key_template``. Only plain
    ``{step_index}`` and ``{trajectory_id}`` fields are accepted; literal
    ``position_key`` values are never interpreted as templates.
    """

    def __init__(self, query: Any) -> None:
        self.template: Optional[str] = None
        if isinstance(query, ScoreQuery):
            self.query = query
        elif isinstance(query, Mapping):
            allowed = {"metric", "config_name", "position_key", "position_key_template"}
            unknown = set(query) - allowed
            if unknown:
                raise ValueError("unknown score query field(s): {}".format(
                    ", ".join(sorted(repr(key) for key in unknown))
                ))
            self.query = ScoreQuery.from_dict(query)
            self.template = query.get("position_key_template")
        else:
            raise TypeError("query must be a ScoreQuery or a mapping")

        nonempty_string(self.query.metric, "query.metric")
        nonempty_string(self.query.config_name, "query.config_name")
        if self.query.position_key is not None:
            nonempty_string(self.query.position_key, "query.position_key")
        if self.template is not None:
            nonempty_string(self.template, "query.position_key_template")
            if self.query.position_key is not None:
                raise ValueError("position_key and position_key_template are mutually exclusive")
            for _, field, format_spec, conversion in Formatter().parse(self.template):
                if field is not None and (
                    field not in {"step_index", "trajectory_id"} or format_spec or conversion
                ):
                    raise ValueError(
                        "position_key_template only accepts plain {step_index} and "
                        "{trajectory_id} fields, without conversion or format specification"
                    )
            # Formatter.parse represents both {step_index} and {step_index:}
            # with an empty format_spec. Accept only the literal plain field
            # syntax (and escaped braces), including that empty-spec case.
            remainder = re.sub(r"\{\{|\}\}|\{step_index\}|\{trajectory_id\}", "", self.template)
            if "{" in remainder or "}" in remainder:
                raise ValueError("position_key_template fields must use plain {step_index} or {trajectory_id}")

    def at(self, key: StepKey) -> ScoreQuery:
        if self.template is None:
            return self.query
        return ScoreQuery(
            metric=self.query.metric,
            config_name=self.query.config_name,
            position_key=self.template.format(
                trajectory_id=key.trajectory_id, step_index=key.step_index
            ),
        )


def row_layout(data: AdvantageDataSource) -> Tuple[Sequence[StepKey], Tuple[bool, ...]]:
    """Read row keys and eligibility only; never interpret the opaque mask."""
    keys = data.row_keys
    if len(keys) != data.n_rows:
        raise ValueError("row_keys must contain exactly N={} keys".format(data.n_rows))
    valid = data.optional_row_values("valid")
    if valid is None:
        return keys, (True,) * data.n_rows
    if len(valid) != data.n_rows:
        raise ValueError("row-aligned valid must contain exactly N={} values".format(data.n_rows))
    return keys, tuple(bool(value) for value in valid)


def selected_score(
    data: AdvantageDataSource,
    key: StepKey,
    selector: ScoreSelector,
    row_index: int,
    fail_on_error: bool,
) -> Tuple[float, str]:
    """Read exactly one identified score, preserving explicit failure status."""
    query = selector.at(key)
    context = "row {} record {} query {}".format(row_index, key, query)
    if not data.has_record(key):
        raise MissingDataError("required record is unavailable for {}".format(context))
    scores = data.scores(key, query)
    if not scores:
        raise MissingDataError("required score is unavailable for {}".format(context))
    if len(scores) != 1:
        raise ValueError("ambiguous score selection: {} matches for {}".format(len(scores), context))
    score = scores[0]
    if score.status not in ("ok", "error"):
        raise ValueError("unknown score status {!r} for {}".format(score.status, context))
    if score.status == "error":
        if score.value is not None:
            raise ValueError("failed score must retain value=None for {}".format(context))
        if fail_on_error:
            raise RuntimeError("evaluation score failed for {}".format(context))
        return 0.0, "error"
    return finite_number(score.value, "score for {}".format(context)), "ok"
