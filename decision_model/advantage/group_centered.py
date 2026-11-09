"""Center selected scores within each actual training group, without std scaling."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List

from decision_model.advantage._selection import (
    ScoreSelector,
    failure_policy,
    finite_number,
    row_layout,
    selected_score,
)
from decision_model.advantage.base import (
    AdvantageConfig,
    BaseStepAdvantage,
    StepAdvantageResult,
    require_row_values,
)
from decision_model.advantage.view import AdvantageDataSource
from decision_model.registry import register_advantage


class GroupCenteredAdvantage(BaseStepAdvantage):
    """Subtract the mean over valid successful rows with the same ``uid``.

    Duplicated rows participate repeatedly. Records with no current rows are
    never included in the mean. ``valid=None`` means no extra row restriction;
    a training bridge must supply its GRPO mask eligibility as ``valid``.
    """

    name = "group_centered"

    def __init__(self, query: Any, fail_on_error: bool = True) -> None:
        self.selector = ScoreSelector(query)
        self.fail_on_error = failure_policy(fail_on_error)

    def compute(self, data: AdvantageDataSource, config: AdvantageConfig) -> StepAdvantageResult:
        keys, valid = row_layout(data)
        groups = require_row_values(data, "uid")
        if len(groups) != data.n_rows:
            raise ValueError("row-aligned uid must contain exactly N={} values".format(data.n_rows))
        values = []
        statuses = []
        eligible = []
        members: Dict[str, List[int]] = {}
        for index, key in enumerate(keys):
            if not isinstance(groups[index], str):
                raise ValueError("row {} uid must be a string".format(index))
            value, status = selected_score(data, key, self.selector, index, self.fail_on_error)
            participates = valid[index] and status == "ok"
            values.append(value)
            statuses.append(status)
            eligible.append(participates)
            if participates:
                members.setdefault(groups[index], []).append(index)

        # The standard-library mean accumulates floats as exact ratios before
        # the final division. This avoids both overflowing a finite group's sum
        # and losing subnormal scores by dividing each member before summation.
        means = {
            group: mean(values[index] for index in indices)
            for group, indices in members.items()
        }
        centered = tuple(
            finite_number(values[index] - means[groups[index]], "centered value for row {}".format(index))
            if participates else 0.0
            for index, participates in enumerate(eligible)
        )
        return StepAdvantageResult(
            centered,
            tuple(statuses),
            {
                "eligible": eligible,
                "group_means": means,
                "group_counts": {group: len(indices) for group, indices in members.items()},
            },
        )


register_advantage(GroupCenteredAdvantage.name, GroupCenteredAdvantage)
