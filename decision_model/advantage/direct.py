"""Direct step scores, aligned to the current training rows."""

from __future__ import annotations

from typing import Any

from decision_model.advantage._selection import ScoreSelector, failure_policy, row_layout, selected_score
from decision_model.advantage.base import AdvantageConfig, BaseStepAdvantage, StepAdvantageResult
from decision_model.advantage.view import AdvantageDataSource
from decision_model.registry import register_advantage


class DirectScoreAdvantage(BaseStepAdvantage):
    """Use one selected score per row; ineligible or skipped failed rows are zero."""

    name = "direct"

    def __init__(self, query: Any, fail_on_error: bool = True) -> None:
        self.selector = ScoreSelector(query)
        self.fail_on_error = failure_policy(fail_on_error)

    def compute(self, data: AdvantageDataSource, config: AdvantageConfig) -> StepAdvantageResult:
        keys, valid = row_layout(data)
        values = []
        statuses = []
        eligible = []
        for index, key in enumerate(keys):
            value, status = selected_score(data, key, self.selector, index, self.fail_on_error)
            participates = valid[index] and status == "ok"
            values.append(value if participates else 0.0)
            statuses.append(status)
            eligible.append(participates)
        return StepAdvantageResult(tuple(values), tuple(statuses), {"eligible": eligible})


register_advantage(DirectScoreAdvantage.name, DirectScoreAdvantage)
