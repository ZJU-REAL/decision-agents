"""Step-level advantage: data interface, data view and dispatch.

``compute_step_advantage`` dispatches to registered implementations, including
the paper's group-centered Teacher scores and optional direct scores.
"""

from __future__ import annotations

from decision_model.advantage.base import (
    AdvantageConfig,
    BaseStepAdvantage,
    HistoryData,
    MissingDataError,
    RowData,
    RowRecordRef,
    Score,
    ScoreQuery,
    StepAdvantageResult,
    StepKey,
    StepRecord,
    compute_step_advantage,
    require_record,
    require_row_values,
    require_score,
    row_indices_with_uid,
)
from decision_model.advantage.view import AdvantageDataSource, InMemoryAdvantageData

__all__ = [
    "AdvantageConfig",
    "AdvantageDataSource",
    "BaseStepAdvantage",
    "HistoryData",
    "InMemoryAdvantageData",
    "MissingDataError",
    "RowData",
    "RowRecordRef",
    "Score",
    "ScoreQuery",
    "StepAdvantageResult",
    "StepKey",
    "StepRecord",
    "compute_step_advantage",
    "require_record",
    "require_row_values",
    "require_score",
    "row_indices_with_uid",
]
