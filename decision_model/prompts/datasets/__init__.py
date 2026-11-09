"""Dataset adapters for ALFWorld and WebShop trajectory records."""

from __future__ import annotations

from decision_model.prompts.datasets.base import (
    BaseDatasetPrompt,
    build_snapshot,
    record_to_snapshot,
    snapshot_from_record,
)

__all__ = [
    "BaseDatasetPrompt",
    "build_snapshot",
    "record_to_snapshot",
    "snapshot_from_record",
]
