"""The data view an advantage implementation reads.

The view is read-only and purely local: no method here contacts the network,
launches collection or issues new evaluations. Missing data is reported as
absence (``None`` / ``()``) where the contract says so, and raised with the
offending key otherwise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from decision_model.advantage.base import (
    HistoryData,
    RowData,
    RowRecordRef,
    Score,
    ScoreQuery,
    StepKey,
    StepRecord,
)


class AdvantageDataSource(ABC):
    """Row space, record space and the mapping between them."""

    # ---- row space -------------------------------------------------------

    @property
    @abstractmethod
    def n_rows(self) -> int:
        """Number of rows currently in training."""
        raise NotImplementedError

    @property
    @abstractmethod
    def row_keys(self) -> Tuple[StepKey, ...]:
        """Record key for each row, in row order."""
        raise NotImplementedError

    @property
    @abstractmethod
    def row_data(self) -> RowData:
        """Row-aligned named training data."""
        raise NotImplementedError

    @property
    @abstractmethod
    def history(self) -> Optional[HistoryData]:
        """Caller-retained history, or ``None`` when unavailable."""
        raise NotImplementedError

    @property
    def history_available(self) -> bool:
        """True when history was provided rather than absent."""
        return self.history is not None

    # ---- record space ----------------------------------------------------

    @abstractmethod
    def record(self, key: StepKey) -> StepRecord:
        """The record for ``key``; raises ``KeyError`` when it is unknown."""
        raise NotImplementedError

    @abstractmethod
    def has_record(self, key: StepKey) -> bool:
        """True when ``key`` exists in the record space."""
        raise NotImplementedError

    @abstractmethod
    def record_optional(self, key: StepKey) -> Optional[StepRecord]:
        """The record for ``key``, or ``None``; never raises for missing keys."""
        raise NotImplementedError

    @abstractmethod
    def rows_for(self, key: StepKey) -> Tuple[int, ...]:
        """Row indices referencing ``key``; empty, single or duplicated."""
        raise NotImplementedError

    @abstractmethod
    def record_keys(self) -> Tuple[StepKey, ...]:
        """Every key in the record space, a superset of the referenced ones."""
        raise NotImplementedError

    # ---- scores ----------------------------------------------------------

    @abstractmethod
    def score(self, key: StepKey, query: Optional[ScoreQuery] = None) -> Optional[Score]:
        """First score stored for ``key`` matching ``query``, or ``None``."""
        raise NotImplementedError

    @abstractmethod
    def scores(self, key: StepKey, query: Optional[ScoreQuery] = None) -> Tuple[Score, ...]:
        """Every score stored for ``key`` matching ``query``."""
        raise NotImplementedError

    # ---- named row-aligned data ------------------------------------------

    @abstractmethod
    def row_value(self, name: str, index: int) -> Any:
        """Value of row-aligned series ``name`` at ``index``; ``KeyError`` if unknown."""
        raise NotImplementedError

    @abstractmethod
    def optional_row_values(self, name: str) -> Optional[Sequence[Any]]:
        """Row-aligned series ``name``, or ``None`` when it was not supplied."""
        raise NotImplementedError


class InMemoryAdvantageData(AdvantageDataSource):
    """Concrete view over explicit in-memory records and row references.

    ``row_refs`` defines the row space: it must name row indices ``0..N-1``
    exactly once each, and N is derived from it. Records are shared, never
    copied per row, so several rows may point at one record and a record may be
    referenced by no row at all.
    """

    def __init__(
        self,
        records: Any,
        row_refs: Sequence[RowRecordRef],
        row_data: RowData,
        history: Optional[HistoryData] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if isinstance(records, Mapping):
            self._records: Dict[StepKey, StepRecord] = dict(records)
        else:
            self._records = {}
            for record in records:
                if not isinstance(record, StepRecord):
                    raise TypeError(
                        "records must contain StepRecord objects, got {!r}".format(
                            type(record).__name__
                        )
                    )
                if record.key in self._records:
                    raise ValueError(
                        "duplicate record for key {} in the supplied sequence".format(record.key)
                    )
                self._records[record.key] = record

        self._row_refs: Tuple[RowRecordRef, ...] = tuple(row_refs)
        n_rows = len(self._row_refs)

        # Row indices must cover 0..N-1 exactly once each.
        seen: Dict[int, StepKey] = {}
        for ref in self._row_refs:
            if ref.row_index in seen:
                raise ValueError(
                    "row index {} is referenced more than once; row_refs must cover "
                    "0..{} exactly once each".format(ref.row_index, n_rows - 1)
                )
            seen[ref.row_index] = ref.key
        missing = [index for index in range(n_rows) if index not in seen]
        if missing:
            raise ValueError(
                "row_refs must cover row indices 0..{} exactly once each; missing {}".format(
                    n_rows - 1, missing
                )
            )

        # Every referenced key must exist in the record space.
        for ref in self._row_refs:
            if ref.key not in self._records:
                raise KeyError(
                    "row {} references record {} which is not present in the record "
                    "space; {} record(s) available".format(
                        ref.row_index, ref.key, len(self._records)
                    )
                )

        if not isinstance(row_data, RowData):
            raise TypeError(
                "row_data must be a RowData, got {!r}".format(type(row_data).__name__)
            )
        self._row_data = row_data
        self._validate_row_data_lengths(row_data, n_rows)

        self._history = history
        self._metadata: Dict[str, Any] = dict(metadata) if metadata else {}

        self._rows_by_key: Dict[StepKey, List[int]] = {}
        for index in range(n_rows):
            self._rows_by_key.setdefault(seen[index], []).append(index)

    @staticmethod
    def _validate_row_data_lengths(row_data: RowData, n_rows: int) -> None:
        def _check(name: str, length: int) -> None:
            if length != n_rows:
                raise ValueError(
                    "row-aligned data {!r} has {} entries but the row space has N={} "
                    "rows".format(name, length, n_rows)
                )

        _check("uid", len(row_data.uid))
        _check("reward", len(row_data.reward))
        if row_data.valid is not None:
            _check("valid", len(row_data.valid))
        for name, sequence in row_data.values.items():
            if hasattr(sequence, "__len__"):
                _check(name, len(sequence))

    # ---- row space -------------------------------------------------------

    @property
    def n_rows(self) -> int:
        return len(self._row_refs)

    @property
    def row_keys(self) -> Tuple[StepKey, ...]:
        keys: List[StepKey] = []
        for ref in sorted(self._row_refs, key=lambda r: r.row_index):
            record = self._records.get(ref.key)
            if record is None:
                raise KeyError(
                    "row {} references missing record {}".format(ref.row_index, ref.key)
                )
            keys.append(record.key)
        return tuple(keys)

    @property
    def row_data(self) -> RowData:
        return self._row_data

    @property
    def history(self) -> Optional[HistoryData]:
        return self._history

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Caller-supplied metadata that came with the data."""
        return self._metadata

    # ---- record space ----------------------------------------------------

    def record(self, key: StepKey) -> StepRecord:
        try:
            return self._records[key]
        except KeyError:
            raise KeyError(
                "unknown record {}; {} record(s) available in the record space".format(
                    key, len(self._records)
                )
            ) from None

    def has_record(self, key: StepKey) -> bool:
        return key in self._records

    def record_optional(self, key: StepKey) -> Optional[StepRecord]:
        return self._records.get(key)

    def rows_for(self, key: StepKey) -> Tuple[int, ...]:
        return tuple(self._rows_by_key.get(key, ()))

    def record_keys(self) -> Tuple[StepKey, ...]:
        keys = list(self._records.keys())
        keys.sort(key=lambda k: (k.trajectory_id, k.step_index))
        return tuple(keys)

    # ---- scores ----------------------------------------------------------

    def score(self, key: StepKey, query: Optional[ScoreQuery] = None) -> Optional[Score]:
        record = self._records.get(key)
        if record is None:
            return None
        return record.score(query)

    def scores(self, key: StepKey, query: Optional[ScoreQuery] = None) -> Tuple[Score, ...]:
        record = self._records.get(key)
        if record is None:
            return ()
        return record.scores_matching(query)

    # ---- named row-aligned data ------------------------------------------

    def row_value(self, name: str, index: int) -> Any:
        if name == "uid":
            sequence: Sequence[Any] = self._row_data.uid
        elif name == "reward":
            sequence = self._row_data.reward
        elif name == "valid":
            if self._row_data.valid is None:
                raise KeyError(
                    "row-aligned data 'valid' is unavailable for this data view"
                )
            sequence = self._row_data.valid
        elif name in self._row_data.values:
            sequence = self._row_data.values[name]
        else:
            raise KeyError(
                "unknown row-aligned data {!r}; available: {}".format(
                    name, ", ".join(self._available_row_names()) or "<none>"
                )
            )
        if index < 0:
            index += len(sequence)
        if index < 0 or index >= len(sequence):
            raise IndexError(
                "row index {} is out of range for row-aligned data {!r} with {} "
                "entr(ies)".format(index, name, len(sequence))
            )
        return sequence[index]

    def optional_row_values(self, name: str) -> Optional[Sequence[Any]]:
        if name == "uid":
            return self._row_data.uid
        if name == "reward":
            return self._row_data.reward
        if name == "valid":
            return self._row_data.valid
        return self._row_data.values.get(name)

    def _available_row_names(self) -> List[str]:
        names = ["uid", "reward"]
        if self._row_data.valid is not None:
            names.append("valid")
        names.extend(self._row_data.values.keys())
        return names

    def available_row_names(self) -> Tuple[str, ...]:
        """Every name that can be read from the row space."""
        return tuple(self._available_row_names())

    def __repr__(self) -> str:
        return "InMemoryAdvantageData(n_rows={}, n_records={})".format(
            self.n_rows, len(self._records)
        )
