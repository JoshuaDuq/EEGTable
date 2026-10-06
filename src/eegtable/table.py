from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from eegtable._validation import validate_fraction_array
from eegtable.bands import Band
from eegtable.naming import feature_name

SpaceKind = Literal["channel", "roi", "global", "pair", "state"]
Normalization = Literal["raw", "log10", "log_ratio", "db", "percent"]
RowId = tuple[str, int, str]
"""Immutable ``(recording, epoch, event)`` identity for one epoch row."""


@dataclass(frozen=True)
class ComputationSpec:
    """Canonical, immutable description of an algorithm and its parameters."""

    method: str
    parameters_json: str

    def __post_init__(self) -> None:
        if not self.method:
            raise ValueError("computation method must be non-empty.")
        parsed = json.loads(self.parameters_json)
        canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if canonical != self.parameters_json:
            raise ValueError("parameters_json must use canonical JSON serialization.")

    @classmethod
    def create(cls, method: str, **parameters: object) -> ComputationSpec:
        canonical = json.dumps(
            _json_value(parameters), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return cls(method=method, parameters_json=canonical)

    @property
    def parameters(self) -> Mapping[str, object]:
        """Decoded parameters."""
        parsed = json.loads(self.parameters_json)
        if not isinstance(parsed, dict):
            raise TypeError("canonical computation parameters must be a JSON object.")
        return parsed

    @property
    def parameter_hash(self) -> str:
        """Stable SHA-256 digest of the complete computation specification."""
        payload = f"{self.method}\n{self.parameters_json}".encode()
        return hashlib.sha256(payload).hexdigest()

    def record(self) -> dict[str, object]:
        """JSON-serializable representation."""
        return {"method": self.method, "parameters": dict(self.parameters)}


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"computation parameter {value!r} is not JSON-serializable.")


# A string no field can hold, encoded once, marking where the computation's text goes.
_SPLICE = "\x00computation\x00"
_SPLICE_JSON = json.dumps(_SPLICE)


def _band_record(band: Band | None) -> dict[str, object] | None:
    if band is None:
        return None
    return {"name": band.name, "fmin": band.fmin, "fmax": band.fmax}


_ROW_ID_FIELDS = ("recording", "epoch", "event")
_GROUP_FIELD = "group"
_LONG_FIELDS = (
    "feature",
    "measure",
    "band",
    "band_fmin",
    "band_fmax",
    "space",
    "space_kind",
    "window",
    "window_tmin",
    "window_tmax",
    "normalization",
    "unit",
    "source",
)


def _long_record(meta: FeatureMeta) -> tuple[object, ...]:
    tmin, tmax = meta.window_bounds if meta.window_bounds is not None else (None, None)
    band = meta.band
    return (
        meta.name,
        meta.measure,
        meta.band_label,
        band.fmin if band is not None else None,
        band.fmax if band is not None else None,
        meta.space,
        meta.space_kind,
        meta.window,
        tmin,
        tmax,
        meta.normalization,
        meta.unit,
        meta.source,
    )


def _matches(field: object, value: object) -> bool:
    # Feature names and recipes write a band by its name, so a name selects it too.
    if isinstance(field, Band) and isinstance(value, str):
        return field.name == value
    return field == value


def _label(field: object) -> object:
    return field.name if isinstance(field, Band) else field


@dataclass(frozen=True)
class FeatureMeta:
    """Structured description of one feature column.

    Every field is constant across epochs. Per-epoch facts belong in
    :attr:`FeatureTable.flags`, not here.

    Parameters
    ----------
    measure : str
        Measure label, e.g. ``"band_power"``, ``"peak_freq"``, ``"slope"``. It names
        the quantity, not the function that produced it, and is what
        :meth:`FeatureTable.select` and ``eegtable.model.Selection`` match on.
    band : Band or None
        The band this column was computed over, or None for measures spanning
        a fitted range rather than a band.
    space : str
        Channel name, ROI name, or ``"global"``.
    space_kind : {"channel", "roi", "global", "pair", "state"}
        Which of those ``space`` is. ``"pair"`` marks a derived unit relating two
        nodes, such as an asymmetry or a connection; ``"state"`` marks a
        microstate class, which is a spatial mode rather than a location.
    window : str or None
        Time window name, or None when the spectrum spans the whole segment.
    normalization : {"raw", "log10", "log_ratio", "db", "percent"}
        Normalization applied to the value.
    unit : str
        Physical unit, or a description of the normalized scale.
    source : str
        Provenance of the spectra, e.g. ``"morlet"`` or ``"multitaper"``.
    freq_resolution_hz : float or None
        Median spacing of the frequency bins inside ``band``. Reported so a
        caller can judge whether the grid supported the measure; it never
        gates anything.
    """

    measure: str
    band: Band | None
    space: str
    space_kind: SpaceKind
    window: str | None
    normalization: Normalization
    unit: str
    source: str
    window_bounds: tuple[float, float] | None
    computation: ComputationSpec
    freq_resolution_hz: float | None = None
    phase_band: Band | None = None
    amplitude_band: Band | None = None
    nodes: tuple[str, str] | None = None

    def record(self) -> dict[str, object]:
        """JSON-serializable form of every field that defines this column."""
        return {**self.fields_record(), "computation": self.computation.record()}

    def fields_record(self) -> dict[str, object]:
        """:meth:`record` without the computation, which a sidecar stores once per table."""
        return {
            "measure": self.measure,
            "band": _band_record(self.band),
            "space": self.space,
            "space_kind": self.space_kind,
            "window": self.window,
            "window_bounds": self.window_bounds,
            "normalization": self.normalization,
            "unit": self.unit,
            "source": self.source,
            "freq_resolution_hz": self.freq_resolution_hz,
            "phase_band": _band_record(self.phase_band),
            "amplitude_band": _band_record(self.amplitude_band),
            "nodes": self.nodes,
        }

    # Cached because the record is immutable and hashing it is not cheap: reading a cohort
    # named every one of its 13,000 columns five times.
    @cached_property
    def parameter_hash(self) -> str:
        """Stable digest of every field that defines this feature column."""
        # The canonical JSON of record(), byte for byte, with the computation spliced in from
        # its own canonical text: decoding and re-encoding it, a 64-channel list for every
        # connectivity column, was most of the cost of naming a table.
        fields = json.dumps(
            _json_value({**self.fields_record(), "computation": _SPLICE}),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        computation = (
            f'{{"method":{json.dumps(self.computation.method)},'
            f'"parameters":{self.computation.parameters_json}}}'
        )
        return hashlib.sha256(fields.replace(_SPLICE_JSON, computation, 1).encode()).hexdigest()

    @property
    def band_label(self) -> str | None:
        """The band as the name writes it: phase and amplitude bands for coupling."""
        if self.phase_band is not None and self.amplitude_band is not None:
            return f"phase-{self.phase_band.name}--amp-{self.amplitude_band.name}"
        return self.band.name if self.band is not None else None

    @cached_property
    def name(self) -> str:
        """The canonical feature name for this column."""
        readable = feature_name(
            measure=self.measure,
            band=self.band_label,
            space=self.space,
            window=self.window,
            normalization=self.normalization,
        )
        return f"{readable}_p{self.parameter_hash[:12]}"


@dataclass(frozen=True, eq=False)
class FeatureTable:
    """Feature values with one metadata record per column.

    Parameters
    ----------
    values : ndarray, shape (n_epochs, n_features)
        Feature values. NaN marks a value withheld because of a data
        condition; see ``coverage`` for how much valid input it had.
    coverage : ndarray, shape (n_epochs, n_features)
        Fraction of numerically finite input that produced each value, in
        ``[0, 1]``. This is not an artifact-free-data score.
    meta : tuple of FeatureMeta
        One record per column.
    flags : mapping of str to ndarray, optional
        Per-cell boolean annotations, each shaped like ``values``.
    row_labels : tuple of str, optional
        Names for the rows when they are **not** epochs. None, the default, means
        one row per epoch. A measure estimated across trials, such as inter-trial
        phase coherence, has one row per trial group and names them here, so a
        table of group rows cannot be silently joined to a table of epoch rows.
    row_ids : tuple of (str, int, str), optional
        Recording, original epoch index and event identity for every epoch row.
        Required when concatenating per-epoch tables. Group-row tables use
        ``row_labels`` instead, and cannot also carry ``row_ids``.
    support : ndarray, shape (n_epochs, n_features), optional
        Fraction of the requested window each value rests on, in ``[0, 1]``. Morlet
        power is averaged only over coefficients whose whole wavelet fits the
        window, so a short window, or a low frequency, can leave a sliver of it.
        None, the default, means every value rests on its whole window, as Welch,
        multitaper and time-domain values do.
    """

    values: npt.NDArray[np.float64]
    coverage: npt.NDArray[np.float64]
    meta: tuple[FeatureMeta, ...]
    flags: Mapping[str, npt.NDArray[np.bool_]] = field(default_factory=dict)
    row_labels: tuple[str, ...] | None = None
    row_ids: tuple[RowId, ...] | None = None
    support: npt.NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if np.iscomplexobj(self.values):
            raise TypeError("Feature values must be real.")
        if self.values.ndim != 2:
            raise ValueError(f"values must be 2-D (n_epochs, n_features), got {self.values.shape}.")
        if self.coverage.shape != self.values.shape:
            raise ValueError(
                f"coverage shape {self.coverage.shape} does not match values {self.values.shape}."
            )
        validate_fraction_array(self.coverage, "coverage")
        if self.support is not None:
            if self.support.shape != self.values.shape:
                raise ValueError(
                    f"support shape {self.support.shape} does not match values "
                    f"{self.values.shape}."
                )
            validate_fraction_array(self.support, "support")
        if len(self.meta) != self.values.shape[1]:
            raise ValueError(
                f"meta has {len(self.meta)} records but values has {self.values.shape[1]} columns."
            )
        for key, array in self.flags.items():
            if array.dtype != np.bool_:
                raise TypeError(f"flag {key!r} must contain boolean values.")
            if array.shape != self.values.shape:
                raise ValueError(
                    f"flag {key!r} shape {array.shape} does not match values {self.values.shape}."
                )
        if self.row_labels is not None and len(self.row_labels) != self.values.shape[0]:
            raise ValueError(
                f"row_labels has {len(self.row_labels)} entries but values has "
                f"{self.values.shape[0]} rows."
            )
        if self.row_ids is not None and len(self.row_ids) != self.values.shape[0]:
            raise ValueError(
                f"row_ids has {len(self.row_ids)} entries but values has "
                f"{self.values.shape[0]} rows."
            )
        if self.row_labels is not None and self.row_ids is not None:
            raise ValueError("group row_labels and epoch row_ids are mutually exclusive.")
        if self.row_ids is not None:
            for row_id in self.row_ids:
                if (
                    len(row_id) != 3
                    or not isinstance(row_id[0], str)
                    or not isinstance(row_id[1], int)
                    or not isinstance(row_id[2], str)
                ):
                    raise ValueError(
                        "each row_id must be (recording: str, epoch: int, event: str)."
                    )
        names = self.names
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate feature names: {duplicates}")

    @property
    def names(self) -> list[str]:
        """Canonical name of every column, in order."""
        return [m.name for m in self.meta]

    @property
    def n_rows(self) -> int:
        """Number of rows: epochs, or trial groups when ``row_labels`` is set."""
        return int(self.values.shape[0])

    def __repr__(self) -> str:
        # The generated repr printed every array and metadata record, hundreds of kilobytes.
        rows = "epoch" if self.row_labels is None else "group"
        measures = sorted({m.measure for m in self.meta})
        shown = ", ".join(measures[:5]) + (", …" if len(measures) > 5 else "")
        return (
            f"FeatureTable({self.n_rows} {rows} rows × {len(self.meta)} columns; "
            f"measures: {shown or 'none'})"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Render the values as a DataFrame with canonical column names.

        Epoch rows are indexed by ``(recording, epoch, event)``. MNE keeps each epoch's
        original number in ``epochs.metadata`` after a drop, so
        ``frame.join(epochs.metadata, on="epoch")`` pairs every row with its own
        trial, where a positional index would shift them. Trial-group rows are
        indexed by their label, and rows without an identity by position.
        """
        return pd.DataFrame(self.values, columns=self.names, index=self._row_index())

    def to_long(self) -> pd.DataFrame:
        """Return one row per cell: the row's identity, the column's metadata, the value.

        This is the shape mixed models, R and plotting libraries read. Identity columns
        are ``recording``, ``epoch`` and ``event`` (``group`` for trial-group rows),
        followed by ``feature`` (the canonical name) and the column's ``measure``,
        ``band``, ``band_fmin``, ``band_fmax``, ``space``, ``space_kind``, ``window``,
        ``window_tmin``, ``window_tmax``, ``normalization``, ``unit`` and ``source``,
        then ``value``, ``coverage`` and ``support``. Each flag becomes a boolean column
        of its name.
        """
        n_rows, n_columns = self.values.shape
        rows = self._row_frame().iloc[np.repeat(np.arange(n_rows), n_columns)]
        columns = pd.DataFrame([_long_record(m) for m in self.meta], columns=list(_LONG_FIELDS))
        columns = columns.iloc[np.tile(np.arange(n_columns), n_rows)]
        frame = pd.concat([rows.reset_index(drop=True), columns.reset_index(drop=True)], axis=1)
        frame["value"] = self.values.reshape(-1)
        frame["coverage"] = self.coverage.reshape(-1)
        frame["support"] = self.support.reshape(-1) if self.support is not None else 1.0
        for key in sorted(self.flags):
            if key in frame.columns:
                raise ValueError(f"flag {key!r} would overwrite the long table's {key!r} column.")
            frame[key] = self.flags[key].reshape(-1)
        return frame

    def _row_index(self) -> pd.Index:
        if self.row_ids is not None:
            return pd.MultiIndex.from_frame(self._row_frame())
        if self.row_labels is not None:
            return pd.Index(self.row_labels, name=_GROUP_FIELD)
        return pd.RangeIndex(self.n_rows)

    def _row_frame(self) -> pd.DataFrame:
        if self.row_ids is not None:
            return pd.DataFrame(list(self.row_ids), columns=list(_ROW_ID_FIELDS))
        if self.row_labels is not None:
            return pd.DataFrame({_GROUP_FIELD: list(self.row_labels)})
        return pd.DataFrame({"row": np.arange(self.n_rows)})

    def select(self, **conditions: object) -> FeatureTable:
        """Return the columns whose metadata matches every given field.

        Parameters
        ----------
        **conditions
            Field name to required value, e.g. ``select(space="C4")``. A band field
            also matches the band's name, as in ``select(band="alpha")``.

        Returns
        -------
        FeatureTable
            A new table holding only the matching columns. A value that no column
            carries at all warns, since that is usually a misspelled label rather
            than a feature this recording lacks.
        """
        known = {f.name for f in fields(FeatureMeta)}
        unknown = set(conditions) - known
        if unknown:
            raise ValueError(
                f"{sorted(unknown)} is not a FeatureMeta field; known fields: {sorted(known)}"
            )
        for key, value in conditions.items():
            present = [getattr(m, key) for m in self.meta]
            if present and not any(_matches(field, value) for field in present):
                labels = sorted({str(_label(field)) for field in present})
                warnings.warn(
                    f"no column has {key}={value!r}; this table's {key} values are {labels}.",
                    UserWarning,
                    stacklevel=2,
                )
        keep = [
            i
            for i, m in enumerate(self.meta)
            if all(_matches(getattr(m, key), value) for key, value in conditions.items())
        ]
        return self._columns(np.asarray(keep, dtype=int))

    def drop_missing(self, max_fraction: float) -> FeatureTable:
        """Return the columns missing in at most ``max_fraction`` of the rows.

        The fraction ignores any target, so a cohort can be thinned before cross-validation;
        each fold's own missingness limit still decides what its model sees.
        """
        if not 0.0 <= max_fraction <= 1.0:
            raise ValueError(f"max_fraction must be between 0 and 1, got {max_fraction}.")
        missing = (
            (~np.isfinite(self.values)).mean(axis=0)
            if self.n_rows
            else np.zeros(self.values.shape[1])
        )
        return self._columns(np.flatnonzero(missing <= max_fraction))

    def take(
        self, rows: Sequence[int] | npt.NDArray[np.bool_] | npt.NDArray[np.intp]
    ) -> FeatureTable:
        """Return the given rows, in the given order, with their identities and flags.

        Parameters
        ----------
        rows : sequence of int, or ndarray of bool
            Row positions, or a mask with one entry per row. A row may be taken once.
        """
        index = np.asarray(rows)
        if index.dtype == np.bool_:
            if index.shape != (self.n_rows,):
                raise ValueError(f"a row mask needs {self.n_rows} entries, got {index.shape}.")
            index = np.flatnonzero(index)
        if index.size == 0:
            index = index.astype(np.intp)
        if index.ndim != 1 or not np.issubdtype(index.dtype, np.integer):
            raise ValueError("rows must be integer row positions or a boolean row mask.")
        if np.any(index < 0) or np.any(index >= self.n_rows):
            raise ValueError(f"row positions must lie in [0, {self.n_rows}).")
        # Each row is an identity, so taking one twice would give a join two targets for it.
        if np.unique(index).size != index.size:
            raise ValueError("take would repeat a row; each row may be taken once.")
        return FeatureTable(
            values=self.values[index],
            coverage=self.coverage[index],
            meta=self.meta,
            flags={key: flag[index] for key, flag in self.flags.items()},
            row_labels=(
                None if self.row_labels is None else tuple(self.row_labels[i] for i in index)
            ),
            row_ids=None if self.row_ids is None else tuple(self.row_ids[i] for i in index),
            support=None if self.support is None else self.support[index],
        )

    def _columns(self, index: npt.NDArray[np.intp]) -> FeatureTable:
        return FeatureTable(
            values=self.values[:, index],
            coverage=self.coverage[:, index],
            meta=tuple(self.meta[i] for i in index),
            flags={k: v[:, index] for k, v in self.flags.items()},
            row_labels=self.row_labels,
            row_ids=self.row_ids,
            support=None if self.support is None else self.support[:, index],
        )


def concat(tables: Sequence[FeatureTable]) -> FeatureTable:
    """Join feature tables column-wise.

    Every table must have the same number of rows and the same row semantics.
    Rows are never aligned or reindexed; a mismatch is an error rather than
    something to repair.

    Parameters
    ----------
    tables : sequence of FeatureTable
        Tables to join, in order.

    Returns
    -------
    FeatureTable
        One table holding every column. A flag present in some inputs and
        absent in others is False where it was absent.
    """
    if not tables:
        raise ValueError("concat requires at least one table.")
    n_rows = tables[0].n_rows
    row_labels = tables[0].row_labels
    row_ids = tables[0].row_ids
    for table in tables:
        # Row semantics first: it explains a count mismatch too, and is the more
        # useful error when a cross-trial estimate meets a per-epoch one.
        if table.row_labels != row_labels:
            raise ValueError(
                "concat requires matching row semantics: one table has rows "
                f"{row_labels!r} and another {table.row_labels!r}. A measure estimated "
                "across trials cannot be joined to one estimated per epoch without "
                "deciding how to broadcast it."
            )
        if row_labels is None and table.row_ids != row_ids:
            raise ValueError(
                "concat requires exact row identities in the same order; the recording, "
                "epoch or event identity differs."
            )
        if table.n_rows != n_rows:
            raise ValueError(f"concat requires matching n_rows; got {n_rows} and {table.n_rows}.")
    if row_labels is None and row_ids is None:
        raise ValueError(
            "concat requires row_ids for per-epoch tables; row counts alone cannot prove "
            "that recordings, epochs and events are aligned."
        )
    flag_keys = sorted({key for table in tables for key in table.flags})
    flags = {
        key: np.concatenate(
            [table.flags.get(key, np.zeros(table.values.shape, dtype=bool)) for table in tables],
            axis=1,
        )
        for key in flag_keys
    }
    return FeatureTable(
        values=np.concatenate([t.values for t in tables], axis=1),
        coverage=np.concatenate([t.coverage for t in tables], axis=1),
        meta=tuple(m for t in tables for m in t.meta),
        flags=flags,
        row_labels=row_labels,
        row_ids=row_ids,
        support=_joined_support(tables, axis=1),
    )


def stack_rows(
    tables: Sequence[FeatureTable],
    *,
    columns: Literal["identical", "union"] = "identical",
) -> FeatureTable:
    """Stack per-epoch feature tables in input order.

    This is the cohort-building counterpart to :func:`concat`, which joins
    feature columns for the same epochs. Cross-trial tables are deliberately
    excluded because their group rows are not independent epochs.

    Parameters
    ----------
    tables : sequence of FeatureTable
        Per-epoch tables to stack, in order.
    columns : {"identical", "union"}
        How to reconcile feature columns. ``"identical"``, the default, requires
        every table to carry the same ordered metadata. ``"union"`` keeps every
        column any table measured, in first-seen order, and marks a column a
        recording did not measure as NaN with zero coverage. Recordings in a
        cohort differ in their bad channels, so their schemas differ; the union
        is the cohort matrix the fold-local harmonization in
        :mod:`eegtable.model` is defined over.

    Returns
    -------
    FeatureTable
        One table holding every row. A flag present in some inputs and absent in
        others is False where it was absent.
    """
    if columns not in ("identical", "union"):
        raise ValueError(f'columns must be "identical" or "union"; got {columns!r}.')
    if not tables:
        raise ValueError("stack_rows requires at least one table.")

    meta = tables[0].meta
    identity_groups: list[tuple[RowId, ...]] = []
    for position, table in enumerate(tables):
        if table.row_labels is not None or table.row_ids is None:
            raise ValueError("stack_rows accepts per-epoch tables with row_ids only.")
        if columns == "identical" and table.meta != meta:
            raise ValueError(
                "stack_rows requires the same ordered feature metadata; "
                f"{_schema_difference(tables[0], table, position)}"
            )
        identity_groups.append(table.row_ids)

    row_ids = tuple(row_id for identities in identity_groups for row_id in identities)
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("stack_rows found duplicate row_ids across input tables.")

    if columns == "union":
        return _stack_rows_union(tables, row_ids)

    flag_names = sorted({name for table in tables for name in table.flags})
    flags = {
        name: np.concatenate(
            [table.flags.get(name, np.zeros(table.values.shape, dtype=bool)) for table in tables],
            axis=0,
        )
        for name in flag_names
    }
    return FeatureTable(
        values=np.concatenate([table.values for table in tables], axis=0),
        coverage=np.concatenate([table.coverage for table in tables], axis=0),
        meta=meta,
        flags=flags,
        row_ids=row_ids,
        support=_joined_support(tables, axis=0),
    )


def _joined_support(tables: Sequence[FeatureTable], *, axis: int) -> npt.NDArray[np.float64] | None:
    # A table without support rests on its whole windows, so it joins as complete.
    if all(table.support is None for table in tables):
        return None
    return np.concatenate(
        [
            table.support if table.support is not None else np.ones(table.values.shape)
            for table in tables
        ],
        axis=axis,
    )


def _schema_difference(first: FeatureTable, table: FeatureTable, position: int) -> str:
    recording = table.row_ids[0][0] if table.row_ids else "no rows"
    reference, names = first.names, table.names
    extra = [name for name in names if name not in set(reference)]
    missing = [name for name in reference if name not in set(names)]
    if not extra and not missing:
        return f"table {position} ({recording}) has the first table's columns in another order."

    def count(names: list[str]) -> str:
        return f"{len(names)} column{'' if len(names) == 1 else 's'} ({', '.join(names[:3])})"

    parts = []
    if extra:
        parts.append(f"has {count(extra)} the first table lacks")
    if missing:
        parts.append(f"lacks {count(missing)} of the first table's")
    # A fit of the recording's own, such as its microstate templates, enters the column
    # names, so those columns can never match another recording's.
    return (
        f"table {position} ({recording}) {' and '.join(parts)}. Recordings that dropped "
        "different channels, or fitted something of their own such as microstate templates, "
        "differ in their columns: stack with columns='union', or leave the fitted measure out."
    )


def _stack_rows_union(tables: Sequence[FeatureTable], row_ids: tuple[RowId, ...]) -> FeatureTable:
    return _stack_union(tables, row_ids=row_ids)


def _stack_union(
    tables: Sequence[FeatureTable],
    *,
    row_ids: tuple[RowId, ...] | None = None,
    row_labels: tuple[str, ...] | None = None,
) -> FeatureTable:
    """Stack tables onto the union of their columns, NaN where a table lacks one."""
    # A column's identity is its whole FeatureMeta, which FeatureTable already
    # requires to be unique within a table, so first-seen order places each one.
    positions: dict[FeatureMeta, int] = {}
    placements = [
        np.array(
            [positions.setdefault(record, len(positions)) for record in table.meta],
            dtype=np.intp,
        )
        for table in tables
    ]

    n_rows, n_columns = sum(table.n_rows for table in tables), len(positions)
    values = np.full((n_rows, n_columns), np.nan)
    # A column a recording never measured had no finite input, so its coverage is 0.
    coverage = np.zeros((n_rows, n_columns))
    flag_names = sorted({name for table in tables for name in table.flags})
    flags = {name: np.zeros((n_rows, n_columns), dtype=bool) for name in flag_names}
    # Like coverage, a column a recording never measured rests on none of its window.
    restricted = any(table.support is not None for table in tables)
    support = np.zeros((n_rows, n_columns)) if restricted else None

    start = 0
    for table, placement in zip(tables, placements, strict=True):
        rows = slice(start, start + table.n_rows)
        values[rows, placement] = table.values
        coverage[rows, placement] = table.coverage
        for name, array in table.flags.items():
            flags[name][rows, placement] = array
        if support is not None:
            support[rows, placement] = 1.0 if table.support is None else table.support
        start += table.n_rows

    return FeatureTable(
        values=values,
        coverage=coverage,
        meta=tuple(positions),
        flags=flags,
        row_ids=row_ids,
        row_labels=row_labels,
        support=support,
    )
