"""Read and write a :class:`~eegtable.FeatureTable` without losing its metadata.

A table is stored as three files beside one another:

- ``<name>.tsv``, the values, one row per epoch or trial group, NaN as ``n/a``;
- ``<name>_coverage.tsv``, the coverage matrix in the same layout;
- ``<name>_support.tsv``, only for tables whose values rest on part of their
  windows (Morlet power), the fraction of each window each value rests on;
- ``<name>.json``, a sidecar holding the :class:`~eegtable.FeatureMeta` of every
  column, the row semantics and any per-cell flags.

The TSV files open in anything that reads tabular data. The sidecar is what lets
:func:`read_table` rebuild the table exactly, so columns can still be selected by
band, space or window rather than by parsing their names.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from eegtable._validation import validate_names
from eegtable.bands import Band
from eegtable.provenance import file_hash
from eegtable.table import (
    ComputationSpec,
    FeatureMeta,
    FeatureTable,
    _json_value,
    stack_rows,
)

_SCHEMA = 3
"""Schema 3 shares computations across columns; schema 2, still read, wrote each in full."""

_NA = "n/a"
"""Missing-value marker, following the BIDS convention for tabular files."""

_EPOCH_KEY = "epoch"
_GROUP_KEY = "group"
_ROW_UID = "__eegtable_row_id"
# Tables written before the rename carry the old column name; they stay readable.
_LEGACY_ROW_UID = "__eegfeat_row_id"
# The descriptors eegfeat's reader kept as text, whatever they looked like.
_LEGACY_TEXT_COLUMNS = ("recording", "event")


def _row_uids(
    row_ids: Sequence[Sequence[object]] | None,
    row_labels: Sequence[str] | None,
    n_rows: int,
) -> list[str]:
    if row_ids is not None:
        return [
            json.dumps(
                [str(recording), int(cast(Any, epoch)), str(event)],
                separators=(",", ":"),
            )
            for recording, epoch, event in row_ids
        ]

    if row_labels is not None:
        return [json.dumps(["group", str(label)], separators=(",", ":")) for label in row_labels]

    return [json.dumps(["epoch", i], separators=(",", ":")) for i in range(n_rows)]


@dataclass(frozen=True)
class FeatureDataset:
    """Per-epoch features and their aligned target descriptors."""

    table: FeatureTable
    targets: pd.DataFrame


def write_table(
    table: FeatureTable,
    path: str | os.PathLike[str],
    *,
    rows: pd.DataFrame | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    """Write a feature table as TSV files and a JSON sidecar.

    Parameters
    ----------
    table : FeatureTable
        The table to write.
    path : path-like
        Destination of the values file. Must end in ``.tsv``; the coverage file
        and sidecar are named after it.
    rows : DataFrame, optional
        Descriptive columns written between the row key and the features, one
        row per table row, e.g. the event and metadata of each epoch. They are
        restored as aligned targets by :func:`read_dataset`, but are not feature
        columns returned by :func:`read_table`. The output is BIDS-style tabular
        data, not a validated BIDS derivative dataset. Literal ``n/a`` labels are
        refused because that token is reserved for missing values.
    provenance : mapping, optional
        JSON-serializable record of how the table was produced, stored in the
        sidecar as given.

    Returns
    -------
    tuple of Path
        The values file, the coverage file and the sidecar, in that order.
    """
    target = Path(path)
    if target.suffix != ".tsv":
        raise ValueError(f"write_table writes a .tsv file; got {target.name!r}.")

    key, key_values = _row_key(table)
    descriptors = _descriptors(rows, table, key)
    row_frame = pd.concat([pd.DataFrame({key: key_values}), descriptors], axis=1)
    for column in row_frame:
        if row_frame[column].eq(_NA).any():
            raise ValueError(
                f"Row descriptor {column!r} contains literal {_NA!r}, reserved for missing values."
            )
    names = table.names
    uid_frame = pd.DataFrame({_ROW_UID: _row_uids(table.row_ids, table.row_labels, table.n_rows)})

    values_frame = pd.concat(
        [
            row_frame,
            uid_frame,
            pd.DataFrame(table.values, columns=names),
        ],
        axis=1,
    )

    def matrix_frame(matrix: npt.NDArray[np.float64]) -> pd.DataFrame:
        return pd.concat(
            [pd.DataFrame({key: key_values}), uid_frame, pd.DataFrame(matrix, columns=names)],
            axis=1,
        )

    coverage_path = target.with_name(f"{target.stem}_coverage.tsv")
    support_path = target.with_name(f"{target.stem}_support.tsv")
    sidecar_path = target.with_suffix(".json")
    sidecar = _sidecar(table, key, list(descriptors.columns), coverage_path.name, provenance)
    if table.support is not None:
        sidecar["support"] = support_path.name
    sidecar["row_text_columns"] = [
        column
        for column in [key, *descriptors.columns]
        if column in {_GROUP_KEY, "recording", "event"}
        or (
            column in descriptors
            and (
                pd.api.types.is_string_dtype(descriptors[column].dtype)
                or isinstance(descriptors[column].dtype, pd.CategoricalDtype)
            )
        )
    ]

    payloads = [(target, values_frame), (coverage_path, matrix_frame(table.coverage))]
    if table.support is not None:
        payloads.append((support_path, matrix_frame(table.support)))
    with TemporaryDirectory(prefix=".eegtable-", dir=target.parent) as temporary:
        staging = Path(temporary)
        staged = [staging / path.name for path, _ in payloads]
        for (_, frame), path in zip(payloads, staged, strict=True):
            _write_tsv(frame, path)
        sidecar["schema"] = _SCHEMA
        sidecar["files"] = {path.name: file_hash(path) for path in staged}
        staged_sidecar = staging / sidecar_path.name
        _write_text(json.dumps(sidecar, indent=2, allow_nan=False) + "\n", staged_sidecar)
        _publish_bundle(
            (*staged, staged_sidecar),
            (*(path for path, _ in payloads), sidecar_path),
            staging / "backup",
            # A support file left from an earlier bundle would describe values it never saw.
            remove=() if table.support is not None else (support_path,),
        )
    return target, coverage_path, sidecar_path


def read_table(path: str | os.PathLike[str]) -> FeatureTable:
    """Read a feature table written by :func:`write_table`.

    Parameters
    ----------
    path : path-like
        The values ``.tsv`` file. Its sidecar and coverage file are found beside it.

    Returns
    -------
    FeatureTable
        Values, coverage, metadata, flags and row labels as they were written.
    """
    source = Path(path)
    return _table_from_sidecar(source, _read_sidecar(source))


def _table_from_sidecar(source: Path, sidecar: Mapping[str, Any]) -> FeatureTable:
    shared = sidecar.get("computations")
    specs: dict[str, ComputationSpec] = {}
    meta = tuple(_meta_from_record(record, shared, specs) for record in sidecar["columns"])
    names = [m.name for m in meta]

    if "n_rows" not in sidecar:
        raise ValueError("Legacy feature bundle has no row-identity manifest; regenerate it.")

    expected_uids = _row_uids(
        sidecar["row_ids"],
        sidecar["row_labels"],
        int(sidecar["n_rows"]),
    )

    values = _read_matrix(source, names, expected_uids)
    coverage = _read_matrix(
        source.with_name(sidecar["coverage"]),
        names,
        expected_uids,
    )
    support = (
        _read_matrix(source.with_name(sidecar["support"]), names, expected_uids)
        if sidecar.get("support")
        else None
    )
    column_index = {name: i for i, name in enumerate(names)}
    flags: dict[str, npt.NDArray[np.bool_]] = {}
    for key, cells in sidecar["flags"].items():
        flag = np.zeros(values.shape, dtype=bool)
        for name, row_indices in cells.items():
            if not isinstance(row_indices, list) or any(
                type(row) is not int or not 0 <= row < values.shape[0] for row in row_indices
            ):
                raise ValueError(
                    f"sidecar flag {key!r} for {name!r} requires integer row positions "
                    f"in [0, {values.shape[0]})."
                )
            flag[np.asarray(row_indices, dtype=int), column_index[name]] = True
        flags[key] = flag

    labels = sidecar["row_labels"]
    identifiers = sidecar["row_ids"]
    return FeatureTable(
        values=values,
        coverage=coverage,
        meta=meta,
        flags=flags,
        row_labels=None if labels is None else tuple(str(label) for label in labels),
        row_ids=(
            None
            if identifiers is None
            else tuple(
                (str(recording), int(epoch), str(event)) for recording, epoch, event in identifiers
            )
        ),
        support=support,
    )


def read_dataset(
    paths: Sequence[str | os.PathLike[str]],
    *,
    columns: Literal["identical", "union"] = "union",
) -> FeatureDataset:
    """Load and vertically combine runner-generated per-epoch feature bundles.

    Parameters
    ----------
    paths : sequence of path-like
        The ``*_features.tsv`` files to combine.
    columns : {"union", "identical"}
        How :func:`eegtable.stack_rows` reconciles the feature columns. The
        default keeps every column any recording measured; pass ``"identical"``
        to require one schema across the cohort.
    """
    if not paths:
        raise ValueError("read_dataset requires at least one feature table path.")

    tables: list[FeatureTable] = []
    target_frames: list[pd.DataFrame] = []
    for path in paths:
        source = Path(path)
        sidecar = _read_sidecar(source)
        table = _table_from_sidecar(source, sidecar)
        if table.row_ids is None:
            raise ValueError("read_dataset accepts per-epoch feature tables only.")
        tables.append(table)
        target_frames.append(_read_targets(source, table, sidecar))

    # Recordings differ in their bad channels, so their feature schemas differ; the
    # cohort is the union, with NaN where a recording did not measure a column.
    return FeatureDataset(
        table=stack_rows(tables, columns=columns),
        targets=pd.concat(target_frames, ignore_index=True),
    )


def _read_descriptor_frame(source: Path, sidecar: Mapping[str, Any]) -> pd.DataFrame:
    columns = sidecar["row_columns"]
    validate_names(columns, "descriptor columns")
    return pd.read_csv(
        source,
        sep="\t",
        na_values=[_NA],
        keep_default_na=False,
        dtype={column: str for column in sidecar["row_text_columns"]},
        float_precision="round_trip",
        usecols=lambda column: column in columns,
    )


def _read_targets(
    source: Path,
    table: FeatureTable,
    sidecar: Mapping[str, Any],
) -> pd.DataFrame:
    row_columns: list[str] = sidecar["row_columns"]
    validate_names(row_columns, "descriptor columns")
    if not row_columns or row_columns[0] != _EPOCH_KEY:
        raise ValueError(f"{source.name} sidecar does not declare an epoch row key.")
    descriptor_columns = row_columns[1:]
    frame = _read_descriptor_frame(source, sidecar)
    if _EPOCH_KEY not in frame.columns:
        raise ValueError(f"{source.name} lacks the epoch row key its sidecar describes.")
    missing_descriptors = [column for column in descriptor_columns if column not in frame.columns]
    if missing_descriptors:
        raise ValueError(
            f"{source.name} lacks descriptor columns its sidecar describes: {missing_descriptors}"
        )

    identifiers = pd.DataFrame(table.row_ids, columns=["recording", "epoch", "event"])
    serialized_epochs = pd.to_numeric(frame[_EPOCH_KEY], errors="raise").to_numpy()
    epochs_are_integral = np.equal(serialized_epochs, np.floor(serialized_epochs))
    canonical_epochs = identifiers[_EPOCH_KEY].to_numpy()
    if not np.all(epochs_are_integral) or not np.array_equal(serialized_epochs, canonical_epochs):
        raise ValueError(f"{source.name} epoch row key disagrees with canonical row_ids.")

    descriptors = frame[descriptor_columns].copy()
    for column in identifiers.columns.intersection(descriptors.columns):
        actual = descriptors[column].to_numpy()
        expected = identifiers[column].to_numpy()
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"{source.name} descriptor {column!r} disagrees with canonical row_ids."
            )
    descriptors = descriptors.drop(columns=identifiers.columns, errors="ignore")
    return pd.concat([identifiers, descriptors], axis=1)


def _legacy_sidecar(source: Path, sidecar: dict[str, Any]) -> dict[str, Any]:
    # A bundle written by eegfeat, before the schema and its manifests: the same values and
    # coverage files, which stay readable so that results computed from them can be reproduced.
    # Its reader took "recording" and "event" as text and parsed everything else; it had no
    # checksums to verify.
    path = source.with_suffix(".json")
    coverage = source.stem + "_coverage.tsv"
    if sidecar.get("coverage") != coverage or not (source.parent / coverage).is_file():
        raise ValueError(f"{path}: invalid legacy feature bundle file manifest.")
    row_columns = sidecar.get("row_columns")
    if not isinstance(row_columns, list):
        raise ValueError(f"{path}: legacy feature bundle declares no descriptor columns.")
    validate_names(row_columns, "descriptor columns")
    text_columns = [column for column in row_columns if column in _LEGACY_TEXT_COLUMNS]
    return {**sidecar, "row_text_columns": text_columns}


def _read_sidecar(source: Path) -> dict[str, Any]:
    path = source.with_suffix(".json")
    try:
        sidecar = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        # macOS writes a "._" twin beside every file on an exFAT drive, and globs find them.
        hint = (
            " It is a macOS AppleDouble resource file; leave out names starting with '._'."
            if path.name.startswith("._")
            else ""
        )
        raise ValueError(f"{path} is not a feature sidecar ({exc}).{hint}") from exc
    if isinstance(sidecar, dict) and "schema" not in sidecar and "eegfeat_version" in sidecar:
        return _legacy_sidecar(source, sidecar)
    if not isinstance(sidecar, dict) or sidecar.get("schema") not in (2, _SCHEMA):
        raise ValueError(f"{path}: unsupported feature bundle schema; regenerate the bundle.")
    text_columns = sidecar.get("row_text_columns")
    if not isinstance(text_columns, list):
        raise ValueError(f"{path}: missing descriptor type manifest; regenerate the bundle.")
    validate_names(text_columns, "text descriptor columns")
    if not set(text_columns).issubset(sidecar["row_columns"]):
        raise ValueError(f"{path}: text descriptor columns are not declared row columns.")
    coverage = sidecar["coverage"]
    support = sidecar.get("support")
    expected = {source.name, source.stem + "_coverage.tsv"}
    if support is not None:
        expected.add(source.stem + "_support.tsv")
    if (
        coverage != source.stem + "_coverage.tsv"
        or support not in (None, source.stem + "_support.tsv")
        or set(sidecar["files"]) != expected
    ):
        raise ValueError(f"{path}: invalid feature bundle file manifest.")
    for name, checksum in sidecar["files"].items():
        payload = source.parent / name
        if not payload.is_file() or file_hash(payload) != checksum:
            raise ValueError(f"{payload}: feature bundle checksum mismatch.")
    return sidecar


def _row_key(table: FeatureTable) -> tuple[str, list[int] | list[str]]:
    if table.row_labels is None:
        if table.row_ids is None:
            return _EPOCH_KEY, list(range(table.n_rows))
        return _EPOCH_KEY, [epoch for _, epoch, _ in table.row_ids]
    return _GROUP_KEY, list(table.row_labels)


def _descriptors(rows: pd.DataFrame | None, table: FeatureTable, key: str) -> pd.DataFrame:
    if rows is None:
        return pd.DataFrame(index=range(table.n_rows))
    if len(rows) != table.n_rows:
        raise ValueError(f"rows has {len(rows)} entries but the table has {table.n_rows} rows.")
    validate_names(list(rows.columns), "descriptor columns")
    reserved = {key, _ROW_UID, *table.names}
    clashes = sorted(str(column) for column in rows.columns if column in reserved)
    if clashes:
        raise ValueError(
            f"descriptor columns {clashes} would shadow the row key {key!r} or a feature."
        )
    return rows.reset_index(drop=True)


def _sidecar(
    table: FeatureTable,
    key: str,
    descriptor_columns: list[str],
    coverage_name: str,
    provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from eegtable import __version__

    names = table.names
    computations: dict[str, dict[str, object]] = {}
    flags = {
        flag: {
            names[column]: np.flatnonzero(cells[:, column]).tolist()
            for column in range(cells.shape[1])
            if cells[:, column].any()
        }
        for flag, cells in table.flags.items()
    }
    sidecar: dict[str, Any] = {
        "eegtable_version": __version__,
        "n_rows": table.n_rows,
        "rows": "epochs" if table.row_labels is None else "groups",
        "row_labels": None if table.row_labels is None else list(table.row_labels),
        "row_ids": None if table.row_ids is None else list(table.row_ids),
        "row_columns": [key, *descriptor_columns],
        "coverage": coverage_name,
        "columns": [_column_record(m, computations) for m in table.meta],
        "computations": computations,
        "flags": flags,
    }
    if provenance is not None:
        sidecar["provenance"] = dict(provenance)
    return sidecar


def _column_record(meta: FeatureMeta, computations: dict[str, dict[str, object]]) -> dict[str, Any]:
    # The record the column's name is hashed from, so no field can be written differently.
    # Its computation goes into a table shared by the bundle: many columns, every pair of a
    # connectivity matrix, carry the same one, channel list and all.
    key = meta.computation.parameter_hash
    if key not in computations:
        computations[key] = meta.computation.record()
    fields = cast(dict[str, Any], _json_value(meta.fields_record()))
    return {"name": meta.name, **fields, "computation": key, "parameter_hash": meta.parameter_hash}


def _computation(
    record: Mapping[str, Any],
    shared: Mapping[str, Any] | None,
    specs: dict[str, ComputationSpec],
) -> ComputationSpec:
    written = record["computation"]
    if not isinstance(written, str):
        # Schema 2 and eegfeat bundles wrote each column's computation in full.
        return ComputationSpec.create(written["method"], **dict(written["parameters"]))
    if written not in specs:
        if shared is None or written not in shared:
            raise ValueError(
                f"sidecar column {record['name']!r} names a computation the sidecar lacks."
            )
        # Not checked against its key here: every column's own hash covers its computation,
        # so a swapped or altered entry already fails that check.
        entry = shared[written]
        specs[written] = ComputationSpec.create(entry["method"], **dict(entry["parameters"]))
    return specs[written]


def _meta_from_record(
    record: Mapping[str, Any],
    shared: Mapping[str, Any] | None = None,
    specs: dict[str, ComputationSpec] | None = None,
) -> FeatureMeta:
    computation = _computation(record, shared, {} if specs is None else specs)
    bounds = record["window_bounds"]
    phase_band = record.get("phase_band")
    amplitude_band = record.get("amplitude_band")
    nodes = record.get("nodes")
    meta = FeatureMeta(
        measure=record["measure"],
        band=_band_from_record(record["band"]),
        space=record["space"],
        space_kind=record["space_kind"],
        window=record["window"],
        window_bounds=None if bounds is None else (float(bounds[0]), float(bounds[1])),
        normalization=record["normalization"],
        unit=record["unit"],
        source=record["source"],
        computation=computation,
        freq_resolution_hz=record["freq_resolution_hz"],
        phase_band=_band_from_record(phase_band),
        amplitude_band=_band_from_record(amplitude_band),
        nodes=None if nodes is None else (str(nodes[0]), str(nodes[1])),
    )
    if meta.name != record["name"]:
        raise ValueError(
            f"sidecar column {record['name']!r} does not match its own metadata, "
            f"which names it {meta.name!r}."
        )
    if meta.parameter_hash != record["parameter_hash"]:
        raise ValueError(f"sidecar column {record['name']!r} has an invalid parameter hash.")
    return meta


def _band_from_record(record: Mapping[str, Any] | None) -> Band | None:
    if record is None:
        return None
    return Band(str(record["name"]), float(record["fmin"]), float(record["fmax"]))


def _read_matrix(
    path: Path,
    names: list[str],
    expected_uids: Sequence[str],
) -> npt.NDArray[np.float64]:
    frame = pd.read_csv(
        path,
        sep="\t",
        na_values=[_NA],
        keep_default_na=False,
        float_precision="round_trip",
    )

    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} lacks columns its sidecar describes: {missing}")

    uid_column = _ROW_UID if _ROW_UID in frame.columns else _LEGACY_ROW_UID
    if uid_column not in frame.columns:
        raise ValueError(
            f"{path.name} has no {_ROW_UID} column; regenerate this legacy feature bundle."
        )

    actual_uids = frame[uid_column].astype(str).tolist()
    if actual_uids != list(expected_uids):
        raise ValueError(
            f"{path.name}: row identities or row order disagree with the JSON sidecar."
        )

    return np.asarray(
        frame[names].to_numpy(dtype=float),
        dtype=np.float64,
    )


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    frame.to_csv(partial, sep="\t", index=False, na_rep=_NA)
    os.replace(partial, path)


def _write_text(text: str, path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def _publish_bundle(
    staged: tuple[Path, ...],
    final: tuple[Path, ...],
    backup: Path,
    *,
    remove: tuple[Path, ...] = (),
) -> None:
    backup.mkdir()
    saved: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for target in (*final, *remove):
            if target.exists():
                copy = backup / target.name
                os.replace(target, copy)
                saved.append((copy, target))
        for source, target in zip(staged, final, strict=True):
            os.replace(source, target)
            published.append(target)
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        for source, target in saved:
            os.replace(source, target)
        raise
