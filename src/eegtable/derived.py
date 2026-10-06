from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from eegtable.table import ComputationSpec, FeatureMeta, FeatureTable, Normalization

_LOG_UNITS: dict[Normalization, str] = {
    "log10": "log10 ratio",
    "log_ratio": "log10 ratio",
    "db": "dB",
}
"""Unit of a difference taken on each logarithmic scale.

Subtracting two values already in dB leaves a dB difference, not a bare log
ratio: the factor of ten is still in there.
"""

_LOGARITHMIC = tuple(_LOG_UNITS)

_SIGNED_CHANGE: tuple[Normalization, ...] = ("percent",)
"""Scales whose values cross zero, so neither a quotient nor a normalized difference holds.

A percent change is a signed deviation from a baseline, not an amount of power.
Dividing two of them gives a negative "ratio" wherever one band went down, and
``(right - left) / (right + left)`` leaves its own ``[-1, 1]`` bound wherever the
two nearly cancel. Both read as measurements and are neither, so they are refused
rather than emitted with a caveat.
"""


def _check_combinable(table: FeatureTable, index: int, operation: str) -> None:
    normalization = table.meta[index].normalization
    if normalization in _SIGNED_CHANGE:
        raise ValueError(
            f"{operation} is not defined on {normalization!r} values: they are signed "
            "changes from a baseline, so a quotient can come out negative and a "
            "normalized difference can leave [-1, 1]. Derive from 'raw' power, or from "
            "a logarithmic scale where the operation becomes a difference."
        )


def band_ratio(table: FeatureTable, numerator: str, denominator: str) -> FeatureTable:
    """Ratio between two bands, computed per spatial unit and window.

    When the input is already logarithmic the ratio is a difference, and this
    subtracts rather than divides.

    Parameters
    ----------
    table : FeatureTable
        Band power, with both bands present for every spatial unit and window.
    numerator, denominator : str
        Band names.

    Returns
    -------
    FeatureTable
        One column per spatial unit and window, measure
        ``"ratio_{numerator}_{denominator}"``.
    """
    top = _index_by_position(table, numerator)
    bottom = _index_by_position(table, denominator)
    for index in (*top.values(), *bottom.values()):
        _check_combinable(table, index, "band_ratio")
    missing = set(top) ^ set(bottom)
    if missing:
        raise ValueError(
            f"bands {numerator!r} and {denominator!r} do not cover the same spatial units "
            f"and windows; unmatched: {missing}"
        )

    operands = [(top[k], bottom[k]) for k in top]
    measure = f"ratio_{numerator}_{denominator}"
    meta = tuple(
        replace(
            table.meta[i],
            measure=measure,
            band=None,
            unit=_LOG_UNITS.get(table.meta[i].normalization, "ratio"),
            computation=_derived_spec(
                measure, table, (i, "numerator"), (j, "denominator"), ratio=True
            ),
        )
        for i, j in operands
    )
    return FeatureTable(
        values=np.stack([_combine(table, i, j, ratio=True) for i, j in operands], axis=1),
        coverage=np.stack(
            [np.minimum(table.coverage[:, i], table.coverage[:, j]) for i, j in operands], axis=1
        ),
        meta=meta,
        flags=_merge_flags(table, operands),
        row_labels=table.row_labels,
        row_ids=table.row_ids,
        support=_paired_support(table, operands),
    )


def asymmetry(table: FeatureTable, pairs: Sequence[tuple[str, str]]) -> FeatureTable:
    """Hemispheric asymmetry between channel pairs.

    For raw power this is ``(right - left) / (right + left)``. For logarithmic
    input the normalized difference is already a plain difference, so it is
    ``right - left``.

    Parameters
    ----------
    table : FeatureTable
        Band power containing both members of every pair.
    pairs : sequence of (str, str)
        ``(left, right)`` channel names.

    Returns
    -------
    FeatureTable
        One column per pair, band and window, with ``space_kind="pair"``.
    """
    operands: list[tuple[int, int]] = []
    meta: list[FeatureMeta] = []
    for left, right in pairs:
        left_index = _index_by_space(table, left)
        right_index = _index_by_space(table, right)
        if set(left_index) != set(right_index):
            raise ValueError(
                f"Channels {left!r} and {right!r} do not have "
                "matching measurement specifications."
            )

        for key, on_left in left_index.items():
            on_right = right_index[key]
            _check_combinable(table, on_left, "asymmetry")
            _check_combinable(table, on_right, "asymmetry")
            operands.append((on_right, on_left))
            meta.append(
                replace(
                    table.meta[on_left],
                    measure="asymmetry",
                    space=f"{left}-{right}",
                    space_kind="pair",
                    unit=_LOG_UNITS.get(table.meta[on_left].normalization, "a.u."),
                    computation=_derived_spec(
                        "asymmetry",
                        table,
                        (on_right, "right"),
                        (on_left, "left"),
                        ratio=False,
                    ),
                )
            )
    if not operands:
        raise ValueError("no pair matched a band and window present in the table.")
    return FeatureTable(
        values=np.stack([_combine(table, i, j, ratio=False) for i, j in operands], axis=1),
        coverage=np.stack(
            [np.minimum(table.coverage[:, i], table.coverage[:, j]) for i, j in operands], axis=1
        ),
        meta=tuple(meta),
        flags=_merge_flags(table, operands),
        row_labels=table.row_labels,
        row_ids=table.row_ids,
        support=_paired_support(table, operands),
    )


def _derived_spec(
    measure: str,
    table: FeatureTable,
    first: tuple[int, str],
    second: tuple[int, str],
    *,
    ratio: bool,
) -> ComputationSpec:
    """Describe the derivation itself, not the measurement it started from.

    Inheriting the input's spec would name the numerator's algorithm alone, so a
    theta/beta ratio and a theta/alpha ratio would hash identically. Each operand
    is recorded whole: its band bounds, scale, window and own computation.
    """
    first_index, first_role = first
    second_index, second_role = second
    if table.meta[first_index].normalization in _LOGARITHMIC:
        operation = "difference"
    else:
        operation = "quotient" if ratio else "normalized_difference"
    return ComputationSpec.create(
        measure,
        operation=operation,
        **{
            first_role: table.meta[first_index].record(),
            second_role: table.meta[second_index].record(),
        },
    )


def _paired_support(
    table: FeatureTable, operands: Sequence[tuple[int, int]]
) -> npt.NDArray[np.float64] | None:
    # As with coverage, a value combining two columns rests on the scarcer of the two.
    if table.support is None:
        return None
    support = table.support
    return np.stack([np.minimum(support[:, i], support[:, j]) for i, j in operands], axis=1)


def _merge_flags(
    table: FeatureTable, operands: Sequence[tuple[int, int]]
) -> Mapping[str, npt.NDArray[np.bool_]]:
    """Carry a flag on either operand onto the derived column.

    A ratio built from a flagged input is itself suspect, and dropping the flag
    would hide that.
    """
    return {
        key: np.stack([array[:, i] | array[:, j] for i, j in operands], axis=1)
        for key, array in table.flags.items()
    }


def _combine(table: FeatureTable, top: int, bottom: int, *, ratio: bool) -> npt.NDArray[np.float64]:
    a, b = table.values[:, top], table.values[:, bottom]
    if table.meta[top].normalization in _LOGARITHMIC:
        return a - b
    if ratio:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(b != 0.0, a / b, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        total = a + b
        return np.where(total != 0.0, (a - b) / total, np.nan)


def _matching_signature(meta: FeatureMeta) -> tuple[object, ...]:
    """Fields that must agree between two comparable measurements."""
    return (
        meta.measure,
        meta.space_kind,
        meta.window,
        meta.window_bounds,
        meta.normalization,
        meta.unit,
        meta.source,
        meta.computation.method,
        meta.computation.parameters_json,
        meta.phase_band,
        meta.amplitude_band,
    )


def _index_by_position(
    table: FeatureTable,
    band_name: str,
) -> dict[tuple[object, ...], int]:
    found: dict[tuple[object, ...], int] = {}

    for i, meta in enumerate(table.meta):
        if meta.band is None or meta.band.name != band_name:
            continue

        key = (meta.space, *_matching_signature(meta))
        if key in found:
            raise ValueError(
                f"Ambiguous {band_name!r} features at "
                f"{meta.space!r}, window={meta.window!r}. "
                "Select one feature family before deriving ratios."
            )
        found[key] = i

    if not found:
        raise ValueError(f"band {band_name!r} is not present in the table.")

    return found


def _index_by_space(
    table: FeatureTable,
    channel: str,
) -> dict[tuple[object, ...], int]:
    found: dict[tuple[object, ...], int] = {}

    for i, meta in enumerate(table.meta):
        if meta.space != channel or meta.space_kind != "channel":
            continue

        band_key = None if meta.band is None else (meta.band.name, meta.band.fmin, meta.band.fmax)
        key = (band_key, *_matching_signature(meta))

        if key in found:
            raise ValueError(
                f"Ambiguous measurements for channel {channel!r}, "
                f"band={band_key!r}, window={meta.window!r}."
            )

        found[key] = i

    if not found:
        raise KeyError(f"channel {channel!r} is not present in the table.")

    return found
