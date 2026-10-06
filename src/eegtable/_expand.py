from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

from eegtable._validation import blank_non_finite
from eegtable.bands import Band, check_passband
from eegtable.baseline import normalize
from eegtable.groups import SpatialUnit, aggregate
from eegtable.qc import band_coverage
from eegtable.signal import TimeSeries
from eegtable.spectra import (
    Spectra,
    Window,
    band_integration_weights,
    gradient_weights,
    require_within_axis,
    sample_period,
    trapezoid_weights,
)
from eegtable.table import ComputationSpec, FeatureMeta, FeatureTable, Normalization, RowId

Kernel = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
    tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]],
]


@dataclass(frozen=True, eq=False)
class _Column:
    meta: FeatureMeta
    values: npt.NDArray[np.float64]
    coverage: npt.NDArray[np.float64]
    support: npt.NDArray[np.float64] | None = None


def _collect(
    units: Sequence[SpatialUnit],
    windows: Sequence[Window],
    flags: Mapping[str, npt.NDArray[np.bool_]],
    make_meta: Callable[[SpatialUnit, Window], FeatureMeta],
    *,
    skip_window: int | None,
    support: npt.NDArray[np.float64] | None = None,
) -> tuple[list[_Column], dict[str, list[npt.NDArray[np.bool_]]]]:
    columns: list[_Column] = []
    flag_columns: dict[str, list[npt.NDArray[np.bool_]]] = {}
    for spatial in units:
        for w_index, window in enumerate(windows):
            if w_index == skip_window:
                continue
            columns.append(
                _Column(
                    make_meta(spatial, window),
                    spatial.values[:, w_index],
                    spatial.coverage[:, w_index],
                    (
                        None
                        if support is None
                        else support[:, list(spatial.picks), w_index].mean(axis=1)
                    ),
                )
            )
            for key, array in flags.items():
                flag_col: npt.NDArray[np.bool_] = np.asarray(
                    array[:, list(spatial.picks), w_index].any(axis=1), dtype=np.bool_
                )
                flag_columns.setdefault(key, []).append(flag_col)
    return columns, flag_columns


def _assemble(
    columns: Sequence[_Column],
    flag_columns: Mapping[str, Sequence[npt.NDArray[np.bool_]]],
    row_labels: tuple[str, ...] | None = None,
    row_ids: tuple[RowId, ...] | None = None,
) -> FeatureTable:
    restricted = any(c.support is not None for c in columns)
    return FeatureTable(
        values=np.stack([c.values for c in columns], axis=1),
        coverage=np.stack([c.coverage for c in columns], axis=1),
        meta=tuple(c.meta for c in columns),
        flags={key: np.stack(list(arrays), axis=1) for key, arrays in flag_columns.items()},
        row_labels=row_labels,
        row_ids=row_ids,
        support=(
            np.stack(
                [c.support if c.support is not None else np.ones(c.values.shape) for c in columns],
                axis=1,
            )
            if restricted
            else None
        ),
    )


def _members(ch_names: Sequence[str], spatial: SpatialUnit) -> list[str] | None:
    # A single channel is already named by its space; recording it would stop
    # channel pairs from matching for asymmetry.
    if spatial.space_kind == "channel":
        return None
    return sorted(ch_names[i] for i in spatial.picks)


def expand(
    spectra: Spectra,
    kernel: Kernel,
    *,
    measure: str,
    unit: str,
    bands: Sequence[Band] | None,
    groups: Mapping[str, Sequence[str]] | None,
    include_global: bool,
    baseline: str | None,
    mode: Normalization,
    min_bins: int,
    parameters: Mapping[str, object],
    weighting: Literal["trapezoid", "gradient", "band_integral", "uniform"] = "trapezoid",
) -> FeatureTable:
    baseline_index = _baseline_index(spectra, baseline)
    baseline_parameters = (
        {
            "baseline_bounds": (
                spectra.windows[baseline_index].tmin,
                spectra.windows[baseline_index].tmax,
            )
        }
        if baseline_index is not None
        else {}
    )
    columns: list[_Column] = []
    flag_columns: dict[str, list[npt.NDArray[np.bool_]]] = {}
    # Morlet spectra average only fully supported coefficients; when that left any window
    # short of whole, how much of it each value rests on travels with the table.
    restricted = bool(np.any(spectra.support < 1.0))

    for band in bands if bands is not None else (None,):
        if band is not None and spectra.passband is not None:
            check_passband(band, *spectra.passband, source=measure)
        integration_weights = None
        if weighting == "band_integral":
            if band is None:
                raise ValueError("band_integral weighting requires a finite frequency band.")
            integration_weights = band_integration_weights(spectra.freqs, band.fmin, band.fmax)
            mask = integration_weights > 0.0
        else:
            mask = (
                band.mask(spectra.freqs) if band is not None else np.ones(spectra.freqs.size, bool)
            )
        _check_band(band, mask, min_bins, spectra.freqs)

        sub_freqs = spectra.freqs[mask]
        if integration_weights is not None:
            weights = integration_weights[mask]
        elif weighting == "trapezoid":
            weights = trapezoid_weights(sub_freqs)
        elif weighting == "gradient":
            weights = gradient_weights(sub_freqs)
        else:
            weights = np.ones(sub_freqs.size)
        values, flags = kernel(spectra.data[:, :, :, mask], sub_freqs, weights)
        coverage = band_coverage(spectra.coverage[:, :, :, mask], weights)
        support = band_coverage(spectra.support[:, :, :, mask], weights) if restricted else None

        base = values[:, :, baseline_index] if baseline_index is not None else None
        values = normalize(values, baseline=base, mode=mode)
        all_flags = {**spectra.flags, **flags}
        if baseline_index is not None:
            coverage = np.minimum(coverage, coverage[:, :, baseline_index, np.newaxis])
            # A ratio to the baseline rests on both windows, so on the less supported one.
            if support is not None:
                support = np.minimum(support, support[:, :, baseline_index, np.newaxis])
            all_flags = {
                key: array | array[:, :, baseline_index, np.newaxis]
                for key, array in all_flags.items()
            }

        units = aggregate(values, coverage, spectra.ch_names, groups, include_global)
        resolution = float(np.median(np.diff(sub_freqs))) if sub_freqs.size > 1 else None

        def make_meta(
            spatial: SpatialUnit,
            window: Window,
            _b: Band | None = band,
            _r: float | None = resolution,
        ) -> FeatureMeta:
            return FeatureMeta(
                measure=measure,
                band=_b,
                space=spatial.space,
                space_kind=spatial.space_kind,
                window=window.name,
                normalization=mode,
                unit=unit,
                source=spectra.source,
                window_bounds=(window.tmin, window.tmax),
                computation=ComputationSpec.create(
                    measure,
                    input_source=spectra.source,
                    input_computation=spectra.computation.record(),
                    baseline=baseline,
                    normalization=mode,
                    minimum_bins=min_bins,
                    weighting=weighting,
                    spatial_aggregation="arithmetic_mean_of_channel_features",
                    spatial_channels=_members(spectra.ch_names, spatial),
                    parameters=parameters,
                    **baseline_parameters,
                ),
                freq_resolution_hz=_r,
            )

        band_columns, band_flags = _collect(
            units,
            spectra.windows,
            all_flags,
            make_meta,
            skip_window=baseline_index,
            support=support,
        )
        columns.extend(band_columns)
        for key, arrays in band_flags.items():
            flag_columns.setdefault(key, []).extend(arrays)

    return _assemble(columns, flag_columns, row_ids=spectra.row_ids)


def _baseline_index(spectra: Spectra, baseline: str | None) -> int | None:
    if baseline is None:
        return None
    names = [w.name for w in spectra.windows]
    if baseline not in names:
        raise ValueError(f"baseline window {baseline!r} is not among {names}.")
    if len(names) < 2:
        raise ValueError("baseline normalization needs at least one non-baseline window.")
    return names.index(baseline)


def _check_band(
    band: Band | None,
    mask: npt.NDArray[np.bool_],
    min_bins: int,
    freqs: npt.NDArray[np.float64],
) -> None:
    label = band.name if band is not None else "the fitted range"
    n_bins = int(mask.sum())
    if n_bins == 0:
        raise ValueError(
            f"band {label!r} contains no frequencies of the axis spanning "
            f"({freqs[0]}, {freqs[-1]})."
        )
    if n_bins < min_bins:
        raise ValueError(
            f"band {label!r} holds {n_bins} frequency bins but this measure needs at least "
            f"{min_bins} bins. Compute the input on a finer frequency grid."
        )


Series = TypeVar("Series", bound=TimeSeries)

SignalKernel: TypeAlias = Callable[
    [Series, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]],
    dict[str, npt.NDArray[np.float64]],
]
"""Measure one window of a series, given ``(series, trace, times, mask)``.

Generic in the series type, so a kernel written against :class:`BandSignal` stays
typed as such while :func:`expand_signal` also accepts a plain :class:`Signal`.

``trace`` and ``times`` are already restricted to the window; ``mask`` is the
boolean selector that produced them, for a kernel that also has to index an
unwindowed array on the series. Recovering it from ``times`` by value would
work only while the two arrays are the same floats.
"""


def expand_signal(
    signals: Sequence[Series],
    *,
    trace_of: Callable[[Series], npt.NDArray[np.float64]],
    kernel: SignalKernel[Series],
    units: Mapping[str, str],
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None,
    include_global: bool,
    mode: Normalization,
    parameters: Mapping[str, object],
    flags_of: Callable[[Series], Mapping[str, npt.NDArray[np.bool_]]] | None = None,
    reference_coverage_of: Callable[[Series], npt.NDArray[np.float64]] | None = None,
    row_groups: npt.NDArray[np.int_] | None = None,
    row_labels: tuple[str, ...] | None = None,
) -> FeatureTable:
    """Apply a kernel across bands, windows and spatial groups.

    ``row_groups`` maps each epoch to an output row, for measures estimated across
    trials rather than within one. The kernel then returns one value per row
    instead of per epoch, and coverage is averaged over each group's epochs.

    ``flags_of`` annotates cells whose value rests on a condition of the input
    rather than on the measurement, such as a baseline too degenerate to anchor a
    ratio. It returns arrays shaped ``(n_epochs, n_channels)``, broadcast across
    windows, and a flag set on any member channel marks the whole spatial unit.

    ``reference_coverage_of`` returns per-epoch, per-channel reference coverage.
    Each window's coverage is limited by it before spatial or trial aggregation.
    """
    check_signals(signals, windows)
    if (row_groups is None) != (row_labels is None):
        raise ValueError("row_groups and row_labels must be given together.")
    if flags_of is not None and row_groups is not None:
        # Flags are per epoch and channel; a group row has no single epoch to carry
        # them, and silently reducing them would misreport which trial was flagged.
        raise ValueError("flags_of cannot be combined with cross-trial row_groups.")
    columns: list[_Column] = []
    flag_columns: dict[str, list[npt.NDArray[np.bool_]]] = {}

    for signal in signals:
        # Blanked once here rather than in each kernel: the coverage taken from the
        # same signal a few lines below already counts a non-finite sample as
        # missing, and a kernel reducing over the raw trace would contradict it.
        trace = blank_non_finite(trace_of(signal))
        trace_coverage = np.where(np.isfinite(trace), signal.coverage, 0.0)
        # One flag per epoch and channel, held across every window of this signal.
        signal_flags = {
            key: np.repeat(array[:, :, np.newaxis], len(windows), axis=2)
            for key, array in (flags_of(signal) if flags_of is not None else {}).items()
        }
        reference_coverage = (
            None if reference_coverage_of is None else reference_coverage_of(signal)
        )
        by_measure: dict[str, list[npt.NDArray[np.float64]]] = {}
        coverages: list[npt.NDArray[np.float64]] = []
        for window in windows:
            mask = window_mask(signal.times, window)
            measured = kernel(signal, trace[:, :, mask], signal.times[mask], mask)
            for name, values in measured.items():
                by_measure.setdefault(name, []).append(values)
            per_epoch = trace_coverage[:, :, mask].mean(axis=2)
            if reference_coverage is not None:
                per_epoch = np.minimum(per_epoch, reference_coverage)
            coverages.append(
                per_epoch
                if row_groups is None
                else _reduce_rows(per_epoch, row_groups, len(row_labels or ()))
            )
        coverage = np.stack(coverages, axis=2)

        for measure, per_window in by_measure.items():
            if measure not in units:
                raise ValueError(f"kernel returned measure {measure!r} with no unit declared.")
            stacked = np.stack(per_window, axis=2)
            spatial_units = aggregate(stacked, coverage, signal.ch_names, groups, include_global)

            def make_meta(
                spatial: SpatialUnit,
                window: Window,
                _m: str = measure,
                _s: Series = signal,
            ) -> FeatureMeta:
                return FeatureMeta(
                    measure=_m,
                    band=_s.band,
                    space=spatial.space,
                    space_kind=spatial.space_kind,
                    window=window.name,
                    normalization=mode,
                    unit=units[_m],
                    source=_s.source,
                    window_bounds=(window.tmin, window.tmax),
                    computation=ComputationSpec.create(
                        _m,
                        input_source=_s.source,
                        input_computation=_s.computation.record(),
                        normalization=mode,
                        spatial_aggregation="arithmetic_mean_of_channel_features",
                        spatial_channels=_members(_s.ch_names, spatial),
                        parameters=parameters,
                    ),
                    freq_resolution_hz=None,
                )

            new_columns, new_flags = _collect(
                spatial_units, windows, signal_flags, make_meta, skip_window=None
            )
            columns.extend(new_columns)
            for key, arrays in new_flags.items():
                flag_columns.setdefault(key, []).extend(arrays)

    return _assemble(
        columns,
        flag_columns,
        row_labels,
        row_ids=None if row_labels is not None else signals[0].row_ids,
    )


def _reduce_rows(
    per_epoch: npt.NDArray[np.float64],
    row_groups: npt.NDArray[np.int_],
    n_rows: int,
) -> npt.NDArray[np.float64]:
    return np.stack([per_epoch[row_groups == row].mean(axis=0) for row in range(n_rows)])


def check_signals(signals: Sequence[TimeSeries], windows: Sequence[Window]) -> None:
    """Reject signals that cannot be combined column-wise into one table.

    Equal ``row_ids`` also settles the epoch count, since every series validates
    one identity per epoch on construction.
    """
    if not isinstance(signals, Sequence):
        raise TypeError(
            "measures take a sequence of signals, one per band, such as [signal]; got "
            f"{type(signals).__name__}. Build one from epochs with Signal.from_epochs or "
            "BandSignal.from_epochs."
        )
    if not signals:
        raise ValueError("at least one signal is required.")
    if not windows:
        raise ValueError("at least one window is required.")
    first = signals[0]
    for signal in signals[1:]:
        if signal.ch_names != first.ch_names:
            raise ValueError(
                "all signals must share the same channels in the same order; got "
                f"{first.ch_names} and {signal.ch_names}."
            )
        if signal.row_ids != first.row_ids:
            raise ValueError(
                "all signals must share exact row identities in the same order; the "
                "recording, epoch or event identity differs. Rows are never aligned "
                "or reindexed here."
            )
        if not np.isclose(signal.sfreq, first.sfreq):
            raise ValueError(
                "all signals must share the same sampling frequency; got "
                f"{first.sfreq} and {signal.sfreq} Hz."
            )
        if signal.times.shape != first.times.shape or not np.allclose(signal.times, first.times):
            raise ValueError("all signals must share the same time axis.")


def window_mask(times: npt.NDArray[np.float64], window: Window) -> npt.NDArray[np.bool_]:
    """Boolean mask of the samples a window covers, inclusive of both bounds."""
    mask = (times >= window.tmin) & (times <= window.tmax)
    if not mask.any():
        raise ValueError(
            f"window {window.name!r} ({window.tmin}, {window.tmax}) selects no samples "
            f"from a time axis spanning ({times[0]}, {times[-1]})."
        )
    require_within_axis(times, window, tolerance=sample_period(times))
    return mask
