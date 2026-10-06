"""Cycle-by-cycle waveform summaries delegated to ByCycle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from importlib.metadata import version
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt
import pandas as pd

from eegtable._expand import check_signals, expand_signal, window_mask
from eegtable.bands import Band, passband_fraction
from eegtable.signal import Signal
from eegtable.spectra import Window
from eegtable.table import FeatureTable, concat

_DEFAULT_THRESHOLDS: dict[str, float | int] = {
    "amp_fraction_threshold": 0.0,
    "amp_consistency_threshold": 0.5,
    "period_consistency_threshold": 0.5,
    "monotonicity_threshold": 0.8,
    "min_n_cycles": 3,
}
_SHAPE_FIELDS = {
    "cycle_period": ("period", "s"),
    "cycle_rise_time": ("time_rise", "s"),
    "cycle_decay_time": ("time_decay", "s"),
    "cycle_rise_decay_symmetry": ("time_rdsym", "a.u."),
    "cycle_peak_trough_symmetry": ("time_ptsym", "a.u."),
    "cycle_amplitude": ("volt_amp", "V"),
}
_UNITS = {
    **{name: unit for name, (_, unit) in _SHAPE_FIELDS.items()},
    "cycle_burst_fraction": "fraction",
    "cycle_count": "count",
    "cycle_burst_count": "count",
}


def _validate_thresholds(
    burst_thresholds: Mapping[str, float | int] | None,
) -> dict[str, float | int]:
    thresholds = dict(_DEFAULT_THRESHOLDS)
    if burst_thresholds is not None:
        unknown = set(burst_thresholds) - set(thresholds)
        if unknown:
            raise ValueError(f"unknown burst thresholds: {sorted(unknown)}.")
        thresholds.update(burst_thresholds)
    for name, value in thresholds.items():
        if name == "min_n_cycles":
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError("min_n_cycles must be a positive integer.")
        elif (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be finite and between zero and one.")
    return thresholds


def _detect_cycles(
    signal: Signal, band: Band, thresholds: Mapping[str, float | int]
) -> list[list[pd.DataFrame]]:
    from bycycle.features import compute_features

    if not isinstance(signal, Signal):
        raise TypeError("cycle_features requires broadband Signal inputs.")
    if band.fmin <= 0 or band.fmax >= signal.sfreq / 2:
        raise ValueError("cycle bands must lie strictly between zero and Nyquist.")
    if signal.passband is not None and passband_fraction(band, *signal.passband) < 1:
        raise ValueError("cycle bands must lie entirely within the recording passband.")
    if not np.isfinite(signal.data).all():
        raise ValueError("cycle_features requires finite samples throughout each epoch.")
    if np.any(np.ptp(signal.data, axis=-1) == 0):
        raise ValueError("cycle_features requires nonconstant traces.")
    if signal.data.shape[-1] <= np.ceil(3 * signal.sfreq / band.fmin):
        raise ValueError("each epoch must exceed the three-cycle ByCycle filter length.")
    return [
        [
            compute_features(
                trace,
                signal.sfreq,
                (band.fmin, band.fmax),
                center_extrema="peak",
                burst_method="cycles",
                threshold_kwargs=dict(thresholds),
                return_samples=True,
            )
            for trace in epoch
        ]
        for epoch in signal.data
    ]


def _summarize_cycles(
    frames: list[list[pd.DataFrame]],
    signal: Signal,
    bounds: tuple[float, float],
) -> tuple[dict[str, npt.NDArray[np.float64]], dict[str, npt.NDArray[np.bool_]]]:
    shape = signal.data.shape[:2]
    values = {name: np.full(shape, np.nan) for name in _UNITS}
    flags = {
        "cycle_no_complete_cycles": np.zeros(shape, dtype=bool),
        "cycle_no_burst": np.zeros(shape, dtype=bool),
    }
    for epoch, channel in np.ndindex(shape):
        frame = frames[epoch][channel]
        complete = frame[
            (signal.times[frame.sample_last_trough.to_numpy(dtype=int)] >= bounds[0])
            & (signal.times[frame.sample_next_trough.to_numpy(dtype=int)] <= bounds[1])
        ]
        bursting = complete[complete.is_burst]
        count = len(complete)
        burst_count = len(bursting)
        values["cycle_count"][epoch, channel] = count
        values["cycle_burst_count"][epoch, channel] = burst_count
        flags["cycle_no_complete_cycles"][epoch, channel] = count == 0
        flags["cycle_no_burst"][epoch, channel] = burst_count == 0
        if count:
            values["cycle_burst_fraction"][epoch, channel] = burst_count / count
        if burst_count:
            for name, (field, unit) in _SHAPE_FIELDS.items():
                result = float(bursting[field].mean())
                values[name][epoch, channel] = result / signal.sfreq if unit == "s" else result
    return values, flags


def _expand_flags(
    table: FeatureTable,
    signal: Signal,
    flags: Mapping[str, Mapping[str, npt.NDArray[np.bool_]]],
    groups: Mapping[str, Sequence[str]] | None,
) -> FeatureTable:
    columns: dict[str, list[npt.NDArray[np.bool_]]] = {}
    for meta in table.meta:
        members: Sequence[str]
        if meta.space_kind == "channel":
            members = (meta.space,)
        elif meta.space_kind == "roi":
            assert groups is not None
            members = groups[meta.space]
        else:
            members = signal.ch_names
        picks = [signal.ch_names.index(member) for member in members]
        assert meta.window is not None
        for name, array in flags[meta.window].items():
            columns.setdefault(name, []).append(np.asarray(array[:, picks].any(axis=1), dtype=bool))
    return replace(table, flags={name: np.stack(items, axis=1) for name, items in columns.items()})


def cycle_features(
    series: Sequence[Signal],
    *,
    windows: Sequence[Window],
    bands: Sequence[Band],
    burst_thresholds: Mapping[str, float | int] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Mean waveform features of bursting cycles and their cycle fraction.

    ByCycle detects trough-to-trough cycles on each whole epoch, using its
    three-cycle narrowband locator and the original broadband amplitudes.
    Only cycles with both troughs within a requested window count. Waveform
    summaries average the burst-labelled cycles; the burst fraction counts
    burst cycles divided by all complete cycles, rather than sample occupancy.
    No cycles or no bursts give explicit flags and undefined summaries.
    Requires ``eegtable[cycles]``.

    Parameters
    ----------
    series : sequence of Signal
        Broadband epochs, finite and nonconstant, longer than three cycles at each
        band's lower edge.
    windows : sequence of Window
        Analysis windows. A cycle counts when both of its troughs lie inside the
        window.
    bands : sequence of Band
        Bands of ByCycle's narrowband locator, with unique names. Each must lie
        strictly between 0 Hz and the Nyquist frequency and, when the recording's
        passband is known, entirely inside it.
    burst_thresholds : mapping of str to float, optional
        Overrides of ByCycle's cycle-consistency thresholds:
        ``amp_fraction_threshold`` (default 0.0), ``amp_consistency_threshold``
        (0.5), ``period_consistency_threshold`` (0.5) and
        ``monotonicity_threshold`` (0.8), each in ``[0, 1]``, and ``min_n_cycles``
        (3), a positive integer.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        For each band, spatial unit and window: mean ``cycle_period``,
        ``cycle_rise_time`` and ``cycle_decay_time`` in seconds,
        ``cycle_rise_decay_symmetry``, ``cycle_peak_trough_symmetry`` and
        ``cycle_amplitude`` of the burst cycles (NaN without a burst), plus
        ``cycle_burst_fraction`` (NaN without a complete cycle), ``cycle_count``
        and ``cycle_burst_count``. Flags ``cycle_no_complete_cycles`` and
        ``cycle_no_burst`` mark those cases.
    """
    check_signals(series, windows)
    if not bands or len({band.name for band in bands}) != len(bands):
        raise ValueError("at least one cycle band is required, with unique band names.")
    thresholds = _validate_thresholds(burst_thresholds)
    tables = []
    for signal in series:
        for band in bands:
            frames = _detect_cycles(signal, band, thresholds)
            window_flags: dict[str, dict[str, npt.NDArray[np.bool_]]] = {}
            summaries: dict[tuple[float, float], dict[str, npt.NDArray[np.float64]]] = {}
            for window in windows:
                selected_times = signal.times[window_mask(signal.times, window)]
                bounds = (float(selected_times[0]), float(selected_times[-1]))
                values, flags = _summarize_cycles(frames, signal, bounds)
                summaries[bounds] = values
                window_flags[window.name] = flags

            def kernel(
                current: Signal,
                trace: npt.NDArray[np.float64],
                times: npt.NDArray[np.float64],
                mask: npt.NDArray[np.bool_],
                _summaries: Mapping[
                    tuple[float, float], dict[str, npt.NDArray[np.float64]]
                ] = summaries,
            ) -> dict[str, npt.NDArray[np.float64]]:
                del current, trace, mask
                bounds = (float(times[0]), float(times[-1]))
                return _summaries[bounds]

            table = expand_signal(
                [signal],
                trace_of=lambda current: current.data,
                kernel=kernel,
                units=_UNITS,
                windows=windows,
                groups=groups,
                include_global=include_global,
                mode="raw",
                parameters={
                    "backend": "bycycle",
                    "backend_version": version("bycycle"),
                    "band": asdict(band),
                    "center_extrema": "peak",
                    "burst_method": "cycles",
                    "burst_thresholds": thresholds,
                    "locator_filter_cycles": 3,
                    "cycle_selection": "both_troughs_inside_window",
                    "shape_aggregation": "mean_of_burst_cycles",
                    "burst_fraction_denominator": "complete_cycle_count",
                },
            )
            table = replace(table, meta=tuple(replace(meta, band=band) for meta in table.meta))
            tables.append(_expand_flags(table, signal, window_flags, groups))
    return concat(tables)
