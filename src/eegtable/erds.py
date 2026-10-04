from __future__ import annotations

import functools
import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from eegtable._expand import expand_signal, window_mask
from eegtable._validation import blank_non_finite, minimum_sample_count
from eegtable.baseline import normalize as _normalize
from eegtable.signal import BandSignal
from eegtable.spectra import Window
from eegtable.table import FeatureTable

_MIN_BASELINE_FRACTION = 1e-6
"""Smallest baseline power that can anchor a ratio, as a fraction of the epoch's own.

A baseline near zero turns a quiet channel into an ERDS value of order 1e6
percent, which is arithmetically valid and physically meaningless. The guard is
relative rather than an absolute number of V², because absolute power is a
property of the band and the montage, not of the data being valid: a gamma
envelope of well under a microvolt is ordinary EEG, and an absolute floor near
1e-12 V² discards it. A baseline a millionth of the same channel's power over
the whole epoch is a dropout, not a quiet channel, at any montage scale.

Refused baselines yield NaN and set the ``baseline_degenerate`` flag, so a
withheld value is distinguishable from one that was never measurable.
"""

_EXTREME_POWER_RATIO = 1e4
"""Power ratio above which a value is reported but flagged rather than withheld.

A baseline a hundred times smaller in amplitude than the epoch's peak yields an
ERDS of order 1e6 percent. That is suspicious, but it is a measurement, not a
missing value: a hundredfold response is what stimulation artifact and muscle
look like, and no scale-free rule separates that from a true response. So the
value stands and ``baseline_extreme_ratio`` marks it for the caller's own QC.
"""

ErdsScale = Literal["percent", "db"]

# Blanked power, baseline reference (NaN where degenerate), baseline spread, degenerate.
_Prepared = tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]
]

_UNITS: dict[str, dict[str, str]] = {
    "percent": {
        "erds_mean": "%",
        "erds_slope": "%/s",
        "erd_magnitude": "%",
        "erd_duration": "s",
        "ers_magnitude": "%",
        "ers_duration": "s",
        "erds_peak_latency": "s",
        "erds_onset_latency": "s",
        "erds_rebound_latency": "s",
    },
    "db": {
        "erds_mean": "dB",
        "erds_slope": "dB/s",
        "erd_magnitude": "dB",
        "erd_duration": "s",
        "ers_magnitude": "dB",
        "ers_duration": "s",
        "erds_peak_latency": "s",
        "erds_onset_latency": "s",
        "erds_rebound_latency": "s",
    },
}


def erds_mean(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "db",
) -> FeatureTable:
    """Mean of the ERDS trace over the window.

    Negative values indicate lower power relative to baseline; positive values
    indicate higher power. This summary alone does not test a task effect.

    Power is referenced to the supplied baseline separately for each epoch
    and channel. The baseline need not be a pre-stimulus interval.

    Decibels are the default. Percent change summarizes additive changes
    relative to baseline; decibels summarize multiplicative changes. Both
    remain sensitive to baseline estimation and the distribution of power.

    **Two ways to average decibels.** This function averages the per-sample dB
    trace over the window. :func:`~eegtable.mean_tfr_power` with a baseline and
    ``normalize="db"`` takes the dB of the window-mean power instead. They are not
    the same estimator. For identical positive power samples, mean log power
    is no greater than log mean power; an exponential population gives a gap
    of about 2.51 dB. These functions also use different power estimators:
    Hilbert band power here and Morlet power in ``mean_tfr_power``.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window each trial is referenced to. Required: ERDS without a baseline is
        not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "db"
        Decibels or percent change from baseline; see :func:`erds_mean` for
        the averaging convention.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "erds_mean",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
    )


def erds_slope(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "db",
) -> FeatureTable:
    """Least-squares slope of the ERDS trace against time.

    Positive slopes indicate increasing baseline-relative power across the
    window; negative slopes indicate decreasing power. The sign alone does
    not identify recovery or a task response.
    NaN with fewer than three finite samples.

    Power is referenced to the supplied baseline separately for each epoch
    and channel. The baseline need not be a pre-stimulus interval.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window each trial is referenced to. Required: ERDS without a baseline is
        not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "db"
        Decibels or percent change from baseline; see :func:`erds_mean` for
        the averaging convention.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "erds_slope",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
    )


def erd_magnitude(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "db",
) -> FeatureTable:
    """Mean depth of the desynchronized part of the trace.

    The mean of ``abs(trace)`` over finite samples below zero. Returns ``0.0``
    when usable samples exist but none are negative, and NaN when no sample
    is usable. Conditioning on negative excursions can produce a positive
    magnitude during unchanged stationary activity. Interpret it against an
    appropriate experimental or statistical reference.

    Power is referenced to the supplied baseline separately for each epoch
    and channel. The baseline need not be a pre-stimulus interval.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window each trial is referenced to. Required: ERDS without a baseline is
        not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "db"
        Decibels or percent change from baseline; see :func:`erds_mean` for
        the averaging convention.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "erd_magnitude",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
    )


def erd_duration(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "db",
) -> FeatureTable:
    """Time spent desynchronized.

    Count of finite samples below zero divided by the sampling rate. Returns
    ``0.0`` when usable samples exist but none are negative, and NaN when no
    sample is usable. If instantaneous power is exponential and the baseline
    equals its population mean, the expected below-baseline fraction is
    ``1 - 1/e``, approximately 0.632. Estimated baselines and other power
    distributions change this reference; half the window is not a general null.

    Power is referenced to the supplied baseline separately for each epoch
    and channel. The baseline need not be a pre-stimulus interval.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window each trial is referenced to. Required: ERDS without a baseline is
        not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "db"
        Decibels or percent change from baseline; see :func:`erds_mean` for
        the averaging convention.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "erd_duration",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
    )


def ers_magnitude(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "db",
) -> FeatureTable:
    """Mean height of the synchronized part of the trace.

    The mean of the trace over finite samples above zero. Returns ``0.0`` when
    usable samples exist but none are positive, and NaN when no sample is
    usable. Conditioning on positive excursions can produce a positive
    magnitude during unchanged stationary activity. Interpret it against an
    appropriate experimental or statistical reference.

    Power is referenced to the supplied baseline separately for each epoch
    and channel. The baseline need not be a pre-stimulus interval.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window each trial is referenced to. Required: ERDS without a baseline is
        not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "db"
        Decibels or percent change from baseline; see :func:`erds_mean` for
        the averaging convention.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "ers_magnitude",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
    )


def ers_duration(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "db",
) -> FeatureTable:
    """Time spent synchronized.

    Count of finite samples above zero divided by the sampling rate. Returns
    ``0.0`` when usable samples exist but none are positive, and NaN when no
    sample is usable. If instantaneous power is exponential and the baseline
    equals its population mean, the expected above-baseline fraction is
    ``1/e``, approximately 0.368. Estimated baselines and other power
    distributions change this reference; half the window is not a general null.

    Power is referenced to the supplied baseline separately for each epoch
    and channel. The baseline need not be a pre-stimulus interval.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window each trial is referenced to. Required: ERDS without a baseline is
        not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "db"
        Decibels or percent change from baseline; see :func:`erds_mean` for
        the averaging convention.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "ers_duration",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
    )


def erds_peak_latency(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "db",
) -> FeatureTable:
    """Time of the largest excursion from baseline, in either direction.

    Located by ``argmax(abs(trace))``, so a deep desynchronization outranks a shallower
    synchronization.

    Power is referenced to the supplied baseline separately for each epoch
    and channel. The baseline need not be a pre-stimulus interval.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window each trial is referenced to. Required: ERDS without a baseline is
        not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "db"
        Decibels or percent change from baseline; see :func:`erds_mean` for
        the averaging convention.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "erds_peak_latency",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
    )


def erds_onset_latency(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "db",
    min_duration_ms: float | None = None,
    min_duration_cycles: float = 6.0,
) -> FeatureTable:
    """Time the trace first leaves the baseline's own variability and stays out.

    The start of the first run of consecutive samples whose absolute raw-power
    departure from the baseline mean exceeds one baseline standard deviation and
    persists for the required duration. The criterion is independent of percent
    versus decibel output. NaN when no run lasts that long.

    The default persistence is six cycles of the band's low-frequency edge.
    This expresses the duration relative to that edge; envelope autocorrelation
    and filter bandwidth also affect detection. The persistence rule does not
    define a significance level or guarantee a false-onset rate. A detected
    excursion alone is not evidence of a task response.

    Power is referenced to the supplied baseline separately for each epoch
    and channel. The baseline need not be a pre-stimulus interval.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window each trial is referenced to. Required: ERDS without a baseline is
        not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "db"
        Decibels, or percent change from baseline. See :func:`erds_mean`.
    min_duration_ms : float, optional
        Shortest excursion that counts as an onset, in milliseconds. When given it
        replaces ``min_duration_cycles`` for every band. Zero retains the first
        single-sample crossing when one exists.
    min_duration_cycles : float, default 6.0
        Shortest excursion in cycles of each band's ``fmin``, used when
        ``min_duration_ms`` is None. A band with ``fmin`` of zero has no cycle
        length and must be given ``min_duration_ms``.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    if min_duration_ms is not None and (not np.isfinite(min_duration_ms) or min_duration_ms < 0.0):
        raise ValueError(f"min_duration_ms must be finite and non-negative, got {min_duration_ms}.")
    if not np.isfinite(min_duration_cycles) or min_duration_cycles < 0.0:
        raise ValueError(
            f"min_duration_cycles must be finite and non-negative, got {min_duration_cycles}."
        )
    if min_duration_ms is None:
        for signal in signals:
            if signal.band.fmin <= 0.0:
                raise ValueError(
                    f"band {signal.band.name!r} starts at 0 Hz, so a persistence in cycles is "
                    "undefined; pass min_duration_ms."
                )
    return _erds_measure(
        signals,
        "erds_onset_latency",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
        onset_persistence=(min_duration_ms, min_duration_cycles),
    )


def erds_rebound_latency(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "db",
) -> FeatureTable:
    """Latency of the maximum of the ERDS trace strictly after the peak latency.

    Returns NaN when no finite sample follows the peak latency within the
    analysis window. The later maximum need not be positive or exceed baseline;
    the name does not impose an ERS criterion.

    Power is referenced to the supplied baseline separately for each epoch
    and channel. The baseline need not be a pre-stimulus interval.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window each trial is referenced to. Required: ERDS without a baseline is
        not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "db"
        Decibels or percent change from baseline; see :func:`erds_mean` for
        the averaging convention.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "erds_rebound_latency",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
    )


def _erds_measure(
    signals: Sequence[BandSignal],
    measure: str,
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None,
    include_global: bool,
    normalize: ErdsScale,
    onset_persistence: tuple[float | None, float] = (None, 6.0),
) -> FeatureTable:
    if normalize not in ("percent", "db"):
        raise ValueError(f"normalize must be 'percent' or 'db', got {normalize!r}.")
    min_duration_ms, min_duration_cycles = onset_persistence

    def onset_seconds(signal: BandSignal) -> float:
        if min_duration_ms is not None:
            return min_duration_ms / 1000.0
        return min_duration_cycles / signal.band.fmin

    # The power and its baseline feed the trace, the flags and every window's onset. The
    # expander finishes one signal before the next, so one entry spares the recomputation
    # while holding a single band's power.
    @functools.lru_cache(maxsize=1)
    def prepared(signal: BandSignal) -> _Prepared:
        power = _power(signal)
        return (power, *_baseline_reference(power, signal.times, baseline))

    def trace_of(signal: BandSignal) -> npt.NDArray[np.float64]:
        power, reference, _, _ = prepared(signal)
        return _normalize(power, baseline=reference, mode=normalize)

    def kernel(
        signal: BandSignal,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
    ) -> dict[str, npt.NDArray[np.float64]]:
        if measure != "erds_onset_latency":
            return {measure: _measures(signal, trace, times)[measure]}
        power, reference, deviation, _ = prepared(signal)
        # The expander's own selector, not one recovered from the time values. A missing
        # sample or baseline is NaN here, and NaN never crosses.
        departure = np.abs(power[:, :, mask] - reference[:, :, np.newaxis])
        onset_crossing = departure > deviation[:, :, np.newaxis]
        onset_samples = minimum_sample_count(onset_seconds(signal), signal.sfreq)
        usable = np.asarray(np.isfinite(trace).any(axis=2), dtype=np.bool_)
        return {measure: _onset(times, usable, onset_crossing, onset_samples)}

    def flags_of(signal: BandSignal) -> dict[str, npt.NDArray[np.bool_]]:
        power, reference, _, degenerate = prepared(signal)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
            # A degenerate reference is NaN, and no peak compares above it.
            extreme = np.nanmax(power, axis=2) > _EXTREME_POWER_RATIO * reference
        return {
            "baseline_degenerate": degenerate,
            "baseline_extreme_ratio": np.asarray(extreme, dtype=np.bool_),
        }

    return expand_signal(
        signals,
        trace_of=trace_of,
        kernel=kernel,
        flags_of=flags_of,
        units={measure: _UNITS[normalize][measure]},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode=normalize,
        parameters={
            "baseline": {"name": baseline.name, "tmin": baseline.tmin, "tmax": baseline.tmax},
            # Only the onset is defined by the persistence, so only its columns carry it.
            **(
                {
                    "onset_criterion": "absolute_power_deviation_exceeds_baseline_sd_sustained",
                    "onset_min_duration_ms": min_duration_ms,
                    "onset_min_duration_cycles": (
                        None if min_duration_ms is not None else min_duration_cycles
                    ),
                }
                if measure == "erds_onset_latency"
                else {}
            ),
        },
    )


def _power(signal: BandSignal) -> npt.NDArray[np.float64]:
    """Instantaneous power with non-finite samples blanked.

    Read here rather than through the analysis trace, so nothing has blanked it
    yet. It matters more than a wrong mean: an infinity makes the baseline
    reference non-finite, which is read as degenerate, and every ERDS measure for
    that channel is withheld over a single bad sample.
    """
    return blank_non_finite(signal.power)


def _baseline_reference(
    power: npt.NDArray[np.float64], times: npt.NDArray[np.float64], baseline: Window
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Baseline mean power, NaN where it cannot anchor a ratio; its spread; those cells."""
    within = power[:, :, window_mask(times, baseline)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        warnings.filterwarnings("ignore", "Degrees of freedom <= 0", RuntimeWarning)
        reference = np.nanmean(within, axis=2)
        deviation = np.nanstd(within, axis=2)
        # The whole epoch, not just the baseline, so the comparison is against this
        # channel's own scale rather than an assumed unit of measurement.
        scale = np.nanmean(power, axis=2)
    with np.errstate(invalid="ignore"):
        degenerate: npt.NDArray[np.bool_] = np.asarray(
            ~np.isfinite(reference)
            | (reference <= 0.0)
            | (np.isfinite(scale) & (reference <= _MIN_BASELINE_FRACTION * scale)),
            dtype=np.bool_,
        )
    return np.where(degenerate, np.nan, reference), deviation, degenerate


def _measures(
    signal: BandSignal,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.float64]]:
    finite = np.isfinite(trace)
    usable: npt.NDArray[np.bool_] = np.asarray(finite.any(axis=2), dtype=np.bool_)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        mean = np.where(usable, np.nanmean(trace, axis=2), np.nan)
    peak_index = _argmax_masked(np.abs(trace), finite)
    return {
        "erds_mean": mean,
        "erds_slope": _slope(trace, times, finite),
        "erd_magnitude": _signed_magnitude(trace, finite, usable, negative=True),
        "erd_duration": _signed_duration(trace, finite, usable, signal.sfreq, negative=True),
        "ers_magnitude": _signed_magnitude(trace, finite, usable, negative=False),
        "ers_duration": _signed_duration(trace, finite, usable, signal.sfreq, negative=False),
        "erds_peak_latency": np.where(usable, times[peak_index], np.nan),
        "erds_rebound_latency": _rebound(trace, times, finite, usable, peak_index),
    }


def _argmax_masked(
    values: npt.NDArray[np.float64], finite: npt.NDArray[np.bool_]
) -> npt.NDArray[np.int_]:
    return np.asarray(np.argmax(np.where(finite, values, -np.inf), axis=2), dtype=np.int_)


def _onset(
    times: npt.NDArray[np.float64],
    usable: npt.NDArray[np.bool_],
    crossed: npt.NDArray[np.bool_],
    min_samples: int,
) -> npt.NDArray[np.float64]:
    """Start of the first run of at least ``min_samples`` consecutive crossings.

    A single sample beyond one baseline SD is met by chance on essentially every
    trial, so a crossing only counts once it has persisted. A non-finite sample
    breaks a run: missing data is not evidence that the excursion continued.
    """
    n_times = crossed.shape[2]
    run = np.zeros(crossed.shape[:2], dtype=int)
    start = np.full(crossed.shape[:2], -1, dtype=int)
    for index in range(n_times):
        run = np.where(crossed[:, :, index], run + 1, 0)
        qualifies = (run >= min_samples) & (start < 0)
        start = np.where(qualifies, index - min_samples + 1, start)
    found = start >= 0
    return np.where(usable & found, times[np.maximum(start, 0)], np.nan)


def _rebound(
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    finite: npt.NDArray[np.bool_],
    usable: npt.NDArray[np.bool_],
    peak_index: npt.NDArray[np.int_],
) -> npt.NDArray[np.float64]:
    after = np.arange(trace.shape[2])[np.newaxis, np.newaxis, :] > peak_index[:, :, np.newaxis]
    eligible = finite & after
    any_eligible: npt.NDArray[np.bool_] = np.asarray(eligible.any(axis=2), dtype=np.bool_)
    index = np.argmax(np.where(eligible, trace, -np.inf), axis=2)
    return np.where(usable & any_eligible, times[index], np.nan)


def _slope(
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    finite: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    # Center both variables before products to avoid cancellation on shifted time axes.
    count = finite.sum(axis=2).astype(float)
    t = np.where(finite, times - times[0], 0.0)
    y = np.where(finite, trace, 0.0)
    divisor = np.maximum(count, 1)[..., np.newaxis]
    t = np.where(finite, t - t.sum(axis=2, keepdims=True) / divisor, 0.0)
    y = np.where(finite, y - y.sum(axis=2, keepdims=True) / divisor, 0.0)
    denominator = (t * t).sum(axis=2)
    numerator = (t * y).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((count > 2) & (denominator > 0.0), numerator / denominator, np.nan)


def _signed_magnitude(
    trace: npt.NDArray[np.float64],
    finite: npt.NDArray[np.bool_],
    usable: npt.NDArray[np.bool_],
    *,
    negative: bool,
) -> npt.NDArray[np.float64]:
    selected = finite & (trace < 0.0 if negative else trace > 0.0)
    total = np.where(selected, np.abs(trace) if negative else trace, 0.0).sum(axis=2)
    count = selected.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        magnitude = np.where(count > 0, total / count, 0.0)
    # Absent is zero; unmeasurable is NaN. They are different statements.
    return np.where(usable, magnitude, np.nan)


def _signed_duration(
    trace: npt.NDArray[np.float64],
    finite: npt.NDArray[np.bool_],
    usable: npt.NDArray[np.bool_],
    sfreq: float,
    *,
    negative: bool,
) -> npt.NDArray[np.float64]:
    selected = finite & (trace < 0.0 if negative else trace > 0.0)
    return np.where(usable, selected.sum(axis=2) / sfreq, np.nan)
