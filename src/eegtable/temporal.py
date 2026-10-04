from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.integrate import trapezoid
from scipy.signal import find_peaks

from eegtable._expand import SignalKernel, expand_signal
from eegtable._validation import blank_non_finite
from eegtable.signal import TimeSeries
from eegtable.spectra import Window
from eegtable.table import FeatureTable

Polarity = Literal["positive", "negative", "absolute"]

_SEARCH = {"positive": 1.0, "negative": -1.0}


def variance(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Variance of the signal within each window.

    Computed over finite samples within each window, with valid sample fractions
    recorded in the parallel ``coverage`` matrix.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes. The bands axis, if any, comes from this
        sequence.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Variance in the squared units of the input.
    """
    return _measure(series, "variance", "V^2", _variance_kernel, windows, groups, include_global)


def peak_to_peak(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Range of the signal within each window, maximum minus minimum.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Peak-to-peak amplitude in the units of the input.
    """
    return _measure(series, "ptp", "V", _ptp_kernel, windows, groups, include_global)


def mean_amplitude(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Mean of the signal within each window.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Mean amplitude in the units of the input.
    """
    return _measure(series, "mean_amplitude", "V", _mean_kernel, windows, groups, include_global)


def area_under_curve(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Signed area under the signal within each window.

    Integrated by the trapezoid rule over each contiguous run of finite samples,
    and summed. A gap is skipped rather than interpolated across, so a stretch of
    missing data contributes nothing instead of contributing a straight line.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Area in the units of the input times seconds.
    """
    return _measure(series, "auc", "V*s", _auc_kernel, windows, groups, include_global)


def peak_amplitude(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    polarity: Polarity = "absolute",
    prominence: float | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Signed value of the extremum within each window.

    ``polarity`` is specified as an explicit parameter (``"positive"``,
    ``"negative"``, or ``"absolute"``), independent of window labels or naming conventions.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    polarity : {"absolute", "positive", "negative"}, default "absolute"
        Whether to find the largest value, the most negative, or the largest
        excursion in either direction. The value returned is always signed.
    prominence : float, optional
        When given, choose the local peak with the greatest prominence among
        those meeting the minimum. Search contiguous finite stretches separately;
        missing samples cannot establish a peak's baseline. Return NaN when no
        local peak meets the requirement.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Peak amplitude in the units of the input.
    """
    return _peak(
        series, "peak_amplitude", "V", windows, polarity, prominence, groups, include_global
    )


def peak_latency(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    polarity: Polarity = "absolute",
    prominence: float | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Time of the extremum within each window.

    See :func:`peak_amplitude` for how the extremum is located.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    polarity : {"absolute", "positive", "negative"}, default "absolute"
        Which extremum to locate.
    prominence : float, optional
        Minimum prominence for a local peak to qualify.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Latency in seconds, relative to the epoch origin.
    """
    return _peak(series, "peak_latency", "s", windows, polarity, prominence, groups, include_global)


def hjorth_mobility(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    r"""Hjorth mobility expressed on a frequency-equivalent scale.

    .. math:: \mathrm{mobility} = \frac{1}{2\pi}
              \sqrt{\frac{\operatorname{Var}(\mathrm{d}x/\mathrm{d}t)}
              {\operatorname{Var}(x)}}

    The derivative is approximated by ``diff(x) * sfreq`` and the result is
    divided by :math:`2\pi`, giving hertz rather than conventional mobility's
    inverse-second scale. A well-sampled sinusoid approaches its frequency.
    Finite differences have gain :math:`\sin(\pi f/f_s)/(\pi f/f_s)` relative
    to a continuous derivative, so sampling rate and bandwidth still affect
    the estimate.

    For a stationary signal with a well-resolved spectrum, this scale
    approximates the power-weighted root-mean-square frequency.
    :func:`~eegtable.spectral_centroid` instead reports the power-weighted mean
    frequency within its selected band. These summaries need not agree.

    Hjorth activity is the variance of the signal, which :func:`~eegtable.variance`
    already computes; it is not duplicated under a second name.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Mobility in Hz. NaN with fewer than two finite first differences or
        non-positive signal variance. Differences across gaps are excluded.

    References
    ----------
    Hjorth, B. (1970). EEG analysis based on time domain properties.
    Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.
    """
    return _measure(
        series, "hjorth_mobility", "Hz", _mobility_kernel, windows, groups, include_global
    )


def hjorth_complexity(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    r"""Ratio of derivative mobility to signal mobility.

    .. math:: \mathrm{complexity} =
              \frac{\mathrm{mobility}(\mathrm{d}x/\mathrm{d}t)}{\mathrm{mobility}(x)}

    A stationary sinusoid approaches one; broadband spectral content can
    increase the ratio. The explicit sampling-interval factors cancel, so the
    result is dimensionless. Finite-difference frequency response, window
    boundaries, and preprocessing can still make it depend on sampling rate.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Complexity, dimensionless. NaN with fewer than two finite second
        differences, insufficient finite signal/first-difference samples, or
        non-positive signal or first-difference variance.

    References
    ----------
    Hjorth, B. (1970). EEG analysis based on time domain properties.
    Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.
    """
    return _measure(
        series, "hjorth_complexity", "a.u.", _complexity_kernel, windows, groups, include_global
    )


def root_mean_square(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Root mean square amplitude within each window.

    Equal to the standard deviation for a signal with no offset, and larger when
    there is one; :func:`~eegtable.variance` removes the mean first and this does not.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        RMS amplitude in the units of the input.
    """
    return _measure(series, "rms", "V", _rms_kernel, windows, groups, include_global)


def skewness(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Asymmetry of the amplitude distribution within each window.

    Zero for a symmetric distribution, positive when the long tail points up.
    Dimensionless, so it does not change with the recording's amplitude scale.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Skewness, dimensionless.
    """
    return _measure(series, "skewness", "a.u.", _skewness_kernel, windows, groups, include_global)


def kurtosis(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Tail weight of the amplitude distribution within each window.

    Fisher excess kurtosis is zero for a Gaussian population. Large positive
    sample values can reflect extreme amplitude observations, including
    artifacts, but are not specific to an artifact mechanism.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Excess kurtosis, dimensionless.
    """
    return _measure(series, "kurtosis", "a.u.", _kurtosis_kernel, windows, groups, include_global)


def line_length(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Mean absolute rate of change within each window.

    Computed as ``mean(abs(diff(x))) * sfreq`` over finite adjacent differences.
    EEG input in volts gives V/s. This scales differences by elapsed time;
    finite sampling and filtering can still change the estimate when a signal
    is resampled. Dividing by ``sfreq`` recovers the per-sample form.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Mean absolute slope, in the units of the input per second.
    """
    return _measure(
        series, "line_length", "V/s", _line_length_kernel, windows, groups, include_global
    )


def zero_crossing_rate(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Sign changes per second within each window.

    A crossing is a sign change between successive nonzero finite samples in
    one contiguous finite segment. Zeros retain the previous sign, and missing
    samples reset it. The count is divided by the full sampled window duration.
    Expressing a count per second does not remove sampling, bandwidth, or
    window-boundary effects.

    For a sufficiently smooth stationary zero-mean Gaussian process, Rice's
    formula relates the continuous crossing rate to twice the corresponding
    frequency-scaled Hjorth mobility. Discrete crossings and finite-difference
    mobility only approximate that relation; it need not hold near Nyquist or
    for non-Gaussian signals.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes. An envelope is non-negative, so its rate
        is zero by construction; this measure is for signals that cross zero.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Crossings per second.
    """
    return _measure(
        series, "zero_crossing_rate", "1/s", _zero_crossing_kernel, windows, groups, include_global
    )


def amplitude_quantile(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    q: float = 0.5,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """A quantile of the amplitude distribution within each window.

    The default is the median, which is far less sensitive to a single transient
    than :func:`~eegtable.mean_amplitude`.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    q : float, default 0.5
        Quantile in ``[0, 1]``. Note this is a fraction, not a percentage.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        The quantile, in the units of the input.
    """
    if not np.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be a finite fraction in [0, 1], got {q}.")

    def kernel(
        s: TimeSeries,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del s, times, mask
        _, usable = _finite(trace)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
            values = np.nanquantile(trace, q, axis=2)
        return {"amplitude_quantile": np.where(usable, values, np.nan)}

    return expand_signal(
        series,
        trace_of=_trace_of,
        kernel=kernel,
        units={"amplitude_quantile": "V"},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
        parameters={"q": q},
    )


def _measure(
    series: Sequence[TimeSeries],
    measure: str,
    unit: str,
    kernel: SignalKernel[TimeSeries],
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None,
    include_global: bool,
) -> FeatureTable:
    return expand_signal(
        series,
        trace_of=_trace_of,
        kernel=kernel,
        units={measure: unit},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
        parameters={},
    )


def _peak(
    series: Sequence[TimeSeries],
    measure: str,
    unit: str,
    windows: Sequence[Window],
    polarity: Polarity,
    prominence: float | None,
    groups: Mapping[str, Sequence[str]] | None,
    include_global: bool,
) -> FeatureTable:
    if polarity not in ("positive", "negative", "absolute"):
        raise ValueError(
            f"polarity must be 'positive', 'negative' or 'absolute', got {polarity!r}."
        )
    if prominence is not None and not prominence > 0.0:
        raise ValueError(f"prominence must be positive when given, got {prominence}.")

    def kernel(
        s: TimeSeries,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del s, mask
        amplitude, latency = _find_peak(trace, times, polarity, prominence)
        return {measure: amplitude if measure == "peak_amplitude" else latency}

    return expand_signal(
        series,
        trace_of=_trace_of,
        kernel=kernel,
        units={measure: unit},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
        parameters={"polarity": polarity, "prominence": prominence},
    )


def _trace_of(series: TimeSeries) -> npt.NDArray[np.float64]:
    return series.amplitude


def _finite(
    trace: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    finite = np.isfinite(trace)
    usable: npt.NDArray[np.bool_] = np.asarray(finite.any(axis=2), dtype=np.bool_)
    return finite, usable


def _variance_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del series, times, mask
    finite, usable = _finite(trace)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Degrees of freedom <= 0", RuntimeWarning)
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        values = np.where(usable, np.nanvar(trace, axis=2), np.nan)
    del finite
    return {"variance": values}


def _ptp_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del series, times, mask
    finite, usable = _finite(trace)
    high = np.where(finite, trace, -np.inf).max(axis=2)
    low = np.where(finite, trace, np.inf).min(axis=2)
    return {"ptp": np.where(usable, high - low, np.nan)}


def _mean_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del series, times, mask
    _, usable = _finite(trace)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        return {"mean_amplitude": np.where(usable, np.nanmean(trace, axis=2), np.nan)}


def _finite_variance(values: npt.NDArray[np.float64], minimum: int) -> npt.NDArray[np.float64]:
    """Variance over finite samples, NaN where too few of them survive."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Degrees of freedom <= 0", RuntimeWarning)
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        variance = np.nanvar(blank_non_finite(values), axis=2)
    enough = np.isfinite(values).sum(axis=2) >= minimum
    return np.asarray(np.where(enough, variance, np.nan), dtype=np.float64)


def _mobility_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del times, mask
    # Per second, not per sample: np.diff alone would carry the sampling interval
    # into the result and make the same signal report differently at another rate.
    derivative = np.diff(trace, axis=2) * series.sfreq
    # Two derivative samples, so three of the signal: the variance of a single
    # value is zero, which would report a still signal rather than an unusable one.
    signal_variance = _finite_variance(trace, 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(
            signal_variance > 0.0, _finite_variance(derivative, 2) / signal_variance, np.nan
        )
        mobility = np.sqrt(ratio) / (2.0 * np.pi)
    return {"hjorth_mobility": np.asarray(mobility, dtype=np.float64)}


def _complexity_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del series, times, mask
    # A ratio of two mobilities, so the sampling interval cancels and the raw
    # differences are enough; scaling them would divide out again.
    first = np.diff(trace, axis=2)
    second = np.diff(trace, n=2, axis=2)
    signal_variance = _finite_variance(trace, 3)
    first_variance = _finite_variance(first, 2)
    # Two second-difference samples, so four of the signal; see _mobility_kernel.
    second_variance = _finite_variance(second, 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        complexity = np.where(
            (signal_variance > 0.0) & (first_variance > 0.0),
            np.sqrt(second_variance * signal_variance) / first_variance,
            np.nan,
        )
    return {"hjorth_complexity": np.asarray(complexity, dtype=np.float64)}


def _rms_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del series, times, mask
    _, usable = _finite(trace)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        values = np.sqrt(np.nanmean(np.square(blank_non_finite(trace)), axis=2))
    return {"rms": np.where(usable, values, np.nan)}


def _moment_kernel(
    trace: npt.NDArray[np.float64], measure: str, minimum: int
) -> dict[str, npt.NDArray[np.float64]]:
    # scipy returns a masked array under nan_policy="omit"; fill it so the column is
    # plain float, and withhold cells with too few samples for the moment to exist.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = (
            stats.skew(trace, axis=2, nan_policy="omit")
            if measure == "skewness"
            else stats.kurtosis(trace, axis=2, nan_policy="omit")
        )
    values: npt.NDArray[np.float64] = np.ma.filled(np.ma.asarray(raw, dtype=float), np.nan)
    enough = np.isfinite(trace).sum(axis=2) >= minimum
    return {measure: np.asarray(np.where(enough, values, np.nan), dtype=np.float64)}


def _skewness_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del series, times, mask
    return _moment_kernel(trace, "skewness", 3)


def _kurtosis_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del series, times, mask
    return _moment_kernel(trace, "kurtosis", 4)


def _line_length_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del times, mask
    steps = blank_non_finite(np.abs(np.diff(blank_non_finite(trace), axis=2)))
    usable = np.isfinite(steps).any(axis=2)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        # Per second: np.diff alone carries the sampling interval into the result.
        values = np.nanmean(steps, axis=2) * series.sfreq
    return {"line_length": np.where(usable, values, np.nan)}


def _zero_crossing_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del times, mask
    finite = np.isfinite(trace)
    # A sample of exactly zero continues the run it sits in. Counting it as a
    # crossing on the way in and again on the way out reports two where a signal
    # that merely touched the axis made none.
    sign = np.sign(np.where(finite, trace, np.nan))
    previous = np.zeros(sign.shape[:2])
    crossings = np.zeros(sign.shape[:2])
    for index in range(sign.shape[2]):
        current = sign[:, :, index]
        moved = np.isfinite(current) & (current != 0.0)
        crossings += moved & (previous != 0.0) & (current != previous)
        previous = np.where(moved, current, previous)
        previous = np.where(finite[:, :, index], previous, 0.0)
    seconds = float(trace.shape[2]) / series.sfreq
    usable = finite.any(axis=2) & (seconds > 0)
    return {"zero_crossing_rate": np.where(usable, crossings / seconds, np.nan)}


def _auc_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del series, mask
    n_epochs, n_channels, _ = trace.shape
    out = np.full((n_epochs, n_channels), np.nan)
    for epoch in range(n_epochs):
        for channel in range(n_channels):
            out[epoch, channel] = _auc_one(trace[epoch, channel], times)
    return {"auc": out}


def _auc_one(trace: npt.NDArray[np.float64], times: npt.NDArray[np.float64]) -> float:
    valid = np.flatnonzero(np.isfinite(trace) & np.isfinite(times))
    if valid.size < 2:
        return float("nan")
    runs = np.split(valid, np.flatnonzero(np.diff(valid) > 1) + 1)
    total, measured = 0.0, False
    for run in runs:
        if run.size < 2:
            continue
        measured = True
        total += float(trapezoid(trace[run], times[run]))
    return total if measured else float("nan")


def _prominent_peak(search: npt.NDArray[np.float64], prominence: float) -> int | None:
    finite = np.flatnonzero(np.isfinite(search))
    runs = np.split(finite, np.flatnonzero(np.diff(finite) > 1) + 1)
    best_index = None
    best_prominence = -np.inf
    for run in runs:
        peaks, properties = find_peaks(search[run], prominence=prominence)
        if peaks.size:
            candidate = int(np.argmax(properties["prominences"]))
            strength = float(properties["prominences"][candidate])
            if strength > best_prominence:
                best_index = int(run[peaks[candidate]])
                best_prominence = strength
    return best_index


def _find_peak(
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    polarity: Polarity,
    prominence: float | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    finite, usable = _finite(trace)
    sign = _SEARCH.get(polarity)
    search = np.abs(trace) if sign is None else sign * trace
    filled = np.where(finite, search, -np.inf)

    index = np.argmax(filled, axis=2)
    if prominence is not None:
        for epoch in range(trace.shape[0]):
            for channel in range(trace.shape[1]):
                peak = _prominent_peak(search[epoch, channel], prominence)
                usable[epoch, channel] = peak is not None
                if peak is not None:
                    index[epoch, channel] = peak

    amplitude = np.take_along_axis(trace, index[..., np.newaxis], axis=2)[..., 0]
    return (
        np.where(usable, amplitude, np.nan),
        np.where(usable, times[index], np.nan),
    )
