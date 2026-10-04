from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from eegtable._validation import (
    validate_fraction_array,
    validate_names,
    validate_nonempty_shape,
)
from eegtable.identity import epoch_row_ids
from eegtable.signal import _passband
from eegtable.table import ComputationSpec, RowId

WindowStatistic = Literal["mean", "median"]


@dataclass(frozen=True)
class Window:
    """A named time window in seconds relative to the epoch origin.

    Parameters
    ----------
    name : str
        Window label, used in feature names.
    tmin, tmax : float
        Bounds in seconds. Infinite bounds denote the whole segment.
    """

    name: str
    tmin: float
    tmax: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Window name must be a non-empty string.")
        # As for Band: 0 and 0.0 are different canonical JSON tokens, so an int-bounded
        # window would name its columns differently from the identical float-bounded one.
        object.__setattr__(self, "tmin", float(self.tmin))
        object.__setattr__(self, "tmax", float(self.tmax))
        if not self.tmin < self.tmax:
            raise ValueError(
                f"Window {self.name!r} requires tmin < tmax, got {self.tmin} >= {self.tmax}."
            )


def trapezoid_weights(freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Trapezoidal integration weights for a frequency axis.

    Weighting by bin width rather than averaging bins is what makes a band value
    an estimate of the integral over the band. It matters because a log-spaced
    grid samples low frequencies far more densely than high ones, so a plain
    mean would weight the bottom of every band too heavily.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Ascending frequency axis.

    Returns
    -------
    ndarray, shape (n_freqs,)
        Weights summing to ``freqs[-1] - freqs[0]``.
    """
    f = np.asarray(freqs, dtype=float)
    if f.size <= 1:
        return np.ones(f.size, dtype=float)
    weights = np.zeros(f.size, dtype=float)
    weights[0] = (f[1] - f[0]) / 2.0
    weights[1:-1] = (f[2:] - f[:-2]) / 2.0
    weights[-1] = (f[-1] - f[-2]) / 2.0
    return weights


def gradient_weights(freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Central-difference bin widths for a frequency axis.

    Identical to :func:`trapezoid_weights` in the interior and twice its value
    at each endpoint. Centroid, bandwidth, and spectral-edge descriptors use
    these weights; spectral entropy uses equal bin weights. Band power uses
    piecewise-linear quadrature over exact numerical boundaries.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Ascending frequency axis.

    Returns
    -------
    ndarray, shape (n_freqs,)
        Bin widths.
    """
    f = np.asarray(freqs, dtype=float)
    if f.size <= 1:
        return np.ones(f.size, dtype=float)
    out: npt.NDArray[np.float64] = np.asarray(np.gradient(f), dtype=np.float64)
    return out


def band_integration_weights(
    freqs: npt.NDArray[np.float64], fmin: float, fmax: float
) -> npt.NDArray[np.float64]:
    """Piecewise-linear quadrature weights over exact frequency bounds.

    The bounds are closed: the integral runs up to and including ``fmax``, with
    an interpolated contribution from the bin above it. That is not the
    half-open convention :meth:`eegtable.Band.mask` uses to assign bins to bands,
    and :meth:`~eegtable.Band.mask` explains where the two meet. No area is
    double counted between adjacent bands -- the shared boundary splits the
    edge bin between them -- but the supports are not identical.
    """
    axis = np.asarray(freqs, dtype=float)
    if axis.size < 2 or axis[0] > fmin or axis[-1] < fmax:
        raise ValueError(
            f"frequency axis ({axis[0]}, {axis[-1]}) does not span the complete "
            f"band [{fmin}, {fmax}]."
        )
    weights = np.zeros(axis.size)
    for index, (left, right) in enumerate(zip(axis[:-1], axis[1:], strict=True)):
        start = max(left, fmin)
        stop = min(right, fmax)
        if stop <= start:
            continue
        width = right - left
        start_fraction = (start - left) / width
        stop_fraction = (stop - left) / width
        square_difference = stop_fraction**2 - start_fraction**2
        weights[index] += width * (stop_fraction - start_fraction - square_difference / 2.0)
        weights[index + 1] += width * square_difference / 2.0
    return weights


@dataclass(frozen=True, eq=False)
class Spectra:
    """Power spectra over epochs, channels and time windows.

    This is the single spectral container for the library. It is built from an
    MNE ``Spectrum`` or ``EpochsTFR`` while retaining whether values are a PSD or
    time-frequency power, so dimensionally incompatible reductions are rejected.

    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_channels, n_windows, n_freqs)
        Power.
    freqs : ndarray, shape (n_freqs,)
        Strictly ascending frequency axis in Hz.
    ch_names : tuple of str
        Channel names, one per channel axis entry.
    windows : tuple of Window
        Time windows, one per window axis entry.
    coverage : ndarray, same shape as ``data``
        Fraction of support-valid coefficients that were finite, per frequency.
    support : ndarray, same shape as ``data``
        Fraction of the requested window whose coefficients have complete
        temporal support inside that window. This is distinct from numerical
        finiteness: clean data can have full ``coverage`` but partial ``support``.
    source : str
        Provenance, e.g. ``"morlet"``, ``"multitaper"``, ``"welch"``.
    representation : {"psd", "time_frequency_power", "aperiodic_ratio"}
        A density per Hz, wavelet time-frequency power, or a dimensionless
        ratio to the aperiodic fit.
    """

    data: npt.NDArray[np.float64]
    freqs: npt.NDArray[np.float64]
    ch_names: tuple[str, ...]
    windows: tuple[Window, ...]
    coverage: npt.NDArray[np.float64]
    source: str
    representation: Literal["psd", "time_frequency_power", "aperiodic_ratio"]
    support: npt.NDArray[np.float64]
    row_ids: tuple[RowId, ...]
    computation: ComputationSpec
    flags: Mapping[str, npt.NDArray[np.bool_]] = field(default_factory=dict)
    passband: tuple[float | None, float | None] | None = None
    """The recording's filter edges, when the source object reported them.

    Carried so a measure can tell that a requested band lies outside what
    preprocessing left behind. None when the spectra came from bare arrays.
    """

    def __post_init__(self) -> None:
        if self.data.ndim != 4:
            raise ValueError(
                "data must be 4-D (n_epochs, n_channels, n_windows, n_freqs), "
                f"got {self.data.shape}."
            )
        validate_nonempty_shape(self.data.shape, "spectral data")
        if np.any(np.isfinite(self.data) & (self.data < 0.0)):
            raise ValueError("spectral data must not contain negative power.")
        if self.representation not in ("psd", "time_frequency_power", "aperiodic_ratio"):
            raise ValueError(f"unknown spectral representation {self.representation!r}.")
        n_channels, n_windows, n_freqs = self.data.shape[1:]
        if len(self.ch_names) != n_channels:
            raise ValueError(
                f"ch_names has {len(self.ch_names)} entries but data has {n_channels} channels."
            )
        validate_names(self.ch_names, "ch_names")
        if len(self.windows) != n_windows:
            raise ValueError(
                f"windows has {len(self.windows)} entries but data has {n_windows} windows."
            )
        validate_names(tuple(window.name for window in self.windows), "window names")
        if self.freqs.ndim != 1 or self.freqs.size != n_freqs:
            raise ValueError(
                f"freqs must be 1-D of length {n_freqs}, got shape {self.freqs.shape}."
            )
        if not np.isfinite(self.freqs).all():
            raise ValueError("freqs must contain only finite values.")
        if np.any(self.freqs < 0.0):
            raise ValueError("freqs must be non-negative.")
        if n_freqs > 1 and not np.all(np.diff(self.freqs) > 0):
            raise ValueError("freqs must be strictly ascending.")
        if self.coverage.shape != self.data.shape:
            raise ValueError(
                f"coverage shape {self.coverage.shape} does not match data {self.data.shape}."
            )
        validate_fraction_array(self.coverage, "coverage")
        if self.support.shape != self.data.shape:
            raise ValueError(
                f"support shape {self.support.shape} does not match data {self.data.shape}."
            )
        validate_fraction_array(self.support, "support")
        if len(self.row_ids) != self.n_epochs:
            raise ValueError(
                f"row_ids has {len(self.row_ids)} entries but data has {self.n_epochs} epochs."
            )
        for name, flag in self.flags.items():
            if flag.shape != self.data.shape[:3]:
                raise ValueError(
                    f"flag {name!r} shape {flag.shape} does not match spectral cells "
                    f"{self.data.shape[:3]}."
                )

    @property
    def n_epochs(self) -> int:
        """Number of epochs."""
        return int(self.data.shape[0])

    @classmethod
    def from_spectrum(
        cls, spectrum: Any, *, recording: str, estimator_parameters: dict[str, object]
    ) -> Spectra:
        """Build from an MNE ``Spectrum`` or ``EpochsSpectrum``.

        Parameters
        ----------
        spectrum : mne.time_frequency.Spectrum or EpochsSpectrum
            A computed power spectrum. A continuous ``Spectrum`` gains a
            leading epoch axis of length 1. Every channel retained in the
            input is preserved, including explicitly selected bad channels.
        estimator_parameters : dict
            Parameters used to compute the spectrum. Multitaper spectra must
            be computed with ``normalization="full"`` and declare it here;
            MNE's default ``"length"`` does not provide density in V²/Hz.

        Returns
        -------
        Spectra
            With a single window named ``"all"`` spanning the whole segment.
        """
        data = np.asarray(spectrum.get_data(picks=spectrum.ch_names))
        if np.iscomplexobj(data):
            raise ValueError(
                "Spectrum contains complex coefficients, not power; use output='power'."
            )
        data = np.asarray(data, dtype=float)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if data.ndim != 3:
            raise ValueError(
                f"expected a (channels, freqs) or (epochs, channels, freqs) spectrum, "
                f"got shape {data.shape}."
            )
        data = data[:, :, np.newaxis, :]
        params = dict(estimator_parameters)
        # MNE's array spectra report "unknown"; only then does the declaration decide.
        actual = getattr(spectrum, "method", "unknown")
        method = str(params.pop("method", actual))
        # The density check below keys on the method, so a known one cannot be overridden.
        if actual != "unknown" and method != actual:
            raise ValueError(
                f"estimator_parameters declare method {method!r}, but the spectrum was "
                f"computed with {actual!r}."
            )
        if method == "multitaper" and params.get("normalization") != "full":
            raise ValueError(
                'Multitaper PSD requires normalization="full" when computing the '
                "spectrum and in estimator_parameters to establish density units."
            )
        return cls(
            data=data,
            freqs=np.asarray(spectrum.freqs, dtype=float),
            ch_names=tuple(spectrum.ch_names),
            windows=(Window("all", -np.inf, np.inf),),
            coverage=np.isfinite(data).astype(float),
            source=method,
            representation="psd",
            support=np.ones(data.shape, dtype=float),
            row_ids=epoch_row_ids(spectrum, recording, data.shape[0]),
            computation=ComputationSpec.create(
                method,
                **params,
                # MNE keeps none of the estimator's own keyword arguments on the
                # Spectrum (no n_per_seg, n_overlap or window as of 1.13), so
                # anything the caller leaves out of estimator_parameters is lost.
                # The grid and the sampling rate are on the object, and recording
                # them means two runs that differ in resolution, range or rate
                # cannot land on the same column even if the declaration matches.
                **_axis_identity(np.asarray(spectrum.freqs, dtype=float), spectrum),
            ),
            passband=_passband(spectrum),
        )

    @classmethod
    def from_tfr(
        cls,
        tfr: Any,
        windows: Sequence[Window],
        *,
        recording: str,
        n_cycles: float | npt.NDArray[np.float64],
        sfreq: float,
        statistic: WindowStatistic = "mean",
    ) -> Spectra:
        """Build from an MNE ``EpochsTFR`` by averaging over time windows.

        MNE's Morlet wavelets have energy 2 (norm sqrt(2)). For stationary
        signals and predominantly positive-frequency wavelets, coefficient power
        divided by the original sampling rate has the scale of a one-sided,
        wavelet-smoothed density. This conversion is applied here and reported
        in V²/Hz for EEG input in volts. Discrete sampling, low cycle counts,
        smoothing, and time reduction affect agreement with Welch or multitaper
        PSD estimates of the same recording.

        Parameters
        ----------
        tfr : mne.time_frequency.EpochsTFR
            Real-valued power, not baseline-corrected. Every channel retained
            in the input is preserved, including explicitly selected bad channels.
        windows : sequence of Window
            Time windows to average over, inclusive of both bounds.
        n_cycles : float or ndarray
            Morlet cycle count used to compute the TFR. Required because MNE
            does not store it on the TFR object and safe temporal attribution
            cannot be inferred without it.
        sfreq : float
            Sampling rate of the data the TFR was computed from, in Hz. Required
            for the same reason: a TFR computed with ``decim`` reports the
            decimated rate as its own, and MNE keeps no record of the original,
            so reading it off the object would scale a decimated TFR wrongly.
        statistic : {"mean", "median"}, default "mean"
            Temporal reduction of supported coefficients. The median is divided
            by ln 2, calibrating the population median to the mean of an
            exponential power distribution. This calibration does not hold for
            every signal: constant coefficient power is multiplied by 1/ln 2.
            Mean and corrected-median estimates can therefore differ across
            windows or bands with different power distributions.

        Returns
        -------
        Spectra
            One spectrum per window, in V²/Hz.
        """
        if statistic not in ("mean", "median"):
            raise ValueError(f"statistic must be 'mean' or 'median', got {statistic!r}.")
        if getattr(tfr, "baseline", None) is not None:
            raise ValueError(
                "this TFR is already baseline-corrected "
                f"(baseline={tfr.baseline!r}); normalizing it again is meaningless. "
                "Pass an uncorrected TFR and use the baseline argument of the feature function."
            )
        if not windows:
            raise ValueError("from_tfr requires at least one window.")
        method = str(getattr(tfr, "method", "unknown"))
        # The support mask and the recorded computation model MNE's Morlet wavelets; a TFR
        # that says it was computed otherwise has another temporal support.
        if method not in ("morlet", "unknown"):
            raise ValueError(
                f"from_tfr models Morlet wavelets, but this TFR was computed with {method!r}; "
                "compute it with method='morlet'."
            )

        data = np.asarray(tfr.get_data(picks=tfr.ch_names))
        if np.iscomplexobj(data):
            raise ValueError("from_tfr requires real power; got a complex TFR.")
        data = np.asarray(data, dtype=float)
        if data.ndim != 4:
            raise ValueError(
                "expected an EpochsTFR of shape (epochs, channels, freqs, times), "
                f"got shape {data.shape}."
            )

        if not np.isfinite(sfreq) or sfreq <= 0.0:
            raise ValueError(f"sfreq must be finite and positive, got {sfreq}.")

        times = np.asarray(tfr.times, dtype=float)
        freqs = np.asarray(tfr.freqs, dtype=float)
        per_window = [
            _reduce_window(data, times, freqs, window, n_cycles, statistic) for window in windows
        ]
        # Recorded only when it departs from the mean, so columns computed before the
        # option existed keep their names.
        reduction = {} if statistic == "mean" else {"window_statistic": statistic}
        return cls(
            # Energy-2 wavelets: E|coefficient|^2 = sfreq * one-sided PSD, so dividing is what
            # turns MNE's rate-dependent number into a density. Done on the reduced windows,
            # since a mean or median commutes with a positive scale, rather than on a copy
            # of the whole TFR.
            data=np.stack([values for values, _, _ in per_window], axis=2) / float(sfreq),
            freqs=freqs,
            ch_names=tuple(tfr.ch_names),
            windows=tuple(windows),
            coverage=np.stack([cover for _, cover, _ in per_window], axis=2),
            source=method,
            representation="time_frequency_power",
            support=np.stack([support for _, _, support in per_window], axis=2),
            row_ids=epoch_row_ids(tfr, recording, data.shape[0]),
            computation=ComputationSpec.create(
                "morlet",
                n_cycles=np.asarray(n_cycles, dtype=float),
                frequencies_hz=freqs,
                # Recorded because it changes every value: a table built before
                # this scaling existed must not land on the same column.
                scaling="power_divided_by_sfreq",
                sfreq_hz=float(sfreq),
                **reduction,
            ),
            passband=_passband(tfr),
        )


def _axis_identity(freqs: npt.NDArray[np.float64], spectrum: Any) -> dict[str, object]:
    """Exact identity of a frequency axis, in four fields rather than the whole grid.

    The axis itself can be a thousand numbers and would be repeated in every
    column's provenance and in the sidecar. The digest is the same identity at
    constant size; the three readable fields beside it are so a human reading the
    provenance can still see what grid was used.
    """
    sfreq = getattr(spectrum, "sfreq", None)
    return {
        "n_frequencies": int(freqs.size),
        "frequency_first_hz": float(freqs[0]) if freqs.size else None,
        "frequency_last_hz": float(freqs[-1]) if freqs.size else None,
        "frequency_axis_sha256": hashlib.sha256(
            np.ascontiguousarray(freqs, dtype=np.float64).tobytes()
        ).hexdigest()[:16],
        "sfreq_hz": float(sfreq) if sfreq is not None else None,
    }


def support_restricted_mask(
    times: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    window: Window,
    n_cycles: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Per-frequency time mask of coefficients a window can account for.

    MNE constructs a Morlet wavelet to five Gaussian standard deviations in
    either direction. With ``sigma_t = n_cycles / (2*pi*f)``, its temporal
    half-support is therefore ``5*n_cycles / (2*pi*f)`` seconds. Only
    coefficients whose complete wavelet lies inside both the window and the
    available time range are attributable to it.

    Parameters
    ----------
    times : ndarray, shape (n_times,)
        Time axis in seconds.
    freqs : ndarray, shape (n_freqs,)
        Frequency axis in Hz.
    window : Window
        The window to restrict to.
    n_cycles : float or ndarray
        Finite positive cycle count, scalar or one value per frequency.

    Returns
    -------
    ndarray of bool, shape (n_freqs, n_times)
        True where the coefficient is attributable to the window. A row is all
        False when no coefficient at that frequency fits.
    """
    f = np.asarray(freqs, dtype=float)
    if f.ndim != 1 or f.size == 0 or not np.isfinite(f).all() or np.any(f <= 0.0):
        raise ValueError("freqs must be a non-empty 1-D array of finite positive values.")
    cycles = np.asarray(n_cycles, dtype=float)
    if cycles.ndim != 0 and cycles.shape != f.shape:
        raise ValueError("n_cycles must be a scalar or one value per frequency.")
    if not np.isfinite(cycles).all() or np.any(cycles <= 0.0):
        raise ValueError("n_cycles must contain finite positive values.")
    cycles = np.broadcast_to(cycles, f.shape)
    half_support = 5.0 * cycles / (2.0 * np.pi * f)
    axis = np.asarray(times, dtype=float)
    lower = max(window.tmin, float(axis[0])) + half_support
    upper = min(window.tmax, float(axis[-1])) - half_support
    t = axis[np.newaxis, :]
    return (t >= lower[:, np.newaxis]) & (t <= upper[:, np.newaxis])


def _reduce_window(
    data: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    window: Window,
    n_cycles: float | npt.NDArray[np.float64],
    statistic: WindowStatistic = "mean",
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    requested = (times >= window.tmin) & (times <= window.tmax)
    if not requested.any():
        raise ValueError(
            f"window {window.name!r} ({window.tmin}, {window.tmax}) selects no samples "
            f"from a time axis spanning ({times[0]}, {times[-1]})."
        )
    mask_2d = support_restricted_mask(times, freqs, window, n_cycles)
    if not mask_2d.any():
        raise ValueError(
            f"window {window.name!r} ({window.tmin}, {window.tmax}) retains no coefficients "
            f"at any frequency once Morlet support is accounted for. Widen the window or "
            f"lower n_cycles."
        )
    # Supported coefficients lie inside the requested span, which is contiguous on an
    # ascending axis: a slice is a view, where a boolean index would copy the whole TFR.
    first, last = np.flatnonzero(requested)[[0, -1]]
    span = slice(first, last + 1)
    selected = np.where(mask_2d[:, span], data[..., span], np.nan)
    finite = np.isfinite(selected)
    selected[~finite] = np.nan  # an infinity is missing, not a value for nanmean to average
    n_selected = mask_2d.sum(axis=1).astype(float)
    with warnings.catch_warnings():
        # A frequency whose support never fits the window is an all-NaN slice by
        # design; the finite.any() guard already discards its mean. np.errstate
        # does not suppress this one, because nanmean raises it through warnings.
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        if statistic == "median":
            # Gaussian power is exponential at each time point, and an exponential's
            # median is ln 2 of its mean.
            reduced = np.nanmedian(selected, axis=3) / np.log(2.0)
        else:
            reduced = np.nanmean(selected, axis=3)
        values = np.where(finite.any(axis=3), reduced, np.nan)
    coverage = finite.sum(axis=3) / np.where(n_selected > 0, n_selected, np.nan)
    support_by_frequency = n_selected / float(requested.sum())
    support = np.broadcast_to(support_by_frequency, values.shape)
    return values, np.nan_to_num(coverage, nan=0.0), support
