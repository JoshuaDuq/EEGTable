from __future__ import annotations

import hashlib
import inspect
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from eegtable._validation import (
    validate_fraction_array,
    validate_multitaper_bandwidth,
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
        _check_declaration(method, params, np.asarray(spectrum.freqs, dtype=float), spectrum)
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
                # Decimation and its offset change which coefficients enter the mean.
                # MNE retains their times even when it no longer retains the decimation.
                n_times=int(times.size),
                time_axis_sha256=hashlib.sha256(
                    np.ascontiguousarray(times, dtype="<f8").tobytes()
                ).hexdigest(),
                **reduction,
            ),
            passband=_passband(tfr),
        )

    @classmethod
    def welch(
        cls,
        epochs: Any,
        windows: Sequence[Window] | None = None,
        *,
        recording: str,
        fmin: float = 1.0,
        fmax: float = 45.0,
        n_fft: int | None = None,
        n_overlap: int | None = None,
        statistic: WindowStatistic = "mean",
        picks: str | Sequence[str] = "eeg",
        exclude: str | Sequence[str] = "bads",
        n_jobs: int = 1,
    ) -> Spectra:
        """Compute Welch power spectra of epochs, one per window, recording the settings.

        The estimator the batch runner uses, so the same settings name the same
        columns either way. Unlike :meth:`from_spectrum`, nothing has to be declared a
        second time: the settings that produce the spectra are the ones recorded.

        Parameters
        ----------
        epochs : mne.Epochs
            Epoched data. Channels are picked from a copy.
        windows : sequence of Window, optional
            Time windows, each estimated from its own samples. An infinite window
            spans the epochs and is recorded with their actual bounds. The default is
            the whole epoch.
        recording : str
            The identity every row carries.
        fmin, fmax : float
            Frequency range kept, in Hz.
        n_fft : int, optional
            Segment length in samples, shared by every window so their grids align.
            By default two seconds, capped by the shortest window's sample count.
        n_overlap : int, optional
            Segment overlap in samples; half of ``n_fft`` by default.
        statistic : {"mean", "median"}, default "mean"
            How segments are averaged. MNE corrects the median for its bias, so both
            estimate the same density for Gaussian data. A median needs at least
            three segments.
        picks, exclude : str or sequence of str
            MNE channel selection; good EEG channels by default.
        n_jobs : int, default 1
            Passed to MNE. It never changes a value or a column name.

        Returns
        -------
        Spectra
            One PSD per window, in V²/Hz for EEG in volts.
        """
        selected = epochs.copy().pick(picks, exclude=exclude)
        times, sfreq = np.asarray(selected.times, dtype=float), float(selected.info["sfreq"])
        resolved = tuple(_resolved(times, window) for window in (windows or (_WHOLE,)))
        data = np.asarray(selected.get_data(), dtype=float)
        masks = [_within(times, window) for window in resolved]
        shortest = min(int(mask.sum()) for mask in masks)
        segment = n_fft if n_fft is not None else default_n_fft(sfreq, shortest)
        estimates = [
            welch_psd(
                data[:, :, mask],
                sfreq,
                window=window,
                fmin=float(fmin),
                fmax=float(fmax),
                n_fft=segment,
                n_overlap=n_overlap,
                statistic=statistic,
                n_jobs=n_jobs,
            )
            for window, mask in zip(resolved, masks, strict=True)
        ]
        # Exactly the settings a recipe records, so the same estimate has the same name.
        settings: dict[str, object] = {"method": "welch", "fmin": float(fmin), "fmax": float(fmax)}
        if n_fft is not None:
            settings["n_fft"] = int(n_fft)
        if n_overlap is not None:
            settings["n_overlap"] = int(n_overlap)
        if statistic != "mean":
            settings["window_statistic"] = statistic
        return psd_spectra(
            estimates,
            resolved,
            ch_names=tuple(selected.ch_names),
            method="welch",
            settings=settings,
            row_ids=epoch_row_ids(selected, recording, data.shape[0]),
            passband=_passband(selected),
        )

    @classmethod
    def multitaper(
        cls,
        epochs: Any,
        windows: Sequence[Window] | None = None,
        *,
        recording: str,
        fmin: float = 1.0,
        fmax: float = 45.0,
        bandwidth: float = 2.0,
        picks: str | Sequence[str] = "eeg",
        exclude: str | Sequence[str] = "bads",
        n_jobs: int = 1,
    ) -> Spectra:
        """Compute multitaper power spectra of epochs, one per window, recording the settings.

        The batch runner's estimator, with density normalization (``"full"``). The
        grid follows the window length, so windows estimated together must be equally
        long.

        Parameters
        ----------
        epochs : mne.Epochs
            Epoched data. Channels are picked from a copy.
        windows : sequence of Window, optional
            As for :meth:`welch`; the whole epoch by default.
        recording : str
            The identity every row carries.
        fmin, fmax : float
            Frequency range kept, in Hz.
        bandwidth : float, default 2.0
            Full frequency smoothing in Hz. Fixed rather than MNE's
            ``8 / window_length``, which smooths a 1 s window over ±4 Hz, wider than
            the delta or theta band. Refused when it leaves fewer than one taper.
        picks, exclude : str or sequence of str
            MNE channel selection; good EEG channels by default.
        n_jobs : int, default 1
            Passed to MNE. It never changes a value or a column name.

        Returns
        -------
        Spectra
            One PSD per window, in V²/Hz for EEG in volts.
        """
        selected = epochs.copy().pick(picks, exclude=exclude)
        times, sfreq = np.asarray(selected.times, dtype=float), float(selected.info["sfreq"])
        resolved = tuple(_resolved(times, window) for window in (windows or (_WHOLE,)))
        data = np.asarray(selected.get_data(), dtype=float)
        estimates = [
            multitaper_psd(
                data[:, :, _within(times, window)],
                sfreq,
                window=window,
                fmin=float(fmin),
                fmax=float(fmax),
                bandwidth=float(bandwidth),
                n_jobs=n_jobs,
            )
            for window in resolved
        ]
        settings: dict[str, object] = {
            "method": "multitaper",
            "fmin": float(fmin),
            "fmax": float(fmax),
            "bandwidth": float(bandwidth),
        }
        return psd_spectra(
            estimates,
            resolved,
            ch_names=tuple(selected.ch_names),
            method="multitaper",
            settings=settings,
            row_ids=epoch_row_ids(selected, recording, data.shape[0]),
            passband=_passband(selected),
        )

    @classmethod
    def morlet(
        cls,
        epochs: Any,
        windows: Sequence[Window],
        *,
        recording: str,
        freqs: npt.ArrayLike,
        n_cycles: float | npt.ArrayLike,
        decim: int = 1,
        statistic: WindowStatistic = "mean",
        picks: str | Sequence[str] = "eeg",
        exclude: str | Sequence[str] = "bads",
        n_jobs: int = 1,
    ) -> Spectra:
        """Compute Morlet time-frequency power of epochs and reduce it to windows.

        :meth:`from_tfr` on a TFR computed here, so ``n_cycles`` and the original
        sampling rate, which MNE keeps on neither object, are passed for you. Only
        coefficients whose whole wavelet fits a window are averaged; the fraction of
        the window that leaves is :attr:`support`.

        Parameters
        ----------
        epochs : mne.Epochs
            Epoched data, not baseline-corrected. Channels are picked from a copy.
        windows : sequence of Window
            Time windows; an infinite one spans the epochs.
        recording : str
            The identity every row carries.
        freqs : array-like
            Wavelet frequencies in Hz.
        n_cycles : float or array-like
            Cycles per wavelet, one value or one per frequency.
        decim : int, default 1
            Keep every ``decim``-th time point of the TFR.
        statistic : {"mean", "median"}, default "mean"
            Temporal reduction; see :meth:`from_tfr`.
        picks, exclude : str or sequence of str
            MNE channel selection; good EEG channels by default.
        n_jobs : int, default 1
            Passed to MNE. It never changes a value or a column name.

        Returns
        -------
        Spectra
            One wavelet-smoothed density per window, in V²/Hz for EEG in volts.
        """
        selected = epochs.copy().pick(picks, exclude=exclude)
        times = np.asarray(selected.times, dtype=float)
        cycles = np.asarray(n_cycles, dtype=float)
        tfr = selected.compute_tfr(
            "morlet",
            freqs=np.asarray(freqs, dtype=float),
            n_cycles=n_cycles,
            picks="all",
            decim=decim,
            output="power",
            average=False,
            return_itc=False,
            n_jobs=n_jobs,
            verbose=False,
        )
        return cls.from_tfr(
            tfr,
            tuple(_resolved(times, window) for window in windows),
            recording=recording,
            n_cycles=cycles if cycles.ndim else float(cycles),
            sfreq=float(selected.info["sfreq"]),
            statistic=statistic,
        )


# Settings of the run rather than of the estimate. Declared, they would enter the identity
# and split one feature into a column per setting, as n_jobs once did for band signals.
_EXECUTION_KEYS = frozenset({"n_jobs", "verbose"})


def _estimator_keys(method: str) -> frozenset[str] | None:
    import mne  # type: ignore[import-untyped]
    from mne.time_frequency import (  # type: ignore[import-untyped]
        psd_array_multitaper,
        psd_array_welch,
    )

    estimators = {"welch": psd_array_welch, "multitaper": psd_array_multitaper}
    if method not in estimators:
        return None
    functions = (estimators[method], mne.BaseEpochs.compute_psd, mne.io.BaseRaw.compute_psd)
    names = {name for function in functions for name in inspect.signature(function).parameters}
    return frozenset(names - {"self", "x", "sfreq", "method", "method_kw"} - _EXECUTION_KEYS)


def _check_declaration(
    method: str, declared: Mapping[str, object], freqs: npt.NDArray[np.float64], spectrum: Any
) -> None:
    # MNE keeps none of these on the Spectrum, so the declaration is the only record of them.
    # It is hashed into every column, so it has to be checked against what the object shows.
    execution = sorted(_EXECUTION_KEYS & set(declared))
    if execution:
        raise ValueError(
            f"estimator_parameters declare {execution}, which change how the spectrum is "
            "computed but not its values; leave them out, or one feature splits into a column "
            "per setting."
        )
    known = _estimator_keys(method)
    unknown = sorted(set(declared) - known) if known is not None else []
    if unknown:
        raise ValueError(
            f"estimator_parameters declare {unknown}, which MNE's {method} estimator does not "
            f"take. Its parameters: {sorted(known or ())}."
        )
    if freqs.size < 2:
        return
    step = float(np.median(np.diff(freqs)))
    slack = 1e-6 * step
    sfreq = getattr(spectrum, "sfreq", None)
    rate = float(sfreq) if sfreq is not None else None
    n_fft = declared.get("n_fft")
    expected = (
        rate / int(n_fft)
        if method == "welch" and isinstance(n_fft, int | np.integer) and rate is not None
        else None
    )
    if expected is not None and not np.isclose(step, expected, rtol=1e-6):
        raise ValueError(
            f"estimator_parameters declare n_fft = {n_fft}, which gives {expected:g} Hz bins "
            f"at {rate} Hz, but this spectrum's bins are {step:g} Hz apart."
        )
    # MNE keeps the grid's bins inside [fmin, fmax], so the first and last kept bins lie within
    # one step of the declared bounds; a bound at or past Nyquist keeps the grid's own end.
    fmin, fmax = declared.get("fmin"), declared.get("fmax")
    nyquist = rate / 2.0 if rate is not None else np.inf
    if (
        isinstance(fmin, int | float)
        and np.isfinite(fmin)
        and not fmin - slack <= freqs[0] < fmin + step + slack
    ):
        raise ValueError(
            f"estimator_parameters declare fmin = {fmin} Hz, but this spectrum starts at "
            f"{freqs[0]:g} Hz."
        )
    if (
        isinstance(fmax, int | float)
        and np.isfinite(fmax)
        and fmax < nyquist
        and not fmax - step - slack < freqs[-1] <= fmax + slack
    ):
        raise ValueError(
            f"estimator_parameters declare fmax = {fmax} Hz, but this spectrum ends at "
            f"{freqs[-1]:g} Hz."
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


def require_within_axis(
    times: npt.NDArray[np.float64], window: Window, *, tolerance: float
) -> None:
    # Clipped, the column would keep naming the requested span while measuring a shorter
    # one. An infinite bound says "to the edge" on purpose, so it is never refused.
    first, last = float(times[0]), float(times[-1])
    if (np.isfinite(window.tmin) and window.tmin < first - tolerance) or (
        np.isfinite(window.tmax) and window.tmax > last + tolerance
    ):
        raise ValueError(
            f"window {window.name!r} ({window.tmin}, {window.tmax}) reaches outside the data, "
            f"which spans ({first:.3f}, {last:.3f}) s. Its column would name a span it never "
            "measured; narrow the window, or use an infinite bound to run to the edge."
        )


def _step(times: npt.NDArray[np.float64]) -> float:
    return float(np.median(np.diff(times))) if times.size > 1 else 0.0


def sample_period(times: npt.NDArray[np.float64]) -> float:
    """How far a window may reach past the data: one sample, and float slack."""
    # A sample stands for the interval up to the next one, so epochs whose last sample is at
    # 29.99 s hold data to 30 s, and the window a user writes for them, 0 to 30 s, is whole.
    return _step(times) * (1.0 + 1e-6)


_WHOLE = Window("all", -math.inf, math.inf)
_WELCH_SEGMENT_SEC = 2.0
_WELCH_TAPER = "hann"
_WELCH_BLOCK_BYTES = 9e6


def _resolved(times: npt.NDArray[np.float64], window: Window) -> Window:
    # As the runner measures it: an infinite window spans these epochs and records their
    # bounds, and a finite one must lie within them.
    if math.isinf(window.tmin) and math.isinf(window.tmax):
        return Window(window.name, float(times[0]), float(times[-1]))
    require_within_axis(times, window, tolerance=sample_period(times))
    return window


def _within(times: npt.NDArray[np.float64], window: Window) -> npt.NDArray[np.bool_]:
    mask: npt.NDArray[np.bool_] = (times >= window.tmin) & (times <= window.tmax)
    return mask


def default_n_fft(sfreq: float, shortest: int) -> int:
    """Default Welch segment: two seconds, capped by the shortest window it must fit."""
    return min(int(round(_WELCH_SEGMENT_SEC * sfreq)), int(shortest))


def welch_psd(
    data: npt.NDArray[np.float64],
    sfreq: float,
    *,
    window: Window,
    fmin: float,
    fmax: float,
    n_fft: int,
    n_overlap: int | None,
    statistic: WindowStatistic,
    n_jobs: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Welch PSD of one window's samples, shaped (epochs, channels, freqs)."""
    from mne.time_frequency import psd_array_welch

    if n_fft > data.shape[-1]:
        raise ValueError(
            f"n_fft = {n_fft} is longer than window {window.name!r}, which holds "
            f"{data.shape[-1]} samples; lower n_fft or widen the window."
        )
    overlap = n_fft // 2 if n_overlap is None else n_overlap
    # An overlap of n_fft or more is left for MNE to refuse.
    if statistic == "median" and overlap < n_fft:
        segments = 1 + (data.shape[-1] - n_fft) // (n_fft - overlap)
        if segments < 3:
            raise ValueError(
                f"window {window.name!r} holds {segments} Welch segment(s) of n_fft = "
                f"{n_fft}, and a median of fewer than 3 segments is their mean; lower "
                "n_fft or widen the window."
            )
    resolution = sfreq / n_fft
    # MNE estimates an input above 10 MB one row at a time in a Python loop, and one below
    # it in a single vectorized call, so blocks of epochs under that size give the same
    # numbers far faster. Input with NaN goes in whole: MNE reads NaN at the same samples in
    # every row as rejected spans and NaN in some rows as broken channels, and a block
    # boundary could change which of the two it sees.
    rows = max(1, int(_WELCH_BLOCK_BYTES // max(1, data[:1].nbytes)))
    blocks = (
        [data]
        if np.isnan(data).any()
        else [data[start : start + rows] for start in range(0, data.shape[0], rows)]
    )
    estimates = [
        psd_array_welch(
            block,
            sfreq,
            fmin=max(0.0, fmin - resolution),
            fmax=min(sfreq / 2.0, fmax + resolution),
            n_fft=n_fft,
            n_overlap=overlap,
            window=_WELCH_TAPER,
            # MNE divides the median by its bias for the segment count, so for
            # Gaussian data both statistics estimate the same density.
            average=statistic,
            n_jobs=n_jobs,
            verbose=False,
        )
        for block in blocks
    ]
    psd = np.concatenate([np.asarray(block_psd, dtype=float) for block_psd, _ in estimates])
    return psd, np.asarray(estimates[0][1], dtype=float)


def multitaper_psd(
    data: npt.NDArray[np.float64],
    sfreq: float,
    *,
    window: Window,
    fmin: float,
    fmax: float,
    bandwidth: float,
    n_jobs: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Multitaper PSD of one window's samples, density-normalized."""
    from mne.time_frequency import psd_array_multitaper

    resolution = sfreq / data.shape[-1]
    validate_multitaper_bandwidth(
        bandwidth, sfreq, data.shape[-1], window.name, "spectra.bandwidth"
    )
    psd, freqs = psd_array_multitaper(
        data,
        sfreq,
        fmin=max(0.0, fmin - resolution),
        fmax=min(sfreq / 2.0, fmax + resolution),
        bandwidth=bandwidth,
        normalization="full",
        n_jobs=n_jobs,
        verbose=False,
    )
    return np.asarray(psd, dtype=float), np.asarray(freqs, dtype=float)


def psd_spectra(
    estimates: Sequence[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    windows: Sequence[Window],
    *,
    ch_names: tuple[str, ...],
    method: str,
    settings: Mapping[str, object],
    row_ids: tuple[RowId, ...],
    passband: tuple[float | None, float | None] | None,
) -> Spectra:
    """Assemble per-window PSD estimates, which must share one frequency grid."""
    freqs = estimates[0][1]
    for (_, other), window in zip(estimates[1:], windows[1:], strict=True):
        if not np.array_equal(other, freqs):
            raise ValueError(
                f"window {window.name!r} yields a different frequency grid from "
                f"{windows[0].name!r}. Multitaper grids follow the window length, so windows "
                "measured together must be equally long; or use welch or morlet."
            )
    data = np.stack([psd for psd, _ in estimates], axis=2)
    return Spectra(
        data=data,
        freqs=freqs,
        ch_names=ch_names,
        windows=tuple(windows),
        coverage=np.isfinite(data).astype(float),
        source=method,
        representation="psd",
        support=np.ones(data.shape, dtype=float),
        row_ids=row_ids,
        computation=ComputationSpec.create(
            method,
            normalization="full" if method == "multitaper" else "density",
            settings=dict(settings),
        ),
        passband=passband,
    )


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
    # One step of this axis: decimated, the TFR keeps every n-th sample, so its last point can
    # fall up to one step before the end of the epochs it came from.
    require_within_axis(times, window, tolerance=sample_period(times))
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
