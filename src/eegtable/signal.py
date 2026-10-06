from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import mne  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from scipy.signal import hilbert

from eegtable._validation import (
    validate_fraction_array,
    validate_names,
    validate_nonempty_shape,
)
from eegtable.bands import Band, check_passband
from eegtable.identity import epoch_row_ids
from eegtable.table import ComputationSpec, RowId

# MNE derives an "auto" FIR length from the narrower of the two transition
# bandwidths, not from fmin: length_seconds = _HAMMING_LENGTH_FACTOR / min(trans).
# These constants mirror mne.filter._length_factors["hamming"] and the "auto"
# transition rule, so _required_filter_length reproduces MNE exactly.
_HAMMING_LENGTH_FACTOR = 3.3
_TRANSITION_FRACTION = 0.25
_MIN_TRANSITION_HZ = 2.0


@runtime_checkable
class TimeSeries(Protocol):
    """What a feature function needs from a time-domain container.

    Structural, not inherited: :class:`Signal` and :class:`BandSignal` satisfy it
    without sharing a base class, so a measure defined on a trace works on a raw
    recording and on a band envelope alike.
    """

    @property
    def times(self) -> npt.NDArray[np.float64]: ...

    @property
    def ch_names(self) -> tuple[str, ...]: ...

    @property
    def sfreq(self) -> float: ...

    @property
    def coverage(self) -> npt.NDArray[np.float64]: ...

    @property
    def row_ids(self) -> tuple[RowId, ...]: ...

    @property
    def computation(self) -> ComputationSpec: ...

    @property
    def amplitude(self) -> npt.NDArray[np.float64]:
        """The real-valued series a time-domain measure reads."""

    @property
    def band(self) -> Band | None:
        """The band this series is restricted to, or None for broadband."""

    @property
    def source(self) -> str:
        """Provenance, recorded on every feature derived from this series."""


@dataclass(frozen=True, eq=False)
class Signal:
    """A broadband time series over epochs, channels and time.

    The raw counterpart to :class:`BandSignal`: no filtering, no analytic signal,
    nothing derived. Measures that read the signal itself rather than a band take
    this.

    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_channels, n_times)
        Signal amplitude.
    times : ndarray, shape (n_times,)
        Time axis in seconds, uniformly spaced at ``1 / sfreq``, relative to
        the epoch origin.
    ch_names : tuple of str
        Channel names, one per channel axis entry.
    sfreq : float
        Sampling frequency in Hz.
    coverage : ndarray, same shape as ``data``
        Fraction of each sample that was finite, in ``[0, 1]``.
    """

    data: npt.NDArray[np.float64]
    times: npt.NDArray[np.float64]
    ch_names: tuple[str, ...]
    sfreq: float
    coverage: npt.NDArray[np.float64]
    row_ids: tuple[RowId, ...]
    computation: ComputationSpec
    passband: tuple[float | None, float | None] | None = None
    """The recording's filter edges, when the source object reported them.

    Carried so a measure that takes this signal *and* a band -- spectral
    connectivity, for one -- can tell that the band lies outside what
    preprocessing left behind. None when the signal came from bare arrays.
    """

    def __post_init__(self) -> None:
        if np.iscomplexobj(self.data):
            raise TypeError("Signal data must be real; use BandSignal for analytic signals.")
        _validate_series(self.data, self.times, self.ch_names, self.coverage, self.sfreq, "data")
        _validate_row_ids(self.row_ids, self.n_epochs)

    @property
    def amplitude(self) -> npt.NDArray[np.float64]:
        """The signal itself."""
        return self.data

    @property
    def band(self) -> Band | None:
        """Always None: a raw signal is broadband."""
        return None

    @property
    def source(self) -> str:
        """Provenance label recorded on derived features."""
        return "signal"

    @property
    def n_epochs(self) -> int:
        """Number of epochs."""
        return int(self.data.shape[0])

    @classmethod
    def from_epochs(
        cls,
        epochs: Any,
        *,
        recording: str,
        picks: str | Sequence[str] = "eeg",
        exclude: str | Sequence[str] = "bads",
    ) -> Signal:
        """Wrap an ``mne.Epochs`` without transforming it.

        Parameters
        ----------
        epochs : mne.Epochs
            Epoched data.
        picks : str or sequence of str, default "eeg"
            MNE channel selection. By default, select good EEG channels.
            Explicit channel names retain every named channel, including bads.
        exclude : str or sequence of str, default "bads"
            Channels excluded from a channel-type selection; use () to retain all.

        Returns
        -------
        Signal
        """
        selected = epochs.copy().pick(picks, exclude=exclude)
        data = np.asarray(selected.get_data())
        return cls(
            data=data,
            times=np.asarray(selected.times, dtype=float),
            ch_names=tuple(selected.ch_names),
            sfreq=float(selected.info["sfreq"]),
            coverage=np.isfinite(data).astype(float),
            row_ids=epoch_row_ids(selected, recording, data.shape[0]),
            computation=ComputationSpec.create("mne.Epochs.get_data", picks=picks, exclude=exclude),
            passband=_passband(selected),
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        data: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        ch_names: tuple[str, ...],
        sfreq: float,
        row_ids: tuple[RowId, ...],
        coverage: npt.NDArray[np.float64] | None = None,
        computation: ComputationSpec | None = None,
    ) -> Signal:
        """Build from arrays.

        Parameters
        ----------
        data : ndarray, shape (n_epochs, n_channels, n_times)
            Signal amplitude.
        times : ndarray, shape (n_times,)
            Time axis in seconds.
        ch_names : tuple of str
            Channel names.
        sfreq : float
            Sampling frequency in Hz.
        coverage : ndarray, optional
            Per-sample coverage. Defaults to where ``data`` is finite.

        Returns
        -------
        Signal
        """
        array = np.asarray(data)
        if np.iscomplexobj(array):
            raise TypeError("Signal data must be real; use BandSignal for analytic signals.")
        array = np.asarray(array, dtype=float)
        _require_real_time_axis(times)
        return cls(
            data=array,
            times=np.asarray(times, dtype=float),
            ch_names=tuple(ch_names),
            sfreq=float(sfreq),
            coverage=np.isfinite(array).astype(float) if coverage is None else coverage,
            row_ids=row_ids,
            computation=(
                ComputationSpec.create("provided-array") if computation is None else computation
            ),
        )


def _passband(source: Any) -> tuple[float | None, float | None] | None:
    """Filter edges from an MNE object's ``info``, or None if it has none."""
    info = getattr(source, "info", None)
    if info is None:
        return None
    try:
        return float(info["highpass"]), float(info["lowpass"])
    except (KeyError, TypeError, ValueError):  # pragma: no cover - exotic info dicts
        return None


def _require_real_time_axis(times: npt.ArrayLike) -> None:
    if np.iscomplexobj(times):
        raise TypeError("times must be real values in seconds.")


def _validate_series(
    values: npt.NDArray[Any],
    times: npt.NDArray[np.float64],
    ch_names: tuple[str, ...],
    coverage: npt.NDArray[np.float64],
    sfreq: float,
    label: str,
) -> None:
    if values.ndim != 3:
        raise ValueError(
            f"{label} must be 3-D (n_epochs, n_channels, n_times), got {values.shape}."
        )
    validate_nonempty_shape(values.shape, label)
    n_channels, n_times = values.shape[1:]
    if len(ch_names) != n_channels:
        raise ValueError(
            f"ch_names has {len(ch_names)} entries but {label} has {n_channels} channels."
        )
    validate_names(ch_names, "ch_names")
    _require_real_time_axis(times)
    if times.ndim != 1 or times.size != n_times:
        raise ValueError(f"times must be 1-D of length {n_times}, got {times.shape}.")
    if not np.isfinite(times).all():
        raise ValueError("times must contain only finite values.")
    if n_times > 1 and not np.all(np.diff(times) > 0):
        raise ValueError("times must be strictly ascending.")
    if coverage.shape != values.shape:
        raise ValueError(f"coverage shape {coverage.shape} does not match {label} {values.shape}.")
    validate_fraction_array(coverage, "coverage")
    if not np.isfinite(sfreq) or sfreq <= 0.0:
        raise ValueError(f"sfreq must be finite and positive, got {sfreq}.")
    if n_times > 1 and not np.allclose(np.diff(times) * sfreq, 1.0, rtol=1e-7, atol=0.0):
        raise ValueError("times must be uniformly spaced at 1 / sfreq.")


def _validate_row_ids(row_ids: tuple[RowId, ...], n_epochs: int) -> None:
    if len(row_ids) != n_epochs:
        raise ValueError(f"row_ids has {len(row_ids)} entries but data has {n_epochs} epochs.")


@dataclass(frozen=True, eq=False)
class BandSignal:
    """A band-filtered analytic signal over epochs, channels and time.

    Only the complex analytic signal is stored; ``envelope``, ``phase`` and
    ``power`` are derived on access via property descriptors to minimize memory overhead.

    Parameters
    ----------
    analytic : ndarray of complex, shape (n_epochs, n_channels, n_times)
        Analytic signal, the bandpass output plus its Hilbert transform.
    times : ndarray, shape (n_times,)
        Time axis in seconds, uniformly spaced at ``1 / sfreq``, relative to
        the epoch origin.
    ch_names : tuple of str
        Channel names, one per channel axis entry.
    band : Band
        The band this signal was filtered to.
    sfreq : float
        Sampling frequency in Hz.
    coverage : ndarray, same shape as ``analytic``
        Fraction of each sample that was finite, in ``[0, 1]``.
    """

    analytic: npt.NDArray[np.complex128]
    times: npt.NDArray[np.float64]
    ch_names: tuple[str, ...]
    band: Band
    sfreq: float
    coverage: npt.NDArray[np.float64]
    row_ids: tuple[RowId, ...]
    computation: ComputationSpec

    def __post_init__(self) -> None:
        _validate_series(
            self.analytic,
            self.times,
            self.ch_names,
            self.coverage,
            self.sfreq,
            "analytic",
        )
        if not np.iscomplexobj(self.analytic):
            raise TypeError("analytic must be complex; a real array has already lost its phase.")
        _validate_row_ids(self.row_ids, self.n_epochs)

    @property
    def envelope(self) -> npt.NDArray[np.float64]:
        """Instantaneous amplitude."""
        return np.abs(self.analytic)

    @property
    def phase(self) -> npt.NDArray[np.float64]:
        """Instantaneous phase in radians; undefined for zero or non-finite signals."""
        valid = np.isfinite(self.analytic) & (self.envelope > 0.0)
        return np.where(valid, np.angle(self.analytic), np.nan)

    @property
    def power(self) -> npt.NDArray[np.float64]:
        """Instantaneous power, the squared envelope."""
        return np.abs(self.analytic) ** 2

    @property
    def amplitude(self) -> npt.NDArray[np.float64]:
        """The envelope: a band signal's amplitude over time."""
        return self.envelope

    @property
    def source(self) -> str:
        """Provenance label recorded on derived features."""
        return "hilbert"

    @property
    def n_epochs(self) -> int:
        """Number of epochs."""
        return int(self.analytic.shape[0])

    @classmethod
    def from_arrays(
        cls,
        *,
        analytic: npt.NDArray[np.complex128],
        times: npt.NDArray[np.float64],
        ch_names: tuple[str, ...],
        band: Band,
        sfreq: float,
        row_ids: tuple[RowId, ...],
        coverage: npt.NDArray[np.float64] | None = None,
        computation: ComputationSpec | None = None,
    ) -> BandSignal:
        """Build from arrays the caller filtered themselves.

        Parameters
        ----------
        analytic : ndarray of complex, shape (n_epochs, n_channels, n_times)
            Analytic signal.
        times : ndarray, shape (n_times,)
            Time axis in seconds.
        ch_names : tuple of str
            Channel names.
        band : Band
            The band the signal was filtered to.
        sfreq : float
            Sampling frequency in Hz.
        coverage : ndarray, optional
            Per-sample coverage. Defaults to where ``analytic`` is finite.

        Returns
        -------
        BandSignal
        """
        array = np.asarray(analytic)
        if coverage is None:
            coverage = np.isfinite(array).astype(float)
        _require_real_time_axis(times)
        return cls(
            analytic=array,
            times=np.asarray(times, dtype=float),
            ch_names=tuple(ch_names),
            band=band,
            sfreq=float(sfreq),
            coverage=coverage,
            row_ids=row_ids,
            computation=(
                ComputationSpec.create("provided-analytic-array")
                if computation is None
                else computation
            ),
        )

    @classmethod
    def from_epochs(
        cls,
        epochs: Any,
        band: Band,
        *,
        recording: str,
        picks: str | Sequence[str] = "eeg",
        exclude: str | Sequence[str] = "bads",
        pad_sec: float = 0.5,
        pad_cycles: float = 3.0,
        n_jobs: int = 1,
    ) -> BandSignal:
        """Bandpass epochs and take their Hilbert transform.

        Parameters
        ----------
        epochs : mne.Epochs
            Epoched data.
        band : Band
            Band to filter to. ``band.fmax`` must be below Nyquist.
        picks : str or sequence of str, default "eeg"
            MNE channel selection. By default, select good EEG channels.
            Explicit channel names retain every named channel, including bads.
        exclude : str or sequence of str, default "bads"
            Channels excluded from a channel-type selection; use () to retain all.
        pad_sec : float, default 0.5
            Minimum reflect padding in seconds.
        pad_cycles : float, default 3.0
            Padding expressed in cycles of ``band.fmin``. The padding actually
            applied is the larger of the two, clamped to one sample short of the
            epoch length.
        n_jobs : int, default 1
            Passed to MNE's filter.

        Returns
        -------
        BandSignal
            With the padding removed, so ``times`` matches ``epochs.times``.
        """
        selected = epochs.copy().pick(picks, exclude=exclude)
        sfreq = float(selected.info["sfreq"])
        # Nyquist itself is excluded, not just frequencies above it: MNE needs a
        # non-zero upper transition band below sfreq/2 to design the filter at all.
        if band.fmax >= sfreq / 2.0:
            raise ValueError(
                f"band {band.name!r} reaches {band.fmax} Hz, at or above the Nyquist "
                f"frequency {sfreq / 2.0} of this recording."
            )
        check_passband(
            band, selected.info["highpass"], selected.info["lowpass"], source="this band signal"
        )

        data = np.asarray(selected.get_data())
        if np.iscomplexobj(data):
            raise TypeError(
                "BandSignal.from_epochs requires real samples; "
                "use BandSignal.from_arrays for analytic signals."
            )
        data = np.asarray(data, dtype=float)
        n_epochs, n_channels, n_times = data.shape
        flat = data.reshape(-1, n_times)

        pad = _padding_samples(pad_sec, pad_cycles, band.fmin, sfreq, n_times)
        padded = np.pad(flat, ((0, 0), (pad, pad)), mode="reflect") if pad else flat

        # Refuse rather than filter with a truncated kernel: a band this narrow cannot
        # be resolved from an epoch this short, and reflecting more of it adds no
        # information. MNE would only warn, and the distortion is invisible downstream.
        required = _required_filter_length(band, sfreq)
        if required > padded.shape[-1]:
            minimum = _minimum_epoch_samples(required, pad_sec, pad_cycles, band.fmin, sfreq)
            raise ValueError(
                f"band {band.name!r} ({band.fmin}-{band.fmax} Hz) needs a {required}-sample "
                f"({required / sfreq:.3f} s) filter, but these {n_times}-sample "
                f"({n_times / sfreq:.3f} s) epochs pad to only {padded.shape[-1]} samples. "
                f"Use epochs of at least {minimum} samples ({minimum / sfreq:.3f} s), raise "
                f"pad_sec or pad_cycles, or choose a band with a wider transition."
            )

        filtered = mne.filter.filter_data(
            padded,
            sfreq,
            l_freq=band.fmin,
            h_freq=band.fmax,
            filter_length="auto",
            n_jobs=n_jobs,
            verbose=False,
        )
        analytic = hilbert(filtered, axis=-1)
        if pad:
            # A copy, not a view: a view would keep the padded array, often twice the
            # epochs' length, alive for as long as the band signal is.
            analytic = np.ascontiguousarray(analytic[:, pad:-pad])

        analytic = analytic.reshape(n_epochs, n_channels, n_times)
        return cls.from_arrays(
            analytic=analytic,
            times=np.asarray(selected.times, dtype=float),
            ch_names=tuple(selected.ch_names),
            band=band,
            sfreq=sfreq,
            # Coverage is taken from the analytic signal, not the input: a single
            # non-finite input sample propagates through the FIR convolution and the
            # Hilbert transform and destroys the whole epoch, so input finiteness
            # would claim a channel is intact when every output sample is NaN.
            coverage=np.isfinite(analytic).astype(float),
            row_ids=epoch_row_ids(selected, recording, n_epochs),
            # n_jobs is left out: it spreads the same filtering over workers without changing
            # a value, and the identity must not split one feature into per-run columns.
            computation=ComputationSpec.create(
                "mne.filter.filter_data+scipy.signal.hilbert",
                band={"name": band.name, "fmin": band.fmin, "fmax": band.fmax},
                pad_sec=pad_sec,
                pad_cycles=pad_cycles,
                filter_length=required,
                picks=picks,
                exclude=exclude,
            ),
        )


def _padding_samples(
    pad_sec: float, pad_cycles: float, fmin: float, sfreq: float, n_times: int
) -> int:
    for name, value in (("pad_sec", pad_sec), ("pad_cycles", pad_cycles)):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}.")
    cycles_sec = pad_cycles / fmin if np.isfinite(fmin) and fmin > 0 and pad_cycles > 0 else 0.0
    seconds = max(pad_sec, cycles_sec)
    if not (np.isfinite(seconds) and seconds > 0) or n_times <= 1:
        return 0
    return max(0, min(int(round(seconds * sfreq)), n_times - 1))


def _transition_bandwidths(band: Band, sfreq: float) -> tuple[float, float]:
    """MNE's ``'auto'`` transition bandwidths for this band, in Hz."""
    lower = (
        min(max(band.fmin * _TRANSITION_FRACTION, _MIN_TRANSITION_HZ), band.fmin)
        if band.fmin > 0
        else np.inf  # fmin of zero is a low-pass: only the upper edge constrains it
    )
    upper = min(max(band.fmax * _TRANSITION_FRACTION, _MIN_TRANSITION_HZ), sfreq / 2.0 - band.fmax)
    return lower, upper


def _required_filter_length(band: Band, sfreq: float) -> int:
    """Samples MNE's ``filter_length="auto"`` will use for this band.

    Verified to agree exactly with ``mne.filter.filter_data`` across sampling
    rates and bands; :func:`_minimum_epoch_samples` inverts it to say how long an
    epoch has to be before that filter fits.
    """
    narrowest = min(_transition_bandwidths(band, sfreq))
    if not (np.isfinite(narrowest) and narrowest > 0.0):
        raise ValueError(
            f"band {band.name!r} ({band.fmin}-{band.fmax} Hz) leaves no transition band "
            f"below the Nyquist frequency {sfreq / 2.0} Hz."
        )
    length = max(int(np.ceil(_HAMMING_LENGTH_FACTOR / narrowest * sfreq)), 1)
    return length + (length - 1) % 2  # firwin needs an odd length


def _minimum_epoch_samples(
    required: int, pad_sec: float, pad_cycles: float, fmin: float, sfreq: float
) -> int:
    """Shortest epoch whose reflect-padded length reaches ``required`` samples.

    Padding is capped at ``n_times - 1`` by ``np.pad(mode="reflect")``, so the
    padded length is ``n_times + 2*pad`` once the cap stops binding and
    ``3*n_times - 2`` while it does. Both regimes are checked and the smaller
    admissible epoch wins.
    """
    cycles_sec = pad_cycles / fmin if np.isfinite(fmin) and fmin > 0 and pad_cycles > 0 else 0.0
    seconds = max(pad_sec, cycles_sec)
    uncapped = int(round(seconds * sfreq)) if np.isfinite(seconds) and seconds > 0 else 0

    candidates = []
    capped = -(-(required + 2) // 3)  # ceil, while pad == n_times - 1
    # The cap binds up to and including n_times == uncapped + 1, where both regimes
    # give the same padded length; stopping one short leaves a gap in which neither
    # candidate qualifies and the answer overshoots.
    if capped <= uncapped + 1:
        candidates.append(capped)
    full = required - 2 * uncapped
    if full >= uncapped + 1:
        candidates.append(full)
    return max(min(candidates, default=required), 2)
