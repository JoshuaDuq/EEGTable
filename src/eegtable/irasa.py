"""IRASA decomposition and features delegated to NeuroDSP."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from numbers import Real

import numpy as np
import numpy.typing as npt

from eegtable._expand import check_signals, expand_signal
from eegtable.bands import Band, passband_fraction
from eegtable.signal import Signal
from eegtable.spectra import Window, band_integration_weights
from eegtable.table import FeatureTable, concat

_DEFAULT_FACTORS = tuple(np.round(np.arange(1.1, 1.95, 0.05), 4))


@dataclass(frozen=True)
class _Decomposition:
    freqs: npt.NDArray[np.float64]
    aperiodic: npt.NDArray[np.float64]
    periodic: npt.NDArray[np.float64]
    offset: npt.NDArray[np.float64]
    slope: npt.NDArray[np.float64]


def _validate_factors(hset: Sequence[float]) -> npt.NDArray[np.float64]:
    factors = np.round(np.asarray(hset, dtype=float), 4)
    if (
        factors.ndim != 1
        or factors.size == 0
        or not np.isfinite(factors).all()
        or np.any((factors <= 1) | (factors >= 2))
        or np.unique(factors).size != factors.size
    ):
        raise ValueError(
            "hset must contain distinct finite noninteger factors strictly between 1 and 2."
        )
    return factors


def _integration_slice(freqs: npt.NDArray[np.float64], fit_band: Band) -> slice:
    left = int(np.searchsorted(freqs, fit_band.fmin, side="right")) - 1
    right = int(np.searchsorted(freqs, fit_band.fmax, side="left")) + 1
    return slice(left, right)


def _validate_signal(signal: Signal, fit_band: Band, largest_factor: float, nperseg: int) -> None:
    if not isinstance(signal, Signal):
        raise TypeError("irasa requires broadband Signal inputs.")
    if fit_band.fmax * largest_factor >= signal.sfreq / 2:
        raise ValueError("fit_range times the largest hset factor must lie below Nyquist.")
    freqs = np.asarray(np.fft.rfftfreq(nperseg, d=1 / signal.sfreq), dtype=np.float64)
    support = freqs[_integration_slice(freqs, fit_band)]
    expanded = Band(
        "resampling_support",
        float(support[0]) / largest_factor,
        float(support[-1]) * largest_factor,
    )
    if expanded.fmax >= signal.sfreq / 2:
        raise ValueError("IRASA bracketing bins and resampling support must lie below Nyquist.")
    if signal.passband is not None and passband_fraction(expanded, *signal.passband) < 1:
        raise ValueError(
            "the expanded IRASA resampling frequency range must lie within the passband."
        )


def _decompose(
    trace: npt.NDArray[np.float64],
    sfreq: float,
    nperseg: int,
    factors: npt.NDArray[np.float64],
    fit_band: Band,
) -> _Decomposition:
    from neurodsp.aperiodic import compute_irasa, fit_irasa

    if trace.shape[-1] < np.ceil(nperseg * factors.max()):
        raise ValueError(
            "each IRASA window must support a complete Welch segment after downsampling."
        )
    if not np.isfinite(trace).all():
        raise ValueError("irasa requires finite samples; gaps cannot be deleted or interpolated.")
    if np.any(np.ptp(trace, axis=-1) == 0):
        raise ValueError("irasa requires nonconstant traces.")
    freqs, aperiodic, periodic = compute_irasa(
        trace,
        sfreq,
        f_range=None,
        hset=factors,
        thresh=None,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        avg_type="mean",
        window="hann",
    )
    support = _integration_slice(freqs, fit_band)
    freqs = freqs[support]
    aperiodic = aperiodic[..., support]
    periodic = periodic[..., support]
    if (
        not np.isfinite(aperiodic).all()
        or np.any(aperiodic <= 0)
        or not np.isfinite(periodic).all()
    ):
        raise ValueError("IRASA did not produce finite spectra with positive aperiodic power.")
    mask = fit_band.mask(freqs)
    if mask.sum() < 5:
        raise ValueError("IRASA fit_range must contain at least five frequency bins.")
    offset = np.empty(trace.shape[:2])
    slope = np.empty(trace.shape[:2])
    for cell in np.ndindex(offset.shape):
        offset[cell], slope[cell] = fit_irasa(freqs[mask], aperiodic[cell][mask])
    if not np.isfinite(offset).all() or not np.isfinite(slope).all():
        raise ValueError("IRASA did not produce finite log-log fit parameters.")
    return _Decomposition(freqs, aperiodic, periodic, offset, slope)


def irasa(
    series: Sequence[Signal],
    *,
    windows: Sequence[Window],
    bands: Sequence[Band],
    fit_range: tuple[float, float] = (2.0, 40.0),
    hset: Sequence[float] = _DEFAULT_FACTORS,
    segment_seconds: float = 2.0,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """IRASA aperiodic offset/slope and signed component band integrals.

    Uses NeuroDSP's irregular resampling, arithmetic-mean Welch PSD with Hann
    windows and 50 percent overlap, and its log-log ``fit_irasa`` regression.
    Periodic residuals retain their sign; no clipping or thresholding occurs.
    Broadband finite inputs must support the resampled frequency range and a
    full Welch segment at every factor. The fit range is half-open; component
    integrals reach both exact band boundaries. Requires ``eegtable[irasa]``.

    Parameters
    ----------
    series : sequence of Signal
        Broadband epochs, finite and nonconstant. Each window must hold at least
        ``segment_seconds`` times the largest factor in ``hset``.
    windows : sequence of Window
        Analysis windows, each decomposed separately.
    bands : sequence of Band
        Bands, with unique names inside ``fit_range``, for the aperiodic and
        periodic component integrals.
    fit_range : tuple of float, default (2.0, 40.0)
        Frequency range of the log-log aperiodic fit, in Hz. The lower edge must
        be above zero, at least five bins must fall inside, and the range
        stretched by the ``hset`` factors must stay below Nyquist and, when the
        passband is known, inside it.
    hset : sequence of float, optional
        Resampling factors, distinct and strictly between 1 and 2. The default
        runs from 1.1 to 1.9 in steps of 0.05.
    segment_seconds : float, default 2.0
        Welch segment length; at least four samples.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        ``irasa_offset`` (log10 power) and ``irasa_slope`` (log10 power per log10
        Hz) of the aperiodic fit, then ``irasa_aperiodic_power`` and
        ``irasa_periodic_power`` integrals for each band, in V² for EEG in volts.
    """
    check_signals(series, windows)
    factors = _validate_factors(hset)
    fit_band = Band("fit", *fit_range)
    if fit_band.fmin <= 0:
        raise ValueError("fit_range must start above zero Hz.")
    if (
        isinstance(segment_seconds, bool)
        or not isinstance(segment_seconds, Real)
        or not np.isfinite(segment_seconds)
        or segment_seconds <= 0
    ):
        raise ValueError("segment_seconds must be finite and positive.")
    if len({band.name for band in bands}) != len(bands):
        raise ValueError("bands must have unique names.")
    for band in bands:
        if band.fmin < fit_band.fmin or band.fmax > fit_band.fmax:
            raise ValueError(f"band {band.name!r} must lie within fit_range.")
    nperseg = int(round(segment_seconds * series[0].sfreq))
    if nperseg < 4:
        raise ValueError("segment_seconds must yield at least four samples per Welch segment.")
    for signal in series:
        _validate_signal(signal, fit_band, float(factors.max()), nperseg)
    parameters = {
        "backend": "neurodsp",
        "backend_version": version("neurodsp"),
        "fit_range": fit_range,
        "hset": factors,
        "nperseg": nperseg,
        "noverlap": nperseg // 2,
        "welch_average": "mean",
        "welch_window": "hann",
        "periodic_threshold": None,
        "integration_support": "outward_bracketing_bins",
    }
    cache: dict[tuple[int, float, float], _Decomposition] = {}

    def table_for(band: Band | None) -> FeatureTable:
        units = (
            {"irasa_offset": "log10 power", "irasa_slope": "log10 power per log10 Hz"}
            if band is None
            else {"irasa_aperiodic_power": "V^2", "irasa_periodic_power": "V^2"}
        )

        def kernel(
            signal: Signal,
            trace: npt.NDArray[np.float64],
            times: npt.NDArray[np.float64],
            mask: npt.NDArray[np.bool_],
        ) -> dict[str, npt.NDArray[np.float64]]:
            del mask
            key = (id(signal), float(times[0]), float(times[-1]))
            if key not in cache:
                cache[key] = _decompose(trace, signal.sfreq, nperseg, factors, fit_band)
            result = cache[key]
            if band is None:
                return {"irasa_offset": result.offset, "irasa_slope": result.slope}
            weights = band_integration_weights(result.freqs, band.fmin, band.fmax)
            return {
                "irasa_aperiodic_power": result.aperiodic @ weights,
                "irasa_periodic_power": result.periodic @ weights,
            }

        table = expand_signal(
            series,
            trace_of=lambda signal: signal.data,
            kernel=kernel,
            units=units,
            windows=windows,
            groups=groups,
            include_global=include_global,
            mode="raw",
            parameters={**parameters, "band": None if band is None else asdict(band)},
        )
        return replace(table, meta=tuple(replace(meta, band=band) for meta in table.meta))

    return concat([table_for(None), *(table_for(band) for band in bands)])
