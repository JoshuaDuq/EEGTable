from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from eegtable._expand import expand
from eegtable.aperiodic import aperiodic_ratio
from eegtable.bands import BANDS_STANDARD, Band
from eegtable.spectra import Spectra
from eegtable.table import FeatureTable, Normalization

_PSD_MEAN_UNITS: dict[str, str] = {
    "raw": "V^2/Hz",
    "log10": "log10(V^2/Hz)",
    "log_ratio": "log10 ratio",
    "db": "dB",
    "percent": "%",
}

_PSD_INTEGRAL_UNITS: dict[str, str] = {
    "raw": "V^2",
    "log10": "log10(V^2)",
    "log_ratio": "log10 ratio",
    "db": "dB",
    "percent": "%",
}

# Morlet power arrives as a density: Spectra.from_tfr divides MNE's energy-2
# wavelet power by the sampling rate, so these are the same units as mean_psd.
_TFR_MEAN_UNITS: dict[str, str] = {
    "raw": "V^2/Hz",
    "log10": "log10(V^2/Hz)",
    "log_ratio": "log10 ratio",
    "db": "dB",
    "percent": "%",
}

# The spectrum divided by its fitted aperiodic component, a dimensionless ratio.
_PERIODIC_UNITS: dict[str, str] = {
    "raw": "ratio to the aperiodic fit",
    "log10": "log10 ratio to the aperiodic fit",
    "log_ratio": "log10 ratio",
    "db": "dB",
    "percent": "%",
}


def mean_psd(
    spectra: Spectra,
    *,
    bands: Sequence[Band] = BANDS_STANDARD,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    baseline: str | None = None,
    normalize: Normalization = "raw",
) -> FeatureTable:
    """Frequency-weighted mean power spectral density in each band.

    With complete data, the value is the piecewise-linear band integral divided
    by the bandwidth. Exact band boundaries are interpolated. Non-finite bins
    are omitted and the remaining frequency weights are renormalized, so a
    partial estimate does not equal the full-band integral divided by its width.

    Parameters
    ----------
    spectra : Spectra
        Input spectra.
    bands : sequence of Band, default BANDS_STANDARD
        Bands to compute.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    baseline : str, optional
        Name of the window to normalize against. That window is consumed and
        does not appear in the output. Normalized coverage is the minimum of
        analysis and baseline coverage; baseline flags propagate to the output.
    normalize : {"raw", "log10", "log_ratio", "db", "percent"}, default "raw"
        Normalization. ``"log_ratio"``, ``"db"`` and ``"percent"`` require ``baseline``.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and emitted window.
    """
    _require_representation(spectra, "psd", "mean_psd")
    return expand(
        spectra,
        _weighted_band_mean,
        measure="mean_psd",
        unit=_PSD_MEAN_UNITS[normalize],
        bands=bands,
        groups=groups,
        include_global=include_global,
        baseline=baseline,
        mode=normalize,
        min_bins=1,
        parameters={"quantity": "mean_power_spectral_density"},
        weighting="band_integral",
    )


def integrated_band_power(
    spectra: Spectra,
    *,
    bands: Sequence[Band] = BANDS_STANDARD,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    baseline: str | None = None,
    normalize: Normalization = "raw",
) -> FeatureTable:
    """Integral of a power spectral density over exact band boundaries.

    Piecewise-linear quadrature integrates from each band's numerical ``fmin``
    through ``fmax``, including interpolated boundary contributions. Raw EEG PSD
    input therefore yields V² rather than V²/Hz.
    """
    _require_representation(spectra, "psd", "integrated_band_power")
    return expand(
        spectra,
        _weighted_band_integral,
        measure="band_power",
        unit=_PSD_INTEGRAL_UNITS[normalize],
        bands=bands,
        groups=groups,
        include_global=include_global,
        baseline=baseline,
        mode=normalize,
        min_bins=2,
        parameters={"quantity": "integrated_power_spectral_density"},
        weighting="band_integral",
    )


def mean_tfr_power(
    spectra: Spectra,
    *,
    bands: Sequence[Band] = BANDS_STANDARD,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    baseline: str | None = None,
    normalize: Normalization = "raw",
) -> FeatureTable:
    """Frequency-weighted mean of wavelet time-frequency power in each band.

    :meth:`~eegtable.Spectra.from_tfr` divides MNE's Morlet power by the original
    sampling rate. The band mean is reported on a smoothed density scale in
    V²/Hz for EEG input in volts. Wavelet smoothing and time reduction can make
    it differ from :func:`mean_psd` for the same recording. Non-finite bins are
    omitted and the remaining frequency weights are renormalized.

    With a ``baseline`` and ``normalize="db"`` this is the decibel of the
    **window-mean** power. :func:`~eegtable.erds_mean` averages a per-sample dB
    trace from Hilbert band power instead. For identical positive power samples,
    the mean log power is no greater than the log of mean power; the population
    gap is about 2.51 dB for exponential power. Different spectral estimators and
    temporal support also affect comparisons between these functions.
    """
    _require_representation(spectra, "time_frequency_power", "mean_tfr_power")
    return expand(
        spectra,
        _weighted_band_mean,
        measure="mean_tfr_power",
        unit=_TFR_MEAN_UNITS[normalize],
        bands=bands,
        groups=groups,
        include_global=include_global,
        baseline=baseline,
        mode=normalize,
        min_bins=1,
        parameters={"quantity": "mean_time_frequency_power"},
        weighting="band_integral",
    )


def periodic_power(
    spectra: Spectra,
    *,
    bands: Sequence[Band] = BANDS_STANDARD,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    baseline: str | None = None,
    normalize: Normalization = "raw",
    fit_range: tuple[float, float] = (2.0, 40.0),
    peak_rejection_z: float = 2.5,
    max_iterations: int = 3,
) -> FeatureTable:
    """Frequency-weighted mean ratio to the fitted aperiodic component.

    Every epoch, channel and window is divided by its own fitted aperiodic
    component, as :func:`~eegtable.aperiodic_ratio` does, and the band value is the
    frequency-weighted mean of that ratio, weighted as :func:`mean_psd` weighs
    power. A pure power law gives 1.0 in every band. A power-law gain that the
    fitted line can follow is absorbed by the fit. Artifacts need not follow
    that model; this ratio does not provide artifact correction.

    The ratio is dimensionless and invariant to a positive overall power scale.
    PSD and time-frequency estimates can still differ because their spectral
    smoothing and temporal reductions differ.

    Parameters
    ----------
    spectra : Spectra
        Input spectra, a power spectral density or time-frequency power.
    bands : sequence of Band, default BANDS_STANDARD
        Bands to compute.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    baseline : str, optional
        Name of the window to normalize against, itself divided by its own
        aperiodic fit. That window is consumed and does not appear in the output.
    normalize : {"raw", "log10", "log_ratio", "db", "percent"}, default "raw"
        Normalization. ``"log_ratio"``, ``"db"`` and ``"percent"`` require ``baseline``.
    fit_range : tuple of float, default (2.0, 40.0)
        Frequency range of the aperiodic fit, in Hz.
    peak_rejection_z : float, default 2.5
        Residual threshold of the fit, in robust deviations.
    max_iterations : int, default 3
        Maximum refit rounds of the fit.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and emitted window. A cell whose
        aperiodic fit fails is NaN and carries ``aperiodic_fit_failed``.
    """
    ratio = aperiodic_ratio(
        spectra,
        fit_range=fit_range,
        peak_rejection_z=peak_rejection_z,
        max_iterations=max_iterations,
    )
    return expand(
        ratio,
        _weighted_band_mean,
        measure="periodic_power",
        unit=_PERIODIC_UNITS[normalize],
        bands=bands,
        groups=groups,
        include_global=include_global,
        baseline=baseline,
        mode=normalize,
        min_bins=1,
        # The fit's settings are recorded with the input, the aperiodic ratio.
        parameters={"quantity": "mean_power_over_the_aperiodic_fit"},
        weighting="band_integral",
    )


def _weighted_band_mean(
    data: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
    del freqs
    finite = np.isfinite(data)
    spread: npt.NDArray[np.float64] = np.asarray(
        np.broadcast_to(weights, data.shape), dtype=np.float64
    )
    numerator = np.sum(np.where(finite, data * spread, 0.0), axis=3)
    # A frequency that produced nothing leaves the average rather than entering it as zero.
    denominator = np.sum(np.where(finite, spread, 0.0), axis=3)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(denominator > 0.0, numerator / denominator, np.nan)
    return values, {}


def _weighted_band_integral(
    data: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
    del freqs
    finite = np.isfinite(data)
    complete = finite.all(axis=3)
    spread = np.asarray(np.broadcast_to(weights, data.shape), dtype=np.float64)
    values = np.sum(np.where(finite, data * spread, 0.0), axis=3)
    return np.where(complete, values, np.nan), {}


def _require_representation(spectra: Spectra, expected: str, operation: str) -> None:
    if spectra.representation != expected:
        raise ValueError(
            f"{operation} requires spectral representation {expected!r}, got "
            f"{spectra.representation!r}."
        )
