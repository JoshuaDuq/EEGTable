from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

from eegtable._expand import Kernel, expand
from eegtable.aperiodic import aperiodic_ratio
from eegtable.bands import Band
from eegtable.baseline import power_floor
from eegtable.spectra import Spectra, band_integration_weights
from eegtable.table import FeatureTable


def peak_frequency(
    spectra: Spectra,
    *,
    band: Band,
    aperiodic_adjusted: bool = True,
    smoothing_hz: float = 1.0,
    min_prominence: float = 0.1,
    interpolate: bool = True,
    fit_range: tuple[float, float] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Frequency of the strongest qualifying spectral peak within a band.

    The default search applies aperiodic adjustment, smoothing, and a prominence
    criterion. Each setting can be disabled:

    - ``aperiodic_adjusted`` divides out the fitted 1/f component first, via
      :func:`~eegtable.aperiodic_ratio`, so the peak is measured relative to the
      fitted background. An unadjusted maximum can be driven by low-frequency
      background power. The default fit range extends outside the analysis band;
      a narrow fit range can poorly constrain the background.
    - ``smoothing_hz`` averages over a centred window of that width before the
      search. It omits non-finite bins and renormalizes the retained integration
      weights, while preserving missing bins. Smoothing can attenuate narrow
      peaks or merge nearby maxima; choose its width for the intended spectral
      resolution. Zero disables smoothing.
    - ``min_prominence`` selects local maxima using SciPy's topographic
      prominence in log10 power. The qualifying peak with greatest smoothed
      linear power is retained. Interpolation also uses linear power.
      When none qualifies, the value is NaN and ``"no_peak"`` is set.
      Missing frequency bins separate the search into contiguous finite runs.

    Every column reports ``freq_resolution_hz``, and every cell carries
    ``"edge_hit"``, set when the maximum landed on the first or last bin of the
    band so the true peak may lie outside it. Bands holding fewer than three bins
    raise, because an interior maximum is undefined there.

    Parameters
    ----------
    spectra : Spectra
        Input spectra.
    band : Band
        Band to search.
    aperiodic_adjusted : bool, default True
        Divide out the fitted 1/f component before searching. Sets the measure
        name to ``"peak_freq_adjusted"``.
    smoothing_hz : float, default 1.0
        Width of the smoothing window in Hz. Zero disables smoothing.
    min_prominence : float, default 0.1
        Minimum topographic prominence in log10 power. Zero selects the bare
        in-band argmax, which may be an edge rather than a local peak.
    interpolate : bool, default True
        Refine the result with a parabola through the retained linear-power peak
        and its neighbors, using their actual frequency coordinates. This does
        not establish the accuracy of a physiological peak frequency.
    fit_range : tuple of float, optional
        Frequency range for the aperiodic fit. Defaults to
        ``(min(2.0, band.fmin), max(40.0, band.fmax))``. Requires
        ``aperiodic_adjusted``.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Peak frequency in Hz, with the ``"edge_hit"`` and ``"no_peak"`` flags.
    """
    if not np.isfinite(smoothing_hz) or smoothing_hz < 0.0:
        raise ValueError(f"smoothing_hz must be finite and >= 0, got {smoothing_hz}.")
    if not np.isfinite(min_prominence) or min_prominence < 0.0:
        raise ValueError(f"min_prominence must be finite and >= 0, got {min_prominence}.")
    if fit_range is not None and not aperiodic_adjusted:
        raise ValueError("fit_range applies only when aperiodic_adjusted is True.")

    if aperiodic_adjusted:
        span = fit_range or (min(2.0, band.fmin), max(40.0, band.fmax))
        spectra = aperiodic_ratio(spectra, fit_range=span)

    def kernel(
        data: npt.NDArray[np.float64],
        freqs: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
        return _find_peak(data, freqs, weights, smoothing_hz, min_prominence, interpolate)

    return expand(
        spectra,
        kernel,
        measure="peak_freq_adjusted" if aperiodic_adjusted else "peak_freq",
        unit="Hz",
        bands=(band,),
        groups=groups,
        include_global=include_global,
        baseline=None,
        mode="raw",
        min_bins=3,
        parameters={
            "aperiodic_adjusted": aperiodic_adjusted,
            "smoothing_hz": smoothing_hz,
            "min_prominence": min_prominence,
            "peak_selection": "scipy_log_smoothed_power_prominence",
            "interpolate": interpolate,
            "fit_range": fit_range,
        },
    )


def _smooth(
    values: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    smoothing_hz: float,
) -> npt.NDArray[np.float64]:
    if smoothing_hz <= 0.0 or freqs.size <= 3:
        return values
    finite = np.isfinite(values)
    radius = smoothing_hz / 2.0
    out = np.full(values.shape, np.nan)
    for index, frequency in enumerate(freqs):
        weights = band_integration_weights(
            freqs,
            max(float(freqs[0]), frequency - radius),
            min(float(freqs[-1]), frequency + radius),
        )
        spread = np.broadcast_to(weights, values.shape)
        selected = finite & (spread > 0.0)
        count = np.where(selected, spread, 0.0).sum(axis=3)
        total = np.where(selected, values * spread, 0.0).sum(axis=3)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[..., index] = np.where(count > 0, total / count, np.nan)
    return np.where(finite, out, np.nan)


def _prominent_indices(
    power: npt.NDArray[np.float64],
    log_power: npt.NDArray[np.float64],
    min_prominence: float,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.bool_]]:
    indices = np.zeros(power.shape[:3], dtype=int)
    found = np.zeros(power.shape[:3], dtype=bool)
    for cell in np.ndindex(power.shape[:3]):
        finite = np.isfinite(log_power[cell])
        boundaries = np.flatnonzero(np.diff(np.r_[False, finite, False]))
        candidates = []
        for start, stop in boundaries.reshape(-1, 2):
            peaks, _ = find_peaks(log_power[cell][start:stop], prominence=min_prominence)
            candidates.extend((start + peaks).tolist())
        if candidates:
            indices[cell] = candidates[int(np.argmax(power[cell][candidates]))]
            found[cell] = True
    return indices, found


def _find_peak(
    data: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    smoothing_hz: float,
    min_prominence: float,
    interpolate: bool,
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
    del weights
    finite = np.isfinite(data)
    usable = (finite & (data > 0.0)).any(axis=3)
    present = np.where(finite, data, np.nan)

    power = _smooth(present, freqs, smoothing_hz)
    filled = np.where(np.isfinite(power), power, -np.inf)
    index = np.argmax(filled, axis=3)
    if min_prominence > 0.0:
        floor = power_floor(np.fmax.reduce(power, axis=3, keepdims=True))
        log_power = np.log10(np.maximum(power, floor))
        index, found = _prominent_indices(power, log_power, min_prominence)
        usable &= found

    last = freqs.size - 1
    interior = (index > 0) & (index < last)
    safe = np.clip(index, 1, max(last - 1, 1))
    peak = freqs[index]

    if interpolate:
        left, centre, right = (_at(filled, np.clip(safe + o, 0, last)) for o in (-1, 0, 1))
        left_step = freqs[safe] - freqs[safe - 1]
        right_step = freqs[safe + 1] - freqs[safe]
        with np.errstate(invalid="ignore", divide="ignore"):
            # Fit around the centre using divided differences on the actual frequency grid.
            left_slope = (centre - left) / left_step
            right_slope = (right - centre) / right_step
            quadratic = (right_slope - left_slope) / (left_step + right_step)
            linear = left_slope + quadratic * left_step
            delta = -linear / (2.0 * quadratic)
        delta = np.where(np.isfinite(delta), np.clip(delta, -left_step / 2, right_step / 2), 0.0)
        peak = peak + np.where(interior, delta, 0.0)

    return (
        np.where(usable, peak, np.nan),
        {"edge_hit": usable & ~interior, "no_peak": ~usable},
    )


def _at(values: npt.NDArray[np.float64], index: npt.NDArray[np.int_]) -> npt.NDArray[np.float64]:
    return np.take_along_axis(values, index[..., np.newaxis], axis=3)[..., 0]


def spectral_centroid(
    spectra: Spectra,
    *,
    band: Band,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Centre of mass of the spectrum within a band.

    Computed as ``sum(f * P * df) / sum(P * df)``, so a non-uniform frequency
    grid is handled correctly.

    Parameters
    ----------
    spectra : Spectra
        Input spectra.
    band : Band
        Band to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Centroid frequency in Hz.
    """
    return _descriptor(
        spectra,
        _centroid_kernel,
        "spectral_centroid",
        "Hz",
        band,
        groups,
        include_global,
    )


def spectral_bandwidth(
    spectra: Spectra,
    *,
    band: Band,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Spread of the spectrum about its centroid, within a band.

    The mass-weighted standard deviation of frequency.

    Parameters
    ----------
    spectra : Spectra
        Input spectra.
    band : Band
        Band to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Bandwidth in Hz.
    """
    return _descriptor(
        spectra,
        _bandwidth_kernel,
        "spectral_bandwidth",
        "Hz",
        band,
        groups,
        include_global,
    )


def spectral_entropy(
    spectra: Spectra,
    *,
    band: Band,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Shannon entropy of the normalized spectrum, scaled to ``[0, 1]``.

    One means power is spread evenly across the band; zero means it is
    concentrated in a single bin. Normalized by ``log(n_bins)``, so values from
    bands holding different numbers of bins are not directly comparable. The
    discrete definition requires an approximately uniform frequency grid.

    Parameters
    ----------
    spectra : Spectra
        Input spectra.
    band : Band
        Band to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Normalized entropy, dimensionless.
    """
    return _descriptor(
        spectra,
        _entropy_kernel,
        "spectral_entropy",
        "a.u.",
        band,
        groups,
        include_global,
        weighting="uniform",
    )


def spectral_edge(
    spectra: Spectra,
    *,
    band: Band,
    percentile: float = 0.95,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Frequency below which a given fraction of the band's power lies.

    Returns the frequency of the bin at which the normalized cumulative mass
    first reaches ``percentile``. It is not interpolated, so the result is
    always a frequency present on the input grid.

    Parameters
    ----------
    spectra : Spectra
        Input spectra.
    band : Band
        Band to summarize.
    percentile : float, default 0.95
        Cumulative power fraction in ``(0, 1]``. Note this is a fraction, not a
        percentage.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Edge frequency in Hz.
    """
    if not np.isfinite(percentile) or not 0.0 < percentile <= 1.0:
        raise ValueError(f"percentile must be a finite fraction in (0, 1], got {percentile}.")

    def kernel(
        data: npt.NDArray[np.float64],
        freqs: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
        mass, _ = _mass(data, weights)
        cumulative = np.cumsum(mass, axis=3)
        # Use the same accumulation for the total so the final fraction is exactly one.
        total = cumulative[..., -1]
        with np.errstate(invalid="ignore", divide="ignore"):
            cumulative = cumulative / total[..., np.newaxis]
        reached = cumulative >= percentile
        index = reached.argmax(axis=3)
        return np.where(total > 0.0, freqs[index], np.nan), {}

    return _descriptor(
        spectra,
        kernel,
        "spectral_edge",
        "Hz",
        band,
        groups,
        include_global,
        {"percentile": percentile},
    )


def _descriptor(
    spectra: Spectra,
    kernel: Kernel,
    measure: str,
    unit: str,
    band: Band,
    groups: Mapping[str, Sequence[str]] | None,
    include_global: bool,
    parameters: Mapping[str, object] | None = None,
    weighting: Literal["trapezoid", "gradient", "band_integral", "uniform"] = "gradient",
) -> FeatureTable:
    return expand(
        spectra,
        kernel,
        measure=measure,
        unit=unit,
        bands=(band,),
        groups=groups,
        include_global=include_global,
        baseline=None,
        mode="raw",
        min_bins=3,
        parameters={} if parameters is None else parameters,
        weighting=weighting,
    )


def _mass(
    data: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    spread = np.asarray(np.broadcast_to(weights, data.shape), dtype=np.float64)
    mass = np.where(np.isfinite(data), data * spread, 0.0)
    return mass, mass.sum(axis=3)


def _centroid_kernel(
    data: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
    mass, total = _mass(data, weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid = np.where(total > 0.0, (mass * freqs).sum(axis=3) / total, np.nan)
    return centroid, {}


def _bandwidth_kernel(
    data: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
    mass, total = _mass(data, weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid = np.where(total > 0.0, (mass * freqs).sum(axis=3) / total, np.nan)
        deviation = (freqs[np.newaxis, np.newaxis, np.newaxis, :] - centroid[..., np.newaxis]) ** 2
        variance = np.where(total > 0.0, (mass * deviation).sum(axis=3) / total, np.nan)
    return np.sqrt(variance), {}


def _entropy_kernel(
    data: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
    spacing = np.diff(freqs)
    tolerance = np.finfo(float).eps * max(1.0, abs(float(spacing[0]))) * 8.0
    if not np.allclose(spacing, spacing[0], rtol=1e-6, atol=tolerance):
        raise ValueError(
            "spectral_entropy requires an approximately uniform frequency grid; "
            "recompute or interpolate the PSD onto a uniform-Hz grid."
        )
    mass, total = _mass(data, weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        probabilities = np.where(total[..., np.newaxis] > 0.0, mass / total[..., np.newaxis], 0.0)
        # 0 log 0 is 0 here, so an empty bin contributes nothing rather than NaN.
        terms = np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0)
        entropy = -terms.sum(axis=3) / np.log(float(mass.shape[3]))
    return np.where(total > 0.0, entropy, np.nan), {}
