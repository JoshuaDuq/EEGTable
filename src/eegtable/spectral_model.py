"""Explicit specparam 2.0.0rc7 spectral parameterization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from importlib.metadata import version
from numbers import Integral, Real
from typing import Literal

import numpy as np
import numpy.typing as npt

from eegtable._expand import expand
from eegtable.bands import Band, passband_fraction
from eegtable.spectra import Spectra
from eegtable.table import FeatureTable, concat

_APERIODIC_UNITS = {
    "specparam_offset": "log10 power",
    "specparam_exponent": "a.u.",
    "specparam_knee": "Hz^exponent",
    "specparam_n_peaks": "count",
    "specparam_r_squared": "a.u.",
    "specparam_error": "log10 power",
}
_PEAK_UNITS = {
    "specparam_peak_cf": "Hz",
    "specparam_peak_height": "log10 power above aperiodic",
    "specparam_peak_width": "Hz",
}


def _validate_spectra(
    spectra: Spectra, fit_band: Band, bands: Sequence[Band]
) -> npt.NDArray[np.bool_]:
    if spectra.representation != "psd":
        raise ValueError("spectral_parameterization requires raw, linear PSD input.")
    if fit_band.fmin <= 0:
        raise ValueError("fit_range must start above zero Hz.")
    if spectra.freqs[0] > fit_band.fmin or spectra.freqs[-1] < fit_band.fmax:
        raise ValueError("the frequency axis must span the complete fit_range.")
    spacing = np.diff(spectra.freqs)
    if spacing.size == 0 or not np.allclose(spacing, spacing[0], rtol=1e-7, atol=1e-10):
        raise ValueError("spectral_parameterization requires uniformly spaced frequencies.")
    if spectra.passband is not None and passband_fraction(fit_band, *spectra.passband) < 1:
        raise ValueError("fit_range must lie entirely within the recording passband.")
    if len({band.name for band in bands}) != len(bands):
        raise ValueError("peak bands must have unique names.")
    for band in bands:
        if band.fmin < fit_band.fmin or band.fmax > fit_band.fmax:
            raise ValueError(f"peak band {band.name!r} must lie within fit_range.")
    mask = fit_band.mask(spectra.freqs)
    if mask.sum() < 5:
        raise ValueError("fit_range must contain at least five frequency bins.")
    power = spectra.data[..., mask]
    if not np.isfinite(power).all() or np.any(power <= 0):
        raise ValueError("spectral_parameterization requires finite, strictly positive PSD values.")
    return mask


def _validate_settings(
    aperiodic_mode: str,
    peak_width_limits: tuple[float, float],
    max_n_peaks: int,
    min_peak_height: float,
    peak_threshold: float,
) -> None:
    if aperiodic_mode not in ("fixed", "knee"):
        raise ValueError("aperiodic_mode must be 'fixed' or 'knee'.")
    if (
        len(peak_width_limits) != 2
        or not np.isfinite(peak_width_limits).all()
        or not 0 < peak_width_limits[0] < peak_width_limits[1]
    ):
        raise ValueError("peak_width_limits must contain increasing, finite, positive widths.")
    if isinstance(max_n_peaks, bool) or not isinstance(max_n_peaks, Integral) or max_n_peaks < 0:
        raise ValueError("max_n_peaks must be a nonnegative integer.")
    for name, value in (("min_peak_height", min_peak_height), ("peak_threshold", peak_threshold)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative.")


def _feature_column(
    spectra: Spectra,
    values: npt.NDArray[np.float64],
    fit_band: Band,
    *,
    measure: str,
    unit: str,
    output_band: Band | None,
    parameters: Mapping[str, object],
    groups: Mapping[str, Sequence[str]] | None,
    include_global: bool,
    flags: dict[str, npt.NDArray[np.bool_]],
) -> FeatureTable:
    def kernel(
        data: npt.NDArray[np.float64],
        freqs: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
        del data, freqs, weights
        return values, flags

    table = expand(
        spectra,
        kernel,
        measure=measure,
        unit=unit,
        bands=(fit_band,),
        groups=groups,
        include_global=include_global,
        baseline=None,
        mode="raw",
        min_bins=5,
        parameters={
            **parameters,
            "peak_band": None if output_band is None else asdict(output_band),
        },
        weighting="uniform",
    )
    # Every parameter rests on the full fitted range, including band peak parameters.
    return replace(table, meta=tuple(replace(meta, band=output_band) for meta in table.meta))


def spectral_parameterization(
    spectra: Spectra,
    *,
    bands: Sequence[Band] = (),
    fit_range: tuple[float, float] = (2.0, 40.0),
    aperiodic_mode: Literal["fixed", "knee"] = "fixed",
    peak_width_limits: tuple[float, float] = (0.5, 12.0),
    max_n_peaks: int = 6,
    min_peak_height: float = 0.0,
    peak_threshold: float = 2.0,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Fit fixed/knee aperiodic activity and Gaussian peaks with specparam.

    Requires positive, finite, linear PSD on a uniform frequency grid. The fit
    uses ``[fit_range[0], fit_range[1])``. Each named band returns the highest
    fitted peak whose center lies in that half-open band: center frequency,
    height above the aperiodic component, and bandwidth (twice Gaussian sigma).
    An absent peak is NaN with ``spectral_no_peak``. Backend fitting errors
    propagate because ``SpectralModel(debug=True)`` is mandatory. Each epoch,
    channel and window is fitted separately. Requires ``eegtable[spectral-model]``.

    Parameters
    ----------
    spectra : Spectra
        Linear power spectral density (``representation="psd"``) on a uniform
        frequency grid, finite and positive inside ``fit_range``.
    bands : sequence of Band
        Bands, with unique names inside ``fit_range``, in which to report the
        highest peak. Empty by default, giving only the aperiodic parameters and
        fit diagnostics.
    fit_range : tuple of float, default (2.0, 40.0)
        Frequency range fitted, in Hz. The axis must span it, it must hold at
        least five bins and lie above zero, and, when the passband is known, lie
        entirely inside it.
    aperiodic_mode : {"fixed", "knee"}, default "fixed"
        ``"knee"`` adds a knee parameter to the offset and exponent.
    peak_width_limits : tuple of float, default (0.5, 12.0)
        Lower and upper bounds of the fitted peak width, in Hz.
    max_n_peaks : int, default 6
        Most peaks fitted per spectrum.
    min_peak_height : float, default 0.0
        Absolute height a peak needs above the aperiodic fit, in log10 power.
    peak_threshold : float, default 2.0
        Relative threshold for detecting a peak, in standard deviations of the
        flattened spectrum.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        ``specparam_offset``, ``specparam_exponent``, ``specparam_knee`` (knee
        mode), ``specparam_n_peaks``, ``specparam_r_squared`` and
        ``specparam_error`` (mean absolute error in log10 power), then
        ``specparam_peak_cf``, ``specparam_peak_height`` and
        ``specparam_peak_width`` for each band.
    """
    from specparam import SpectralModel

    _validate_settings(
        aperiodic_mode, peak_width_limits, max_n_peaks, min_peak_height, peak_threshold
    )
    fit_band = Band("fit", *fit_range)
    mask = _validate_spectra(spectra, fit_band, bands)
    parameters = {
        "backend": "specparam",
        "backend_version": version("specparam"),
        "fit_range": fit_range,
        "aperiodic_mode": aperiodic_mode,
        "periodic_mode": "gaussian",
        "peak_width_limits": peak_width_limits,
        "max_n_peaks": max_n_peaks,
        "min_peak_height": min_peak_height,
        "peak_threshold": peak_threshold,
        "peak_selection": "highest_height_with_center_in_half_open_band",
        "error_metric": "mae_log10_power",
    }
    units = dict(_APERIODIC_UNITS)
    if aperiodic_mode == "fixed":
        units.pop("specparam_knee")
    shape = spectra.data.shape[:3]
    fitted = {measure: np.empty(shape) for measure in units}
    peaks = {band.name: np.full((*shape, 3), np.nan) for band in bands}
    for cell in np.ndindex(shape):
        model = SpectralModel(
            aperiodic_mode=aperiodic_mode,
            periodic_mode="gaussian",
            peak_width_limits=peak_width_limits,
            max_n_peaks=max_n_peaks,
            min_peak_height=min_peak_height,
            peak_threshold=peak_threshold,
            debug=True,
            verbose=False,
        )
        model.fit(spectra.freqs[mask], spectra.data[cell][mask])
        for field in ("offset", "exponent"):
            fitted[f"specparam_{field}"][cell] = model.get_params("aperiodic", field)
        if aperiodic_mode == "knee":
            fitted["specparam_knee"][cell] = model.get_params("aperiodic", "knee")
        fitted["specparam_n_peaks"][cell] = model.results.n_peaks
        fitted["specparam_r_squared"][cell] = model.get_metrics("gof", "rsquared")
        fitted["specparam_error"][cell] = model.get_metrics("error", "mae")
        periodic = model.get_params("periodic")
        if periodic.size:
            for band in bands:
                selected = periodic[band.mask(periodic[:, 0])]
                if selected.size:
                    peaks[band.name][cell] = selected[np.argmax(selected[:, 1])]
    if any(not np.isfinite(values).all() for values in fitted.values()):
        raise ValueError("specparam did not produce finite model parameters and diagnostics.")

    def column(
        values: npt.NDArray[np.float64],
        measure: str,
        unit: str,
        band: Band | None = None,
        flags: dict[str, npt.NDArray[np.bool_]] | None = None,
    ) -> FeatureTable:
        return _feature_column(
            spectra,
            values,
            fit_band,
            measure=measure,
            unit=unit,
            output_band=band,
            parameters=parameters,
            groups=groups,
            include_global=include_global,
            flags={} if flags is None else flags,
        )

    tables = [column(fitted[measure], measure, unit) for measure, unit in units.items()]
    for band in bands:
        for index, (measure, unit) in enumerate(_PEAK_UNITS.items()):
            values = peaks[band.name][..., index]
            tables.append(
                column(values, measure, unit, band, {"spectral_no_peak": np.isnan(values)})
            )
    return concat(tables)
