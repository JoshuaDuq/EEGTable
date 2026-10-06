from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt

from eegtable._expand import Kernel, expand
from eegtable.bands import Band, passband_fraction
from eegtable.spectra import Spectra
from eegtable.table import ComputationSpec, FeatureTable, concat

_MIN_FIT_POINTS = 5
# scipy's median_abs_deviation(scale="normal") divides by this: the standard normal's 0.75
# quantile, which makes the MAD estimate a standard deviation.
_NORMAL_MAD = 0.6744897501960817
_CELLS_PER_BLOCK = 16384

# In the order _fit_one returns the three measures.
_UNITS: dict[str, str] = {
    "slope": "log10 power per log10 Hz",
    "offset": "log10 power",
    "r_squared": "a.u.",
}


def _validate_fit_band(spectra: Spectra, fit_range: tuple[float, float]) -> Band:
    band = Band("fit", *fit_range)
    if spectra.passband is not None and passband_fraction(band, *spectra.passband) < 1.0:
        raise ValueError(
            f"fit_range {fit_range} must lie within the recording passband "
            f"{spectra.passband}; filtered-out frequencies bias the aperiodic fit."
        )
    return band


def aperiodic(
    spectra: Spectra,
    *,
    fit_range: tuple[float, float] = (2.0, 40.0),
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    peak_rejection_z: float = 2.5,
    max_iterations: int = 3,
) -> FeatureTable:
    """Fit the aperiodic (1/f) component of the spectrum.

    Fits ``log10(P) = offset + slope * log10(f)`` over ``fit_range``, then
    iteratively discards points lying more than ``peak_rejection_z`` robust
    deviations **above** the fit and refits. Only positive residuals are
    rejected. Peaks above the background can bias the fitted slope; their
    influence depends on location and shape. Negative residuals are retained.

    Each retained frequency bin has equal regression weight. Grid density,
    fit range, filtering, and departures from a straight power law can affect
    the fit. This estimator does not fit a knee or Gaussian peaks.

    Parameters
    ----------
    spectra : Spectra
        Input spectra.
    fit_range : tuple of float, default (2.0, 40.0)
        Frequency range to fit, in Hz.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    peak_rejection_z : float, default 2.5
        Residual threshold in robust deviations.
    max_iterations : int, default 3
        Maximum refit rounds.

    Returns
    -------
    FeatureTable
        Columns for ``slope``, ``offset``, and ``r_squared``. Slope is negative
        for a decreasing power law. Fit quality is scored only on retained bins;
        it does not assess the full spectral decomposition. Fewer than five
        usable bins yields NaN parameters. Constant retained log power gives
        undefined ``r_squared`` even when slope and offset can be estimated.
    """
    _validate_fit_settings(peak_rejection_z, max_iterations)
    band = _validate_fit_band(spectra, fit_range)
    mask = band.mask(spectra.freqs)
    # One fit per cell serves all three measures; expand only lays out the columns.
    fits = _fit_cells(
        spectra.data[..., mask], spectra.freqs[mask], peak_rejection_z, max_iterations
    )

    def column(index: int) -> Kernel:
        def kernel(
            data: npt.NDArray[np.float64],
            freqs: npt.NDArray[np.float64],
            weights: npt.NDArray[np.float64],
        ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
            del data, freqs, weights
            return fits[..., index], {}

        return kernel

    tables = [
        expand(
            spectra,
            column(index),
            measure=which,
            unit=_UNITS[which],
            bands=(band,),
            groups=groups,
            include_global=include_global,
            baseline=None,
            mode="raw",
            min_bins=_MIN_FIT_POINTS,
            parameters={
                "fit_range": fit_range,
                "peak_rejection_z": peak_rejection_z,
                "max_iterations": max_iterations,
            },
        )
        for index, which in enumerate(_UNITS)
    ]

    # The fit range is not a named band, so these columns are broadband.
    stripped = [replace(t, meta=tuple(replace(m, band=None) for m in t.meta)) for t in tables]
    return concat(stripped)


def aperiodic_ratio(
    spectra: Spectra,
    *,
    fit_range: tuple[float, float] = (2.0, 40.0),
    peak_rejection_z: float = 2.5,
    max_iterations: int = 3,
) -> Spectra:
    """Divide the spectrum by its fitted aperiodic component.

    Returns spectra in which a pure power law is flat at 1.0, so an oscillation is
    measured relative to the fitted background. This is a dimensionless ratio,
    rather than a subtracted periodic PSD. An unadjusted spectral maximum can
    be driven by low-frequency background power.

    The fit is the same iterative, positive-residual-rejecting fit used by
    :func:`aperiodic`. A cell whose fit cannot be estimated is entirely NaN and
    carries ``aperiodic_fit_failed``; raw spectra are never silently substituted.

    Parameters
    ----------
    spectra : Spectra
        Input spectra.
    fit_range : tuple of float, default (2.0, 40.0)
        Frequency range to fit, in Hz. The fitted curve is divided out across the
        whole frequency axis, not only this range.
    peak_rejection_z : float, default 2.5
        Residual threshold in robust deviations.
    max_iterations : int, default 3
        Maximum refit rounds.

    Returns
    -------
    Spectra
        Dimensionless power divided by the fitted aperiodic component. The DC
        bin is NaN with zero coverage because the fit is undefined there.
    """
    _validate_fit_settings(peak_rejection_z, max_iterations)
    mask = _validate_fit_band(spectra, fit_range).mask(spectra.freqs)
    n_bins = int(mask.sum())
    if n_bins < _MIN_FIT_POINTS:
        raise ValueError(
            f"fit_range {fit_range} holds {n_bins} frequency bins of the axis spanning "
            f"({spectra.freqs[0]}, {spectra.freqs[-1]}), but the aperiodic fit needs at "
            f"least {_MIN_FIT_POINTS}."
        )

    fits = _fit_cells(
        spectra.data[..., mask], spectra.freqs[mask], peak_rejection_z, max_iterations
    )
    slope, offset = fits[..., 0, np.newaxis], fits[..., 1, np.newaxis]
    failed = ~(np.isfinite(slope) & np.isfinite(offset))[..., 0]
    curve = 10.0 ** (offset + slope * _log_frequency(spectra.freqs))
    out = spectra.data / curve
    out[failed] = np.nan

    return replace(
        spectra,
        data=out,
        representation="aperiodic_ratio",
        coverage=np.where(np.isfinite(out), spectra.coverage, 0.0),
        source=f"{spectra.source}+aperiodic_ratio",
        computation=ComputationSpec.create(
            "aperiodic_ratio",
            input_computation=spectra.computation.record(),
            fit_range=fit_range,
            peak_rejection_z=peak_rejection_z,
            max_iterations=max_iterations,
        ),
        flags={**spectra.flags, "aperiodic_fit_failed": failed},
    )


def _validate_fit_settings(peak_rejection_z: float, max_iterations: int) -> None:
    if (
        isinstance(peak_rejection_z, bool)
        or not isinstance(peak_rejection_z, Real)
        or not np.isfinite(peak_rejection_z)
        or peak_rejection_z <= 0.0
    ):
        raise ValueError(f"peak_rejection_z must be finite and positive, got {peak_rejection_z}.")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, Integral)
        or max_iterations < 1
    ):
        raise ValueError(f"max_iterations must be a positive integer, got {max_iterations!r}.")


def _fit_cells(
    data: npt.NDArray[np.float64], freqs: npt.NDArray[np.float64], z: float, iterations: int
) -> npt.NDArray[np.float64]:
    """Slope, offset and r_squared of every cell, stacked on a last axis of three."""
    log_f = _log_frequency(freqs)
    # The row count is explicit: a fit range outside the axis leaves no bins to infer it from.
    cells = data.reshape(math.prod(data.shape[:3]), data.shape[-1])
    # Every cell at once rather than one polyfit each, in blocks that bound the memory a
    # 256-channel recording would otherwise take for its intermediate arrays.
    blocks = [
        _fit_block(log_f, cells[start : start + _CELLS_PER_BLOCK], z, iterations)
        for start in range(0, cells.shape[0], _CELLS_PER_BLOCK)
    ]
    fits = np.concatenate(blocks) if blocks else np.empty((0, 3))
    return fits.reshape(*data.shape[:3], 3)


def _log_frequency(freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # A power law has no value at 0 Hz: NaN there keeps a DC bin out of every fit.
    positive = freqs > 0.0
    log_f = np.full(freqs.shape, np.nan)
    log_f[positive] = np.log10(freqs[positive])
    return log_f


def _fit_block(
    log_f: npt.NDArray[np.float64],
    power: npt.NDArray[np.float64],
    z: float,
    iterations: int,
) -> npt.NDArray[np.float64]:
    """Fit each row: a line in log-log space, refitted without positive outliers."""
    x = np.broadcast_to(log_f, power.shape)
    usable = np.isfinite(power) & (power > 0.0) & np.isfinite(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(usable, np.log10(np.where(usable, power, 1.0)), np.nan)
    enough = usable.sum(axis=1) >= _MIN_FIT_POINTS
    keep = usable
    slope, offset = _lines(x, log_p, keep)
    active = enough.copy()
    for _ in range(iterations):
        residuals = log_p - (offset[:, np.newaxis] + slope[:, np.newaxis] * x)
        kept = np.where(keep, residuals, np.nan)
        with warnings.catch_warnings():
            # A row with too few points to fit is all NaN by design; it never refits.
            warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
            centre = np.nanmedian(kept, axis=1)
            mad = np.nanmedian(np.abs(kept - centre[:, np.newaxis]), axis=1) / _NORMAL_MAD
        with np.errstate(invalid="ignore"):
            tightened = keep & (residuals <= z * mad[:, np.newaxis])
        # A row stops as the cell-by-cell fit did: no spread left, too few points left, or
        # nothing new rejected. It keeps the line it had.
        active &= ~(
            ~np.isfinite(mad)
            | (mad < 1e-12)
            | (tightened.sum(axis=1) < _MIN_FIT_POINTS)
            | (tightened == keep).all(axis=1)
        )
        if not active.any():
            break
        # Refit on every tightened mask, so the line and r_squared share one point set.
        keep = np.where(active[:, np.newaxis], tightened, keep)
        refit_slope, refit_offset = _lines(x, log_p, keep)
        slope = np.where(active, refit_slope, slope)
        offset = np.where(active, refit_offset, offset)
    fits = np.stack([slope, offset, _r_squared(x, log_p, keep, slope, offset)], axis=1)
    fits[~enough] = np.nan
    return fits


def _lines(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], keep: npt.NDArray[np.bool_]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Least-squares slope and intercept of each row's kept points."""
    n = keep.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_mean = np.where(keep, x, 0.0).sum(axis=1) / n
        y_mean = np.where(keep, y, 0.0).sum(axis=1) / n
        dx = np.where(keep, x - x_mean[:, np.newaxis], 0.0)
        dy = np.where(keep, y - y_mean[:, np.newaxis], 0.0)
        slope = (dx * dy).sum(axis=1) / (dx * dx).sum(axis=1)
    return slope, y_mean - slope * x_mean


def _r_squared(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    keep: npt.NDArray[np.bool_],
    slope: npt.NDArray[np.float64],
    offset: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Fit quality over the points each line was actually fitted to.

    Rejected peaks are excluded, so this statistic describes the retained
    straight-line fit. A bend within the fitted range can reduce its value and
    bias the slope; the statistic alone does not validate the decomposition.
    """
    n = keep.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(keep, y, 0.0).sum(axis=1) / n
        fitted = offset[:, np.newaxis] + slope[:, np.newaxis] * x
        residual = np.where(keep, (y - fitted) ** 2, 0.0).sum(axis=1)
        total = np.where(keep, (y - mean[:, np.newaxis]) ** 2, 0.0).sum(axis=1)
        r_squared = 1.0 - residual / total
    defined = (n >= _MIN_FIT_POINTS) & np.isfinite(slope) & np.isfinite(offset) & (total > 0.0)
    return np.where(defined, r_squared, np.nan)
