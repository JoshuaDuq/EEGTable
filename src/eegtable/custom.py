"""Measures of your own, computed with the library's bands, windows, ROIs and provenance.

A kernel reduces arrays; everything around it is the same machinery the built-in
measures use: bands and windows become columns, channels are averaged into ROIs and
the global mean, non-finite input lowers coverage, Morlet support is carried, and
every column is named from its complete definition.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from eegtable._expand import expand, expand_signal
from eegtable._validation import blank_non_finite
from eegtable.bands import Band
from eegtable.power import _unit
from eegtable.signal import TimeSeries
from eegtable.spectra import Spectra, Window
from eegtable.table import FeatureTable, Normalization


def _real_values(values: Any) -> npt.NDArray[np.float64]:
    if np.iscomplexobj(values):
        raise TypeError("Custom measures must return real feature values.")
    return np.asarray(values, dtype=float)


def spectral_measure(
    spectra: Spectra,
    kernel: Callable[..., Any],
    *,
    measure: str,
    unit: str,
    bands: Sequence[Band] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    baseline: str | None = None,
    normalize: Normalization = "raw",
    parameters: Mapping[str, object] | None = None,
) -> FeatureTable:
    """Apply your own spectral kernel across bands, windows and spatial units.

    Parameters
    ----------
    spectra : Spectra
        Power spectra, e.g. from :meth:`Spectra.welch`.
    kernel : callable
        ``kernel(power, freqs, weights)`` returning ``values``, or
        ``(values, flags)``. ``power`` has shape ``(epochs, channels, windows,
        freqs)`` restricted to one band's bins, with non-finite bins as NaN;
        ``weights`` are those bins' widths in Hz. ``values`` has shape ``(epochs,
        channels, windows)``; ``flags`` maps names to boolean arrays of that shape.
    measure : str
        The label columns carry and :meth:`FeatureTable.select` matches.
    unit : str
        Physical unit of the values, before normalization.
    bands : sequence of Band, optional
        Bands to apply the kernel to. None applies it once to every bin.
    groups, include_global
        ROIs, and whether to add the mean across all channels, as for the
        built-in measures.
    baseline : str, optional
        Name of the window to normalize against with ``normalize``.
    normalize : {"raw", "log10", "log_ratio", "db", "percent"}, default "raw"
        As for :func:`eegtable.integrated_band_power`.
    parameters : mapping, optional
        The kernel's own settings. They enter every column's identity, so two
        settings never share a column; values must be JSON-serializable.

    Returns
    -------
    FeatureTable
        One column per band, window and spatial unit. The kernel's qualified name
        is part of each column's identity, so give it a named function rather than
        a lambda.
    """

    def adapted(
        data: npt.NDArray[np.float64],
        freqs: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
        result = kernel(blank_non_finite(data), freqs, weights)
        values, flags = result if isinstance(result, tuple) else (result, {})
        return _real_values(values), dict(flags)

    return expand(
        spectra,
        adapted,
        measure=measure,
        unit=_unit(
            {
                "raw": unit,
                "log10": f"log10({unit})",
                "log_ratio": "log10 ratio",
                "db": "dB",
                "percent": "%",
            },
            normalize,
        ),
        bands=bands,
        groups=groups,
        include_global=include_global,
        baseline=baseline,
        mode=normalize,
        min_bins=1,
        parameters=_identity(kernel, parameters),
    )


def signal_measure(
    series: Sequence[TimeSeries],
    kernel: Callable[..., Any],
    *,
    measure: str,
    unit: str,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    parameters: Mapping[str, object] | None = None,
) -> FeatureTable:
    """Apply your own time-domain kernel across series, windows and spatial units.

    Parameters
    ----------
    series : sequence of TimeSeries
        Broadband :class:`Signal` or band-limited :class:`BandSignal` inputs; a band
        signal is measured on its envelope. Each series' band becomes a column field.
    kernel : callable
        ``kernel(trace, times)`` returning ``values``. ``trace`` has shape
        ``(epochs, channels, samples)`` for one window, with non-finite samples as
        NaN; ``times`` holds those samples' times in seconds. ``values`` has shape
        ``(epochs, channels)``.
    measure : str
        The label columns carry and :meth:`FeatureTable.select` matches.
    unit : str
        Physical unit of the values.
    windows : sequence of Window
        Analysis windows; finite bounds must lie within the data.
    groups, include_global
        ROIs, and whether to add the mean across all channels, as for the
        built-in measures.
    parameters : mapping, optional
        The kernel's own settings, entered into every column's identity.

    Returns
    -------
    FeatureTable
        One column per series, window and spatial unit. Coverage is the fraction of
        each window's samples that were finite.
    """

    def adapted(
        signal: TimeSeries,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del signal, mask
        return {measure: _real_values(kernel(trace, times))}

    return expand_signal(
        series,
        trace_of=lambda signal: signal.amplitude,
        kernel=adapted,
        units={measure: unit},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
        parameters=_identity(kernel, parameters),
    )


def _identity(
    kernel: Callable[..., Any], parameters: Mapping[str, object] | None
) -> dict[str, object]:
    if parameters is not None and "kernel" in parameters:
        raise ValueError("parameters cannot override the kernel identity.")
    # The label is the user's choice; the kernel's own name keeps two different
    # computations under one label from being stacked as one feature.
    name = f"{getattr(kernel, '__module__', '?')}.{getattr(kernel, '__qualname__', repr(kernel))}"
    return {"kernel": name, **dict(parameters or {})}
