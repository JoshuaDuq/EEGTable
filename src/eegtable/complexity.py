from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from numbers import Integral, Real
from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from eegtable._expand import expand_signal
from eegtable.signal import TimeSeries
from eegtable.spectra import Window
from eegtable.table import FeatureTable

_PAIR_BUDGET = 2_000_000
"""Template pairs compared at once.

Sample entropy is inherently O(n^2) in the window length. Comparing every pair at
once would need gigabytes on a multi-second window, so the comparison is chunked;
this bounds the working set without changing any result.
"""


def _antropy_feature(
    series: Sequence[TimeSeries],
    windows: Sequence[Window],
    measure: Callable[[npt.NDArray[np.float64]], float],
    name: str,
    minimum_samples: int,
    parameters: Mapping[str, object],
    groups: Mapping[str, Sequence[str]] | None,
    include_global: bool,
) -> FeatureTable:
    def kernel(
        signal: TimeSeries,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del signal, times, mask
        if trace.shape[-1] < minimum_samples:
            raise ValueError(f"{name} requires at least {minimum_samples} samples per window.")
        if not np.isfinite(trace).all():
            raise ValueError(f"{name} requires finite samples; gaps cannot be deleted.")
        return {name: _per_channel(trace, lambda values: measure(np.ascontiguousarray(values)))}

    return expand_signal(
        series,
        trace_of=lambda signal: signal.amplitude,
        kernel=kernel,
        units={name: "a.u."},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
        parameters={"backend": "antropy", **parameters},
    )


def permutation_entropy(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    order: int = 3,
    delay: int = 1,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Normalized ordinal-pattern entropy using AntroPy.

    Embeddings use ``order`` samples separated by ``delay`` samples. Entropy
    is divided by ``log2(order!)``. Ties follow AntroPy's deterministic ordinal
    ordering. Non-finite or unembeddable windows raise ``ValueError``.
    """
    import antropy

    for name, value, minimum in (("order", order, 2), ("delay", delay, 1)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return _antropy_feature(
        series,
        windows,
        lambda values: float(antropy.perm_entropy(values, order, delay, normalize=True)),
        "permutation_entropy",
        (order - 1) * delay + 1,
        {"order": order, "delay": delay, "normalize": True},
        groups,
        include_global,
    )


def lempel_ziv_complexity(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    symbolization: Literal["median", "mean"] = "median",
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Normalized Lempel--Ziv complexity of an explicitly binary signal.

    Samples at or above each channel/window's median or mean become 1; the
    others become 0. AntroPy normalizes the substring count by
    ``n / log2(n)``. EEGTable requires both symbols to occur and raises
    ``ValueError`` for a one-symbol sequence, rather than using AntroPy's
    observed-alphabet normalization with an alphabet of size one.
    """
    import antropy

    if symbolization not in ("median", "mean"):
        raise ValueError("symbolization must be 'median' or 'mean'.")

    def measure(values: npt.NDArray[np.float64]) -> float:
        threshold = np.median(values) if symbolization == "median" else np.mean(values)
        symbols = (values >= threshold).astype(np.uint32)
        if np.unique(symbols).size != 2:
            raise ValueError("lempel_ziv_complexity requires both binary symbols in each window.")
        return float(antropy.lziv_complexity(symbols, normalize=True))

    return _antropy_feature(
        series,
        windows,
        measure,
        "lempel_ziv_complexity",
        2,
        {"symbolization": symbolization, "threshold_comparison": ">=", "normalize": True},
        groups,
        include_global,
    )


def detrended_fluctuation(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """DFA scaling exponent using AntroPy's linear detrending implementation.

    Nonoverlapping block sizes span 4 to 10 percent of the window length,
    geometrically spaced by 1.2. At least 50 samples are required to support
    two distinct block sizes. Constant windows raise ``ValueError``.
    """
    import antropy

    def measure(values: npt.NDArray[np.float64]) -> float:
        if np.ptp(values) == 0.0:
            raise ValueError("detrended_fluctuation requires a nonconstant window.")
        exponent = float(antropy.detrended_fluctuation(values))
        if not np.isfinite(exponent):
            raise ValueError("detrended_fluctuation could not estimate a finite scaling exponent.")
        return exponent

    return _antropy_feature(
        series,
        windows,
        measure,
        "dfa_exponent",
        50,
        {"detrending_order": 1, "minimum_block": 4, "maximum_fraction": 0.1, "factor": 1.2},
        groups,
        include_global,
    )


def sample_entropy(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    order: int = 2,
    r: float = 0.2,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Sample entropy of the signal within each window.

    The negative log probability that two template vectors matching over ``order``
    samples still match over ``order + 1``. Larger values mean less
    self-similarity. Two templates match when their Chebyshev distance is strictly
    below ``r`` times the window's standard deviation.

    Undefined rather than zero when no pair of templates matches at all: the
    result is NaN when no length-``order`` pair matches, and infinite when some do
    but no length-``order + 1`` pair does.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows. Cost grows with the square of the window length.
    order : int, default 2
        Embedding dimension, the template length.
    r : float, default 0.2
        Tolerance as a fraction of the window's standard deviation.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Sample entropy, dimensionless.
    """
    _validate(order, r)

    def kernel(
        s: TimeSeries,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del s, times, mask
        return {"sampen": _per_channel(trace, lambda x: _sample_entropy(x, order, r))}

    return expand_signal(
        series,
        trace_of=lambda s: s.amplitude,
        kernel=kernel,
        units={"sampen": "nats"},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
        parameters={"order": order, "r": r},
    )


def multiscale_entropy(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    scales: Sequence[int] = (1, 2, 3, 4, 5),
    order: int = 2,
    r: float = 0.2,
    tolerance_mode: Literal["original_sd", "scale_sd"] = "original_sd",
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Sample entropy after coarse-graining, at each of several scales.

    Coarse-graining averages non-overlapping blocks of ``scale`` samples, so scale
    1 is the signal itself and larger scales describe slower structure. The
    ``tolerance_mode="original_sd"`` implements classical MSE by deriving the
    tolerance from the original window and holding it constant across scales.
    ``"scale_sd"`` recomputes it from each coarse-grained series.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    scales : sequence of int, default (1, 2, 3, 4, 5)
        Coarse-graining factors. Each becomes its own column, named
        ``mse{scale:02d}``.
    order : int, default 2
        Embedding dimension.
    r : float, default 0.2
        Tolerance as a fraction of the selected standard deviation.
    tolerance_mode : {"original_sd", "scale_sd"}, default "original_sd"
        Classical fixed tolerance, or scale-dependent tolerance.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        One column per scale. A scale that leaves too few samples to embed gives
        NaN rather than a value drawn from too little data.
    """
    _validate(order, r)
    if tolerance_mode not in ("original_sd", "scale_sd"):
        raise ValueError(
            f"tolerance_mode must be 'original_sd' or 'scale_sd', got {tolerance_mode!r}."
        )
    raw_scales = list(scales)
    if (
        not raw_scales
        or any(
            isinstance(scale, bool) or not isinstance(scale, Integral) or scale < 1
            for scale in raw_scales
        )
        or len(set(raw_scales)) != len(raw_scales)
    ):
        raise ValueError(f"scales must be positive integers, got {list(scales)!r}.")
    ordered = [int(scale) for scale in raw_scales]

    def kernel(
        s: TimeSeries,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del s, times, mask

        def at_scale(x: npt.NDArray[np.float64], scale: int) -> float:
            coarse = _coarse_grain(x, scale)
            tolerance = _tolerance(x, r) if tolerance_mode == "original_sd" else None
            return _sample_entropy(coarse, order, r, tolerance=tolerance)

        return {
            _scale_name(scale): _per_channel(trace, lambda x, k=scale: at_scale(x, k))
            for scale in ordered
        }

    return expand_signal(
        series,
        trace_of=lambda s: s.amplitude,
        kernel=kernel,
        units={_scale_name(scale): "nats" for scale in ordered},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
        parameters={
            "scales": ordered,
            "order": order,
            "r": r,
            "tolerance_mode": tolerance_mode,
            "coarse_graining": "nonoverlapping_mean",
        },
    )


def higuchi_fractal_dimension(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    k_max: int = 10,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    r"""Higuchi's fractal dimension of the signal within each window.

    The curve is re-traced at a range of strides :math:`k`, and its mean length
    :math:`L(k)` falls as :math:`k^{-D}`. The dimension :math:`D` is the slope of
    :math:`\log L(k)` against :math:`-\log k`, and it measures how much structure
    survives coarse sampling. Smooth curves tend toward dimension one;
    irregular sampled traces can yield larger estimates. This finite-stride
    regression is not clipped to the ideal graph-dimension interval ``[1, 2]``
    or proof of fractal scaling. Sampling rate, window length, and stride range
    affect it.

    This geometric scaling estimate differs from the recurrence estimate in
    :func:`~eegtable.sample_entropy`. For fixed ``k_max``, its cost is linear
    in window length, whereas sample-entropy pair counting is quadratic.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    k_max : int, default 10
        Largest stride, in samples. The window needs at least ``2 * k_max``
        samples. Compare estimates at consistent sampling rates and stride ranges.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Fractal dimension, dimensionless. NaN for a window with a non-finite
        sample or too few samples for the largest stride.

    References
    ----------
    Higuchi, T. (1988). Approach to an irregular time series on the basis of the
    fractal theory. Physica D: Nonlinear Phenomena, 31(2), 277-283.
    """
    if isinstance(k_max, bool) or not isinstance(k_max, Integral) or k_max < 2:
        raise ValueError(f"k_max must be an integer of at least 2, got {k_max!r}.")
    strides = int(k_max)

    def kernel(
        s: TimeSeries,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del s, times, mask
        return {"higuchi_fd": _higuchi(trace, strides)}

    return expand_signal(
        series,
        trace_of=lambda s: s.amplitude,
        kernel=kernel,
        units={"higuchi_fd": "a.u."},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
        parameters={"k_max": strides},
    )


def _higuchi(trace: npt.NDArray[np.float64], k_max: int) -> npt.NDArray[np.float64]:
    """Curve length against stride, fitted in log-log space, for every trace at once."""
    values = np.asarray(trace, dtype=float)
    n = values.shape[-1]
    if n < 2 * k_max:
        return np.full(values.shape[:-1], np.nan)

    lengths = np.empty((*values.shape[:-1], k_max))
    for k in range(1, k_max + 1):
        total = np.zeros(values.shape[:-1])
        for m in range(k):
            steps = (n - m - 1) // k
            # (n - 1) / (steps * k) restores the scale the stride removed, and the
            # remaining 1/k makes lengths at different strides comparable.
            curve = np.abs(np.diff(values[..., m::k], axis=-1)).sum(axis=-1)
            total += curve * (n - 1) / (steps * k * k)
        lengths[..., k - 1] = total / k

    log_k = -np.log(np.arange(1, k_max + 1, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_length = np.log(lengths)
    # A zero length, from a flat stretch at that stride, has no logarithm to fit.
    usable = np.isfinite(log_length)
    count = usable.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_mean = np.where(usable, log_k, 0.0).sum(axis=-1) / count
        y_mean = np.where(usable, log_length, 0.0).sum(axis=-1) / count
        dx = np.where(usable, log_k - x_mean[..., np.newaxis], 0.0)
        dy = np.where(usable, log_length - y_mean[..., np.newaxis], 0.0)
        slope = (dx * dy).sum(axis=-1) / (dx * dx).sum(axis=-1)
    # A gap would shorten one sub-curve and not the others, tilting the fit; the
    # measure is about how length scales, so a partial curve is not comparable.
    complete = np.isfinite(values).all(axis=-1)
    return np.where(complete & (count >= 2), slope, np.nan)


def _scale_name(scale: int) -> str:
    return f"mse{int(scale):02d}"


def _validate(order: int, r: float) -> None:
    if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
        raise ValueError(f"order must be a positive integer, got {order!r}.")
    if isinstance(r, bool) or not isinstance(r, Real) or not np.isfinite(r) or r <= 0.0:
        raise ValueError(f"r must be finite and positive, got {r!r}.")


def _per_channel(
    trace: npt.NDArray[np.float64],
    measure: object,
) -> npt.NDArray[np.float64]:
    n_epochs, n_channels, _ = trace.shape
    out = np.full((n_epochs, n_channels), np.nan)
    for epoch in range(n_epochs):
        for channel in range(n_channels):
            out[epoch, channel] = measure(trace[epoch, channel])  # type: ignore[operator]
    return out


def _coarse_grain(x: npt.NDArray[np.float64], scale: int) -> npt.NDArray[np.float64]:
    values = np.asarray(x, dtype=float)
    if scale <= 1:
        return values.copy()
    blocks = values.size // scale
    if blocks <= 0:
        return np.array([], dtype=float)
    shaped = values[: blocks * scale].reshape(blocks, scale)
    valid = np.isfinite(shaped).all(axis=1)
    averaged = np.full(blocks, np.nan)
    averaged[valid] = shaped[valid].mean(axis=1)
    return averaged


def _tolerance(x: npt.NDArray[np.float64], r: float) -> float:
    finite = np.asarray(x, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    if np.ptp(finite) == 0.0:
        return 0.0
    return r * float(np.std(finite))


def _sample_entropy(
    x: npt.NDArray[np.float64],
    order: int,
    r: float,
    *,
    tolerance: float | None = None,
) -> float:
    values = np.asarray(x, dtype=float)
    if values.size < order + 2:
        return float("nan")

    threshold = _tolerance(values, r) if tolerance is None else tolerance
    if not np.isfinite(threshold) or threshold <= 0.0:
        return float("nan")
    candidates = sliding_window_view(values, order + 1)
    templates = candidates[np.isfinite(candidates).all(axis=1)]
    n_templates = templates.shape[0]
    if n_templates < 2:
        return float("nan")

    matched_short = 0
    matched_long = 0
    chunk = max(1, _PAIR_BUDGET // n_templates)
    for start in range(0, n_templates, chunk):
        stop = min(start + chunk, n_templates)
        block = templates[start:stop]
        # A Chebyshev match is a conjunction over the embedding dimensions, so the
        # pair mask is narrowed one dimension at a time. Reducing a stacked
        # (pairs x dimensions) distance array instead would hold order + 1 floats
        # per pair where this holds one byte.
        close = np.abs(block[:, 0, np.newaxis] - templates[np.newaxis, :, 0]) < threshold
        # Each unordered pair once, matching antropy's positive-offset iteration.
        close &= np.arange(n_templates)[np.newaxis, :] > np.arange(start, stop)[:, np.newaxis]
        for dimension in range(1, order):
            close &= (
                np.abs(block[:, dimension, np.newaxis] - templates[np.newaxis, :, dimension])
                < threshold
            )
        matched_short += int(close.sum())
        close &= np.abs(block[:, order, np.newaxis] - templates[np.newaxis, :, order]) < threshold
        matched_long += int(close.sum())

    if matched_short == 0:
        return float("nan")
    if matched_long == 0:
        return float("inf")
    return float(-np.log(matched_long / matched_short))
