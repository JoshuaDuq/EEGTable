"""Explicit anti-aliasing and final-grid cropping contracts."""

from __future__ import annotations

from math import gcd
from typing import Any

from .config import DecimationSettings, FilterSettings, ResamplingSettings


def validate_sampling(
    sfreq: float,
    settings: DecimationSettings | ResamplingSettings,
    filters: FilterSettings,
    tmin: float,
    n_times: int,
) -> None:
    if isinstance(settings, DecimationSettings):
        cutoff = filters.h_freq
        target = sfreq / settings.factor
        if cutoff is None:
            raise ValueError("sampling: decimation requires explicit filter.h_freq")
        # MNE's "auto" low-pass transition; its stopband must end below the new Nyquist.
        transition = min(max(cutoff * 0.25, 2), sfreq / 2 - cutoff)
        if cutoff > target / 3 or cutoff + transition >= target / 2:
            raise ValueError("sampling: low-pass cutoff and transition exceed safe decimation band")
        return
    if settings.sfreq >= sfreq:
        raise ValueError("sampling.sfreq: must be below input sampling frequency")
    # MNE rounds the output length but keeps the requested rate, which would stretch the
    # grid; and Epochs.crop rounds relative to t=0, so the origin must lie on the new grid.
    output_length = n_times * settings.sfreq / sfreq
    if abs(output_length - round(output_length)) > 1e-9 or round(output_length) < 2:
        raise ValueError(
            "sampling.sfreq: padded epoch must map to a whole number (>=2) of output samples"
        )
    if abs(tmin * settings.sfreq - round(tmin * settings.sfreq)) > 1e-9:
        raise ValueError(
            "sampling.sfreq: output grid must contain the epoch origin; adjust epochs.padding"
        )
    divisor = gcd(n_times, round(output_length))
    up, down = round(output_length) // divisor, n_times // divisor
    # SciPy resample_poly's default Kaiser FIR spans 10*max(up, down) upsampled samples.
    support_seconds = 10 * max(up, down) / up / sfreq
    if settings.padding < support_seconds:
        raise ValueError(
            f"sampling.padding: {settings.padding} below FIR half-support {support_seconds}"
        )


def resample_epochs(
    epochs: Any,
    settings: DecimationSettings | ResamplingSettings,
    filters: FilterSettings,
    *,
    n_jobs: int = 1,
) -> Any:
    """Decimate or polyphase-resample a copy of padded epochs to a lower rate.

    Decimation keeps every ``factor``-th sample without filtering, so the
    configured low-pass must already sit at most a third of the new rate, with
    its transition ending below the new Nyquist. Polyphase resampling only
    downsamples; the padded epoch must map to a whole number of output samples,
    the epoch origin must lie on the new grid, and the padding must cover the
    anti-aliasing filter's half-support.

    Parameters
    ----------
    epochs : mne.Epochs
        Padded epochs.
    settings : DecimationSettings or ResamplingSettings
        Decimation factor, or target rate and padding.
    filters : FilterSettings
        The filtering already applied; decimation checks its ``h_freq``.
    n_jobs : int, default 1
        Passed to MNE resampling.

    Returns
    -------
    mne.Epochs
    """
    validate_sampling(
        epochs.info["sfreq"], settings, filters, float(epochs.times[0]), len(epochs.times)
    )
    if isinstance(settings, DecimationSettings):
        result = epochs.copy().decimate(settings.factor)
    else:
        result = epochs.copy().resample(
            settings.sfreq, method="polyphase", window=("kaiser", 5.0), pad="reflect", n_jobs=n_jobs
        )
    # Rejection is finished. Epochs.decimate keeps the old reject window, and when tmin
    # moves onto the new grid read_epochs refuses a window that starts before it.
    result.reject_tmin = result.reject_tmax = None
    return result


def crop_epochs(epochs: Any, tmin: float, tmax: float) -> Any:
    """Crop the padding from a copy of epochs, without extrapolating.

    The available samples nearest the requested bounds are kept; the endpoints may
    differ from ``tmin`` and ``tmax`` by up to one output sample, and a larger gap
    raises.

    Parameters
    ----------
    epochs : mne.Epochs
        Padded epochs on their final grid.
    tmin, tmax : float
        Analysis window, in seconds.

    Returns
    -------
    mne.Epochs
    """
    # Bounds were rounded to the acquisition grid at construction and decimation keeps
    # every n-th sample from t=0, so the available endpoints can sit up to one output
    # sample inside the request. Crop to what exists; never extrapolate.
    tolerance = 1.0 / epochs.info["sfreq"] + 1e-9
    if epochs.times[0] - tmin > tolerance or tmax - epochs.times[-1] > tolerance:
        raise ValueError("crop-epochs: requested bounds outside available samples")
    result = epochs.copy().crop(max(tmin, epochs.times[0]), min(tmax, epochs.times[-1]))
    if abs(result.times[0] - tmin) > tolerance or abs(result.times[-1] - tmax) > tolerance:
        raise ValueError("crop-epochs: output grid displacement exceeds one sample")
    return result
