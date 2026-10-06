from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

# The first Slepian taper keeps 90% of its power in the band, MNE's low-bias criterion, only
# from a time-halfbandwidth product of 0.675, a bandwidth of 1.35 frequency bins. Below it
# MNE falls back to that leaky taper with just a warning.
MULTITAPER_MIN_BINS = 1.35


def minimum_sample_count(duration_seconds: float, sfreq: float) -> int:
    """Round a minimum duration up, allowing one ULP of conversion roundoff."""
    samples = np.nextafter(duration_seconds * sfreq, -np.inf)
    return max(1, int(np.ceil(samples)))


def validate_fraction_array(values: npt.NDArray[np.float64], name: str) -> None:
    """Require finite fractions on the closed unit interval."""
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real fractions.")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} must contain finite values in [0, 1].")


def validate_nonempty_shape(shape: tuple[int, ...], name: str) -> None:
    """Reject arrays with an empty scientific axis."""
    if any(length == 0 for length in shape):
        raise ValueError(f"{name} axes must be non-empty, got shape {shape}.")


def validate_names(names: Sequence[str], name: str) -> None:
    """Require non-empty, unique labels."""
    if any(not isinstance(value, str) or not value for value in names):
        raise ValueError(f"{name} must contain non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} must be unique.")


def validate_multitaper_bandwidth(
    bandwidth: float, sfreq: float, n_times: int, window: str, name: str
) -> None:
    """Require a band wide enough for one taper of the window to stay inside it."""
    floor = MULTITAPER_MIN_BINS * sfreq / n_times
    if bandwidth < floor:
        raise ValueError(
            f"{name} = {bandwidth} Hz is below {floor:.3f} Hz, {MULTITAPER_MIN_BINS} frequency "
            f"bins of window {window!r} ({n_times} samples): in a narrower band no Slepian "
            "taper keeps 90% of its power. Raise the bandwidth or widen the window."
        )


def blank_non_finite(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Non-finite samples replaced by NaN, so a nan-aware reduction really skips them.

    Coverage and every kernel's finiteness guard call a sample missing when it is
    not finite, but ``np.nanmean`` and friends skip only NaN: an infinity survives
    the very average those guards excluded it from, and ``np.nanvar`` turns the
    whole window into NaN instead of dropping the one bad sample.
    """
    finite = np.isfinite(values)
    # Clean data is the common case and is handed back untouched: the test costs a
    # pass over a bool array, where blanking costs a second float array the size of
    # the recording.
    if bool(finite.all()):
        return np.asarray(values, dtype=np.float64)
    return np.asarray(np.where(finite, values, np.nan), dtype=np.float64)
