"""Epoch construction and explicit final transforms."""

from __future__ import annotations

from typing import Any

import mne  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from scipy.signal import detrend

from .config import EventEpochSettings, FixedEpochSettings, ReferenceSettings, ThresholdSettings
from .events import EventData, fixed_bounds
from .raw import good_eeg_names, restore_reference_channels, validate_geometry


def analysis_bounds(
    settings: EventEpochSettings | FixedEpochSettings, sfreq: float
) -> tuple[float, float]:
    if isinstance(settings, FixedEpochSettings):
        return fixed_bounds(settings, sfreq)
    return settings.tmin, settings.tmax


def validate_epochs(epochs: Any) -> None:
    if not isinstance(epochs, mne.BaseEpochs):
        raise TypeError("epochs: expected MNE BaseEpochs")
    if not len(epochs):
        raise ValueError(f"epochs: all epochs rejected; drop log: {epochs.drop_log}")
    good_eeg_names(epochs)
    if not np.isfinite(epochs.get_data()).all():
        raise ValueError("epochs: nonfinite samples")


def validate_metadata_precision(
    original: pd.DataFrame | None, restored: pd.DataFrame | None
) -> None:
    if original is None:
        return
    if restored is None:
        raise ValueError("metadata: missing after serialization")
    # JSON restores nulls as either None or NaN and infers all-missing column dtypes.
    original_values = original.astype(object).where(original.notna(), None)
    restored_values = restored.astype(object).where(restored.notna(), None)
    # JSON has limited decimal precision; refuse material metadata rounding.
    try:
        pd.testing.assert_frame_equal(
            restored_values, original_values, check_exact=False, rtol=1e-9, atol=0
        )
    except AssertionError as exc:
        raise ValueError(
            "metadata: cannot be preserved at native JSON precision; "
            f"rescale numeric columns or store exact values as strings. {exc}"
        ) from exc


def make_epochs(
    raw: Any,
    events: EventData,
    settings: EventEpochSettings | FixedEpochSettings,
    rejection: ThresholdSettings | None = None,
) -> Any:
    """Cut padded epochs around the resolved events.

    Epochs span the analysis window widened by ``settings.padding`` on each side,
    with no baseline, projection, decimation or detrending. Epochs overlapping a
    BAD annotation are dropped; no thresholds are applied here. At least one epoch
    must remain, with finite samples and a good EEG channel.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous recording.
    events : EventData
        From :func:`~eegtable.preprocessing.events.resolve_events`; its metadata
        becomes the epochs' metadata.
    settings : EventEpochSettings or FixedEpochSettings
        Analysis window and padding.
    rejection : ThresholdSettings, optional
        Only its ``tmin`` and ``tmax`` are used here: they fix the window that
        later threshold rejection judges, inside the analysis window. None judges
        the whole analysis window.

    Returns
    -------
    mne.Epochs
        Preloaded epochs.
    """
    # The rejection window is fixed here, on the analysis bounds, not on the padding.
    tmin, tmax = analysis_bounds(settings, events.original_sfreq)
    reject_tmin = tmin if rejection is None or rejection.tmin is None else rejection.tmin
    reject_tmax = tmax if rejection is None or rejection.tmax is None else rejection.tmax
    if not tmin <= reject_tmin <= reject_tmax <= tmax:
        raise ValueError("rejection: window must be inside the analysis interval")
    epochs = mne.Epochs(
        raw,
        events.events,
        dict(events.event_id),
        tmin=tmin - settings.padding,
        tmax=tmax + settings.padding,
        baseline=None,
        picks=list(raw.ch_names),
        preload=True,
        reject=None,
        flat=None,
        proj=False,
        decim=1,
        reject_tmin=reject_tmin,
        reject_tmax=reject_tmax,
        detrend=None,
        on_missing="raise",
        reject_by_annotation=True,
        metadata=events.metadata,
        event_repeated="error",
    )
    validate_epochs(epochs)
    return epochs


def interpolate_channels(epochs: Any) -> Any:
    """Spline-interpolate the bad EEG channels of a copy and mark them good.

    Bad labels on other channel types stay. Without bad EEG channels the copy is
    returned unchanged; otherwise every EEG channel needs a finite nonzero position
    and at least four must be good.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs with finite samples.

    Returns
    -------
    mne.Epochs
    """
    # Only EEG bads are targets; EOG/ECG bad labels are restored afterwards.
    validate_epochs(epochs)
    working = epochs.copy()
    eeg = set(
        name
        for name, kind in zip(epochs.ch_names, epochs.get_channel_types(), strict=True)
        if kind == "eeg"
    )
    targets = [name for name in epochs.info["bads"] if name in eeg]
    if not targets:
        return working
    validate_geometry(epochs)
    other = [name for name in epochs.info["bads"] if name not in eeg]
    working.info["bads"] = targets
    working.interpolate_bads(reset_bads=True, method={"eeg": "spline"})
    working.info["bads"] = other
    return working


def reference_epochs(epochs: Any, settings: ReferenceSettings) -> Any:
    """Re-reference the EEG of a copy of epochs or continuous data.

    Parameters
    ----------
    epochs : mne.Epochs or mne.io.Raw
        Data to re-reference.
    settings : ReferenceSettings
        ``channels`` is ``"average"`` of the good EEG channels (at least two), a
        tuple of good EEG channel names, or None to leave the reference unchanged.
        ``add_channels`` first restores missing acquisition-reference electrodes
        as zero channels, which a custom reference already applied rules out.

    Returns
    -------
    mne.Epochs or mne.io.Raw
    """
    working = restore_reference_channels(epochs, settings.add_channels)
    if settings.channels is None:
        return working
    good = good_eeg_names(working)
    if settings.channels == "average":
        if len(good) < 2:
            raise ValueError("reference.channels: average requires >=2 good EEG channels")
        reference: str | list[str] = "average"
    else:
        if not set(settings.channels) <= set(good):
            raise ValueError(
                "reference.channels: reference must contain only good retained EEG names"
            )
        reference = list(settings.channels)
    working.set_eeg_reference(ref_channels=reference, projection=False)
    return working


def detrend_epochs(epochs: Any, method: str) -> Any:
    """Remove the mean (``"constant"``) or linear trend (``"linear"``) of each EEG trace.

    Uses SciPy's ``detrend`` on a copy; other channel types are unchanged.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to detrend.
    method : {"constant", "linear"}
        Detrending type.

    Returns
    -------
    mne.Epochs
    """
    if method not in ("constant", "linear"):
        raise ValueError("epochs.detrend: expected constant or linear")
    picks = [
        name
        for name, kind in zip(epochs.ch_names, epochs.get_channel_types(), strict=True)
        if kind == "eeg"
    ]
    return epochs.copy().apply_function(detrend, picks=picks, type=method, channel_wise=False)


def baseline_epochs(epochs: Any, baseline: tuple[float | None, float | None]) -> Any:
    """Subtract the mean of a baseline interval from a copy, with MNE's ``apply_baseline``.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs on their final time grid.
    baseline : tuple of (float or None, float or None)
        Interval in seconds; None extends to the epoch's first or last sample.
        Bounds may lie at most one sample outside the epoch.

    Returns
    -------
    mne.Epochs
    """
    # Like MNE, accept bounds up to one sample outside the rounded final grid.
    tstep = 1.0 / epochs.info["sfreq"]
    start = epochs.times[0] if baseline[0] is None else baseline[0]
    stop = epochs.times[-1] if baseline[1] is None else baseline[1]
    if start < epochs.times[0] - tstep or stop > epochs.times[-1] + tstep:
        raise ValueError("epochs.baseline: bounds must be within final epoch times")
    return epochs.copy().apply_baseline(baseline)
