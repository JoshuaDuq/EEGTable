"""Separate rejection fitting, transformation, and explicit manual exclusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._deps import require
from .config import AutoRejectSettings, ThresholdSettings
from .epochs import validate_epochs
from .raw import good_eeg_names, validate_geometry


@dataclass(frozen=True)
class RejectionModel:
    """A fitted autoreject model with the layout and window it was fitted on.

    Parameters
    ----------
    model : autoreject.AutoReject
        The fitted model.
    channels : tuple of str
        Channel names of the training epochs, in order.
    sfreq : float
        Their sample rate, in Hz.
    tmin, tmax : float
        Analysis window the model judges, in seconds.
    """

    model: Any
    channels: tuple[str, ...]
    sfreq: float
    tmin: float
    tmax: float


def fit_rejection(
    epochs: Any, settings: AutoRejectSettings, tmin: float, tmax: float
) -> RejectionModel:
    """Fit autoreject on the good EEG channels within the analysis window.

    Padding samples outside ``[tmin, tmax]`` do not enter the fit. Nothing is
    rejected or repaired here. Requires ``eegtable[preprocessing-auto]``.

    Parameters
    ----------
    epochs : mne.Epochs
        Padded epochs with finite nonzero EEG positions and at least as many
        epochs as cross-validation folds.
    settings : AutoRejectSettings
        Interpolation and consensus grids, folds and seed; every ``n_interpolate``
        must be below the number of good EEG channels.
    tmin, tmax : float
        Analysis window, in seconds.

    Returns
    -------
    RejectionModel
    """
    autoreject = require("autoreject", "preprocessing-auto")
    validate_epochs(epochs)
    validate_geometry(epochs)
    picks = good_eeg_names(epochs)
    if len(epochs) < settings.cv:
        raise ValueError("rejection.cv: more folds than usable epochs")
    if max(settings.n_interpolate) >= len(picks):
        raise ValueError("rejection.n_interpolate: must be below good EEG channel count")
    model = autoreject.AutoReject(
        n_interpolate=list(settings.n_interpolate),
        consensus=list(settings.consensus),
        cv=settings.cv,
        random_state=settings.random_state,
        picks=picks,
        n_jobs=1,
    )
    # Fit on the analysis window only; padding samples must not drive rejection.
    model.fit(epochs.copy().crop(tmin, tmax))
    return RejectionModel(model, tuple(epochs.ch_names), float(epochs.info["sfreq"]), tmin, tmax)


def reject_epochs(epochs: Any, settings: ThresholdSettings) -> Any:
    """Drop epochs whose peak-to-peak amplitude breaks the thresholds, on a copy.

    The window judged is the one :func:`~eegtable.preprocessing.epochs.make_epochs`
    fixed. At least one epoch must remain.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to screen.
    settings : ThresholdSettings
        ``reject`` and ``flat`` peak-to-peak limits in volts, by channel type; each
        named type must be present.

    Returns
    -------
    mne.Epochs
    """
    validate_epochs(epochs)
    present = set(epochs.get_channel_types())
    if not set(settings.reject or {}) | set(settings.flat or {}) <= present:
        raise ValueError("rejection: threshold names absent channel type")
    # make_epochs fixed the decision window (reject_tmin/reject_tmax), which survives
    # FIF round trips, so it is not restated here.
    working = epochs.copy().drop_bad(reject=settings.reject, flat=settings.flat)
    validate_epochs(working)
    return working


def apply_rejection(epochs: Any, model: RejectionModel) -> tuple[Any, Any]:
    """Apply a fitted autoreject model: drop bad epochs and interpolate bad channels.

    The decisions are made on the fitted analysis window and applied to the padded
    epochs. At least one epoch must remain.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs with the fitted channels, in order, and sample rate.
    model : RejectionModel
        From :func:`fit_rejection`.

    Returns
    -------
    epochs : mne.Epochs
        The cleaned copy.
    log : autoreject.RejectLog
        Bad epochs and per-epoch channel labels.
    """
    if tuple(epochs.ch_names) != model.channels or epochs.info["sfreq"] != model.sfreq:
        raise ValueError("rejection: incompatible channels or sample rate")
    # Decide on the fitted window, then repair the padded epochs with that same log.
    log = model.model.get_reject_log(epochs.copy().crop(model.tmin, model.tmax))
    result = model.model.transform(epochs.copy(), reject_log=log)
    validate_epochs(result)
    return result, log


def apply_epoch_review(epochs: Any, original_ids: tuple[int, ...]) -> Any:
    """Drop reviewed epochs from a copy, logged with reason ``"USER"``.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs under review.
    original_ids : tuple of int
        Unique original event rows (``epochs.selection``) to drop, not displayed
        positions. Each must still be present, and at least one epoch must remain.

    Returns
    -------
    mne.Epochs
    """
    if any(type(value) is not int for value in original_ids) or len(set(original_ids)) != len(
        original_ids
    ):
        raise ValueError("review.epochs: expected unique integer original event IDs")
    # IDs are original event rows (epochs.selection), never displayed row positions.
    missing = set(original_ids) - set(epochs.selection)
    if missing:
        raise ValueError(f"review.epochs: original event IDs already absent: {sorted(missing)}")
    working = epochs.copy().drop(
        np.flatnonzero(np.isin(epochs.selection, original_ids)), reason="USER"
    )
    validate_epochs(working)
    return working
