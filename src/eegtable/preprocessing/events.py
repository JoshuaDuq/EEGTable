"""Original-grid event identities and sample-exact fixed epoch geometry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import mne  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import EventSettings, FixedEpochSettings
from .raw import validate_raw


@dataclass(frozen=True)
class EventData:
    """Resolved events on the acquisition grid, from :func:`resolve_events`.

    Parameters
    ----------
    events : ndarray of int, shape (n_events, 3)
        MNE event array after the delay correction.
    event_id : dict of str to int
        Event names and codes; ``{"fixed": 1}`` for fixed-length epochs.
    metadata : DataFrame or None
        One row per event.
    original_sfreq : float
        Acquisition sample rate, in Hz.
    original_row : ndarray of int
        Row number of each event in the resolved array.
    original_samples : ndarray of int
        Event samples before the delay correction.
    delay : float
        Delay correction, in seconds.
    shift_samples : int
        Samples subtracted from every event, ``round(delay * sfreq)``.
    """

    events: NDArray[np.int64]
    event_id: dict[str, int]
    metadata: pd.DataFrame | None
    original_sfreq: float
    original_row: NDArray[np.int64]
    original_samples: NDArray[np.int64]
    delay: float = 0.0
    shift_samples: int = 0


def validate_events(raw: Any, events: NDArray[Any], event_id: dict[str, int]) -> None:
    if (
        events.ndim != 2
        or events.shape[1] != 3
        or len(events) == 0
        or events.dtype.kind not in "iu"
    ):
        raise ValueError("events: expected nonempty integer (n, 3) array")
    if np.any(events[1:, 0] <= events[:-1, 0]):
        raise ValueError("events: samples must be strictly increasing and unique")
    if events[0, 0] < raw.first_samp or events[-1, 0] >= raw.first_samp + raw.n_times:
        raise ValueError("events: sample outside recording")
    missing = set(event_id.values()) - set(events[:, 2])
    if missing:
        raise ValueError(f"events.event_id: missing codes {sorted(missing)}")


def fixed_bounds(settings: FixedEpochSettings, sfreq: float) -> tuple[float, float]:
    samples = round(settings.duration * sfreq)
    stride = (settings.duration - settings.overlap) * sfreq
    if samples < 2 or not np.isclose(samples, settings.duration * sfreq, atol=1e-9, rtol=0):
        raise ValueError(
            "epochs.duration: must align with acquisition samples and contain >=2 samples"
        )
    if not np.isclose(stride, round(stride), atol=1e-9, rtol=0):
        raise ValueError("epochs.overlap: stride must align with acquisition samples")
    return 0.0, (samples - 1) / sfreq


def resolve_events(
    raw: Any,
    settings: EventSettings | FixedEpochSettings,
    *,
    events: NDArray[Any] | None = None,
    metadata: pd.DataFrame | None = None,
) -> EventData:
    """Resolve the epoching events on the acquisition sample grid.

    Event-related settings read events from annotations, a stim channel or an MNE
    event file; fixed-length settings make one event per window. Event samples
    must be unique, strictly increasing and inside the recording both before and
    after the ``delay`` correction, and every code in ``event_id`` must occur.

    Parameters
    ----------
    raw : mne.io.Raw
        Recording the events belong to.
    settings : EventSettings or FixedEpochSettings
        Event source, codes and delay, or fixed-epoch geometry. A fixed
        ``duration`` and its stride must align with acquisition samples.
    events : ndarray of int, shape (n_events, 3), optional
        MNE event array used instead of the configured source; refused for fixed
        epochs.
    metadata : DataFrame, optional
        One row per resolved event, in event order.

    Returns
    -------
    EventData
        Delay-corrected events with their original samples and row numbers.
    """
    validate_raw(raw)
    sfreq = float(raw.info["sfreq"])
    if isinstance(settings, FixedEpochSettings):
        fixed_bounds(settings, sfreq)
        if events is not None:
            raise ValueError("events: explicit array incompatible with fixed epochs")
        if settings.stop is not None and settings.stop > raw.n_times / sfreq:
            raise ValueError("epochs.stop: exceeds acquisition duration")
        resolved = mne.make_fixed_length_events(
            raw,
            start=settings.start,
            stop=settings.stop,
            duration=settings.duration,
            overlap=settings.overlap,
            first_samp=True,
        )
        event_id, delay = {"fixed": 1}, 0.0
    else:
        event_id, delay = dict(settings.event_id), settings.delay
        if events is not None:
            resolved = np.asarray(events).copy()
        elif settings.source == "annotations":
            resolved, _ = mne.events_from_annotations(raw, event_id=event_id, use_rounding=True)
        elif settings.source == "stim":
            resolved = mne.find_events(
                raw,
                stim_channel=settings.stim_channel,
                shortest_event=settings.shortest_event,
                min_duration=settings.min_duration,
            )
        else:
            resolved = mne.read_events(settings.path)
    validate_events(raw, resolved, event_id)
    original = resolved[:, 0].copy()
    shift = round(delay * sfreq)
    resolved = resolved.astype(np.int64, copy=True)
    resolved[:, 0] -= shift
    validate_events(raw, resolved, event_id)
    if metadata is not None:
        if not isinstance(metadata, pd.DataFrame) or len(metadata) != len(resolved):
            raise ValueError("epochs.metadata: one row per input event required")
        metadata = metadata.copy().reset_index(drop=True)
    return EventData(
        resolved,
        event_id,
        metadata,
        sfreq,
        np.arange(len(resolved), dtype=np.int64),
        original,
        delay,
        shift,
    )


def _bids_annotation_labels(sidecar: pd.DataFrame) -> pd.Series[Any]:
    labels = sidecar["trial_type"].copy()
    if "value" in sidecar:
        value_counts = sidecar.groupby("trial_type", dropna=False)["value"].transform(
            "nunique", dropna=False
        )
        hierarchical = value_counts > 1
        labels.loc[hierarchical] += "/" + sidecar.loc[hierarchical, "value"].fillna("na").astype(
            str
        )
    return labels


def attach_bids_metadata(
    raw: Any,
    events: EventData,
    source: EventSettings | FixedEpochSettings,
    bids: Mapping[str, Any],
) -> EventData:
    """Align event sidecar rows on original acquisition samples, then add entities."""
    metadata = pd.DataFrame(index=np.arange(len(events.events)))
    if isinstance(source, EventSettings):
        if bids["events"] is None:
            raise ValueError("BIDS event epochs require an events.tsv sidecar.")
        sidecar = pd.DataFrame(bids["events"])
        onset = sidecar["onset"].to_numpy(dtype=float)
        samples = np.rint(onset * events.original_sfreq).astype(np.int64)
        if "sample" in sidecar:
            supplied = pd.to_numeric(sidecar["sample"], errors="raise").to_numpy(dtype=float)
            present = np.isfinite(supplied)
            if np.any(supplied[present] != samples[present]):
                raise ValueError(
                    "BIDS events.tsv sample and onset disagree on the acquisition grid."
                )
        samples += raw.first_samp
        selected = np.isin(samples, events.original_samples)
        if source.source == "annotations" and "trial_type" in sidecar:
            ambiguous = pd.Series(samples).duplicated(keep=False).to_numpy()
            label_matches = _bids_annotation_labels(sidecar).isin(source.event_id).to_numpy()
            selected &= ~ambiguous | label_matches
        matches = samples[selected]
        if np.unique(matches).size != matches.size or set(matches) != set(events.original_samples):
            raise ValueError("BIDS events.tsv must match each selected event sample exactly once.")
        metadata = sidecar.loc[selected].copy()
        metadata["bids_event_row"] = np.flatnonzero(selected)
        metadata.index = matches
        metadata = metadata.loc[events.original_samples].reset_index(drop=True)
    entities = bids["entities"]
    descriptors = {f"bids_{name}": value for name, value in entities.items()}
    descriptors["subject_id"] = f"sub-{entities['subject']}"
    for name in ("session", "task", "run"):
        if name in entities:
            descriptors[name] = entities[name]
    if bids["participant"] is not None:
        descriptors.update(
            {
                f"participant_{name}": value
                for name, value in bids["participant"].items()
                if name != "participant_id"
            }
        )
    for name, value in descriptors.items():
        if name in metadata:
            raise ValueError(f"BIDS event metadata collides with descriptor {name!r}.")
        metadata[name] = value
    if events.metadata is not None:
        shared = set(metadata).intersection(events.metadata)
        if shared:
            raise ValueError(
                f"Explicit epoch metadata collides with BIDS columns: {sorted(shared, key=str)}."
            )
        metadata = pd.concat([metadata, events.metadata], axis=1)
    return replace(events, metadata=metadata)
