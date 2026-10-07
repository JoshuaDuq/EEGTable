"""Copy-preserving raw preparation and continuous FIR filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from .config import AnnotationSettings, ChannelSettings, CropSettings, FilterSettings


def physiology_names(raw: Any) -> list[str]:
    return [
        name
        for name, kind in zip(raw.ch_names, raw.get_channel_types(), strict=True)
        if kind in ("eeg", "eog", "ecg")
    ]


def good_eeg_names(raw: Any) -> list[str]:
    result = [
        name
        for name, kind in zip(raw.ch_names, raw.get_channel_types(), strict=True)
        if kind == "eeg" and name not in raw.info["bads"]
    ]
    if not result:
        raise ValueError("channels: at least one good EEG channel required")
    return result


def validate_source_raw(raw: Any) -> None:
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw: expected MNE BaseRaw")
    if not np.isfinite(raw.info["sfreq"]) or raw.info["sfreq"] <= 0 or raw.n_times == 0:
        raise ValueError("raw: positive sample rate and nonempty data required")
    picks = physiology_names(raw)
    if not picks:
        return
    for start in range(0, raw.n_times, 100_000):
        data = raw.get_data(picks=picks, start=start, stop=min(start + 100_000, raw.n_times))
        if not np.isfinite(data).all():
            raise ValueError("raw: nonfinite physiology samples")


def validate_raw(raw: Any) -> None:
    validate_source_raw(raw)
    good_eeg_names(raw)


def require_names(raw: Any, requested: tuple[str, ...] | list[str], path: str) -> None:
    missing = set(requested) - set(raw.ch_names)
    if missing:
        raise ValueError(f"{path}: missing channels {sorted(missing)}")


def validate_geometry(raw: Any) -> None:
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    positions = np.array([raw.info["chs"][index]["loc"][:3] for index in picks])
    if not np.isfinite(positions).all() or np.any(np.linalg.norm(positions, axis=1) == 0):
        raise ValueError("channels.montage: finite nonzero EEG positions required")
    if len(good_eeg_names(raw)) < 4:
        raise ValueError("channels.montage: at least four good EEG electrodes required")


def prepare_channels(raw: Any, settings: ChannelSettings) -> Any:
    """Rename, type, derive, drop and locate channels on a copy of a recording.

    In order: renames, channel types, bipolar derivations (the sources are kept),
    drops, the montage, bad labels added to those already marked, and the policy
    for inactive projectors. Every named channel must exist. The result needs at
    least one good EEG channel and finite physiology samples; with
    ``interpolate_bads``, also finite nonzero positions for every EEG channel and
    at least four good ones.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous recording; loaded into memory on the copy.
    settings : ChannelSettings
        Channel preparation settings.

    Returns
    -------
    mne.io.Raw
        The prepared copy.
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw: expected MNE BaseRaw")
    working = raw.copy().load_data()
    require_names(working, list(settings.rename), "channels.rename")
    renamed = [settings.rename.get(name, name) for name in working.ch_names]
    if len(set(renamed)) != len(renamed):
        raise ValueError("channels.rename: resulting names collide")
    working.rename_channels(dict(settings.rename))
    require_names(working, list(settings.types), "channels.types")
    working.set_channel_types(dict(settings.types))
    for bipolar in settings.bipolar:
        require_names(working, [bipolar.anode, bipolar.cathode], "channels.bipolar")
        if bipolar.name in working.ch_names:
            raise ValueError(f"channels.bipolar: channel {bipolar.name} already exists")
        kinds = dict(zip(working.ch_names, working.get_channel_types(), strict=True))
        if any(
            kinds[name] not in ("eeg", "eog", "ecg") for name in (bipolar.anode, bipolar.cathode)
        ):
            raise ValueError("channels.bipolar: sources must be voltage physiology channels")
        derived = mne.set_bipolar_reference(
            working,
            bipolar.anode,
            bipolar.cathode,
            ch_name=bipolar.name,
            drop_refs=False,
            copy=True,
        ).pick([bipolar.name])
        derived.set_channel_types({bipolar.name: bipolar.type})
        # Bipolar referencing removes EEG projectors and changes reference metadata;
        # those side effects belong only to the auxiliary channel's temporary copy.
        working.add_channels([derived], force_update_info=True)
    require_names(working, settings.drop, "channels.drop")
    if settings.drop:
        working.drop_channels(list(settings.drop))
    if settings.montage is not None:
        working.set_montage(_montage(settings.montage), on_missing="raise")
    require_names(working, settings.bads, "channels.bads")
    working.info["bads"] = list(dict.fromkeys([*working.info["bads"], *settings.bads]))
    inactive = [
        index for index, projector in enumerate(working.info["projs"]) if not projector["active"]
    ]
    if inactive:
        if settings.projections == "error":
            raise ValueError(
                "channels.projections: inactive projectors require apply or discard-inactive"
            )
        if settings.projections == "apply":
            working.apply_proj()
        else:
            working.del_proj(inactive)
    validate_raw(working)
    if settings.interpolate_bads:
        validate_geometry(working)
    return working


def _montage(source: str | Path) -> Any:
    # A standard name, a digitized FIF, or any electrode file MNE reads (sfp, elc, bvef, ...).
    if not isinstance(source, Path):
        return source
    if source.name.lower().endswith((".fif", ".fif.gz")):
        return mne.channels.read_dig_fif(source)
    # head_size=None keeps the file's own coordinates instead of rescaling to a 95 mm sphere.
    return mne.channels.read_custom_montage(source, head_size=None)


def restore_reference_channels(
    inst: Any, channels: tuple[str, ...], montage: str | Path | None = None
) -> Any:
    working = inst.copy().load_data()
    if not channels:
        return working
    overlap = set(channels) & set(working.ch_names)
    if overlap:
        raise ValueError(f"reference.add_channels: already present {sorted(overlap)}")
    if working.info["custom_ref_applied"]:
        raise ValueError(
            "reference.add_channels: restore acquisition electrodes before re-referencing; "
            "their signals cannot be recovered as zeros after a custom reference"
        )
    working = mne.add_reference_channels(working, list(channels), copy=False)
    if montage is not None:
        working.set_montage(_montage(montage), on_missing="raise")
    return working


def crop_raw(raw: Any, settings: CropSettings) -> Any:
    """Copy of a recording kept between ``settings.tmin`` and ``settings.tmax``.

    Times are seconds from the recording's first sample; a ``tmax`` beyond its
    last sample raises.
    """
    validate_raw(raw)
    if settings.tmax > raw.times[-1]:
        raise ValueError(f"crop.tmax: {settings.tmax} exceeds recording endpoint {raw.times[-1]}")
    return raw.copy().crop(settings.tmin, settings.tmax)


def annotate_raw(
    raw: Any,
    settings: AnnotationSettings,
    *,
    acquisition_first_samp: int | None = None,
    acquisition_n_times: int | None = None,
) -> Any:
    """Copy of a recording with the configured manual BAD spans added.

    Only ``settings.bad_spans`` are applied here; the amplitude, break and muscle
    detectors run in :func:`~eegtable.preprocessing.quality.detect_annotations`.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous recording, possibly already cropped.
    settings : AnnotationSettings
        Annotation settings; span onsets are seconds from the first acquired
        sample. A span past the end of the acquisition raises; the part of a span
        outside a cropped recording is dropped.
    acquisition_first_samp, acquisition_n_times : int, optional
        First sample and length of the original acquisition, when ``raw`` has
        been cropped since; None uses ``raw`` itself.

    Returns
    -------
    mne.io.Raw
    """
    # Spans are acquisition-relative; after a crop only the retained part is kept.
    validate_raw(raw)
    working = raw.copy()
    sfreq = raw.info["sfreq"]
    first_samp = raw.first_samp if acquisition_first_samp is None else acquisition_first_samp
    n_times = raw.n_times if acquisition_n_times is None else acquisition_n_times
    offset = (raw.first_samp - first_samp) / sfreq
    onsets, durations, descriptions = [], [], []
    for span in settings.bad_spans:
        if span.onset + span.duration > n_times / sfreq:
            raise ValueError("annotations.bad_spans: span exceeds acquisition duration")
        start = max(span.onset, offset)
        stop = min(span.onset + span.duration, offset + raw.n_times / sfreq)
        if start < stop:
            onsets.append(start - offset)
            durations.append(stop - start)
            descriptions.append(span.description)
    if onsets:
        normalized = attached_annotations(raw, mne.Annotations(onsets, durations, descriptions))
        working.annotations.append(
            normalized.onset,
            normalized.duration,
            normalized.description,
            ch_names=normalized.ch_names,
        )
    return working


def attached_annotations(raw: Any, annotations: Any) -> Any:
    # MNE's detectors return zero-based onsets on undated recordings and meas_date-relative
    # onsets on dated ones; set_annotations resolves both to the frame raw.annotations uses.
    return raw.copy().set_annotations(annotations).annotations


def acquisition_spans(
    raw: Any, annotations: Any, acquisition_first_samp: int
) -> list[dict[str, Any]]:
    # Seconds from the first acquired sample, the clock annotations.bad_spans is written in.
    attached = attached_annotations(raw, annotations)
    origin = acquisition_first_samp / raw.info["sfreq"]
    return [
        {
            "onset": float(onset - origin),
            "duration": float(duration),
            "description": str(description),
        }
        for onset, duration, description in zip(
            attached.onset, attached.duration, attached.description, strict=True
        )
    ]


def filter_coefficients(sfreq: float, settings: FilterSettings) -> NDArray[np.float64]:
    for value in (settings.l_freq, settings.h_freq):
        if value is not None and value >= sfreq / 2:
            raise ValueError(f"filter: cutoff {value} exceeds Nyquist {sfreq / 2}")
    return np.asarray(
        mne.filter.create_filter(
            None,
            sfreq,
            settings.l_freq,
            settings.h_freq,
            method="fir",
            phase="zero",
            fir_window="hamming",
            fir_design="firwin",
            verbose=False,
        ),
        dtype=float,
    )


def _validate_filter_support(raw: Any, support: int) -> None:
    intervals: list[tuple[int, int]] = []
    boundaries = {0, raw.n_times}
    for onset, duration, description in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description, strict=True
    ):
        if description.lower().startswith(("bad", "edge")):
            start, stop = np.round(
                (np.array([onset, onset + duration]) - raw.first_time) * raw.info["sfreq"]
            ).astype(int)
            start, stop = max(0, int(start)), min(raw.n_times, int(stop))
            boundaries.update((start, stop))
            if stop > start:
                intervals.append((start, stop))
    ordered = sorted(boundaries)
    for start, stop in zip(ordered[:-1], ordered[1:], strict=True):
        excluded = any(left <= start and stop <= right for left, right in intervals)
        if not excluded and 0 < stop - start < support:
            raise ValueError(
                f"filter: clean segment {start}:{stop} shorter than FIR support {support}"
            )


def _validate_jobs(n_jobs: int) -> None:
    if type(n_jobs) is not int or n_jobs == 0:
        raise ValueError("n_jobs: expected nonzero integer")


def filter_raw(raw: Any, settings: FilterSettings, *, n_jobs: int = 1) -> Any:
    """Zero-phase FIR band-, high- or low-pass filter of a copy's EEG, EOG and ECG.

    A Hamming-window ``firwin`` design, skipping BAD and edge annotations. A clean
    segment between them shorter than the filter raises.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous recording.
    settings : FilterSettings
        ``l_freq`` and ``h_freq`` in Hz, below Nyquist; both None returns an
        unfiltered copy.
    n_jobs : int, default 1
        Nonzero MNE filtering jobs.

    Returns
    -------
    mne.io.Raw
    """
    validate_raw(raw)
    _validate_jobs(n_jobs)
    working = raw.copy().load_data()
    if settings.l_freq is None and settings.h_freq is None:
        return working
    coefficients = filter_coefficients(raw.info["sfreq"], settings)
    _validate_filter_support(raw, len(coefficients))
    return working.filter(
        settings.l_freq,
        settings.h_freq,
        picks=physiology_names(raw),
        method="fir",
        phase="zero",
        fir_window="hamming",
        fir_design="firwin",
        pad="reflect_limited",
        skip_by_annotation=("edge", "bad"),
        n_jobs=n_jobs,
    )


def notch_coefficients(sfreq: float, centers: tuple[float, ...]) -> NDArray[np.float64]:
    # Mirrors Raw.notch_filter's FIR design: stop width freq/200, 1 Hz transition split
    # across both edges; the design fails if that support crosses 0 Hz or Nyquist.
    freqs = np.asarray(centers, dtype=float)
    widths = freqs / 200.0
    lower, upper = freqs - widths / 2 - 0.5, freqs + widths / 2 + 0.5
    if np.any(lower <= 0) or np.any(upper >= sfreq / 2):
        raise ValueError("filter.notch_freqs: transition support must lie inside Nyquist")
    return np.asarray(
        mne.filter.create_filter(
            None,
            sfreq,
            upper,
            lower,
            l_trans_bandwidth=0.5,
            h_trans_bandwidth=0.5,
            method="fir",
            phase="zero",
            fir_window="hamming",
            fir_design="firwin",
            verbose=False,
        ),
        dtype=float,
    )


def notch_raw(raw: Any, settings: FilterSettings, *, n_jobs: int = 1) -> Any:
    """Zero-phase FIR notch of a copy's EEG, EOG and ECG at ``settings.notch_freqs``.

    MNE's notch design, skipping BAD and edge annotations. Each notch's transition
    band must lie between 0 Hz and Nyquist, and a clean segment shorter than the
    filter raises. No notch frequencies returns an unfiltered copy. The notch is
    not recorded in the stored highpass or lowpass.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous recording.
    settings : FilterSettings
        Its ``notch_freqs``, in Hz, are used.
    n_jobs : int, default 1
        Nonzero MNE filtering jobs.

    Returns
    -------
    mne.io.Raw
    """
    validate_raw(raw)
    _validate_jobs(n_jobs)
    working = raw.copy().load_data()
    if not settings.notch_freqs:
        return working
    coefficients = notch_coefficients(raw.info["sfreq"], settings.notch_freqs)
    _validate_filter_support(raw, len(coefficients))
    return working.notch_filter(
        np.asarray(settings.notch_freqs),
        picks=physiology_names(raw),
        method="fir",
        phase="zero",
        fir_window="hamming",
        fir_design="firwin",
        pad="reflect_limited",
        skip_by_annotation=("edge", "bad"),
        n_jobs=n_jobs,
    )
