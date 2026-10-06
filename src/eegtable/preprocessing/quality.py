"""Inspectable diagnostics and explicit channel/segment decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mne  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from ._deps import require
from ._validation import names
from .config import AnnotationSettings, BadChannelSettings, BadSpan, StimulationSettings
from .raw import annotate_raw, good_eeg_names, physiology_names, require_names, validate_geometry


@dataclass(frozen=True)
class QualityCandidates:
    """Detector output for review; nothing in it has been applied.

    Parameters
    ----------
    bads : tuple of str
        Candidate bad channels.
    annotations : tuple of mne.Annotations
        Candidate BAD spans, one set per detector.
    evidence : dict
        Detector scores and verdicts.
    """

    bads: tuple[str, ...] = ()
    annotations: tuple[Any, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)


def detect_annotations(
    raw: Any, settings: AnnotationSettings, events: NDArray[np.int64] | None = None
) -> QualityCandidates:
    """Candidate BAD spans and bad channels from the configured annotation detectors.

    Runs MNE's amplitude detector on the good EEG channels, its break detector
    around ``events`` and its muscle z-score detector, each only when configured.
    Nothing is applied to ``raw``, and manual ``bad_spans`` are not included; see
    :func:`~eegtable.preprocessing.raw.annotate_raw`.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous recording.
    settings : AnnotationSettings
        Detector settings. The muscle band must lie below Nyquist and inside the
        recording's stored passband.
    events : ndarray of int, shape (n_events, 3), optional
        Task events, required by the break detector.

    Returns
    -------
    QualityCandidates
        Amplitude-detector channels as ``bads``, the detected annotations, and
        the amplitude channels and muscle scores as evidence.
    """
    annotations, bads = [], []
    evidence: dict[str, Any] = {}
    if settings.amplitude is not None:
        amplitude = settings.amplitude
        spans, channels = mne.preprocessing.annotate_amplitude(
            raw,
            peak=amplitude.peak,
            flat=amplitude.flat,
            bad_percent=amplitude.bad_percent,
            min_duration=amplitude.min_duration,
            picks=good_eeg_names(raw),
        )
        annotations.append(spans)
        bads.extend(channels)
        evidence["amplitude"] = channels
    if settings.breaks is not None:
        if events is None or not len(events):
            raise ValueError("annotations.breaks: explicit task events required")
        breaks = settings.breaks
        annotations.append(
            mne.preprocessing.annotate_break(
                raw,
                events=events,
                min_break_duration=breaks.min_break_duration,
                t_start_after_previous=breaks.t_start_after_previous,
                t_stop_before_next=breaks.t_stop_before_next,
            )
        )
    if settings.muscle is not None:
        muscle = settings.muscle
        low, high = muscle.filter_freq
        nyquist = raw.info["sfreq"] / 2
        if high >= nyquist:
            raise ValueError(
                f"annotations.muscle.filter_freq exceeds the {nyquist:g} Hz Nyquist frequency"
            )
        if low < raw.info["highpass"] or high > raw.info["lowpass"]:
            raise ValueError(
                "annotations.muscle.filter_freq: diagnostic band outside source passband"
            )
        spans, scores = mne.preprocessing.annotate_muscle_zscore(
            raw,
            ch_type="eeg",
            filter_freq=muscle.filter_freq,
            threshold=muscle.threshold,
            min_length_good=muscle.min_length_good,
        )
        annotations.append(spans)
        # MNE returns NaN for samples excluded by BAD annotations; JSON uses null.
        evidence["muscle_scores"] = np.where(np.isnan(scores), np.asarray(None), scores).tolist()
    return QualityCandidates(tuple(dict.fromkeys(bads)), tuple(annotations), evidence)


def detect_bad_channels(raw: Any, settings: BadChannelSettings) -> QualityCandidates:
    """Candidate bad EEG channels from PyPREP's ``NoisyChannels`` tests.

    Only the requested tests run, on a detrended copy of the good EEG channels with
    annotated segments omitted, after a notch on that copy when ``notch_freqs`` is
    set. Over ``repeats`` RANSAC draws, a channel is a candidate when a strict
    majority flags it. Nothing is marked bad. Requires
    ``eegtable[preprocessing-auto]``.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous recording. ``high_frequency`` needs a sampling rate above
        100 Hz and a stored lowpass of at least 50 Hz; RANSAC needs electrode
        positions and at least 16 good EEG channels.
    settings : BadChannelSettings
        Tests, RANSAC and its repeats, seed and diagnostic notch.

    Returns
    -------
    QualityCandidates
        Candidate ``bads`` and, as evidence, every draw's verdicts by test.
    """
    # Explicit methods only; find_all_bads would run unrequested tests.
    pyprep = require("pyprep", "preprocessing-auto")
    if "high_frequency" in settings.methods and (
        raw.info["sfreq"] <= 100 or raw.info["lowpass"] < 50
    ):
        raise ValueError("bad_channels.methods: high_frequency requires bandwidth through 50 Hz")
    if settings.ransac:
        validate_geometry(raw)
        if len(good_eeg_names(raw)) < 16:
            raise ValueError("bad_channels.ransac: at least 16 good EEG channels required")
    diagnostic = raw.copy().pick(good_eeg_names(raw))
    if settings.notch_freqs:
        # PREP removes line noise before its deviation test; the copy is diagnostic only.
        diagnostic.notch_filter(list(settings.notch_freqs), picks="eeg")
    repeats = []
    for repeat in range(settings.repeats):
        detector = pyprep.NoisyChannels(
            diagnostic,
            random_state=settings.random_state + repeat,
            do_detrend=True,
            reject_by_annotation="omit",
        )
        operations = {
            "flat": detector.find_bad_by_nan_flat,
            "deviation": detector.find_bad_by_deviation,
            "high_frequency": detector.find_bad_by_hfnoise,
            "correlation": detector.find_bad_by_correlation,
            "snr": detector.find_bad_by_SNR,
        }
        for method in ("flat", "deviation", "high_frequency", "correlation", "snr"):
            if method in settings.methods:
                operations[method]()
        if settings.ransac:
            detector.find_bad_by_ransac()
        repeats.append((tuple(detector.get_bads()), detector.get_bads(as_dict=True)))
    # A channel is a candidate when a strict majority of the repeated RANSAC draws agree.
    counts: dict[str, int] = {}
    for bads, _ in repeats:
        for name in bads:
            counts[name] = counts.get(name, 0) + 1
    majority = [name for name in diagnostic.ch_names if counts.get(name, 0) > len(repeats) / 2]
    return QualityCandidates(
        tuple(majority), evidence={"repeats": [details for _, details in repeats]}
    )


def detect_bridges(raw: Any) -> QualityCandidates:
    """Bridged EEG electrode pairs, as review evidence.

    Uses MNE's ``compute_bridged_electrodes``; nothing is marked bad. Requires
    finite nonzero positions for every EEG channel and at least four good ones.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous recording.

    Returns
    -------
    QualityCandidates
        No candidates; ``evidence["bridged"]`` lists each pair with its median
        electrical distance.
    """
    validate_geometry(raw)
    pairs, distances = mne.preprocessing.compute_bridged_electrodes(raw.copy())
    # MNE names pairs by channel index but indexes the distance matrix by EEG pick position.
    # That (epochs, ch, ch) matrix is NaN off its upper triangle and far too large for every
    # downstream checkpoint; keep each bridged pair with its median electrical distance.
    position = {int(index): k for k, index in enumerate(mne.pick_types(raw.info, eeg=True))}
    bridged = [
        {
            "pair": [raw.ch_names[i], raw.ch_names[j]],
            "electrical_distance": float(np.nanmedian(distances[:, position[i], position[j]])),
        }
        for i, j in pairs
    ]
    return QualityCandidates(evidence={"bridged": bridged})


def apply_raw_review(
    raw: Any,
    bads: tuple[str, ...],
    spans: tuple[BadSpan, ...],
    *,
    acquisition_first_samp: int | None = None,
    acquisition_n_times: int | None = None,
) -> Any:
    """Apply a raw review decision to a copy: set the bad channels, add BAD spans.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous recording.
    bads : tuple of str
        Replaces ``info["bads"]``; empty marks every channel good. Every name must
        exist, and at least one good EEG channel must remain.
    spans : tuple of BadSpan
        BAD spans to append, with onsets in seconds from the first acquired sample.
    acquisition_first_samp, acquisition_n_times : int, optional
        As for :func:`~eegtable.preprocessing.raw.annotate_raw`.

    Returns
    -------
    mne.io.Raw
    """
    names(bads, "review.raw.bads")
    require_names(raw, bads, "review.raw.bads")
    working = annotate_raw(
        raw,
        AnnotationSettings(bad_spans=spans),
        acquisition_first_samp=acquisition_first_samp,
        acquisition_n_times=acquisition_n_times,
    )
    working.info["bads"] = list(bads)
    good_eeg_names(working)
    return working


def _validate_stimulation_input(
    raw: Any, events: NDArray[np.int64], settings: StimulationSettings
) -> None:
    require_names(raw, settings.channels, "stimulation.channels")
    if not set(settings.channels) <= set(physiology_names(raw)):
        raise ValueError("stimulation.channels: only measured physiology channels may be repaired")
    if not set(settings.event_ids) <= set(events[:, 2]):
        raise ValueError("stimulation.event_ids: requested event code absent")


def stimulation_intervals(
    raw: Any, events: NDArray[np.int64], settings: StimulationSettings
) -> list[tuple[int, int]]:
    _validate_stimulation_input(raw, events, settings)
    # fix_stim_artifact ceils both bounds; the recorded windows must be the repaired ones.
    offsets = np.ceil(np.array([settings.tmin, settings.tmax]) * raw.info["sfreq"]).astype(int)
    if offsets[0] >= offsets[1]:
        raise ValueError("stimulation: repair window contains no samples")
    selected = events[np.isin(events[:, 2], settings.event_ids), 0]
    intervals = [(int(sample + offsets[0]), int(sample + offsets[1])) for sample in selected]
    for start, stop in intervals:
        if start <= raw.first_samp or stop >= raw.first_samp + raw.n_times - 1:
            raise ValueError(
                "stimulation: repair window requires neighboring samples inside recording"
            )
    if any(right[0] <= left[1] for left, right in zip(intervals[:-1], intervals[1:], strict=True)):
        raise ValueError("stimulation: overlapping repair windows")
    if settings.baseline is not None:
        baseline = np.ceil(np.array(settings.baseline) * raw.info["sfreq"]).astype(int)
        if baseline[0] >= baseline[1]:
            raise ValueError("stimulation.baseline: no samples")
        if any(
            sample + baseline[0] < raw.first_samp
            or sample + baseline[1] >= raw.first_samp + raw.n_times
            for sample in selected
        ):
            raise ValueError("stimulation.baseline: outside recording")
    return intervals


def repair_stimulation(raw: Any, events: NDArray[np.int64], settings: StimulationSettings) -> Any:
    """Repair stimulation artifacts on a copy with MNE's ``fix_stim_artifact``.

    The window ``[tmin, tmax]`` around every event with one of the configured codes
    is replaced on the named channels. Every code must occur, every window must hold
    samples and keep a neighboring sample inside the recording on both sides, and
    windows must not overlap.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous recording.
    events : ndarray of int, shape (n_events, 3)
        Events on the recording's sample grid.
    settings : StimulationSettings
        Event codes, EEG, EOG or ECG channels, window, mode and baseline.

    Returns
    -------
    mne.io.Raw
    """
    stimulation_intervals(raw, events, settings)
    working = raw.copy().load_data()
    for event_id in settings.event_ids:
        mne.preprocessing.fix_stim_artifact(
            working,
            events=events,
            event_id=event_id,
            tmin=settings.tmin,
            tmax=settings.tmax,
            mode=settings.mode,
            baseline=settings.baseline,
            picks=list(settings.channels),
        )
    return working
