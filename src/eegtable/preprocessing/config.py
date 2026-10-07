"""Validated preprocessing settings and a strict YAML boundary."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._validation import choice, integer, mapping, names, number, positive, thresholds

# Longest first, so a derived name loses ".fif.gz" whole.
SUFFIXES = (".fif.gz", ".fif", ".edf", ".bdf", ".vhdr", ".set")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class FilterSettings:
    """Continuous filtering of the EEG, EOG and ECG channels.

    Parameters
    ----------
    l_freq : float, optional
        High-pass edge in Hz, below ``h_freq``. With ``h_freq`` also None, the filter
        stage is skipped.
    h_freq : float, optional
        Low-pass edge in Hz. Decimation requires it.
    notch_freqs : tuple of float
        Line frequencies to notch, in Hz, unique and ascending. Empty skips the notch
        stage.
    """

    l_freq: float | None = None
    h_freq: float | None = None
    notch_freqs: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for name in ("l_freq", "h_freq"):
            value = getattr(self, name)
            if value is not None:
                positive(value, f"filter.{name}")
        if self.l_freq is not None and self.h_freq is not None and self.l_freq >= self.h_freq:
            raise ValueError("filter.l_freq: must be below h_freq")
        if not isinstance(self.notch_freqs, tuple):
            raise TypeError("filter.notch_freqs: expected tuple")
        for value in self.notch_freqs:
            positive(value, "filter.notch_freqs")
        if tuple(sorted(set(self.notch_freqs))) != self.notch_freqs:
            raise ValueError("filter.notch_freqs: must be unique and ascending")


@dataclass(frozen=True)
class BipolarSettings:
    """A bipolar EOG or ECG channel derived from two recorded channels.

    Parameters
    ----------
    name : str
        Name of the new channel, which must not exist yet.
    anode, cathode : str
        EEG, EOG or ECG channels; the new channel is ``anode - cathode`` and both
        sources are kept.
    type : {"eog", "ecg"}
        Channel type of the new channel.
    """

    name: str
    anode: str
    cathode: str
    type: str

    def __post_init__(self) -> None:
        names((self.name, self.anode, self.cathode), "channels.bipolar")
        choice(self.type, ("eog", "ecg"), "channels.bipolar.type")


@dataclass(frozen=True)
class ChannelSettings:
    """Channel preparation: names, types, derivations, drops, montage and bad labels.

    Parameters
    ----------
    rename : mapping of str to str
        Old to new channel names, applied first; the new names must not collide.
    types : mapping of str to str
        Channel name, after renaming, to MNE channel type.
    drop : tuple of str
        Channels removed after the bipolar derivations.
    bads : tuple of str
        Channels added to the recording's bad labels.
    montage : str or pathlib.Path, optional
        A standard MNE montage name, or a digitized FIF or any electrode file
        ``mne.channels.read_custom_montage`` reads, positions taken as written.
        Every EEG channel must have a position in it.
    interpolate_bads : bool, default False
        Spline-interpolate the bad EEG channels of the epochs after rejection and
        review; requires finite positions for every EEG channel.
    bipolar : tuple of BipolarSettings
        Bipolar channels to derive.
    projections : {"error", "apply", "discard-inactive"}, default "error"
        What to do with inactive projectors in the recording: refuse it, apply them,
        or delete them.
    """

    rename: Mapping[str, str] = field(default_factory=dict)
    types: Mapping[str, str] = field(default_factory=dict)
    drop: tuple[str, ...] = ()
    bads: tuple[str, ...] = ()
    montage: str | Path | None = None
    interpolate_bads: bool = False
    bipolar: tuple[BipolarSettings, ...] = ()
    projections: str = "error"

    def __post_init__(self) -> None:
        for key in ("drop", "bads"):
            names(getattr(self, key), f"channels.{key}")
        for key in ("rename", "types"):
            values = getattr(self, key)
            if not isinstance(values, Mapping):
                raise TypeError(f"channels.{key}: expected mapping")
            names(tuple(values), f"channels.{key}")
            if any(not isinstance(value, str) or not value for value in values.values()):
                raise ValueError(f"channels.{key}: values must be nonempty strings")
        names(tuple(self.rename.values()), "channels.rename")
        if self.montage is not None and not isinstance(self.montage, (str, Path)):
            raise TypeError("channels.montage: expected a standard name or Path")
        if type(self.interpolate_bads) is not bool:
            raise TypeError("channels.interpolate_bads: expected boolean")
        choice(self.projections, ("error", "apply", "discard-inactive"), "channels.projections")
        if not isinstance(self.bipolar, tuple) or any(
            not isinstance(item, BipolarSettings) for item in self.bipolar
        ):
            raise TypeError("channels.bipolar: expected tuple of BipolarSettings")
        names(tuple(item.name for item in self.bipolar), "channels.bipolar")


@dataclass(frozen=True)
class BadSpan:
    """A manual BAD interval.

    Parameters
    ----------
    onset : float
        Nonnegative seconds from the first acquired sample, unchanged by cropping.
    duration : float
        Positive length in seconds.
    description : str
        Annotation label, starting with ``BAD`` in any case.
    """

    onset: float
    duration: float
    description: str

    def __post_init__(self) -> None:
        number(self.onset, "annotations.bad_spans.onset", 0)
        positive(self.duration, "annotations.bad_spans.duration")
        if not isinstance(self.description, str) or not self.description.upper().startswith("BAD"):
            raise ValueError("annotations.bad_spans.description: must start with BAD")


@dataclass(frozen=True)
class AmplitudeSettings:
    """MNE's amplitude detector, run on the good EEG channels.

    Parameters
    ----------
    peak : mapping of str to float, optional
        ``{"eeg": volts}``: peak-to-peak amplitude above which spans are marked.
    flat : mapping of str to float, optional
        ``{"eeg": volts}``: peak-to-peak amplitude below which spans are marked. At
        least one of ``peak`` and ``flat`` is required.
    bad_percent : float, default 5.0
        Percentage of the recording, in ``[0, 100]``, beyond which a channel breaking
        a threshold is suggested bad instead of having its spans marked.
    min_duration : float, default 0.005
        Seconds a threshold must be broken for in a row to count.
    """

    peak: Mapping[str, float] | None = None
    flat: Mapping[str, float] | None = None
    bad_percent: float = 5.0
    min_duration: float = 0.005

    def __post_init__(self) -> None:
        for name in ("peak", "flat"):
            value = getattr(self, name)
            thresholds(value, f"annotations.amplitude.{name}")
            if value is not None and set(value) != {"eeg"}:
                raise ValueError(f"annotations.amplitude.{name}: only eeg is supported")
        if self.peak is None and self.flat is None:
            raise ValueError("annotations.amplitude: peak or flat required")
        number(self.bad_percent, "annotations.amplitude.bad_percent", 0)
        if self.bad_percent > 100:
            raise ValueError("annotations.amplitude.bad_percent: must be <= 100")
        positive(self.min_duration, "annotations.amplitude.min_duration")


@dataclass(frozen=True)
class BreakSettings:
    """MNE's detector of breaks between task events; requires event epochs.

    Parameters
    ----------
    min_break_duration : float
        Shortest gap between events, in seconds, treated as a break.
    t_start_after_previous : float
        Seconds after the previous event before a break span starts.
    t_stop_before_next : float
        Seconds before the next event at which a break span stops. The two margins
        must sum to less than ``min_break_duration``.
    """

    min_break_duration: float
    t_start_after_previous: float
    t_stop_before_next: float

    def __post_init__(self) -> None:
        positive(self.min_break_duration, "annotations.breaks.min_break_duration")
        number(self.t_start_after_previous, "annotations.breaks.t_start_after_previous", 0)
        number(self.t_stop_before_next, "annotations.breaks.t_stop_before_next", 0)
        if self.t_start_after_previous + self.t_stop_before_next >= self.min_break_duration:
            raise ValueError("annotations.breaks: margins must sum to less than minimum duration")


@dataclass(frozen=True)
class MuscleSettings:
    """MNE's muscle z-score detector, run on the EEG channels.

    Parameters
    ----------
    filter_freq : tuple of float
        Ascending band in Hz whose envelope is z-scored. It must lie below Nyquist
        and inside the recording's stored passband.
    threshold : float
        Z-score above which spans are marked.
    min_length_good : float
        Shortest good stretch, in seconds, kept between marked spans.
    """

    filter_freq: tuple[float, float]
    threshold: float
    min_length_good: float

    def __post_init__(self) -> None:
        if not isinstance(self.filter_freq, tuple) or len(self.filter_freq) != 2:
            raise ValueError("annotations.muscle.filter_freq: expected pair")
        for value in self.filter_freq:
            positive(value, "annotations.muscle.filter_freq")
        if self.filter_freq[0] >= self.filter_freq[1]:
            raise ValueError("annotations.muscle.filter_freq: expected ordered band")
        positive(self.threshold, "annotations.muscle.threshold")
        number(self.min_length_good, "annotations.muscle.min_length_good", 0)


@dataclass(frozen=True)
class AnnotationSettings:
    """Manual BAD spans and the annotation detectors.

    Manual spans are applied directly. Detector spans are candidates that take
    effect only through the raw review; a detector set to None does not run.

    Parameters
    ----------
    bad_spans : tuple of BadSpan
        Manual BAD intervals.
    amplitude : AmplitudeSettings, optional
        Amplitude detector.
    breaks : BreakSettings, optional
        Break detector; requires event epochs.
    muscle : MuscleSettings, optional
        Muscle detector.
    """

    bad_spans: tuple[BadSpan, ...] = ()
    amplitude: AmplitudeSettings | None = None
    breaks: BreakSettings | None = None
    muscle: MuscleSettings | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.bad_spans, tuple) or any(
            not isinstance(span, BadSpan) for span in self.bad_spans
        ):
            raise TypeError("annotations.bad_spans: expected tuple of BadSpan")
        for name, expected in (
            ("amplitude", AmplitudeSettings),
            ("breaks", BreakSettings),
            ("muscle", MuscleSettings),
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"annotations.{name}: expected {expected.__name__}")


@dataclass(frozen=True)
class CropSettings:
    """Span of the continuous recording to keep.

    Parameters
    ----------
    tmin, tmax : float
        Nonnegative bounds in seconds from the start of the recording, with ``tmin``
        below ``tmax``.
    """

    tmin: float
    tmax: float

    def __post_init__(self) -> None:
        number(self.tmin, "crop.tmin", 0)
        number(self.tmax, "crop.tmax", 0)
        if self.tmin >= self.tmax:
            raise ValueError("crop.tmax: must exceed tmin")


@dataclass(frozen=True)
class EventSettings:
    """Where event-related epochs get their events.

    Parameters
    ----------
    source : {"annotations", "stim", "file"}
        Read events from the annotations, a stim channel, or an MNE event file.
    event_id : mapping of str to int
        Event names and their unique positive codes; every code must occur.
    stim_channel : str, optional
        Stim channel; required for ``source="stim"`` and allowed only there.
    path : pathlib.Path, optional
        MNE event file read by ``mne.read_events``; required for ``source="file"``
        and allowed only there.
    shortest_event : int, default 2
        Shortest stim-channel event, in samples; ``source="stim"`` only.
    min_duration : float, default 0.0
        Shortest stim-channel event, in seconds; ``source="stim"`` only.
    delay : float, default 0.0
        Seconds subtracted from every event, rounded to samples, so a positive delay
        moves events earlier.
    """

    source: str
    event_id: Mapping[str, int]
    stim_channel: str | None = None
    path: Path | None = None
    shortest_event: int = 2
    min_duration: float = 0.0
    delay: float = 0.0

    def __post_init__(self) -> None:
        choice(self.source, ("annotations", "stim", "file"), "epochs.events.source")
        if not isinstance(self.event_id, Mapping) or not self.event_id:
            raise ValueError("epochs.events.event_id: nonempty mapping required")
        names(tuple(self.event_id), "epochs.events.event_id")
        for value in self.event_id.values():
            integer(value, "epochs.events.event_id", 1)
        if len(set(self.event_id.values())) != len(self.event_id):
            raise ValueError("epochs.events.event_id: duplicate event codes")
        integer(self.shortest_event, "epochs.events.shortest_event", 1)
        number(self.min_duration, "epochs.events.min_duration", 0)
        number(self.delay, "epochs.events.delay")
        if self.source == "stim":
            names((self.stim_channel,), "epochs.events.stim_channel")
        elif self.stim_channel is not None or self.shortest_event != 2 or self.min_duration != 0:
            raise ValueError("epochs.events: stim settings require source=stim")
        if (self.source == "file") != isinstance(self.path, Path):
            raise ValueError("epochs.events.path: required only for source=file")


def _validate_epoch_common(baseline: object, padding: float, detrend: str | None) -> None:
    number(padding, "epochs.padding", 0)
    choice(detrend, (None, "constant", "linear"), "epochs.detrend")
    if baseline is not None:
        if not isinstance(baseline, tuple) or len(baseline) != 2:
            raise ValueError("epochs.baseline: expected pair or null")
        for value in baseline:
            if value is not None:
                number(value, "epochs.baseline")
        if all(value is not None for value in baseline) and baseline[0] > baseline[1]:
            raise ValueError("epochs.baseline: expected ordered bounds")


@dataclass(frozen=True)
class EventEpochSettings:
    """Epochs around events.

    Parameters
    ----------
    events : EventSettings
        Event source.
    tmin, tmax : float
        Analysis window around each event, in seconds, with ``tmin`` below ``tmax``.
    baseline : tuple of (float or None, float or None), optional
        Baseline interval applied to the final epochs; None applies none.
    padding : float, default 0.0
        Seconds cut on both sides for filtering and resampling, then removed. An
        epoch whose padding overlaps a BAD span or leaves the recording is dropped.
    detrend : {"constant", "linear"}, optional
        Detrending of the final EEG.
    metadata : pathlib.Path, optional
        TSV with one row per resolved event, in event order before any rejection,
        attached as epoch metadata.
    """

    events: EventSettings
    tmin: float
    tmax: float
    baseline: tuple[float | None, float | None] | None = None
    padding: float = 0.0
    detrend: str | None = None
    metadata: Path | None = None
    kind: str = field(default="events", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.events, EventSettings):
            raise TypeError("epochs.events: expected EventSettings")
        number(self.tmin, "epochs.tmin")
        number(self.tmax, "epochs.tmax")
        if self.tmin >= self.tmax:
            raise ValueError("epochs.tmax: must exceed tmin")
        if self.metadata is not None and not isinstance(self.metadata, Path):
            raise TypeError("epochs.metadata: expected Path")
        _validate_epoch_common(self.baseline, self.padding, self.detrend)


@dataclass(frozen=True)
class FixedEpochSettings:
    """Consecutive fixed-length epochs.

    Parameters
    ----------
    duration : float
        Epoch length in seconds; it must align with acquisition samples and hold at
        least two.
    overlap : float, default 0.0
        Overlap of consecutive epochs in seconds, below ``duration``. The stride
        ``duration - overlap`` must align with acquisition samples.
    start : float, default 0.0
        First epoch onset, in seconds from the start of the recording.
    stop : float, optional
        End of the epoched span, in seconds; no epoch extends past it. None runs to
        the end of the recording.
    baseline : tuple of (float or None, float or None), optional
        Baseline interval applied to the final epochs; None applies none.
    padding : float, default 0.0
        Seconds cut on both sides for filtering and resampling, then removed.
    detrend : {"constant", "linear"}, optional
        Detrending of the final EEG.
    """

    duration: float
    overlap: float = 0.0
    start: float = 0.0
    stop: float | None = None
    baseline: tuple[float | None, float | None] | None = None
    padding: float = 0.0
    detrend: str | None = None
    kind: str = field(default="fixed", init=False)

    def __post_init__(self) -> None:
        positive(self.duration, "epochs.duration")
        number(self.overlap, "epochs.overlap", 0)
        number(self.start, "epochs.start", 0)
        if self.overlap >= self.duration:
            raise ValueError("epochs.overlap: must be less than duration")
        if self.stop is not None:
            number(self.stop, "epochs.stop", 0)
            if self.stop <= self.start:
                raise ValueError("epochs.stop: must exceed start")
        _validate_epoch_common(self.baseline, self.padding, self.detrend)


@dataclass(frozen=True)
class ReferenceSettings:
    """Final EEG reference of the epochs.

    Parameters
    ----------
    channels : "average" or tuple of str, optional
        ``"average"`` of the good EEG channels, or names of good EEG channels. None
        leaves the reference unchanged.
    add_channels : tuple of str
        Missing acquisition-reference electrodes, restored as zero channels before
        the first re-reference, including an artifact reference, and placed with the
        configured montage. Requires ``channels``.
    """

    channels: str | tuple[str, ...] | None = None
    add_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.channels, str):
            choice(self.channels, ("average",), "reference.channels")
        elif self.channels is not None:
            names(self.channels, "reference.channels")
            if not self.channels:
                raise ValueError("reference.channels: must not be empty")
        names(self.add_channels, "reference.add_channels")
        if self.add_channels and self.channels is None:
            raise ValueError("reference.add_channels: requires final reference")


@dataclass(frozen=True)
class ThresholdSettings:
    """Peak-to-peak epoch rejection.

    Parameters
    ----------
    reject : mapping of str to float, optional
        Largest peak-to-peak amplitude, in volts, by channel type (``eeg``, ``eog``
        or ``ecg``).
    flat : mapping of str to float, optional
        Smallest peak-to-peak amplitude, in volts, by channel type; below ``reject``
        for the same type. At least one of ``reject`` and ``flat`` is required.
    tmin, tmax : float, optional
        Window judged, in seconds, inside the analysis window; None uses its bound.
    """

    reject: Mapping[str, float] | None = None
    flat: Mapping[str, float] | None = None
    tmin: float | None = None
    tmax: float | None = None
    method: str = field(default="thresholds", init=False)

    def __post_init__(self) -> None:
        thresholds(self.reject, "rejection.reject")
        thresholds(self.flat, "rejection.flat")
        if self.reject is None and self.flat is None:
            raise ValueError("rejection: reject or flat required")
        for name in ("tmin", "tmax"):
            value = getattr(self, name)
            if value is not None:
                number(value, f"rejection.{name}")
        if self.tmin is not None and self.tmax is not None and self.tmin > self.tmax:
            raise ValueError("rejection.tmin: must not exceed tmax")
        for key in (self.reject or {}).keys() & (self.flat or {}).keys():
            if self.flat[key] >= self.reject[key]:  # type: ignore[index]
                raise ValueError(f"rejection.flat.{key}: must be below reject")


@dataclass(frozen=True)
class AutoRejectSettings:
    """Autoreject, fitted on the analysis window of the epochs.

    Parameters
    ----------
    n_interpolate : tuple of int
        Candidate numbers of channels to interpolate per epoch, each below the
        number of good EEG channels.
    consensus : tuple of float
        Candidate fractions of bad channels, in ``[0, 1]``, at which an epoch is
        dropped.
    cv : int
        Cross-validation folds, at least 2 and at most the number of epochs.
    random_state : int, default 42
        Seed.
    """

    n_interpolate: tuple[int, ...]
    consensus: tuple[float, ...]
    cv: int
    random_state: int = 42
    method: str = field(default="autoreject", init=False)

    def __post_init__(self) -> None:
        integer(self.cv, "rejection.cv", 2)
        integer(self.random_state, "rejection.random_state")
        if not self.n_interpolate or not self.consensus:
            raise ValueError("rejection: nonempty interpolation and consensus grids required")
        for value in self.n_interpolate:
            integer(value, "rejection.n_interpolate")
        for fraction in self.consensus:
            number(fraction, "rejection.consensus", 0)
            if fraction > 1:
                raise ValueError("rejection.consensus: must be <= 1")


@dataclass(frozen=True)
class DecimationSettings:
    """Integer decimation of the epochs.

    Parameters
    ----------
    factor : int
        Keep every ``factor``-th sample; at least 2. The low-pass ``filter.h_freq``
        must be at most a third of the new rate, with its transition ending below the
        new Nyquist.
    """

    factor: int
    method: str = field(default="decimate", init=False)

    def __post_init__(self) -> None:
        integer(self.factor, "sampling.factor", 2)


@dataclass(frozen=True)
class ResamplingSettings:
    """Polyphase downsampling of the padded epochs.

    Parameters
    ----------
    sfreq : float
        Target rate in Hz, below the current one. The padded epoch must map to a
        whole number of samples and the epoch origin must lie on the new grid.
    padding : float
        Must equal ``epochs.padding`` and cover the anti-aliasing filter's
        half-support.
    """

    sfreq: float
    padding: float
    method: str = field(default="resample", init=False)

    def __post_init__(self) -> None:
        positive(self.sfreq, "sampling.sfreq")
        positive(self.padding, "sampling.padding")


ICLABEL_CLASSES = (
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
)


@dataclass(frozen=True)
class ICLabelSettings:
    """ICLabel classification of ICA components, as review evidence.

    Parameters
    ----------
    threshold : float, default 0.8
        Probability, in ``[0, 1]``, at or above which a component whose winning class
        is outside ``keep`` is suggested for exclusion.
    keep : tuple of str
        ICLabel classes never suggested for exclusion. Default
        ``("brain", "other")``.
    """

    threshold: float = 0.8
    keep: tuple[str, ...] = ("brain", "other")

    def __post_init__(self) -> None:
        number(self.threshold, "artifact.ica.iclabel.threshold", 0)
        if self.threshold > 1:
            raise ValueError("artifact.ica.iclabel.threshold: must be <= 1")
        names(self.keep, "artifact.ica.iclabel.keep")
        for label in self.keep:
            choice(label, ICLABEL_CLASSES, "artifact.ica.iclabel.keep")


@dataclass(frozen=True)
class ICASettings:
    """ICA, fitted on a high-passed training copy of the continuous data.

    Parameters
    ----------
    l_freq : float, default 1.0
        High-pass of the training copy, in Hz, applied only when the data's own is
        lower.
    n_components : int, optional
        Components to fit, at least 2 and at most the EEG rank; None uses the rank.
    random_state : int, default 42
        Seed.
    max_iter : int, default 1000
        Iteration limit. FastICA or Picard reaching it raises as unconverged; MNE's
        infomax reports no convergence, so an infomax fit stops there without raising.
    reject, flat : mapping of str to float, optional
        Peak-to-peak limits in volts, by channel type, for the training segments.
    tstep : float, default 2.0
        Training segment length, in seconds.
    eog_channels : tuple of str
        Channels for MNE's EOG component detector.
    ecg_channel : str, optional
        Channel for MNE's ECG component detector.
    method : {"fastica", "infomax", "picard"}, default "fastica"
        ``"infomax"`` uses the extended update; ``"picard"`` runs with
        ``ortho=False`` and ``extended=True``.
    iclabel : ICLabelSettings, optional
        Requires ``"infomax"`` or ``"picard"`` and an average artifact reference.
    """

    l_freq: float = 1.0
    n_components: int | None = None
    random_state: int = 42
    max_iter: int = 1000
    reject: Mapping[str, float] | None = None
    flat: Mapping[str, float] | None = None
    tstep: float = 2.0
    eog_channels: tuple[str, ...] = ()
    ecg_channel: str | None = None
    method: str = "fastica"
    iclabel: ICLabelSettings | None = None

    def __post_init__(self) -> None:
        choice(self.method, ("fastica", "infomax", "picard"), "artifact.ica.method")
        if self.iclabel is not None and not isinstance(self.iclabel, ICLabelSettings):
            raise TypeError("artifact.ica.iclabel: expected ICLabelSettings")
        if self.iclabel is not None and self.method == "fastica":
            # ICLabel was trained on extended infomax; picard's extended mode approximates it.
            raise ValueError("artifact.ica.iclabel: requires method infomax or picard")
        positive(self.l_freq, "artifact.ica.l_freq")
        if self.n_components is not None:
            integer(self.n_components, "artifact.ica.n_components", 2)
        integer(self.random_state, "artifact.ica.random_state")
        integer(self.max_iter, "artifact.ica.max_iter", 1)
        positive(self.tstep, "artifact.ica.tstep")
        thresholds(self.reject, "artifact.ica.reject")
        thresholds(self.flat, "artifact.ica.flat")
        names(self.eog_channels, "artifact.ica.eog_channels")
        if self.ecg_channel is not None:
            names((self.ecg_channel,), "artifact.ica.ecg_channel")


@dataclass(frozen=True)
class SSPSettings:
    """EEG projectors computed around EOG and ECG events.

    Parameters
    ----------
    n_eeg : int
        Projectors per artifact channel, at least 1.
    eog_channels : tuple of str
        EOG channels whose events train projectors.
    ecg_channel : str, optional
        ECG channel whose events train projectors. At least one artifact channel is
        required.
    l_freq : float, default 1.0
        Lower edge of the training band, in Hz.
    h_freq : float, default 35.0
        Upper edge of the training band, in Hz, below Nyquist.
    tmin : float, default -0.2
        Start of the window around each event, in seconds.
    tmax : float, default 0.5
        End of that window, in seconds.
    reject : mapping of str to float, optional
        Peak-to-peak limits in volts, by channel type, for the training epochs.
    """

    n_eeg: int
    eog_channels: tuple[str, ...] = ()
    ecg_channel: str | None = None
    l_freq: float = 1.0
    h_freq: float = 35.0
    tmin: float = -0.2
    tmax: float = 0.5
    reject: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        integer(self.n_eeg, "artifact.ssp.n_eeg", 1)
        names(self.eog_channels, "artifact.ssp.eog_channels")
        if self.ecg_channel is not None:
            names((self.ecg_channel,), "artifact.ssp.ecg_channel")
        if not self.eog_channels and self.ecg_channel is None:
            raise ValueError("artifact.ssp: an artifact channel is required")
        positive(self.l_freq, "artifact.ssp.l_freq")
        positive(self.h_freq, "artifact.ssp.h_freq")
        number(self.tmin, "artifact.ssp.tmin")
        number(self.tmax, "artifact.ssp.tmax")
        if self.l_freq >= self.h_freq or self.tmin >= self.tmax:
            raise ValueError("artifact.ssp: invalid filter/time bounds")
        thresholds(self.reject, "artifact.ssp.reject")


@dataclass(frozen=True)
class RegressionSettings:
    """EOG regression of the EEG.

    Parameters
    ----------
    eog_channels : tuple of str
        EOG channels regressed out; at least one.
    tstep : float, default 2.0
        Training segment length, in seconds.
    reject, flat : mapping of str to float, optional
        Peak-to-peak limits in volts, by channel type, for the training segments.
    """

    eog_channels: tuple[str, ...]
    tstep: float = 2.0
    reject: Mapping[str, float] | None = None
    flat: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        names(self.eog_channels, "artifact.regression.eog_channels")
        if not self.eog_channels:
            raise ValueError("artifact.regression.eog_channels: must not be empty")
        positive(self.tstep, "artifact.regression.tstep")
        thresholds(self.reject, "artifact.regression.reject")
        thresholds(self.flat, "artifact.regression.flat")


@dataclass(frozen=True)
class ArtifactSettings:
    """Artifact correction, fitted on the continuous data and applied to the epochs.

    Parameters
    ----------
    method : {"ica", "ssp", "regression"}
        Kind of operator.
    settings : ICASettings, SSPSettings or RegressionSettings
        Settings matching ``method``.
    reference : "average" or tuple of str, optional
        EEG reference applied to the continuous data before fitting; the epochs are
        cut from this re-referenced data. Required for regression, and ``"average"``
        for ICLabel. None fits on the data as referenced.
    """

    method: str
    settings: ICASettings | SSPSettings | RegressionSettings
    reference: str | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        expected = {"ica": ICASettings, "ssp": SSPSettings, "regression": RegressionSettings}
        choice(self.method, tuple(expected), "artifact.method")
        if not isinstance(self.settings, expected[self.method]):
            raise TypeError(f"artifact.{self.method}: settings do not match method")
        ReferenceSettings(channels=self.reference)
        if self.method == "regression" and self.reference is None:
            raise ValueError("artifact.reference: regression requires explicit reference")
        if (
            isinstance(self.settings, ICASettings)
            and self.settings.iclabel is not None
            and self.reference != "average"
        ):
            raise ValueError("artifact.ica.iclabel: requires artifact.reference: average")


@dataclass(frozen=True)
class BadChannelSettings:
    """PyPREP bad-channel detection, whose candidates go to the raw review.

    Parameters
    ----------
    methods : tuple of str
        Tests among ``"flat"``, ``"deviation"``, ``"correlation"``,
        ``"high_frequency"`` and ``"snr"``. ``"snr"`` requires ``"correlation"`` and
        ``"high_frequency"``. Default ``("flat", "deviation", "correlation")``.
    random_state : int, default 42
        Detector seed; draw ``k`` of the repeats uses ``random_state + k``.
    ransac : bool, default False
        Also run RANSAC, which needs electrode positions and at least 16 good EEG
        channels.
    repeats : int, default 1
        RANSAC draws that vote; a channel is a candidate when a strict majority flags
        it. More than one requires ``ransac``.
    notch_freqs : tuple of float
        Line frequencies, in Hz, notched on the diagnostic copy only.
    method : {"pyprep"}, default "pyprep"
        Detector.
    """

    methods: tuple[str, ...] = ("flat", "deviation", "correlation")
    random_state: int = 42
    ransac: bool = False
    repeats: int = 1
    notch_freqs: tuple[float, ...] = ()
    method: str = "pyprep"

    def __post_init__(self) -> None:
        choice(self.method, ("pyprep",), "bad_channels.method")
        integer(self.repeats, "bad_channels.repeats", 1)
        if self.repeats > 1 and not self.ransac:
            raise ValueError("bad_channels.repeats: only RANSAC is stochastic; repeats need ransac")
        if not isinstance(self.notch_freqs, tuple):
            raise TypeError("bad_channels.notch_freqs: expected tuple")
        for value in self.notch_freqs:
            positive(value, "bad_channels.notch_freqs")
        names(self.methods, "bad_channels.methods")
        if not self.methods:
            raise ValueError("bad_channels.methods: must not be empty")
        for method in self.methods:
            choice(
                method,
                ("flat", "deviation", "correlation", "high_frequency", "snr"),
                "bad_channels.methods",
            )
        if "snr" in self.methods and not {"high_frequency", "correlation"} <= set(self.methods):
            raise ValueError("bad_channels.methods: snr requires high_frequency and correlation")
        integer(self.random_state, "bad_channels.random_state")
        if type(self.ransac) is not bool:
            raise TypeError("bad_channels.ransac: expected boolean")


@dataclass(frozen=True)
class StimulationSettings:
    """Stimulation-artifact repair with MNE's ``fix_stim_artifact``.

    Parameters
    ----------
    event_ids : tuple of int
        Unique positive codes of the stimulation events; each must occur.
    channels : tuple of str
        EEG, EOG or ECG channels to repair.
    tmin, tmax : float
        Repair window around each event, in seconds, with ``tmin`` below ``tmax``.
    mode : {"linear", "window", "constant"}
        How the window is replaced: linear interpolation, a tapering window, or a
        constant from ``baseline``.
    baseline : tuple of float, optional
        Interval around each event, in seconds, whose mean ``"constant"`` uses;
        required by that mode.
    """

    event_ids: tuple[int, ...]
    channels: tuple[str, ...]
    tmin: float
    tmax: float
    mode: str
    baseline: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.event_ids or len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("stimulation.event_ids: nonempty unique IDs required")
        for value in self.event_ids:
            integer(value, "stimulation.event_ids", 1)
        names(self.channels, "stimulation.channels")
        if not self.channels:
            raise ValueError("stimulation.channels: explicit channels required")
        number(self.tmin, "stimulation.tmin")
        number(self.tmax, "stimulation.tmax")
        if self.tmin >= self.tmax:
            raise ValueError("stimulation.tmax: must exceed tmin")
        choice(self.mode, ("linear", "window", "constant"), "stimulation.mode")
        if self.mode == "constant" and self.baseline is None:
            raise ValueError("stimulation.baseline: required for constant mode")
        if self.baseline is not None:
            if len(self.baseline) != 2:
                raise ValueError("stimulation.baseline: expected pair")
            for bound in self.baseline:
                number(bound, "stimulation.baseline")
            if self.baseline[0] >= self.baseline[1]:
                raise ValueError("stimulation.baseline: expected ordered bounds")


@dataclass(frozen=True)
class ProcessingSettings:
    """Numerical settings of every preprocessing stage.

    An optional stage set to None, or left empty, does not run.

    Parameters
    ----------
    epochs : EventEpochSettings or FixedEpochSettings
        Epoch construction.
    channels : ChannelSettings
        Channel preparation.
    annotations : AnnotationSettings
        Manual BAD spans and annotation detectors; breaks require event epochs.
    filter : FilterSettings
        Continuous notch and pass-band filtering.
    reference : ReferenceSettings
        Final EEG reference.
    crop : CropSettings, optional
        Span of the continuous recording to keep.
    artifact : ArtifactSettings, optional
        Artifact correction.
    rejection : ThresholdSettings or AutoRejectSettings, optional
        Epoch rejection.
    sampling : DecimationSettings or ResamplingSettings, optional
        Final sampling rate. Decimation requires ``filter.h_freq``; resampling
        padding must equal ``epochs.padding``.
    bad_channels : BadChannelSettings, optional
        PyPREP channel detection.
    bridges : bool, default False
        Detect bridged electrodes as raw-review evidence.
    stimulation : StimulationSettings, optional
        Stimulation-artifact repair.
    """

    epochs: EventEpochSettings | FixedEpochSettings
    channels: ChannelSettings = field(default_factory=ChannelSettings)
    annotations: AnnotationSettings = field(default_factory=AnnotationSettings)
    filter: FilterSettings = field(default_factory=FilterSettings)
    reference: ReferenceSettings = field(default_factory=ReferenceSettings)
    crop: CropSettings | None = None
    artifact: ArtifactSettings | None = None
    rejection: ThresholdSettings | AutoRejectSettings | None = None
    sampling: DecimationSettings | ResamplingSettings | None = None
    bad_channels: BadChannelSettings | None = None
    bridges: bool = False
    stimulation: StimulationSettings | None = None

    def __post_init__(self) -> None:
        expected_types = {
            "epochs": (EventEpochSettings, FixedEpochSettings),
            "channels": (ChannelSettings,),
            "annotations": (AnnotationSettings,),
            "filter": (FilterSettings,),
            "reference": (ReferenceSettings,),
            "crop": (CropSettings, type(None)),
            "artifact": (ArtifactSettings, type(None)),
            "rejection": (ThresholdSettings, AutoRejectSettings, type(None)),
            "sampling": (DecimationSettings, ResamplingSettings, type(None)),
            "bad_channels": (BadChannelSettings, type(None)),
            "stimulation": (StimulationSettings, type(None)),
        }
        for name, expected in expected_types.items():
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name}: invalid settings type")
        if type(self.bridges) is not bool:
            raise TypeError("bridges: expected boolean")
        if isinstance(self.epochs, FixedEpochSettings) and self.annotations.breaks is not None:
            raise ValueError("annotations.breaks: requires event-related epochs")
        if (
            isinstance(self.sampling, ResamplingSettings)
            and self.sampling.padding != self.epochs.padding
        ):
            raise ValueError("sampling.padding: must match epochs.padding")
        if isinstance(self.sampling, DecimationSettings) and self.filter.h_freq is None:
            raise ValueError("sampling.factor: decimation requires explicit filter.h_freq")


@dataclass(frozen=True)
class BIDSInputSettings:
    """How a recording is read through MNE-BIDS.

    Parameters
    ----------
    root : pathlib.Path
        BIDS dataset root.
    canonical_channels : tuple of str, optional
        Channel order to impose, a permutation of exactly the BIDS channel names;
        None keeps the order of ``channels.tsv``.
    """

    root: Path
    canonical_channels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("input.bids.root: expected Path")
        if self.canonical_channels is not None:
            names(self.canonical_channels, "input.canonical_channels")
            if not self.canonical_channels:
                raise ValueError("input.canonical_channels: must be nonempty")


@dataclass(frozen=True)
class InputSettings:
    """The recording a configuration reads.

    Parameters
    ----------
    path : pathlib.Path
        Recording file ending in ``.fif``, ``.fif.gz``, ``.edf``, ``.bdf``, ``.vhdr``
        or ``.set``.
    bids : BIDSInputSettings, optional
        Read the file through MNE-BIDS with its sidecars instead of directly.
    """

    path: Path
    bids: BIDSInputSettings | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("input.path: expected Path")
        if not str(self.path).lower().endswith(SUFFIXES):
            raise ValueError("input.path: unsupported recording suffix")
        if self.bids is not None and not isinstance(self.bids, BIDSInputSettings):
            raise TypeError("input.bids: expected BIDSInputSettings")


@dataclass(frozen=True)
class OutputSettings:
    """Where a recording's export and checkpoints are written.

    Parameters
    ----------
    directory : pathlib.Path
        Export directory; checkpoints go in its ``.preprocessing/<name>``
        subdirectory.
    name : str
        File name stem of the export: a letter or digit, then letters, digits,
        ``_``, ``.`` or ``-``.
    """

    directory: Path
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.directory, Path):
            raise TypeError("output.directory: expected Path")
        if not isinstance(self.name, str) or not _SAFE_NAME.fullmatch(self.name):
            raise ValueError("output.name: expected safe nonempty filename stem")


@dataclass(frozen=True)
class WorkflowSettings:
    """Policies of the review gates in the checkpointed workflow.

    Parameters
    ----------
    raw_review : {"required", "suggested", "disabled"}, default "required"
        ``"required"`` stops at ``review-raw`` until a decision is saved,
        ``"suggested"`` saves the detectors' verdict as the decision and continues,
        and ``"disabled"`` skips the gate, leaving detector candidates unapplied.
    artifact_review : {"required", "suggested"}, default "required"
        The same for ``review-artifact``, when artifact correction is configured.
    epoch_review : {"required", "optional", "disabled"}, default "optional"
        Only ``"required"`` stops at ``review-epochs``; the other two skip it.
    """

    raw_review: str = "required"
    artifact_review: str = "required"
    epoch_review: str = "optional"

    def __post_init__(self) -> None:
        choice(self.raw_review, ("required", "suggested", "disabled"), "workflow.raw_review")
        choice(self.artifact_review, ("required", "suggested"), "workflow.artifact_review")
        choice(self.epoch_review, ("required", "optional", "disabled"), "workflow.epoch_review")


@dataclass(frozen=True)
class PreprocessingConfig:
    """One recording's preprocessing configuration, as :func:`load_recipe` returns them.

    Parameters
    ----------
    input : InputSettings
        The recording to read.
    output : OutputSettings
        Export location and name.
    processing : ProcessingSettings
        Numerical settings of every stage.
    workflow : WorkflowSettings
        Review-gate policies.
    """

    input: InputSettings
    output: OutputSettings
    processing: ProcessingSettings
    workflow: WorkflowSettings = field(default_factory=WorkflowSettings)

    def __post_init__(self) -> None:
        for name, expected in (
            ("input", InputSettings),
            ("output", OutputSettings),
            ("processing", ProcessingSettings),
            ("workflow", WorkflowSettings),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name}: expected {expected.__name__}")


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
        raise ModuleNotFoundError(
            "Install YAML support: pip install 'eegtable[preprocessing]'"
        ) from exc

    class UniqueLoader(yaml.SafeLoader):
        pass

    def construct(loader: Any, node: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str):
                raise ValueError("YAML: mapping keys must be strings")
            if key in result:
                raise ValueError(f"YAML: duplicate key {key!r}")
            result[key] = loader.construct_object(value_node)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct)
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    return value


def _section(value: object, path: str, keys: str) -> dict[str, Any]:
    return dict(mapping(value, path, set(keys.split())))


def _sequence(value: object, path: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{path}: expected YAML sequence")
    return tuple(value)


def _path(value: object, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: expected nonempty path string")
    return (base / value).resolve()


def _fill(value: object, fields: Mapping[str, str], name: str) -> object:
    # Per-recording placeholders; anything else in braces is a mistake, not a literal.
    if not isinstance(value, str):
        return value
    try:
        return value.format_map(fields)
    except (KeyError, IndexError, ValueError, AttributeError) as exc:
        raise ValueError(f"{name}: unknown placeholder in {value!r} ({exc})") from exc


def _parse_channels(value: object, base: Path) -> ChannelSettings:
    data = _section(
        value, "channels", "rename types drop bads montage interpolate_bads bipolar projections"
    )
    for key in ("drop", "bads"):
        if key in data:
            data[key] = _sequence(data[key], f"channels.{key}")
    if isinstance(data.get("montage"), dict):
        montage = _section(data["montage"], "channels.montage", "path")
        data["montage"] = _path(montage.get("path"), base, "channels.montage.path")
    if "bipolar" in data:
        data["bipolar"] = tuple(
            BipolarSettings(**_section(item, "channels.bipolar", "name anode cathode type"))
            for item in _sequence(data["bipolar"], "channels.bipolar")
        )
    return ChannelSettings(**data)


def _parse_annotations(value: object) -> AnnotationSettings:
    data = _section(value, "annotations", "bad_spans amplitude breaks muscle")
    if "bad_spans" in data:
        data["bad_spans"] = tuple(
            BadSpan(**_section(item, "annotations.bad_spans", "onset duration description"))
            for item in _sequence(data["bad_spans"], "annotations.bad_spans")
        )
    if data.get("amplitude") is not None:
        data["amplitude"] = AmplitudeSettings(
            **_section(
                data["amplitude"], "annotations.amplitude", "peak flat bad_percent min_duration"
            )
        )
    if data.get("breaks") is not None:
        data["breaks"] = BreakSettings(
            **_section(
                data["breaks"],
                "annotations.breaks",
                "min_break_duration t_start_after_previous t_stop_before_next",
            )
        )
    if data.get("muscle") is not None:
        muscle = _section(
            data["muscle"], "annotations.muscle", "filter_freq threshold min_length_good"
        )
        muscle["filter_freq"] = _sequence(
            muscle.get("filter_freq"), "annotations.muscle.filter_freq"
        )
        data["muscle"] = MuscleSettings(**muscle)
    return AnnotationSettings(**data)


def _parse_epochs(
    value: object, base: Path, fields: Mapping[str, str]
) -> EventEpochSettings | FixedEpochSettings:
    data = _section(
        value,
        "epochs",
        "kind events tmin tmax duration overlap start stop baseline padding detrend metadata",
    )
    kind = data.pop("kind", None)
    choice(kind, ("events", "fixed"), "epochs.kind")
    if data.get("baseline") is not None:
        data["baseline"] = _sequence(data["baseline"], "epochs.baseline")
    if kind == "fixed":
        return FixedEpochSettings(
            **_section(data, "epochs", "duration overlap start stop baseline padding detrend")
        )
    data = _section(data, "epochs", "events tmin tmax baseline padding detrend metadata")
    events = _section(
        data.get("events"),
        "epochs.events",
        "source event_id stim_channel path shortest_event min_duration delay",
    )
    if "path" in events:
        events["path"] = _path(
            _fill(events["path"], fields, "epochs.events.path"), base, "epochs.events.path"
        )
    data["events"] = EventSettings(**events)
    if data.get("metadata") is not None:
        data["metadata"] = _path(
            _fill(data["metadata"], fields, "epochs.metadata"), base, "epochs.metadata"
        )
    return EventEpochSettings(**data)


def _parse_artifact(value: object) -> ArtifactSettings:
    data = _section(value, "artifact", "method reference ica ssp regression")
    method = data.get("method")
    if not isinstance(method, str):
        raise ValueError("artifact.method: expected string")
    choice(method, ("ica", "ssp", "regression"), "artifact.method")
    data = _section(data, "artifact", f"method reference {method}")
    if isinstance(data.get("reference"), list):
        data["reference"] = tuple(data["reference"])
    allowed = {
        "ica": "l_freq n_components random_state max_iter reject flat tstep "
        "eog_channels ecg_channel method iclabel",
        "ssp": "n_eeg eog_channels ecg_channel l_freq h_freq tmin tmax reject",
        "regression": "eog_channels tstep reject flat",
    }
    settings = _section(data.get(method), f"artifact.{method}", allowed[method])
    if "eog_channels" in settings:
        settings["eog_channels"] = _sequence(
            settings["eog_channels"], f"artifact.{method}.eog_channels"
        )
    if settings.get("iclabel") is not None:
        iclabel = _section(settings["iclabel"], "artifact.ica.iclabel", "threshold keep")
        if "keep" in iclabel:
            iclabel["keep"] = _sequence(iclabel["keep"], "artifact.ica.iclabel.keep")
        settings["iclabel"] = ICLabelSettings(**iclabel)
    constructors = {"ica": ICASettings, "ssp": SSPSettings, "regression": RegressionSettings}
    return ArtifactSettings(method, constructors[method](**settings), data.get("reference"))


def load_recipe(path: str | Path) -> dict[str, PreprocessingConfig]:
    """One configuration per recording the recipe selects, keyed by label.

    Parameters
    ----------
    path : path-like
        YAML recipe. Relative paths in it resolve against its directory; unknown
        and duplicate keys raise, as do two recordings that would write the same
        bundle.

    Returns
    -------
    dict of str to PreprocessingConfig
        Keyed by export name when the names are unique, otherwise by bundle path
        relative to ``output.directory``.
    """
    path = Path(path).resolve()
    base = path.parent
    data = _section(
        read_yaml(path),
        "config",
        "input output workflow channels annotations filter epochs reference crop artifact "
        "rejection sampling bad_channels bridges stimulation",
    )
    source = _section(
        data.pop("input", None),
        "input",
        "kind path root pattern subjects sessions tasks acquisitions runs canonical_channels",
    )
    kind = source.pop("kind", "files")
    choice(kind, ("files", "bids"), "input.kind")
    output = _section(data.pop("output", None), "output", "directory name")
    workflow = WorkflowSettings(
        **_section(data.pop("workflow", {}), "workflow", "raw_review artifact_review epoch_review")
    )
    directory = _path(output.get("directory"), base, "output.directory")
    if "name" in output and "root" in source:
        raise ValueError("output.name: not allowed with input.root; names come from the files")
    configs: dict[str, PreprocessingConfig] = {}
    sources = (
        _bids_sources(source, base)
        if kind == "bids"
        else [(InputSettings(file), relative) for file, relative in _sources(source, base)]
    )
    for input_settings, relative in sources:
        file = input_settings.path
        name = output["name"] if output.get("name") is not None else _derive_name(file)
        bundle = (relative / name).as_posix()
        if bundle in configs:
            other = configs[bundle].input.path.name
            raise ValueError(f"output: {other} and {file.name} write the same bundle {bundle}")
        configs[bundle] = PreprocessingConfig(
            input_settings,
            OutputSettings(directory / relative, name),
            _processing(dict(data), base, {"name": name, "parent": str(file.parent)}),
            workflow,
        )
    by_name = len({config.output.name for config in configs.values()}) == len(configs)
    return {config.output.name if by_name else key: config for key, config in configs.items()}


def load_config(path: str | Path) -> PreprocessingConfig:
    """The configuration of a recipe that selects exactly one recording.

    Parameters
    ----------
    path : path-like
        YAML recipe, read as by :func:`load_recipe`.

    Returns
    -------
    PreprocessingConfig
        Raises unless the recipe selects exactly one recording.
    """
    recordings = load_recipe(path)
    if len(recordings) != 1:
        raise ValueError(f"{path}: selects {len(recordings)} recordings; use load_recipe")
    return next(iter(recordings.values()))


def _sources(source: Mapping[str, Any], base: Path) -> list[tuple[Path, Path]]:
    source = _section(dict(source), "input", "path root pattern")
    # Each recording with its directory relative to the root, which the output tree mirrors.
    if ("path" in source) == ("root" in source):
        raise ValueError("input: expected exactly one of path and root")
    if "path" in source:
        if "pattern" in source:
            raise ValueError("input.pattern: only allowed with input.root")
        return [(_path(source["path"], base, "input.path"), Path())]
    root = _path(source["root"], base, "input.root")
    if not root.is_dir():
        raise ValueError(f"input.root: {root} is not a directory")
    pattern = source.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("input.pattern: required with input.root")
    # anchor, not is_absolute(): Windows calls "/tmp/*.fif" relative, and glob then refuses it.
    if Path(pattern).anchor or ".." in Path(pattern).parts:
        raise ValueError("input.pattern: must stay below input.root")
    # Hidden components are skipped: macOS leaves ._ AppleDouble files beside every file
    # on an external drive, and pathlib's glob matches them where a shell would not.
    files = sorted(
        file
        for file in root.glob(pattern)
        if file.is_file() and not any(part.startswith(".") for part in file.relative_to(root).parts)
    )
    if not files:
        raise ValueError(f"input.pattern: no files match {pattern!r} under {root}")
    for file in files:
        if not file.name.lower().endswith(SUFFIXES):
            raise ValueError(f"input.pattern: {file.name} is not a recording; narrow the pattern")
    return [(file, file.parent.relative_to(root)) for file in files]


def _bids_sources(source: Mapping[str, Any], base: Path) -> list[tuple[InputSettings, Path]]:
    from eegtable.bids import BIDSQuery, discover_bids

    source = _section(
        dict(source), "input", "root subjects sessions tasks acquisitions runs canonical_channels"
    )
    root = _path(source.pop("root", None), base, "input.root")
    canonical = source.pop("canonical_channels", None)
    canonical_channels = (
        None if canonical is None else _sequence(canonical, "input.canonical_channels")
    )
    bids_settings = BIDSInputSettings(root, canonical_channels)
    entities = {name: _sequence(value, f"input.{name}") for name, value in source.items()}
    query = BIDSQuery(root=root, **entities)
    return [
        (InputSettings(Path(path.fpath), bids_settings), path.fpath.parent.relative_to(root))
        for path in discover_bids(query)
    ]


def _derive_name(file: Path) -> str:
    name = file.name
    for suffix in SUFFIXES:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    for marker in ("_raw", "_eeg"):
        if name.endswith(marker) and len(name) > len(marker):
            name = name[: -len(marker)]
            break
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError(
            f"output.name: {file.name} gives {name!r}; set output.name or rename the file"
        )
    return name


def _processing(data: dict[str, Any], base: Path, fields: Mapping[str, str]) -> ProcessingSettings:
    data["epochs"] = _parse_epochs(data.get("epochs"), base, fields)
    data["channels"] = _parse_channels(data.get("channels", {}), base)
    data["annotations"] = _parse_annotations(data.get("annotations", {}))
    filters = _section(data.get("filter", {}), "filter", "l_freq h_freq notch_freqs")
    if "notch_freqs" in filters:
        filters["notch_freqs"] = _sequence(filters["notch_freqs"], "filter.notch_freqs")
    data["filter"] = FilterSettings(**filters)
    reference = _section(data.get("reference", {}), "reference", "channels add_channels")
    if isinstance(reference.get("channels"), list):
        reference["channels"] = tuple(reference["channels"])
    if "add_channels" in reference:
        reference["add_channels"] = _sequence(reference["add_channels"], "reference.add_channels")
    data["reference"] = ReferenceSettings(**reference)
    if data.get("crop") is not None:
        data["crop"] = CropSettings(**_section(data["crop"], "crop", "tmin tmax"))
    if data.get("artifact") is not None:
        data["artifact"] = _parse_artifact(data["artifact"])
    if data.get("rejection") is not None:
        rejection = _section(
            data["rejection"],
            "rejection",
            "method reject flat tmin tmax n_interpolate consensus cv random_state",
        )
        method = rejection.pop("method", None)
        choice(method, ("thresholds", "autoreject"), "rejection.method")
        if method == "thresholds":
            data["rejection"] = ThresholdSettings(
                **_section(rejection, "rejection", "reject flat tmin tmax")
            )
        else:
            rejection = _section(rejection, "rejection", "n_interpolate consensus cv random_state")
            for key in ("n_interpolate", "consensus"):
                rejection[key] = _sequence(rejection.get(key), f"rejection.{key}")
            data["rejection"] = AutoRejectSettings(**rejection)
    if data.get("sampling") is not None:
        sampling = _section(data["sampling"], "sampling", "method factor sfreq padding")
        method = sampling.pop("method", None)
        choice(method, ("decimate", "resample"), "sampling.method")
        data["sampling"] = (
            DecimationSettings(**_section(sampling, "sampling", "factor"))
            if method == "decimate"
            else ResamplingSettings(**_section(sampling, "sampling", "sfreq padding"))
        )
    if data.get("bad_channels") is not None:
        bads = _section(
            data["bad_channels"],
            "bad_channels",
            "method methods random_state ransac repeats notch_freqs",
        )
        if "notch_freqs" in bads:
            bads["notch_freqs"] = _sequence(bads["notch_freqs"], "bad_channels.notch_freqs")
        if "methods" in bads:
            bads["methods"] = _sequence(bads["methods"], "bad_channels.methods")
        data["bad_channels"] = BadChannelSettings(**bads)
    bridges = data.get("bridges")
    if bridges is not None and type(bridges) is not bool:
        raise TypeError("bridges: expected true or false")
    data["bridges"] = bool(bridges)
    if data.get("stimulation") is not None:
        stimulation = _section(
            data["stimulation"], "stimulation", "event_ids channels tmin tmax mode baseline"
        )
        for key in ("event_ids", "channels", "baseline"):
            if stimulation.get(key) is not None:
                stimulation[key] = _sequence(stimulation[key], f"stimulation.{key}")
        data["stimulation"] = StimulationSettings(**stimulation)
    return ProcessingSettings(**data)
