"""Raw BrainVision (or artifact-cleaned FIF) thermal-pain EEG to BIDS, markers intact.

Self-contained copy of EEG_fMRI_Pipeline's studies/pain_study/scripts/conversion/
eeg_raw_to_bids.py, kept here because the first conversion of this dataset lost every
marker name: mne-bids rewrites BrainVision through pybv, which writes each event as
"Stimulus/S <code>", and the default filter dropped the recording's own BAD intervals.

    python paradigm_specific/thermal_pain/eeg_raw_to_bids.py --source-root SOURCE \\
        --bids-root BIDS --task thermalactive --source-layout fastr_corrected_1khz_hp05 \\
        --source-glob "ThermalPain*_run*_fastr.vhdr" --trim-to-volume-bounds \\
        --canonicalize-thermode-markers

--canonicalize-thermode-markers is not optional for this dataset: some subjects (sub-0016)
recorded the thermode as Stim_on/S  1 only.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import timedelta
from pathlib import Path
from typing import Literal

import mne
import numpy as np

logger = logging.getLogger(__name__)

SourceFormat = Literal["brainvision", "native-fif"]

CANONICAL_THERMODE_MARKER = "Trig_therm/T  1"
LEGACY_STIMULATION_MARKER = "Stim_on/S  1"
THERMODE_EVENT_COUNT = 11
# A gap this many times the median volume interval starts a new acquisition block.
VOLUME_BLOCK_GAP_FACTOR = 1.5
# A "Bad Interval" marker reads back as "Bad Interval/...", which MNE treats as a BAD span
# when epoching; dropping it would silently keep data the recording itself flagged.
ALWAYS_KEPT_MARKERS = ("Pulse Artifact", "Bad Interval")
NON_EEG_CHANNEL_TYPES = {"HEOG": "eog", "VEOG": "eog", "ECG": "ecg"}


def normalize_string(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def parse_subject_id(path: Path) -> str:
    match = re.search(r"sub-([A-Za-z0-9]+)", str(path))
    if not match:
        raise ValueError(f"Could not parse subject from path: {path}")
    return match.group(1)


def run_index(path: Path) -> int:
    match = re.search(r"run[-_]?(\d+)", path.stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Run number is required in EEG filename: {path}")
    return int(match.group(1))


def find_source_files(
    source_root: Path, source_format: SourceFormat, task: str, layout: str, glob: str = "*.vhdr"
) -> list[Path]:
    if source_format == "brainvision":
        if "/" in layout or "\\" in layout:
            raise ValueError(
                f"Source layout must be one directory name under sub-*/eeg/, got {layout!r}"
            )
        # A layout directory can hold other paradigms' recordings too; the glob names ours.
        pattern = f"sub-*/eeg/{layout}/**/{glob}"
    else:
        pattern = f"sub-*/eeg/sub-*_task-{task}_run-*_desc-mriartifactclean_raw.fif"
    files = sorted(
        p for p in source_root.glob(pattern) if p.is_file() and not p.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"No {source_format} EEG files found under {source_root}")
    return files


def read_raw(source_file: Path, source_format: SourceFormat) -> mne.io.BaseRaw:
    if source_format == "brainvision":
        return mne.io.read_raw_brainvision(source_file, preload=False, verbose=False)
    return mne.io.read_raw_fif(source_file, preload=True, verbose=False)


def set_channel_types(raw: mne.io.BaseRaw) -> None:
    present = {name: kind for name, kind in NON_EEG_CHANNEL_TYPES.items() if name in raw.ch_names}
    if present:
        raw.set_channel_types(present, on_unit_change="ignore")


def set_montage(raw: mne.io.BaseRaw, montage: str) -> None:
    # The recordings spell FPz against the 10-05 montage.
    if "FPz" in raw.ch_names and "Fpz" not in raw.ch_names:
        raw.rename_channels({"FPz": "Fpz"})
    raw.set_montage(mne.channels.make_standard_montage(montage), on_missing="raise")


def data_relative_onsets(raw: mne.io.BaseRaw) -> np.ndarray:
    return np.asarray(raw.annotations.onset, dtype=float) - raw.first_time


# raw.annotations.onset is on the raw's absolute timeline, but set_annotations() reads an
# orig_time=None object as relative to the first sample. Undo that so rebuilding annotations
# from onsets read off a cropped raw is an identity rather than a shift by the crop offset.
def set_annotations_at_absolute_onsets(raw: mne.io.BaseRaw, annotations: mne.Annotations) -> None:
    if annotations.orig_time is None and raw.first_time:
        annotations = mne.Annotations(
            onset=np.asarray(annotations.onset, dtype=float) - raw.first_time,
            duration=annotations.duration,
            description=annotations.description,
            orig_time=None,
        )
    raw.set_annotations(annotations)


def filter_annotations(
    raw: mne.io.BaseRaw, prefixes: list[str] | None, keep_all: bool, zero_base: bool
) -> None:
    if len(raw.annotations) == 0:
        return
    if keep_all:
        keep = list(range(len(raw.annotations)))
    else:
        wanted = (
            ["Trig_", "Volume"]
            if prefixes is None
            else [normalize_string(p) for p in prefixes if str(p).strip()]
        )
        wanted.extend(ALWAYS_KEPT_MARKERS)
        keep = [
            i
            for i, description in enumerate(raw.annotations.description)
            if description.lower().startswith(("bad", "edge"))
            or any(normalize_string(description).startswith(prefix) for prefix in wanted)
        ]
        if not keep:
            logger.warning("No annotations matched %s; the run will have no events.", wanted)
    retained = raw.annotations[keep]
    if zero_base and len(retained):
        # Move the data origin with the markers so events keep their acquired samples.
        raw.crop(tmin=max(0.0, float(retained.onset.min() - raw.first_time)))
    set_annotations_at_absolute_onsets(raw, retained)


def first_contiguous_volume_onsets(onsets: list[float]) -> list[float]:
    ordered = sorted(onsets)
    if len(ordered) < 2:
        return ordered
    intervals = np.diff(ordered)
    block = [ordered[0]]
    for current, interval in zip(ordered[1:], intervals, strict=True):
        if interval > VOLUME_BLOCK_GAP_FACTOR * float(np.median(intervals)):
            break
        block.append(current)
    return block


def volume_onsets(raw: mne.io.BaseRaw) -> list[float]:
    pattern = re.compile(r"(^|[/,])V\s*1(\D|$)")
    return [
        float(onset)
        for onset, description in zip(
            raw.annotations.onset, raw.annotations.description, strict=True
        )
        if normalize_string(description).startswith("Volume/V")
        or pattern.search(normalize_string(description))
    ]


# Crop to the first contiguous volume block, ending one TR after its last marker (clipped to
# the recording), so unsaved dummy volumes and a later scanner restart are left out.
def trim_to_volume_bounds(raw: mne.io.BaseRaw) -> bool:
    block = first_contiguous_volume_onsets(volume_onsets(raw))
    if len(block) < 2:
        return False
    start = block[0] - raw.first_time
    repetition_time = float(np.median(np.diff(block)))
    end = min(block[-1] + repetition_time - raw.first_time, float(raw.times[-1]))
    logger.info("Trimming to volume bounds: %.3f s to %.3f s.", start, end)
    raw.crop(tmin=start, tmax=end)
    return True


def trim_to_first_event(raw: mne.io.BaseRaw, prefix: str) -> float:
    onsets = [
        float(onset)
        for onset, description in zip(
            raw.annotations.onset, raw.annotations.description, strict=True
        )
        if normalize_string(description).startswith(normalize_string(prefix))
    ]
    if not onsets:
        raise ValueError(f"Cannot trim: no annotation starts with {prefix!r} in this recording")
    # Annotation onsets are absolute against orig_time; crop takes a time in raw.times.
    raw.crop(tmin=min(onsets) - raw.first_time)
    return min(onsets)


def canonicalize_thermode_markers(raw: mne.io.BaseRaw) -> bool:
    descriptions = list(raw.annotations.description)
    canonical = descriptions.count(CANONICAL_THERMODE_MARKER)
    if canonical == THERMODE_EVENT_COUNT:
        return False
    if canonical:
        raise ValueError(
            f"Expected {THERMODE_EVENT_COUNT} {CANONICAL_THERMODE_MARKER} markers, "
            f"found {canonical}."
        )
    legacy = descriptions.count(LEGACY_STIMULATION_MARKER)
    if legacy != THERMODE_EVENT_COUNT:
        raise ValueError(
            f"Thermode canonicalization expected {THERMODE_EVENT_COUNT} "
            f"{LEGACY_STIMULATION_MARKER} "
            f"markers, found {legacy}."
        )
    raw.annotations.rename({LEGACY_STIMULATION_MARKER: CANONICAL_THERMODE_MARKER})
    return True


# A volume marker can fall one sample past the last recorded sample when the scanner outlived
# the EEG; that one is dropped. Anything else outside the data is an error.
def discard_unrecorded_terminal_volumes(raw: mne.io.BaseRaw) -> None:
    if len(raw.annotations) == 0:
        return
    samples = raw.time_as_index(data_relative_onsets(raw), use_rounding=True)
    valid = (samples >= 0) & (samples < raw.n_times)
    if valid.all():
        return
    invalid = np.flatnonzero(~valid)
    end = float(raw.first_time + raw.times[-1])
    onsets = raw.annotations.onset[invalid]
    removable = (
        np.array_equal(
            invalid, np.arange(len(raw.annotations) - invalid.size, len(raw.annotations))
        )
        and np.all(raw.annotations.description[invalid] == "Volume/V  1")
        and np.all(raw.annotations.duration[invalid] == 0.0)
        and np.all((onsets > end) & (onsets <= end + 1.0 / float(raw.info["sfreq"])))
    )
    if not removable:
        details = ", ".join(
            f"{d!r} at {o:.9f}s"
            for d, o in zip(raw.annotations.description[invalid], onsets, strict=True)
        )
        raise ValueError(f"Annotations fall outside recorded EEG data: {details}")
    set_annotations_at_absolute_onsets(raw, raw.annotations[valid])


def brainvision_marker(description: str) -> tuple[str, str]:
    # MNE reads a BrainVision marker as "Type/Description"; a description without a type is
    # filed under Comment, the BrainVision type for free text.
    kind, slash, text = str(description).partition("/")
    if not slash:
        if kind.lower().startswith("bad"):
            return "Bad Interval", kind
        if kind.lower().startswith("edge"):
            return "edge", kind
        return "Comment", kind
    return kind, text


def restore_marker_names(vhdr: Path, raw: mne.io.BaseRaw) -> None:
    # mne-bids hands pybv a bare event array, so every marker comes out as "Stimulus/S <code>"
    # and the paradigm's types and names are gone. Rebuild the marker file from the
    # annotations that produced it; only pybv's "New Segment" (recording date) is kept.
    vmrk = vhdr.with_suffix(".vmrk")
    lines = vmrk.read_text(encoding="utf-8").splitlines()
    markers = [line.split("=", 1)[1] for line in lines if line.startswith("Mk")]
    segments = [marker for marker in markers if marker.startswith("New Segment")]
    if len(markers) - len(segments) != len(raw.annotations):
        raise ValueError(
            f"{vmrk.name}: {len(markers) - len(segments)} markers written for "
            f"{len(raw.annotations)} annotations"
        )
    sfreq = float(raw.info["sfreq"])
    for onset, duration, description in zip(
        data_relative_onsets(raw),
        raw.annotations.duration,
        raw.annotations.description,
        strict=True,
    ):
        kind, name = (field.replace(",", r"\1") for field in brainvision_marker(description))
        position = int(round(float(onset) * sfreq)) + 1
        size = max(1, int(round(float(duration) * sfreq)))
        segments.append(f"{kind},{name},{position},{size},0")
    kept = [line for line in lines if not line.startswith("Mk")]
    numbered = [f"Mk{number}={marker}" for number, marker in enumerate(segments, start=1)]
    vmrk.write_text("\n".join([*kept, *numbered]) + "\n", encoding="utf-8")


def ensure_dataset_description(bids_root: Path, name: str) -> None:
    from mne_bids import make_dataset_description

    bids_root.mkdir(parents=True, exist_ok=True)
    if not (bids_root / "dataset_description.json").exists():
        make_dataset_description(path=bids_root, name=name, dataset_type="raw", overwrite=False)


def ensure_task_events_json(bids_root: Path, task: str) -> None:
    out = bids_root / f"task-{task}_events.json"
    if out.exists():
        return
    schema = {
        "onset": {
            "Description": "Event onset in seconds from the start of the EEG run.",
            "Units": "s",
        },
        "duration": {"Description": "Event duration in seconds.", "Units": "s"},
        "trial_type": {"Description": "Event label (BrainVision/MNE annotation description)."},
        "value": {"Description": "Event code (trigger ID)."},
        "sample": {"Description": "Event onset sample index (first sample is 0)."},
    }
    out.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")


def run_raw_to_bids(
    source_root: Path,
    bids_root: Path,
    task: str,
    *,
    subjects: list[str] | None = None,
    montage: str = "easycap-M1",
    line_freq: float = 60.0,
    overwrite: bool = False,
    zero_base_onsets: bool = False,
    trim_volume_bounds: bool = False,
    trim_to_first_event_prefix: str | None = None,
    event_prefixes: list[str] | None = None,
    keep_all_annotations: bool = False,
    canonicalize_thermode: bool = False,
    source_format: SourceFormat = "brainvision",
    source_layout: str = "analyzer_brainvision_processed_1khz",
    source_glob: str = "*.vhdr",
) -> int:
    from mne_bids import BIDSPath, write_raw_bids

    files = find_source_files(source_root, source_format, task, source_layout, source_glob)
    if subjects:
        files = [path for path in files if parse_subject_id(path) in set(subjects)]
        if not files:
            raise FileNotFoundError(
                f"No matching {source_format} files for subjects: {sorted(subjects)}"
            )
    ensure_dataset_description(bids_root, f"{task} EEG")
    ensure_task_events_json(bids_root, task)
    for index, source_file in enumerate(files, 1):
        raw = read_raw(source_file, source_format)
        set_channel_types(raw)
        if canonicalize_thermode:
            canonicalize_thermode_markers(raw)
        if montage:
            set_montage(raw, montage)
        raw.info["line_freq"] = line_freq
        trimmed = trim_volume_bounds and trim_to_volume_bounds(raw)
        if trim_to_first_event_prefix:
            trim_to_first_event(raw, trim_to_first_event_prefix)
            trimmed = True
        if trimmed and not raw.preload:
            raw.load_data()
        filter_annotations(raw, event_prefixes, keep_all_annotations, zero_base_onsets)
        if zero_base_onsets and not raw.preload:
            raw.load_data()
        discard_unrecorded_terminal_volumes(raw)
        bids_path = BIDSPath(
            subject=parse_subject_id(source_file),
            task=task,
            run=run_index(source_file),
            datatype="eeg",
            suffix="eeg",
            root=bids_root,
        )
        # BrainVision starts at sample zero; scans.tsv must date that exported sample.
        if raw.info["meas_date"] is not None:
            raw.set_meas_date(raw.info["meas_date"] + timedelta(seconds=raw.first_time))
        write_raw_bids(
            raw=raw,
            bids_path=bids_path,
            overwrite=overwrite,
            allow_preload=raw.preload,
            format="BrainVision",
            verbose=False,
        )
        restore_marker_names(bids_path.copy().update(extension=".vhdr").fpath, raw)
        logger.info("[%d/%d] wrote %s", index, len(files), bids_path.basename)
    return len(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--source-format", choices=("brainvision", "native-fif"), default="brainvision"
    )
    parser.add_argument(
        "--source-layout",
        default="analyzer_brainvision_processed_1khz",
        help="directory under sub-*/eeg/ holding the .vhdr files to convert",
    )
    parser.add_argument(
        "--source-glob",
        default="*.vhdr",
        help="file pattern inside the layout, e.g. 'ThermalPain*_run*.vhdr' "
        "when it also holds other tasks",
    )
    parser.add_argument("--subject", action="append", default=None)
    parser.add_argument("--montage", default="easycap-M1")
    parser.add_argument("--line-freq", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trim-to-volume-bounds", action="store_true")
    parser.add_argument("--trim-to-first-event", default=None, metavar="PREFIX")
    parser.add_argument("--zero-base-onsets", action="store_true")
    parser.add_argument(
        "--event-prefix",
        action="append",
        default=None,
        help="keep markers starting with this (default: Trig_ and Volume); "
        "BAD intervals always stay",
    )
    parser.add_argument("--keep-all-annotations", action="store_true")
    parser.add_argument(
        "--canonicalize-thermode-markers",
        action="store_true",
        help="rewrite exactly 11 Stim_on markers to Trig_therm when the latter are absent",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    count = run_raw_to_bids(
        args.source_root,
        args.bids_root,
        args.task,
        subjects=args.subject,
        montage=args.montage,
        line_freq=args.line_freq,
        overwrite=args.overwrite,
        zero_base_onsets=args.zero_base_onsets,
        trim_volume_bounds=args.trim_to_volume_bounds,
        trim_to_first_event_prefix=args.trim_to_first_event,
        event_prefixes=args.event_prefix,
        keep_all_annotations=args.keep_all_annotations,
        canonicalize_thermode=args.canonicalize_thermode_markers,
        source_format=args.source_format,
        source_layout=args.source_layout,
        source_glob=args.source_glob,
    )
    print(f"Converted {count} recording(s) into {args.bids_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
