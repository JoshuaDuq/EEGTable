"""Explicit BIDS discovery and ingestion through MNE-BIDS."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from eegtable._validation import validate_names
from eegtable.provenance import file_hash

if TYPE_CHECKING:
    from eegtable.preprocessing.config import ProcessingSettings
    from eegtable.preprocessing.pipeline import PreprocessingResult

__all__ = ["BIDSQuery", "BIDSRecording", "discover_bids", "preprocess_bids", "read_bids"]


def _require_mne_bids() -> Any:
    try:
        import mne_bids
    except ImportError as exc:
        raise ImportError("BIDS ingestion requires: pip install eegtable[bids]") from exc
    return mne_bids


@dataclass(frozen=True)
class BIDSQuery:
    """EEG recording selection expressed as BIDS entities, without filename parsing."""

    root: Path
    subjects: tuple[str, ...] | None = None
    sessions: tuple[str, ...] | None = None
    tasks: tuple[str, ...] | None = None
    acquisitions: tuple[str, ...] | None = None
    runs: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("BIDS root must be a Path.")
        for name in ("subjects", "sessions", "tasks", "acquisitions", "runs"):
            values = getattr(self, name)
            if values is not None:
                if not isinstance(values, tuple) or not values:
                    raise ValueError(f"BIDS {name} must be a nonempty tuple of entity labels.")
                validate_names(values, f"BIDS {name}")
                if any(not value.isalnum() for value in values):
                    raise ValueError(f"BIDS {name} must contain unprefixed alphanumeric labels.")


def _validate_root(root: Path) -> None:
    if not root.is_dir():
        raise ValueError(f"BIDS root is not a directory: {root}.")
    if not (root / "dataset_description.json").is_file():
        raise FileNotFoundError(f"BIDS dataset_description.json is missing under {root}.")


def discover_bids(query: BIDSQuery) -> tuple[Any, ...]:
    """Discover raw EEG data via MNE-BIDS, excluding sidecars and derivatives."""
    _validate_root(query.root)
    mne_bids = _require_mne_bids()
    paths = mne_bids.find_matching_paths(
        query.root,
        subjects=query.subjects,
        sessions=query.sessions,
        tasks=query.tasks,
        acquisitions=query.acquisitions,
        runs=query.runs,
        datatypes="eeg",
        suffixes="eeg",
        extensions=(".vhdr", ".edf", ".bdf", ".set"),
        check=True,
    )
    paths = [
        path for path in paths if "derivatives" not in path.fpath.relative_to(query.root).parts
    ]
    if not paths:
        raise ValueError("BIDS query selected no EEG recordings.")
    return tuple(sorted(paths, key=lambda path: str(path.fpath)))


def _bids_path(path: Any, root: Path | None = None) -> Any:
    mne_bids = _require_mne_bids()
    if isinstance(path, mne_bids.BIDSPath):
        resolved = path.copy()
        if root is not None:
            resolved.update(root=root)
    elif isinstance(path, (str, Path)):
        if root is None:
            raise ValueError("An explicit BIDS root is required when reading a filesystem path.")
        resolved = mne_bids.get_bids_path_from_fname(path, check=True).update(root=root)
    else:
        raise TypeError("Expected a BIDSPath or a recording path with an explicit BIDS root.")
    if resolved.root is None or resolved.subject is None or resolved.task is None:
        raise ValueError("A BIDS recording needs explicit root, subject, and task entities.")
    if resolved.datatype != "eeg" or resolved.suffix != "eeg":
        raise ValueError("BIDS ingestion supports raw EEG recordings only.")
    _validate_root(Path(resolved.root))
    if not resolved.fpath.is_file():
        raise FileNotFoundError(resolved.fpath)
    if not resolved.fpath.resolve().is_relative_to(Path(resolved.root).resolve()):
        raise ValueError("BIDS recording must reside within its root.")
    return resolved


def _read_tsv(path: Path, numeric: tuple[str, ...] = ()) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=["n/a"])
    return _parse_numeric(frame, numeric)


def _parse_numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(
        list[dict[str, Any]],
        frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records"),
    )


def read_bids_metadata(
    path: Any, *, root: Path | None = None, canonical_channels: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """Read sidecar content and hashes for ingestion and checkpoint identities."""
    bids_path = _bids_path(path, root)
    root = Path(bids_path.root)
    channels_path = Path(bids_path.find_matching_sidecar(suffix="channels", extension=".tsv"))
    recording_json = Path(bids_path.find_matching_sidecar(suffix="eeg", extension=".json"))
    channels = _read_tsv(channels_path)
    if not {"name", "type", "units"} <= set(channels):
        raise ValueError("BIDS channels.tsv requires name, type, and units columns.")
    validate_names(tuple(channels["name"]), "BIDS channel names")
    if canonical_channels is not None:
        validate_names(canonical_channels, "canonical_channels")
        if set(canonical_channels) != set(channels["name"]):
            raise ValueError("canonical_channels must contain exactly the BIDS channel names.")
        channels = channels.set_index("name").loc[list(canonical_channels)].reset_index()
    sidecars = [channels_path, recording_json, root / "dataset_description.json"]
    events_path = bids_path.find_matching_sidecar(
        suffix="events", extension=".tsv", on_error="ignore"
    )
    events = None
    if events_path is not None:
        events_path = Path(events_path)
        events = _read_tsv(events_path, ("onset", "duration", "sample", "response_time"))
        if not {"onset", "duration"} <= set(events):
            raise ValueError("BIDS events.tsv requires onset and duration columns.")
        onset = events["onset"].to_numpy(dtype=float)
        duration = events["duration"].to_numpy(dtype=float)
        # BIDS allows n/a for an unavailable duration, which stays missing.
        if not np.isfinite(onset).all() or np.isinf(duration).any() or np.any(duration < 0):
            raise ValueError(
                "BIDS events need finite onsets and nonnegative finite or n/a durations."
            )
        sidecars.append(events_path)
    participant = None
    participants_path = root / "participants.tsv"
    if participants_path.exists():
        participants = _read_tsv(participants_path)
        if "participant_id" not in participants:
            raise ValueError("BIDS participants.tsv requires participant_id.")
        validate_names(tuple(participants["participant_id"]), "BIDS participant_id")
        selected = participants.loc[participants["participant_id"] == f"sub-{bids_path.subject}"]
        if len(selected) != 1:
            raise ValueError(
                "BIDS participants.tsv must identify the recording subject exactly once."
            )
        # Parsed for this subject only, so another row's unparsable age cannot block it.
        participant = _records(_parse_numeric(selected.copy(), ("age",)))[0]
        sidecars.append(participants_path)
    for suffix, extension in (
        ("events", ".json"),
        ("electrodes", ".tsv"),
        ("coordsystem", ".json"),
    ):
        sidecar = bids_path.find_matching_sidecar(
            suffix=suffix, extension=extension, on_error="ignore"
        )
        if sidecar is not None:
            sidecars.append(Path(sidecar))
    participants_json = root / "participants.json"
    if participants_json.is_file():
        sidecars.append(participants_json)
    scans_path = (
        _require_mne_bids()
        .BIDSPath(
            root=root,
            subject=bids_path.subject,
            session=bids_path.session,
            suffix="scans",
            extension=".tsv",
        )
        .fpath
    )
    if scans_path.is_file():
        sidecars.append(scans_path)
    return {
        "reader": "mne_bids.read_raw_bids",
        "mne_bids_version": version("mne-bids"),
        "root": str(root),
        "path": str(bids_path.fpath),
        "entities": {key: value for key, value in bids_path.entities.items() if value is not None},
        "participant": participant,
        "events": None if events is None else _records(events),
        "channels": _records(channels),
        "canonical_channels": list(channels["name"]),
        "sidecar_hashes": {str(sidecar): file_hash(sidecar) for sidecar in sidecars},
    }


@dataclass(frozen=True, eq=False)
class BIDSRecording:
    """Raw EEG and the BIDS metadata needed to construct labeled epochs."""

    raw: Any
    entities: Mapping[str, str]
    participant: Mapping[str, Any] | None
    events: pd.DataFrame | None
    channels: pd.DataFrame
    event_id: Mapping[str, int] | None
    provenance: Mapping[str, Any]


def read_bids(
    path: Any, *, root: Path | None = None, canonical_channels: tuple[str, ...] | None = None
) -> BIDSRecording:
    """Read EEG and sidecars through MNE-BIDS with strict channel matching.

    A canonical channel order is an explicit permutation of the complete channel
    set; missing or extra channels fail. No channel is dropped or renamed.
    """
    bids_path = _bids_path(path, root)
    provenance = read_bids_metadata(bids_path, canonical_channels=canonical_channels)
    has_events = provenance["events"] is not None
    loaded = _require_mne_bids().read_raw_bids(
        bids_path,
        extra_params={"preload": True},
        return_event_dict=has_events,
        on_ch_mismatch="raise",
        verbose=False,
    )
    if has_events:
        raw, event_id = loaded
    else:
        raw, event_id = loaded, None
    canonical = provenance["canonical_channels"]
    if set(raw.ch_names) != set(canonical):
        raise ValueError("Raw channels and BIDS canonical_channels disagree.")
    raw.reorder_channels(canonical)
    return BIDSRecording(
        raw=raw,
        entities=provenance["entities"],
        participant=provenance["participant"],
        events=None if provenance["events"] is None else pd.DataFrame(provenance["events"]),
        channels=pd.DataFrame(provenance["channels"]),
        event_id=event_id,
        provenance=provenance,
    )


def preprocess_bids(
    recording: BIDSRecording,
    settings: ProcessingSettings,
    *,
    decisions: dict[str, dict[str, Any]] | None = None,
    n_jobs: int = 1,
) -> PreprocessingResult:
    """Run the existing preprocessing pipeline with explicit BIDS metadata."""
    from eegtable.preprocessing.pipeline import preprocess

    return preprocess(
        recording.raw,
        settings,
        decisions=decisions,
        n_jobs=n_jobs,
        provenance={"bids": dict(recording.provenance)},
    )
