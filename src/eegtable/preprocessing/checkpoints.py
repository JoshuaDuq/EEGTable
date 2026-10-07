"""Immutable native checkpoints, payload validation, and atomic pointer publication."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import mne  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from ._deps import require
from ._validation import validate_sampling_precision
from .artifacts import ArtifactModel, review_artifact
from .epochs import validate_metadata_precision
from .events import EventData
from .pipeline import StageData, result_from_state
from .provenance import canonical_json, file_hash, identity, serializable
from .rejection import RejectionModel


@dataclass(frozen=True)
class Checkpoint:
    """An immutable stage checkpoint, verified against its manifest when loaded.

    Parameters
    ----------
    artifact_id : str
        Identity of the stage result, derived from its settings, its parents'
        identities, its review decision, its reset token and package versions.
    path : pathlib.Path
        Checkpoint directory.
    state : StageData
        Data at this stage: the continuous recording before ``epoch`` and the
        epochs from then on, with events, fitted models, provenance and detector
        candidates.
    manifest : dict
        Stage, parents, settings and payload file hashes.
    """

    artifact_id: str
    path: Path
    state: StageData
    manifest: dict[str, Any]


def payload_files(path: Path) -> list[Path]:
    # Hidden entries belong to the OS, not the checkpoint: Finder's .DS_Store, exFAT's ._ files.
    return sorted(
        item for item in path.iterdir() if item.is_file() and not item.name.startswith(".")
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def write_pointer(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def save_state(path: Path, state: StageData) -> None:
    if state.raw is not None:
        state.raw.save(path / "data_raw.fif", fmt="double", overwrite=False)
    metadata: dict[str, Any] = {"provenance": state.provenance, "candidates": state.candidates}
    if state.epochs is not None:
        state.epochs.save(path / "data_epo.fif", fmt="double", overwrite=False)
    if state.events is not None:
        events = state.events
        metadata["events"] = {
            "events": events.events,
            "event_id": events.event_id,
            "original_sfreq": events.original_sfreq,
            "original_row": events.original_row,
            "original_samples": events.original_samples,
            "delay": events.delay,
            "shift_samples": events.shift_samples,
        }
        if events.metadata is not None:
            events.metadata.to_json(
                path / "metadata.json", orient="table", index=False, double_precision=15
            )
    if state.artifact is not None:
        artifact = state.artifact
        metadata["artifact"] = {
            "method": artifact.method,
            "fit_id": artifact.fit_id,
            "compatibility": artifact.compatibility,
            "evidence": artifact.evidence,
        }
        if artifact.method == "ica":
            artifact.model.save(path / "model-ica.fif")
        elif artifact.method == "ssp":
            mne.write_proj(path / "model-proj.fif", artifact.model)
        else:
            artifact.model.save(path / "model-regression.h5")
    if state.reviewed is not None:
        metadata["decision"] = state.reviewed.decision
    if state.rejection is not None:
        model = state.rejection
        model.model.save(path / "rejection.h5")
        metadata["rejection"] = {
            "channels": model.channels,
            "sfreq": model.sfreq,
            "tmin": model.tmin,
            "tmax": model.tmax,
        }
    write_json(path / "state.json", metadata)


def load_state(path: Path) -> StageData:
    metadata = json.loads((path / "state.json").read_text(encoding="utf-8"))
    raw = (
        mne.io.read_raw_fif(path / "data_raw.fif", preload=True)
        if (path / "data_raw.fif").exists()
        else None
    )
    epochs = (
        mne.read_epochs(path / "data_epo.fif", preload=True, proj=False)
        if (path / "data_epo.fif").exists()
        else None
    )
    events = None
    if "events" in metadata:
        event_data = metadata["events"]
        for key in ("events", "original_row", "original_samples"):
            event_data[key] = np.asarray(event_data[key], dtype=np.int64)
        frame = (
            pd.read_json(path / "metadata.json", orient="table")
            if (path / "metadata.json").exists()
            else None
        )
        events = EventData(metadata=frame, **event_data)
    artifact, reviewed, rejection = None, None, None
    if "artifact" in metadata:
        details = metadata["artifact"]
        method = details["method"]
        if method == "ica":
            model = mne.preprocessing.read_ica(path / "model-ica.fif")
        elif method == "ssp":
            model = mne.read_proj(path / "model-proj.fif")
        else:
            model = mne.preprocessing.read_eog_regression(path / "model-regression.h5")
        artifact = ArtifactModel(model=model, **details)
        if "decision" in metadata:
            reviewed = review_artifact(artifact, metadata["decision"])
    if "rejection" in metadata:
        auto = require("autoreject", "preprocessing-auto")
        details = metadata["rejection"]
        details["channels"] = tuple(details["channels"])
        rejection = RejectionModel(auto.read_auto_reject(path / "rejection.h5"), **details)
    return StageData(
        raw,
        events,
        epochs,
        artifact,
        reviewed,
        rejection,
        metadata["provenance"],
        metadata["candidates"],
    )


def verify_manifest(path: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 1:
        raise ValueError(f"{path}: unsupported checkpoint schema")
    if manifest["stage"] == "export" and not isinstance(manifest.get("export_provenance_id"), str):
        raise ValueError(f"{path}: export provenance identity missing; explicit reset required")
    for name, expected in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError(f"{path}: invalid payload path {name}")
        payload = path / name
        if not payload.is_file() or file_hash(payload) != expected:
            raise ValueError(
                f"{payload}: checkpoint payload hash mismatch; explicit reset required"
            )
    actual_files = {item.name for item in payload_files(path)} - {"manifest.json"}
    if actual_files != set(manifest["files"]):
        raise ValueError(f"{path}: checkpoint payload inventory mismatch")
    return manifest


def load_checkpoint(path: Path, expected_id: str) -> Checkpoint:
    manifest = verify_manifest(path)
    if manifest["artifact_id"] != expected_id:
        raise ValueError(f"{path}: stale checkpoint identity")
    return Checkpoint(expected_id, path, load_state(path), manifest)


def publish_checkpoint(
    workspace: Path,
    stage: str,
    artifact_id: str,
    state: StageData,
    parents: dict[str, str],
    settings: dict[str, Any],
) -> Checkpoint:
    root = workspace / stage
    root.mkdir(parents=True, exist_ok=True)
    destination = root / artifact_id
    if destination.exists():
        checkpoint = load_checkpoint(destination, artifact_id)
    else:
        with TemporaryDirectory(prefix=".pending-", dir=root) as temporary:
            staged = Path(temporary) / "checkpoint"
            staged.mkdir()
            save_state(staged, state)
            restored = load_state(staged)
            # FIF applies the channel calibration on write and read: expect one ulp, relatively.
            if state.raw is not None:
                validate_sampling_precision(state.raw, restored.raw)
                np.testing.assert_allclose(
                    restored.raw.get_data(), state.raw.get_data(), rtol=1e-14, atol=0
                )
            if state.epochs is not None:
                validate_sampling_precision(state.epochs, restored.epochs)
                np.testing.assert_allclose(
                    restored.epochs.get_data(), state.epochs.get_data(), rtol=1e-14, atol=0
                )
                np.testing.assert_array_equal(restored.epochs.selection, state.epochs.selection)
                validate_metadata_precision(state.epochs.metadata, restored.epochs.metadata)
            if state.events is not None:
                assert restored.events is not None
                validate_metadata_precision(state.events.metadata, restored.events.metadata)
            manifest = {
                "schema": 1,
                "stage": stage,
                "artifact_id": artifact_id,
                "parents": parents,
                "settings": serializable(settings),
                "files": {item.name: file_hash(item) for item in payload_files(staged)},
            }
            if stage == "export":
                manifest["export_provenance_id"] = identity(result_from_state(state).provenance)
            write_json(staged / "manifest.json", manifest)
            verify_manifest(staged)
            os.rename(staged, destination)
        checkpoint = load_checkpoint(destination, artifact_id)
    write_pointer(workspace / f"{stage}.json", {"schema": 1, "artifact_id": artifact_id})
    return checkpoint
