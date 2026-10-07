"""Identity-bound headless review and disposable native MNE viewers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ._deps import require
from ._validation import mapping
from .checkpoints import write_pointer
from .config import BadSpan, read_yaml
from .execution import (
    Workflow,
    decision_path,
    enabled_parents,
    pending_path,
    read_decision,
    read_stage,
    require_stage,
    source_identity,
    stage_identities,
    suggested_decision,
)
from .pipeline import execute_numeric
from .provenance import identity
from .quality import apply_raw_review
from .stages import enabled, get_stage


class ReviewCancelled(ValueError):
    pass


def _confirm_choices(title: str, labels: list[str]) -> list[int]:
    require("PyQt6", "preprocessing-gui")
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QDialog,
        QDialogButtonBox,
        QVBoxLayout,
    )

    application = QApplication.instance() or QApplication([])
    dialog = QDialog()
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    boxes = [QCheckBox(label) for label in labels]
    for box in boxes:
        layout.addWidget(box)
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        raise ReviewCancelled("review: cancelled; no decision saved")
    application.processEvents()
    return [index for index, box in enumerate(boxes) if box.isChecked()]


def new_bad_spans(before: Any, after: Any, acquisition_first_samp: int) -> list[dict[str, Any]]:
    # Only spans the reviewer added are the decision; the ones already on the record would be
    # appended twice. Onsets are acquisition seconds, the same clock as annotations.bad_spans.
    def rows(inst: Any) -> list[tuple[float, float, str]]:
        annotations = inst.annotations
        return [
            (float(onset), float(duration), str(description))
            for onset, duration, description in zip(
                annotations.onset, annotations.duration, annotations.description, strict=True
            )
        ]

    origin = acquisition_first_samp / after.info["sfreq"]
    existing = set(rows(before))
    return [
        {"onset": onset - origin, "duration": duration, "description": description}
        for onset, duration, description in rows(after)
        if description.upper().startswith("BAD") and (onset, duration, description) not in existing
    ]


def viewer_decision(stage: str, state: Any) -> dict[str, Any]:
    # Viewers get disposable copies; a decision exists only after explicit confirmation.
    require("mne_qt_browser", "preprocessing-gui")
    require("PyQt6", "preprocessing-gui")
    import mne  # type: ignore[import-untyped]

    with mne.viz.use_browser_backend("qt"):
        if stage == "review-raw":
            # The reviewer starts from the detectors' verdict and edits it in the viewer.
            suggested = suggested_decision(stage, state)
            first_samp = state.provenance["original_first_samp"]
            raw = apply_raw_review(
                state.raw,
                tuple(suggested["bads"]),
                tuple(BadSpan(**span) for span in suggested["spans"]),
                acquisition_first_samp=first_samp,
                acquisition_n_times=state.provenance["original_n_times"],
            )
            raw.plot(block=True)
            _confirm_choices("Save raw review?", [])
            return {
                "bads": list(raw.info["bads"]),
                "spans": new_bad_spans(state.raw, raw, first_samp),
            }
        if stage == "review-epochs":
            epochs = state.epochs.copy()
            original = set(epochs.selection)
            epochs.plot(block=True)
            _confirm_choices("Save epoch review?", [])
            return {"exclude": sorted(int(value) for value in original - set(epochs.selection))}
        artifact = state.artifact
        if artifact.method == "ica":
            model = artifact.model.copy()
            # Paged 20 to a figure; all of them at once is taller than the screen and
            # topomap figures do not scroll.
            model.plot_components()
            model.plot_sources(state.raw.copy(), block=True)
            _confirm_choices("Save ICA review?", [])
            return {"fit_id": artifact.fit_id, "exclude": list(model.exclude)}
        if artifact.method == "ssp":
            mne.viz.plot_projs_topomap(artifact.model, info=state.raw.info)
            selected = _confirm_choices(
                "Apply SSP projectors", [str(index) for index in range(len(artifact.model))]
            )
            return {"fit_id": artifact.fit_id, "include": selected}
        artifact.model.plot()
        selected = _confirm_choices("Apply EOG regression", ["Apply reviewed regression operator"])
        return {"fit_id": artifact.fit_id, "apply": bool(selected)}


def open_viewer(state: Any) -> None:
    # A fitted ICA is what a fit-artifact checkpoint is for; every other checkpoint is its data.
    require("mne_qt_browser", "preprocessing-gui")
    import mne

    with mne.viz.use_browser_backend("qt"):
        if state.epochs is None and state.artifact is not None and state.artifact.method == "ica":
            model = state.artifact.model.copy()
            # Picks would force every component into one figure, which on a full montage is
            # taller than any screen; topomap figures have no scrollbars, so the components
            # past the fold are unreachable. The default pages them 20 to a figure instead.
            model.plot_components()
            model.plot_sources(state.raw.copy(), block=True)
            return
        (state.epochs if state.epochs is not None else state.raw).copy().plot(block=True)


def gate_view(workflow: Workflow, stage: str) -> dict[str, Any]:
    # What a reviewer needs to decide, read from checkpoint metadata and headers: the data
    # itself is never loaded except for the epoch amplitudes.
    if not get_stage(stage).review:
        raise ValueError(
            f"{stage}: not a review gate; --json describes review-raw, "
            "review-artifact, or review-epochs"
        )
    if not enabled(stage, workflow.config.processing, workflow.config.workflow):
        raise ValueError(f"{stage}: disabled by the workflow policy; no decision applies")
    identities = stage_identities(workflow, source_identity(workflow) or "pending")
    parents = {
        name: require_stage(workflow, name, identities) for name in enabled_parents(workflow, stage)
    }
    parent, path = next(iter(parents.items()))
    metadata = json.loads((path / "state.json").read_text(encoding="utf-8"))
    view = {
        "stage": stage,
        "parent": parent,
        "parent_id": identity({name: identities[name] for name in parents}),
    }
    if stage == "review-raw":
        return {**view, **_raw_gate(path, metadata)}
    if stage == "review-epochs":
        return {**view, **_epoch_gate(path)}
    return {**view, **_artifact_gate(path, metadata["artifact"])}


def _item(
    id: Any, label: str, tags: list[str], score: float | None, suggested: bool
) -> dict[str, Any]:
    return {"id": id, "label": label, "tags": tags, "score": score, "suggested": suggested}


def _raw_gate(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    import mne

    raw = mne.io.read_raw_fif(path / "data_raw.fif", preload=False, verbose="error")
    candidates = metadata["candidates"]
    reasons = _bad_channel_reasons(candidates.get("bad_channels"))
    for name in candidates.get("evidence", {}).get("amplitude", []):
        reasons.setdefault(name, []).insert(0, "amplitude")
    recorded, suggested = set(raw.info["bads"]), set(candidates.get("bads", []))
    items = [
        _item(
            name,
            name,
            [kind, *(["recorded"] if name in recorded else []), *reasons.get(name, [])],
            None,
            name in recorded or name in suggested,
        )
        for name, kind in zip(raw.ch_names, raw.get_channel_types(), strict=True)
    ]
    spans = [{**span, "suggested": True} for span in candidates.get("spans", [])]
    acquisition = metadata["provenance"]
    duration = acquisition["original_n_times"] / acquisition["original_sfreq"]
    return {"field": "bads", "items": items, "spans": spans, "duration": duration}


def _bad_channel_reasons(evidence: dict[str, Any] | None) -> dict[str, list[str]]:
    # pyprep's get_bads(as_dict=True) once per repeat: the tests that flagged a channel, and
    # how many repeats agreed when there was more than one.
    if not evidence:
        return {}
    repeats = evidence["repeats"]
    tests: dict[str, set[str]] = {}
    votes: dict[str, int] = {}
    for verdict in repeats:
        for key, names in verdict.items():
            for name in names:
                if key == "bad_all":
                    votes[name] = votes.get(name, 0) + 1
                elif key.startswith("bad_by_"):
                    tests.setdefault(name, set()).add(key.removeprefix("bad_by_").lower())
    return {
        name: sorted(found) + ([f"{votes.get(name, 0)}/{len(repeats)}"] if len(repeats) > 1 else [])
        for name, found in tests.items()
    }


def _artifact_gate(path: Path, details: dict[str, Any]) -> dict[str, Any]:
    import mne

    method, evidence = details["method"], details["evidence"]
    view = {"fit_id": details["fit_id"], "method": method}
    if method == "ica":
        model = mne.preprocessing.read_ica(path / "model-ica.fif", verbose="error")
        detectors: dict[int, list[str]] = {}
        for detector, indices in evidence["suggested"].items():
            for index in indices:
                detectors.setdefault(int(index), []).append(detector)
        iclabel, suggested = evidence.get("iclabel"), set(evidence["suggested_exclude"])
        items = []
        for index in range(model.n_components_):
            tags, score = list(detectors.get(index, [])), None
            if iclabel is not None:
                tags.insert(0, iclabel["labels"][index])
                score = round(float(max(iclabel["probabilities"][index])), 3)
            items.append(_item(index, f"ICA{index:03d}", tags, score, index in suggested))
        return {**view, "field": "exclude", "items": items}
    if method == "ssp":
        projectors = mne.read_proj(path / "model-proj.fif", verbose="error")
        items = [
            _item(index, projector["desc"], [], None, True)
            for index, projector in enumerate(projectors)
        ]
        return {**view, "field": "include", "items": items}
    apply = _item(0, "Apply the reviewed EOG regression", [], None, True)
    return {**view, "field": "apply", "items": [apply]}


def _epoch_gate(path: Path) -> dict[str, Any]:
    import mne

    epochs = mne.read_epochs(path / "data_epo.fif", preload=True, proj=False, verbose="error")
    names = {code: name for name, code in epochs.event_id.items()}
    # Peak-to-peak in microvolts, the worst channel per epoch: what a reviewer sorts by.
    amplitude = np.ptp(epochs.get_data(picks="eeg"), axis=2).max(axis=1) * 1e6
    items = [
        _item(int(original), f"epoch {original}", [names[int(code)]], round(float(ptp), 1), False)
        for original, code, ptp in zip(
            epochs.selection, epochs.events[:, 2], amplitude, strict=True
        )
    ]
    return {"field": "exclude", "items": items}


def save_review(
    workflow: Workflow, target: str, decisions: Path | None = None, *, suggested: bool = False
) -> Path:
    if target not in ("raw", "artifact", "epochs"):
        raise ValueError("review: expected raw, artifact, or epochs")
    if suggested and decisions is not None:
        raise ValueError("review: --suggested and --decisions are exclusive")
    stage = f"review-{target}"
    if not enabled(stage, workflow.config.processing, workflow.config.workflow):
        raise ValueError(f"{stage}: disabled by the workflow policy; no decision applies")
    identities = stage_identities(workflow, source_identity(workflow) or "pending")
    parents = {
        name: read_stage(workflow, name, identities) for name in enabled_parents(workflow, stage)
    }
    parent_ids = {name: checkpoint.artifact_id for name, checkpoint in parents.items()}
    state = next(iter(parents.values())).state
    pending = pending_path(workflow, stage)
    if suggested:
        decision = {"parent_id": identity(parent_ids), **suggested_decision(stage, state)}
    elif decisions is not None:
        decision = read_yaml(decisions)
    else:
        # A filled-in pending file is the headless review; the viewer is for the rest.
        decision = read_yaml(pending) if pending.exists() else {}
        if not decision or None in decision.values():
            try:
                decision = {"parent_id": identity(parent_ids), **viewer_decision(stage, state)}
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(f"{exc}; or fill in {pending}") from exc
    if decision.get("parent_id") != identity(parent_ids):
        raise ValueError("review.parent_id: stale or missing reviewed checkpoint identity")
    fields = {
        "raw": {"bads", "spans"},
        "epochs": {"exclude"},
        "artifact": {"fit_id", "exclude", "include", "apply"},
    }
    mapping(decision, f"review.{target}", fields[target] | {"parent_id"})
    payload = {key: value for key, value in decision.items() if key != "parent_id"}
    if any(value is None for value in payload.values()):
        raise ValueError("review: pending null choices must be replaced by explicit decisions")
    execute_numeric(stage, state, workflow.config.processing, payload)
    filelock = require("filelock", "preprocessing")
    with filelock.FileLock(workflow.workspace / ".writer.lock", timeout=0):
        path = decision_path(workflow, stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(
                f"{path}: decision already exists; reset --from {stage} to replace"
            )
        write_pointer(path, decision)
        read_decision(workflow, stage, parent_ids)
    return path
