"""Explicit single-step, sequential and dependency-aware checkpoint execution."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

import mne  # type: ignore[import-untyped]
import numpy as np

from ._deps import require
from .checkpoints import Checkpoint, load_checkpoint, publish_checkpoint, write_pointer
from .checks import validate_source_destinations
from .config import (
    EventEpochSettings,
    PreprocessingConfig,
    ProcessingSettings,
    WorkflowSettings,
    read_yaml,
)
from .io import validate_bundle, write_result
from .pipeline import StageData, execute_numeric, result_from_state
from .provenance import file_hash, fingerprint, identity
from .stages import STAGES, enabled, get_stage, stage_settings

POLICIES = {
    "review-raw": "raw_review",
    "review-artifact": "artifact_review",
    "review-epochs": "epoch_review",
}
HELP = {
    "parent_id": "Identity of the reviewed checkpoint. Leave as written.",
    "fit_id": "Identity of the fitted model. Leave as written.",
    "bads": "Bad channels; replaces the recording's list. [] marks every channel good.",
    "spans": "BAD intervals to append, each {onset: seconds from the start of the recording, "
    "duration: seconds, description: BAD_...}. [] appends none.",
    "exclude": "Indices to drop: ICA components, or original epoch numbers. [] drops none.",
    "include": "SSP projector indices to apply. [] applies none.",
    "apply": "true applies the reviewed EOG regression; false leaves the data uncorrected.",
}


@dataclass(frozen=True)
class Workflow:
    """One recording's configuration bound to its checkpoint workspace.

    Parameters
    ----------
    config : PreprocessingConfig
        The recording's configuration.
    workspace : pathlib.Path
        Directory holding its checkpoints, pointers and review decisions.
    """

    config: PreprocessingConfig
    workspace: Path


@dataclass(frozen=True)
class StepStatus:
    """State of one stage, from :func:`list_steps`.

    Parameters
    ----------
    stage : str
        Stage name.
    state : str
        ``"completed"``, ``"stale"``, ``"pending"``, ``"needs-review"`` or
        ``"disabled"``.
    reason : str
        Why the stage is disabled; empty otherwise.
    path : pathlib.Path or None
        Checkpoint directory the stage's pointer names, if any.
    next_action : str
        Command-line action, with ``CONFIG`` standing for the recipe path.
    """

    stage: str
    state: str
    reason: str
    path: Path | None
    next_action: str


@dataclass(frozen=True)
class StepResult:
    """Outcome of running one stage.

    Parameters
    ----------
    stage : str
        Stage name.
    state : str
        ``"completed"``, or ``"needs-review"`` when a review gate has no decision.
    path : pathlib.Path or None
        The checkpoint directory, or the pending decision file of a review.
    next_action : str
        Follow-up command-line action, with ``CONFIG`` standing for the recipe path.
    """

    stage: str
    state: str
    path: Path | None
    next_action: str


@dataclass(frozen=True)
class RunOutcome:
    """Outcome of :func:`run_until`.

    Parameters
    ----------
    state : str
        ``"completed"``, or ``"needs-review"`` when a review gate stopped the run.
    steps : tuple of StepResult
        The stages run, in order.
    next_action : str
        Follow-up command-line action, with ``CONFIG`` standing for the recipe path.
    """

    state: str
    steps: tuple[StepResult, ...]
    next_action: str


def open_workflow(config: PreprocessingConfig) -> Workflow:
    """Bind a recording's configuration to its checkpoint workspace.

    Nothing is read or written. The workspace is
    ``output.directory / ".preprocessing" / output.name``.

    Parameters
    ----------
    config : PreprocessingConfig
        One recording's configuration, e.g. a value of :func:`load_recipe`.

    Returns
    -------
    Workflow
    """
    return Workflow(config, config.output.directory / ".preprocessing" / config.output.name)


def _enabled(workflow: Workflow, stage: str) -> bool:
    return enabled(stage, workflow.config.processing, workflow.config.workflow)


def data_source(workflow: Workflow, name: str) -> str:
    # A disabled stage forwards its data parent, parents[0]; a side branch such as a fit
    # ends with the stage that consumes it.
    while not _enabled(workflow, name):
        name = get_stage(name).parents[0]
    return name


def enabled_parents(workflow: Workflow, name: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(data_source(workflow, parent) for parent in get_stage(name).parents))


def _pointer(workflow: Workflow, name: str) -> str | None:
    path = workflow.workspace / f"{name}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != 1:
        raise ValueError(f"{path}: unsupported pointer schema")
    artifact_id = data.get("artifact_id")
    if (
        not isinstance(artifact_id, str)
        or len(artifact_id) != 64
        or any(c not in "0123456789abcdef" for c in artifact_id)
    ):
        raise ValueError(f"{path}: invalid artifact identity")
    return artifact_id


def decision_path(workflow: Workflow, name: str) -> Path:
    return workflow.workspace / "decisions" / f"{name}.yaml"


def pending_path(workflow: Workflow, name: str) -> Path:
    return decision_path(workflow, name).with_suffix(".pending.yaml")


def source_identity(workflow: Workflow) -> str | None:
    # The load checkpoint recorded the source fingerprint; reading it back spares the source.
    pointer = _pointer(workflow, "load")
    if pointer is None:
        return None
    state = json.loads(
        (workflow.workspace / "load" / pointer / "state.json").read_text(encoding="utf-8")
    )
    identity: str = state["provenance"]["input_hash"]
    return identity


def _reset_tokens(workflow: Workflow) -> dict[str, str]:
    path = workflow.workspace / "resets.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _external_file_hashes(stage: str, settings: ProcessingSettings) -> dict[str, str]:
    paths: dict[str, Path] = {}
    if stage == "prepare" and isinstance(settings.channels.montage, Path):
        paths["channels.montage"] = settings.channels.montage
    if stage == "events" and isinstance(settings.epochs, EventEpochSettings):
        if settings.epochs.events.path is not None:
            paths["epochs.events.path"] = settings.epochs.events.path
        if settings.epochs.metadata is not None:
            paths["epochs.metadata"] = settings.epochs.metadata
    return {name: file_hash(path) for name, path in paths.items()}


def _input_provenance(config: PreprocessingConfig) -> dict[str, Any]:
    if config.input.bids is None:
        return {}
    from eegtable.bids import read_bids_metadata

    settings = config.input.bids
    return {
        "bids": read_bids_metadata(
            config.input.path, root=settings.root, canonical_channels=settings.canonical_channels
        )
    }


def stage_identities(workflow: Workflow, source_id: str) -> dict[str, str]:
    reset = _reset_tokens(workflow)
    resolved: dict[str, str] = {}
    versions = {name: version(name) for name in ("mne", "numpy", "scipy")}
    # The package was called eegfeat when these identities were first recorded; the key stays so
    # that renaming it does not mark every finished preprocessing stage stale.
    versions["eegfeat"] = version("eegtable")
    for stage in STAGES:
        parents = {name: resolved[name] for name in stage.parents}
        settings = stage_settings(stage, workflow.config.processing)
        if stage.review:
            settings[POLICIES[stage.name]] = _policy(workflow, stage.name)
        if stage.name == "load" and workflow.config.input.bids is not None:
            bids = _input_provenance(workflow.config)["bids"]
            settings["bids_input"] = {
                name: bids[name]
                for name in (
                    "reader",
                    "mne_bids_version",
                    "root",
                    "path",
                    "canonical_channels",
                    "sidecar_hashes",
                )
            }
        files = _external_file_hashes(stage.name, workflow.config.processing)
        if files:
            settings["external_files"] = files
        path = decision_path(workflow, stage.name)
        decision = read_yaml(path) if stage.review and path.exists() else None
        resolved[stage.name] = identity(
            {
                "implementation": 1,
                "stage": stage.name,
                "source": source_id if stage.name == "load" else None,
                "parents": parents,
                "enabled": _enabled(workflow, stage.name),
                "settings": settings,
                "decision": decision,
                "reset": reset.get(stage.name),
                "versions": versions,
            }
        )
    return resolved


def fold_calibration(raw: Any) -> Any:
    # FIF stores the channel cal/range as float32 and applies them on write and read, so a
    # non-FIF source calibration such as 1e-7 V/bit breaks the exact checkpoint round trip.
    # Preloaded data is already calibrated: a unit calibration makes the round trip exact.
    for channel in raw.info["chs"]:
        channel["cal"] = 1.0
        channel["range"] = 1.0
    raw._cals = np.ones_like(raw._cals)
    return raw


def load_source(workflow: Workflow) -> Any:
    config = workflow.config
    if config.input.bids is None:
        raw = mne.io.read_raw(config.input.path, preload=True)
    else:
        from eegtable.bids import read_bids

        settings = config.input.bids
        raw = read_bids(
            config.input.path, root=settings.root, canonical_channels=settings.canonical_channels
        ).raw
    raw = fold_calibration(raw)
    validate_source_destinations(raw, workflow.config)
    return raw


def require_stage(workflow: Workflow, name: str, identities: dict[str, str]) -> Path:
    # Metadata only: payloads are verified when a stage consumes them, or by status --verify.
    actual = _pointer(workflow, name)
    if actual is None:
        raise ValueError(f"{name}: missing parent; run CONFIG --until {name}")
    if actual != identities[name]:
        raise ValueError(f"{name}: stale checkpoint; reset CONFIG --from {name} and run again")
    path = workflow.workspace / name / actual
    if not (path / "manifest.json").is_file():
        raise ValueError(
            f"{name}: checkpoint payload missing; reset CONFIG --from {name} and run again"
        )
    return path


def read_stage(workflow: Workflow, name: str, identities: dict[str, str]) -> Checkpoint:
    return load_checkpoint(require_stage(workflow, name, identities), identities[name])


def read_checkpoint(workflow: Workflow, stage: str) -> Checkpoint:
    """Load the checkpoint holding a stage's data, verifying its payload.

    A disabled stage resolves to the enabled stage whose data it passes on. The
    checkpoint must exist and match the current recipe and saved decisions;
    otherwise this raises with the command that repairs it.

    Parameters
    ----------
    workflow : Workflow
        From :func:`open_workflow`.
    stage : str
        Stage name, e.g. ``"filter"`` or ``"epoch"``.

    Returns
    -------
    Checkpoint
    """
    get_stage(stage)
    identities = stage_identities(workflow, source_identity(workflow) or "pending")
    return read_stage(workflow, data_source(workflow, stage), identities)


def list_steps(workflow: Workflow) -> tuple[StepStatus, ...]:
    """State of every stage, in pipeline order, from checkpoint metadata.

    A stage is ``"completed"`` when its checkpoint matches the current recipe and
    saved decisions, ``"stale"`` when a checkpoint exists but no longer matches,
    ``"needs-review"`` for a review gate whose parents have checkpoints but which
    has no decision for them, ``"disabled"`` when the recipe turns it off, and
    ``"pending"`` otherwise. Neither the source recording nor checkpoint payloads
    are read.

    Parameters
    ----------
    workflow : Workflow
        From :func:`open_workflow`.

    Returns
    -------
    tuple of StepStatus
    """
    # Reads metadata only; payload hashes are verified when a checkpoint is consumed.
    identities = stage_identities(workflow, source_identity(workflow) or "pending")
    result = []
    for stage in STAGES:
        pointer = _pointer(workflow, stage.name)
        state = "pending"
        reason = ""
        if not _enabled(workflow, stage.name):
            state, reason = "disabled", "not selected in configuration"
        elif pointer is not None:
            state = "completed" if pointer == identities[stage.name] else "stale"
        elif stage.review and all(
            _pointer(workflow, parent) is not None
            for parent in enabled_parents(workflow, stage.name)
        ):
            # A saved decision leaves only the run to do; the gate no longer waits on anyone.
            state = "pending" if _decided(workflow, stage.name, identities) else "needs-review"
        path = workflow.workspace / stage.name / pointer if pointer else None
        actions = {
            "needs-review": f"review CONFIG {stage.name.removeprefix('review-')}",
            "stale": f"reset CONFIG --from {stage.name}",
        }
        action = actions.get(state, f"run CONFIG --until {stage.name}")
        result.append(StepStatus(stage.name, state, reason, path, action))
    return tuple(result)


def _template(workflow: Workflow, stage: str, state: StageData, parents: dict[str, str]) -> None:
    path, pending = decision_path(workflow, stage), pending_path(workflow, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_id = identity(parents)
    # A pending file for this very checkpoint may already hold the reviewer's edits.
    if path.exists() or (pending.exists() and read_yaml(pending).get("parent_id") == parent_id):
        return
    template: dict[str, Any] = {"parent_id": parent_id}
    if stage == "review-raw":
        template.update(bads=None, spans=None)
    elif stage == "review-epochs":
        template["exclude"] = None
    else:
        assert state.artifact is not None
        template["fit_id"] = state.artifact.fit_id
        template[
            {"ica": "exclude", "ssp": "include", "regression": "apply"}[state.artifact.method]
        ] = None
    import yaml

    lines = [
        f"# {HELP[key]}\n{yaml.safe_dump({key: value}, default_flow_style=False)}"
        for key, value in template.items()
    ]
    pending.write_text("".join(lines), encoding="utf-8")


def _decided(workflow: Workflow, stage: str, identities: dict[str, str]) -> bool:
    path = decision_path(workflow, stage)
    parents = {name: identities[name] for name in enabled_parents(workflow, stage)}
    return path.exists() and read_yaml(path).get("parent_id") == identity(parents)


def read_decision(workflow: Workflow, stage: str, parents: dict[str, str]) -> dict[str, Any] | None:
    path = decision_path(workflow, stage)
    if not path.exists():
        return None
    decision = read_yaml(path)
    if decision.pop("parent_id", None) != identity(parents):
        raise ValueError(f"{path}: stale review parent identity; review the current checkpoint")
    if any(value is None for value in decision.values()):
        raise ValueError(f"{path}: review decision is still pending")
    return decision


def suggested_decision(stage: str, state: StageData) -> dict[str, Any]:
    # The detectors' own verdicts, taken as-is; the saved decision records that choice.
    if stage == "review-raw":
        bads = [*state.raw.info["bads"], *state.candidates.get("bads", [])]
        return {"bads": list(dict.fromkeys(bads)), "spans": list(state.candidates.get("spans", []))}
    if stage == "review-epochs":
        raise ValueError("review.epochs: no detector suggests epochs; pass --decisions")
    artifact = state.artifact
    assert artifact is not None
    if artifact.method == "ica":
        return {"fit_id": artifact.fit_id, "exclude": list(artifact.evidence["suggested_exclude"])}
    if artifact.method == "ssp":
        return {"fit_id": artifact.fit_id, "include": list(range(len(artifact.model)))}
    return {"fit_id": artifact.fit_id, "apply": True}


def _policy(workflow: Workflow, stage: str) -> str:
    policy: str = getattr(workflow.config.workflow, POLICIES.get(stage, "epoch_review"))
    return policy


def _join_branches(epoch: StageData, review: StageData) -> StageData:
    # The fit/review branch carries its own provenance; the epoch branch is the data.
    stages = epoch.provenance["stages"]
    provenance = {
        **epoch.provenance,
        "artifact": review.provenance["artifact"],
        "artifact_decision": review.provenance["artifact_decision"],
        "settings": {**epoch.provenance["settings"], **review.provenance["settings"]},
        "stages": [*stages, *(name for name in review.provenance["stages"] if name not in stages)],
    }
    return replace(epoch, artifact=review.artifact, reviewed=review.reviewed, provenance=provenance)


def _run_locked(
    workflow: Workflow, stage: str, source: Any, source_id: str, n_jobs: int, overwrite: bool
) -> StepResult:
    # Decisions and resets may have changed on disk before the lock; the source cannot.
    identities = stage_identities(workflow, source_id)
    definition = get_stage(stage)
    if not _enabled(workflow, stage):
        raise ValueError(f"{stage}: disabled by configuration; no operation performed")
    current = _pointer(workflow, stage)
    if current is not None and current != identities[stage]:
        raise ValueError(
            f"{stage}: stale checkpoint; reset CONFIG --from {stage} before recomputing"
        )
    if current == identities[stage]:
        path = require_stage(workflow, stage, identities)
        if stage == "export":
            output = workflow.config.output
            manifest = output.directory / f"{output.name}_preprocessing.json"
            # A deleted bundle is republished from the checkpoint; an intact one is verified.
            if overwrite or not manifest.exists():
                state = load_checkpoint(path, current).state
                write_result(result_from_state(state), output, overwrite=overwrite)
            else:
                validate_bundle(manifest)
        return StepResult(stage, "completed", path, "next CONFIG")
    parents = {
        name: read_stage(workflow, name, identities) for name in enabled_parents(workflow, stage)
    }
    parent_ids = {name: checkpoint.artifact_id for name, checkpoint in parents.items()}
    state = (
        next(iter(parents.values())).state
        if parents
        else StageData(
            source,
            provenance={
                **_policies(workflow.config.workflow),
                **_input_provenance(workflow.config),
            },
        )
    )
    if stage == "apply-artifact":
        state = _join_branches(parents["epoch"].state, parents["review-artifact"].state)
    state = replace(state, provenance={**state.provenance, **_policies(workflow.config.workflow)})
    decision = None
    if definition.review:
        decision = read_decision(workflow, stage, parent_ids)
        if decision is None and _policy(workflow, stage) == "suggested":
            decision = suggested_decision(stage, state)
            path = decision_path(workflow, stage)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_pointer(path, {"parent_id": identity(parent_ids), **decision})
            # The decision is part of this stage's identity, so it changes now.
            identities = stage_identities(workflow, source_id)
        if decision is None:
            _template(workflow, stage, state, parent_ids)
            return StepResult(
                stage,
                "needs-review",
                pending_path(workflow, stage),
                f"review CONFIG {stage.removeprefix('review-')}",
            )
    state = execute_numeric(stage, state, workflow.config.processing, decision, n_jobs=n_jobs)
    if stage == "export":
        write_result(result_from_state(state), workflow.config.output, overwrite=overwrite)
    checkpoint = publish_checkpoint(
        workflow.workspace,
        stage,
        identities[stage],
        state,
        parent_ids,
        stage_settings(definition, workflow.config.processing),
    )
    return StepResult(
        stage,
        "completed",
        checkpoint.path,
        "next CONFIG" if stage != "export" else "Feature-ready epochs exported",
    )


def _policies(workflow: WorkflowSettings) -> dict[str, Any]:
    return {name: getattr(workflow, name) for name in POLICIES.values()}


def run_step(
    workflow: Workflow, stage: str, *, n_jobs: int = 1, overwrite: bool = False
) -> StepResult:
    """Run one stage whose enabled parents already have current checkpoints.

    Reads and fingerprints the source recording, then runs the stage under the
    workspace's writer lock; a second writer raises instead of waiting. A current
    checkpoint is reused and a stale one raises until it is reset. A review stage
    without a saved decision writes a pending decision file and returns
    ``"needs-review"``, unless its policy is ``"suggested"``, which saves the
    detectors' verdict as the decision and runs.

    Parameters
    ----------
    workflow : Workflow
        From :func:`open_workflow`.
    stage : str
        Stage name; a disabled stage raises.
    n_jobs : int, default 1
        Passed to filtering and resampling.
    overwrite : bool, default False
        Let ``export`` replace existing bundle files, including republishing a
        bundle that already matches its checkpoint. A missing bundle is
        republished without it.

    Returns
    -------
    StepResult
    """
    get_stage(stage)
    if not _enabled(workflow, stage):
        raise ValueError(f"{stage}: disabled by configuration")
    source = load_source(workflow)
    return _step(workflow, stage, source, fingerprint(source), n_jobs, overwrite)


def _step(
    workflow: Workflow, stage: str, source: Any, source_id: str, n_jobs: int, overwrite: bool
) -> StepResult:
    # Validate prerequisites before creating any output directories.
    identities = stage_identities(workflow, source_id)
    for parent in enabled_parents(workflow, stage):
        require_stage(workflow, parent, identities)
    workflow.workspace.mkdir(parents=True, exist_ok=True)
    filelock = require("filelock", "preprocessing")
    with filelock.FileLock(workflow.workspace / ".writer.lock", timeout=0):
        return _run_locked(workflow, stage, source, source_id, n_jobs, overwrite)


def run_next(workflow: Workflow, *, n_jobs: int = 1, overwrite: bool = False) -> StepResult:
    """Run the first stage, in pipeline order, that is neither completed nor disabled.

    Runs as :func:`run_step`. When every stage is completed, ``export`` runs again,
    which verifies the bundle or republishes it.

    Parameters
    ----------
    workflow : Workflow
        From :func:`open_workflow`.
    n_jobs : int, default 1
        Passed to filtering and resampling.
    overwrite : bool, default False
        As for :func:`run_step`.

    Returns
    -------
    StepResult
    """
    pending = (
        status.stage
        for status in list_steps(workflow)
        if status.state not in ("completed", "disabled")
    )
    source = load_source(workflow)
    return _step(workflow, next(pending, "export"), source, fingerprint(source), n_jobs, overwrite)


def run_until(
    workflow: Workflow,
    stage: str = "export",
    *,
    n_jobs: int = 1,
    overwrite: bool = False,
    on_step: Callable[[StepResult, int, int], None] | None = None,
) -> RunOutcome:
    """Run, in order, every enabled stage up to and including ``stage``.

    The source recording is read and fingerprinted once. Each stage runs as by
    :func:`run_step`, reusing current checkpoints, and the run stops at the first
    review gate without a decision.

    Parameters
    ----------
    workflow : Workflow
        From :func:`open_workflow`.
    stage : str, default "export"
        Last stage to run.
    n_jobs : int, default 1
        Passed to filtering and resampling.
    overwrite : bool, default False
        As for :func:`run_step`.
    on_step : callable, optional
        Called as ``on_step(result, index, total)`` after each stage, ``index``
        counting from 1.

    Returns
    -------
    RunOutcome
    """
    get_stage(stage)
    needed: set[str] = set()

    def visit(name: str) -> None:
        for parent in get_stage(name).parents:
            visit(parent)
        if _enabled(workflow, name):
            needed.add(name)

    visit(stage)
    ordered = [definition.name for definition in STAGES if definition.name in needed]
    # One read and one hash of the source serve every step; stages never mutate it.
    source = load_source(workflow)
    source_id = fingerprint(source)
    results = []
    for index, name in enumerate(ordered, 1):
        result = _step(workflow, name, source, source_id, n_jobs, overwrite)
        results.append(result)
        if on_step is not None:
            on_step(result, index, len(ordered))
        if result.state == "needs-review":
            return RunOutcome("needs-review", tuple(results), result.next_action)
    return RunOutcome(
        "completed", tuple(results), results[-1].next_action if results else "No enabled stages"
    )


def reset_from(workflow: Workflow, stage: str) -> tuple[str, ...]:
    """Retire a stage and every stage that depends on it.

    Their checkpoint pointers are removed and saved decisions renamed aside;
    checkpoint payloads stay on disk. The stage also gets a new reset token, so
    it is recomputed on the next run even with unchanged settings.

    Parameters
    ----------
    workflow : Workflow
        From :func:`open_workflow`.
    stage : str
        First stage to retire.

    Returns
    -------
    tuple of str
        The retired stages, in pipeline order.
    """
    # Pointers and decisions are retired, never the immutable payloads.
    get_stage(stage)
    descendants = {stage}
    for definition in STAGES:
        if set(definition.parents) & descendants:
            descendants.add(definition.name)
    workflow.workspace.mkdir(parents=True, exist_ok=True)
    filelock = require("filelock", "preprocessing")
    with filelock.FileLock(workflow.workspace / ".writer.lock", timeout=0):
        tokens = _reset_tokens(workflow)
        tokens[stage] = uuid4().hex
        write_pointer(workflow.workspace / "resets.json", tokens)
        for name in descendants:
            pointer = workflow.workspace / f"{name}.json"
            if pointer.exists():
                pointer.unlink()
            decision = decision_path(workflow, name)
            if decision.exists():
                decision.rename(decision.with_name(f"{decision.stem}.{uuid4().hex}.yaml"))
    return tuple(definition.name for definition in STAGES if definition.name in descendants)
