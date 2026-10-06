"""One numerical stage path shared by in-memory and persisted workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import (
    ArtifactModel,
    ReviewedArtifact,
    apply_artifact,
    fit_eog_regression,
    fit_ssp,
    reference_artifact_data,
    review_artifact,
)
from .checks import validate_processing
from .config import (
    AutoRejectSettings,
    BadSpan,
    EventEpochSettings,
    ICASettings,
    ProcessingSettings,
    RegressionSettings,
    SSPSettings,
    ThresholdSettings,
    WorkflowSettings,
)
from .epochs import (
    analysis_bounds,
    baseline_epochs,
    detrend_epochs,
    interpolate_channels,
    make_epochs,
    reference_epochs,
    validate_epochs,
)
from .events import EventData, attach_bids_metadata, resolve_events
from .ica import fit_ica
from .provenance import fingerprint, serializable
from .quality import (
    _validate_stimulation_input,
    apply_raw_review,
    detect_annotations,
    detect_bad_channels,
    detect_bridges,
    repair_stimulation,
    stimulation_intervals,
)
from .raw import (
    acquisition_spans,
    annotate_raw,
    crop_raw,
    filter_raw,
    notch_raw,
    prepare_channels,
    require_names,
    restore_reference_channels,
    validate_source_raw,
)
from .rejection import (
    RejectionModel,
    apply_epoch_review,
    apply_rejection,
    fit_rejection,
    reject_epochs,
)
from .sampling import crop_epochs, resample_epochs
from .stages import STAGES, enabled, get_stage, stage_settings


@dataclass(frozen=True)
class StageData:
    raw: Any  # the continuous recording until the epoch stage, then None
    events: EventData | None = None
    epochs: Any = None
    artifact: ArtifactModel | None = None
    reviewed: ReviewedArtifact | None = None
    rejection: RejectionModel | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    candidates: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessingResult:
    """Feature-ready epochs and their records, from :func:`preprocess` or an export.

    Parameters
    ----------
    epochs : mne.Epochs
        Final epochs.
    events : DataFrame
        One row per input event: ``original_row``, ``original_sample``,
        ``event_sample``, ``event_code``, ``label``, ``event_sample_sfreq``,
        ``retained``, ``epoch_row`` and ``drop_reason``, plus BIDS event metadata
        for BIDS input.
    provenance : dict
        Settings, stages run, review decisions, and the final sampling rate, bad
        channels, time bounds and epoch counts.
    repairs : DataFrame or None
        When autoreject ran, one row per repaired epoch and channel:
        ``original_row``, ``channel`` and ``label`` (``"bad"`` or
        ``"interpolated"``).
    """

    epochs: Any
    events: pd.DataFrame
    provenance: dict[str, Any]
    repairs: pd.DataFrame | None = None


def _repair_ledger(repair: dict[str, Any]) -> pd.DataFrame:
    # autoreject labels: 0 good, 1 bad, 2 interpolated; long form keyed by original event row.
    labels = np.asarray(repair["labels"])
    rows, columns = np.nonzero(labels)
    return pd.DataFrame(
        {
            "original_row": np.asarray(repair["original_rows"])[rows],
            "channel": np.asarray(repair["channels"])[columns],
            "label": np.where(labels[rows, columns] == 2, "interpolated", "bad"),
        }
    )


def result_from_state(state: StageData) -> PreprocessingResult:
    validate_epochs(state.epochs)
    if state.events is None:
        raise ValueError("report: missing original events")
    events = state.events
    retained = {int(original): index for index, original in enumerate(state.epochs.selection)}
    labels = {code: name for name, code in events.event_id.items()}
    ledger = pd.DataFrame(
        {
            "original_row": events.original_row,
            "original_sample": events.original_samples,
            "event_sample": events.events[:, 0],
            "event_code": events.events[:, 2],
            "label": [labels.get(int(code), "") for code in events.events[:, 2]],
            "event_sample_sfreq": events.original_sfreq,
            "retained": [index in retained for index in range(len(events.events))],
            "epoch_row": pd.array(
                [retained.get(index) for index in range(len(events.events))], dtype="Int64"
            ),
            "drop_reason": [";".join(reasons) for reasons in state.epochs.drop_log],
        }
    )
    if "bids" in state.provenance and events.metadata is not None:
        shared = set(ledger).intersection(events.metadata)
        if shared:
            raise ValueError(
                f"BIDS event metadata collides with ledger columns: {sorted(shared, key=str)}."
            )
        ledger = pd.concat([ledger, events.metadata.reset_index(drop=True)], axis=1)
    repair = state.provenance.get("repair")
    provenance = {
        **state.provenance,
        "final_sfreq": state.epochs.info["sfreq"],
        "final_bads": state.epochs.info["bads"],
        "final_times": [state.epochs.tmin, state.epochs.tmax],
        "retained": len(state.epochs),
        "original_events": len(events.events),
    }
    if repair is not None:
        # The per-epoch label matrix lives in the repair ledger, not the manifest.
        provenance["repair"] = {key: value for key, value in repair.items() if key != "labels"}
    return PreprocessingResult(
        state.epochs, ledger, provenance, None if repair is None else _repair_ledger(repair)
    )


Operation = Callable[["StageData", ProcessingSettings, Any, int], "StageData"]


def _stage_load(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    raw = state.raw
    validate_source_raw(raw)
    return replace(
        state,
        raw=raw.copy(),
        provenance={
            **state.provenance,
            "input_hash": fingerprint(raw),
            "original_first_samp": raw.first_samp,
            "original_n_times": raw.n_times,
            "original_sfreq": raw.info["sfreq"],
            "original_description": raw.info["description"],
        },
    )


def _stage_prepare(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    prepared = prepare_channels(state.raw, settings.channels)
    validate_processing(prepared, settings)
    return replace(state, raw=prepared)


def _stage_events(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    epoch_settings = settings.epochs
    metadata = (
        pd.read_csv(epoch_settings.metadata, sep="\t")
        if isinstance(epoch_settings, EventEpochSettings) and epoch_settings.metadata
        else None
    )
    source = (
        epoch_settings.events if isinstance(epoch_settings, EventEpochSettings) else epoch_settings
    )
    events = resolve_events(state.raw, source, metadata=metadata)
    if "bids" in state.provenance:
        events = attach_bids_metadata(state.raw, events, source, state.provenance["bids"])
    return replace(state, events=events)


def _stage_crop_raw(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert settings.crop is not None
    return replace(state, raw=crop_raw(state.raw, settings.crop))


def _acquisition(state: StageData) -> dict[str, int]:
    # Span onsets are seconds from the first acquired sample, even after crop-raw.
    return {
        "acquisition_first_samp": state.provenance["original_first_samp"],
        "acquisition_n_times": state.provenance["original_n_times"],
    }


def _require_decision(stage: str, decision: Any) -> dict[str, Any]:
    if decision is None:
        raise ValueError(f"{stage}: explicit decision required")
    return dict(decision)


def _stage_annotate(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    raw, events = state.raw, state.events
    annotated = annotate_raw(raw, settings.annotations, **_acquisition(state))
    suggestions = detect_annotations(
        annotated, settings.annotations, None if events is None else events.events
    )
    first_samp = state.provenance["original_first_samp"]
    spans = [
        span
        for annotations in suggestions.annotations
        for span in acquisition_spans(annotated, annotations, first_samp)
    ]
    return replace(
        state,
        raw=annotated,
        candidates={
            "bads": list(suggestions.bads),
            "spans": spans,
            "evidence": serializable(suggestions.evidence),
        },
    )


def _stage_detect_bads(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    candidates = dict(state.candidates)
    if settings.bad_channels is not None:
        found = detect_bad_channels(state.raw, settings.bad_channels)
        candidates["bads"] = list(dict.fromkeys([*candidates.get("bads", []), *found.bads]))
        candidates["bad_channels"] = serializable(found.evidence)
    if settings.bridges:
        candidates["bridges"] = serializable(detect_bridges(state.raw).evidence)
    return replace(state, candidates=candidates)


def _stage_review_raw(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    decision = _require_decision("review-raw", decision)
    spans = tuple(BadSpan(**span) for span in decision["spans"])
    reviewed = apply_raw_review(state.raw, tuple(decision["bads"]), spans, **_acquisition(state))
    return replace(state, raw=reviewed, provenance={**state.provenance, "raw_decision": decision})


def _stage_repair_stim(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert settings.stimulation is not None and state.events is not None
    events = state.events.events
    _validate_stimulation_input(state.raw, events, settings.stimulation)
    retained = events[
        np.isin(events[:, 2], settings.stimulation.event_ids)
        & (events[:, 0] >= state.raw.first_samp)
        & (events[:, 0] <= state.raw.last_samp)
    ]
    repaired, intervals = state.raw, []
    if len(retained):
        stimulation = replace(
            settings.stimulation,
            event_ids=tuple(
                code for code in settings.stimulation.event_ids if code in retained[:, 2]
            ),
        )
        intervals = stimulation_intervals(state.raw, retained, stimulation)
        repaired = repair_stimulation(state.raw, retained, stimulation)
    return replace(
        state,
        raw=repaired,
        provenance={**state.provenance, "repaired_samples": intervals},
    )


def _stage_notch(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    return replace(state, raw=notch_raw(state.raw, settings.filter, n_jobs=n_jobs))


def _stage_filter(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    return replace(state, raw=filter_raw(state.raw, settings.filter, n_jobs=n_jobs))


def _stage_artifact_reference(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert settings.artifact is not None and settings.artifact.reference is not None
    raw, provenance = state.raw, state.provenance
    if settings.reference.add_channels:
        raw = restore_reference_channels(
            raw, settings.reference.add_channels, settings.channels.montage
        )
        provenance = {
            **provenance,
            "restored_reference_channels": list(settings.reference.add_channels),
        }
    return replace(
        state,
        raw=reference_artifact_data(raw, settings.artifact.reference),
        provenance=provenance,
    )


def _stage_fit_artifact(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert settings.artifact is not None
    model_settings = settings.artifact.settings
    if isinstance(model_settings, ICASettings):
        model = fit_ica(state.raw, model_settings)
    elif isinstance(model_settings, SSPSettings):
        model = fit_ssp(state.raw, model_settings)
    else:
        assert isinstance(model_settings, RegressionSettings)
        model = fit_eog_regression(state.raw, model_settings)
    record = {"method": model.method, "fit_id": model.fit_id, "evidence": model.evidence}
    return replace(state, artifact=model, provenance={**state.provenance, "artifact": record})


def _stage_review_artifact(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert state.artifact is not None
    decision = _require_decision("review-artifact", decision)
    return replace(
        state,
        reviewed=review_artifact(state.artifact, decision),
        provenance={**state.provenance, "artifact_decision": decision},
    )


def _stage_epoch(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert state.events is not None
    thresholds = settings.rejection if isinstance(settings.rejection, ThresholdSettings) else None
    epochs = make_epochs(state.raw, state.events, settings.epochs, thresholds)
    # Nothing downstream reads the continuous data; carrying it would copy it into every
    # later checkpoint.
    return replace(state, raw=None, epochs=epochs)


def _stage_apply_artifact(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert state.reviewed is not None
    return replace(state, epochs=apply_artifact(state.epochs, state.reviewed))


def _stage_fit_rejection(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert isinstance(settings.rejection, AutoRejectSettings) and state.events is not None
    tmin, tmax = analysis_bounds(settings.epochs, state.events.original_sfreq)
    return replace(state, rejection=fit_rejection(state.epochs, settings.rejection, tmin, tmax))


def _stage_reject(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    if isinstance(settings.rejection, ThresholdSettings):
        return replace(state, epochs=reject_epochs(state.epochs, settings.rejection))
    assert state.rejection is not None
    cleaned, log = apply_rejection(state.epochs, state.rejection)
    # autoreject labels channels it did not fit with NaN; keep only fitted channels.
    fitted = np.isfinite(log.labels).all(axis=0)
    repair = {
        "original_rows": state.epochs.selection,
        "channels": [name for name, keep in zip(log.ch_names, fitted, strict=True) if keep],
        "bad_epochs": log.bad_epochs,
        "labels": log.labels[:, fitted],
    }
    return replace(state, epochs=cleaned, provenance={**state.provenance, "repair": repair})


def _stage_review_epochs(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    decision = _require_decision("review-epochs", decision)
    return replace(
        state,
        epochs=apply_epoch_review(state.epochs, tuple(decision["exclude"])),
        provenance={**state.provenance, "epoch_decision": decision},
    )


def _stage_interpolate(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    return replace(state, epochs=interpolate_channels(state.epochs))


def _stage_reference(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    epochs, reference, provenance = state.epochs, settings.reference, state.provenance
    if reference.add_channels:
        restored = provenance.get("restored_reference_channels", [])
        if restored:
            if restored != list(reference.add_channels):
                raise ValueError("reference.add_channels: restored electrodes disagree with recipe")
            require_names(epochs, reference.add_channels, "reference.add_channels")
        else:
            epochs = restore_reference_channels(
                epochs, reference.add_channels, settings.channels.montage
            )
            provenance = {**provenance, "restored_reference_channels": list(reference.add_channels)}
        reference = replace(reference, add_channels=())
    return replace(state, epochs=reference_epochs(epochs, reference), provenance=provenance)


def _stage_resample(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert settings.sampling is not None
    epochs = resample_epochs(state.epochs, settings.sampling, settings.filter, n_jobs=n_jobs)
    return replace(state, epochs=epochs)


def _stage_crop_epochs(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert state.events is not None
    bounds = analysis_bounds(settings.epochs, state.events.original_sfreq)
    return replace(state, epochs=crop_epochs(state.epochs, *bounds))


def _stage_detrend(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert settings.epochs.detrend is not None
    return replace(state, epochs=detrend_epochs(state.epochs, settings.epochs.detrend))


def _stage_baseline(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert settings.epochs.baseline is not None
    return replace(state, epochs=baseline_epochs(state.epochs, settings.epochs.baseline))


def _stage_report(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    result_from_state(state)
    return state


OPERATIONS: dict[str, Operation] = {
    "load": _stage_load,
    "prepare": _stage_prepare,
    "events": _stage_events,
    "crop-raw": _stage_crop_raw,
    "annotate": _stage_annotate,
    "detect-bads": _stage_detect_bads,
    "review-raw": _stage_review_raw,
    "repair-stim": _stage_repair_stim,
    "notch": _stage_notch,
    "filter": _stage_filter,
    "artifact-reference": _stage_artifact_reference,
    "fit-artifact": _stage_fit_artifact,
    "review-artifact": _stage_review_artifact,
    "epoch": _stage_epoch,
    "apply-artifact": _stage_apply_artifact,
    "fit-rejection": _stage_fit_rejection,
    "reject": _stage_reject,
    "review-epochs": _stage_review_epochs,
    "interpolate": _stage_interpolate,
    "reference": _stage_reference,
    "resample": _stage_resample,
    "crop-epochs": _stage_crop_epochs,
    "detrend": _stage_detrend,
    "baseline": _stage_baseline,
    "report": _stage_report,
    "export": _stage_report,
}


def execute_numeric(
    stage: str,
    state: StageData,
    settings: ProcessingSettings,
    decision: dict[str, Any] | None = None,
    *,
    n_jobs: int = 1,
) -> StageData:
    definition = get_stage(stage)
    provenance = {
        **state.provenance,
        "stages": [*state.provenance.get("stages", []), stage],
        "settings": {
            **state.provenance.get("settings", {}),
            **stage_settings(definition, settings),
        },
    }
    return OPERATIONS[stage](replace(state, provenance=provenance), settings, decision, n_jobs)


def preprocess(
    raw: Any,
    settings: ProcessingSettings,
    *,
    decisions: dict[str, dict[str, Any]] | None = None,
    n_jobs: int = 1,
    provenance: Mapping[str, Any] | None = None,
) -> PreprocessingResult:
    """Run every enabled stage on an in-memory recording and return the epochs.

    The stages and numerical operations are those of the checkpointed workflow,
    but nothing is written and the input is not modified. There is no unattended
    review policy: raw and epoch review run only when ``decisions`` holds
    ``"review-raw"`` or ``"review-epochs"``, so without them detector candidates
    are not applied. An enabled artifact correction requires a
    ``"review-artifact"`` decision naming the fitted model's ``fit_id``; fit and
    review with the separate numerical functions when that decision depends on
    inspecting the fit.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous recording with finite EEG, EOG and ECG samples.
    settings : ProcessingSettings
        Channel, filter, epoch, artifact, rejection and sampling settings.
    decisions : dict of str to dict, optional
        Review decisions by stage: ``"review-raw"`` with ``bads`` and ``spans``,
        ``"review-artifact"`` with ``fit_id`` and ``exclude``, ``include`` or
        ``apply``, and ``"review-epochs"`` with ``exclude``, as in a saved
        decision file.
    n_jobs : int, default 1
        Passed to filtering and resampling.
    provenance : mapping, optional
        Entries added to the result's provenance.

    Returns
    -------
    PreprocessingResult
    """
    # Same catalog as the checkpointed path; a review stage runs only with its decision.
    reviews = {} if decisions is None else decisions
    workflow = WorkflowSettings(
        raw_review="required" if "review-raw" in reviews else "disabled",
        epoch_review="required" if "review-epochs" in reviews else "disabled",
    )
    state = StageData(
        raw,
        provenance={
            **({} if provenance is None else provenance),
            "settings": serializable(settings),
            "raw_review": workflow.raw_review,
            "artifact_review": workflow.artifact_review,
            "epoch_review": workflow.epoch_review,
        },
    )
    for stage in STAGES:
        if enabled(stage.name, settings, workflow) and stage.name not in ("report", "export"):
            state = execute_numeric(
                stage.name, state, settings, reviews.get(stage.name), n_jobs=n_jobs
            )
    return result_from_state(state)
