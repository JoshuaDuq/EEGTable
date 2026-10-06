import json
from dataclasses import replace

import pytest

from eegtable.preprocessing import list_steps, open_workflow, read_checkpoint, reset_from, run_until
from eegtable.preprocessing.config import (
    AmplitudeSettings,
    AnnotationSettings,
    ArtifactSettings,
    ICASettings,
    WorkflowSettings,
)
from eegtable.preprocessing.review import save_review

from .test_execution import config_for


def _loud(raw, tmp_path, **workflow):
    # Fp1 is the only amplitude candidate; C3 is marked bad on the file itself.
    import mne

    data = raw.get_data()
    data[0] *= 10.0
    loud = mne.io.RawArray(data, raw.info.copy(), first_samp=raw.first_samp)
    loud.info["bads"] = ["C3"]
    base = config_for(loud, tmp_path)
    return replace(
        base,
        workflow=WorkflowSettings(**workflow),
        processing=replace(
            base.processing,
            annotations=AnnotationSettings(amplitude=AmplitudeSettings(peak={"eeg": 5e-5})),
        ),
    )


def test_suggested_raw_policy_records_a_bound_decision_and_continues(raw, tmp_path):
    workflow = open_workflow(_loud(raw, tmp_path, raw_review="suggested"))
    assert run_until(workflow, "review-raw").state == "completed"
    decision = json.loads(
        (workflow.workspace / "decisions" / "review-raw.yaml").read_text(encoding="utf-8")
    )
    assert decision["bads"] == ["C3", "Fp1"]
    assert decision["spans"] == []
    assert len(decision["parent_id"]) == 64
    assert read_checkpoint(workflow, "review-raw").state.raw.info["bads"] == ["C3", "Fp1"]


def test_resumed_raw_review_records_current_policy(raw, tmp_path):
    config = _loud(raw, tmp_path, raw_review="required")
    workflow = open_workflow(config)
    assert run_until(workflow, "review-raw").state == "needs-review"
    load_id = read_checkpoint(workflow, "load").artifact_id
    workflow = open_workflow(replace(config, workflow=WorkflowSettings(raw_review="suggested")))

    assert run_until(workflow, "epoch").state == "completed"
    assert read_checkpoint(workflow, "epoch").state.provenance["raw_review"] == "suggested"
    assert read_checkpoint(workflow, "load").artifact_id == load_id


def test_changed_raw_policy_invalidates_review_and_downstream_only(raw, tmp_path):
    config = _loud(raw, tmp_path, raw_review="suggested")
    workflow = open_workflow(config)
    assert run_until(workflow, "epoch").state == "completed"
    changed = open_workflow(replace(config, workflow=WorkflowSettings(raw_review="required")))

    states = {step.stage: step.state for step in list_steps(changed)}
    assert states["annotate"] == "completed"
    assert states["review-raw"] == "stale"
    assert states["epoch"] == "stale"


def test_suggested_decision_goes_stale_with_its_detector(raw, tmp_path):
    config = _loud(raw, tmp_path, raw_review="suggested")
    workflow = open_workflow(config)
    run_until(workflow, "review-raw")
    quiet = replace(
        config.processing,
        annotations=AnnotationSettings(amplitude=AmplitudeSettings(peak={"eeg": 5e-3})),
    )
    workflow = open_workflow(replace(config, processing=quiet))
    with pytest.raises(ValueError, match="stale"):
        run_until(workflow, "review-raw")
    reset_from(workflow, "annotate")
    assert run_until(workflow, "review-raw").state == "completed"
    assert read_checkpoint(workflow, "review-raw").state.raw.info["bads"] == ["C3"]


def test_review_suggested_keeps_marked_bads(raw, tmp_path):
    workflow = open_workflow(_loud(raw, tmp_path, raw_review="required"))
    assert run_until(workflow, "review-raw").state == "needs-review"
    save_review(workflow, "raw", suggested=True)
    assert run_until(workflow, "review-raw").state == "completed"
    assert read_checkpoint(workflow, "review-raw").state.raw.info["bads"] == ["C3", "Fp1"]


def test_suggested_artifact_policy_excludes_detected_components(mixture, tmp_path):
    base = config_for(mixture, tmp_path)
    artifact = ArtifactSettings(
        "ica", ICASettings(n_components=4, eog_channels=("VEOG",), ecg_channel="ECG"), "average"
    )
    config = replace(
        base,
        processing=replace(base.processing, artifact=artifact),
        workflow=WorkflowSettings(raw_review="disabled", artifact_review="suggested"),
    )
    workflow = open_workflow(config)
    assert run_until(workflow, "review-artifact").state == "completed"
    state = read_checkpoint(workflow, "review-artifact").state
    assert state.reviewed.decision["exclude"] == state.artifact.evidence["suggested_exclude"]
    assert state.provenance["artifact_review"] == "suggested"
