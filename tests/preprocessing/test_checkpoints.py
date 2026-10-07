from dataclasses import replace
from datetime import date

import mne
import numpy as np
import pandas as pd
import pytest

from eegtable.preprocessing import list_steps, open_workflow, run_step, run_until
from eegtable.preprocessing.config import (
    ArtifactSettings,
    AutoRejectSettings,
    BadChannelSettings,
    FilterSettings,
    ICASettings,
    ICLabelSettings,
)

from .test_execution import config_for


def test_baseline_change_does_not_invalidate_upstream(raw, tmp_path):
    workflow = open_workflow(config_for(raw, tmp_path))
    run_until(workflow, "crop-epochs")
    config = workflow.config
    settings = replace(
        config.processing, epochs=replace(config.processing.epochs, baseline=(0, 0.2))
    )
    changed = open_workflow(replace(config, processing=settings))
    statuses = {item.stage: item.state for item in list_steps(changed)}
    assert statuses["epoch"] == statuses["crop-epochs"] == "completed"
    assert statuses["baseline"] == "pending"


def test_source_change_requires_explicit_reset(raw, tmp_path):
    workflow = open_workflow(config_for(raw, tmp_path))
    run_step(workflow, "load")
    raw.apply_function(lambda values: values * 2, picks=["C3"])
    raw.save(workflow.config.input.path, fmt="double", overwrite=True)
    with pytest.raises(ValueError, match="reset"):
        run_step(workflow, "load")


def test_source_description_change_requires_explicit_reset(raw, tmp_path):
    from eegtable.preprocessing import read_checkpoint

    raw.info["description"] = "Original acquisition notes"
    workflow = open_workflow(config_for(raw, tmp_path))
    run_step(workflow, "load")
    assert read_checkpoint(workflow, "load").state.provenance["original_description"] == (
        raw.info["description"]
    )
    raw.info["description"] = "Corrected acquisition notes"
    raw.save(workflow.config.input.path, fmt="double", overwrite=True)

    with pytest.raises(ValueError, match="reset"):
        run_step(workflow, "load")


@pytest.mark.parametrize("field,value", [("his_id", "subject-B"), ("birthday", date(2001, 2, 3))])
def test_source_subject_change_requires_explicit_reset(raw, tmp_path, field, value):
    raw.info["subject_info"] = {"his_id": "subject-A", "birthday": date(2000, 1, 2)}
    workflow = open_workflow(config_for(raw, tmp_path))
    run_step(workflow, "load")
    raw.info["subject_info"][field] = value
    raw.save(workflow.config.input.path, fmt="double", overwrite=True)

    with pytest.raises(ValueError, match="reset"):
        run_step(workflow, "load")


@pytest.mark.parametrize(
    "package,stage,field,value",
    [
        ("pyprep", "detect-bads", "bad_channels", BadChannelSettings()),
        (
            "autoreject",
            "fit-rejection",
            "rejection",
            AutoRejectSettings((1,), (0.5,), 2),
        ),
        (
            "scikit-learn",
            "fit-rejection",
            "rejection",
            AutoRejectSettings((1,), (0.5,), 2),
        ),
        (
            "scikit-learn",
            "fit-artifact",
            "artifact",
            ArtifactSettings("ica", ICASettings(n_components=4)),
        ),
        (
            "python-picard",
            "fit-artifact",
            "artifact",
            ArtifactSettings("ica", ICASettings(method="picard")),
        ),
        (
            "mne-icalabel",
            "fit-artifact",
            "artifact",
            ArtifactSettings(
                "ica", ICASettings(method="infomax", iclabel=ICLabelSettings()), "average"
            ),
        ),
        (
            "onnxruntime",
            "fit-artifact",
            "artifact",
            ArtifactSettings(
                "ica", ICASettings(method="infomax", iclabel=ICLabelSettings()), "average"
            ),
        ),
    ],
)
def test_scientific_package_change_invalidates_owning_stage(
    raw, tmp_path, monkeypatch, package, stage, field, value
):
    from eegtable.preprocessing import execution

    base = config_for(raw, tmp_path)
    workflow = open_workflow(replace(base, processing=replace(base.processing, **{field: value})))
    installed_version = execution.version
    monkeypatch.setattr(
        execution, "version", lambda name: "before" if name == package else installed_version(name)
    )
    before = execution.stage_identities(workflow, "source")
    monkeypatch.setattr(
        execution, "version", lambda name: "after" if name == package else installed_version(name)
    )
    after = execution.stage_identities(workflow, "source")

    assert before["load"] == after["load"]
    assert before[stage] != after[stage]
    assert before["export"] != after["export"]


def test_disabled_filter_change_does_not_invalidate_events(raw, tmp_path):
    workflow = open_workflow(config_for(raw, tmp_path))
    run_until(workflow, "epoch")
    changed = open_workflow(
        replace(
            workflow.config,
            processing=replace(workflow.config.processing, filter=FilterSettings(h_freq=40)),
        )
    )
    statuses = {item.stage: item.state for item in list_steps(changed)}
    assert statuses["events"] == "completed"
    assert statuses["epoch"] == "stale"


def test_reset_retires_descendants_and_keeps_payloads(raw, tmp_path):
    from eegtable.preprocessing import reset_from

    workflow = open_workflow(config_for(raw, tmp_path))
    run_until(workflow, "epoch")
    payloads = sorted(workflow.workspace.glob("*/*/manifest.json"))
    assert reset_from(workflow, "events")[:2] == ("events", "crop-raw")
    statuses = {item.stage: item.state for item in list_steps(workflow)}
    assert statuses["prepare"] == "completed"
    assert statuses["events"] == statuses["epoch"] == "pending"
    assert sorted(workflow.workspace.glob("*/*/manifest.json")) == payloads
    assert run_until(workflow, "epoch").state == "completed"


def test_second_writer_is_refused(raw, tmp_path):
    import filelock

    workflow = open_workflow(config_for(raw, tmp_path))
    run_step(workflow, "load")
    with (
        filelock.FileLock(workflow.workspace / ".writer.lock", timeout=0),
        pytest.raises(filelock.Timeout),
    ):
        run_step(workflow, "prepare")


def test_hidden_files_beside_payloads_are_ignored(raw, tmp_path):
    from eegtable.preprocessing import read_checkpoint

    workflow = open_workflow(config_for(raw, tmp_path))
    result = run_step(workflow, "load")
    # Finder drops .DS_Store into any browsed folder; exFAT drives add ._ AppleDouble files.
    (result.path / ".DS_Store").write_bytes(b"\0")
    (result.path / "._data_raw.fif").write_bytes(b"\0")
    read_checkpoint(workflow, "load")


def test_epoch_checkpoints_hold_no_raw(raw, tmp_path):
    from eegtable.preprocessing import read_checkpoint

    workflow = open_workflow(config_for(raw, tmp_path))
    run_until(workflow, "crop-epochs")
    assert read_checkpoint(workflow, "events").state.raw is not None
    for stage in ("epoch", "crop-epochs"):
        checkpoint = read_checkpoint(workflow, stage)
        assert checkpoint.state.raw is None
        assert not (checkpoint.path / "data_raw.fif").exists()


def test_checkpoint_refuses_epoch_metadata_precision_loss(raw, tmp_path):
    from eegtable.preprocessing.checkpoints import publish_checkpoint
    from eegtable.preprocessing.pipeline import StageData

    epochs = mne.Epochs(
        raw,
        np.array([[raw.first_samp + 100, 0, 1]]),
        event_id={"stimulus": 1},
        tmin=0,
        tmax=0.5,
        baseline=None,
        metadata=pd.DataFrame({"effect": [1e-11]}),
        preload=True,
        verbose=False,
    )

    with pytest.raises(ValueError, match="metadata.*precision.*rescale"):
        publish_checkpoint(tmp_path, "epoch", "a" * 64, StageData(None, epochs=epochs), {}, {})

    assert not (tmp_path / "epoch.json").exists()
    assert not (tmp_path / "epoch" / ("a" * 64)).exists()


def test_checkpoint_refuses_event_metadata_precision_loss(raw, tmp_path):
    from eegtable.preprocessing.checkpoints import publish_checkpoint
    from eegtable.preprocessing.config import EventSettings
    from eegtable.preprocessing.events import resolve_events
    from eegtable.preprocessing.pipeline import StageData

    events = resolve_events(
        raw,
        EventSettings("annotations", {"stimulus": 1}),
        events=np.array([[raw.first_samp + 100, 0, 1]]),
        metadata=pd.DataFrame({"effect": [1.23456789e-14]}),
    )

    with pytest.raises(ValueError, match="metadata.*precision.*rescale"):
        publish_checkpoint(tmp_path, "events", "b" * 64, StageData(raw, events), {}, {})

    assert not (tmp_path / "events.json").exists()
    assert not (tmp_path / "events" / ("b" * 64)).exists()
