from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import pytest

from eegtable import bids
from eegtable.preprocessing import load_recipe, open_workflow, read_checkpoint, run_until
from eegtable.preprocessing.config import EventEpochSettings, EventSettings, ProcessingSettings

mne_bids = pytest.importorskip("mne_bids")
pytest.importorskip("pybv")


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "bids"
    paths = []
    rng = np.random.default_rng(21)
    for subject in ("01", "02"):
        raw = mne.io.RawArray(
            rng.normal(0, 1e-6, (4, 1000)),
            mne.create_info(["C3", "C4", "P3", "P4"], 100.0, "eeg"),
            verbose=False,
        )
        raw.set_montage("colin27_1020")
        raw.info["line_freq"] = 60.0
        path = mne_bids.BIDSPath(
            root=root, subject=subject, session="a", task="motor", run="01", datatype="eeg"
        )
        path = mne_bids.write_raw_bids(
            raw,
            path,
            events=np.array([[100, 0, 1], [300, 0, 2], [500, 0, 1], [700, 0, 2]]),
            event_id={"left": 1, "right": 2},
            event_metadata=pd.DataFrame(
                {
                    "trial_type": ["left", "right", "left", "right"],
                    "reaction_time": [0.2, 0.3, 0.4, 0.5],
                }
            ),
            extra_columns_descriptions={
                "reaction_time": "Reaction time in seconds",
                "trial_type": "Movement direction",
            },
            format="BrainVision",
            allow_preload=True,
            verbose=False,
        )
        paths.append(path)
    participants_path = root / "participants.tsv"
    participants = pd.read_csv(participants_path, sep="\t")
    participants["age"] = [31, 44]
    participants["site"] = ["A", "B"]
    participants.to_csv(participants_path, sep="\t", index=False, na_rep="n/a")
    return root, paths


def test_bids_discovery_uses_entities_and_excludes_sidecars(dataset):
    root, paths = dataset
    result = bids.discover_bids(bids.BIDSQuery(root, subjects=("01",), tasks=("motor",)))
    assert len(result) == 1
    assert result[0].fpath == paths[0].fpath
    assert result[0].suffix == "eeg"
    assert result[0].extension == ".vhdr"


def test_bids_read_keeps_entities_participants_events_channels_and_provenance(dataset):
    root, paths = dataset
    result = bids.read_bids(paths[0])
    assert result.entities["subject"] == "01"
    assert result.entities["session"] == "a"
    assert result.entities["run"] == "01"
    assert result.participant["participant_id"] == "sub-01"
    assert result.participant["age"] == 31
    assert result.participant["site"] == "A"
    assert list(result.events["reaction_time"]) == [0.2, 0.3, 0.4, 0.5]
    assert result.event_id == {"left": 1, "right": 2}
    assert list(result.channels["name"]) == result.raw.ch_names
    assert result.provenance["reader"] == "mne_bids.read_raw_bids"
    assert result.provenance["root"] == str(root)
    assert str(root / "participants.tsv") in result.provenance["sidecar_hashes"]
    events_json = paths[0].find_matching_sidecar(suffix="events", extension=".json")
    assert str(events_json) in result.provenance["sidecar_hashes"]
    assert str(root / "sub-01/ses-a/sub-01_ses-a_scans.tsv") in result.provenance["sidecar_hashes"]


@pytest.mark.parametrize("numeric_field", ["channels", "events"])
def test_bids_numeric_looking_labels_retain_their_text_identity(tmp_path, numeric_field):
    channels = ["01", "02"] if numeric_field == "channels" else ["C3", "C4"]
    labels = ("01", "02") if numeric_field == "events" else ("left", "right")
    raw = mne.io.RawArray(
        np.random.default_rng(18).normal(0, 1e-6, (2, 600)),
        mne.create_info(channels, 100.0, "eeg"),
        verbose=False,
    )
    raw.info["line_freq"] = 60.0
    path = mne_bids.write_raw_bids(
        raw,
        mne_bids.BIDSPath(root=tmp_path / "bids", subject="01", task="labels", datatype="eeg"),
        events=np.array([[100, 0, 1], [300, 0, 2]]),
        event_id={labels[0]: 1, labels[1]: 2},
        format="BrainVision",
        allow_preload=True,
        verbose=False,
    )

    recording = bids.read_bids(path)

    assert recording.raw.ch_names == channels
    assert recording.channels["name"].tolist() == channels
    assert recording.events["trial_type"].tolist() == list(labels)
    result = bids.preprocess_bids(
        recording,
        ProcessingSettings(
            EventEpochSettings(EventSettings("annotations", {labels[0]: 1}), 0, 0.5)
        ),
    )
    assert result.epochs.metadata["trial_type"].tolist() == [labels[0]]


def test_bids_hierarchical_event_names_keep_original_event_metadata(dataset):
    _, paths = dataset
    events_path = paths[0].find_matching_sidecar(suffix="events", extension=".tsv")
    events = pd.read_csv(events_path, sep="\t")
    events["trial_type"] = "movement"
    events.to_csv(events_path, sep="\t", index=False)
    recording = bids.read_bids(paths[0])
    assert recording.event_id == {"movement/1": 1, "movement/2": 2}

    result = bids.preprocess_bids(
        recording,
        ProcessingSettings(
            EventEpochSettings(EventSettings("annotations", {"movement/1": 1}), 0, 0.5)
        ),
    )

    assert result.epochs.metadata["reaction_time"].tolist() == [0.2, 0.4]


def test_canonical_channels_reorder_explicitly_and_missing_channels_fail(dataset):
    _, paths = dataset
    canonical = ("P4", "P3", "C4", "C3")
    recording = bids.read_bids(paths[0], canonical_channels=canonical)
    assert recording.raw.ch_names == list(canonical)
    assert list(recording.channels["name"]) == list(canonical)
    with pytest.raises(ValueError, match="canonical_channels"):
        bids.read_bids(paths[0], canonical_channels=("C3", "C4", "P3", "Cz"))


def test_bids_preprocessing_aligns_selected_event_metadata(dataset):
    _, paths = dataset
    recording = bids.read_bids(paths[0])
    settings = ProcessingSettings(
        EventEpochSettings(EventSettings("annotations", {"left": 1}), 0.0, 0.5)
    )
    result = bids.preprocess_bids(recording, settings)
    assert len(result.epochs) == 2
    assert list(result.epochs.metadata["reaction_time"]) == [0.2, 0.4]
    assert set(result.epochs.metadata["subject_id"]) == {"sub-01"}
    assert set(result.epochs.metadata["run"]) == {"01"}
    assert set(result.epochs.metadata["participant_age"]) == {31}
    assert result.provenance["bids"]["entities"]["subject"] == "01"
    assert list(result.events["reaction_time"]) == [0.2, 0.4]


def test_bids_recipe_is_explicit_and_checkpointed_workflow_carries_metadata(dataset, tmp_path):
    root, _ = dataset
    recipe = tmp_path / "bids.yaml"
    recipe.write_text(
        f"input:\n  kind: bids\n  root: {root}\n  subjects: ['01']\n"
        "  tasks: [motor]\n  canonical_channels: [C3, C4, P3, P4]\n"
        "output: {directory: out}\n"
        "workflow: {raw_review: disabled, epoch_review: disabled}\n"
        "epochs:\n  kind: events\n  tmin: 0\n  tmax: 0.5\n"
        "  events: {source: annotations, event_id: {left: 1, right: 2}}\n",
        encoding="utf-8",
    )
    configs = load_recipe(recipe)
    assert len(configs) == 1
    config = next(iter(configs.values()))
    assert config.input.bids.root == root
    workflow = open_workflow(config)
    assert run_until(workflow, "crop-epochs").state == "completed"
    state = read_checkpoint(workflow, "crop-epochs").state
    assert len(state.epochs) == 4
    assert list(state.epochs.metadata["reaction_time"]) == [0.2, 0.3, 0.4, 0.5]
    assert state.provenance["bids"]["entities"]["subject"] == "01"


def test_bids_metadata_changes_make_checkpoint_stale(dataset, tmp_path):
    root, _ = dataset
    recipe = tmp_path / "bids.yaml"
    recipe.write_text(
        f"input: {{kind: bids, root: {root}, subjects: ['01']}}\n"
        "output: {directory: out}\nworkflow: {raw_review: disabled}\n"
        "epochs: {kind: fixed, duration: 1.0}\n",
        encoding="utf-8",
    )
    workflow = open_workflow(next(iter(load_recipe(recipe).values())))
    run_until(workflow, "load")
    participants = pd.read_csv(root / "participants.tsv", sep="\t")
    participants.loc[0, "age"] = 32
    participants.to_csv(root / "participants.tsv", sep="\t", index=False)
    with pytest.raises(ValueError, match="stale checkpoint"):
        run_until(workflow, "load")


def test_bids_channel_sidecar_mismatch_is_an_error(dataset):
    _, paths = dataset
    channels_path = paths[0].find_matching_sidecar(suffix="channels", extension=".tsv")
    channels = pd.read_csv(channels_path, sep="\t")
    channels.loc[0, "name"] = "unknown"
    channels.to_csv(channels_path, sep="\t", index=False)
    with pytest.raises(RuntimeError, match="channel"):
        bids.read_bids(paths[0])


def test_cli_check_validates_bids_sidecar_sample_alignment(dataset, tmp_path, capsys):
    from eegtable.runner.cli import main

    root, paths = dataset
    events_path = paths[0].find_matching_sidecar(suffix="events", extension=".tsv")
    events = pd.read_csv(events_path, sep="\t")
    events.loc[0, "sample"] = 101
    events.to_csv(events_path, sep="\t", index=False)
    recipe = tmp_path / "bids.yaml"
    recipe.write_text(
        f"input: {{kind: bids, root: {root}, subjects: ['01']}}\n"
        "output: {directory: out}\n"
        "epochs:\n  kind: events\n  tmin: 0\n  tmax: 0.5\n"
        "  events: {source: annotations, event_id: {left: 1, right: 2}}\n",
        encoding="utf-8",
    )
    assert main(["preprocess", "check", str(recipe)]) == 1
    assert "sample and onset disagree" in capsys.readouterr().out


def test_fixed_epochs_carry_descriptors_without_inventing_event_matches(dataset):
    from eegtable.preprocessing.config import FixedEpochSettings

    _, paths = dataset
    result = bids.preprocess_bids(
        bids.read_bids(paths[0]), ProcessingSettings(FixedEpochSettings(duration=1.0))
    )
    assert set(result.epochs.metadata["subject_id"]) == {"sub-01"}
    assert set(result.epochs.metadata["participant_age"]) == {31}
    assert "reaction_time" not in result.epochs.metadata


def test_missing_event_sidecar_is_supported_for_fixed_epochs(dataset):
    from eegtable.preprocessing.config import FixedEpochSettings

    _, paths = dataset
    events_path = paths[0].find_matching_sidecar(suffix="events", extension=".tsv")
    events_path.unlink()
    recording = bids.read_bids(paths[0])
    assert recording.events is None
    assert recording.event_id is None
    result = bids.preprocess_bids(recording, ProcessingSettings(FixedEpochSettings(duration=1.0)))
    assert len(result.epochs) == 10
    label = next(name for name in recording.raw.annotations.description if "Stimulus" in name)
    settings = ProcessingSettings(
        EventEpochSettings(EventSettings("annotations", {label: 1}), 0.0, 0.5)
    )
    with pytest.raises(ValueError, match="events.tsv sidecar"):
        bids.preprocess_bids(recording, settings)


def test_participant_duplicates_fail_before_reading(dataset):
    root, paths = dataset
    participants = pd.read_csv(root / "participants.tsv", sep="\t")
    participants = pd.concat([participants, participants.iloc[:1]])
    participants.to_csv(root / "participants.tsv", sep="\t", index=False, na_rep="n/a")
    with pytest.raises(ValueError, match="participant_id.*unique"):
        bids.read_bids(paths[0])


def test_simultaneous_unselected_bids_event_does_not_replace_selected_metadata(dataset):
    _, paths = dataset
    events_path = paths[0].find_matching_sidecar(suffix="events", extension=".tsv")
    events = pd.read_csv(events_path, sep="\t")
    cue = events.iloc[:1].copy()
    cue["trial_type"] = "cue"
    cue["value"] = 3
    cue["reaction_time"] = 9.0
    pd.concat([events, cue]).sort_values("onset").to_csv(events_path, sep="\t", index=False)
    result = bids.preprocess_bids(
        bids.read_bids(paths[0]),
        ProcessingSettings(EventEpochSettings(EventSettings("annotations", {"left": 1}), 0, 0.5)),
    )
    assert list(result.epochs.metadata["reaction_time"]) == [0.2, 0.4]


def test_simultaneous_hierarchical_bids_events_keep_the_selected_value(dataset):
    _, paths = dataset
    events_path = paths[0].find_matching_sidecar(suffix="events", extension=".tsv")
    events = pd.read_csv(events_path, sep="\t")
    events["trial_type"] = "movement"
    other = events.iloc[:1].copy()
    other["value"] = 2
    other["reaction_time"] = 9.0
    pd.concat([events, other]).sort_values("onset").to_csv(events_path, sep="\t", index=False)

    result = bids.preprocess_bids(
        bids.read_bids(paths[0]),
        ProcessingSettings(
            EventEpochSettings(EventSettings("annotations", {"movement/1": 1}), 0, 0.5)
        ),
    )

    assert result.epochs.metadata["reaction_time"].tolist() == [0.2, 0.4]


@pytest.mark.parametrize(
    "source",
    [
        "input: {root: bids, subjects: ['01'], pattern: '**/*.vhdr'}",
        "input: {kind: bids, root: bids, pattern: '**/*.vhdr'}",
        "input: {kind: bids, root: bids, subjects: [1]}",
        "input: {kind: bids, root: bids, subjects: [sub-01]}",
    ],
)
def test_bids_recipe_requires_explicit_mode_and_valid_entity_labels(dataset, tmp_path, source):
    recipe = tmp_path / "invalid.yaml"
    recipe.write_text(
        source + "\noutput: {directory: out}\nepochs: {kind: fixed, duration: 1.0}\n",
        encoding="utf-8",
    )
    with pytest.raises((TypeError, ValueError)):
        load_recipe(recipe)
