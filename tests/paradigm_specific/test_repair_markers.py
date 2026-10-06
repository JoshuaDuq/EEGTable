import importlib.util
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
import yaml

from eegtable.preprocessing.provenance import file_hash, identity
from eegtable.runner.batch import _upstream_provenance

SCRIPT = (
    Path(__file__).resolve().parents[2] / "paradigm_specific" / "thermal_pain" / "repair_markers.py"
)
spec = importlib.util.spec_from_file_location("repair_markers", SCRIPT)
assert spec is not None and spec.loader is not None
repair = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = repair
spec.loader.exec_module(repair)

HEAD = (
    "Brain Vision Data Exchange Marker File, Version 1.0\n\n"
    "[Common Infos]\nDataFile={data}\n\n[Marker Infos]\n"
)
NAMED = [
    "Volume,V  1,1,1,0",
    "Volume,V  1,901,1,0",
    "Trig_therm,T  1,300,1,0",
    "Bad Interval,Bad_Gradient,700,50,0",
]
GENERIC = ["Stimulus,S  2,1,1,0", "Stimulus,S  1,300,1,0", "Stimulus,S  2,901,1,0"]


def vmrk(path, data, markers):
    path.write_text(
        HEAD.format(data=data) + "\n".join(f"Mk{i}={m}" for i, m in enumerate(markers, 1)) + "\n",
        encoding="utf-8",
    )


def test_splice_keeps_the_derivatives_header_and_takes_the_named_markers(tmp_path):
    fixed, old = tmp_path / "fixed.vmrk", tmp_path / "old.vmrk"
    vmrk(fixed, "fixed.eeg", NAMED)
    vmrk(old, "sub-01_desc-bcgnet_eeg.eeg", GENERIC)
    repair.splice_markers(fixed, old)
    text = old.read_text(encoding="utf-8")
    assert "DataFile=sub-01_desc-bcgnet_eeg.eeg" in text
    assert [line.split("=", 1)[1] for line in text.splitlines() if line.startswith("Mk")] == NAMED


def test_repair_ignores_appledouble_fixed_marker_sources(tmp_path, monkeypatch):
    fixed_root, old_root = tmp_path / "fixed", tmp_path / "old"
    fixed_root.mkdir()
    old_root.mkdir()
    name = "sub-01_task-thermalactive_run-1_eeg.vmrk"
    vmrk(fixed_root / name, "fixed.eeg", NAMED)
    vmrk(old_root / name, "old.eeg", GENERIC)
    (fixed_root / f"._{name}").write_bytes(b"\x00\x05\x16\x07")
    original_rglob = Path.rglob

    def source_files(path, pattern):
        matches = original_rglob(path, pattern)
        if path == fixed_root:
            return sorted(matches, key=lambda item: item.name.startswith("._"))
        return matches

    monkeypatch.setattr(Path, "rglob", source_files)
    assert (
        repair.main(
            [
                "--fixed-bids",
                str(fixed_root),
                "--derivatives",
                str(old_root),
                "--archive",
                str(tmp_path / "archive"),
            ]
        )
        == 0
    )
    assert repair.marker_lines(old_root / name) == NAMED


def test_splice_refuses_when_the_generic_positions_do_not_match(tmp_path):
    fixed, old = tmp_path / "fixed.vmrk", tmp_path / "old.vmrk"
    vmrk(fixed, "fixed.eeg", NAMED)
    vmrk(old, "old.eeg", ["Stimulus,S  2,5,1,0", "Stimulus,S  1,300,1,0", "Stimulus,S  2,901,1,0"])
    with pytest.raises(ValueError, match="positions"):
        repair.splice_markers(fixed, old)
    assert "Stimulus,S  2,5,1,0" in old.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "markers",
    [
        ["Stimulus,S  1,1,1,0", "Stimulus,S  2,300,1,0", "Stimulus,S  2,901,1,0"],
        [*GENERIC, GENERIC[1]],
        [GENERIC[0], "Stimulus,S  1,300,50,0", GENERIC[2]],
        [GENERIC[0], "Stimulus,S  1,300,1,1", GENERIC[2]],
    ],
    ids=["swapped-codes", "duplicate-event", "changed-duration", "changed-channel"],
)
def test_splice_refuses_incompatible_events_at_matching_samples(tmp_path, markers):
    fixed, old = tmp_path / "fixed.vmrk", tmp_path / "old.vmrk"
    vmrk(fixed, "fixed.eeg", NAMED)
    vmrk(old, "old.eeg", markers)
    original = old.read_bytes()
    with pytest.raises(ValueError, match="marker"):
        repair.splice_markers(fixed, old)
    assert old.read_bytes() == original


def test_splice_accepts_a_new_segment_without_treating_it_as_a_task_event(tmp_path):
    fixed, old = tmp_path / "fixed.vmrk", tmp_path / "old.vmrk"
    segment = "New Segment,,1,1,0,20200101000000000000"
    vmrk(fixed, "fixed.eeg", [segment, *NAMED])
    vmrk(old, "old.eeg", [segment, *GENERIC])
    repair.splice_markers(fixed, old)
    assert repair.marker_lines(old) == [segment, *NAMED]


@pytest.mark.parametrize(
    "old_segments",
    [
        ["New Segment,,1,1,0,20250101000000000000"],
        ["New Segment,,2,1,0,20200101000000000000"],
        [],
        [
            "New Segment,,1,1,0,20200101000000000000",
            "New Segment,,500,1,0,20200101000005000000",
        ],
    ],
    ids=["changed-date", "changed-position", "missing-segment", "extra-discontinuity"],
)
def test_splice_preserves_recording_dates_and_discontinuities(tmp_path, old_segments):
    fixed, old = tmp_path / "fixed.vmrk", tmp_path / "old.vmrk"
    segment = "New Segment,,1,1,0,20200101000000000000"
    vmrk(fixed, "fixed.eeg", [segment, *NAMED])
    vmrk(old, "old.eeg", [*old_segments, *GENERIC])
    original = old.read_bytes()
    with pytest.raises(ValueError, match="New Segment"):
        repair.splice_markers(fixed, old)
    assert old.read_bytes() == original


def bundle(tmp_path):
    stem = tmp_path / "sub-01_task-thermalactive_run-1"
    info = mne.create_info(["Cz"], sfreq=100.0, ch_types=["eeg"])
    events = np.array([[50, 0, 1], [150, 0, 1]])
    epochs = mne.EpochsArray(
        np.zeros((2, 1, 10)), info, events=events, event_id={"Stimulus/S  1": 1}
    )
    provenance = {"events": {"event_id": {"Stimulus/S  1": 1}}}
    epochs.info["description"] = (
        f"preprocessing={stem.name}_preprocessing.json; identity={identity(provenance)}"
    )
    epochs.save(f"{stem}_epo.fif", verbose=False)
    pd.DataFrame({"label": ["Stimulus/S  1", "Stimulus/S  1"], "retained": [True, False]}).to_csv(
        f"{stem}_events.tsv", sep="\t", index=False
    )
    Path(f"{stem}_recipe.yaml").write_text(
        'epochs:\n  events: {source: annotations, event_id: {"Stimulus/S  1": 1}}\n',
        encoding="utf-8",
    )
    manifest = {
        "schema": 1,
        "files": {
            f"{stem.name}_epo.fif": file_hash(Path(f"{stem}_epo.fif")),
            f"{stem.name}_events.tsv": file_hash(Path(f"{stem}_events.tsv")),
        },
        "provenance": provenance,
    }
    Path(f"{stem}_preprocessing.json").write_text(json.dumps(manifest), encoding="utf-8")
    return stem


def test_bundle_repair_renames_everywhere_and_rehashes(tmp_path):
    stem = bundle(tmp_path)
    repair.repair_bundle(Path(f"{stem}_preprocessing.json"))
    assert mne.read_epochs(f"{stem}_epo.fif", verbose=False).event_id == {"Trig_therm/T  1": 1}
    assert pd.read_csv(f"{stem}_events.tsv", sep="\t")["label"].tolist() == ["Trig_therm/T  1"] * 2
    recipe = yaml.safe_load(Path(f"{stem}_recipe.yaml").read_text(encoding="utf-8"))
    assert recipe["epochs"]["events"]["event_id"] == {"Trig_therm/T  1": 1}
    manifest = json.loads(Path(f"{stem}_preprocessing.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["events"]["event_id"] == {"Trig_therm/T  1": 1}
    for name, digest in manifest["files"].items():
        assert file_hash(tmp_path / name) == digest


# The 2026-09-22 repair left the old identity in the FIF, so eegtable run refused every bundle.
def test_bundle_repair_restamps_the_epochs_identity(tmp_path):
    stem = bundle(tmp_path)
    manifest_path = Path(f"{stem}_preprocessing.json")
    repair.repair_bundle(manifest_path)
    epochs = mne.read_epochs(f"{stem}_epo.fif", verbose=False)
    provenance = _upstream_provenance(Path(f"{stem}_epo.fif"), epochs)
    assert provenance is not None
    assert provenance["identity"] == identity(
        json.loads(manifest_path.read_text(encoding="utf-8"))["provenance"]
    )


def test_bundle_repair_preserves_other_trigger_codes(tmp_path):
    stem = bundle(tmp_path)
    epochs_file = Path(f"{stem}_epo.fif")
    epochs = mne.read_epochs(epochs_file, preload=True, verbose=False)
    epochs.events[1, 2] = 10
    epochs.event_id["Stimulus/S  10"] = 10
    epochs.save(epochs_file, overwrite=True, verbose=False)
    ledger = Path(f"{stem}_events.tsv")
    frame = pd.read_csv(ledger, sep="\t")
    frame.loc[1, "label"] = "Stimulus/S  10"
    frame.to_csv(ledger, sep="\t", index=False)
    recipe = Path(f"{stem}_recipe.yaml")
    recipe.write_text(
        "epochs:\n  events: {source: annotations, "
        'event_id: {"Stimulus/S  1": 1, "Stimulus/S  10": 10}}\n',
        encoding="utf-8",
    )
    manifest_path = Path(f"{stem}_preprocessing.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"]["events"]["event_id"]["Stimulus/S  10"] = 10
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    repair.repair_bundle(manifest_path)

    assert mne.read_epochs(epochs_file, verbose=False).event_id == {
        "Trig_therm/T  1": 1,
        "Stimulus/S  10": 10,
    }
    assert pd.read_csv(ledger, sep="\t")["label"].tolist() == [
        "Trig_therm/T  1",
        "Stimulus/S  10",
    ]
    assert yaml.safe_load(recipe.read_text(encoding="utf-8"))["epochs"]["events"]["event_id"] == {
        "Trig_therm/T  1": 1,
        "Stimulus/S  10": 10,
    }
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["provenance"]["events"][
        "event_id"
    ] == {
        "Trig_therm/T  1": 1,
        "Stimulus/S  10": 10,
    }


@pytest.mark.parametrize(
    "description",
    ["Stimulus/S  10", "Stimulus/S  20", "Stimulus/S  1_other", "Comment/Stimulus/S  1"],
)
def test_rename_changes_only_the_two_complete_marker_descriptions(description):
    assert repair.rename(description) == description


def test_bundle_repair_refuses_colliding_event_labels_before_writing(tmp_path):
    stem = bundle(tmp_path)
    manifest_path = Path(f"{stem}_preprocessing.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"]["events"]["event_id"]["Trig_therm/T  1"] = 10
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    originals = {path: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="collid"):
        repair.repair_bundle(manifest_path)
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == originals


def test_bundle_repair_preserves_samples_and_inactive_projectors(tmp_path):
    stem = bundle(tmp_path)
    epochs_file = Path(f"{stem}_epo.fif")
    info = mne.create_info(["Cz", "Pz"], sfreq=100.0, ch_types="eeg")
    samples = np.broadcast_to(np.array([[1e-6], [3e-6]]), (2, 2, 10)).copy()
    epochs = mne.EpochsArray(
        samples,
        info,
        events=np.array([[50, 0, 1], [150, 0, 1]]),
        event_id={"Stimulus/S  1": 1},
    )
    epochs.set_eeg_reference(projection=True)
    epochs.save(epochs_file, overwrite=True, fmt="double", verbose=False)

    repair.repair_bundle(Path(f"{stem}_preprocessing.json"))

    repaired = mne.read_epochs(epochs_file, preload=True, proj=False, verbose=False)
    np.testing.assert_array_equal(repaired.get_data(), samples)
    assert repaired.info["projs"][0]["active"] is False


def test_only_live_bundles_are_repaired(tmp_path):
    for folder in ("sub-01/eeg", "_superseded_v1/sub-01/eeg"):
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "sub-01_task-thermalactive_run-1_preprocessing.json").write_text(
            '{"provenance": {"event_id": {"Stimulus/S  1": 1}}}', encoding="utf-8"
        )
    (tmp_path / "sub-01/eeg/._sub-01_task-thermalactive_run-1_preprocessing.json").write_bytes(
        b"\x00\x05\x16\x07"
    )
    assert [p.parent.parent.name for p in repair.live_bundles(tmp_path)] == ["sub-01"]
