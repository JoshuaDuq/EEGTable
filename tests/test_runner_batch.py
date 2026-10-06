"""Running a recipe over a folder of recordings."""

import hashlib
import json
import os
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eegtable.io import read_dataset, read_table
from eegtable.runner import RunError, check, load_recipe, run, status
from eegtable.runner.progress import JsonReporter
from tests.synthetic import make_epochs, save_epochs

POWER = '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\nspatial = ["global"]\n'
ITPC = (
    '[trials]\nby = "event"\n\n'
    '[[features]]\nmeasure = "itpc"\nbands = ["alpha"]\nspatial = ["global"]\n'
)


@pytest.mark.parametrize("exclude_bads", [True, False])
def test_explicit_channel_names_respect_recipe_bad_channel_exclusion(tmp_path, exclude_bads):
    from eegtable.runner.batch import load_epochs
    from tests.synthetic import make_epochs

    epochs = make_epochs()
    epochs.info["bads"] = ["Fz"]
    source = tmp_path / "test_epo.fif"
    epochs.save(source, overwrite=True, verbose=False)
    path = tmp_path / "recipe.toml"
    path.write_text(
        '[inputs]\nroot = "."\npicks = ["Fz", "Cz"]\n'
        f"exclude_bads = {str(exclude_bads).lower()}\n"
        '[output]\nroot = "out"\n' + POWER,
        encoding="utf-8",
    )

    selected = load_epochs(source, load_recipe(path).inputs)

    assert selected.ch_names == (["Cz"] if exclude_bads else ["Fz", "Cz"])


def _recipe(tmp_path: Path, body: str, output: str = "out"):
    path = tmp_path / "recipe.toml"
    path.write_text(
        f'[inputs]\nroot = "data"\n\n[output]\nroot = "{output}"\n\n{body}', encoding="utf-8"
    )
    return load_recipe(path)


def _two_recordings(tmp_path: Path) -> None:
    save_epochs(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif")
    save_epochs(tmp_path / "data/sub-02/eeg/sub-02_task-rest_epo.fif")


def _features_path(tmp_path: Path, subject: str) -> Path:
    return tmp_path / f"out/{subject}/eeg/{subject}_task-rest_features.tsv"


@pytest.mark.parametrize("naming", ["bids", "neuromag"])
def test_split_epoch_parts_are_one_recording(tmp_path, naming) -> None:
    from dataclasses import replace

    from eegtable.runner.batch import discover, load_epochs

    root = tmp_path / "data"
    root.mkdir()
    epochs = make_epochs(n_epochs=100)
    epochs.save(root / "test_epo.fif", split_size="2MB", split_naming=naming, fmt="double")
    parts = sorted(root.glob("*.fif"))
    assert len(parts) > 1
    recipe = _recipe(tmp_path, POWER)
    recipe = replace(recipe, inputs=replace(recipe.inputs, pattern="**/*.fif"))

    recordings = discover(recipe)

    assert len(recordings) == 1
    assert len(load_epochs(recordings[0].source, recipe.inputs)) == len(epochs)


def test_a_standalone_split_named_file_is_a_recording(tmp_path) -> None:
    from eegtable.runner.batch import discover

    source = tmp_path / "data/test_split-02_epo.fif"
    save_epochs(source)
    assert [item.source for item in discover(_recipe(tmp_path, POWER))] == [source]


# --- outputs ------------------------------------------------------------------


def test_each_recording_gets_its_own_table_in_a_mirrored_tree(tmp_path) -> None:
    _two_recordings(tmp_path)

    result = run(_recipe(tmp_path, POWER))

    assert result.ok
    for subject in ("sub-01", "sub-02"):
        assert read_table(_features_path(tmp_path, subject)).n_rows == 12


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("sklearn") is None,
    reason="scikit-learn is not installed in this environment",
)
def test_runner_outputs_feed_group_disjoint_modeling_without_manual_reassembly(tmp_path) -> None:
    import eegtable.model as efm

    for subject in ("sub-01", "sub-02", "sub-03"):
        save_epochs(tmp_path / f"data/{subject}/eeg/{subject}_task-rest_epo.fif")

    result = run(_recipe(tmp_path, POWER))
    assert result.ok
    dataset = read_dataset(
        [_features_path(tmp_path, subject) for subject in ("sub-01", "sub-02", "sub-03")]
    )
    design = efm.build_design(
        dataset.table,
        dataset.targets,
        target="rating",
        groups="recording",
    )
    folds = efm.loso_folds(design.groups)
    predictions = efm.cross_fit_regression(
        folds,
        design.X,
        design.y,
        design.groups,
        efm.ridge_pipeline(efm.PreprocessingConfig(), seed=42),
        efm.ridge_grid(design.X),
        inner=efm.InnerSplit(grouping="subject", n_splits=2),
        seed=42,
    )

    tested_rows = np.concatenate([prediction.rows for prediction in predictions])
    np.testing.assert_array_equal(np.sort(tested_rows), np.arange(dataset.table.n_rows))
    assert len(predictions) == 3


def test_feature_rows_carry_each_epochs_event_and_metadata(tmp_path) -> None:
    _two_recordings(tmp_path)

    run(_recipe(tmp_path, POWER))

    frame = pd.read_csv(_features_path(tmp_path, "sub-01"), sep="\t")
    assert list(frame.columns[:4]) == ["epoch", "selection", "event", "rating"]
    assert frame["event"].tolist()[:4] == ["left", "right", "left", "right"]


def test_epoch_metadata_can_be_left_out(tmp_path) -> None:
    _two_recordings(tmp_path)
    recipe = _recipe(tmp_path, POWER)
    path = recipe.path
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'root = "out"', 'root = "out"\nepoch_metadata = false'
        ),
        encoding="utf-8",
    )

    run(load_recipe(path))

    frame = pd.read_csv(_features_path(tmp_path, "sub-01"), sep="\t")
    assert "rating" not in frame.columns


def test_cross_trial_measures_are_written_to_their_own_table(tmp_path) -> None:
    _two_recordings(tmp_path)

    run(_recipe(tmp_path, ITPC))

    table = read_table(tmp_path / "out/sub-01/eeg/sub-01_task-rest_crosstrial.tsv")
    assert table.row_labels == ("left", "right")
    assert not _features_path(tmp_path, "sub-01").exists()


def test_sidecar_records_where_the_features_came_from(tmp_path) -> None:
    _two_recordings(tmp_path)
    recipe = _recipe(tmp_path, POWER)

    run(recipe)

    sidecar = json.loads(
        _features_path(tmp_path, "sub-01").with_suffix(".json").read_text(encoding="utf-8")
    )
    provenance = sidecar["provenance"]
    assert provenance["input"] == str(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif")
    assert provenance["recipe_sha256"] == hashlib.sha256(recipe.text.encode()).hexdigest()
    assert provenance["channels"] == ["Fz", "F3", "F4", "Cz", "Pz"]


def test_channels_marked_bad_are_left_out_by_default(tmp_path) -> None:
    save_epochs(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif", bads=["Fz"])

    run(_recipe(tmp_path, POWER))

    sidecar = json.loads(
        _features_path(tmp_path, "sub-01").with_suffix(".json").read_text(encoding="utf-8")
    )
    assert sidecar["provenance"]["channels"] == ["F3", "F4", "Cz", "Pz"]


def test_run_log_records_every_recording(tmp_path) -> None:
    _two_recordings(tmp_path)
    recipe = _recipe(tmp_path, POWER)

    run(recipe)

    log = json.loads((tmp_path / "out/eegtable_run.json").read_text(encoding="utf-8"))
    assert log["recipe_text"] == recipe.text
    assert [(r["label"], r["success"]) for r in log["recordings"]] == [
        ("sub-01_task-rest", True),
        ("sub-02_task-rest", True),
    ]


# --- failures -----------------------------------------------------------------


def test_a_failing_recording_does_not_stop_the_others(tmp_path) -> None:
    save_epochs(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif")
    save_epochs(
        tmp_path / "data/sub-02/eeg/sub-02_task-rest_epo.fif", channels=["Fz", "F4", "Cz", "Pz"]
    )
    recipe = _recipe(
        tmp_path,
        '[rois]\nfront = ["Fz", "F3"]\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\nspatial = ["rois"]\n',
    )

    result = run(recipe)

    first, second = result.recordings
    assert first.success and not second.success
    assert second.error is not None and "F3" in second.error
    assert not result.ok
    assert _features_path(tmp_path, "sub-01").exists()
    assert not _features_path(tmp_path, "sub-02").exists()
    log = json.loads((tmp_path / "out/eegtable_run.json").read_text(encoding="utf-8"))
    assert "F3" in log["recordings"][1]["error"]


def test_existing_results_are_not_overwritten_by_default(tmp_path) -> None:
    _two_recordings(tmp_path)
    run(_recipe(tmp_path, POWER))

    with pytest.raises(RunError, match="overwrite"):
        run(_recipe(tmp_path, POWER))


def test_overwrite_replaces_results_and_clears_stale_ones(tmp_path) -> None:
    _two_recordings(tmp_path)
    run(_recipe(tmp_path, POWER + "\n" + ITPC))
    stale = tmp_path / "out/sub-01/eeg/sub-01_task-rest_crosstrial.tsv"
    assert stale.exists()

    result = run(_recipe(tmp_path, POWER), overwrite=True)

    assert result.ok
    assert _features_path(tmp_path, "sub-01").exists()
    assert not stale.exists()
    assert not stale.with_suffix(".json").exists()


def test_failed_overwrite_preserves_the_previous_complete_result(tmp_path, monkeypatch) -> None:
    save_epochs(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif")
    recipe = _recipe(tmp_path, POWER)
    first = run(recipe)
    assert first.ok
    values = _features_path(tmp_path, "sub-01")
    existing = {
        path: path.read_bytes()
        for path in (
            values,
            values.with_suffix(".json"),
            values.with_name(f"{values.stem}_coverage.tsv"),
        )
    }

    import eegtable.runner.batch as batch

    def fail(*args, **kwargs):
        raise RuntimeError("intentional replacement failure")

    monkeypatch.setattr(batch, "compute_features", fail)
    result = run(recipe, overwrite=True)

    assert not result.ok
    assert all(path.read_bytes() == content for path, content in existing.items())


def test_hidden_files_are_not_recordings(tmp_path) -> None:
    # macOS writes "._" AppleDouble companions beside files on external drives.
    save_epochs(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif")
    (tmp_path / "data/sub-01/eeg/._sub-01_task-rest_epo.fif").write_bytes(b"\0" * 4096)
    (tmp_path / "data/.cache").mkdir()
    save_epochs(tmp_path / "data/.cache/sub-09_task-rest_epo.fif")

    result = run(_recipe(tmp_path, POWER))

    assert [r.label for r in result.recordings] == ["sub-01_task-rest"]


def test_no_matching_file_is_an_error(tmp_path) -> None:
    (tmp_path / "data").mkdir()

    with pytest.raises(RunError, match="no files match"):
        run(_recipe(tmp_path, POWER))


def test_missing_input_root_is_an_error(tmp_path) -> None:
    with pytest.raises(RunError, match="data"):
        run(_recipe(tmp_path, POWER))


def test_inputs_that_would_share_outputs_are_rejected(tmp_path) -> None:
    save_epochs(tmp_path / "data/sub-01_epo.fif")
    save_epochs(tmp_path / "data/sub-01-epo.fif")
    recipe = tmp_path / "recipe.toml"
    recipe.write_text(
        '[inputs]\nroot = "data"\npattern = "*epo.fif"\n\n[output]\nroot = "out"\n\n' + POWER,
        encoding="utf-8",
    )

    with pytest.raises(RunError, match="sub-01"):
        run(load_recipe(recipe))


# --- progress -----------------------------------------------------------------


def test_progress_events_follow_the_tui_protocol(tmp_path) -> None:
    _two_recordings(tmp_path)
    stream = StringIO()

    run(_recipe(tmp_path, POWER), reporter=JsonReporter(stream))

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    one, two = "sub-01_task-rest", "sub-02_task-rest"
    assert [(e["event"], e.get("subject"), e.get("step")) for e in events] == [
        ("start", None, None),
        ("subject_start", one, None),
        ("progress", one, "read"),
        ("progress", one, "integrated_band_power"),
        ("progress", one, "write"),
        ("log", one, None),
        ("subject_done", one, None),
        ("subject_start", two, None),
        ("progress", two, "read"),
        ("progress", two, "integrated_band_power"),
        ("progress", two, "write"),
        ("log", two, None),
        ("subject_done", two, None),
        ("log", None, None),
        ("complete", None, None),
    ]
    assert events[0]["subjects"] == [one, two] and events[0]["total_subjects"] == 2
    steps = [(e["current"], e["total"]) for e in events if e["event"] == "progress"][:3]
    assert steps == [(1, 3), (2, 3), (3, 3)]
    assert [e["success"] for e in events if e["event"] == "subject_done"] == [True, True]
    assert events[-1]["success"] is True


def test_a_failed_recording_is_reported_as_such(tmp_path) -> None:
    save_epochs(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif", channels=["Fz", "Cz"])
    stream = StringIO()
    recipe = _recipe(
        tmp_path,
        '[rois]\nfront = ["Fz", "F3"]\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["rois"]\n',
    )

    run(recipe, reporter=JsonReporter(stream))

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    done = next(e for e in events if e["event"] == "subject_done")
    error = next(e for e in events if e["event"] == "log" and e.get("level") == "error")
    assert done["success"] is False
    assert "F3" in error["message"]
    assert events[-1]["success"] is False


# --- check --------------------------------------------------------------------


def test_check_trials_the_first_recording_without_writing(tmp_path) -> None:
    _two_recordings(tmp_path)

    report = check(_recipe(tmp_path, POWER + "\n" + ITPC))

    assert [r.label for r in report.recordings] == ["sub-01_task-rest", "sub-02_task-rest"]
    assert report.trial is not None
    assert report.trial.recording.label == "sub-01_task-rest"
    assert report.trial.features.epochs is not None
    assert report.trial.features.epochs.n_rows == 12
    assert report.trial.features.crosstrial is not None
    assert not (tmp_path / "out").exists()


def test_check_warns_that_global_means_split_when_recordings_keep_different_channels(
    tmp_path,
) -> None:
    # The global mean is identified by its members: without Pz, sub-02 measures a different
    # feature, and the cohort table would hold two half-empty "global" columns.
    save_epochs(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif")
    save_epochs(tmp_path / "data/sub-02/eeg/sub-02_task-rest_epo.fif", bads=["Pz"])

    report = check(_recipe(tmp_path, POWER))

    assert any("sub-02_task-rest" in w and "Pz" in w for w in report.warnings)


def test_check_does_not_warn_about_channels_every_recording_keeps(tmp_path) -> None:
    _two_recordings(tmp_path)
    report = check(_recipe(tmp_path, POWER))
    assert not any("channels" in w for w in report.warnings)


def test_check_warns_when_a_welch_window_holds_a_single_segment(tmp_path) -> None:
    # The 1 s window caps n_fft at its own 251 samples, so "Welch" averages nothing.
    _two_recordings(tmp_path)
    report = check(_recipe(tmp_path, "[windows]\nstim = [0.0, 1.0]\n\n" + POWER))
    assert any("stim" in w and "one Welch segment" in w for w in report.warnings)


def test_check_does_not_warn_when_welch_segments_overlap(tmp_path) -> None:
    _two_recordings(tmp_path)
    body = "[spectra]\nn_fft = 125\n\n[windows]\nstim = [0.0, 1.0]\n\n" + POWER
    report = check(_recipe(tmp_path, body))
    assert not any("Welch" in w for w in report.warnings)


MORLET = (
    '[spectra]\nmethod = "morlet"\nfmin = 4.0\nn_freqs = 10\n\n[windows]\nstim = [0.0, 1.0]\n\n'
    '[[features]]\nmeasure = "mean_tfr_power"\nbands = ["alpha"]\nspatial = ["global"]\n'
)


def test_a_morlet_run_publishes_its_support_beside_the_values(tmp_path) -> None:
    # Morlet power rests on the part of each window its wavelets fit, so the bundle carries
    # that fraction, and status still reads the recording as done.
    _two_recordings(tmp_path)
    recipe = _recipe(tmp_path, MORLET)

    assert run(recipe).ok

    path = _features_path(tmp_path, "sub-01")
    assert path.with_name(path.stem + "_support.tsv").is_file()
    assert read_table(path).support is not None
    assert {entry.state for entry in status(recipe)} == {"done"}


def test_check_lists_results_already_on_disk(tmp_path) -> None:
    _two_recordings(tmp_path)
    run(_recipe(tmp_path, POWER))

    report = check(_recipe(tmp_path, POWER))

    assert _features_path(tmp_path, "sub-01") in report.existing


# --- status -------------------------------------------------------------------


def _states(recipe) -> dict[str, str]:
    return {entry.label: entry.state for entry in status(recipe)}


def test_status_before_any_run_is_missing_for_every_recording(tmp_path) -> None:
    _two_recordings(tmp_path)

    assert _states(_recipe(tmp_path, POWER)) == {
        "sub-01_task-rest": "missing",
        "sub-02_task-rest": "missing",
    }


def test_status_after_a_run_is_done_and_lists_the_results(tmp_path) -> None:
    _two_recordings(tmp_path)
    recipe = _recipe(tmp_path, POWER)
    run(recipe)

    entries = status(recipe)

    assert [entry.state for entry in entries] == ["done", "done"]
    assert _features_path(tmp_path, "sub-01") in entries[0].outputs


def test_status_names_the_error_of_a_recording_that_failed(tmp_path) -> None:
    save_epochs(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif")
    save_epochs(
        tmp_path / "data/sub-02/eeg/sub-02_task-rest_epo.fif", channels=["Fz", "F4", "Cz", "Pz"]
    )
    recipe = _recipe(
        tmp_path,
        '[rois]\nfront = ["Fz", "F3"]\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\nspatial = ["rois"]\n',
    )
    run(recipe)

    first, second = status(recipe)

    assert first.state == "done"
    assert second.state == "failed" and "F3" in second.reason


def test_a_changed_computation_makes_results_stale(tmp_path) -> None:
    _two_recordings(tmp_path)
    run(_recipe(tmp_path, POWER))

    entries = status(_recipe(tmp_path, POWER.replace("alpha", "beta")))

    assert [entry.state for entry in entries] == ["stale", "stale"]
    assert "recipe" in entries[0].reason


def test_comments_and_repointed_inputs_do_not_make_results_stale(tmp_path) -> None:
    # Moving the data to another drive and repointing inputs.root changes the recipe's
    # text, not what it computes.
    _two_recordings(tmp_path)
    run(_recipe(tmp_path, POWER))
    (tmp_path / "data").rename(tmp_path / "moved")
    path = tmp_path / "recipe.toml"
    path.write_text(
        '# repointed after the move\n[inputs]\nroot = "moved"\npattern = "**/*_epo.fif"\n\n'
        '[output]\nroot = "out"\n\n' + POWER,
        encoding="utf-8",
    )

    assert set(_states(load_recipe(path)).values()) == {"done"}


def test_an_input_rewritten_after_its_results_makes_them_stale(tmp_path) -> None:
    _two_recordings(tmp_path)
    recipe = _recipe(tmp_path, POWER)
    run(recipe)
    source = tmp_path / "data/sub-02/eeg/sub-02_task-rest_epo.fif"
    later = _features_path(tmp_path, "sub-02").stat().st_mtime + 60
    source.write_bytes(source.read_bytes() + b"changed")
    os.utime(source, (later, later))

    first, second = status(recipe)

    assert first.state == "done"
    assert second.state == "stale" and "input" in second.reason


def test_a_missing_file_of_a_table_makes_results_partial(tmp_path) -> None:
    _two_recordings(tmp_path)
    recipe = _recipe(tmp_path, POWER)
    run(recipe)
    values = _features_path(tmp_path, "sub-01")
    values.with_name(f"{values.stem}_coverage.tsv").unlink()

    first, second = status(recipe)

    assert first.state == "partial" and "coverage" in first.reason
    assert second.state == "done"


def test_a_missing_table_makes_results_partial(tmp_path) -> None:
    _two_recordings(tmp_path)
    recipe = _recipe(tmp_path, POWER + "\n" + ITPC)
    run(recipe)
    crosstrial = tmp_path / "out/sub-01/eeg/sub-01_task-rest_crosstrial.tsv"
    for path in (
        crosstrial,
        crosstrial.with_suffix(".json"),
        crosstrial.with_name(f"{crosstrial.stem}_coverage.tsv"),
    ):
        path.unlink()

    first, _ = status(recipe)

    assert first.state == "partial" and "crosstrial" in first.reason


# --- resume -------------------------------------------------------------------


def test_resume_computes_only_the_recordings_without_results(tmp_path) -> None:
    save_epochs(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif")
    recipe = _recipe(tmp_path, POWER)
    run(recipe)
    before = _features_path(tmp_path, "sub-01").read_bytes()
    save_epochs(tmp_path / "data/sub-02/eeg/sub-02_task-rest_epo.fif")

    result = run(recipe, resume=True)

    assert result.ok
    assert [r.label for r in result.recordings] == ["sub-02_task-rest"]
    assert _features_path(tmp_path, "sub-01").read_bytes() == before
    assert set(_states(recipe).values()) == {"done"}
    log = json.loads((tmp_path / "out/eegtable_run.json").read_text(encoding="utf-8"))
    assert log["skipped"] == ["sub-01_task-rest"]


def test_resume_retries_a_recording_that_failed(tmp_path) -> None:
    save_epochs(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif")
    broken = tmp_path / "data/sub-02/eeg/sub-02_task-rest_epo.fif"
    save_epochs(broken, channels=["Fz", "F4", "Cz", "Pz"])
    recipe = _recipe(
        tmp_path,
        '[rois]\nfront = ["Fz", "F3"]\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\nspatial = ["rois"]\n',
    )
    run(recipe)
    save_epochs(broken)

    result = run(recipe, resume=True)

    assert [(r.label, r.success) for r in result.recordings] == [("sub-02_task-rest", True)]


def test_resume_will_not_replace_stale_results_without_overwrite(tmp_path) -> None:
    _two_recordings(tmp_path)
    run(_recipe(tmp_path, POWER))
    changed = _recipe(tmp_path, POWER.replace("alpha", "beta"))

    with pytest.raises(RunError, match="overwrite"):
        run(changed, resume=True)

    result = run(changed, resume=True, overwrite=True)
    assert result.ok and len(result.recordings) == 2
    assert set(_states(changed).values()) == {"done"}


def test_resume_with_everything_done_computes_nothing(tmp_path) -> None:
    _two_recordings(tmp_path)
    recipe = _recipe(tmp_path, POWER)
    run(recipe)
    before = _features_path(tmp_path, "sub-01").stat().st_mtime_ns

    result = run(recipe, resume=True)

    assert result.ok and result.recordings == ()
    assert _features_path(tmp_path, "sub-01").stat().st_mtime_ns == before


# --- parallel workers -----------------------------------------------------------


def _three_recordings(tmp_path: Path) -> None:
    for subject in ("sub-01", "sub-02", "sub-03"):
        save_epochs(tmp_path / f"data/{subject}/eeg/{subject}_task-rest_epo.fif")


def test_workers_compute_the_same_tables_as_one_process(tmp_path) -> None:
    _three_recordings(tmp_path)
    serial = run(_recipe(tmp_path, POWER + "\n" + ITPC, output="serial"))
    parallel = run(_recipe(tmp_path, POWER + "\n" + ITPC, output="parallel"), workers=2)

    assert serial.ok and parallel.ok
    assert [r.label for r in parallel.recordings] == [r.label for r in serial.recordings]
    for subject in ("sub-01", "sub-02", "sub-03"):
        for table in ("features", "crosstrial"):
            name = f"{subject}/eeg/{subject}_task-rest_{table}.tsv"
            np.testing.assert_array_equal(
                read_table(tmp_path / "parallel" / name).values,
                read_table(tmp_path / "serial" / name).values,
            )
    log = json.loads((tmp_path / "parallel/eegtable_run.json").read_text(encoding="utf-8"))
    assert [r["label"] for r in log["recordings"]] == [r.label for r in serial.recordings]


def test_workers_forward_every_recordings_progress(tmp_path) -> None:
    _three_recordings(tmp_path)
    stream = StringIO()

    run(_recipe(tmp_path, POWER), workers=2, reporter=JsonReporter(stream))

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    labels = {f"{s}_task-rest" for s in ("sub-01", "sub-02", "sub-03")}
    for kind in ("subject_start", "subject_done"):
        assert {e["subject"] for e in events if e["event"] == kind} == labels
    steps = {(e["subject"], e["step"]) for e in events if e["event"] == "progress"}
    assert steps == {
        (label, step) for label in labels for step in ("read", "integrated_band_power", "write")
    }
    assert events[-1]["event"] == "complete" and events[-1]["success"] is True


def test_a_failing_recording_does_not_stop_the_other_workers(tmp_path) -> None:
    save_epochs(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif")
    save_epochs(
        tmp_path / "data/sub-02/eeg/sub-02_task-rest_epo.fif", channels=["Fz", "F4", "Cz", "Pz"]
    )
    recipe = _recipe(
        tmp_path,
        '[rois]\nfront = ["Fz", "F3"]\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\nspatial = ["rois"]\n',
    )

    first, second = run(recipe, workers=2).recordings

    assert first.success and not second.success
    assert second.error is not None and "F3" in second.error


def _work_or_crash(recording, *args):
    # Stands in for a real crash (a segfault in a compiled dependency, the OOM killer). At
    # module level, so spawned workers import it by name, and with it an unpatched batch.
    from eegtable.runner.batch import _work

    with recording.source.with_suffix(".attempts").open("a", encoding="utf-8") as attempts:
        attempts.write("attempt\n")
    if recording.label.startswith("sub-02"):
        os._exit(1)
    return _work(recording, *args)


def test_a_worker_process_that_dies_does_not_rerun_submitted_recordings(tmp_path, monkeypatch):
    import eegtable.runner.batch as batch

    _three_recordings(tmp_path)
    monkeypatch.setattr(batch, "_work", _work_or_crash)

    result = run(_recipe(tmp_path, POWER), workers=2)

    outcome = {r.label: r for r in result.recordings}
    assert not outcome["sub-02_task-rest"].success
    assert "worker process" in (outcome["sub-02_task-rest"].error or "")
    assert outcome["sub-03_task-rest"].success
    attempts = list((tmp_path / "data").rglob("*.attempts"))
    assert len(attempts) == 3
    assert all(path.read_text(encoding="utf-8") == "attempt\n" for path in attempts)
    assert not result.ok
    assert outcome["sub-02_task-rest"].recording.failure_path.exists()


def test_workers_must_be_at_least_one(tmp_path) -> None:
    _two_recordings(tmp_path)

    with pytest.raises(RunError, match="workers"):
        run(_recipe(tmp_path, POWER), workers=0)


# --- what a run leaves behind as it goes ----------------------------------------


def test_the_run_log_is_written_after_every_recording(tmp_path) -> None:
    # A run killed half way must still say what it finished.
    _two_recordings(tmp_path)
    seen: list[list[str]] = []

    class Watcher(JsonReporter):
        def recording_done(self, label, success, message):
            log = json.loads((tmp_path / "out/eegtable_run.json").read_text(encoding="utf-8"))
            seen.append([entry["label"] for entry in log["recordings"]])

    run(_recipe(tmp_path, POWER), reporter=Watcher(StringIO()))

    assert seen == [["sub-01_task-rest"], ["sub-01_task-rest", "sub-02_task-rest"]]
    log = json.loads((tmp_path / "out/eegtable_run.json").read_text(encoding="utf-8"))
    assert log["finished"] is True


def _one_failing(tmp_path: Path):
    save_epochs(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif")
    save_epochs(
        tmp_path / "data/sub-02/eeg/sub-02_task-rest_epo.fif", channels=["Fz", "F4", "Cz", "Pz"]
    )
    return _recipe(
        tmp_path,
        '[rois]\nfront = ["Fz", "F3"]\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\nspatial = ["rois"]\n',
    )


def test_a_failure_is_kept_beside_the_recordings_results(tmp_path) -> None:
    # Another run into the same output root replaces the run log; the failure must
    # survive it.
    recipe = _one_failing(tmp_path)
    run(recipe)
    (tmp_path / "out/eegtable_run.json").unlink()

    failed = json.loads(
        (tmp_path / "out/sub-02/eeg/sub-02_task-rest_failed.json").read_text(encoding="utf-8")
    )

    assert "F3" in failed["error"] and failed["traceback"]
    assert {e.label: e.state for e in status(recipe)}["sub-02_task-rest"] == "failed"


def test_a_failure_in_a_run_log_written_before_the_rename_is_still_reported(tmp_path) -> None:
    recipe = _one_failing(tmp_path)
    run(recipe)
    (tmp_path / "out/sub-02/eeg/sub-02_task-rest_failed.json").unlink()
    (tmp_path / "out/eegtable_run.json").rename(tmp_path / "out/eegfeat_run.json")

    failed = {e.label: e for e in status(recipe)}["sub-02_task-rest"]

    assert failed.state == "failed" and "F3" in failed.reason


def test_a_later_success_clears_the_recorded_failure(tmp_path) -> None:
    recipe = _one_failing(tmp_path)
    run(recipe)
    save_epochs(tmp_path / "data/sub-02/eeg/sub-02_task-rest_epo.fif")

    run(recipe, resume=True)

    assert not (tmp_path / "out/sub-02/eeg/sub-02_task-rest_failed.json").exists()
    assert {e.label: e.state for e in status(recipe)}["sub-02_task-rest"] == "done"


# --- check timing -----------------------------------------------------------------


def test_check_times_the_trial_recording_and_each_entry(tmp_path) -> None:
    _two_recordings(tmp_path)

    report = check(_recipe(tmp_path, POWER + "\n" + ITPC))

    assert report.trial is not None
    assert report.trial.seconds > 0.0
    assert [t.measure for t in report.trial.features.timings] == ["integrated_band_power", "itpc"]
    assert report.trial.epochs_total == 12


def test_a_quick_check_computes_only_the_first_epochs(tmp_path) -> None:
    _two_recordings(tmp_path)

    report = check(_recipe(tmp_path, POWER + "\n" + ITPC), quick=True)

    assert report.trial is not None
    assert report.trial.n_epochs == 4 and report.trial.epochs_total == 12
    assert report.trial.features.epochs is not None
    assert report.trial.features.epochs.n_rows == 4


def test_workers_share_the_cores_between_their_thread_pools() -> None:
    # Each worker's BLAS and OpenMP pools would otherwise size themselves to the whole
    # machine, and N workers would oversubscribe it N times over.
    from eegtable.runner.batch import _worker_threads

    limits = _worker_threads(workers=4, cpu_count=10, environ={})

    assert limits["OMP_NUM_THREADS"] == "2"
    assert limits["OPENBLAS_NUM_THREADS"] == "2" and limits["VECLIB_MAXIMUM_THREADS"] == "2"


def test_a_thread_limit_the_user_set_is_left_alone() -> None:
    from eegtable.runner.batch import _worker_threads

    limits = _worker_threads(workers=4, cpu_count=10, environ={"OMP_NUM_THREADS": "3"})

    assert "OMP_NUM_THREADS" not in limits and limits["MKL_NUM_THREADS"] == "2"


def test_a_pool_that_breaks_before_a_submission_does_not_end_the_run(tmp_path, monkeypatch):
    # A worker can die between two polls, so the next submission is the first to see
    # the broken pool. Everything must still be computed.
    from concurrent.futures.process import BrokenProcessPool

    import eegtable.runner.batch as batch

    _three_recordings(tmp_path)
    submissions = []

    class BreaksOnce(batch.ProcessPoolExecutor):
        def submit(self, *args, **kwargs):
            submissions.append(args[1].label)
            if len(submissions) == 2:
                raise BrokenProcessPool("a worker died")
            return super().submit(*args, **kwargs)

    monkeypatch.setattr(batch, "ProcessPoolExecutor", BreaksOnce)

    result = run(_recipe(tmp_path, POWER), workers=2)

    assert [(r.label, r.success) for r in result.recordings] == [
        ("sub-01_task-rest", True),
        ("sub-02_task-rest", True),
        ("sub-03_task-rest", True),
    ]
