import io
import json
import webbrowser

import numpy as np
import pandas as pd
import pytest

from eegtable.preprocessing import load_config, open_workflow, read_checkpoint
from eegtable.preprocessing.config import read_yaml
from eegtable.runner import load_recipe, run
from eegtable.runner.cli import main


def test_cli_init_and_catalog(tmp_path, capsys):
    config = tmp_path / "preprocessing.yaml"
    assert main(["preprocess", "init", str(config), "--mode", "resting"]) == 0
    assert main(["preprocess", "steps", str(config)]) == 0
    output = capsys.readouterr().out
    assert "review-artifact" in output
    assert "export" in output
    assert not (tmp_path / "preprocessed").exists()


@pytest.mark.parametrize(
    "command,flags",
    [
        ("run", ["--until", "--recording", "--n-jobs", "--overwrite", "--progress-json"]),
        ("check", ["--recording"]),
        ("status", ["--recording", "--verify", "--json"]),
        ("step", ["--recording", "--n-jobs", "--overwrite"]),
        ("next", ["--recording", "--n-jobs", "--overwrite"]),
        ("inspect", ["--recording", "--report", "--json"]),
        ("review", ["--recording", "--decisions", "--suggested"]),
        ("reset", ["--from", "--recording"]),
        ("init", ["--mode"]),
    ],
)
def test_every_flag_is_documented(command, flags, capsys):
    with pytest.raises(SystemExit) as stop:
        main(["preprocess", command, "--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    assert all(flag in text for flag in flags)
    assert "recipe" in text.lower()


def _write_config(raw, tmp_path, raw_review):
    raw.save(tmp_path / "recording_raw.fif", fmt="double", verbose=False)
    config = tmp_path / "preprocessing.yaml"
    config.write_text(
        "input: {path: recording_raw.fif}\noutput: {directory: preprocessed, name: recording}\n"
        f"workflow: {{raw_review: {raw_review}}}\nepochs: {{kind: fixed, duration: 2.0}}\n",
        encoding="utf-8",
    )
    return config


def _write_cohort(raw, tmp_path, raw_review, *labels):
    for label in labels:
        (tmp_path / "raw" / label).mkdir(parents=True)
        raw.save(tmp_path / "raw" / label / f"{label}_raw.fif", fmt="double", verbose=False)
    config = tmp_path / "study.yaml"
    config.write_text(
        "input: {root: raw, pattern: '**/*_raw.fif'}\noutput: {directory: preprocessed}\n"
        f"workflow: {{raw_review: {raw_review}}}\nepochs: {{kind: fixed, duration: 2.0}}\n",
        encoding="utf-8",
    )
    return config


def test_cli_return_codes_and_review_gate(raw, tmp_path, capsys):
    config = _write_config(raw, tmp_path, "required")
    assert main(["preprocess", "check", str(config)]) == 0
    assert not (tmp_path / "preprocessed").exists()
    assert main(["preprocess", "run", str(config)]) == 3
    assert f"eegtable preprocess review {config} raw" in capsys.readouterr().out
    assert main(["preprocess", "status", str(config)]) == 0
    assert "review-raw: needs-review" in capsys.readouterr().out
    assert main(["preprocess", "step", str(config), "export"]) == 2
    assert main(["preprocess", "step", str(config), "notch"]) == 2
    assert main(["preprocess", "reset", str(config), "--from", "events"]) == 0


def test_export_feeds_feature_runner_and_hides_checkpoints(raw, tmp_path, capsys):
    config = _write_config(raw, tmp_path, "disabled")
    assert main(["preprocess", "run", str(config)]) == 0
    assert "eegtable init" in capsys.readouterr().out
    assert main(["preprocess", "run", str(config)]) == 0
    assert main(["preprocess", "status", str(config), "--verify"]) == 0
    recipe = tmp_path / "recipe.toml"
    recipe.write_text(
        '[inputs]\nroot = "preprocessed"\npattern = "**/*_epo.fif"\npicks = "eeg"\n'
        '[output]\nroot = "features"\n[bands]\nalpha = [8.0, 13.0]\n'
        '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\n'
        'spatial = ["global"]\n',
        encoding="utf-8",
    )
    result = run(load_recipe(recipe))
    tables = sorted(path.name for path in (tmp_path / "features").rglob("*_features.tsv"))
    assert tables == ["recording_features.tsv"]
    assert len(result.recordings) == 1
    features = pd.read_csv(tmp_path / "features" / "recording_features.tsv", sep="\t")
    assert len(features) == 15
    assert np.isfinite(features.select_dtypes("number").to_numpy()).all()


@pytest.mark.filterwarnings("ignore:Invalid tag with only")
def test_cohort_run_mirrors_tree_and_isolates_failures(raw, tmp_path, capsys):
    config = _write_cohort(raw, tmp_path, "disabled", "sub-01", "sub-02")
    (tmp_path / "raw" / "sub-00").mkdir()
    (tmp_path / "raw" / "sub-00" / "sub-00_raw.fif").write_bytes(b"not a recording")
    assert main(["preprocess", "run", str(config)]) == 1
    out = capsys.readouterr().out
    assert "[1/3] sub-00" in out and "✗" in out
    assert "Opening raw data file" not in out
    assert (tmp_path / "preprocessed" / "sub-01" / "sub-01_epo.fif").exists()
    assert (tmp_path / "preprocessed" / "sub-02" / "sub-02_epo.fif").exists()
    assert "eegtable init" not in out
    assert main(["preprocess", "run", str(config), "--recording", "sub-02"]) == 0
    assert "eegtable init" in capsys.readouterr().out


def test_cohort_gate_prints_complete_commands_and_status_summary(raw, tmp_path, capsys):
    config = _write_cohort(raw, tmp_path, "required", "sub-01", "sub-02")
    assert main(["preprocess", "run", str(config)]) == 3
    out = capsys.readouterr().out
    assert f"eegtable preprocess review {config} --recording sub-01 raw" in out
    assert main(["preprocess", "status", str(config)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("sub-01") and "awaiting review-raw" in line for line in lines)
    assert main(["preprocess", "status", str(config), "--recording", "sub-02"]) == 0
    assert "review-raw: needs-review" in capsys.readouterr().out


def test_cohort_commands_need_a_recording(raw, tmp_path, capsys):
    config = _write_cohort(raw, tmp_path, "disabled", "sub-01", "sub-02")
    assert main(["preprocess", "step", str(config), "load"]) == 2
    assert "--recording" in capsys.readouterr().err
    assert main(["preprocess", "run", str(config), "--recording", "sub-03"]) == 2
    assert "sub-01, sub-02" in capsys.readouterr().err


def test_cohort_flagless_review_walks_pending_recordings(raw, tmp_path, capsys):
    config = _write_cohort(raw, tmp_path, "required", "sub-01", "sub-02")
    assert main(["preprocess", "run", str(config)]) == 3
    for label in ("sub-01", "sub-02"):
        pending = tmp_path / "preprocessed" / label / ".preprocessing" / label / "decisions"
        pending = pending / "review-raw.pending.yaml"
        text = (
            pending.read_text(encoding="utf-8")
            .replace("bads: null", "bads: []")
            .replace("spans: null", "spans: []")
        )
        pending.write_text(text, encoding="utf-8")
    decisions = tmp_path / "one.yaml"
    decisions.write_text(pending.read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["preprocess", "review", str(config), "raw", "--decisions", str(decisions)]) == 2
    assert "--recording" in capsys.readouterr().err
    assert main(["preprocess", "review", str(config), "raw"]) == 0
    assert main(["preprocess", "run", str(config)]) == 0


def test_progress_json_uses_runner_events(raw, tmp_path, capsys):
    config = _write_cohort(raw, tmp_path, "disabled", "sub-01")
    assert main(["preprocess", "run", str(config), "--progress-json"]) == 0
    events = [json.loads(line)["event"] for line in capsys.readouterr().out.splitlines()]
    assert events[0] == "start" and events[-1] == "complete"
    assert {"subject_start", "progress", "subject_done"} <= set(events)


def test_init_refuses_to_overwrite(tmp_path, capsys):
    config = tmp_path / "preprocessing.yaml"
    assert main(["preprocess", "init", str(config)]) == 0
    assert main(["preprocess", "init", str(config)]) == 2
    assert "already exists" in capsys.readouterr().err


@pytest.mark.filterwarnings("ignore:This filename .* does not conform to MNE naming")
def test_check_applies_the_same_source_checks_as_run(raw, tmp_path, capsys):
    raw.save(tmp_path / "rec_epo.fif", fmt="double", verbose=False)
    config = tmp_path / "preprocessing.yaml"
    config.write_text(
        "input: {path: rec_epo.fif}\noutput: {directory: ., name: rec}\n"
        "workflow: {raw_review: disabled}\nepochs: {kind: fixed, duration: 2.0}\n",
        encoding="utf-8",
    )
    assert main(["preprocess", "check", str(config)]) == 1
    assert "collides" in capsys.readouterr().out


def test_stale_checkpoint_is_reported_as_a_command(raw, tmp_path, capsys):
    config = _write_config(raw, tmp_path, "disabled")
    assert main(["preprocess", "run", str(config)]) == 0
    config.write_text(
        config.read_text(encoding="utf-8").replace("duration: 2.0", "duration: 1.0"),
        encoding="utf-8",
    )
    assert main(["preprocess", "run", str(config)]) == 1
    out = capsys.readouterr().out
    assert f"eegtable preprocess reset {config} --from events" in out
    assert "CONFIG" not in out
    assert main(["preprocess", "status", str(config)]) == 0
    assert f"Next: eegtable preprocess reset {config} --from events" in capsys.readouterr().out


def test_cohort_status_suggests_the_next_command(raw, tmp_path, capsys):
    config = _write_cohort(raw, tmp_path, "required", "sub-01", "sub-02")
    assert main(["preprocess", "status", str(config)]) == 0
    assert f"Next: eegtable preprocess run {config}" in capsys.readouterr().out
    assert main(["preprocess", "run", str(config)]) == 3
    capsys.readouterr()
    assert main(["preprocess", "status", str(config)]) == 0
    assert f"Next: eegtable preprocess review {config} raw" in capsys.readouterr().out


def test_cohort_status_verify_rereads_every_recording(raw, tmp_path, capsys):
    config = _write_cohort(raw, tmp_path, "disabled", "sub-01", "sub-02")
    assert main(["preprocess", "run", str(config), "--until", "load"]) == 0
    assert main(["preprocess", "status", str(config), "--verify"]) == 0
    capsys.readouterr()
    (payload,) = (tmp_path / "preprocessed" / "sub-02").rglob("data_raw.fif")
    with payload.open("ab") as stream:
        stream.write(b"corrupt")
    assert main(["preprocess", "status", str(config), "--verify"]) == 2
    assert "hash mismatch" in capsys.readouterr().err


def test_stage_arguments_are_checked_before_any_recording_runs(raw, tmp_path, capsys):
    config = _write_cohort(raw, tmp_path, "disabled", "sub-01", "sub-02")
    assert main(["preprocess", "run", str(config), "--until", "nope"]) == 2
    assert "nope" in capsys.readouterr().err
    assert not (tmp_path / "preprocessed").exists()


@pytest.mark.parametrize("error,traced", [(RuntimeError, True), (ValueError, False)])
def test_only_unexpected_failures_keep_a_traceback(
    raw, tmp_path, capsys, monkeypatch, error, traced
):
    from eegtable.preprocessing import execution

    config = _write_cohort(raw, tmp_path, "disabled", "sub-01")

    def boom(*args, **kwargs):
        raise error("boom")

    monkeypatch.setattr(execution, "run_until", boom)
    assert main(["preprocess", "run", str(config)]) == 1
    captured = capsys.readouterr()
    assert "✗ boom" in captured.out
    assert ("Traceback" in captured.err) is traced


def test_run_summary_points_at_the_pending_review(raw, tmp_path, capsys):
    config = _write_cohort(raw, tmp_path, "required", "sub-01", "sub-02")
    assert main(["preprocess", "run", str(config)]) == 3
    assert f"Next: eegtable preprocess review {config} raw" in capsys.readouterr().out


def test_stage_reporter_shows_progress_only_on_a_terminal():
    from eegtable.preprocessing.cli import StageReporter

    class Tty(io.StringIO):
        def isatty(self):
            return True

    tty = Tty()
    reporter = StageReporter(tty)
    reporter.step("sub-01", "filter", 10, 26)
    reporter.recording_done("sub-01", True, "done")
    assert "· filter (10/26)" in tty.getvalue()
    assert tty.getvalue().endswith("✓ done\n")
    plain = io.StringIO()
    StageReporter(plain).step("sub-01", "filter", 10, 26)
    assert plain.getvalue() == ""


def test_inspect_report_fits_the_checkpoint(raw, tmp_path, monkeypatch):
    import webbrowser

    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda uri: opened.append(uri))
    config = _write_config(raw, tmp_path, "disabled")
    assert main(["preprocess", "run", str(config), "--until", "crop-epochs"]) == 0
    assert main(["preprocess", "inspect", str(config), "load", "--report"]) == 0
    assert main(["preprocess", "inspect", str(config), "crop-epochs", "--report"]) == 0
    assert len(opened) == 2 and all(uri.endswith("-inspection.html") for uri in opened)


def _write_recipe(raw, tmp_path, body):
    raw.save(tmp_path / "recording_raw.fif", fmt="double", verbose=False)
    config = tmp_path / "preprocessing.yaml"
    config.write_text(
        "input: {path: recording_raw.fif}\noutput: {directory: preprocessed, name: recording}\n"
        + body,
        encoding="utf-8",
    )
    return config


def _json_output(capsys):
    return json.loads(capsys.readouterr().out)


def _pending(tmp_path, stage):
    decisions = tmp_path / "preprocessed" / ".preprocessing" / "recording" / "decisions"
    return read_yaml(decisions / f"{stage}.pending.yaml")


def test_status_json_reports_each_recording_and_its_next_action(raw, tmp_path, capsys):
    config = _write_cohort(raw, tmp_path, "required", "sub-01", "sub-02")
    assert main(["preprocess", "run", str(config)]) == 3
    capsys.readouterr()
    assert main(["preprocess", "status", str(config), "--json"]) == 0
    report = _json_output(capsys)
    assert [entry["label"] for entry in report["recordings"]] == ["sub-01", "sub-02"]
    first = report["recordings"][0]
    assert first["summary"] == "awaiting review-raw"
    assert first["next"] == {"kind": "review", "stage": "review-raw", "target": "raw"}
    states = {stage["stage"]: stage["state"] for stage in first["stages"]}
    assert states["load"] == "completed"
    assert states["review-raw"] == "needs-review"
    assert states["export"] == "pending"


def test_status_json_has_no_next_action_once_exported(raw, tmp_path, capsys):
    config = _write_config(raw, tmp_path, "disabled")
    assert main(["preprocess", "run", str(config)]) == 0
    capsys.readouterr()
    assert main(["preprocess", "status", str(config), "--json"]) == 0
    (entry,) = _json_output(capsys)["recordings"]
    assert entry["summary"] == "exported"
    assert entry["next"] is None


def test_status_json_points_a_stale_recording_at_reset(raw, tmp_path, capsys):
    config = _write_config(raw, tmp_path, "disabled")
    assert main(["preprocess", "run", str(config)]) == 0
    config.write_text(
        config.read_text(encoding="utf-8").replace("duration: 2.0", "duration: 1.0"),
        encoding="utf-8",
    )
    capsys.readouterr()
    assert main(["preprocess", "status", str(config), "--json"]) == 0
    (entry,) = _json_output(capsys)["recordings"]
    assert entry["summary"] == "stale at events"
    assert entry["next"] == {"kind": "reset", "stage": "events"}


def test_status_json_points_an_unstarted_recording_at_run(raw, tmp_path, capsys):
    config = _write_config(raw, tmp_path, "disabled")
    assert main(["preprocess", "status", str(config), "--json"]) == 0
    (entry,) = _json_output(capsys)["recordings"]
    assert entry["next"] == {"kind": "run", "stage": "load"}


def test_inspect_json_describes_the_raw_gate(raw, tmp_path, capsys, monkeypatch):
    import mne

    data = raw.get_data()
    data[0] *= 10.0
    loud = mne.io.RawArray(data, raw.info.copy(), first_samp=raw.first_samp)
    config = _write_recipe(
        loud,
        tmp_path,
        "workflow: {raw_review: required}\nannotations: {amplitude: {peak: {eeg: 5.0e-5}}}\n"
        "epochs: {kind: fixed, duration: 2.0}\n",
    )
    assert main(["preprocess", "run", str(config)]) == 3
    capsys.readouterr()
    assert main(["preprocess", "inspect", str(config), "review-raw", "--json"]) == 0
    gate = _json_output(capsys)
    assert gate["stage"] == "review-raw"
    assert gate["parent_id"] == _pending(tmp_path, "review-raw")["parent_id"]
    assert gate["field"] == "bads"
    assert [item["id"] for item in gate["items"]] == loud.ch_names
    # The gate has no checkpoint of its own; the viewer opens the checkpoint it reviews.
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda uri: opened.append(uri))
    assert main(["preprocess", "inspect", str(config), gate["parent"], "--report"]) == 0
    assert opened[0].endswith("-inspection.html")
    by_name = {item["id"]: item for item in gate["items"]}
    assert by_name["Fp1"]["suggested"] and not by_name["C3"]["suggested"]
    assert by_name["Fp1"]["tags"] == ["eeg", "amplitude"]
    assert by_name["VEOG"]["tags"][0] == "eog"
    assert gate["duration"] == pytest.approx(loud.n_times / loud.info["sfreq"])
    assert all(
        set(span) == {"onset", "duration", "description", "suggested"} and span["suggested"]
        for span in gate["spans"]
    )


def test_inspect_json_describes_the_ica_gate(mixture, tmp_path, capsys):
    config = _write_recipe(
        mixture,
        tmp_path,
        "workflow: {raw_review: disabled}\nepochs: {kind: fixed, duration: 2.0}\n"
        "artifact: {method: ica, reference: average, "
        "ica: {n_components: 4, eog_channels: [VEOG], ecg_channel: ECG}}\n",
    )
    assert main(["preprocess", "run", str(config)]) == 3
    capsys.readouterr()
    assert main(["preprocess", "inspect", str(config), "review-artifact", "--json"]) == 0
    gate = _json_output(capsys)
    template = _pending(tmp_path, "review-artifact")
    assert (gate["parent_id"], gate["fit_id"]) == (template["parent_id"], template["fit_id"])
    assert (gate["method"], gate["field"]) == ("ica", "exclude")
    assert [item["id"] for item in gate["items"]] == [0, 1, 2, 3]
    assert gate["items"][0]["label"] == "ICA000"
    workflow = open_workflow(load_config(config))
    evidence = read_checkpoint(workflow, "fit-artifact").state.artifact.evidence
    suggested = [item["id"] for item in gate["items"] if item["suggested"]]
    assert suggested == evidence["suggested_exclude"]
    for item in gate["items"]:
        assert bool(item["tags"]) == item["suggested"]
        assert set(item["tags"]) <= {"VEOG", "ECG"}


def test_inspect_json_describes_the_epoch_gate(raw, tmp_path, capsys):
    config = _write_recipe(
        raw,
        tmp_path,
        "workflow: {raw_review: disabled, epoch_review: required}\n"
        "epochs: {kind: fixed, duration: 2.0}\n",
    )
    assert main(["preprocess", "run", str(config)]) == 3
    capsys.readouterr()
    assert main(["preprocess", "inspect", str(config), "review-epochs", "--json"]) == 0
    gate = _json_output(capsys)
    assert gate["stage"] == "review-epochs"
    assert gate["parent_id"] == _pending(tmp_path, "review-epochs")["parent_id"]
    assert gate["field"] == "exclude"
    assert len(gate["items"]) == 15
    first = gate["items"][0]
    assert (first["id"], first["label"]) == (0, "epoch 0")
    assert len(first["tags"]) == 1 and first["score"] > 0
    assert not any(item["suggested"] for item in gate["items"])


def test_inspect_json_is_only_for_review_gates(raw, tmp_path, capsys):
    config = _write_config(raw, tmp_path, "required")
    assert main(["preprocess", "run", str(config)]) == 3
    assert main(["preprocess", "inspect", str(config), "load", "--json"]) == 2
    assert "review" in capsys.readouterr().err
    assert main(["preprocess", "inspect", str(config), "review-raw", "--json", "--report"]) == 2
    assert "exclusive" in capsys.readouterr().err


def test_review_accepts_a_json_decision_file(raw, tmp_path, capsys):
    config = _write_config(raw, tmp_path, "required")
    assert main(["preprocess", "run", str(config)]) == 3
    capsys.readouterr()
    assert main(["preprocess", "inspect", str(config), "review-raw", "--json"]) == 0
    gate = _json_output(capsys)
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps({"parent_id": gate["parent_id"], "bads": ["C3"], "spans": []}), encoding="utf-8"
    )
    assert main(["preprocess", "review", str(config), "raw", "--decisions", str(decision)]) == 0
    assert main(["preprocess", "run", str(config)]) == 0


def test_a_saved_decision_turns_the_gate_pending_until_run(raw, tmp_path, capsys):
    config = _write_config(raw, tmp_path, "required")
    assert main(["preprocess", "run", str(config)]) == 3
    assert main(["preprocess", "review", str(config), "raw", "--suggested"]) == 0
    capsys.readouterr()
    assert main(["preprocess", "status", str(config), "--json"]) == 0
    (entry,) = _json_output(capsys)["recordings"]
    states = {stage["stage"]: stage["state"] for stage in entry["stages"]}
    assert states["review-raw"] == "pending"
    assert entry["next"] == {"kind": "run", "stage": "review-raw"}
    assert main(["preprocess", "status", str(config)]) == 0
    assert f"Next: eegtable preprocess run {config}" in capsys.readouterr().out
