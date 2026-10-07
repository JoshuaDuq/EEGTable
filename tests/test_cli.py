"""The ``eegtable`` command line: exit codes, output, and the progress stream."""

import io
import json
import re
import shlex
import subprocess
import sys
from importlib import util
from pathlib import Path

import pytest

from eegtable.runner import load_recipe
from eegtable.runner.cli import main
from tests.synthetic import save_epochs

POWER = '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\nspatial = ["global"]\n'
FRONTAL_ROI = (
    '[rois]\nfront = ["Fz", "F3"]\n\n'
    '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\nspatial = ["rois"]\n'
)

# A dB scale needs a baseline window: the recipe loads, and the entry fails when computed.
DB_WITHOUT_BASELINE = (
    '\n[[features]]\nmeasure = "mean_psd"\nbands = ["alpha"]\nspatial = ["global"]\n'
    'normalize = "db"\n'
)


def _recipe(tmp_path: Path, body: str = POWER) -> Path:
    path = tmp_path / "recipe.toml"
    path.write_text(
        f'[inputs]\nroot = "data"\n\n[output]\nroot = "out"\n\n{body}', encoding="utf-8"
    )
    return path


def _recording(tmp_path: Path, subject: str, **kwargs: object) -> None:
    save_epochs(tmp_path / f"data/{subject}/eeg/{subject}_task-rest_epo.fif", **kwargs)


def test_run_writes_results_and_exits_zero(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")

    code = main(["run", str(_recipe(tmp_path))])

    out = capsys.readouterr().out
    assert code == 0
    assert "sub-01_task-rest" in out
    assert "12 epochs" in out and "1 features" in out
    assert (tmp_path / "out/sub-01/eeg/sub-01_task-rest_features.tsv").exists()


def test_progress_json_prints_one_json_event_per_line(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")

    code = main(["run", str(_recipe(tmp_path)), "--progress-json"])

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 0
    assert events[0]["event"] == "start" and events[-1]["event"] == "complete"


def test_run_exits_one_when_a_recording_fails(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")
    _recording(tmp_path, "sub-02", channels=["Fz", "Cz"])

    code = main(["run", str(_recipe(tmp_path, FRONTAL_ROI))])

    assert code == 1
    assert "sub-02_task-rest" in capsys.readouterr().out


def test_invalid_recipe_exits_two_and_lists_its_problems(tmp_path, capsys) -> None:
    code = main(["run", str(_recipe(tmp_path, '[[features]]\nmeasure = "band_powr"\n'))])

    assert code == 2
    assert "band_powr" in capsys.readouterr().err


def test_invalid_recipe_is_an_error_event_in_the_progress_stream(tmp_path, capsys) -> None:
    body = '[[features]]\nmeasure = "band_powr"\n'

    code = main(["run", str(_recipe(tmp_path, body)), "--progress-json"])

    (event,) = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 2
    assert event["event"] == "error" and "band_powr" in event["message"]


def test_existing_results_exit_two_unless_overwritten(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")
    recipe = str(_recipe(tmp_path))
    assert main(["run", recipe]) == 0

    assert main(["run", recipe]) == 2
    assert "--overwrite" in capsys.readouterr().err
    assert main(["run", recipe, "--overwrite"]) == 0


def test_check_exits_zero_and_writes_nothing(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")

    code = main(["check", str(_recipe(tmp_path))])

    assert code == 0
    assert "sub-01_task-rest" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_check_prints_warnings_and_still_exits_zero(tmp_path, capsys) -> None:
    # A split global mean is a problem the run would not stop for, so check reports it
    # beside the readiness line instead of failing.
    _recording(tmp_path, "sub-01")
    _recording(tmp_path, "sub-02", bads=["Pz"])

    code = main(["check", str(_recipe(tmp_path))])

    out = capsys.readouterr().out
    assert code == 0
    assert "sub-02_task-rest lacks Pz" in out and "Ready" in out


def test_check_exits_one_when_the_trial_recording_fails(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")

    code = main(["check", str(_recipe(tmp_path, POWER + DB_WITHOUT_BASELINE))])

    assert code == 1
    assert "baseline" in capsys.readouterr().err


def test_check_finds_a_named_channel_that_a_later_recording_marks_bad(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")
    _recording(tmp_path, "sub-02", bads=["F3"])

    code = main(["check", str(_recipe(tmp_path, FRONTAL_ROI))])

    out = capsys.readouterr().out
    assert code == 1
    assert "sub-02_task-rest" in out and "F3" in out and "marked bad" in out
    assert "Ready" not in out


def test_check_finds_a_named_channel_that_a_later_recording_lacks(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")
    _recording(tmp_path, "sub-02", channels=["Fz", "Cz"])

    code = main(["check", str(_recipe(tmp_path, FRONTAL_ROI))])

    out = capsys.readouterr().out
    assert code == 1
    assert "sub-02_task-rest" in out and "F3" in out


def test_check_finds_an_asymmetry_channel_that_a_later_recording_marks_bad(
    tmp_path, capsys
) -> None:
    _recording(tmp_path, "sub-01")
    _recording(tmp_path, "sub-02", bads=["F4"])
    body = (
        '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\n'
        'spatial = ["channels"]\nasymmetry = [["F3", "F4"]]\n'
    )

    code = main(["check", str(_recipe(tmp_path, body))])

    out = capsys.readouterr().out
    assert code == 1
    assert "sub-02_task-rest" in out and "F4" in out


def test_check_ignores_rois_that_no_entry_uses(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")
    _recording(tmp_path, "sub-02", bads=["F3"])

    code = main(["check", str(_recipe(tmp_path, '[rois]\nfront = ["Fz", "F3"]\n\n' + POWER))])

    assert code == 0
    assert "Ready" in capsys.readouterr().out


def test_a_failed_check_names_the_recipe_entry_it_came_from(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")

    code = main(["check", str(_recipe(tmp_path, POWER + DB_WITHOUT_BASELINE))])

    err = capsys.readouterr().err
    assert code == 1
    assert "features[1] (mean_psd)" in err
    assert "requires a baseline" in err


def test_a_failed_recording_in_a_run_names_the_recipe_entry(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")

    code = main(["run", str(_recipe(tmp_path, POWER + DB_WITHOUT_BASELINE))])

    out = capsys.readouterr().out
    assert code == 1
    assert "features[1] (mean_psd)" in out
    assert "requires a baseline" in out


def test_init_writes_a_recipe_that_loads(tmp_path) -> None:
    path = tmp_path / "recipe.toml"

    assert main(["init", str(path)]) == 0
    assert load_recipe(path).features


def test_init_does_not_overwrite_an_existing_file(tmp_path, capsys) -> None:
    path = tmp_path / "recipe.toml"
    path.write_text("# mine\n", encoding="utf-8")

    assert main(["init", str(path)]) == 2
    assert path.read_text(encoding="utf-8") == "# mine\n"


def test_init_into_a_missing_folder_exits_two(tmp_path, capsys) -> None:
    assert main(["init", str(tmp_path / "nope/recipe.toml")]) == 2
    assert "nope" in capsys.readouterr().err


def test_suggested_commands_can_be_pasted_into_a_shell(tmp_path, capsys) -> None:
    folder = tmp_path / "my study"
    folder.mkdir()
    new = str(folder / "new recipe.toml")
    assert main(["init", new]) == 0
    assert f"eegtable check {shlex.quote(new)}" in capsys.readouterr().out
    _recording(folder, "sub-01")
    recipe = str(_recipe(folder))

    main(["status", recipe])
    assert f"Next: eegtable run {shlex.quote(recipe)} --resume" in capsys.readouterr().out
    main(["check", recipe])
    assert f"Ready: eegtable run {shlex.quote(recipe)}" in capsys.readouterr().out


def test_text_output_survives_a_legacy_code_page(tmp_path, monkeypatch) -> None:
    # Windows gives redirected output its ANSI code page, often cp1252, which has no arrows.
    _recording(tmp_path, "sub-01")
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stdout)

    assert main(["status", str(_recipe(tmp_path))]) == 0

    stdout.flush()
    assert "sub-01_task-rest" in stdout.buffer.getvalue().decode("cp1252")


def test_the_package_runs_as_a_module() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "eegtable", "--version"], capture_output=True, text=True
    )

    assert completed.returncode == 0
    assert completed.stdout.startswith("eegtable ")


def test_check_reports_a_channel_the_first_recording_lacks_instead_of_computing_it(
    tmp_path, capsys
) -> None:
    # Surveying the cohort's channels is cheap and says why; computing a recording that
    # already fails the survey can only raise from inside a measure.
    _recording(tmp_path, "sub-01", bads=["F3"])
    _recording(tmp_path, "sub-02")

    code = main(["check", str(_recipe(tmp_path, FRONTAL_ROI))])

    out = capsys.readouterr().out
    assert code == 1
    assert "sub-01_task-rest" in out and "F3" in out and "marked bad" in out
    assert "sub-02_task-rest" not in out


def test_status_lists_each_recording_and_what_to_run_next(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")
    recipe = str(_recipe(tmp_path))
    assert main(["run", recipe]) == 0
    _recording(tmp_path, "sub-02")
    capsys.readouterr()

    code = main(["status", recipe])

    out = capsys.readouterr().out
    assert code == 0
    assert "sub-01_task-rest" in out and "done" in out
    assert "sub-02_task-rest" in out and "missing" in out
    assert f"eegtable run {recipe} --resume" in out


def test_status_json_describes_every_recording_for_a_front_end(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")
    _recording(tmp_path, "sub-02", channels=["Fz", "Cz"])
    recipe = str(_recipe(tmp_path, FRONTAL_ROI))
    main(["run", recipe])
    capsys.readouterr()

    code = main(["status", recipe, "--json"])

    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["counts"] == {"done": 1, "missing": 0, "failed": 1, "stale": 0, "partial": 0}
    first, second = report["recordings"]
    assert first["label"] == "sub-01_task-rest" and first["state"] == "done"
    assert first["outputs"] and first["input"].endswith("sub-01_task-rest_epo.fif")
    assert second["state"] == "failed" and "F3" in second["reason"]
    assert report["next"] == ["run", recipe, "--resume"]


def test_status_json_asks_for_overwrite_when_results_are_stale(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")
    main(["run", str(_recipe(tmp_path))])
    recipe = str(_recipe(tmp_path, POWER.replace("alpha", "beta")))
    capsys.readouterr()

    main(["status", recipe, "--json"])

    assert json.loads(capsys.readouterr().out)["next"] == ["run", recipe, "--resume", "--overwrite"]


def test_status_json_has_nothing_next_when_everything_is_done(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")
    recipe = str(_recipe(tmp_path))
    main(["run", recipe])
    capsys.readouterr()

    main(["status", recipe, "--json"])

    assert json.loads(capsys.readouterr().out)["next"] is None


def test_status_of_an_invalid_recipe_exits_two(tmp_path, capsys) -> None:
    code = main(["status", str(_recipe(tmp_path, '[[features]]\nmeasure = "band_powr"\n'))])

    assert code == 2
    assert "band_powr" in capsys.readouterr().err


def test_run_resume_skips_what_is_done(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")
    recipe = str(_recipe(tmp_path))
    assert main(["run", recipe]) == 0
    _recording(tmp_path, "sub-02")
    capsys.readouterr()

    code = main(["run", recipe, "--resume"])

    out = capsys.readouterr().out
    assert code == 0
    assert "sub-02_task-rest" in out
    assert "1 up to date" in out


def test_run_with_workers_computes_every_recording(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")
    _recording(tmp_path, "sub-02")

    code = main(["run", str(_recipe(tmp_path)), "--workers", "2"])

    assert code == 0
    for subject in ("sub-01", "sub-02"):
        assert (tmp_path / f"out/{subject}/eeg/{subject}_task-rest_features.tsv").exists()


def test_check_times_the_trial_and_projects_the_run(tmp_path, capsys) -> None:
    for subject in ("sub-01", "sub-02", "sub-03"):
        _recording(tmp_path, subject)

    code = main(["check", str(_recipe(tmp_path)), "--workers", "3"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Slowest" in out and "integrated_band_power" in out
    projected = next(line for line in out.splitlines() if "Projected" in line)
    assert "3 recordings" in projected and "with --workers 3" in projected


def test_a_quick_check_says_how_much_it_computed(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")

    code = main(["check", str(_recipe(tmp_path)), "--quick"])

    out = capsys.readouterr().out
    assert code == 0
    assert "first 4 of 12 epochs" in out
    assert "scaled from 4 of 12 epochs" in out


@pytest.mark.parametrize("template", ["task", "resting"])
def test_each_init_template_checks_cleanly_on_real_epochs(tmp_path, capsys, template) -> None:
    _recording(tmp_path, "sub-01", tmin=-1.0, seconds=2.5)
    path = tmp_path / "recipe.toml"

    assert main(["init", str(path), "--template", template]) == 0
    text = path.read_text(encoding="utf-8")
    text = text.replace('root = "derivatives/preprocessed"', 'root = "data"', 1)
    text = text.replace('root = "derivatives/eegtable"', 'root = "out"', 1)
    path.write_text(text, encoding="utf-8")

    assert main(["check", str(path)]) == 0, capsys.readouterr()


def _uncommented(text: str) -> str:
    # Uncomments each commented-out section and its keys, leaving prose and lone options.
    lines, inside = [], False
    for line in text.splitlines():
        if re.fullmatch(r"# \[\[?\w+\]\]?", line):
            inside = True
        elif not re.match(r"# \w+ = ", line):
            inside = False
        lines.append(line[2:] if inside else line)
    return "\n".join(lines) + "\n"


def test_the_basic_templates_commented_sections_check_cleanly(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01", tmin=-1.0, seconds=2.5)
    path = tmp_path / "recipe.toml"
    assert main(["init", str(path)]) == 0
    text = _uncommented(path.read_text(encoding="utf-8"))
    assert "\n[windows]\n" in text and "\n[bands]\n" in text and 'measure = "itpc"' in text
    text = text.replace('root = "derivatives/preprocessed"', 'root = "data"', 1)
    text = text.replace('root = "derivatives/eegtable"', 'root = "out"', 1)
    path.write_text(text, encoding="utf-8")

    assert main(["check", str(path)]) == 0, capsys.readouterr()


def test_a_quick_check_scales_the_computing_but_not_the_reading(tmp_path) -> None:
    # Reading the file costs the same however many epochs are then computed.
    from eegtable.runner.batch import CheckReport, Trial
    from eegtable.runner.cli import _timing_rows
    from eegtable.runner.compute import EntryTiming, RecordingFeatures

    recordings = tuple(object() for _ in range(2))
    trial = Trial(
        recording=None,  # type: ignore[arg-type]
        n_epochs=4,
        channels=("Fz",),
        features=RecordingFeatures(None, None, (EntryTiming(0, "variance", 6.0),)),
        seconds=10.0,
        epochs_total=12,
        read_seconds=4.0,
    )
    rows = dict(_timing_rows(CheckReport(recordings, (), trial), workers=1))  # type: ignore[arg-type]

    assert rows["Time"].startswith("22 s for this recording")


@pytest.mark.parametrize("dependency", ["yaml", "sklearn", "filelock"])
@pytest.mark.parametrize("command", ["check", "run"])
def test_model_commands_explain_missing_dependencies(
    tmp_path, capsys, monkeypatch, dependency, command
) -> None:
    find_spec = util.find_spec
    monkeypatch.setattr(
        util, "find_spec", lambda name: None if name == dependency else find_spec(name)
    )

    code = main(["model", command, str(tmp_path / "recipe.yml")])

    assert code == 2
    error = capsys.readouterr().err
    assert dependency in error
    assert "pip install 'eegtable[model]'" in error


def test_model_init_needs_no_modeling_dependencies(tmp_path, monkeypatch) -> None:
    find_spec = util.find_spec
    monkeypatch.setattr(
        util,
        "find_spec",
        lambda name: None if name in {"yaml", "sklearn", "filelock"} else find_spec(name),
    )
    path = tmp_path / "model.yaml"

    assert main(["model", "init", str(path)]) == 0
    assert path.is_file()
