import pytest

from eegtable.preprocessing import load_config, load_recipe
from eegtable.preprocessing.config import WorkflowSettings

EPOCHS = "epochs: {kind: fixed, duration: 2.0}\n"


def write(tmp_path, text, *files):
    for name in files:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    recipe = tmp_path / "study.yaml"
    recipe.write_text(text + EPOCHS, encoding="utf-8")
    return recipe


def test_root_recipe_mirrors_tree_and_derives_names(tmp_path):
    recipe = write(
        tmp_path,
        "input: {root: raw, pattern: '**/*_eeg.fif'}\noutput: {directory: out}\n",
        "raw/sub-01/sub-01_eeg.fif",
        "raw/sub-02/sub-02_eeg.fif",
        "raw/sub-02/._sub-02_eeg.fif",
        "raw/notes_eeg.txt",
    )
    recordings = load_recipe(recipe)
    assert list(recordings) == ["sub-01", "sub-02"]
    second = recordings["sub-02"]
    assert second.input.path == tmp_path / "raw/sub-02/sub-02_eeg.fif"
    assert second.output.directory == tmp_path / "out/sub-02"
    assert second.output.name == "sub-02"


def test_path_recipe_keeps_explicit_name_and_derives_when_absent(tmp_path):
    explicit = write(tmp_path, "input: {path: a_raw.fif}\noutput: {directory: out, name: s}\n")
    assert load_recipe(explicit)["s"].output.name == "s"
    derived = write(tmp_path, "input: {path: a_raw.fif}\noutput: {directory: out}\n")
    assert list(load_recipe(derived)) == ["a"]
    assert load_config(derived).output.name == "a"


def test_labels_fall_back_to_relative_paths_on_collision(tmp_path):
    recipe = write(
        tmp_path,
        "input: {root: raw, pattern: '**/*.fif'}\noutput: {directory: out}\n",
        "raw/sub-01/rec_eeg.fif",
        "raw/sub-02/rec_eeg.fif",
    )
    assert list(load_recipe(recipe)) == ["sub-01/rec", "sub-02/rec"]


def test_placeholders_resolve_per_recording(tmp_path):
    recipe = write(
        tmp_path,
        "input: {root: raw, pattern: '*.fif'}\noutput: {directory: out}\n"
        "epochs:\n  kind: events\n  tmin: -0.1\n  tmax: 0.5\n  metadata: 'meta/{name}.tsv'\n"
        "  events: {source: file, path: '{parent}/{name}_events.tsv', event_id: {s: 1}}\n",
        "raw/x_raw.fif",
    )
    recipe.write_text(recipe.read_text(encoding="utf-8").removesuffix(EPOCHS), encoding="utf-8")
    config = load_recipe(recipe)["x"]
    assert config.processing.epochs.events.path == tmp_path / "raw/x_events.tsv"
    assert config.processing.epochs.metadata == tmp_path / "meta/x.tsv"


@pytest.mark.parametrize(
    "text,files,match",
    [
        ("input: {path: a.fif, root: raw}\noutput: {directory: out}\n", (), "input"),
        ("input: {path: a.fif, pattern: '*'}\noutput: {directory: out}\n", (), "input.pattern"),
        ("input: {root: raw}\noutput: {directory: out}\n", ("raw/a.fif",), "input.pattern"),
        (
            "input: {root: raw, pattern: '*.fif'}\noutput: {directory: out, name: s}\n",
            ("raw/a.fif",),
            "output.name",
        ),
        ("input: {root: raw, pattern: '*.fif'}\noutput: {directory: out}\n", (), "input.root"),
        (
            "input: {root: raw, pattern: '*.fif'}\noutput: {directory: out}\n",
            ("raw/",),
            "no files match",
        ),
        (
            "input: {root: raw, pattern: '*'}\noutput: {directory: out}\n",
            ("raw/a.fif", "raw/a.json"),
            "a.json",
        ),
        (
            "input: {root: raw, pattern: '*.fif'}\noutput: {directory: out}\n",
            ("raw/a_raw.fif", "raw/a_eeg.fif"),
            "same bundle",
        ),
        ("input: {path: '@_raw.fif'}\noutput: {directory: out}\n", (), "output.name"),
        ("input: {path: a.fif}\noutput: {directory: out, name: ''}\n", (), "output.name"),
        (
            "input: {root: raw, pattern: '/tmp/*.fif'}\noutput: {directory: out}\n",
            ("raw/a.fif",),
            "input.pattern",
        ),
        (
            "input: {root: raw, pattern: '../*.fif'}\noutput: {directory: out}\n",
            ("raw/a.fif", "b.fif"),
            "input.pattern",
        ),
        (
            "input: {path: a.fif}\noutput: {directory: out}\n"
            "epochs: {kind: events, events: {source: annotations, event_id: {s: 1}},"
            " tmin: 0, tmax: 1, metadata: '{stem}.tsv'}\n",
            (),
            "epochs.metadata",
        ),
    ],
)
def test_invalid_recipes(tmp_path, text, files, match):
    recipe = write(tmp_path, text, *[f for f in files if not f.endswith("/")])
    for folder in (f for f in files if f.endswith("/")):
        (tmp_path / folder).mkdir(exist_ok=True)
    if "metadata" in text:
        recipe.write_text(recipe.read_text(encoding="utf-8").removesuffix(EPOCHS), encoding="utf-8")
    with pytest.raises((ValueError, TypeError), match=match):
        load_recipe(recipe)


def test_load_config_refuses_a_cohort(tmp_path):
    recipe = write(
        tmp_path,
        "input: {root: raw, pattern: '*.fif'}\noutput: {directory: out}\n",
        "raw/a.fif",
        "raw/b.fif",
    )
    with pytest.raises(ValueError, match="2 recordings.*load_recipe"):
        load_config(recipe)


@pytest.mark.parametrize("kwargs", [{"raw_review": "suggested"}, {"artifact_review": "suggested"}])
def test_review_policies_accept_suggested(kwargs):
    WorkflowSettings(**kwargs)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"artifact_review": "disabled"}, "artifact_review"),
        ({"epoch_review": "suggested"}, "epoch_review"),
    ],
)
def test_review_policies_reject_unknown_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        WorkflowSettings(**kwargs)


@pytest.mark.parametrize("value,expected", [("true", True), ("false", False), ("null", False)])
def test_bridges_is_a_boolean(tmp_path, value, expected):
    recipe = write(
        tmp_path, f"input: {{path: a.fif}}\noutput: {{directory: out}}\nbridges: {value}\n"
    )
    assert load_recipe(recipe)["a"].processing.bridges is expected
