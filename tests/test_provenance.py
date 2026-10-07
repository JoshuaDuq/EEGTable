"""Content identities protect portable feature bundles and resumed runs."""

import json
import os
from pathlib import Path

import pytest

from eegtable.io import read_table
from eegtable.runner import load_recipe, run, status
from tests.synthetic import save_epochs


def _extract(tmp_path, *, pattern="**/*_epo.fif"):
    source = tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif"
    save_epochs(source)
    recipe_path = tmp_path / "recipe.toml"
    recipe_path.write_text(
        f'[inputs]\nroot = "data"\npattern = "{pattern}"\n[output]\nroot = "out"\n'
        '[[features]]\nmeasure = "integrated_band_power"\n'
        'bands = ["alpha"]\nspatial = ["global"]\n',
        encoding="utf-8",
    )
    recipe = load_recipe(recipe_path)
    assert run(recipe).ok
    output = tmp_path / "out/sub-01/eeg/sub-01_task-rest_features.tsv"
    return recipe, source, output


def test_extraction_records_portable_identity_and_resolved_computation(tmp_path):
    _, _, output = _extract(tmp_path)
    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    provenance = sidecar["provenance"]
    assert provenance["recording"] == "sub-01/eeg/sub-01_task-rest_epo.fif"
    assert len(provenance["input_sha256"]) == 64
    assert provenance["software"]["mne"]
    assert provenance["environment"]["eegtable"]
    assert set(provenance["software"]) == {"python", "mne", "numpy", "scipy", "pandas"}
    assert len(provenance["code_sha256"]) == 64
    assert provenance["resolved_settings"]["spectra"]["method"] == "welch"
    assert set(sidecar["files"]) == {output.name, output.stem + "_coverage.tsv"}
    assert read_table(output).row_ids[0][0] == provenance["recording"]


def test_changed_input_is_stale_even_when_its_timestamp_is_preserved(tmp_path):
    recipe, source, _ = _extract(tmp_path)
    timestamp = source.stat().st_mtime_ns
    source.write_bytes(source.read_bytes() + b"changed")
    os.utime(source, ns=(timestamp, timestamp))
    assert status(recipe)[0].state == "stale"


def test_touching_unchanged_input_does_not_invalidate_results(tmp_path):
    recipe, source, _ = _extract(tmp_path)
    later = source.stat().st_mtime + 60
    os.utime(source, (later, later))
    assert status(recipe)[0].state == "done"


@pytest.mark.parametrize("package", ["mne-bids", "onnxruntime", "scikit-learn"])
def test_unused_package_change_preserves_extraction_cache(tmp_path, monkeypatch, package):
    import eegtable.provenance as provenance

    recipe, _, _ = _extract(tmp_path)
    installed_version = provenance.version
    monkeypatch.setattr(
        provenance,
        "version",
        lambda name: "changed" if name == package else installed_version(name),
    )

    assert status(recipe)[0].state == "done"
    resumed = run(recipe, resume=True)
    assert resumed.ok and len(resumed.skipped) == 1 and not resumed.recordings


@pytest.mark.parametrize("module", ["preprocessing/review.py", "model/metrics.py", "cycles.py"])
def test_unused_source_change_preserves_extraction_cache(tmp_path, monkeypatch, module):
    import eegtable.provenance as provenance

    recipe, _, _ = _extract(tmp_path)
    changed = Path(provenance.__file__).parent / module
    original_hash = provenance.file_hash
    monkeypatch.setattr(
        provenance, "file_hash", lambda path: "changed" if path == changed else original_hash(path)
    )

    assert status(recipe)[0].state == "done"


@pytest.mark.parametrize(
    "module",
    [
        "power.py",
        "_expand.py",
        "groups.py",
        "runner/compute.py",
        "runner/measures.py",
        "__init__.py",
    ],
)
def test_relevant_source_change_invalidates_extraction_cache(tmp_path, monkeypatch, module):
    import eegtable.provenance as provenance

    recipe, _, _ = _extract(tmp_path)
    changed = Path(provenance.__file__).parent / module
    original_hash = provenance.file_hash
    monkeypatch.setattr(
        provenance, "file_hash", lambda path: "changed" if path == changed else original_hash(path)
    )

    assert status(recipe)[0].state == "stale"


def test_required_package_change_invalidates_extraction_cache(tmp_path, monkeypatch):
    import eegtable.provenance as provenance

    recipe, _, _ = _extract(tmp_path)
    installed_version = provenance.version
    monkeypatch.setattr(
        provenance, "version", lambda name: "changed" if name == "mne" else installed_version(name)
    )

    assert status(recipe)[0].state == "stale"


@pytest.mark.parametrize(
    "measure,package",
    [("permutation_entropy", "antropy"), ("microstate_coverage", "scikit-learn")],
)
def test_selected_optional_package_versions_are_read_fresh(monkeypatch, measure, package):
    import eegtable.provenance as provenance
    from eegtable.runner.provenance import extraction_software

    recipe = load_recipe({"features": [{"measure": measure}]})
    monkeypatch.setattr(provenance, "version", lambda name: "first")
    before = extraction_software(recipe)
    monkeypatch.setattr(
        provenance, "version", lambda name: "changed" if name == package else "first"
    )

    assert before[package] == "first"
    assert extraction_software(recipe) == {**before, package: "changed"}


def test_plugin_microstates_record_segmentation_dependencies(monkeypatch):
    import eegtable.provenance as provenance
    from eegtable.runner.measures import MEASURES, Measure
    from eegtable.runner.provenance import extraction_software

    plugin = Measure(
        "plugin_microstates", MEASURES["microstate_coverage"].function, ("demo-plugin", "first")
    )
    monkeypatch.setitem(MEASURES, plugin.name, plugin)
    monkeypatch.setattr(provenance, "version", lambda name: "installed")
    recipe = load_recipe({"features": [{"measure": plugin.name}]})

    assert extraction_software(recipe)["scikit-learn"] == "installed"


@pytest.mark.parametrize(
    "measure,package",
    [
        ("wpli", "mne-connectivity"),
        ("permutation_entropy", "antropy"),
        ("spectral_parameterization", "specparam"),
        ("irasa", "neurodsp"),
        ("cycle_features", "bycycle"),
        ("pac_surrogates", "tensorpac"),
    ],
)
def test_plugin_alias_records_implementation_dependencies(monkeypatch, measure, package):
    import eegtable.provenance as provenance
    from eegtable.runner.measures import MEASURES, Measure
    from eegtable.runner.provenance import extraction_software

    plugin = Measure("plugin_alias", MEASURES[measure].function, ("demo-plugin", "first"))
    monkeypatch.setitem(MEASURES, plugin.name, plugin)
    monkeypatch.setattr(provenance, "version", lambda name: "installed")
    entry = {"measure": plugin.name}
    if plugin.kind == "pac":
        entry["pairs"] = [["theta", "gamma"]]
    recipe = load_recipe({"features": [entry]})

    assert package in extraction_software(recipe)


def test_plugin_dependency_change_invalidates_extraction_cache(tmp_path, monkeypatch):
    import eegtable.provenance as provenance
    from eegtable.runner.measures import MEASURES, Measure

    pytest.importorskip("antropy")
    plugin = Measure(
        "plugin_entropy", MEASURES["permutation_entropy"].function, ("demo-plugin", "registered")
    )
    monkeypatch.setitem(MEASURES, plugin.name, plugin)
    monkeypatch.setattr(provenance, "version", lambda name: "first")
    source = tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif"
    save_epochs(source)
    recipe_path = tmp_path / "recipe.toml"
    recipe_path.write_text(
        '[inputs]\nroot = "data"\n[output]\nroot = "out"\n'
        '[[features]]\nmeasure = "plugin_entropy"\nseries = ["broadband"]\n',
        encoding="utf-8",
    )
    recipe = load_recipe(recipe_path)
    assert run(recipe).ok
    assert status(recipe)[0].state == "done"
    monkeypatch.setattr(
        provenance, "version", lambda name: "changed" if name == "antropy" else "first"
    )

    assert status(recipe)[0].state == "stale"
    resumed = run(recipe, resume=True, overwrite=True)
    assert resumed.ok and len(resumed.recordings) == 1 and not resumed.skipped


def test_unused_package_change_during_computation_does_not_block_publication(tmp_path, monkeypatch):
    import eegtable.provenance as provenance
    import eegtable.runner.batch as batch

    recipe, _, _ = _extract(tmp_path)
    compute = batch.compute_features
    installed_version = provenance.version

    def modify_unused_package(*args, **kwargs):
        features = compute(*args, **kwargs)
        monkeypatch.setattr(
            provenance,
            "version",
            lambda name: "changed" if name == "mne-bids" else installed_version(name),
        )
        return features

    monkeypatch.setattr(batch, "compute_features", modify_unused_package)

    assert run(recipe, overwrite=True).ok


def test_renamed_input_is_stale_when_output_stem_and_bytes_match(tmp_path):
    recipe, source, _ = _extract(tmp_path, pattern="**/*epo.fif")
    renamed = source.with_name(source.name.replace("_epo.fif", "-epo.fif"))
    source.rename(renamed)
    entry = status(recipe)[0]
    assert entry.state == "stale"
    assert "recording identity" in entry.reason


def test_modified_output_is_rejected_and_reported_as_partial(tmp_path):
    recipe, _, output = _extract(tmp_path)
    output.write_text(output.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        read_table(output)
    entry = status(recipe)[0]
    assert entry.state == "partial" and "checksum" in entry.reason


def test_missing_provenance_manifest_is_not_accepted_as_current(tmp_path):
    recipe, _, output = _extract(tmp_path)
    path = output.with_suffix(".json")
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    del sidecar["provenance"]["input_sha256"]
    path.write_text(json.dumps(sidecar), encoding="utf-8")
    assert status(recipe)[0].state == "partial"


def test_input_change_during_computation_fails_without_publishing(tmp_path, monkeypatch):
    import eegtable.runner.batch as batch

    source = tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif"
    save_epochs(source)
    recipe_path = tmp_path / "recipe.toml"
    recipe_path.write_text(
        '[inputs]\nroot = "data"\n[output]\nroot = "out"\n'
        '[[features]]\nmeasure = "integrated_band_power"\n'
        'bands = ["alpha"]\nspatial = ["global"]\n',
        encoding="utf-8",
    )
    compute = batch.compute_features

    def modify_input(*args, **kwargs):
        features = compute(*args, **kwargs)
        source.write_bytes(source.read_bytes() + b"changed during computation")
        return features

    monkeypatch.setattr(batch, "compute_features", modify_input)
    result = run(load_recipe(recipe_path))
    assert not result.ok
    assert "input changed during extraction" in result.failed[0].error.lower()
    assert not result.recordings[0].outputs
    assert not list((tmp_path / "out").rglob("*_features.tsv"))


def _split_recording(tmp_path, split_naming):
    import mne
    import numpy as np

    source = tmp_path / "data/sub-01_task-rest_epo.fif"
    source.parent.mkdir()
    epochs = mne.EpochsArray(
        np.random.default_rng(42).normal(scale=1e-5, size=(20, 2, 10_000)),
        mne.create_info(["C3", "C4"], 1000, "eeg"),
        verbose="error",
    )
    paths = epochs.save(source, split_size="2MB", split_naming=split_naming, verbose="error")
    assert len(paths) == 2
    recipe_path = tmp_path / "recipe.toml"
    recipe_path.write_text(
        f'[inputs]\nroot = "data"\npattern = "{paths[0].name}"\n'
        '[output]\nroot = "out"\n[[features]]\n'
        'measure = "integrated_band_power"\nbands = ["alpha"]\nspatial = ["global"]\n',
        encoding="utf-8",
    )
    return load_recipe(recipe_path), paths


@pytest.mark.parametrize("split_naming", ["neuromag", "bids"])
def test_changed_split_recording_is_stale_and_recomputed(tmp_path, split_naming):
    recipe, paths = _split_recording(tmp_path, split_naming)
    assert run(recipe).ok
    assert status(recipe)[0].state == "done"
    paths[1].write_bytes(paths[1].read_bytes() + b"changed")
    assert status(recipe)[0].state == "stale"
    resumed = run(recipe, resume=True, overwrite=True)
    assert resumed.ok and len(resumed.recordings) == 1 and not resumed.skipped


def test_split_recording_manifest_is_portable(tmp_path):
    import shutil

    from eegtable.provenance import file_hash

    recipe, paths = _split_recording(tmp_path, "neuromag")
    result = run(recipe)
    assert result.ok
    sidecar = result.recordings[0].recording.features_path.with_suffix(".json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))["provenance"]
    assert provenance["input_files"] == {path.name: file_hash(path) for path in paths}
    moved = tmp_path / "moved"
    moved.mkdir()
    for name in ("data", "out"):
        shutil.move(tmp_path / name, moved / name)
    shutil.move(recipe.path, moved / recipe.path.name)
    assert status(load_recipe(moved / recipe.path.name))[0].state == "done"


def test_missing_split_recording_is_stale(tmp_path):
    recipe, paths = _split_recording(tmp_path, "neuromag")
    assert run(recipe).ok
    paths[1].unlink()
    assert status(recipe)[0].state == "stale"


@pytest.mark.parametrize("changed", ["edit", "delete"])
def test_split_change_during_computation_fails_without_publishing(tmp_path, monkeypatch, changed):
    import eegtable.runner.batch as batch

    recipe, paths = _split_recording(tmp_path, "neuromag")
    compute = batch.compute_features

    def modify_split(*args, **kwargs):
        features = compute(*args, **kwargs)
        if changed == "edit":
            paths[1].write_bytes(paths[1].read_bytes() + b"changed")
        else:
            paths[1].unlink()
        return features

    monkeypatch.setattr(batch, "compute_features", modify_split)
    result = run(recipe)
    assert not result.ok
    assert not list((tmp_path / "out").rglob("*_features.tsv"))


def test_software_versions_include_installed_preprocessing_dependencies(monkeypatch):
    import eegtable.provenance as provenance

    monkeypatch.setattr(provenance, "version", lambda name: "test-version")
    versions = provenance.software_versions()
    assert {
        "PyYAML",
        "pyprep",
        "autoreject",
        "mne-icalabel",
        "onnxruntime",
        "python-picard",
        "h5io",
        "h5py",
        "filelock",
    } <= set(versions)


@pytest.mark.parametrize("changed", ["manifest", "payload"])
def test_upstream_change_during_computation_fails_without_publishing(
    tmp_path, monkeypatch, changed
):
    import mne

    import eegtable.runner.batch as batch
    from eegtable.provenance import file_hash, identity

    source = tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif"
    save_epochs(source)
    upstream = {"retained": 12, "original_events": 14}
    epochs = mne.read_epochs(source, preload=True, verbose="error")
    epochs.info["description"] = f"preprocessing=preprocessing.json; identity={identity(upstream)}"
    epochs.save(source, overwrite=True, verbose="error")
    payload = source.parent / "events.tsv"
    payload.write_text("original_row\tretained\n0\ttrue\n", encoding="utf-8")
    manifest = source.parent / "preprocessing.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "provenance": upstream,
                "files": {source.name: file_hash(source), payload.name: file_hash(payload)},
            }
        ),
        encoding="utf-8",
    )
    recipe_path = tmp_path / "recipe.toml"
    recipe_path.write_text(
        '[inputs]\nroot = "data"\n[output]\nroot = "out"\n'
        '[[features]]\nmeasure = "integrated_band_power"\n'
        'bands = ["alpha"]\nspatial = ["global"]\n',
        encoding="utf-8",
    )
    compute = batch.compute_features

    def modify_upstream(*args, **kwargs):
        features = compute(*args, **kwargs)
        path = manifest if changed == "manifest" else payload
        path.write_bytes(path.read_bytes() + b"\n")
        return features

    monkeypatch.setattr(batch, "compute_features", modify_upstream)
    result = run(load_recipe(recipe_path))
    assert not result.ok
    assert "preprocessing" in result.failed[0].error.lower()
    assert not list((tmp_path / "out").rglob("*_features.tsv"))


@pytest.mark.parametrize("changed", ["extraction_software", "extraction_hash"])
def test_environment_change_during_computation_fails_without_publishing(
    tmp_path, monkeypatch, changed
):
    import eegtable.runner.batch as batch

    source = tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif"
    save_epochs(source)
    recipe_path = tmp_path / "recipe.toml"
    recipe_path.write_text(
        '[inputs]\nroot = "data"\n[output]\nroot = "out"\n'
        '[[features]]\nmeasure = "integrated_band_power"\n'
        'bands = ["alpha"]\nspatial = ["global"]\n',
        encoding="utf-8",
    )
    original = getattr(batch, changed)
    compute = batch.compute_features

    def modify_environment(*args, **kwargs):
        features = compute(*args, **kwargs)
        monkeypatch.setattr(
            batch,
            changed,
            lambda recipe: (
                {"changed": "version"} if changed == "extraction_software" else "changed"
            ),
        )
        return features

    monkeypatch.setattr(batch, "compute_features", modify_environment)
    result = run(load_recipe(recipe_path))
    monkeypatch.setattr(batch, changed, original)
    assert not result.ok
    assert "changed during extraction" in result.failed[0].error.lower()
    assert not list((tmp_path / "out").rglob("*_features.tsv"))
