"""The ``eegtable`` command on real epochs files.

``check`` must validate a recipe against the recordings and try the first one
without writing; ``run`` must write a bundle per recording, with cross-trial
measures in their own table; and the JSON progress stream must be parseable.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from eegtable.io import read_dataset, read_table
from tests.validation.loaders import Recording

RECIPE = """
[inputs]
root = "epochs"
pattern = "**/*_epo.fif"

[output]
root = "features"

[bands]
mu = [8.0, 13.0]
beta = [13.0, 30.0]

# Equal lengths on purpose: a multitaper grid follows the window length, and the
# runner refuses to put two grids under one column.
[windows]
baseline = [-1.0, 0.0]
movement = [2.0, 3.0]

[rois]
hand = ["C3", "C4"]

[spectra]
method = "multitaper"
bandwidth = 2.0

[trials]
by = "metadata"
column = "condition"

[[features]]
measure = "integrated_band_power"
normalize = "log10"
spatial = ["rois", "global"]

[[features]]
measure = "erds_mean"
baseline = "baseline"
spatial = ["rois"]

[[features]]
measure = "itpc"
bands = ["mu"]
spatial = ["rois"]
"""

DATASET = "eegbci"


def _eegtable(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "eegtable", *args], cwd=cwd, capture_output=True, text=True
    )


@pytest.fixture(scope="module")
def workspace(eegbci_recordings: list[Recording], tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("cli")
    for recording in eegbci_recordings[:2]:
        path = root / "epochs" / recording.name / f"{recording.name}_task-motor_epo.fif"
        path.parent.mkdir(parents=True)
        recording.epochs.save(path, fmt="double", overwrite=True, verbose="error")
    (root / "recipe.toml").write_text(RECIPE, encoding="utf-8")
    return root


@pytest.mark.validates(
    "eegtable command",
    kind="behaviour",
    claim="eegtable check validates a recipe against real files and writes nothing",
    criterion="exit 0; recordings listed; no output directory",
)
def test_check_validates_without_writing(workspace: Path) -> None:
    result = _eegtable("check", "recipe.toml", cwd=workspace)
    assert result.returncode == 0, result.stderr
    assert "2" in result.stdout and "S001" in result.stdout
    assert not (workspace / "features").exists()


@pytest.mark.validates(
    "eegtable command",
    kind="behaviour",
    claim="eegtable check names a channel the recordings lack",
    criterion="exit 1 and the channel named",
)
def test_check_reports_a_channel_the_data_lacks(workspace: Path) -> None:
    broken = RECIPE.replace('hand = ["C3", "C4"]', 'hand = ["C3", "C4", "Nope"]')
    (workspace / "broken.toml").write_text(broken, encoding="utf-8")
    result = _eegtable("check", "broken.toml", cwd=workspace)
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert "Nope" in result.stdout + result.stderr


@pytest.mark.validates(
    "eegtable command",
    "runner",
    kind="behaviour",
    claim="eegtable run writes per-epoch and cross-trial bundles with a JSON progress stream",
    criterion="two bundles of each kind, as the recipe says; a second run refused",
)
def test_run_writes_a_bundle_per_recording(workspace: Path) -> None:
    result = _eegtable("run", "recipe.toml", "--progress-json", cwd=workspace)
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert events, "the JSON progress stream was empty"
    assert all(isinstance(event, dict) for event in events)

    features = sorted((workspace / "features").rglob("*_features.tsv"))
    crosstrial = sorted((workspace / "features").rglob("*_crosstrial.tsv"))
    assert len(features) == 2 and len(crosstrial) == 2

    dataset = read_dataset(features)
    measures = {meta.measure for meta in dataset.table.meta}
    assert measures == {"band_power", "erds_mean"}
    assert all(
        meta.source == "multitaper" for meta in dataset.table.select(measure="band_power").meta
    )
    assert set(dataset.targets["condition"]) == {"rest", "left", "right"}

    groups = read_table(crosstrial[0])
    assert groups.row_labels == ("left", "rest", "right")
    assert {meta.measure for meta in groups.meta} == {"itpc"}

    # A second run must refuse to overwrite earlier results unless told to.
    again = _eegtable("run", "recipe.toml", cwd=workspace)
    assert again.returncode == 2, again.stderr
    forced = _eegtable("run", "recipe.toml", "--overwrite", cwd=workspace)
    assert forced.returncode == 0, forced.stderr
