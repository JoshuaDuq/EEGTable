"""A recipe applied to epochs in memory, as a notebook would, without files."""

import numpy as np
import pytest

import eegtable as ef
from eegtable.io import read_table
from eegtable.runner import RecipeError, load_recipe, run
from tests.synthetic import make_epochs

ALPHA_GLOBAL = {
    "features": [{"measure": "integrated_band_power", "bands": ["alpha"], "spatial": ["global"]}]
}


def test_extract_applies_a_recipe_mapping_to_epochs_in_memory() -> None:
    epochs = make_epochs()

    features = ef.extract(epochs, ALPHA_GLOBAL, recording="sub-01")

    assert features.epochs is not None
    assert features.epochs.n_rows == len(epochs)
    assert {row_id[0] for row_id in features.epochs.row_ids or ()} == {"sub-01"}
    assert features.crosstrial is None


def test_extract_computes_what_the_runner_writes(tmp_path) -> None:
    # The notebook path and `eegtable run` must agree, value for value and name for name,
    # or a result would depend on how it was produced.
    epochs = make_epochs()
    source = tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif"
    source.parent.mkdir(parents=True)
    # Double precision: FIF defaults to float32, which alone would make the values differ.
    epochs.save(source, fmt="double", verbose="error")
    recipe_path = tmp_path / "recipe.toml"
    recipe_path.write_text(
        '[inputs]\nroot = "data"\n\n[output]\nroot = "out"\n\n'
        "[windows]\nbase = [-0.5, 0.0]\nstim = [0.25, 1.25]\n\n"
        '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\n\n'
        '[[features]]\nmeasure = "erds_mean"\nbands = ["alpha"]\nbaseline = "base"\n',
        encoding="utf-8",
    )
    recipe = load_recipe(recipe_path)
    assert run(recipe).ok
    written = read_table(tmp_path / "out/sub-01/eeg/sub-01_task-rest_features.tsv")

    extracted = ef.extract(epochs, recipe, recording="sub-01/eeg/sub-01_task-rest_epo.fif")

    assert extracted.epochs is not None
    assert extracted.epochs.names == written.names
    np.testing.assert_allclose(extracted.epochs.values, written.values, rtol=1e-12)


def test_extract_picks_channels_on_a_copy_and_leaves_the_epochs_alone() -> None:
    # The recipe keeps good EEG channels, as the runner does when it loads a file; the
    # caller's own object must come back exactly as it was.
    epochs = make_epochs(bads=["Pz"])
    channels = list(epochs.ch_names)

    features = ef.extract(epochs, ALPHA_GLOBAL, recording="sub-01")

    assert features.epochs is not None
    assert features.epochs.meta[0].computation.parameters["spatial_channels"] == [
        "Cz",
        "F3",
        "F4",
        "Fz",
    ]
    assert list(epochs.ch_names) == channels and epochs.info["bads"] == ["Pz"]


def test_a_recipe_mapping_is_validated_like_a_file() -> None:
    with pytest.raises(RecipeError, match="band_powr"):
        load_recipe({"features": [{"measure": "band_powr"}]})
