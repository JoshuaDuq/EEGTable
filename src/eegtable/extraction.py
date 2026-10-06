"""A recipe applied to epochs in memory: the batch runner's computation, without files."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from eegtable.runner.compute import RecordingFeatures
    from eegtable.runner.recipe import Recipe


def extract(
    epochs: Any,
    recipe: Recipe | str | os.PathLike[str] | Mapping[str, Any],
    *,
    recording: str,
    n_jobs: int = 1,
) -> RecordingFeatures:
    """Compute a recipe's features for epochs already in memory.

    This is what ``eegtable run`` computes for one recording, without reading or
    writing files: the recipe's channel selection, spectra, band signals and
    measures, giving the same values under the same column names.

    Parameters
    ----------
    epochs : mne.Epochs
        The recording. Channels are picked from a copy, as ``[inputs]`` asks, so
        the object passed in is left unchanged.
    recipe : Recipe, path-like or mapping
        A loaded recipe, a TOML file, or a mapping with the same structure, as
        :func:`eegtable.runner.load_recipe` accepts. A mapping needs only the
        sections it uses, e.g. ``{"features": [{"measure": "integrated_band_power"}]}``.
    recording : str
        The identity every row carries, as the runner records a file's path.
    n_jobs : int, default 1
        Passed to MNE's filtering and spectral estimation. It never changes a value.

    Returns
    -------
    RecordingFeatures
        ``epochs``, the per-epoch table, and ``crosstrial``, the table of measures
        estimated across trials; either is None when the recipe has no such entry.
    """
    # Imported here: the runner looks its measures up on this package by name.
    from eegtable.runner.batch import _pick
    from eegtable.runner.compute import compute_features
    from eegtable.runner.recipe import Recipe, load_recipe

    loaded = recipe if isinstance(recipe, Recipe) else load_recipe(recipe)
    selected = _pick(epochs.copy().load_data(), loaded.inputs)
    return compute_features(selected, loaded, recording=recording, n_jobs=n_jobs)
