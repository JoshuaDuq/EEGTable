"""Batch feature extraction from preprocessed epochs files, driven by a recipe.

The library computes features from MNE objects you build yourself. The runner is
the layer above it: it reads a recipe, finds the epochs files, computes the
spectra and band signals each measure needs, and writes one feature table per
recording. It is what ``eegtable run`` executes.
"""

from __future__ import annotations

from eegtable.extraction import extract
from eegtable.runner.batch import (
    CheckReport,
    Recording,
    RecordingResult,
    RecordingStatus,
    RunError,
    RunResult,
    Trial,
    TrialError,
    check,
    discover,
    run,
    status,
)
from eegtable.runner.compute import RecordingFeatures
from eegtable.runner.recipe import Recipe, RecipeError, load_recipe

__all__ = [
    "CheckReport",
    "Recipe",
    "RecipeError",
    "Recording",
    "RecordingFeatures",
    "RecordingResult",
    "RecordingStatus",
    "RunError",
    "RunResult",
    "Trial",
    "TrialError",
    "check",
    "discover",
    "extract",
    "load_recipe",
    "run",
    "status",
]
