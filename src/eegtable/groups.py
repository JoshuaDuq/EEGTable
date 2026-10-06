from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from eegtable._validation import validate_names
from eegtable.table import SpaceKind


@dataclass(frozen=True, eq=False)
class SpatialUnit:
    """One spatial level of a feature: a channel, an ROI, or the global mean."""

    space: str
    space_kind: SpaceKind
    picks: tuple[int, ...]
    values: npt.NDArray[np.float64]
    coverage: npt.NDArray[np.float64]


def aggregate(
    values: npt.NDArray[np.float64],
    coverage: npt.NDArray[np.float64],
    ch_names: Sequence[str],
    groups: Mapping[str, Sequence[str]] | None,
    include_global: bool,
) -> list[SpatialUnit]:
    """Reduce the channel axis to channels, ROIs, or the global mean.

    Parameters
    ----------
    values : ndarray, shape (n_epochs, n_channels, n_windows)
        Per-channel feature values, already normalized.
    coverage : ndarray, same shape as ``values``
        Per-channel coverage.
    ch_names : sequence of str
        Channel names matching the channel axis.
    groups : mapping of str to sequence of str, or None
        ROI name to member channel names. None yields one unit per channel.
    include_global : bool
        Whether to append a unit holding the mean across all channels.

    Returns
    -------
    list of SpatialUnit
        Each carrying arrays of shape ``(n_epochs, n_windows)``.
    """
    index = {name: i for i, name in enumerate(ch_names)}
    units: list[SpatialUnit] = []

    if groups is None:
        units.extend(
            SpatialUnit(name, "channel", (i,), values[:, i, :], coverage[:, i, :])
            for i, name in enumerate(ch_names)
        )
    else:
        for roi, members in groups.items():
            if not members:
                raise ValueError(f"group {roi!r} has no channels.")
            validate_names(members, f"group {roi!r} channels")
            missing = [m for m in members if m not in index]
            if missing:
                raise KeyError(f"group {roi!r} names unknown channels: {missing}")
            picks = [index[m] for m in members]
            units.append(
                SpatialUnit(roi, "roi", tuple(picks), *_mean_over_channels(values, coverage, picks))
            )

    if include_global:
        picks = list(range(len(ch_names)))
        units.append(
            SpatialUnit(
                "global", "global", tuple(picks), *_mean_over_channels(values, coverage, picks)
            )
        )
    return units


def _mean_over_channels(
    values: npt.NDArray[np.float64],
    coverage: npt.NDArray[np.float64],
    picks: Sequence[int],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    subset = values[:, picks, :]
    finite = np.isfinite(subset)
    with warnings.catch_warnings():
        # A channel that produced no value is absent from the group, not a zero in
        # it; a group where none did is an all-NaN slice by design.
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        mean = np.where(
            finite.any(axis=1), np.nanmean(np.where(finite, subset, np.nan), axis=1), np.nan
        )
    members = coverage[:, picks, :]
    # A member without a value put none of its input into the mean. A unit with no member
    # value at all is withheld, and reports the input it had, as a withheld channel does.
    averaged = np.where(finite, members, 0.0).mean(axis=1)
    return mean, np.where(finite.any(axis=1), averaged, members.mean(axis=1))
