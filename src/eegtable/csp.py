"""Common spatial patterns, fitted where they cannot see the labels they predict.

CSP is supervised: it uses the class labels to find spatial filters, so fitting
it on a whole dataset and then cross-validating the classifier reports an
accuracy the method did not earn. The leak is invisible in the numbers and
survives every downstream precaution.

:func:`csp_features` produces descriptive held-out features, not a fixed design
for classifier cross-validation: one fold's training features can depend on its
test labels through the other CSP fits. For prediction, fit
:class:`CommonSpatialPattern` inside every training fold and use that same fit
to transform both its training and test rows, including during inner tuning.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh

from eegtable._expand import window_mask
from eegtable._validation import blank_non_finite
from eegtable.signal import Signal
from eegtable.spectra import Window
from eegtable.table import ComputationSpec, FeatureMeta, FeatureTable

__all__ = ["CommonSpatialPattern", "csp_features"]


class _Fold(Protocol):
    """What this module needs of a fold; ``eegtable.model.Fold`` satisfies it."""

    @property
    def train(self) -> npt.NDArray[np.intp]: ...

    @property
    def test(self) -> npt.NDArray[np.intp]: ...


def _fold_indices(fold: Any) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    if hasattr(fold, "train") and hasattr(fold, "test"):
        return np.asarray(fold.train), np.asarray(fold.test)
    train, test = fold
    return np.asarray(train), np.asarray(test)


def _row_indices(rows: npt.NDArray[np.intp] | None, n_epochs: int) -> npt.NDArray[np.intp]:
    if rows is None:
        return np.arange(n_epochs, dtype=np.intp)
    indices = np.asarray(rows)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("CSP row indices must be a nonempty 1-D array.")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("CSP row indices must be integers.")
    if np.any(indices < 0) or np.any(indices >= n_epochs):
        raise ValueError("CSP row indices are out of range.")
    if np.unique(indices).size != indices.size:
        raise ValueError("CSP row indices must not contain duplicates.")
    return indices.astype(np.intp)


def _covariance(epochs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Trace-normalized covariance, averaged over epochs.

    Normalizing each epoch by its own trace before averaging is the original
    algorithm's step for removing between-subject and between-trial magnitude
    differences, so one loud epoch cannot decide the filters on its own.
    """
    if not np.isfinite(epochs).all():
        raise ValueError("CSP fitting requires finite data in every selected training epoch.")
    n_channels = epochs.shape[1]
    total = np.zeros((n_channels, n_channels), dtype=float)
    for epoch in epochs:
        if not np.any(np.ptp(epoch, axis=-1) > 0.0):
            raise ValueError("CSP fitting requires nonzero temporal variance in every epoch.")
        centred = epoch - epoch.mean(axis=-1, keepdims=True)
        covariance = centred @ centred.T
        trace = np.trace(covariance)
        if not np.isfinite(trace) or trace <= 0.0:
            raise ValueError("CSP fitting requires finite positive covariance traces.")
        total += covariance / trace
    return total / int(epochs.shape[0])


def _shrink_covariance(
    covariance: npt.NDArray[np.float64], regularization: float
) -> npt.NDArray[np.float64]:
    rank = covariance.shape[0]
    return np.asarray(
        (1.0 - regularization) * covariance
        + regularization * np.trace(covariance) / rank * np.eye(rank),
        dtype=float,
    )


@dataclass(frozen=True, eq=False)
class CommonSpatialPattern:
    """Spatial filters maximizing the variance ratio between two classes.

    Parameters
    ----------
    filters : ndarray, shape (n_components, n_channels)
        One spatial filter per component, in the order the components are
        emitted: alternating ends of the eigenvalue spectrum, so the first
        favours ``classes[0]`` and the second ``classes[1]``.
    patterns : ndarray, shape (n_components, n_channels)
        The corresponding forward patterns, which are what may be read
        topographically. A filter is not a pattern and does not map to the scalp.
    eigenvalues : ndarray, shape (n_components,)
        Fraction of the pooled variance each component assigns to ``classes[0]``.
    ch_names : tuple of str
        Channels the filters were fitted on, in order.
    classes : tuple of int
        The two class labels, ascending.
    computation : ComputationSpec
        How the filters were fitted.
    """

    filters: npt.NDArray[np.float64]
    patterns: npt.NDArray[np.float64]
    eigenvalues: npt.NDArray[np.float64]
    ch_names: tuple[str, ...]
    classes: tuple[int, int]
    computation: ComputationSpec

    @property
    def n_components(self) -> int:
        """Number of spatial filters."""
        return int(self.filters.shape[0])

    @classmethod
    def fit(
        cls,
        signal: Signal,
        labels: npt.NDArray[np.intp] | Sequence[int],
        *,
        rows: npt.NDArray[np.intp] | None = None,
        n_components: int = 4,
        regularization: float = 0.0,
    ) -> CommonSpatialPattern:
        """Fit filters on the given rows only.

        Parameters
        ----------
        signal : Signal
            Broadband or band-filtered epochs. CSP assumes the band is already
            chosen: its variance ratio is only meaningful within one.
        labels : array-like of int
            One label per epoch of ``signal``, exactly two distinct values.
        rows : ndarray of int, optional
            Which epochs may contribute. None uses all of them, which is correct
            for description and leaks for prediction.
        n_components : int, default 4
            Number of filters, taken in pairs from the two ends of the spectrum,
            so it must be even and at most the rank of the data: the channel count,
            less one for an average reference and one per removed ICA component.
        regularization : float, default 0.0
            Shrinkage toward a sphere, in ``[0, 1)``. Raise it when there are
            more channels than epochs.

        Returns
        -------
        CommonSpatialPattern
        """
        y = np.asarray(labels)
        if y.shape != (signal.data.shape[0],):
            raise ValueError(
                f"labels must have one entry per epoch; got {y.shape} for "
                f"{signal.data.shape[0]} epochs."
            )
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("labels must be integers; class identities cannot be rounded.")
        if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)):
            raise ValueError(f"n_components must be an even integer, got {n_components!r}.")
        if n_components < 2 or n_components % 2:
            raise ValueError(f"n_components must be even and at least 2, got {n_components}.")
        if n_components > len(signal.ch_names):
            raise ValueError(
                f"n_components ({n_components}) exceeds the {len(signal.ch_names)} channels "
                "available; CSP cannot return more filters than sensors."
            )
        if not 0.0 <= regularization < 1.0:
            raise ValueError(f"regularization must be in [0, 1), got {regularization}.")

        selected = _row_indices(rows, y.size)
        present = np.unique(y[selected])
        if present.size != 2:
            raise ValueError(
                f"CSP separates exactly two classes; these rows carry {present.tolist()}."
            )
        first, second = (int(present[0]), int(present[1]))

        covariances = [
            _covariance(signal.data[selected[y[selected] == label]]) for label in (first, second)
        ]
        pooled = covariances[0] + covariances[1]
        # An average reference, or ICA components removed, leaves the pooled covariance
        # singular. As in MNE's CSP, the problem is solved in its non-null subspace, where it
        # is well posed, rather than shrunk toward a sphere the data do not span.
        spread, axes = np.linalg.eigh(pooled)
        basis = axes[:, spread > spread.max() * spread.size * np.finfo(float).eps]
        if n_components > basis.shape[1]:
            raise ValueError(
                f"n_components ({n_components}) exceeds the rank ({basis.shape[1]}) of these "
                "epochs' covariance; a reference or removed ICA components take dimensions "
                "away that no spatial filter can recover."
            )
        # Shrink only within the measured subspace; regularization cannot restore
        # dimensions removed by referencing or artifact correction.
        reduced_covariances = [
            _shrink_covariance(basis.T @ covariance @ basis, regularization)
            for covariance in covariances
        ]
        values, reduced = eigh(
            reduced_covariances[0], reduced_covariances[0] + reduced_covariances[1]
        )
        vectors = basis @ reduced

        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        # Alternate ends: the largest eigenvalue favours the first class, the
        # smallest the second, and a pair from each end is what CSP is for.
        half = n_components // 2
        picks = np.empty(n_components, dtype=int)
        picks[0::2] = np.arange(half)
        picks[1::2] = values.size - 1 - np.arange(half)

        filters = np.asarray(vectors[:, picks].T, dtype=float)
        patterns = np.asarray(np.linalg.pinv(vectors)[picks], dtype=float)
        return cls(
            filters=filters,
            patterns=patterns,
            eigenvalues=np.asarray(values[picks], dtype=float),
            ch_names=tuple(signal.ch_names),
            classes=(first, second),
            computation=ComputationSpec.create(
                "csp",
                n_components=n_components,
                regularization=regularization,
                n_fit_epochs=int(selected.size),
                classes=[first, second],
                covariance="trace_normalized_per_epoch_mean",
                input_computation=signal.computation.record(),
            ),
        )

    def transform(
        self, signal: Signal, *, rows: npt.NDArray[np.intp] | None = None
    ) -> npt.NDArray[np.float64]:
        """Project epochs and return each component's log relative power.

        A sample that is non-finite in any channel is left out of every
        component's variance.

        Parameters
        ----------
        signal : Signal
            Epochs with the fitted channels in the same order.
        rows : ndarray of int, optional
            Epochs to transform; unique and in range. None transforms all.

        Returns
        -------
        ndarray, shape (n_rows, n_components)
            ``log(var_j / sum_k var_k)``, using the natural logarithm and summing
            variance over the retained components. An overall positive amplitude
            scale cancels; changing the number of retained components changes
            the denominator. This differs from log absolute component power.
        """
        if tuple(signal.ch_names) != self.ch_names:
            raise ValueError(
                "this CSP was fitted on different channels, in a different order; "
                f"fitted on {self.ch_names} and asked to transform {tuple(signal.ch_names)}."
            )
        selected = _row_indices(rows, signal.data.shape[0])
        projected = np.einsum("ij,njt->nit", self.filters, signal.data[selected])
        with np.errstate(invalid="ignore", divide="ignore"):
            # Every filter mixes every channel, so one bad sample is bad in all
            # components at that instant. Blanked, it drops out of the variance the
            # way a NaN sample already does instead of voiding the whole epoch.
            variance = np.nanvar(blank_non_finite(projected), axis=2)
            total = variance.sum(axis=1, keepdims=True)
            values = np.log(np.where(total > 0.0, variance / total, np.nan))
        return np.asarray(values, dtype=float)


def _fold_signature(folds: Sequence[Any]) -> str:
    """A digest of the exact split, so a table cannot be reused under another one."""
    payload = ";".join(
        f"{np.asarray(train).tolist()}|{np.asarray(test).tolist()}"
        for train, test in (_fold_indices(fold) for fold in folds)
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def csp_features(
    signal: Signal,
    labels: npt.NDArray[np.intp] | Sequence[int],
    *,
    folds: Sequence[Any],
    window: Window | None = None,
    n_components: int = 4,
    regularization: float = 0.0,
) -> FeatureTable:
    """Cross-fitted CSP features: every row transformed by filters that never saw it.

    For each fold the filters are fitted on its training rows and applied to its
    test rows, so no row's features depend on its own label. This table is for
    description, not downstream classifier cross-validation, even with the same
    folds: training rows can encode test labels through other folds' CSP fits.
    ``build_design`` rejects these columns. For prediction, fit CSP on each
    training fold and transform both train and test rows with that same fit.
    Repeat that procedure inside inner cross-validation when tuning.

    Every row must be tested exactly once. Forward-only splits that leave the
    earliest runs untested cannot produce this descriptive table.

    Parameters
    ----------
    signal : Signal
        Epochs, already restricted to the band of interest. CSP compares variance
        between classes, which is only meaningful within one band.
    labels : array-like of int
        One label per epoch, exactly two distinct values across the whole set.
    folds : sequence
        Objects with ``train`` and ``test`` index arrays, such as
        :class:`~eegtable.model.Fold`, or plain ``(train, test)`` pairs.
    window : Window, optional
        Restrict covariance fitting and held-out projected variance to these
        time samples. None uses the complete signal.
    n_components : int, default 4
        Number of filters, even.
    regularization : float, default 0.0
        Shrinkage toward a sphere, in ``[0, 1)``.

    Returns
    -------
    FeatureTable
        One row per epoch and one column per component, ``space_kind="global"``
        because a spatial filter is a weighting of every channel rather than a
        location.
    """
    y = np.asarray(labels)
    n_epochs = int(signal.data.shape[0])
    if y.shape != (n_epochs,):
        raise ValueError(f"labels must have one entry per epoch; got {y.shape} for {n_epochs}.")
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError(
            "csp_features requires exactly two classes across the whole dataset; "
            f"got {classes.tolist()}."
        )
    if not folds:
        raise ValueError(
            "csp_features requires at least one fold; features cannot be cross-fitted."
        )
    if window is not None:
        within = window_mask(signal.times, window)
        signal = replace(
            signal,
            data=signal.data[..., within],
            times=signal.times[within],
            coverage=signal.coverage[..., within],
            computation=ComputationSpec.create(
                "crop",
                tmin=window.tmin,
                tmax=window.tmax,
                input_computation=signal.computation.record(),
            ),
        )

    values = np.full((n_epochs, n_components), np.nan)
    tested = np.zeros(n_epochs, dtype=bool)
    for index, fold in enumerate(folds, start=1):
        train, test = _fold_indices(fold)
        train = _row_indices(train, n_epochs)
        test = _row_indices(test, n_epochs)
        if np.intersect1d(train, test).size:
            raise ValueError(f"fold {index}: train and test rows overlap.")
        if tested[test].any():
            raise ValueError(
                f"fold {index}: some rows are tested twice, so their features would "
                "depend on which fold wrote them last."
            )
        fitted = CommonSpatialPattern.fit(
            signal,
            y,
            rows=train,
            n_components=n_components,
            regularization=regularization,
        )
        values[test] = fitted.transform(signal, rows=test)
        tested[test] = True

    if not tested.all():
        missing = int((~tested).sum())
        raise ValueError(
            f"{missing} of {n_epochs} rows are in no fold's test set, so they have no "
            "features that were fitted without them. Supply folds that test every row."
        )

    label = window.name if window is not None else "all"
    bounds = (window.tmin, window.tmax) if window is not None else None
    computation = ComputationSpec.create(
        "csp_features",
        n_components=n_components,
        regularization=regularization,
        n_folds=len(folds),
        # The split is part of the specification: the same epochs cross-fitted
        # under a different one are different numbers with the same name.
        folds=_fold_signature(folds),
        cross_fitted=True,
        input_computation=signal.computation.record(),
    )
    meta = tuple(
        FeatureMeta(
            measure="csp_log_power",
            band=signal.band,
            space=f"component{component + 1:02d}",
            space_kind="global",
            window=label,
            normalization="raw",
            unit="log relative power",
            source=signal.source,
            window_bounds=bounds,
            computation=computation,
            freq_resolution_hz=None,
        )
        for component in range(n_components)
    )
    return FeatureTable(
        values=values,
        coverage=np.isfinite(values).astype(float),
        meta=meta,
        row_ids=signal.row_ids,
    )
