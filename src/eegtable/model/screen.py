from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from eegtable.model.residualize import residualize_within_subjects

__all__ = ["univariate_screen"]

# Flips are summed in blocks so the flip-by-feature matrix stays small.
_BLOCK = 256


def univariate_screen(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_],
    *,
    feature_names: Sequence[str] | None = None,
    covariates: pd.DataFrame | npt.NDArray[np.float64] | None = None,
    residualize_on: Sequence[str] = (),
    n_flips: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Test every feature for tracking the target within subjects, over the whole family.

    Each subject's correlation between a feature and the target is taken over that
    subject's own trials, averaged across subjects in Fisher z and tested with a
    one-sample t across subjects. Each subject's statistic uses its own trials,
    avoiding shared predictive training data; independence must still follow
    from the study design. Flipping one subject's z for every feature at once
    preserves cross-feature dependence. The maximum ``|t|`` supplies the
    adjustment ``p_fwer``; its randomization interpretation requires independent
    subject vectors and a joint null invariant under sign reversal. ``q`` is
    the Benjamini-Hochberg adjustment of ``p``, whose FDR interpretation also
    requires valid p-values and appropriate dependence assumptions.

    With ``residualize_on``, the features and the target are first residualized on each
    subject's own nuisance design, as in :func:`residualize_within_subjects`, so a
    feature's shared response to a stimulus does not read as tracking.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_features)
        Feature values; NaN is missing. A subject's correlation for a feature uses
        the trials where it is finite and needs at least three of them, with both
        the feature and the target varying.
    y : ndarray, shape (n_trials,)
        Finite target.
    groups : ndarray, shape (n_trials,)
        Subject label of each trial; every trial needs one.
    feature_names : sequence of str, optional
        One name per column, the index of the result; defaults to ``feature_0``,
        ``feature_1``, ...
    covariates : DataFrame or ndarray, optional
        Nuisance values with one row per trial, required by ``residualize_on``.
    residualize_on : sequence of str
        Nuisance columns removed from the features and the target within each
        subject.
    n_flips : int, default 10000
        Random sign-flip draws for ``p_fwer``.
    seed : int, default 42
        Seed of the sign flips.

    Returns
    -------
    DataFrame
        One row per feature: ``r`` (the mean z back in r units), ``t``,
        ``n_subjects``, ``p``, ``q`` and ``p_fwer``. A feature measured in fewer
        than 3 subjects is not tested. Zero mean and zero between-subject variance
        leave ``t``, ``p``, ``q`` and ``p_fwer`` undefined (NaN).
    """
    values = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    labels = np.asarray(groups, dtype=object)
    if values.ndim != 2 or target.shape != (values.shape[0],) or labels.shape != target.shape:
        raise ValueError("X, y and groups must have one row per trial.")
    if not np.all(np.isfinite(target)):
        raise ValueError("The screen needs a finite target for every trial.")
    if pd.isna(labels).any():
        raise ValueError("The screen needs a subject label for every trial.")
    if n_flips < 1:
        raise ValueError(f"n_flips must be at least 1, got {n_flips}.")
    names = (
        [f"feature_{i}" for i in range(values.shape[1])]
        if feature_names is None
        else [str(name) for name in feature_names]
    )
    if len(names) != values.shape[1]:
        raise ValueError(f"{len(names)} feature names for {values.shape[1]} features.")
    if residualize_on:
        if covariates is None:
            raise ValueError("residualize_on needs covariates.")
        every_row, no_row = np.arange(len(target)), np.empty(0, dtype=np.intp)
        values, _ = residualize_within_subjects(
            values, covariates, labels, every_row, no_row, columns=residualize_on
        )
        target, _ = residualize_within_subjects(
            target, covariates, labels, every_row, no_row, columns=residualize_on
        )

    z = np.vstack([_subject_z(values[labels == s], target[labels == s]) for s in pd.unique(labels)])
    measured = np.isfinite(z)
    n_subjects = measured.sum(axis=0)
    testable = n_subjects >= 3
    z = np.where(measured, z, 0.0)
    t = _group_t(z, np.ones(z.shape[0]), measured)

    # The null of the largest |t| over the family, one sign per subject for every feature.
    rng = np.random.default_rng(seed)
    largest = np.concatenate(
        [
            np.nanmax(np.abs(_group_t(z, signs, measured)), axis=1, initial=0.0)
            for signs in np.array_split(
                rng.choice([-1.0, 1.0], size=(n_flips, z.shape[0])),
                max(1, n_flips // _BLOCK),
            )
        ]
    )
    exceeding = n_flips - np.searchsorted(np.sort(largest), np.abs(t), side="left")
    p = np.where(testable, 2.0 * stats.t.sf(np.abs(t), np.maximum(n_subjects - 1, 1)), np.nan)
    q = np.full_like(p, np.nan)
    defined = np.isfinite(p)
    if defined.any():
        q[defined] = stats.false_discovery_control(p[defined])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_z = z.sum(axis=0) / n_subjects
    return pd.DataFrame(
        {
            "r": np.where(testable, np.tanh(mean_z), np.nan),
            "t": t,
            "n_subjects": n_subjects,
            "p": p,
            "q": q,
            "p_fwer": np.where(defined, (1 + exceeding) / (n_flips + 1), np.nan),
        },
        index=pd.Index(names, name="feature"),
    )


def _subject_z(
    values: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    # Pearson r of every feature with the target over the trials where the feature is finite.
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    trials = np.where(finite, values, 0.0)
    paired = np.where(finite, target[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dx = np.where(finite, trials - trials.sum(axis=0) / count, 0.0)
        dy = np.where(finite, paired - paired.sum(axis=0) / count, 0.0)
        r = (dx * dy).sum(axis=0) / np.sqrt((dx * dx).sum(axis=0) * (dy * dy).sum(axis=0))
    # A feature or target constant over its finite trials has no correlation to report.
    varies = _spread(np.where(finite, values, np.nan)) & _spread(
        np.where(finite, target[:, None], np.nan)
    )
    r = np.where((count >= 3) & varies, r, np.nan)
    return np.asarray(np.arctanh(np.clip(r, -0.999999, 0.999999)))


def _spread(values: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    lowest = np.where(np.isnan(values), np.inf, values).min(axis=0, initial=np.inf)
    highest = np.where(np.isnan(values), -np.inf, values).max(axis=0, initial=-np.inf)
    return np.asarray(highest > lowest)


def _group_t(
    z: npt.NDArray[np.float64],
    signs: npt.NDArray[np.float64],
    measured: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    # Center before squaring: subtracting second moments can give negative variance.
    # Process each flip separately to avoid a flips-by-subjects-by-features allocation.
    flips = np.atleast_2d(signs)
    n_subjects = measured.sum(axis=0)
    first_subject = measured.argmax(axis=0)
    features = np.arange(z.shape[1])
    statistics = np.empty((flips.shape[0], z.shape[1]))
    with np.errstate(invalid="ignore", divide="ignore"):
        for row, sign in enumerate(flips):
            flipped = sign[:, None] * z
            reference = flipped[first_subject, features]
            shifted = np.where(measured, flipped - reference, 0.0)
            mean_shift = shifted.sum(axis=0) / n_subjects
            centered = np.where(measured, shifted - mean_shift, 0.0)
            variance = (centered * centered).sum(axis=0) / (n_subjects - 1)
            t = (reference + mean_shift) / np.sqrt(variance / n_subjects)
            statistics[row] = np.where(n_subjects >= 3, t, np.nan)
    return statistics[0] if signs.ndim == 1 else statistics
