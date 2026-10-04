from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from eegtable._validation import blank_non_finite
from eegtable.model import _deps as _deps
from eegtable.model._nuisance import fit_coefficients
from eegtable.model.transformers import PreprocessingConfig, validate_subject_missingness

__all__ = [
    "FoldNuisanceFit",
    "StagedResidualPreprocessor",
    "fit_nuisance_model",
    "fit_staged_residual_preprocessor",
    "reconstruct_staged_permutation_target_for_fold",
    "residualize_targets",
    "residualize_within_subjects",
]

# A residual this small relative to the values it came from is rounding, not variation: a
# thousand times the error of residualizing on a well-conditioned design.
_ROUNDING = 1e-12


@dataclass(frozen=True)
class FoldNuisanceFit:
    train_target: npt.NDArray[np.float64]
    test_target: npt.NDArray[np.float64]
    train_prediction: npt.NDArray[np.float64]
    test_prediction: npt.NDArray[np.float64]
    train_residual: npt.NDArray[np.float64]
    test_residual: npt.NDArray[np.float64]
    details: Mapping[str, object]


def _validate_indices(
    y: npt.NDArray[np.float64],
    train_idx: npt.NDArray[np.intp],
    test_idx: npt.NDArray[np.intp],
) -> None:
    if y.ndim != 1:
        msg = f"Target residualization expects a 1D target vector, got shape {y.shape}."
        raise ValueError(msg)
    if train_idx.size == 0:
        msg = "Target residualization requires non-empty train indices."
        raise ValueError(msg)
    max_index = len(y) - 1
    if np.any(train_idx < 0) or np.any(test_idx < 0):
        msg = "Target residualization indices must be non-negative."
        raise ValueError(msg)
    if np.any(train_idx > max_index) or (test_idx.size > 0 and np.any(test_idx > max_index)):
        msg = "Target residualization indices exceed target length."
        raise ValueError(msg)
    if test_idx.size > 0 and np.intersect1d(train_idx, test_idx).size > 0:
        msg = "Train and test indices must not overlap."
        raise ValueError(msg)


def _validate_training_nuisance_rank(
    nuisance_design: npt.NDArray[np.float64],
    columns: Sequence[str],
    *,
    tolerance: float = 1e-10,
) -> None:
    centered = nuisance_design - np.mean(nuisance_design, axis=0, keepdims=True)
    column_norms = np.linalg.norm(centered, axis=0)
    # Centring a constant with no exact binary form leaves rounding residue, not zeros, so
    # a column is judged constant relative to its own magnitude.
    column_scales = np.max(np.abs(nuisance_design), axis=0) * np.sqrt(nuisance_design.shape[0])
    if np.any(column_norms <= float(tolerance) * column_scales):
        msg = (
            "Target residualization design is rank deficient. "
            f"Columns={list(columns)} contain constant training-fold nuisance terms."
        )
        raise ValueError(msg)

    scaled = centered / column_norms
    singular_values = np.linalg.svd(scaled, compute_uv=False)
    if singular_values.size < len(columns):
        msg = (
            "Target residualization design is rank deficient. "
            f"Columns={list(columns)}, rank={singular_values.size}, parameters={len(columns)}."
        )
        raise ValueError(msg)

    max_singular_value = float(singular_values[0])
    if max_singular_value <= 0.0:
        msg = (
            "Target residualization design is rank deficient. "
            f"Columns={list(columns)} have zero training-fold variance."
        )
        raise ValueError(msg)
    singular_ratios = singular_values / max_singular_value
    if np.any(singular_ratios < float(tolerance)):
        rank = int(np.sum(singular_ratios >= float(tolerance)))
        msg = (
            "Target residualization design is rank deficient. "
            f"Columns={list(columns)}, rank={rank}, parameters={len(columns)}, "
            f"tolerance={float(tolerance)}."
        )
        raise ValueError(msg)


def _design_matrix(
    covariates: pd.DataFrame | npt.NDArray[np.float64],
    rows: npt.NDArray[np.intp],
    columns: Sequence[str],
    *,
    check_rank: bool,
) -> npt.NDArray[np.float64]:
    if isinstance(covariates, pd.DataFrame):
        missing = [c for c in columns if c not in covariates.columns]
        if missing:
            msg = f"Target residualization nuisance columns are missing: {missing}."
            raise ValueError(msg)
        sub = covariates.iloc[rows]
        design_columns: list[npt.NDArray[np.float64]] = [np.ones(len(rows), dtype=np.float64)]
        for col in columns:
            values = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(values)):
                msg = f"Target residualization column '{col}' contains non-finite values."
                raise ValueError(msg)
            design_columns.append(values)
        design = np.column_stack(design_columns)
    else:
        cov_arr = np.asarray(covariates, dtype=np.float64)
        if cov_arr.ndim == 1:
            cov_arr = cov_arr.reshape(-1, 1)
        if cov_arr.shape[1] != len(columns):
            msg = (
                f"Covariates width ({cov_arr.shape[1]}) does not match "
                f"columns count ({len(columns)})."
            )
            raise ValueError(msg)
        sub_arr = cov_arr[rows]
        if not np.all(np.isfinite(sub_arr)):
            msg = "Target residualization covariates contain non-finite values."
            raise ValueError(msg)
        design = np.column_stack([np.ones(len(rows), dtype=np.float64), sub_arr])

    if check_rank:
        _validate_training_nuisance_rank(design[:, 1:], columns)
    return design


def fit_nuisance_model(
    y: npt.NDArray[np.float64] | Sequence[float],
    covariates: pd.DataFrame | npt.NDArray[np.float64],
    train: npt.NDArray[np.intp] | Sequence[int],
    test: npt.NDArray[np.intp] | Sequence[int] | None = None,
    *,
    columns: Sequence[str],
) -> FoldNuisanceFit:
    """Regress a nuisance model out of the target, fitting it on ``train`` alone.

    The coefficients come from ``train`` and are only applied to ``test``, which
    is what keeps confound control out of the cross-validation's way: fitting the
    nuisance model on all rows lets the held-out target inform its own adjustment
    (Snoek, Miletic & Scholte 2019, NeuroImage 184, 741-760).

    Both residuals are returned because they are different objects: the training
    residual is what a model should be fitted on, the test residual is what it
    should be scored against. Residualizing the target also changes the estimand
    -- performance is now on the part of the target the confound does not explain
    -- so it is not a free correction.
    """
    column_names = tuple(str(c).strip() for c in columns if str(c).strip())
    if not column_names:
        msg = "Target residualization requires at least one nuisance column."
        raise ValueError(msg)

    y_values = np.asarray(y, dtype=np.float64)
    train_indices = np.asarray(train, dtype=np.intp)
    test_indices = (
        np.asarray(test, dtype=np.intp) if test is not None else np.empty(0, dtype=np.intp)
    )
    _validate_indices(y_values, train_indices, test_indices)

    design_train = _design_matrix(covariates, train_indices, column_names, check_rank=True)
    y_train = y_values[train_indices]
    if not np.all(np.isfinite(y_train)):
        msg = "Target residualization requires finite train target values."
        raise ValueError(msg)
    if len(y_train) <= design_train.shape[1]:
        msg = (
            "Target residualization requires more training rows than nuisance parameters: "
            f"rows={len(y_train)}, parameters={design_train.shape[1]}."
        )
        raise ValueError(msg)

    coefficients = fit_coefficients(design_train, y_train)
    train_prediction = design_train @ coefficients
    train_residual = y_train - train_prediction

    if test_indices.size > 0:
        design_test = _design_matrix(covariates, test_indices, column_names, check_rank=False)
        y_test = y_values[test_indices]
        if not np.all(np.isfinite(y_test)):
            msg = "Target residualization requires finite test target values."
            raise ValueError(msg)
        test_prediction = design_test @ coefficients
        test_residual = y_test - test_prediction
    else:
        y_test = np.empty(0, dtype=np.float64)
        test_prediction = np.empty(0, dtype=np.float64)
        test_residual = np.empty(0, dtype=np.float64)

    details: dict[str, object] = {
        "columns": list(column_names),
        "n_parameters": int(design_train.shape[1]),
        "n_train": int(len(train_indices)),
        "n_test": int(len(test_indices)),
    }
    return FoldNuisanceFit(
        train_target=y_train,
        test_target=y_test,
        train_prediction=train_prediction,
        test_prediction=test_prediction,
        train_residual=train_residual,
        test_residual=test_residual,
        details=details,
    )


def residualize_targets(
    y: npt.NDArray[np.float64] | Sequence[float],
    covariates: pd.DataFrame | npt.NDArray[np.float64],
    train: npt.NDArray[np.intp] | Sequence[int],
    test: npt.NDArray[np.intp] | Sequence[int],
    *,
    columns: Sequence[str],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """``(train_residual, test_residual)`` from :func:`fit_nuisance_model`.

    The pair :func:`~eegtable.model.cross_fit_regression` uses internally when it
    is given ``residualize_on``; call that rather than this unless you are
    building your own fold loop.
    """
    fit = fit_nuisance_model(y, covariates, train, test, columns=columns)
    return fit.train_residual, fit.test_residual


def residualize_within_subjects(
    values: npt.NDArray[np.float64],
    covariates: pd.DataFrame | npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_],
    train: npt.NDArray[np.intp] | Sequence[int],
    test: npt.NDArray[np.intp] | Sequence[int],
    *,
    columns: Sequence[str],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """``(train_residual, test_residual)`` of each subject's own nuisance model.

    ``values`` is the target, or a feature matrix, with one row per design row. A pooled
    nuisance model removes only the average nuisance effect, so each subject's own
    deviation from it -- its own stimulus response, say -- survives in the residuals of
    both the target and the features and reads as trial-level tracking. Here every
    subject gets its own model instead. A subject with training rows is fitted on those
    alone and its held-out rows are predicted from them. A subject with none, as in a
    leave-one-subject-out fold, is fitted on its own held-out rows. Target and
    feature nuisance fits use that subject's batches separately; feature fits
    use no target. The predictive estimator receives no held-out targets, but
    the residual outcome and test features depend on the held-out batch. This
    defines a within-subject residual association, not a training-only transform
    for prospective prediction of an unseen raw outcome. Missing values remain
    missing.
    """
    column_names = tuple(str(c).strip() for c in columns if str(c).strip())
    if not column_names:
        raise ValueError("Target residualization requires at least one nuisance column.")
    data = np.asarray(values, dtype=np.float64)
    group_labels = np.asarray(groups, dtype=object)
    train_rows = np.asarray(train, dtype=np.intp)
    test_rows = np.asarray(test, dtype=np.intp)
    _validate_indices(np.empty(len(data)), train_rows, test_rows)
    if len(group_labels) != len(data):
        raise ValueError(f"groups has {len(group_labels)} rows and values has {len(data)}.")

    residual = np.full(data.shape, np.nan)
    for subject in pd.unique(group_labels[np.concatenate([train_rows, test_rows])]):
        own_train = train_rows[group_labels[train_rows] == subject]
        own_test = test_rows[group_labels[test_rows] == subject]
        rows = np.concatenate([own_train, own_test])
        try:
            residual[rows] = _subject_residual(
                data,
                covariates,
                fit_rows=own_train if own_train.size else own_test,
                rows=rows,
                columns=column_names,
                extrapolate=bool(own_train.size and own_test.size),
            )
        except ValueError as exc:
            raise ValueError(f"Subject {subject}: {exc}") from exc
    return residual[train_rows], residual[test_rows]


def _subject_residual(
    data: npt.NDArray[np.float64],
    covariates: pd.DataFrame | npt.NDArray[np.float64],
    *,
    fit_rows: npt.NDArray[np.intp],
    rows: npt.NDArray[np.intp],
    columns: tuple[str, ...],
    extrapolate: bool,
) -> npt.NDArray[np.float64]:
    fit_design = _design_matrix(covariates, fit_rows, columns, check_rank=extrapolate)
    design = _design_matrix(covariates, rows, columns, check_rank=False)
    # Comparable units, so the rank cutoff does not depend on the covariates' scale.
    scales = np.linalg.norm(fit_design, axis=0)
    scales[scales == 0.0] = 1.0
    if data.ndim == 1:
        target = data[fit_rows]
        if not np.all(np.isfinite(target)):
            raise ValueError("Target residualization requires finite target values.")
        rank = int(np.linalg.matrix_rank(fit_design / scales))
        if fit_rows.size - rank < 3:
            raise ValueError(
                f"{fit_rows.size} trials and a nuisance design of rank {rank} leave fewer "
                "than 3 residual degrees of freedom to correlate."
            )
        # Predicting held-out rows needs identified coefficients; residualizing the rows
        # that were fitted needs only the projection, which a constant column leaves intact.
        coefficients = (
            fit_coefficients(fit_design, target)
            if extrapolate
            else np.linalg.lstsq(fit_design / scales, target, rcond=None)[0] / scales
        )
        return _without_rounding(data[rows] - design @ coefficients, target)

    # Feature columns are fitted on the rows where they are finite, one missingness pattern
    # at a time; a column left with no residual degrees of freedom is missing.
    finite = np.isfinite(data[fit_rows])
    patterns, inverse = np.unique(finite.T, axis=0, return_inverse=True)
    residual = np.full((rows.size, data.shape[1]), np.nan)
    for which, pattern in enumerate(patterns):
        kept = np.flatnonzero(inverse.ravel() == which)
        pattern_design = fit_design[pattern] / scales
        rank = int(np.linalg.matrix_rank(pattern_design)) if pattern.any() else 0
        if pattern.sum() - rank < 2 or (extrapolate and rank < pattern_design.shape[1]):
            continue
        fitted = data[fit_rows[pattern]][:, kept]
        coefficients = np.linalg.lstsq(pattern_design, fitted, rcond=None)[0] / scales[:, None]
        residual[:, kept] = data[rows][:, kept] - design @ coefficients
    return _without_rounding(residual, data[fit_rows])


def _without_rounding(
    residual: npt.NDArray[np.float64], fitted_values: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    # A value the nuisance design explains exactly, such as a feature constant within a
    # subject, leaves rounding error of about 1e-15 rather than zero, and anything downstream
    # reads that as variation: a correlation, or a unit-variance column after scaling.
    # The rounding scale is fitted too: held-out values must not erase training variation.
    scale = np.max(np.where(np.isfinite(fitted_values), np.abs(fitted_values), 0.0), axis=0)
    residual = blank_non_finite(residual)
    return np.where(np.abs(residual) <= _ROUNDING * scale, 0.0, residual)


def _finite_feature_block(
    X: npt.NDArray[np.float64], rows: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    values = np.asarray(X, dtype=np.float64)[rows].copy()
    values[~np.isfinite(values)] = np.nan
    return values


@dataclass(frozen=True)
class StagedResidualPreprocessor:
    columns: tuple[str, ...]
    feature_support: npt.NDArray[np.bool_]
    feature_medians: npt.NDArray[np.float64]
    feature_coefficients: npt.NDArray[np.float64]
    target_coefficients: npt.NDArray[np.float64]
    power_transform: object
    n_fit_rows: int
    max_subject_missingness: float

    def _design(
        self,
        covariates: pd.DataFrame | npt.NDArray[np.float64],
        rows: npt.NDArray[np.intp],
        *,
        check_rank: bool,
    ) -> npt.NDArray[np.float64]:
        return _design_matrix(covariates, rows, self.columns, check_rank=check_rank)

    def transform_features(
        self,
        X: npt.NDArray[np.float64],
        covariates: pd.DataFrame | npt.NDArray[np.float64],
        rows: npt.NDArray[np.intp],
        groups: Sequence[object] | npt.NDArray[Any],
    ) -> npt.NDArray[np.float64]:
        values = _finite_feature_block(X, rows)[:, self.feature_support]
        validate_subject_missingness(
            values,
            cast(npt.NDArray[np.object_], np.asarray(groups)[rows]),
            maximum=self.max_subject_missingness,
        )
        filled = np.where(np.isnan(values), self.feature_medians[None, :], values)
        design = self._design(covariates, rows, check_rank=False)
        return filled - design @ self.feature_coefficients

    def nuisance_prediction(
        self,
        covariates: pd.DataFrame | npt.NDArray[np.float64],
        rows: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.float64]:
        design = self._design(covariates, rows, check_rank=False)
        return design @ self.target_coefficients

    def transform_target(
        self,
        y: npt.NDArray[np.float64],
        covariates: pd.DataFrame | npt.NDArray[np.float64],
        rows: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.float64]:
        pred = self.nuisance_prediction(covariates, rows)
        residual = np.asarray(y, dtype=np.float64)[rows] - pred
        from sklearn.preprocessing import PowerTransformer

        pt = cast(PowerTransformer, self.power_transform)
        transformed = pt.transform(residual.reshape(-1, 1)).flatten()
        return cast(npt.NDArray[np.float64], transformed)

    def inverse_transform_target(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        from sklearn.preprocessing import PowerTransformer

        pt = cast(PowerTransformer, self.power_transform)
        inversed = pt.inverse_transform(
            np.asarray(values, dtype=np.float64).reshape(-1, 1)
        ).flatten()
        return cast(npt.NDArray[np.float64], inversed)


def fit_staged_residual_preprocessor(
    *,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    covariates: pd.DataFrame | npt.NDArray[np.float64] | None = None,
    meta: pd.DataFrame | npt.NDArray[np.float64] | None = None,
    groups: Sequence[object] | npt.NDArray[Any],
    rows: npt.NDArray[np.intp],
    columns: Sequence[str],
    config: PreprocessingConfig | None = None,
) -> StagedResidualPreprocessor:
    """Fit every data-dependent preprocessing step for one fold, on ``rows`` alone.

    ``rows`` must be the **training** rows of the fold being fitted, never all
    rows. Everything this learns is learned from them: which features clear the
    missingness threshold, the medians that fill the rest, the nuisance
    coefficients for the features, the nuisance coefficients for the target, and
    the Yeo-Johnson transform of the residualized target. Passing every row
    instead is accepted silently -- the signature cannot tell the two apart --
    and makes the held-out rows contribute to their own preprocessing.

    Use it when a fold needs the feature-side and target-side adjustments kept in
    step, as the staged permutation null does. For ordinary cross-fitting, pass
    ``covariates`` and ``residualize_on`` to
    :func:`~eegtable.model.cross_fit_regression`, which does the same thing per
    fold without the caller holding the contract.
    """
    from sklearn.preprocessing import PowerTransformer

    cov = covariates if covariates is not None else meta
    if cov is None:
        msg = "fit_staged_residual_preprocessor requires 'covariates' or 'meta'."
        raise ValueError(msg)

    cfg = config or PreprocessingConfig()
    fit_rows = np.asarray(rows, dtype=np.intp)
    values = _finite_feature_block(X, fit_rows)

    max_missing = cfg.max_feature_missingness
    missing_rate = (
        np.isnan(values).sum(axis=0) / values.shape[0]
        if values.shape[0]
        else np.ones(values.shape[1])
    )
    support = missing_rate <= max_missing
    if not np.any(support):
        msg = (
            f"Every feature exceeds the {max_missing:.1%} missingness limit on this "
            "training split; nothing is left to residualize."
        )
        raise ValueError(msg)

    kept = values[:, support]
    max_subject_missingness = cfg.max_subject_missingness
    validate_subject_missingness(
        kept,
        cast(npt.NDArray[np.object_], np.asarray(groups)[fit_rows]),
        maximum=max_subject_missingness,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        medians = np.nanmedian(kept, axis=0)
    if not np.all(np.isfinite(medians)):
        msg = (
            "A feature retained by the missingness policy has no finite training value "
            "to impute from."
        )
        raise ValueError(msg)
    imputed = np.where(np.isnan(kept), medians[None, :], kept)

    design = _design_matrix(cov, fit_rows, tuple(columns), check_rank=True)
    feature_coefficients = fit_coefficients(design, imputed)

    y_fit = np.asarray(y, dtype=np.float64)[fit_rows]
    if not np.all(np.isfinite(y_fit)):
        msg = "Staged residual learning requires finite training target values."
        raise ValueError(msg)
    if y_fit.size <= design.shape[1]:
        msg = (
            "Staged residual learning requires more training rows than nuisance parameters: "
            f"rows={y_fit.size}, parameters={design.shape[1]}."
        )
        raise ValueError(msg)
    target_coefficients = fit_coefficients(design, y_fit)

    power_transform = PowerTransformer(method="yeo-johnson", standardize=True)
    power_transform.fit((y_fit - design @ target_coefficients).reshape(-1, 1))

    return StagedResidualPreprocessor(
        columns=tuple(columns),
        feature_support=support,
        feature_medians=medians,
        feature_coefficients=feature_coefficients,
        target_coefficients=target_coefficients,
        power_transform=power_transform,
        n_fit_rows=int(fit_rows.size),
        max_subject_missingness=max_subject_missingness,
    )


def reconstruct_staged_permutation_target_for_fold(
    *,
    y: npt.NDArray[np.float64],
    covariates: pd.DataFrame | npt.NDArray[np.float64] | None = None,
    meta: pd.DataFrame | npt.NDArray[np.float64] | None = None,
    train: npt.NDArray[np.intp] | None = None,
    test: npt.NDArray[np.intp] | None = None,
    train_idx: npt.NDArray[np.intp] | None = None,
    test_idx: npt.NDArray[np.intp] | None = None,
    columns: Sequence[str],
    permutation_indices: npt.NDArray[np.intp],
) -> npt.NDArray[np.float64]:
    """Rebuild a permuted target that keeps the nuisance relationship intact.

    The nuisance model is fitted on ``train``, and the target is rebuilt within
    the fold as its nuisance prediction plus a permuted residual, so shuffling
    breaks the feature-target link while leaving the confound-target link where
    it was. This is the Freedman-Lane scheme (Freedman & Lane 1983, J. Bus. Econ.
    Stat. 1(4), 292-298; Winkler et al. 2014, NeuroImage 92, 381-397), and it is
    the conditional null that :func:`~eegtable.model.permutation_test` refuses to
    approximate by permuting raw targets.

    ``permutation_indices`` must be a permutation of every row index that maps
    each fold row to another row of the same fold; both are checked.
    """
    cov = covariates if covariates is not None else meta
    if cov is None:
        msg = "reconstruct_staged_permutation_target_for_fold requires 'covariates' or 'meta'."
        raise ValueError(msg)
    trn = train if train is not None else train_idx
    tst = test if test is not None else test_idx
    if trn is None or tst is None:
        msg = "reconstruct_staged_permutation_target_for_fold requires train and test indices."
        raise ValueError(msg)

    y_arr = np.asarray(y, dtype=np.float64)
    train_indices = np.asarray(trn, dtype=np.intp)
    test_indices = np.asarray(tst, dtype=np.intp)
    fold_mask = np.zeros(len(y_arr), dtype=np.bool_)
    fold_mask[train_indices] = True
    if np.any(fold_mask[test_indices]):
        msg = "Outer fold train and test indices must be disjoint."
        raise ValueError(msg)
    fold_mask[test_indices] = True
    fold_indices = np.flatnonzero(fold_mask)

    nuisance_fit = fit_nuisance_model(
        y=y_arr,
        covariates=cov,
        train=train_indices,
        test=test_indices,
        columns=columns,
    )
    nuisance_prediction = np.full(len(y_arr), np.nan, dtype=np.float64)
    residual = np.full(len(y_arr), np.nan, dtype=np.float64)
    nuisance_prediction[train_indices] = nuisance_fit.train_prediction
    nuisance_prediction[test_indices] = nuisance_fit.test_prediction
    residual[train_indices] = nuisance_fit.train_residual
    residual[test_indices] = nuisance_fit.test_residual

    residual_fold = residual[fold_indices]
    nuisance_prediction_fold = nuisance_prediction[fold_indices]
    if not np.all(np.isfinite(residual_fold)):
        msg = "Staged residual permutation requires finite fold residuals."
        raise ValueError(msg)
    if not np.all(np.isfinite(nuisance_prediction_fold)):
        msg = "Staged residual permutation requires finite nuisance predictions."
        raise ValueError(msg)

    source_indices = np.asarray(permutation_indices, dtype=np.intp)
    if source_indices.shape != (len(y_arr),):
        msg = "Permutation indices must contain one source row per target row."
        raise ValueError(msg)
    if not np.array_equal(np.sort(source_indices), np.arange(len(y_arr), dtype=np.intp)):
        msg = "Permutation indices must be a permutation of all row indices."
        raise ValueError(msg)
    fold_source_indices = source_indices[fold_indices]
    if not np.all(fold_mask[fold_source_indices]):
        msg = "Permutation indices map an outer-fold row outside that fold."
        raise ValueError(msg)
    shifted_residual_fold = residual[fold_source_indices]

    y_perm = y_arr.copy()
    y_perm[fold_indices] = nuisance_prediction_fold + shifted_residual_fold
    return y_perm
