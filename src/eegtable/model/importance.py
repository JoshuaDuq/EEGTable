from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np
import numpy.typing as npt
from sklearn.base import is_classifier
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance as sklearn_perm_importance
from sklearn.metrics import check_scoring
from sklearn.pipeline import FeatureUnion, Pipeline

from eegtable._validation import (
    as_real_array,
    blank_non_finite,
    validate_names,
    validate_real_array,
)
from eegtable.model._deps import require_shap
from eegtable.model.aggregate import _SubjectRScorer
from eegtable.model.crossfit import (
    _default_scoring,
    _fit_fold,
    _FittedFold,
    _validate_and_resolve_inner_groups,
    _validate_outer_folds,
    _validate_residualize_within,
    _validate_row_aligned,
)
from eegtable.model.execution import run_folds, seeded
from eegtable.model.splits import Fold, InnerSplit
from eegtable.model.transformers import Deconfounder, transform_feature_names
from eegtable.table import FeatureMeta

__all__ = [
    "Importance",
    "aggregate_by",
    "permutation_importance",
    "permutation_importance_over_folds",
    "shap_importance",
    "shap_importance_over_folds",
]


@dataclass(frozen=True)
class Importance:
    """Feature importance, with per-fold values from the fold helpers.

    Parameters
    ----------
    feature_names : tuple of str
        One name per input feature.
    values : ndarray, shape (n_features,)
        Importance of each feature. From the fold helpers, the mean over the folds
        that kept the feature; NaN when none did.
    per_fold : ndarray, shape (n_folds, n_features)
        One row per fold that scored any feature, NaN where a fold's harmonization
        dropped the feature. Empty for the single-model functions.
    """

    feature_names: tuple[str, ...]
    values: npt.NDArray[np.float64]
    per_fold: npt.NDArray[np.float64]


def aggregate_by(
    importance: Importance,
    meta: Sequence[FeatureMeta],
    field: str,
) -> dict[str, float]:
    """Sum importance values over the categories of a metadata field.

    Parameters
    ----------
    importance : Importance
        Values for exactly the features in ``meta``, matched by name. Covariates
        have no ``FeatureMeta`` and must be left out first. A NaN value raises.
    meta : sequence of FeatureMeta
        Metadata of the same features.
    field : str
        A :class:`~eegtable.FeatureMeta` field, such as ``"band"`` or ``"space"``.
        Bands are keyed by name, and a missing value by ``"unknown"``.

    Returns
    -------
    dict of str to float
        Total importance of each category.
    """
    valid_fields = {f.name for f in fields(FeatureMeta)}
    if field not in valid_fields:
        raise ValueError(f"Unknown FeatureMeta field: {field!r}")

    names = tuple(importance.feature_names)
    if len(names) != len(set(names)):
        raise ValueError("Importance contains duplicate feature names.")

    metadata_by_name = {m.name: m for m in meta}
    if len(metadata_by_name) != len(meta):
        raise ValueError("Metadata contains duplicate feature names.")

    if set(names) != set(metadata_by_name):
        raise ValueError("Importance and metadata must contain exactly the same feature names.")

    # A feature no fold could score carries NaN, which would silently turn its whole
    # category's total into NaN. Name the features instead of returning that.
    unscored = [
        name for name, value in zip(names, importance.values, strict=True) if np.isnan(value)
    ]
    if unscored:
        msg = (
            f"No fold scored {unscored[:3]}, so their importance is undefined and would make "
            "the totals NaN; drop them or lower the harmonization threshold that removed them."
        )
        raise ValueError(msg)

    totals: dict[str, float] = {}
    for name, value in zip(names, importance.values, strict=True):
        m = metadata_by_name[name]
        raw = getattr(m, field)
        key = "unknown" if raw is None else str(raw.name) if hasattr(raw, "name") else str(raw)
        totals[key] = totals.get(key, 0.0) + float(value)

    return totals


def permutation_importance(
    model: Pipeline,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    *,
    feature_names: Sequence[str] | None = None,
    n_repeats: int = 10,
    seed: int = 42,
    scoring: object = None,
) -> Importance:
    """Permutation importance of one fitted model on rows it was not fitted on.

    Wraps :func:`sklearn.inspection.permutation_importance`: the mean decrease in
    score over ``n_repeats`` shuffles of each input column. Correlated features
    can mask each other's importance.

    Parameters
    ----------
    model : Pipeline
        Fitted model.
    X, y : ndarray
        Held-out rows and their targets.
    feature_names : sequence of str, optional
        One name per column; defaults to ``feature_0``, ``feature_1``, ...
    n_repeats : int, default 10
        Shuffles of each column.
    seed : int, default 42
        Shuffle seed.
    scoring : str or callable, optional
        Score whose decrease is reported; None uses the model's ``score``.

    Returns
    -------
    Importance
        With an empty ``per_fold``.
    """
    X_arr = as_real_array(X, "X")
    y_arr = np.asarray(y)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2-D array of samples and features.")
    names = (
        tuple(f"feature_{i}" for i in range(X_arr.shape[1]))
        if feature_names is None
        else tuple(feature_names)
    )
    if len(names) != X_arr.shape[1]:
        raise ValueError("feature_names must contain one name per column of X.")
    validate_names(names, "feature_names")
    scorer = check_scoring(model, scoring=scoring)

    def score(estimator: Pipeline, values: npt.NDArray[np.float64], targets: object) -> float:
        # Each shuffle must be scored before any in-place preprocessing changes it.
        return float(scorer(estimator, values.copy(), targets))

    res = sklearn_perm_importance(
        model, X_arr, y_arr, scoring=score, n_repeats=n_repeats, random_state=seed
    )
    values = np.asarray(res.importances_mean, dtype=np.float64)
    per_fold = np.empty((0, len(names)), dtype=np.float64)
    return Importance(feature_names=names, values=values, per_fold=per_fold)


def _validate_shap_preprocessing(preprocessor: object) -> None:
    if isinstance(preprocessor, PCA):
        raise ValueError(
            "PCA mixes input features, so original-input SHAP attribution "
            "cannot be recovered from the components."
        )
    if isinstance(preprocessor, Deconfounder) and preprocessor.n_covariates > 0:
        raise ValueError(
            "Deconfounder mixes features and covariates, so original-input SHAP "
            "attribution cannot be recovered from the transformed columns."
        )
    if isinstance(preprocessor, Pipeline):
        for _, step in preprocessor.steps:
            _validate_shap_preprocessing(step)
    elif isinstance(preprocessor, ColumnTransformer):
        for name, transformer, _ in preprocessor.transformers_:
            outputs = preprocessor.output_indices_[name]
            if outputs.start != outputs.stop:
                _validate_shap_preprocessing(transformer)
    elif isinstance(preprocessor, FeatureUnion):
        for _, transformer in preprocessor.transformer_list:
            if transformer != "drop":
                _validate_shap_preprocessing(transformer)


def shap_importance(
    model: Pipeline,
    X: npt.NDArray[np.float64],
    feature_names: Sequence[str],
    *,
    seed: int = 42,
) -> Importance:
    """Mean absolute SHAP value of each input feature for one fitted pipeline.

    The final estimator is explained on the columns its preprocessing passes it,
    with a tree explainer when it has ``feature_importances_``, a linear explainer
    when it has ``coef_``, and otherwise a kernel explainer over at most 100
    background rows. The kernel explainer scores every feature, evaluating
    ``2 * n_features + 2048`` feature coalitions per explained row, so it is slow
    for large ``X``. Requires ``eegtable[importance]``.

    Parameters
    ----------
    model : Pipeline
        Fitted pipeline whose steps report their output feature names. Steps that
        mix input columns, such as PCA or an active Deconfounder, raise:
        original-input SHAP attribution cannot be recovered from those outputs.
    X : ndarray
        Rows to explain.
    feature_names : sequence of str
        Unique name of each column of ``X``.
    seed : int, default 42
        Seed for drawing the kernel explainer's background rows and feature coalitions.

    Returns
    -------
    Importance
        Features the pipeline dropped score 0. For a binary classifier with
        per-class SHAP values, class 1's are used. ``per_fold`` is empty.
    """
    for _, step in model.steps[:-1]:
        _validate_shap_preprocessing(step)
    require_shap()
    import shap

    X_arr = as_real_array(X, "X").copy()
    steps = list(model.steps)
    _, final_estimator = steps[-1]
    X_trans = X_arr
    for _, step in steps[:-1]:
        if hasattr(step, "transform"):
            X_trans = step.transform(X_trans)
    # SHAP explains the columns the final estimator sees, which the preprocessing steps may
    # have dropped; each explained column is mapped back to the input feature it came from.
    explained = transform_feature_names(steps[:-1], feature_names)
    if len(explained) != X_trans.shape[1]:
        msg = (
            f"The pipeline outputs {X_trans.shape[1]} columns but reports {len(explained)} "
            "names, so SHAP values cannot be matched to features."
        )
        raise ValueError(msg)
    positions = _input_positions(explained, feature_names)

    rng = np.random.default_rng(seed)
    if hasattr(final_estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(final_estimator)
        shap_values = explainer.shap_values(X_trans)
    elif hasattr(final_estimator, "coef_"):
        explainer = shap.LinearExplainer(final_estimator, X_trans)
        shap_values = explainer.shap_values(X_trans)
    else:
        predict_fn = (
            final_estimator.predict_proba
            if hasattr(final_estimator, "predict_proba")
            else final_estimator.predict
        )
        bg_size = min(100, len(X_trans))
        bg_indices = rng.choice(len(X_trans), bg_size, replace=False)
        background = X_trans[bg_indices]
        # shap's default l1_reg is a lasso that keeps at most 10 features per row (0.47+) or
        # an AIC lasso (older), zeroing the rest. Without it the weighted least squares
        # needs more coalitions than features; "auto" is shap's 2 * n_features + 2048.
        with seeded(seed, 0):
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_trans, nsamples="auto", l1_reg=False)

    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim == 3:
        # SHAP >= 0.45 places the output/class axis last.
        shap_values = (
            shap_values[:, :, 1]
            if is_classifier(final_estimator) and shap_values.shape[2] == 2
            else np.mean(np.abs(shap_values), axis=2)
        )
    if shap_values.shape != X_trans.shape:
        raise ValueError("SHAP values must align with the explained samples and features.")

    # A feature the pipeline dropped has no effect on the prediction, so its value is 0.
    names = tuple(feature_names)
    values = np.zeros(len(names), dtype=np.float64)
    values[positions] = np.mean(np.abs(shap_values), axis=0)
    per_fold = np.empty((0, len(names)), dtype=np.float64)
    return Importance(feature_names=names, values=values, per_fold=per_fold)


def _input_positions(
    explained: Sequence[str], feature_names: Sequence[str]
) -> npt.NDArray[np.intp]:
    index = {name: i for i, name in enumerate(feature_names)}
    if len(index) != len(feature_names):
        raise ValueError("feature_names must be unique to attribute SHAP values to them.")
    if len(set(explained)) != len(explained):
        raise ValueError("Output feature names must be unique to attribute SHAP values to inputs.")
    unknown = [name for name in explained if name not in index]
    if unknown:
        msg = (
            f"The pipeline outputs {unknown[:3]}, which are not input features, so their SHAP "
            "values belong to no single feature; remove steps that mix features, such as PCA."
        )
        raise ValueError(msg)
    return np.array([index[name] for name in explained], dtype=np.intp)


def _fold_fitter(
    folds: Sequence[Fold],
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    groups: npt.NDArray[np.object_],
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    *,
    inner: InnerSplit,
    seed: int,
    runs: npt.NDArray[np.object_] | None,
    harmonization: str | None,
    covariates: npt.NDArray[np.float64] | None,
    residualize_on: Sequence[str],
    residualize_within: Literal["subject"] | None,
    scoring: object,
    refit: str | bool | None,
) -> Callable[[Fold], _FittedFold]:
    # Importance fits each fold exactly as cross-fitting does, so it explains the model that
    # was evaluated rather than one tuned or trained differently. That includes the fold
    # checks: importance is an entry point of its own, and folds reaching it never passed
    # through cross_fit_*.
    validate_real_array(X, "X")
    _validate_outer_folds(folds, len(X), groups, runs)
    _validate_row_aligned(len(X), y, covariates)
    _validate_residualize_within(residualize_within, residualize_on)
    inner_groups_all = _validate_and_resolve_inner_groups(folds, inner, groups, runs)
    task = _task(pipeline)
    scoring = _default_scoring(task, scoring)

    def fit(f: Fold) -> _FittedFold:
        return _fit_fold(
            task,
            f,
            X,
            y,
            groups,
            inner_groups_all,
            pipeline,
            grid,
            inner=inner,
            seed=seed,
            harmonization=harmonization,
            covariates=covariates,
            residualize_on=residualize_on,
            residualize_within=residualize_within,
            scoring=scoring,
            refit=refit,
        )

    return fit


def _task(pipeline: Pipeline) -> str:
    return "classification" if is_classifier(pipeline) else "regression"


def _importance_scoring(scoring: object, refit: str | bool | None) -> object:
    # Importance is the drop in the metric the model was selected on.
    if isinstance(scoring, Mapping):
        if isinstance(refit, str) and refit in scoring:
            return scoring[refit]
        msg = "With multi-metric scoring, refit must name the metric importance is measured on."
        raise ValueError(msg)
    return scoring


def _fold_values(
    values: npt.NDArray[np.float64], features: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    # A feature harmonized out of a fold was never available to that fold's model.
    fold_imp = np.full(features.size, np.nan, dtype=np.float64)
    fold_imp[features] = values
    return fold_imp


def _combine_folds(
    results: Sequence[npt.NDArray[np.float64]],
    n_folds: int,
    min_complete_fraction: float,
    label: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # run_folds propagates a failed fold, so a result here is always an array; a fold counts
    # as unsuccessful only when no feature survived it.
    successful = [r for r in results if np.any(np.isfinite(r))]
    rate = len(successful) / n_folds if n_folds else 0.0
    if rate < min_complete_fraction:
        msg = f"Insufficient successful folds for {label} ({len(successful)}/{n_folds})."
        raise ValueError(msg)

    per_fold = np.vstack(successful)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=r"Mean of empty slice")
        # The same reading of "missing" the fold filter above used: one feature's
        # non-finite value drops out of the average rather than becoming it.
        values = np.asarray(np.nanmean(blank_non_finite(per_fold), axis=0), dtype=np.float64)
    return per_fold, values


def shap_importance_over_folds(
    folds: Sequence[Fold],
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_],
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    feature_names: Sequence[str],
    *,
    inner: InnerSplit,
    seed: int = 42,
    runs: npt.NDArray[np.object_] | None = None,
    outer_n_jobs: int = 1,
    harmonization: str | None = None,
    covariates: npt.NDArray[np.float64] | None = None,
    residualize_on: Sequence[str] = (),
    residualize_within: Literal["subject"] | None = None,
    scoring: object = None,
    refit: str | bool | None = None,
    min_complete_fraction: float = 0.5,
) -> Importance:
    """SHAP importance on each outer fold's held-out rows, averaged over folds.

    Every fold is fitted exactly as cross-fitting fits it, with the same fold
    checks, tuning, harmonization and residualization, and its held-out rows are
    explained with :func:`shap_importance`. Pass the settings used for evaluation
    so the explained models are the evaluated ones. A classifier ``pipeline`` is
    fitted as a classification task.

    Parameters
    ----------
    folds, X, y, groups, pipeline, grid
        As for :func:`cross_fit_regression` or :func:`cross_fit_classification`.
    feature_names : sequence of str
        Unique name of each column of ``X``.
    inner, runs, outer_n_jobs, harmonization, scoring, refit
        As for the cross-fitting functions.
    seed : int, default 42
        As for the cross-fitting functions; fold ``k`` is explained with seed
        ``seed + k``.
    covariates, residualize_on, residualize_within
        As for :func:`cross_fit_regression`.
    min_complete_fraction : float, default 0.5
        Fraction of folds that must give a finite value for at least one feature;
        fewer raises. A fold that fails to fit raises regardless.

    Returns
    -------
    Importance
    """
    require_shap()
    fit = _fold_fitter(
        folds,
        X,
        y,
        groups,
        pipeline,
        grid,
        inner=inner,
        seed=seed,
        runs=runs,
        harmonization=harmonization,
        covariates=covariates,
        residualize_on=residualize_on,
        residualize_within=residualize_within,
        scoring=scoring,
        refit=refit,
    )
    all_names = list(feature_names)

    def _fold_importance(f: Fold) -> npt.NDArray[np.float64]:
        fitted = fit(f)
        retained = [name for name, keep in zip(all_names, fitted.features, strict=True) if keep]
        imp = shap_importance(fitted.model, fitted.X_test, retained, seed=seed + f.index)
        return _fold_values(imp.values, fitted.features)

    results = run_folds(folds, _fold_importance, outer_n_jobs=outer_n_jobs)
    per_fold, values = _combine_folds(results, len(folds), min_complete_fraction, "SHAP")
    return Importance(feature_names=tuple(all_names), values=values, per_fold=per_fold)


def permutation_importance_over_folds(
    folds: Sequence[Fold],
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    groups: npt.NDArray[np.object_],
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    *,
    inner: InnerSplit,
    feature_names: Sequence[str] | None = None,
    n_repeats: int = 10,
    seed: int = 42,
    runs: npt.NDArray[np.object_] | None = None,
    outer_n_jobs: int = 1,
    harmonization: str | None = None,
    covariates: npt.NDArray[np.float64] | None = None,
    residualize_on: Sequence[str] = (),
    residualize_within: Literal["subject"] | None = None,
    scoring: object = None,
    refit: str | bool | None = None,
    min_complete_fraction: float = 0.5,
) -> Importance:
    """Held-out permutation importance of each outer fold's model, averaged over folds.

    Every fold is fitted exactly as cross-fitting fits it, with the same fold
    checks, tuning, harmonization and residualization; the input columns of its
    held-out rows are then shuffled with :func:`permutation_importance`. The value
    is the mean decrease in the score the model was selected on: subject-level
    ``r`` for regressors by default, the estimator's ``score`` (accuracy) for
    classifiers. Pass the settings used for evaluation so the explained models are
    the evaluated ones.

    Parameters
    ----------
    folds, X, y, groups, pipeline, grid
        As for :func:`cross_fit_regression` or :func:`cross_fit_classification`.
    inner : InnerSplit
        As for the cross-fitting functions.
    feature_names : sequence of str, optional
        One name per column of ``X``, so a score can be matched after fold-local
        column drops; defaults to ``feature_0``, ``feature_1``, ...
    n_repeats : int, default 10
        Shuffles of each column.
    seed : int, default 42
        As for the cross-fitting functions; fold ``k`` shuffles with seed
        ``seed + k``.
    runs, outer_n_jobs, harmonization
        As for the cross-fitting functions.
    covariates, residualize_on, residualize_within
        As for :func:`cross_fit_regression`.
    scoring, refit
        As for the cross-fitting functions; with multi-metric ``scoring``,
        ``refit`` also names the metric the importance is measured on.
    min_complete_fraction : float, default 0.5
        Fraction of folds that must give a finite value for at least one feature;
        fewer raises. A fold that fails to fit raises regardless.

    Returns
    -------
    Importance
    """
    importance_scoring = _importance_scoring(_default_scoring(_task(pipeline), scoring), refit)
    fit = _fold_fitter(
        folds,
        X,
        y,
        groups,
        pipeline,
        grid,
        inner=inner,
        seed=seed,
        runs=runs,
        harmonization=harmonization,
        covariates=covariates,
        residualize_on=residualize_on,
        residualize_within=residualize_within,
        scoring=scoring,
        refit=refit,
    )
    all_names = (
        list(feature_names)
        if feature_names is not None
        else [f"feature_{i}" for i in range(X.shape[1])]
    )

    def _fold_importance(f: Fold) -> npt.NDArray[np.float64]:
        fitted = fit(f)
        retained = [name for name, keep in zip(all_names, fitted.features, strict=True) if keep]
        fold_scoring = importance_scoring
        if isinstance(importance_scoring, _SubjectRScorer):
            # The permuted rows stay in test order, so each keeps its subject.
            test_groups = groups[f.test]

            def fold_scoring(
                estimator: object, X_: npt.NDArray[np.float64], y_: npt.NDArray[np.float64]
            ) -> float:
                return importance_scoring(estimator, X_, y_, test_groups)

        imp = permutation_importance(
            fitted.model,
            fitted.X_test,
            fitted.y_test,
            feature_names=retained,
            n_repeats=n_repeats,
            seed=seed + f.index,
            scoring=fold_scoring,
        )
        return _fold_values(imp.values, fitted.features)

    results = run_folds(folds, _fold_importance, outer_n_jobs=outer_n_jobs)
    per_fold, values = _combine_folds(
        results, len(folds), min_complete_fraction, "permutation importance"
    )
    return Importance(feature_names=tuple(all_names), values=values, per_fold=per_fold)
