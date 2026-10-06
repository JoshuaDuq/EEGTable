from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    VotingClassifier,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, NuSVC

from eegtable.model import _deps as _deps
from eegtable.model.aggregate import _SubjectRScorer
from eegtable.model.execution import seeded
from eegtable.model.splits import InnerSplit, inner_cv

__all__ = [
    "FoldFitError",
    "TunedFit",
    "fit_untuned",
    "tune",
]


class FoldFitError(ValueError, RuntimeError):
    """A fold's estimator could not be fitted: a fault, not a measurement.

    Subclasses ``ValueError`` so existing handlers keep working, and ``RuntimeError`` so a
    caller can count fit failures separately from designs that cannot be fitted at all.
    """


@dataclass(frozen=True)
class TunedFit:
    """Best candidate of :func:`tune`.

    Parameters
    ----------
    estimator : Pipeline
        The chosen candidate, refitted on all training rows of the fold.
    best_params : dict
        Its grid values.
    """

    estimator: Pipeline
    best_params: dict[str, object]


def _raise_for_nonfinite_grid_search_scores(grid: GridSearchCV, fold: int) -> None:
    cv_results = getattr(grid, "cv_results_", None)
    if not isinstance(cv_results, dict):
        return

    bad_score_keys: list[str] = []
    for key, values in cv_results.items():
        is_test_score = key.startswith("mean_test") or (key.startswith("split") and "_test" in key)
        if not is_test_score:
            continue
        scores: npt.NDArray[np.float64] = np.ma.asarray(values, dtype=float).filled(np.nan)
        if scores.size and not np.all(np.isfinite(scores)):
            bad_score_keys.append(str(key))

    if bad_score_keys:
        msg = f"Fold {fold}: non-finite inner CV test scores: {', '.join(bad_score_keys)}."
        raise FoldFitError(msg)


def _assign_random_state(estimator: object, seed: int) -> None:
    if hasattr(estimator, "get_params") and hasattr(estimator, "set_params"):
        random_state_keys = [
            key
            for key in estimator.get_params(deep=True)
            if key == "random_state" or key.endswith("__random_state")
        ]
        if random_state_keys:
            estimator.set_params(**{key: seed for key in random_state_keys})


def _validate_boosting_early_stopping(pipeline: Pipeline) -> None:
    for component in (pipeline, *pipeline.get_params(deep=True).values()):
        if (
            isinstance(component, (HistGradientBoostingRegressor, HistGradientBoostingClassifier))
            and component.early_stopping
        ):
            raise ValueError(
                "Grouped fitting requires early_stopping=False for histogram gradient "
                "boosting; tune max_iter through group-disjoint inner folds."
            )


def _validate_grouped_estimator(pipeline: Pipeline) -> None:
    _validate_boosting_early_stopping(pipeline)
    components = (pipeline, *pipeline.get_params(deep=True).values())
    if any(isinstance(component, CalibratedClassifierCV) for component in components):
        raise ValueError(
            "Grouped fitting requires group-disjoint calibration with preprocessing fitted "
            "inside each calibration split. CalibratedClassifierCV is unsupported here "
            "because calibration groups are not routed through this workflow."
        )
    for component in components:
        if (
            isinstance(component, (SVC, NuSVC))
            and isinstance(component.probability, (bool, np.bool_))
            and bool(component.probability)
        ):
            raise ValueError(
                "SVC probability=True uses internal trial-wise cross-validation; grouped "
                "fitting requires probability=False. Group-disjoint probability calibration "
                "must be performed in a separate calibration workflow."
            )
        if isinstance(component, VotingClassifier) and component.voting == "soft":
            for _, member in component.estimators:
                if isinstance(member, BaseEstimator):
                    members = (member, *member.get_params(deep=True).values())
                    if any(isinstance(model, (SVC, NuSVC)) for model in members):
                        raise ValueError(
                            "Soft voting with SVC requires group-disjoint probability "
                            "calibration, which grouped fitting does not provide. Use an "
                            "explicit hard vote or probability estimators without internal CV."
                        )


def tune(
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    inner_groups_train: npt.NDArray[np.object_],
    *,
    split: InnerSplit,
    seed: int,
    fold: int,
    n_jobs: int = 1,
    scoring: object = None,
    refit: str | bool | None = None,
) -> TunedFit:
    """Grid-search a pipeline over group-disjoint inner splits of one training fold.

    Runs scikit-learn's ``GridSearchCV`` on splits from :func:`inner_cv` and refits
    the best candidate on all of ``X_train``. Every candidate is checked against
    the grouped-fitting restrictions first. A failing candidate or a non-finite
    inner score raises :class:`FoldFitError`. The cross-fitting functions call
    this; call it directly only in a fold loop of your own.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted pipeline; it is cloned, not modified.
    grid : mapping of str to sequence
        Parameter grid.
    X_train, y_train : ndarray
        Training rows of the outer fold only.
    inner_groups_train : ndarray
        Label of each training row that the inner splits keep disjoint: subjects,
        or runs inside a within-subject fold. At least two distinct labels.
    split : InnerSplit
        Grouping, stratification and requested split count; the count is capped
        at the number of distinct labels.
    seed : int
        Assigned to every ``random_state`` in the pipeline; with ``fold`` it also
        seeds stratified splits and the global generators during the search.
    fold : int
        Outer fold index, used in error messages and to vary seeds by fold.
    n_jobs : int, default 1
        ``GridSearchCV`` parallel jobs.
    scoring : str, callable or mapping, optional
        Selection score; None uses the estimator's ``score``.
        :func:`subject_r_scorer` is refused because ``GridSearchCV`` cannot pass it
        the validation subjects.
    refit : str or bool, optional
        With multi-metric ``scoring``, the metric that chooses the candidate.
        False is refused.

    Returns
    -------
    TunedFit
    """
    if refit is False:
        raise ValueError("refit=False cannot return a fitted outer-fold model.")
    _validate_grouped_estimator(pipeline)
    for parameters in ParameterGrid(dict(grid)):
        try:
            candidate = clone(pipeline).set_params(**parameters)
        except ValueError as exc:
            raise FoldFitError(f"Fold {fold}: inner CV failed: {exc}") from exc
        _validate_grouped_estimator(candidate)
    chosen = scoring[refit] if isinstance(scoring, Mapping) and isinstance(refit, str) else scoring
    if isinstance(chosen, _SubjectRScorer):
        raise ValueError(
            "subject_r_scorer needs the subject of every validation row, which GridSearchCV "
            "does not pass to a scorer; tune through cross_fit_regression instead."
        )

    groups_arr = np.asarray(inner_groups_train)
    n_unique = len(np.unique(groups_arr))
    if n_unique < 2:
        msg = f"Fold {fold}: inner CV requires at least 2 unique groups, got {n_unique}."
        raise ValueError(msg)

    with seeded(seed, fold):
        y_strat = np.asarray(y_train, dtype=np.intp) if split.stratified else None
        inner_seed = seed + (fold - 1 if fold > 0 else 0)
        cv = inner_cv(groups_arr, split, y_train=y_strat, random_state=inner_seed)

        pipe_clone = cast(Pipeline, clone(pipeline))
        _assign_random_state(pipe_clone, seed)

        refit_param: str | bool = refit if refit is not None else True
        gs = GridSearchCV(
            estimator=pipe_clone,
            param_grid=grid,
            scoring=scoring,
            cv=cv,
            refit=refit_param,
            error_score="raise",
            n_jobs=n_jobs,
        )

        try:
            gs.fit(X_train, y_train, groups=groups_arr)
        except Exception as exc:
            msg = f"Fold {fold}: inner CV failed: {exc}"
            raise FoldFitError(msg) from exc

    _raise_for_nonfinite_grid_search_scores(gs, fold)
    return TunedFit(
        estimator=cast(Pipeline, gs.best_estimator_),
        best_params=dict(gs.best_params_),
    )


def fit_untuned(
    pipeline: Pipeline,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    *,
    seed: int,
    fold: int = 0,
) -> Pipeline:
    """Fit a clone of a pipeline as configured, without a search.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted pipeline; it is cloned, not modified.
    X, y : ndarray
        Training rows.
    seed : int
        Assigned to every ``random_state`` in the clone. ``seed + fold`` seeds the
        global NumPy and ``random`` generators during the fit; their previous state
        is restored afterwards.
    fold : int, default 0
        Fold index, used in error messages and seeding.

    Returns
    -------
    Pipeline
        The fitted clone. A failed fit raises :class:`FoldFitError`.
    """
    with seeded(seed, fold):
        pipe_clone = cast(Pipeline, clone(pipeline))
        _assign_random_state(pipe_clone, seed)
        try:
            pipe_clone.fit(X, y)
        except Exception as exc:
            msg = f"Fold {fold}: fit failed: {exc}"
            raise FoldFitError(msg) from exc
    return pipe_clone
