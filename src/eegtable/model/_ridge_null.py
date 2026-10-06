"""Every permutation draw of a ridge cross-fit at once.

What a ridge pipeline fits before its regressor never sees the target, and ridge
predictions, target residualization and the Freedman-Lane rebuild are all linear in the
target. Each fold and inner split therefore transforms its features once, then
scikit-learn Ridge fits all permutation targets together for each penalty. The
batched null uses the same solver as the observed fit, to rounding error.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from eegtable.model._nuisance import fit_coefficients
from eegtable.model.aggregate import AggregationConfig, _center_within, _SubjectRScorer
from eegtable.model.crossfit import _chosen_scorer
from eegtable.model.design import harmonize_fold
from eegtable.model.execution import run_folds, seeded
from eegtable.model.residualize import (
    _design_matrix,
    residualize_targets,
    residualize_within_subjects,
)
from eegtable.model.splits import Fold, InnerSplit, inner_cv
from eegtable.model.transformers import (
    Deconfounder,
    DropAllNaNColumns,
    MissingnessThreshold,
    ReplaceInfWithNaN,
    VarianceThreshold,
    _CovariateImputer,
)
from eegtable.model.tuning import _assign_random_state

# Exact types, not subclasses: a subclass could start using the target.
_LABEL_FREE: tuple[type, ...] = (
    ReplaceInfWithNaN,
    DropAllNaNColumns,
    MissingnessThreshold,
    VarianceThreshold,
    Deconfounder,
    _CovariateImputer,
    SimpleImputer,
    StandardScaler,
    PCA,
    FunctionTransformer,
)
# Solvers with the closed-form solution; the iterative ones stop at a tolerance.
_EXACT_SOLVERS = ("auto", "cholesky", "svd")
_DRAWS = 256


def _label_free(step: object) -> bool:
    # Containers are opened and every step inside is checked; any other step must be one
    # of the exact types above.
    if isinstance(step, str):
        return step in ("drop", "passthrough")
    if type(step) is Pipeline:
        return all(_label_free(inner) for _, inner in cast(Pipeline, step).steps)
    if type(step) is ColumnTransformer:
        transformer = cast(ColumnTransformer, step)
        return _label_free(transformer.remainder) and all(
            _label_free(inner) for _, inner, _ in transformer.transformers
        )
    return type(step) in _LABEL_FREE


def ridge_penalty(
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    scoring: object,
    refit: str | bool | None,
    metric_fn: object,
) -> str | None:
    """The grid key of the ridge penalty when the null has a closed form, else None."""
    if (
        type(pipeline) is not Pipeline
        or metric_fn is not None
        # The batched statistic cannot validate additional metrics on each draw.
        or (isinstance(scoring, Mapping) and len(scoring) > 1)
        or not isinstance(_chosen_scorer(scoring, refit), _SubjectRScorer)
    ):
        return None
    name, regressor = pipeline.steps[-1]
    if (
        type(regressor) is not Ridge
        or not regressor.fit_intercept
        or regressor.positive
        or regressor.solver not in _EXACT_SOLVERS
        or np.ndim(regressor.alpha) != 0
    ):
        return None
    if not all(_label_free(step) for _, step in pipeline.steps[:-1]):
        return None
    key = f"{name}__alpha"
    return key if set(grid) <= {key} else None


def ridge_null(
    folds: Sequence[Fold],
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    sources: npt.NDArray[np.intp],
    groups: npt.NDArray[np.object_],
    inner_groups: npt.NDArray[np.object_],
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    penalty: str,
    *,
    inner: InnerSplit,
    seed: int,
    scoring: object,
    refit: str | bool | None,
    aggregation: AggregationConfig,
    harmonization: str | None,
    covariates: npt.NDArray[np.float64] | None,
    residualize_on: Sequence[str],
    residualize_within: Literal["subject"] | None,
    outer_n_jobs: int,
) -> npt.NDArray[np.float64]:
    """The statistic of every draw, one row of ``sources`` each, as permutation_test refits it."""
    scorer = cast(_SubjectRScorer, _chosen_scorer(scoring, refit))
    alphas = np.asarray(grid[penalty] if grid else [pipeline.steps[-1][1].alpha], dtype=np.float64)
    if not np.isfinite(alphas).all() or np.any(alphas <= 0.0):
        raise ValueError(
            "Batched ridge permutation inference requires finite, strictly positive alpha; "
            "unregularized fits can change when targets are fitted together."
        )
    n_draws = sources.shape[0]
    # Draws are taken a block at a time, so memory does not grow with n_permutations.
    blocks = np.array_split(np.arange(n_draws), max(1, -(-n_draws // _DRAWS)))

    def fold_predictions(
        f: Fold,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        def split(fit: npt.NDArray[np.intp], held_out: npt.NDArray[np.intp]) -> _Split:
            return _Split(
                X,
                fit,
                held_out,
                groups=groups,
                pipeline=pipeline,
                seed=seed,
                fold=f.index,
                harmonization=harmonization,
                covariates=covariates,
                residualize_on=residualize_on,
                residualize_within=residualize_within,
            )

        inner_splits: list[_Split] = []
        if len(alphas) > 1:
            # Reuse the feature transforms of the same inner splits as cross-fitting.
            train_groups = inner_groups[f.train]
            splitter = inner_cv(train_groups, inner, random_state=seed + max(f.index - 1, 0))
            inner_splits = [
                split(f.train[local_train], f.train[local_valid])
                for local_train, local_valid in splitter.split(f.train, groups=train_groups)
            ]
        outer = split(f.train, f.test)
        rebuild = (
            _FreedmanLane(f, y, sources, covariates, groups, residualize_on, residualize_within)
            if residualize_on
            else None
        )

        predicted = np.empty((f.test.size, n_draws))
        truth = np.empty((f.test.size, n_draws))
        for block in blocks:
            targets = y[sources[block]].T if rebuild is None else rebuild.targets(block)
            chosen = np.zeros(block.size, dtype=np.intp)
            if inner_splits:
                scores = np.zeros((len(alphas), block.size))
                for inner_split in inner_splits:
                    fitted, held = inner_split.targets(targets)
                    for a, prediction in enumerate(inner_split.predict(fitted, alphas)):
                        scores[a] += _subject_r(
                            prediction, held, groups[inner_split.held_out], None, scorer.config
                        )
                # The mean over splits, and the first of equal scores, as the grid search has it.
                chosen = np.argmax(scores / len(inner_splits), axis=0)
            fitted, held = outer.targets(targets)
            truth[:, block] = held
            for a, prediction in enumerate(outer.predict(fitted, alphas)):
                picked = chosen == a
                predicted[:, block[picked]] = prediction[:, picked]
        return predicted, truth

    results = run_folds(folds, fold_predictions, outer_n_jobs=outer_n_jobs)
    rows = np.concatenate([f.test for f in folds])
    predicted = np.concatenate([p for p, _ in results])
    truth = np.concatenate([t for _, t in results])
    fold_ids = np.concatenate([np.full(f.test.size, f.index) for f in folds])
    if not (np.isfinite(predicted).all() and np.isfinite(truth).all()):
        raise ValueError("Permutation statistic requires finite predictions.")
    return _subject_r(predicted, truth, groups[rows], fold_ids, aggregation)


class _Split:
    """One transformed split, with permutation targets fitted together by Ridge."""

    def __init__(
        self,
        X: npt.NDArray[np.float64],
        fit: npt.NDArray[np.intp],
        held_out: npt.NDArray[np.intp],
        *,
        groups: npt.NDArray[np.object_],
        pipeline: Pipeline,
        seed: int,
        fold: int,
        harmonization: str | None,
        covariates: npt.NDArray[np.float64] | None,
        residualize_on: Sequence[str],
        residualize_within: Literal["subject"] | None,
    ) -> None:
        self.fit, self.held_out = fit, held_out
        self._groups, self._covariates, self._within = groups, covariates, residualize_within
        # residualize_targets' own column naming.
        self._columns = tuple(str(c).strip() for c in residualize_on if str(c).strip())
        X_fit, X_held = X[fit], X[held_out]
        kept = np.ones(X.shape[1], dtype=np.bool_)
        if harmonization is not None:
            X_fit, X_held, kept = harmonize_fold(
                X_fit, X_held, groups[fit], mode=harmonization, n_covariates=0
            )
        self._designs: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None
        if self._columns:
            assert covariates is not None
            if residualize_within is None:
                self._designs = (
                    _design_matrix(covariates, fit, self._columns, check_rank=True),
                    _design_matrix(covariates, held_out, self._columns, check_rank=False),
                )
            else:
                X_fit, X_held = residualize_within_subjects(
                    X[:, kept], covariates, groups, fit, held_out, columns=self._columns
                )
        Z_fit, Z_held = np.asarray(X_fit, dtype=np.float64), np.asarray(X_held, dtype=np.float64)
        if len(pipeline.steps) > 1:
            with seeded(seed, fold):
                steps = cast(Pipeline, clone(Pipeline(pipeline.steps[:-1])))
                _assign_random_state(steps, seed)
                Z_fit = np.asarray(steps.fit_transform(X_fit), dtype=np.float64)
                Z_held = np.asarray(steps.transform(X_held), dtype=np.float64)
        self._features, self._held_features = Z_fit, Z_held
        self._regressor = clone(pipeline.steps[-1][1])

    def targets(
        self, targets: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The fit and held-out targets of every draw, residualized as cross-fitting would."""
        if not self._columns:
            return targets[self.fit], targets[self.held_out]
        assert self._covariates is not None
        if self._designs is None:
            return residualize_within_subjects(
                targets,
                self._covariates,
                self._groups,
                self.fit,
                self.held_out,
                columns=self._columns,
            )
        fit_design, held_design = self._designs
        coefficients = fit_coefficients(fit_design, targets[self.fit])
        return (
            targets[self.fit] - fit_design @ coefficients,
            targets[self.held_out] - held_design @ coefficients,
        )

    def predict(
        self, fitted: npt.NDArray[np.float64], alphas: npt.NDArray[np.float64]
    ) -> Iterator[npt.NDArray[np.float64]]:
        for alpha in alphas:
            model = clone(self._regressor).set_params(alpha=float(alpha))
            model.fit(self._features.copy(), fitted)
            yield np.asarray(model.predict(self._held_features), dtype=np.float64)


class _FreedmanLane:
    """permutation_test's _freedman_lane_target, for a block of draws at a time."""

    def __init__(
        self,
        fold: Fold,
        y: npt.NDArray[np.float64],
        sources: npt.NDArray[np.intp],
        covariates: npt.NDArray[np.float64] | None,
        groups: npt.NDArray[np.object_],
        columns: Sequence[str],
        within: Literal["subject"] | None,
    ) -> None:
        assert covariates is not None
        self._rows = np.concatenate([fold.train, fold.test])
        in_fold = np.zeros(len(y), dtype=np.bool_)
        in_fold[self._rows] = True
        if not np.all(in_fold[sources[:, self._rows]]):
            raise ValueError(
                f"Fold {fold.index}: the permutation moves trials outside the fold, whose "
                "nuisance fit cannot then be kept; use a scheme that exchanges trials within "
                "its runs."
            )
        residual = np.full(len(y), np.nan)
        if within is None:
            residual[fold.train], residual[fold.test] = residualize_targets(
                y, covariates, fold.train, fold.test, columns=columns
            )
        else:
            residual[fold.train], residual[fold.test] = residualize_within_subjects(
                y, covariates, groups, fold.train, fold.test, columns=columns
            )
        self._y, self._residual, self._sources = y, residual, sources

    def targets(self, block: npt.NDArray[np.intp]) -> npt.NDArray[np.float64]:
        rows, residual = self._rows, self._residual
        targets = np.repeat(self._y[:, None], block.size, axis=1)
        kept = (self._y[rows] - residual[rows])[:, None]
        targets[rows] = kept + residual[self._sources[block][:, rows]].T
        return targets


def _subject_r(
    predicted: npt.NDArray[np.float64],
    truth: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_],
    folds: npt.NDArray[np.intp] | None,
    config: AggregationConfig,
) -> npt.NDArray[np.float64]:
    # subject_level_r with undefined="zero", computed for every column (draw) at once.
    z_values, weights = [], []
    for subject in np.unique(groups):
        cell = groups == subject
        p, t = predicted[cell], truth[cell]
        n_trials = int(cell.sum())
        if n_trials < 3:
            raise ValueError(
                f"Subject-level r needs at least 3 held-out trials per subject; {subject} has "
                f"{n_trials}. Hold out more trials or pass scoring."
            )
        # A subject scored by several fold models is centred within each, as the observed is.
        labels = np.zeros(n_trials, dtype=np.intp) if folds is None else folds[cell]
        constant = (np.ptp(p, axis=0) == 0.0) | (np.ptp(t, axis=0) == 0.0)
        p, t = _center_within(p, labels), _center_within(t, labels)
        n_trials -= len(np.unique(labels)) - 1
        denominator = np.sqrt((p * p).sum(axis=0) * (t * t).sum(axis=0))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(constant | (denominator == 0.0), 0.0, (p * t).sum(axis=0) / denominator)
        z_values.append(np.arctanh(np.clip(r, -0.999999, 0.999999)))
        if config.subject_weighting == "trial_count" and n_trials < 4:
            raise ValueError("Trial-count weighting needs more than 3 trials per subject.")
        weights.append(n_trials - 3.0 if config.subject_weighting == "trial_count" else 1.0)
    weight = np.asarray(weights)[:, None]
    return np.asarray(np.tanh((np.asarray(z_values) * weight).sum(axis=0) / weight.sum()))
