from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from eegtable.model.aggregate import subject_r_scorer
from eegtable.model.estimators import (
    elasticnet_grid,
    elasticnet_pipeline,
    ridge_grid,
    ridge_pipeline,
)
from eegtable.model.scoring import scoring_dict
from eegtable.model.splits import InnerSplit
from eegtable.model.transformers import PreprocessingConfig
from eegtable.model.tuning import FoldFitError, fit_untuned, tune

PIPE = Pipeline([("regressor", DummyRegressor(strategy="mean"))])
GRID = {"regressor__strategy": ["mean", "median"]}
BY_SUBJECT = InnerSplit(grouping="subject", n_splits=2)
BY_RUN = InnerSplit(grouping="run", n_splits=2)
TWO_SUBJECTS = np.array(["s1"] * 4 + ["s2"] * 4, dtype=object)
TWO_RUNS = np.array(["r1"] * 4 + ["r2"] * 4, dtype=object)
X8 = np.arange(8.0).reshape(-1, 1)
Y8 = np.arange(8.0)


def test_tuning_refuses_a_training_fold_with_one_group() -> None:
    # With one group the inner split cannot be group-disjoint, so the chosen
    # hyperparameters would be selected on data from the subject being predicted.
    with pytest.raises(ValueError, match="at least 2"):
        tune(
            PIPE,
            GRID,
            X8,
            Y8,
            np.array(["s1"] * 8, dtype=object),
            split=BY_SUBJECT,
            seed=0,
            fold=1,
        )


def test_a_within_subject_fold_tunes_on_runs_not_subjects() -> None:
    # The training block is one subject, so runs are the only split that exists. This is
    # what upstream does at orchestration.py:868, and the reason tune takes an InnerSplit
    # rather than "the groups".
    assert tune(PIPE, GRID, X8, Y8, TWO_RUNS, split=BY_RUN, seed=0, fold=1).best_params


def test_the_fold_is_named_when_tuning_fails() -> None:
    with pytest.raises(ValueError, match="Fold 7"):
        tune(
            PIPE,
            GRID,
            X8,
            Y8,
            np.array(["s1"] * 8, dtype=object),
            split=BY_SUBJECT,
            seed=0,
            fold=7,
        )


def test_tuning_does_not_fall_back_to_an_untuned_fit() -> None:
    # A failed inner search must surface. Falling back to a default fit would report a
    # score for a model nobody selected, indistinguishable from a tuned one downstream.
    with pytest.raises(ValueError):
        tune(
            PIPE,
            {"regressor__nonexistent": [1]},
            X8,
            Y8,
            TWO_SUBJECTS,
            split=BY_SUBJECT,
            seed=0,
            fold=1,
        )


def test_within_subject_inner_cv_failure_raises_instead_of_default_fit() -> None:
    with pytest.raises(ValueError, match="inner CV failed"):
        tune(
            PIPE,
            {"regressor__nonexistent": [1]},
            X8,
            Y8,
            TWO_RUNS,
            split=BY_RUN,
            seed=0,
            fold=1,
        )


class _NaNScoringRegressor(DummyRegressor):
    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: object = None) -> float:
        return float("nan")


@pytest.mark.filterwarnings("ignore:One or more of the test scores are non-finite")
def test_non_finite_inner_scores_are_refused() -> None:
    # A grid point scoring NaN on every inner split is not the best model; selecting it
    # by argmax over NaN picks whichever the sort happened to put first.
    pipe = Pipeline([("regressor", _NaNScoringRegressor(strategy="mean"))])
    with pytest.raises(ValueError, match="finite"):
        tune(pipe, GRID, X8, Y8, TWO_SUBJECTS, split=BY_SUBJECT, seed=0, fold=1)


def test_fit_untuned_sets_available_random_state_parameter() -> None:
    pipe = Pipeline([("rf", RandomForestRegressor(n_estimators=1))])
    fitted = fit_untuned(pipe, X8, Y8, seed=17)
    assert fitted.named_steps["rf"].random_state == 17


def test_within_subject_regression_tunes_with_real_pipeline() -> None:
    cfg = PreprocessingConfig()
    pipe = ridge_pipeline(cfg, seed=42)
    grid = ridge_grid(X8)
    fit = tune(pipe, grid, X8, Y8, TWO_RUNS, split=BY_RUN, seed=42, fold=1)
    assert "regressor__alpha" in fit.best_params


def test_a_candidate_that_zeroes_every_coefficient_does_not_abort_tuning() -> None:
    # The default elastic-net grid reaches penalties that zero every coefficient on a
    # standardized target. Such a candidate predicts a constant, which has no correlation
    # with the target: it must lose the search, not end it.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = 0.5 * X[:, 0] + rng.normal(size=60)
    y = (y - y.mean()) / y.std()
    groups = np.repeat(["s1", "s2", "s3", "s4"], 15).astype(object)
    fit = tune(
        elasticnet_pipeline(PreprocessingConfig(), seed=0),
        elasticnet_grid(),
        X,
        y,
        groups,
        split=InnerSplit(grouping="subject", n_splits=3),
        seed=0,
        fold=1,
        scoring=scoring_dict(),
        refit="r",
    )
    assert fit.best_params["regressor__alpha"] < 10.0


def test_a_failed_inner_search_is_reported_as_a_fit_failure() -> None:
    # Callers such as the permutation null must tell a fit that failed apart from a design
    # that cannot be fitted at all, which stays a plain ValueError.
    with pytest.raises(FoldFitError, match="Fold 1"):
        tune(
            PIPE,
            {"regressor__nonexistent": [1]},
            X8,
            Y8,
            TWO_SUBJECTS,
            split=BY_SUBJECT,
            seed=0,
            fold=1,
        )


class _Diverges(DummyRegressor):
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: object = None) -> _Diverges:
        raise RuntimeError("solver diverged")


def test_an_untuned_fit_that_raises_is_reported_as_a_fit_failure() -> None:
    with pytest.raises(FoldFitError, match="solver diverged"):
        fit_untuned(Pipeline([("regressor", _Diverges())]), X8, Y8, seed=0, fold=3)


def test_tune_refuses_refit_false() -> None:
    # tune() returns a fitted estimator, which GridSearchCV never builds under refit=False.
    with pytest.raises(ValueError, match="refit=False"):
        tune(
            PIPE,
            GRID,
            X8,
            Y8,
            TWO_SUBJECTS,
            split=BY_SUBJECT,
            seed=0,
            fold=1,
            refit=False,
        )


@pytest.mark.parametrize(
    ("scoring", "refit"),
    [
        (subject_r_scorer(), None),
        ({"mse": "neg_mean_squared_error", "r": subject_r_scorer()}, "mse"),
    ],
)
def test_tune_refuses_a_scorer_that_needs_each_rows_subject(scoring, refit) -> None:
    # GridSearchCV hands a scorer the validation rows alone, so it cannot say whose they are.
    with pytest.raises(ValueError, match="subject"):
        tune(
            ridge_pipeline(PreprocessingConfig(), seed=0),
            {"regressor__alpha": [1.0, 10.0]},
            X8,
            Y8,
            TWO_RUNS,
            split=BY_RUN,
            seed=0,
            fold=1,
            scoring=scoring,
            refit=refit,
        )
