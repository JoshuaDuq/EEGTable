from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline

from eegtable.model.crossfit import cross_fit_classification, cross_fit_regression
from eegtable.model.splits import Fold, InnerSplit

VALUES = np.arange(16, dtype=float).reshape(8, 2)
TARGET = np.tile([0, 1], 4)
GROUPS = np.repeat(["s1", "s2"], 4).astype(object)
FOLD = Fold(1, np.arange(4), np.arange(4, 8))


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("columns", [1, 2])
def test_cross_fitting_requires_one_dimensional_targets(task, columns) -> None:
    target = TARGET[:, None].repeat(columns, axis=1)
    evaluate = cross_fit_regression if task == "regression" else cross_fit_classification
    estimator = DummyRegressor() if task == "regression" else DummyClassifier()

    with pytest.raises(ValueError, match="y must be a finite 1-D array"):
        evaluate(
            [FOLD],
            VALUES,
            target,
            GROUPS,
            Pipeline([("model", estimator)]),
            {},
            inner=InnerSplit("subject"),
            seed=0,
        )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_regression_rejects_nonfinite_held_out_targets(invalid) -> None:
    target = np.arange(8, dtype=float)
    target[-1] = invalid

    with pytest.raises(ValueError, match="y must be a finite 1-D array"):
        cross_fit_regression(
            [FOLD],
            VALUES,
            target,
            GROUPS,
            Pipeline([("model", DummyRegressor())]),
            {},
            inner=InnerSplit("subject"),
            seed=0,
        )


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_cross_fitting_rejects_complex_targets(task) -> None:
    evaluate = cross_fit_regression if task == "regression" else cross_fit_classification
    estimator = DummyRegressor() if task == "regression" else DummyClassifier()
    with pytest.raises(ValueError, match="y must be a finite 1-D array"):
        evaluate(
            [FOLD],
            VALUES,
            TARGET.astype(complex),
            GROUPS,
            Pipeline([("model", estimator)]),
            {},
            inner=InnerSplit("subject"),
            seed=0,
        )


class _ColumnRegressor(DummyRegressor):
    def predict(self, X):
        return super().predict(X)[:, None]


class _NonfiniteRegressor(DummyRegressor):
    def predict(self, X):
        return np.full(len(X), np.inf)


class _ComplexRegressor(DummyRegressor):
    def predict(self, X):
        return np.full(len(X), 0.5 + 1j)


@pytest.mark.parametrize(
    "estimator", [_ColumnRegressor(), _NonfiniteRegressor(), _ComplexRegressor()]
)
def test_regression_rejects_malformed_predictions(estimator) -> None:
    with pytest.raises(ValueError, match="regression predictions must be finite and aligned"):
        cross_fit_regression(
            [FOLD],
            VALUES,
            TARGET,
            GROUPS,
            Pipeline([("model", estimator)]),
            {},
            inner=InnerSplit("subject"),
            seed=0,
        )


class _FractionalClassifier(DummyClassifier):
    def predict(self, X):
        return np.full(len(X), 0.5)


class _NonfiniteProbabilityClassifier(DummyClassifier):
    def predict_proba(self, X):
        return np.full((len(X), 2), np.nan)


class _OutOfBoundsProbabilityClassifier(DummyClassifier):
    def predict_proba(self, X):
        return np.tile([-1.0, 2.0], (len(X), 1))


class _UnnormalizedProbabilityClassifier(DummyClassifier):
    def predict_proba(self, X):
        return np.full((len(X), 2), 0.25)


class _ComplexClassifier(DummyClassifier):
    def predict(self, X):
        return np.zeros(len(X), dtype=complex)


class _ComplexProbabilityClassifier(DummyClassifier):
    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5 + 1j)


class _ComplexDecisionClassifier(DummyClassifier):
    def decision_function(self, X):
        return np.full(len(X), 0.5 + 1j)


@pytest.mark.parametrize(
    "estimator, message",
    [
        (DummyRegressor(), "classification requires fitted classes 0/1"),
        (_FractionalClassifier(), "binary predictions are required for every row"),
        (_ComplexClassifier(), "binary predictions are required for every row"),
        (_NonfiniteProbabilityClassifier(), "predictions require finite binary probabilities"),
        (_OutOfBoundsProbabilityClassifier(), "probabilities must be in \\[0, 1\\] and sum to one"),
        (
            _UnnormalizedProbabilityClassifier(),
            "probabilities must be in \\[0, 1\\] and sum to one",
        ),
        (_ComplexProbabilityClassifier(), "predictions require finite binary probabilities"),
        (_ComplexDecisionClassifier(), "Binary decision scores must be finite"),
    ],
)
def test_classification_rejects_malformed_estimator_outputs(estimator, message) -> None:
    with pytest.raises(ValueError, match=message):
        cross_fit_classification(
            [FOLD],
            VALUES,
            TARGET,
            GROUPS,
            Pipeline([("model", estimator)]),
            {},
            inner=InnerSplit("subject"),
            seed=0,
        )


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("name", ["groups", "runs"])
@pytest.mark.parametrize("invalid", ["missing", "column"])
def test_split_labels_must_be_complete_one_dimensional_arrays(task, name, invalid) -> None:
    values = np.arange(8, dtype=float).reshape(-1, 1)
    target = np.tile([0, 1], 4)
    groups = np.full(8, "s1", dtype=object)
    runs = np.repeat([1.0, 2.0], 4)
    invalid_labels = np.full(8, np.nan) if invalid == "missing" else np.arange(8)[:, None]
    if name == "groups":
        groups = invalid_labels
        subject = None
        inner = InnerSplit("subject")
    else:
        runs = invalid_labels
        subject = "s1"
        inner = InnerSplit("run")
    fold = Fold(1, np.arange(4), np.arange(4, 8), subject=subject)
    evaluate = cross_fit_regression if task == "regression" else cross_fit_classification
    estimator = DummyRegressor() if task == "regression" else DummyClassifier()

    with pytest.raises(ValueError, match=f"{name} must contain one nonmissing label per row"):
        evaluate(
            [fold],
            values,
            target,
            groups,
            Pipeline([("model", estimator)]),
            {},
            inner=inner,
            seed=0,
            runs=runs,
        )
