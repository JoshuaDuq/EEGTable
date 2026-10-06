from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline

from eegtable.model.crossfit import cross_fit_classification, cross_fit_regression
from eegtable.model.splits import Fold, InnerSplit


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
