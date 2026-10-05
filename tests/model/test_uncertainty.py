from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection

from eegtable.model.uncertainty import (
    _compute_conformal_quantile,
    _order_stat_quantile,
    prediction_intervals,
)

PIPE = Pipeline([("regressor", DummyRegressor(strategy="mean"))])


@pytest.mark.parametrize("method", ["split", "cv_plus"])
@pytest.mark.parametrize("early_stopping", [True, "auto"])
def test_ungrouped_intervals_allow_boosting_early_stopping(method, early_stopping):
    values = np.random.default_rng(31).normal(size=(80, 3))
    model = Pipeline(
        [("hgb", HistGradientBoostingRegressor(early_stopping=early_stopping, max_iter=3))]
    )
    result = prediction_intervals(model, values, values[:, 0], values[:4], method=method)
    assert np.isfinite(result.lower).all()
    assert np.isfinite(result.upper).all()
    assert np.all(result.lower <= result.upper)


@pytest.mark.parametrize("method", ["split", "cv_plus"])
def test_grouped_intervals_reject_boosting_internal_validation(method):
    values = np.random.default_rng(31).normal(size=(80, 3))
    groups = np.repeat(["a", "b", "c", "d"], 20).astype(object)
    model = Pipeline([("hgb", HistGradientBoostingRegressor(early_stopping=True, max_iter=3))])
    with pytest.raises(ValueError, match="early_stopping=False"):
        prediction_intervals(model, values, values[:, 0], values[:4], method=method, groups=groups)


def test_grouped_quantile_intervals_validate_the_replacement_estimator():
    values = np.random.default_rng(31).normal(size=(80, 3))
    groups = np.repeat(["a", "b", "c", "d"], 20).astype(object)
    model = Pipeline([("hgb", HistGradientBoostingRegressor(early_stopping=True, max_iter=3))])
    result = prediction_intervals(
        model, values, values[:, 0], values[:4], method="quantile", groups=groups
    )
    assert np.isfinite(result.lower).all()
    assert np.isfinite(result.upper).all()


@pytest.mark.parametrize("method", ["split", "cv_plus", "quantile"])
def test_interval_seed_controls_stochastic_model_fits(method):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(80, 3))
    y = X[:, 0] + rng.normal(size=80)
    model = Pipeline(
        [
            ("projection", GaussianRandomProjection(n_components=2)),
            ("regressor", RandomForestRegressor(n_estimators=5)),
        ]
    )

    first = prediction_intervals(model, X, y, X[:4], method=method, seed=7)
    second = prediction_intervals(model, X, y, X[:4], method=method, seed=7)

    np.testing.assert_array_equal(first.lower, second.lower)
    np.testing.assert_array_equal(first.upper, second.upper)
    assert model.named_steps["regressor"].random_state is None


def test_intervals_bracket_their_point_prediction() -> None:
    X = np.arange(40.0).reshape(-1, 1)
    y = np.arange(40.0)
    result = prediction_intervals(PIPE, X, y, X[:5], alpha=0.1, cv_splits=4)
    assert np.all(result.lower <= result.upper)


def test_a_smaller_alpha_gives_wider_intervals() -> None:
    X = np.arange(40.0).reshape(-1, 1)
    y = np.arange(40.0)
    wide = prediction_intervals(PIPE, X, y, X[:5], alpha=0.05, cv_splits=4)
    narrow = prediction_intervals(PIPE, X, y, X[:5], alpha=0.10, cv_splits=4)
    assert np.all((wide.upper - wide.lower) >= (narrow.upper - narrow.lower))


def test_the_calibration_unit_is_recorded_so_coverage_cannot_be_over_read() -> None:
    X = np.arange(40.0).reshape(-1, 1)
    y = np.arange(40.0)
    groups = np.repeat(["s1", "s2", "s3", "s4"], 10).astype(object)
    assert prediction_intervals(PIPE, X, y, X[:5], cv_splits=4).calibration_unit == "trial"
    assert (
        prediction_intervals(PIPE, X, y, X[:5], cv_splits=4, groups=groups).calibration_unit
        == "trial"
    )


def test_no_calibration_data_raises_rather_than_returning_an_infinite_interval() -> None:
    with pytest.raises(ValueError):
        prediction_intervals(
            PIPE,
            np.arange(2.0).reshape(-1, 1),
            np.arange(2.0),
            np.zeros((1, 1)),
            alpha=0.1,
            cv_splits=5,
        )


def test_split_method_produces_valid_intervals() -> None:
    X = np.arange(40.0).reshape(-1, 1)
    y = np.arange(40.0)
    result = prediction_intervals(PIPE, X, y, X[:5], alpha=0.1, method="split")
    assert result.method == "split"
    assert len(result.lower) == 5
    assert np.all(result.lower <= result.upper)


def test_quantile_method_produces_valid_intervals() -> None:
    X = np.arange(40.0).reshape(-1, 1)
    y = np.arange(40.0)
    result = prediction_intervals(PIPE, X, y, X[:5], alpha=0.1, method="quantile", cv_splits=3)
    assert result.method == "quantile"
    assert len(result.lower) == 5
    assert np.all(result.lower <= result.upper)


def test_invalid_alpha_raises() -> None:
    X = np.arange(20.0).reshape(-1, 1)
    y = np.arange(20.0)
    with pytest.raises(ValueError, match="alpha"):
        prediction_intervals(PIPE, X, y, X[:2], alpha=1.5)


def test_split_conformal_with_groups() -> None:
    X = np.arange(40.0).reshape(-1, 1)
    y = np.arange(40.0)
    groups = np.repeat(["s1", "s2", "s3", "s4"], 10).astype(object)
    result = prediction_intervals(PIPE, X, y, X[:5], alpha=0.1, method="split", groups=groups)
    assert result.calibration_unit == "trial"
    assert len(result.lower) == 5


def test_split_conformal_raises_on_small_data() -> None:
    X = np.arange(4.0).reshape(-1, 1)
    y = np.arange(4.0)
    with pytest.raises(ValueError, match="at least 5"):
        prediction_intervals(PIPE, X, y, X[:2], method="split")


def test_invalid_method_raises() -> None:
    X = np.arange(20.0).reshape(-1, 1)
    y = np.arange(20.0)
    with pytest.raises(ValueError, match="Unknown method"):
        prediction_intervals(PIPE, X, y, X[:2], method="magic")  # type: ignore[arg-type]


def test_compute_conformal_quantile_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="Calibration set cannot be empty"):
        _compute_conformal_quantile(np.array([], dtype=float), 0.1)


def test_compute_conformal_quantile_returns_inf_when_insufficient_samples() -> None:
    assert _compute_conformal_quantile(np.array([1.0], dtype=float), 0.01) == float("inf")


def test_compute_conformal_quantile_finite() -> None:
    res = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)
    assert _compute_conformal_quantile(res, 0.1) == 9.0


def test_quantile_intervals_use_the_models_preprocessing() -> None:
    # The quantile regressors sit behind the model's own preprocessing, so a pipeline that
    # imputes missing features also gives intervals for data that contain NaN.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    X[0, 1] = np.nan
    y = X[:, 0] + rng.normal(scale=0.1, size=60)
    pipe = Pipeline([("impute", SimpleImputer()), ("regressor", LinearRegression())])
    result = prediction_intervals(pipe, X, y, X[:5], method="quantile", cv_splits=3)
    assert np.all(np.isfinite(result.lower)) and np.all(np.isfinite(result.upper))


def test_quantile_intervals_reach_their_coverage_on_new_data() -> None:
    # Calibrated in CV+ form, the quantile intervals cover at least 1 - 2 * alpha of new
    # exchangeable trials, here with noise that grows away from zero.
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(600, 1))
    y = X[:, 0] + rng.normal(scale=0.2 + 0.3 * np.abs(X[:, 0]))
    pipe = Pipeline([("regressor", LinearRegression())])
    result = prediction_intervals(
        pipe, X[:200], y[:200], X[200:], alpha=0.1, method="quantile", cv_splits=5
    )
    covered = (y[200:] >= result.lower) & (y[200:] <= result.upper)
    assert covered.mean() >= 0.8


def test_order_stat_quantile_formulas() -> None:
    res = np.arange(1, 10, dtype=float)
    # Upper bound uses ceil((1-alpha)*(n+1)), lower bound uses floor(alpha*(n+1))
    upper = _order_stat_quantile(res, 0.1, tail="upper")
    lower = _order_stat_quantile(res, 0.1, tail="lower")
    assert np.isfinite(upper)
    assert np.isfinite(lower)
    assert lower <= upper


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"y_train": np.ones(9)}, "y_train has 9 rows"),
        ({"groups": np.array(["s1"] * 9, dtype=object)}, "groups has 9 rows"),
        ({"X_test": np.ones((3, 2))}, "X_test has 2 columns"),
    ],
)
def test_prediction_intervals_refuse_misaligned_inputs(
    kwargs: dict[str, np.ndarray], message: str
) -> None:
    rng = np.random.default_rng(0)
    base = {
        "model": PIPE,
        "X_train": rng.normal(size=(10, 3)),
        "y_train": rng.normal(size=10),
        "X_test": rng.normal(size=(3, 3)),
    }
    with pytest.raises(ValueError, match=message):
        prediction_intervals(**{**base, **kwargs})


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_calibration_does_not_drop_nonfinite_scores(invalid) -> None:
    scores = np.arange(20.0)
    scores[-1] = invalid
    with pytest.raises(ValueError, match="finite"):
        _compute_conformal_quantile(scores, 0.1)
    with pytest.raises(ValueError, match="finite"):
        _order_stat_quantile(scores, 0.1, tail="upper")


class NonfiniteRegressor(DummyRegressor):
    def predict(self, X):
        predictions = super().predict(X)
        predictions[0] = np.nan
        return predictions


@pytest.mark.parametrize("method", ["split", "cv_plus"])
def test_intervals_reject_nonfinite_model_predictions(method) -> None:
    X = np.arange(40.0).reshape(-1, 1)
    model = Pipeline([("regressor", NonfiniteRegressor())])
    with pytest.raises(ValueError, match="finite"):
        prediction_intervals(model, X, X[:, 0], X[:5], method=method)


@pytest.mark.parametrize("method", ["split", "cv_plus", "quantile"])
def test_intervals_reject_multidimensional_targets(method) -> None:
    X = np.arange(40.0).reshape(-1, 1)
    with pytest.raises(ValueError, match="y_train must be.*1-D"):
        prediction_intervals(PIPE, X, X, X[:5], method=method)
