from __future__ import annotations

import contextlib
import importlib.util
import types
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from eegtable.model.estimators import ridge_pipeline
from eegtable.model.importance import (
    Importance,
    aggregate_by,
    permutation_importance,
    permutation_importance_over_folds,
    shap_importance,
    shap_importance_over_folds,
)
from eegtable.model.scoring import pearsonr_scorer
from eegtable.model.splits import Fold, InnerSplit, loso_folds
from eegtable.model.transformers import Deconfounder, PreprocessingConfig
from eegtable.table import FeatureMeta

PIPE = Pipeline([("regressor", DummyRegressor(strategy="mean"))])
needs_shap = pytest.mark.skipif(
    importlib.util.find_spec("shap") is None, reason="shap not installed"
)


def test_shap_binary_tree_output_keeps_class_and_feature_axes_separate() -> None:
    class TreeExplainer:
        def __init__(self, model):
            pass

        def shap_values(self, X):
            values = X - X.mean(axis=0)
            return np.stack([-values, values], axis=-1)

    X = np.random.default_rng(0).normal(size=(20, 3))
    model = Pipeline([("classifier", RandomForestClassifier(n_estimators=2, random_state=0))])
    model.fit(X, (X[:, 0] > 0).astype(int))
    fake_shap = types.ModuleType("shap")
    fake_shap.TreeExplainer = TreeExplainer
    with (
        patch("eegtable.model.importance.require_shap", return_value=None),
        patch.dict("sys.modules", {"shap": fake_shap}),
    ):
        result = shap_importance(model, X, ["a", "b", "c"])
    np.testing.assert_allclose(result.values, np.abs(X - X.mean(axis=0)).mean(axis=0))


class _LinearExplainerStandIn:
    # shap is an optional extra that CI does not install. For a linear model with an
    # independent masker, SHAP values are coef * (x - mean), one column per column of the
    # data the explainer receives, which is everything shap_importance relies on.
    def __init__(self, model: object, data: object) -> None:
        self.coef = np.ravel(getattr(model, "coef_"))  # noqa: B009

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        return np.asarray((values - values.mean(axis=0)) * self.coef)


@contextlib.contextmanager
def _shap_stand_in() -> Iterator[None]:
    module = types.ModuleType("shap")
    module.LinearExplainer = _LinearExplainerStandIn  # type: ignore[attr-defined]
    with (
        patch("eegtable.model.importance.require_shap", return_value=None),
        patch.dict("sys.modules", {"shap": module}),
    ):
        yield


def test_permutation_importance_names_every_feature_it_scores() -> None:
    X = np.random.default_rng(0).normal(size=(30, 3))
    y = X[:, 0] * 2.0
    result = permutation_importance(PIPE.fit(X, y), X, y, n_repeats=3, seed=0)
    assert len(result.feature_names) == result.values.size == 3


def test_permutation_scores_receive_unmodified_feature_values() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(80, 3)) * [2, 3, 4] + 10
    target = values @ np.array([2, -1, 1]) + rng.normal(scale=0.2, size=80)
    importances = []
    for copy in (True, False):
        pipeline = Pipeline([("scale", StandardScaler(copy=copy)), ("model", Ridge())])
        pipeline.fit(values[:60].copy(), target[:60])
        held_out = values[60:].copy()
        importances.append(
            permutation_importance(pipeline, held_out, target[60:], n_repeats=4, seed=0)
        )
        np.testing.assert_array_equal(held_out, values[60:])
    np.testing.assert_allclose(importances[0].values, importances[1].values)


def test_importance_aggregates_by_a_metadata_field_not_a_name_fragment(
    alpha_beta_meta: tuple[FeatureMeta, FeatureMeta],
) -> None:
    importance = Importance(
        feature_names=(alpha_beta_meta[0].name, alpha_beta_meta[1].name),
        values=np.array([1.0, 3.0]),
        per_fold=np.empty((0, 2)),
    )
    assert aggregate_by(importance, alpha_beta_meta, "band") == {"alpha": 1.0, "beta": 3.0}


def test_importance_aggregation_is_invariant_to_metadata_order(
    alpha_beta_meta: tuple[FeatureMeta, FeatureMeta],
) -> None:
    importance = Importance(
        feature_names=(alpha_beta_meta[0].name, alpha_beta_meta[1].name),
        values=np.array([1.0, 3.0]),
        per_fold=np.empty((0, 2)),
    )
    reversed_meta = (alpha_beta_meta[1], alpha_beta_meta[0])
    assert aggregate_by(importance, reversed_meta, "band") == {"alpha": 1.0, "beta": 3.0}


def test_shap_importance_raises_when_shap_missing() -> None:
    if importlib.util.find_spec("shap") is None:
        X = np.ones((10, 2))
        with pytest.raises(ModuleNotFoundError, match="SHAP"):
            shap_importance(PIPE, X, ["f1", "f2"])


@needs_shap
def test_shap_importance_computes_with_shap_installed() -> None:
    X = np.random.default_rng(0).normal(size=(20, 2))
    y = X[:, 0] * 1.5
    pipe = Pipeline([("regressor", DummyRegressor())])
    pipe.fit(X, y)
    imp = shap_importance(pipe, X, ["f0", "f1"])
    assert len(imp.feature_names) == 2
    assert imp.values.shape == (2,)


def test_shap_kernel_uses_estimator_predict_fn_for_transformed_features() -> None:
    captured: dict[str, object] = {}

    class FakeKernelExplainer:
        def __init__(self, predict_fn: object, background: object) -> None:
            captured["predict_fn"] = predict_fn
            captured["background"] = background

        def shap_values(self, X_input: object, nsamples: int = 100) -> np.ndarray:
            arr = np.asarray(X_input)
            return np.zeros_like(arr)

    fake_shap = MagicMock()
    fake_shap.KernelExplainer = FakeKernelExplainer

    regressor = DummyRegressor()
    scaler = StandardScaler()
    pipe = Pipeline([("scaler", scaler), ("regressor", regressor)])
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1.0, 2.0])
    pipe.fit(X, y)

    with (
        patch("eegtable.model.importance.require_shap", return_value=None),
        patch.dict("sys.modules", {"shap": fake_shap}),
    ):
        shap_importance(pipe, X, ["f1", "f2"])

    assert captured["predict_fn"] == regressor.predict


def test_shap_stage_requires_min_valid_fold_fraction() -> None:
    folds = [
        Fold(index=0, train=np.array([0, 1]), test=np.array([2])),
        Fold(index=1, train=np.array([0, 2]), test=np.array([1])),
    ]
    X = np.ones((3, 2))
    y = np.ones(3)
    groups = np.array(["s1", "s2", "s3"], dtype=object)
    pipe = Pipeline([("regressor", DummyRegressor())])
    inner = InnerSplit(grouping="subject", n_splits=2)

    with (
        patch("eegtable.model.importance.require_shap", return_value=None),
        patch(
            "eegtable.model.importance.shap_importance",
            side_effect=[
                Importance(
                    feature_names=("f1", "f2"),
                    values=np.array([1.0, 2.0]),
                    per_fold=np.empty((0, 2)),
                ),
                Importance(
                    feature_names=("f1", "f2"),
                    values=np.array([np.nan, np.nan]),
                    per_fold=np.empty((0, 2)),
                ),
            ],
        ),
        pytest.raises(ValueError, match="Insufficient successful folds for SHAP"),
    ):
        shap_importance_over_folds(
            folds,
            X,
            y,
            groups,
            pipe,
            {},
            ["f1", "f2"],
            inner=inner,
            min_complete_fraction=0.8,
        )


def test_permutation_importance_stage_requires_min_valid_fold_fraction() -> None:
    folds = [
        Fold(index=0, train=np.array([0, 1]), test=np.array([2])),
        Fold(index=1, train=np.array([0, 2]), test=np.array([1])),
    ]
    X = np.ones((3, 2))
    y = np.ones(3)
    groups = np.array(["s1", "s2", "s3"], dtype=object)
    pipe = Pipeline([("regressor", DummyRegressor())])
    inner = InnerSplit(grouping="subject", n_splits=2)

    with (
        patch(
            "eegtable.model.importance.permutation_importance",
            side_effect=[
                Importance(
                    feature_names=("f1", "f2"),
                    values=np.array([1.0, 2.0]),
                    per_fold=np.empty((0, 2)),
                ),
                Importance(
                    feature_names=("f1", "f2"),
                    values=np.array([np.nan, np.nan]),
                    per_fold=np.empty((0, 2)),
                ),
            ],
        ),
        pytest.raises(ValueError, match="Insufficient successful folds for permutation importance"),
    ):
        permutation_importance_over_folds(
            folds,
            X,
            y,
            groups,
            pipe,
            {},
            inner=inner,
            min_complete_fraction=0.8,
        )


def test_permutation_importance_over_folds_runs_and_aggregates() -> None:
    # Three trials per held-out subject, the fewest a subject-level r can be scored on.
    folds = [
        Fold(index=0, train=np.arange(6), test=np.array([6, 7, 8])),
        Fold(index=1, train=np.arange(3, 9), test=np.array([0, 1, 2])),
    ]
    X = np.arange(18, dtype=np.float64).reshape(9, 2)
    y = np.array([1.0, 1.2, 1.1, 2.0, 2.1, 2.3, 3.0, 3.2, 3.1])
    groups = np.repeat(["s1", "s2", "s3"], 3).astype(object)
    pipe = Pipeline([("regressor", DummyRegressor())])
    inner = InnerSplit(grouping="subject", n_splits=2)

    res = permutation_importance_over_folds(
        folds, X, y, groups, pipe, {}, inner=inner, feature_names=["f0", "f1"], n_repeats=2
    )
    assert res.per_fold.shape == (2, 2)
    assert res.values.shape == (2,)
    assert res.feature_names == ("f0", "f1")


def test_permutation_importance_over_folds_handles_fold_varying_widths() -> None:
    folds = [
        Fold(index=0, train=np.arange(3, 9), test=np.array([0, 1, 2])),
        Fold(index=1, train=np.arange(6), test=np.array([6, 7, 8])),
    ]
    X = np.ones((9, 3))
    # For subject s1 (rows 0 to 2), f2 is all NaN
    X[0:3, 2] = np.nan
    y = np.arange(9, dtype=float)
    groups = np.repeat(["s1", "s2", "s3"], 3).astype(object)
    pipe = Pipeline([("regressor", DummyRegressor(strategy="mean"))])
    inner = InnerSplit(grouping="subject", n_splits=2)

    res = permutation_importance_over_folds(
        folds,
        X,
        y,
        groups,
        pipe,
        {},
        inner=inner,
        feature_names=["f0", "f1", "f2"],
        harmonization="intersection",
        n_repeats=2,
    )
    assert res.feature_names == ("f0", "f1", "f2")
    assert res.values.shape == (3,)
    assert res.per_fold.shape == (2, 3)
    assert np.isnan(res.per_fold[1, 2])
    assert not np.isnan(res.per_fold[0, 2])


def _with_constant_feature(n_rows: int) -> np.ndarray:
    values = np.random.default_rng(0).normal(size=(n_rows, 3))
    values[:, 1] = 5.0  # constant, so the pipeline's variance step drops it
    return values


def test_shap_importance_reports_zero_for_a_feature_the_pipeline_dropped() -> None:
    # SHAP explains the matrix the final estimator sees, after steps that drop columns. A
    # dropped feature has no effect on the prediction, so it reads 0 instead of taking the
    # value of whichever column moved into its position.
    X = _with_constant_feature(40)
    pipe = ridge_pipeline(PreprocessingConfig(), seed=0).fit(X, X[:, 0] + 0.5 * X[:, 2])
    with _shap_stand_in():
        imp = shap_importance(pipe, X, ["f0", "f1", "f2"])
    assert imp.feature_names == ("f0", "f1", "f2")
    assert imp.values[1] == 0.0
    assert imp.values[0] > imp.values[2] > 0.0


def test_shap_importance_over_folds_keeps_features_aligned_when_one_is_dropped() -> None:
    groups = np.repeat(["s1", "s2", "s3", "s4"], 10).astype(object)
    X = _with_constant_feature(40)
    with _shap_stand_in():
        imp = shap_importance_over_folds(
            loso_folds(groups),
            X,
            X[:, 0] + 0.5 * X[:, 2],
            groups,
            ridge_pipeline(PreprocessingConfig(), seed=0),
            {},
            ["f0", "f1", "f2"],
            inner=InnerSplit(grouping="subject", n_splits=2),
        )
    np.testing.assert_array_equal(imp.per_fold[:, 1], np.zeros(4))
    assert np.all(imp.per_fold[:, 0] > imp.per_fold[:, 2])


def test_shap_importance_refuses_columns_that_are_not_input_features() -> None:
    # PCA components mix every feature, so their SHAP values belong to no single feature.
    X = np.random.default_rng(0).normal(size=(40, 3))
    pipe = ridge_pipeline(PreprocessingConfig(pca_enabled=True), seed=0).fit(X, X[:, 0])
    with _shap_stand_in(), pytest.raises(ValueError, match="input feature"):
        shap_importance(pipe, X, ["f0", "f1", "f2"])


def test_shap_importance_refuses_a_transformer_with_unknown_column_mapping() -> None:
    values = np.random.default_rng(12).normal(size=(40, 2))
    model = Pipeline(
        [
            ("reverse", FunctionTransformer(lambda data: data[:, ::-1])),
            ("model", LinearRegression()),
        ]
    ).fit(values, values[:, 0])
    np.testing.assert_allclose(model.predict(values), values[:, 0], atol=1e-12)

    with _shap_stand_in(), pytest.raises(ValueError, match="output feature names"):
        shap_importance(model, values, ["a", "b"])


def test_shap_importance_uses_explicit_transformer_column_mapping() -> None:
    values = np.random.default_rng(12).normal(size=(40, 2))
    model = Pipeline(
        [
            (
                "reverse",
                FunctionTransformer(
                    lambda data: data[:, ::-1],
                    feature_names_out=lambda transformer, names: names[::-1],
                ),
            ),
            ("identity", "passthrough"),
            ("model", LinearRegression()),
        ]
    ).fit(values, values[:, 0])

    with _shap_stand_in():
        importance = shap_importance(model, values, ["a", "b"])
    np.testing.assert_allclose(
        importance.values,
        [np.abs(values[:, 0] - values[:, 0].mean()).mean(), 0.0],
        atol=1e-12,
    )


@pytest.mark.parametrize("container", ["direct", "pipeline", "columns", "union"])
def test_shap_importance_refuses_covariate_mixing(container) -> None:
    rng = np.random.default_rng(12)
    nuisance, target = rng.normal(size=(2, 100))
    values = np.column_stack([2 * nuisance + target, nuisance])
    transform = Deconfounder(n_covariates=1)
    if container == "pipeline":
        transform = Pipeline([("deconfound", transform)])
    elif container == "columns":
        transform = ColumnTransformer(
            [("deconfound", transform, [0, 1])], verbose_feature_names_out=False
        )
    elif container == "union":
        transform = FeatureUnion([("deconfound", transform)], verbose_feature_names_out=False)
    model = Pipeline([("transform", transform), ("model", LinearRegression())]).fit(values, target)
    changed = values.copy()
    changed[:, 1] += 1
    assert not np.allclose(model.predict(values), model.predict(changed))

    with _shap_stand_in(), pytest.raises(ValueError, match="original-input SHAP attribution"):
        shap_importance(model, values, ["eeg", "nuisance"])


def test_shap_importance_allows_a_deconfounder_without_covariates() -> None:
    values = np.random.default_rng(12).normal(size=(40, 2))
    model = Pipeline([("deconfound", Deconfounder()), ("model", LinearRegression())]).fit(
        values, values[:, 0] + values[:, 1]
    )
    with _shap_stand_in():
        importance = shap_importance(model, values, ["a", "b"])
    assert np.all(importance.values > 0)


@pytest.mark.parametrize("nested", [False, True])
def test_shap_importance_allows_an_inactive_column_transformer_deconfounder(nested) -> None:
    values = np.random.default_rng(12).normal(size=(40, 2))
    transform = ColumnTransformer(
        [("inactive", Deconfounder(n_covariates=1), [])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    if nested:
        transform = Pipeline([("columns", transform)])
    model = Pipeline([("transform", transform), ("model", LinearRegression())]).fit(
        values, values.sum(axis=1)
    )
    np.testing.assert_array_equal(model[:-1].transform(values), values)

    with _shap_stand_in():
        importance = shap_importance(model, values, ["a", "b"])
    np.testing.assert_allclose(importance.values, np.abs(values - values.mean(axis=0)).mean(axis=0))


def _nuisance_design() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # f1 is the nuisance covariate itself; once the target is residualized on it, only f0
    # has anything left to explain.
    rng = np.random.default_rng(0)
    groups = np.repeat([f"s{i}" for i in range(5)], 20).astype(object)
    nuisance = rng.normal(size=100)
    X = np.column_stack([rng.normal(size=100), nuisance])
    y = 0.5 * X[:, 0] + 3.0 * nuisance + 0.1 * rng.normal(size=100)
    return X, y, groups, nuisance.reshape(-1, 1)


def test_permutation_importance_is_measured_on_the_residualized_target() -> None:
    X, y, groups, covariates = _nuisance_design()
    imp = permutation_importance_over_folds(
        loso_folds(groups),
        X,
        y,
        groups,
        Pipeline([("regressor", LinearRegression())]),
        {},
        inner=InnerSplit(grouping="subject", n_splits=2),
        feature_names=["f0", "f1"],
        covariates=covariates,
        residualize_on=["nuisance"],
        n_repeats=5,
    )
    assert imp.values[0] > 0.5
    assert abs(imp.values[1]) < 0.1 * imp.values[0]


def test_shap_importance_explains_the_model_fitted_on_the_residualized_target() -> None:
    X, y, groups, covariates = _nuisance_design()
    with _shap_stand_in():
        imp = shap_importance_over_folds(
            loso_folds(groups),
            X,
            y,
            groups,
            Pipeline([("regressor", LinearRegression())]),
            {},
            ["f0", "f1"],
            inner=InnerSplit(grouping="subject", n_splits=2),
            covariates=covariates,
            residualize_on=["nuisance"],
        )
    assert imp.values[1] < 0.2 * imp.values[0]


@pytest.mark.parametrize("multi_metric", [False, True])
def test_permutation_importance_is_measured_with_the_selection_metric(multi_metric: bool) -> None:
    # Importance is the drop in the metric the model was selected on. A metric that ignores
    # the predictions cannot drop, so every importance is exactly zero.
    rng = np.random.default_rng(0)
    groups = np.repeat(["s1", "s2", "s3"], 10).astype(object)
    X = rng.normal(size=(30, 2))
    y = X[:, 0] + 0.1 * rng.normal(size=30)
    constant_metric = make_scorer(lambda yt, yp: 1.0)
    scoring: object = (
        {"r": pearsonr_scorer(), "constant": constant_metric} if multi_metric else constant_metric
    )
    imp = permutation_importance_over_folds(
        loso_folds(groups),
        X,
        y,
        groups,
        Pipeline([("regressor", LinearRegression())]),
        {},
        inner=InnerSplit(grouping="subject", n_splits=2),
        scoring=scoring,
        refit="constant" if multi_metric else None,
        n_repeats=2,
    )
    np.testing.assert_array_equal(imp.values, np.zeros(2))


class _Column(BaseEstimator, RegressorMixin):
    # Predicts one input column as it is, so the score it loses when permuted is known.
    def __init__(self, column: int = 1) -> None:
        self.column = column

    def fit(self, X, y):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float)[:, self.column]


def test_regression_importance_is_the_drop_in_subject_level_r_by_default() -> None:
    # The model tracks every held-out trial at 100 times its scale: a within-subject r of 1
    # and an R^2 far below zero. Permuting the tracking column costs the whole r, about 1;
    # measured in R^2 the same permutation would cost about 200.
    rng = np.random.default_rng(0)
    groups = np.repeat([f"s{i}" for i in range(6)], 10).astype(object)
    offset = np.repeat(np.arange(6) * 10.0, 10)
    signal = rng.normal(size=groups.size)
    X = np.column_stack([offset, 100.0 * signal])
    imp = permutation_importance_over_folds(
        loso_folds(groups),
        X,
        offset + signal,
        groups,
        Pipeline([("regressor", _Column())]),
        {},
        inner=InnerSplit(grouping="subject", n_splits=2),
        n_repeats=10,
    )
    assert imp.values[0] == 0.0
    assert imp.values[1] == pytest.approx(1.0, abs=0.2)


@pytest.mark.parametrize(
    "entry_point",
    [
        pytest.param("permutation", id="permutation"),
        pytest.param("shap", id="shap"),
    ],
)
def test_importance_over_folds_rejects_overlapping_custom_folds(entry_point: str) -> None:
    # Importance explains the model cross-fitting evaluated, so it has to refuse the same
    # leaky folds cross-fitting refuses; otherwise it reports on a model fitted on its test set.
    bad = Fold(index=1, train=np.array([0, 1, 2, 3]), test=np.array([3, 4, 5]))
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    y = np.array([1.0, 1.2, 2.0, 2.1, 3.0, 3.2])
    groups = np.array(["s1", "s1", "s2", "s2", "s3", "s3"], dtype=object)
    pipe = Pipeline([("regressor", DummyRegressor())])
    inner = InnerSplit(grouping="subject", n_splits=2)

    with pytest.raises(ValueError, match="train/test overlap"):
        if entry_point == "permutation":
            permutation_importance_over_folds([bad], X, y, groups, pipe, {}, inner=inner)
        else:
            with _shap_stand_in():
                shap_importance_over_folds([bad], X, y, groups, pipe, {}, ["f0", "f1"], inner=inner)


def test_importance_over_folds_rejects_mismatched_y_length() -> None:
    folds = [Fold(index=0, train=np.array([0, 1, 2, 3]), test=np.array([4, 5]))]
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    groups = np.array(["s1", "s1", "s2", "s2", "s3", "s3"], dtype=object)
    pipe = Pipeline([("regressor", DummyRegressor())])
    inner = InnerSplit(grouping="subject", n_splits=2)

    with pytest.raises(ValueError, match="y has .* rows and X has"):
        permutation_importance_over_folds(folds, X, np.ones(5), groups, pipe, {}, inner=inner)


def test_aggregation_refuses_a_feature_no_fold_could_score(
    alpha_beta_meta: tuple[FeatureMeta, FeatureMeta],
) -> None:
    # nanmean over folds leaves NaN when a feature was dropped everywhere, and summing that
    # into a category would report the whole category as NaN.
    importance = Importance(
        feature_names=(alpha_beta_meta[0].name, alpha_beta_meta[1].name),
        values=np.array([1.0, np.nan]),
        per_fold=np.array([[1.0, np.nan]]),
    )
    with pytest.raises(ValueError, match="No fold scored"):
        aggregate_by(importance, alpha_beta_meta, "band")


def test_an_infinite_fold_value_is_skipped_like_a_missing_one() -> None:
    # A fold counts as successful when any feature survived it, so one feature's
    # infinity must not become that feature's mean across folds: the guard above
    # already calls a non-finite value missing, and nanmean alone does not.
    from eegtable.model.importance import _combine_folds

    folds = [
        np.array([1.0, np.inf, 3.0]),
        np.array([3.0, 2.0, np.nan]),
        np.array([2.0, 2.0, 5.0]),
    ]
    _, values = _combine_folds(folds, n_folds=3, min_complete_fraction=0.5, label="test")
    assert values == pytest.approx([2.0, 2.0, 4.0])
