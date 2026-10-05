"""Numerical references and grouped-fitting contracts for additional models."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from eegtable.model import estimators
from eegtable.model.crossfit import cross_fit_classification, cross_fit_regression
from eegtable.model.splits import InnerSplit, inner_cv, loso_folds
from eegtable.model.transformers import PreprocessingConfig, _check_subject_missingness
from eegtable.model.tuning import fit_untuned, tune


def _pipeline(name, **kwargs):
    factory = getattr(estimators, name, None)
    assert callable(factory), f"Missing estimator factory: {name}"
    return factory(PreprocessingConfig(), seed=7, **kwargs)


@pytest.mark.parametrize(
    "factory,reference,task",
    [
        (
            "hist_gradient_boosting_pipeline",
            HistGradientBoostingRegressor(early_stopping=False, random_state=7),
            "regression",
        ),
        (
            "hist_gradient_boosting_classifier_pipeline",
            HistGradientBoostingClassifier(
                early_stopping=False, random_state=7, class_weight="balanced"
            ),
            "classification",
        ),
        (
            "svr_pipeline",
            TransformedTargetRegressor(regressor=SVR(), transformer=StandardScaler()),
            "regression",
        ),
        (
            "lda_pipeline",
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            "classification",
        ),
    ],
)
@pytest.mark.parametrize("n_covariates", [0, 1])
def test_new_pipelines_match_scikit_learn_after_preprocessing(
    factory, reference, task, n_covariates
):
    rng = np.random.default_rng(31)
    values = np.column_stack([rng.normal(size=(60, 3)), np.ones(60)])
    values[0, 1] = np.nan
    if n_covariates:
        values = np.column_stack([values, rng.normal(size=60)])
    target = values[:, 0] if task == "regression" else (values[:, 0] > 0).astype(int)
    pipeline = _pipeline(factory, n_covariates=n_covariates).fit(values, target)
    transformed = pipeline[:-1].transform(values)
    reference.fit(transformed, target)
    np.testing.assert_allclose(pipeline.predict(values), reference.predict(transformed))
    if task == "classification":
        np.testing.assert_allclose(
            pipeline.predict_proba(values), reference.predict_proba(transformed)
        )


def test_svr_predictions_preserve_feature_and_target_units():
    rng = np.random.default_rng(31)
    values = rng.normal(size=(60, 3))
    target = np.sin(values[:, 0]) + 0.1 * values[:, 1]
    original = _pipeline("svr_pipeline").fit(values, target)
    rescaled = _pipeline("svr_pipeline").fit(values * 1e-6, 1000 * target + 1234)
    np.testing.assert_allclose(
        1000 * original.predict(values) + 1234,
        rescaled.predict(values * 1e-6),
        rtol=1e-6,
    )


def test_lda_missingness_audit_checks_preprocessing_without_transforming_the_classifier():
    values = np.random.default_rng(31).normal(size=(24, 3))
    target = np.tile([0, 1], 12)
    pipeline = _pipeline("lda_pipeline").fit(values, target)
    groups = np.repeat(["a", "b", "c", "d"], 6).astype(object)
    _check_subject_missingness(pipeline, values, groups)
    values[groups == "a"] = np.nan
    with pytest.raises(ValueError, match="Subject a has missingness"):
        _check_subject_missingness(pipeline, values, groups)


def test_svr_target_scaling_is_fitted_only_on_nested_training_rows(monkeypatch):
    values = np.random.default_rng(31).normal(size=(24, 3))
    target = np.arange(24, dtype=float)
    groups = np.repeat(["a", "b", "c", "d"], 6).astype(object)
    folds = loso_folds(groups)
    inner = InnerSplit("subject", n_splits=2)
    expected_targets = set()
    for fold in folds:
        expected_targets.add(tuple(target[fold.train]))
        splits = inner_cv(groups[fold.train], inner).split(
            values[fold.train], target[fold.train], groups=groups[fold.train]
        )
        for train, _ in splits:
            expected_targets.add(tuple(target[fold.train[train]]))

    fitted_targets = set()
    fit = StandardScaler.fit

    def record_fit(self, X, y=None, **kwargs):
        if X.shape[1] == 1:
            fitted_targets.add(tuple(X[:, 0]))
        return fit(self, X, y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record_fit)
    results = cross_fit_regression(
        folds,
        values,
        target,
        groups,
        _pipeline("svr_pipeline"),
        {"svr__regressor__C": [0.1, 1.0]},
        inner=inner,
        seed=7,
        scoring="neg_mean_squared_error",
    )
    assert fitted_targets == expected_targets
    np.testing.assert_array_equal(
        np.sort(np.concatenate([fold.rows for fold in results])), np.arange(24)
    )
    assert all(np.isfinite(fold.y_pred).all() for fold in results)


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("early_stopping", [True, "auto", np.bool_(True)])
@pytest.mark.parametrize("grid", [{}, {"hgb__max_iter": [2]}])
def test_grouped_fitting_rejects_internal_boosting_validation(task, early_stopping, grid):
    values = np.random.default_rng(31).normal(size=(24, 3))
    target = np.tile([0, 1], 12)
    groups = np.repeat(["a", "b", "c", "d"], 6).astype(object)
    estimator = (
        HistGradientBoostingRegressor if task == "regression" else HistGradientBoostingClassifier
    )
    evaluator = cross_fit_regression if task == "regression" else cross_fit_classification
    with pytest.raises(ValueError, match="early_stopping=False"):
        evaluator(
            loso_folds(groups),
            values,
            target,
            groups,
            Pipeline([("hgb", estimator(early_stopping=early_stopping))]),
            grid,
            inner=InnerSplit("subject", n_splits=2),
            seed=7,
        )


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_grouped_tuning_rejects_a_grid_that_enables_boosting_early_stopping(task):
    values = np.random.default_rng(31).normal(size=(24, 3))
    target = np.tile([0, 1], 12)
    groups = np.repeat(["a", "b", "c", "d"], 6).astype(object)
    estimator = (
        HistGradientBoostingRegressor if task == "regression" else HistGradientBoostingClassifier
    )
    evaluator = cross_fit_regression if task == "regression" else cross_fit_classification
    with pytest.raises(ValueError, match="early_stopping=False"):
        evaluator(
            loso_folds(groups),
            values,
            target,
            groups,
            Pipeline([("hgb", estimator(early_stopping=False))]),
            {"hgb__early_stopping": [True]},
            inner=InnerSplit("subject", n_splits=2),
            seed=7,
        )


def test_direct_grouped_tuning_rejects_boosting_early_stopping():
    values = np.random.default_rng(31).normal(size=(24, 3))
    target = values[:, 0]
    pipeline = Pipeline([("hgb", HistGradientBoostingRegressor(early_stopping=True))])
    with pytest.raises(ValueError, match="early_stopping=False"):
        tune(
            pipeline,
            {"hgb__max_iter": [2]},
            values,
            target,
            np.repeat(["a", "b", "c", "d"], 6).astype(object),
            split=InnerSplit("subject", n_splits=2),
            seed=7,
            fold=0,
        )


@pytest.mark.parametrize("early_stopping", [True, "auto"])
@pytest.mark.parametrize(
    "estimator", [HistGradientBoostingRegressor, HistGradientBoostingClassifier]
)
def test_direct_untuned_fit_preserves_boosting_early_stopping(estimator, early_stopping):
    values = np.random.default_rng(31).normal(size=(80, 3))
    target = np.tile([0, 1], 40)
    reference = estimator(early_stopping=early_stopping, max_iter=3, random_state=7)
    reference.fit(values, target)
    pipeline = Pipeline([("hgb", estimator(early_stopping=early_stopping, max_iter=3))])
    fitted = fit_untuned(pipeline, values, target, seed=7)
    np.testing.assert_array_equal(fitted.predict(values), reference.predict(values))
    assert pipeline.named_steps["hgb"].random_state is None


@pytest.mark.parametrize(
    "estimator",
    [SVC(probability=True), CalibratedClassifierCV(LogisticRegression(), cv=2)],
)
def test_direct_untuned_fit_preserves_existing_calibration_behavior(estimator):
    values = np.random.default_rng(31).normal(size=(24, 3))
    target = np.tile([0, 1], 12)
    fitted = fit_untuned(Pipeline([("model", estimator)]), values, target, seed=7)
    assert np.isfinite(fitted.predict_proba(values)).all()
