from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from eegtable.model.crossfit import cross_fit_regression
from eegtable.model.estimators import (
    elasticnet_grid,
    elasticnet_pipeline,
    ensemble_pipeline,
    logistic_grid,
    logistic_pipeline,
    random_forest_classifier_grid,
    random_forest_classifier_pipeline,
    random_forest_grid,
    random_forest_pipeline,
    ridge_grid,
    ridge_pipeline,
    svm_grid,
    svm_pipeline,
)
from eegtable.model.splits import InnerSplit, loso_folds
from eegtable.model.transformers import PreprocessingConfig

_CONFIG = PreprocessingConfig()


def test_scaled_ridge_matches_penalty_from_retained_training_dimensions() -> None:
    from sklearn.linear_model import Ridge

    from eegtable.model.estimators import scaled_ridge_pipeline

    rng = np.random.default_rng(42)
    values = np.column_stack([rng.normal(size=(20, 3)), np.ones(20)])
    target = rng.normal(size=20)
    pipeline = scaled_ridge_pipeline(_CONFIG, seed=42)
    pipeline.set_params(regressor__alpha=0.1).fit(values, target)
    transformed = pipeline[:-1].transform(values)
    reference = Ridge(alpha=0.1 * transformed.shape[0] * transformed.shape[1])
    reference.fit(transformed, target)
    np.testing.assert_allclose(pipeline.predict(values), reference.predict(transformed))
    assert pipeline.named_steps["regressor"].alpha == 0.1


def test_scaled_ridge_uses_inner_training_dimensions_after_pca(monkeypatch) -> None:
    from eegtable.model.estimators import scaled_ridge_pipeline

    rng = np.random.default_rng(42)
    values = np.column_stack([rng.normal(size=(24, 4)), np.ones(24)])
    target = values[:, 0] + rng.normal(scale=0.2, size=24)
    groups = np.repeat(["a", "b", "c", "d"], 6).astype(object)
    pipeline = scaled_ridge_pipeline(
        PreprocessingConfig(pca_enabled=True, pca_n_components=2), seed=42
    )
    regressor_type = type(pipeline.named_steps["regressor"])
    fit = regressor_type.fit
    fitted_penalties = []

    def record_fit(self, X, y, **fit_params):
        result = fit(self, X, y, **fit_params)
        fitted_penalties.append((len(X), X.shape[1], self.alpha, self.alpha_))
        return result

    monkeypatch.setattr(regressor_type, "fit", record_fit)
    cross_fit_regression(
        loso_folds(groups),
        values,
        target,
        groups,
        pipeline,
        {"regressor__alpha": [0.01, 0.1]},
        inner=InnerSplit("subject", n_splits=2),
        seed=42,
        scoring="neg_mean_squared_error",
    )
    assert {rows for rows, _, _, _ in fitted_penalties} == {6, 12, 18}
    for rows, columns, alpha, effective_alpha in fitted_penalties:
        assert columns == 2
        assert effective_alpha == pytest.approx(alpha * rows * columns)


def test_scaled_ridge_is_invariant_to_replicating_training_rows() -> None:
    from sklearn.base import clone

    from eegtable.model.estimators import scaled_ridge_pipeline

    rng = np.random.default_rng(31)
    values, target = rng.normal(size=(12, 3)), rng.normal(size=12)
    pipeline = scaled_ridge_pipeline(_CONFIG, seed=42)
    fitted = clone(pipeline).fit(values, target)
    repeated = clone(pipeline).fit(np.tile(values, (3, 1)), np.tile(target, 3))
    np.testing.assert_allclose(fitted.predict(values), repeated.predict(values))


def test_the_pipeline_scales_before_it_regularizes() -> None:
    # ElasticNet penalizes coefficients on their own scale, so an unscaled feature in
    # different units is regularized differently from an identical one in volts.
    names = [name for name, _ in elasticnet_pipeline(_CONFIG, seed=0).steps]
    assert names.index("scaler") < names.index("regressor")


def test_the_seed_reaches_the_estimator() -> None:
    pipe = elasticnet_pipeline(_CONFIG, seed=17)
    assert pipe.named_steps["regressor"].random_state == 17


def test_pipelines_are_built_fresh_not_shared() -> None:
    assert elasticnet_pipeline(_CONFIG, seed=0) is not elasticnet_pipeline(_CONFIG, seed=0)


@pytest.mark.parametrize(
    ("pipeline_factory", "grid_factory"),
    [
        (elasticnet_pipeline, elasticnet_grid),
        (ridge_pipeline, lambda: ridge_grid(np.ones((10, 3)))),
        (random_forest_pipeline, random_forest_grid),
        (svm_pipeline, svm_grid),
        (logistic_pipeline, logistic_grid),
        (random_forest_classifier_pipeline, random_forest_classifier_grid),
    ],
)
@pytest.mark.parametrize("n_covariates", [0, 2])
def test_every_grid_key_names_a_step_that_exists_in_its_pipeline(
    pipeline_factory: Callable[..., Pipeline],
    grid_factory: Callable[..., dict[str, list[object]]],
    n_covariates: int,
) -> None:
    # A grid key that does not resolve is not an error in GridSearchCV until fit time,
    # and then it reports a parameter name rather than the typo that caused it.
    pipe = pipeline_factory(_CONFIG, seed=42, n_covariates=n_covariates)
    grid = grid_factory()
    valid_params = set(pipe.get_params())
    assert set(grid.keys()) <= valid_params


def test_default_grids_give_the_same_predictions_whatever_the_feature_units() -> None:
    # A FeatureTable mixes units (V^2, log power, ratios, PLV). A variance threshold on
    # unscaled values selected features by their unit, and PLV-like columns, with variance
    # near 0.003, were removed entirely, which crashed the fold.
    rng = np.random.default_rng(0)
    groups = np.repeat([f"s{i}" for i in range(6)], 20).astype(object)
    X = rng.uniform(0.3, 0.5, size=(120, 8))
    y = 10.0 * X[:, 0] + rng.normal(size=120)
    folds = loso_folds(groups)
    inner = InnerSplit(grouping="subject", n_splits=3)
    pipe = ridge_pipeline(_CONFIG, seed=0)
    as_given, rescaled = (
        cross_fit_regression(folds, values, y, groups, pipe, ridge_grid(X), inner=inner, seed=0)
        for values in (X, X * 1e-6)
    )
    np.testing.assert_allclose(
        np.concatenate([p.y_pred for p in as_given]),
        np.concatenate([p.y_pred for p in rescaled]),
        rtol=1e-6,
    )


def _effective_dof(X: np.ndarray, alpha: float) -> float:
    # Degrees of freedom of a ridge fit on the standardized design the pipeline fits.
    standardized = (X - X.mean(axis=0)) / X.std(axis=0)
    s = np.linalg.svd(standardized, compute_uv=False)
    return float(np.sum(s**2 / (s**2 + alpha)))


@pytest.mark.parametrize("shape", [(1200, 40), (300, 2000)])
def test_the_ridge_grid_spans_unpenalized_to_empty_whatever_the_design_size(shape) -> None:
    # scikit-learn's Ridge does not divide its penalty by the number of trials, so a fixed
    # grid that shrinks a small design barely touches a large one: at 1,200 trials the old
    # largest penalty, 100, still left 37 of 40 degrees of freedom in place.
    X = np.random.default_rng(0).normal(size=shape)
    alphas = ridge_grid(X)["regressor__alpha"]
    rank = min(shape[0] - 1, shape[1])
    assert _effective_dof(X, min(alphas)) > 0.9 * rank
    assert _effective_dof(X, max(alphas)) < 0.1


def test_the_ridge_grid_ignores_columns_the_pipeline_drops_as_empty() -> None:
    X = np.random.default_rng(0).normal(size=(100, 20))
    padded = np.column_stack([X, np.full((100, 20), np.nan)])
    assert ridge_grid(padded) == ridge_grid(X)


def test_elastic_net_selects_the_same_model_whatever_the_target_units() -> None:
    # ElasticNet's L1 penalty does not scale with the target, so a fixed grid shrinks a
    # rating in points and the same rating in hundredths of a point differently: rescaling
    # the target by 1,000 changed the selected alpha and cut the subject-level r by 0.1.
    rng = np.random.default_rng(0)
    groups = np.repeat([f"s{i}" for i in range(6)], 30).astype(object)
    X = rng.normal(size=(180, 30))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=180)
    folds = loso_folds(groups)
    inner = InnerSplit(grouping="subject", n_splits=3)
    pipe = elasticnet_pipeline(_CONFIG, seed=0)
    as_given, rescaled = (
        cross_fit_regression(
            folds, X, scale * y, groups, pipe, elasticnet_grid(), inner=inner, seed=0
        )
        for scale in (1.0, 1000.0)
    )
    assert [p.best_params for p in as_given] == [p.best_params for p in rescaled]
    np.testing.assert_allclose(
        1000.0 * np.concatenate([p.y_pred for p in as_given]),
        np.concatenate([p.y_pred for p in rescaled]),
        rtol=1e-6,
    )


def test_ensemble_pipeline_contains_all_base_classifiers() -> None:
    pipe = ensemble_pipeline(_CONFIG, seed=42)
    ensemble = pipe.named_steps["ensemble"]
    named_estimators = dict(ensemble.estimators)
    assert "svm" in named_estimators
    assert "lr" in named_estimators
    assert "rf" in named_estimators


def test_ensemble_uses_hard_votes_without_trial_wise_svm_calibration() -> None:
    ensemble = ensemble_pipeline(_CONFIG, seed=42).named_steps["ensemble"]
    assert ensemble.voting == "hard"
    assert dict(ensemble.estimators)["svm"].probability is False


def test_svm_pipeline_does_not_calibrate_with_hidden_trial_folds() -> None:
    assert svm_pipeline(_CONFIG, seed=42).named_steps["svm"].probability is False


def test_ensemble_refuses_calibration_without_group_disjoint_splits() -> None:
    with pytest.raises(ValueError, match="group-disjoint calibration"):
        ensemble_pipeline(_CONFIG, seed=42, calibrate_ensemble=True)


def test_logistic_elasticnet_grid_includes_l1_ratio() -> None:
    pipe = logistic_pipeline(_CONFIG, seed=42, penalty="elasticnet")
    grid = logistic_grid(penalty="elasticnet")
    assert "lr__l1_ratio" in grid
    assert set(grid.keys()) <= set(pipe.get_params())


@pytest.mark.parametrize("penalty", ["typo", "L2", "ridge"])
def test_logistic_pipeline_refuses_an_unknown_penalty(penalty: str) -> None:
    # On scikit-learn >= 1.8 the penalty is translated rather than passed through, so an
    # unrecognised value used to fall past every branch and fit a default L2 model in silence.
    with pytest.raises(ValueError, match="penalty must be one of"):
        logistic_pipeline(_CONFIG, seed=0, penalty=penalty)


def test_an_unpenalized_logistic_model_has_no_strength_to_tune() -> None:
    # Tuning C would refit the unpenalized model as an L2 one under the same name.
    assert "lr__C" not in logistic_grid(penalty="none")


def test_unpenalized_logistic_regression_uses_none_before_scikit_learn_1_8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # scikit-learn 1.4 removed the string "none"; from 1.2 the unpenalized model is None.
    from eegtable.model import estimators

    monkeypatch.setattr(estimators.sklearn, "__version__", "1.7.2")
    assert estimators._get_lr_kwargs("none") == {"penalty": None}
