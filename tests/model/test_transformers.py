from __future__ import annotations

import numpy as np
import pytest

from eegtable.model.estimators import ridge_pipeline
from eegtable.model.transformers import (
    Deconfounder,
    DropAllNaNColumns,
    PreprocessingConfig,
    ReplaceInfWithNaN,
    SpatialFeatureSelector,
    VarianceThreshold,
    transform_feature_names,
    validate_subject_missingness,
)


def test_feature_names_follow_the_columns_through_a_covariate_pipeline() -> None:
    # With covariates the features pass through a ColumnTransformer. The names must still
    # drop the columns its steps drop, or importance lands on the wrong feature.
    rng = np.random.default_rng(0)
    values = rng.normal(size=(60, 5))
    values[:, 2] = 1.0
    with_covariate = np.column_stack([values, rng.normal(size=60)])
    pipe = ridge_pipeline(PreprocessingConfig(deconfound=True), seed=0, n_covariates=1)
    pipe.fit(with_covariate, rng.normal(size=60))
    names = transform_feature_names(pipe.steps[:-1], ["f0", "f1", "f2", "f3", "f4", "age"])
    assert names == ["f0", "f1", "f3", "f4"]


def test_a_subject_above_the_missingness_limit_is_named_in_the_error() -> None:
    values = np.array([[1.0, 2.0], [np.nan, np.nan], [np.nan, np.nan]])
    groups = np.array(["s1", "s2", "s2"], dtype=object)
    with pytest.raises(ValueError, match="s2"):
        validate_subject_missingness(values, groups, maximum=0.2)


def test_missingness_is_checked_per_subject_not_over_the_pooled_matrix() -> None:
    # Pooled missingness here is 25%, under the limit; s2's is 100%. A subject with no
    # usable features must fail even when the cohort average looks acceptable.
    values = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan]])
    groups = np.array(["s1", "s1", "s2"], dtype=object)
    with pytest.raises(ValueError, match="s2"):
        validate_subject_missingness(values, groups, maximum=0.3)


def test_missingness_requires_at_least_one_retained_feature() -> None:
    with pytest.raises(ValueError, match="at least one"):
        validate_subject_missingness(
            np.empty((2, 0)), np.array(["s1", "s2"], dtype=object), maximum=0.5
        )


def test_infinities_become_nan_so_imputation_can_see_them() -> None:
    out = ReplaceInfWithNaN().fit_transform(np.array([[1.0, np.inf], [-np.inf, 2.0]]))
    assert np.isnan(out[0, 1]) and np.isnan(out[1, 0])


def test_all_nan_columns_are_dropped_and_the_drop_is_learned_on_fit() -> None:
    # The columns to drop are decided by the training block and reapplied to test data,
    # so a column that happens to be present at test time is still dropped.
    train = np.array([[1.0, np.nan], [2.0, np.nan]])
    step = DropAllNaNColumns().fit(train)
    assert step.transform(np.array([[3.0, 9.0]])).shape == (1, 1)


@pytest.mark.parametrize("constant", [0.1, 1e-6])
def test_variance_threshold_drops_decimal_constants_without_a_variance_floor(constant) -> None:
    from sklearn.feature_selection import VarianceThreshold as ReferenceVarianceThreshold

    variable = np.arange(10) * 1e-12
    train = np.column_stack([np.full(10, constant), variable])
    step = VarianceThreshold().fit(train)

    np.testing.assert_array_equal(step.get_support(), [False, True])
    np.testing.assert_array_equal(
        step.get_support(), ReferenceVarianceThreshold().fit(train).get_support()
    )
    assert step.variances_[0] == 0.0
    np.testing.assert_array_equal(step.transform([[constant + 1.0, 3e-12]]), [[3e-12]])


def test_preprocessing_config_validates_missingness_bounds() -> None:
    with pytest.raises(ValueError, match="max_feature_missingness"):
        PreprocessingConfig(max_feature_missingness=1.5)
    with pytest.raises(ValueError, match="max_subject_missingness"):
        PreprocessingConfig(max_subject_missingness=-0.1)


def test_spatial_feature_selector_requires_feature_names_when_regions_are_requested() -> None:
    selector = SpatialFeatureSelector(allowed_regions=("insula",))
    with pytest.raises(ValueError, match="feature names"):
        selector.fit(np.ones((4, 3), dtype=float))


def test_deconfounder_removes_covariate_projection() -> None:
    cov = np.arange(10, dtype=float).reshape(-1, 1)
    feature = 2.0 * cov + 5.0
    X = np.column_stack([feature, cov])
    deconf = Deconfounder(n_covariates=1).fit(X)
    res = deconf.transform(X)
    assert res.shape == (10, 1)
    assert not np.allclose(res, feature)
    np.testing.assert_allclose(res, 0.0, atol=1e-10)


def test_deconfounding_is_invariant_to_covariate_units() -> None:
    rng = np.random.default_rng(3)
    covariates = rng.normal(size=(40, 2))
    features = (covariates @ np.array([3.0, 5.0]))[:, None]
    scaled = covariates * np.array([1e12, 1e-12])
    train, test = np.arange(30), np.arange(30, 40)

    for nuisance in (covariates, scaled):
        data = np.column_stack([features, nuisance])
        fitted = Deconfounder(n_covariates=2).fit(data[train])
        np.testing.assert_allclose(fitted.transform(data[test]), 0.0, atol=1e-10)
