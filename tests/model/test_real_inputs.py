from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from eegtable.model.aggregate import (
    AggregationConfig,
    bootstrap_mean_ci,
    paired_signflip_p_value,
    subject_level_errors,
    subject_level_r,
)
from eegtable.model.crossfit import cross_fit_regression
from eegtable.model.design import compute_train_group_intersection_mask, harmonize_fold
from eegtable.model.estimators import ridge_pipeline
from eegtable.model.importance import permutation_importance, shap_importance
from eegtable.model.nulls import (
    NullConfig,
    _prediction_statistic,
    changed_fraction,
    is_permutation_valid_run,
    permutation_test,
    permute,
)
from eegtable.model.residualize import (
    fit_nuisance_model,
    fit_staged_residual_preprocessor,
    reconstruct_staged_permutation_target_for_fold,
    residualize_targets,
    residualize_within_subjects,
)
from eegtable.model.screen import univariate_screen
from eegtable.model.splits import InnerSplit, loso_folds
from eegtable.model.transformers import (
    Deconfounder,
    DropAllNaNColumns,
    MissingnessThreshold,
    PreprocessingConfig,
    ReplaceInfWithNaN,
    SpatialFeatureSelector,
    VarianceThreshold,
    validate_subject_missingness,
)


@pytest.fixture
def real_inputs():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(36, 2))
    return {
        "X": features,
        "y": 2 * features[:, 0] + rng.normal(size=36),
        "covariates": rng.normal(size=(36, 1)),
        "groups": np.repeat(["a", "b", "c"], 12).astype(object),
        "train": np.arange(24),
        "test": np.arange(24, 36),
    }


@pytest.mark.parametrize("function", [bootstrap_mean_ci, paired_signflip_p_value])
def test_subject_inference_rejects_complex_values(function):
    with pytest.raises(ValueError, match="real|complex"):
        function(np.array([1 + 1j, 2 + 2j, 3 + 3j]), iterations=10, seed=0)


@pytest.mark.parametrize("function", [subject_level_r, subject_level_errors])
@pytest.mark.parametrize("column", ["y_true", "y_pred"])
def test_subject_metrics_reject_complex_columns(function, column, real_inputs):
    frame = pd.DataFrame(
        {
            "subject_id": real_inputs["groups"],
            "y_true": real_inputs["y"],
            "y_pred": real_inputs["y"],
        }
    )
    frame[column] = frame[column].to_numpy() * (1 + 1j)
    with pytest.raises(ValueError, match="real|complex"):
        function(frame)


@pytest.mark.parametrize("argument", ["X", "y"])
def test_univariate_screen_rejects_complex_data(argument, real_inputs):
    arguments = {key: real_inputs[key] for key in ("X", "y", "groups")}
    arguments[argument] = arguments[argument] * (1 + 1j)
    with pytest.raises(ValueError, match="real|complex"):
        univariate_screen(**arguments, n_flips=10)


@pytest.mark.parametrize("function", [fit_nuisance_model, residualize_targets])
@pytest.mark.parametrize("argument", ["y", "covariates"])
def test_nuisance_fitting_rejects_complex_data(function, argument, real_inputs):
    arguments = {key: value for key, value in real_inputs.items() if key not in ("X", "groups")}
    arguments[argument] = arguments[argument] * (1 + 1j)
    with pytest.raises(ValueError, match="real|complex"):
        function(**arguments, columns=("covariate",))


@pytest.mark.parametrize("argument", ["values", "covariates"])
def test_subject_residualization_rejects_complex_data(argument, real_inputs):
    arguments = {key: value for key, value in real_inputs.items() if key != "X"}
    arguments["values"] = arguments.pop("y")
    arguments[argument] = arguments[argument] * (1 + 1j)
    with pytest.raises(ValueError, match="real|complex"):
        residualize_within_subjects(**arguments, columns=("covariate",))


@pytest.mark.parametrize("argument", ["X", "y", "covariates"])
def test_staged_preprocessing_rejects_complex_data(argument, real_inputs):
    arguments = {key: value for key, value in real_inputs.items() if key not in ("train", "test")}
    arguments[argument] = arguments[argument] * (1 + 1j)
    with pytest.raises(ValueError, match="real|complex"):
        fit_staged_residual_preprocessor(
            **arguments, rows=real_inputs["train"], columns=("covariate",)
        )


@pytest.mark.parametrize("argument", ["y", "covariates"])
def test_staged_permutations_reject_complex_data(argument, real_inputs):
    arguments = {key: value for key, value in real_inputs.items() if key not in ("X", "groups")}
    arguments[argument] = arguments[argument] * (1 + 1j)
    with pytest.raises(ValueError, match="real|complex"):
        reconstruct_staged_permutation_target_for_fold(
            **arguments, columns=("covariate",), permutation_indices=np.arange(36)
        )


@pytest.mark.parametrize("argument", ["X_train", "X_test"])
def test_harmonization_rejects_complex_data(argument, real_inputs):
    arguments = {
        "X_train": real_inputs["X"][:24],
        "X_test": real_inputs["X"][24:],
        "groups_train": real_inputs["groups"][:24],
    }
    arguments[argument] = arguments[argument] * (1 + 1j)
    with pytest.raises(ValueError, match="real|complex"):
        harmonize_fold(**arguments, mode="intersection")


def test_training_intersection_rejects_complex_data(real_inputs):
    with pytest.raises(ValueError, match="real|complex"):
        compute_train_group_intersection_mask(real_inputs["X"] * (1 + 1j), real_inputs["groups"])


@pytest.mark.parametrize("argument", ["y_original", "y_permuted"])
def test_permutation_changed_fraction_rejects_complex_data(argument, real_inputs):
    arguments = {"y_original": real_inputs["y"], "y_permuted": real_inputs["y"]}
    arguments[argument] = arguments[argument] * (1 + 1j)
    with pytest.raises(ValueError, match="real|complex"):
        changed_fraction(**arguments)


def test_permutation_run_validation_rejects_complex_indices():
    with pytest.raises(ValueError, match="real|complex"):
        is_permutation_valid_run(np.array([1 + 1j] * 8))


def test_permute_rejects_complex_targets(real_inputs):
    with pytest.raises(ValueError, match="real|complex"):
        permute(
            real_inputs["y"] * (1 + 1j),
            real_inputs["groups"],
            config=NullConfig(),
            rng=np.random.default_rng(0),
        )


def test_permutation_inference_rejects_complex_targets(real_inputs):
    features, target, groups = (real_inputs[key] for key in ("X", "y", "groups"))
    folds, inner = loso_folds(groups), InnerSplit("subject", n_splits=2)
    pipeline = ridge_pipeline(PreprocessingConfig(), seed=0)
    predictions = cross_fit_regression(
        folds, features, target, groups, pipeline, {}, inner=inner, seed=0
    )
    observed = _prediction_statistic(predictions, groups, AggregationConfig(), None)
    with pytest.raises(ValueError, match="real|complex"):
        permutation_test(
            folds,
            features,
            target * (1 + 1j),
            groups,
            None,
            pipeline,
            {},
            observed,
            config=NullConfig(n_permutations=2),
            inner=inner,
            seed=0,
        )


def test_cross_fitting_rejects_complex_features(real_inputs):
    with pytest.raises(ValueError, match="real|complex"):
        cross_fit_regression(
            loso_folds(real_inputs["groups"]),
            real_inputs["X"] * (1 + 1j),
            real_inputs["y"],
            real_inputs["groups"],
            ridge_pipeline(PreprocessingConfig(), seed=0),
            {},
            inner=InnerSplit("subject", n_splits=2),
            seed=0,
        )


def test_cross_fitting_preserves_real_input_precision(real_inputs):
    features = real_inputs["X"].astype(np.float32)
    target = real_inputs["y"].astype(np.float32)
    groups = real_inputs["groups"]
    folds = loso_folds(groups)
    pipeline = Pipeline([("regressor", Ridge(alpha=1.0))])
    predictions = cross_fit_regression(
        folds,
        features,
        target,
        groups,
        pipeline,
        {},
        inner=InnerSplit("subject", n_splits=2),
        seed=0,
    )
    for fold, prediction in zip(folds, predictions, strict=True):
        fitted = clone(pipeline).fit(features[fold.train], target[fold.train])
        np.testing.assert_array_equal(prediction.y_pred, fitted.predict(features[fold.test]))


@pytest.mark.parametrize("function", [permutation_importance, shap_importance])
def test_importance_rejects_complex_features(function, real_inputs):
    if function is shap_importance and importlib.util.find_spec("shap") is None:
        pytest.skip("shap not installed")
    model = Pipeline([("regressor", Ridge())]).fit(real_inputs["X"], real_inputs["y"])
    with pytest.raises(ValueError, match="real|complex"):
        if function is shap_importance:
            function(model, real_inputs["X"] * (1 + 1j), ("one", "two"))
        else:
            function(model, real_inputs["X"] * (1 + 1j), real_inputs["y"], n_repeats=2)


@pytest.mark.parametrize(
    "transformer",
    [
        ReplaceInfWithNaN(),
        DropAllNaNColumns(),
        MissingnessThreshold(),
        VarianceThreshold(),
        SpatialFeatureSelector(),
        Deconfounder(n_covariates=1),
    ],
)
@pytest.mark.parametrize("operation", ["fit", "transform"])
def test_transformers_reject_complex_features(transformer, operation, real_inputs):
    if operation == "transform":
        transformer.fit(real_inputs["X"])
    with pytest.raises(ValueError, match="real|complex"):
        getattr(transformer, operation)(real_inputs["X"] * (1 + 1j))


def test_subject_missingness_rejects_complex_features(real_inputs):
    with pytest.raises(ValueError, match="real|complex"):
        validate_subject_missingness(
            real_inputs["X"] * (1 + 1j), real_inputs["groups"], maximum=0.5
        )
