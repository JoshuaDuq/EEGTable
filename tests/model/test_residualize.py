from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eegtable.model.residualize import (
    FoldNuisanceFit,
    fit_nuisance_model,
    fit_staged_residual_preprocessor,
    reconstruct_staged_permutation_target_for_fold,
    residualize_targets,
    residualize_within_subjects,
)
from eegtable.model.transformers import PreprocessingConfig

TRAIN = np.arange(8, dtype=np.intp)
TEST = np.arange(8, 12, dtype=np.intp)


@pytest.mark.parametrize("method", ["pooled", "within", "staged"])
@pytest.mark.parametrize(
    "train, message",
    [
        (np.arange(8) + 0.25, "integer"),
        (np.array([0, 1, 2, 3, 4, 5, 6, 6]), "duplicate"),
    ],
)
def test_nuisance_fitting_rejects_invalid_training_rows(method, train, message):
    y = np.random.default_rng(4).normal(size=12)
    covariates = np.arange(12.0).reshape(-1, 1)
    groups = np.full(12, "subject", dtype=object)
    with pytest.raises(ValueError, match=message):
        if method == "pooled":
            fit_nuisance_model(y, covariates, train, TEST, columns=("c",))
        elif method == "within":
            residualize_within_subjects(y, covariates, groups, train, TEST, columns=("c",))
        else:
            fit_staged_residual_preprocessor(
                X=y[:, None],
                y=y,
                covariates=covariates,
                groups=groups,
                rows=train,
                columns=("c",),
            )


@pytest.mark.parametrize("method", ["pooled", "within", "staged"])
def test_nuisance_fitting_requires_covariates_aligned_to_all_rows(method):
    y = np.random.default_rng(4).normal(size=12)
    covariates = np.arange(13.0).reshape(-1, 1)
    groups = np.full(12, "subject", dtype=object)
    with pytest.raises(ValueError, match="covariates.*rows"):
        if method == "pooled":
            fit_nuisance_model(y, covariates, TRAIN, TEST, columns=("c",))
        elif method == "within":
            residualize_within_subjects(y, covariates, groups, TRAIN, TEST, columns=("c",))
        else:
            fit_staged_residual_preprocessor(
                X=y[:, None],
                y=y,
                covariates=covariates,
                groups=groups,
                rows=TRAIN,
                columns=("c",),
            )


@pytest.mark.parametrize("invalid", ["train", "permutation"])
def test_staged_permutation_rejects_fractional_row_indices(invalid):
    y = np.random.default_rng(4).normal(size=12)
    train = TRAIN + 0.25 if invalid == "train" else TRAIN
    source = np.arange(12) + 0.25 if invalid == "permutation" else np.arange(12)
    with pytest.raises(ValueError, match="integer"):
        reconstruct_staged_permutation_target_for_fold(
            y=y,
            covariates=np.arange(12.0).reshape(-1, 1),
            train=train,
            test=TEST,
            columns=("c",),
            permutation_indices=source,
        )


@pytest.mark.parametrize(
    "feature_shape, group_shape, message",
    [
        ((13, 2), (12,), "X"),
        ((12, 2), (13,), "groups"),
        ((12,), (12,), "X"),
        ((12, 2), (12, 1), "groups"),
    ],
)
def test_staged_preprocessing_requires_all_inputs_aligned_to_target(
    feature_shape, group_shape, message
):
    rng = np.random.default_rng(4)
    y = rng.normal(size=12)
    X = rng.normal(size=feature_shape)
    groups = np.full(group_shape, "subject", dtype=object)

    with pytest.raises(ValueError, match=message):
        fit_staged_residual_preprocessor(
            X=X,
            y=y,
            covariates=np.arange(12.0).reshape(-1, 1),
            groups=groups,
            rows=TRAIN,
            columns=("c",),
        )


def test_nuisance_removal_is_invariant_to_covariate_units() -> None:
    covariates = np.random.default_rng(3).normal(size=(40, 2))
    y = 2 + covariates @ np.array([3.0, 5.0])
    fitted = fit_nuisance_model(
        y,
        covariates * [1e12, 1e-12],
        np.arange(30),
        np.arange(30, 40),
        columns=("a", "b"),
    )
    np.testing.assert_allclose(fitted.train_residual, 0, atol=1e-10)
    np.testing.assert_allclose(fitted.test_residual, 0, atol=1e-10)


def test_staged_nuisance_removal_is_invariant_to_covariate_units() -> None:
    rng = np.random.default_rng(3)
    covariates = rng.normal(size=(40, 2))
    X = covariates @ np.array([[3.0, 1.0], [5.0, 2.0]])
    y = X[:, 0] + rng.normal(size=40)
    groups = np.repeat(["a", "b", "c", "d"], 10)
    scaled = covariates * [1e12, 1e-12]
    fitted = fit_staged_residual_preprocessor(
        X=X,
        y=y,
        covariates=scaled,
        groups=groups,
        rows=np.arange(30),
        columns=("a", "b"),
    )
    np.testing.assert_allclose(
        fitted.transform_features(X, scaled, np.arange(40), groups),
        0,
        atol=1e-10,
    )
    reference = fit_nuisance_model(
        y, covariates, np.arange(30), np.arange(30, 40), columns=("a", "b")
    )
    np.testing.assert_allclose(
        fitted.nuisance_prediction(scaled, np.arange(30, 40)),
        reference.test_prediction,
        atol=1e-10,
    )


def test_the_nuisance_model_is_fitted_on_training_rows_only() -> None:
    # If the test rows entered the fit, the test residuals would be centred by
    # construction and the held-out score would be optimistic.
    y = np.concatenate([np.arange(8, dtype=float), np.full(4, 100.0)])
    covariates = np.concatenate([np.arange(8, dtype=float), np.zeros(4)]).reshape(-1, 1)
    _, y_test = residualize_targets(y, covariates, TRAIN, TEST, columns=["c"])
    assert np.all(y_test > 50.0)


def test_a_covariate_that_explains_the_target_leaves_near_zero_residuals() -> None:
    y = np.arange(12, dtype=float)
    covariates = np.arange(12, dtype=float).reshape(-1, 1)
    y_train, _ = residualize_targets(y, covariates, TRAIN, TEST, columns=["c"])
    assert np.allclose(y_train, 0.0, atol=1e-9)


def test_a_rank_deficient_training_design_is_refused() -> None:
    # A constant covariate adds nothing to the intercept; silently inverting a singular
    # design would return residuals that depend on the pseudo-inverse's tie-breaking.
    y = np.arange(12, dtype=float)
    covariates = np.ones((12, 1))
    with pytest.raises(ValueError, match="rank deficient"):
        residualize_targets(y, covariates, TRAIN, TEST, columns=["c"])


def test_train_and_test_indices_may_not_overlap() -> None:
    y = np.arange(12, dtype=float)
    covariates = np.arange(12, dtype=float).reshape(-1, 1)
    with pytest.raises(ValueError, match="overlap"):
        residualize_targets(y, covariates, TRAIN, TRAIN, columns=["c"])


def test_fit_nuisance_model_accepts_dataframe_and_populates_fit() -> None:
    y = np.asarray([100.0, 110.0, 0.0, 10.0, 0.0, 10.0], dtype=float)
    meta = pd.DataFrame(
        {
            "nuisance": [0, 1, 0, 1, 0, 1],
            "subject": ["s1", "s1", "s2", "s2", "s3", "s3"],
        }
    )
    train_idx = np.asarray([2, 3, 4, 5], dtype=np.intp)
    test_idx = np.asarray([0, 1], dtype=np.intp)
    fit = fit_nuisance_model(y, meta, train_idx, test_idx, columns=["nuisance"])
    assert isinstance(fit, FoldNuisanceFit)
    assert np.allclose(fit.train_residual, 0.0)
    assert np.allclose(fit.test_residual, 100.0)
    assert fit.details["columns"] == ["nuisance"]
    assert fit.details["n_train"] == 4
    assert fit.details["n_test"] == 2


def test_residualize_targets_checks_rank_only_on_training_design() -> None:
    y = np.asarray([0.0, 1.0, 2.0, 3.0, 10.0], dtype=float)
    meta = pd.DataFrame({"nuisance": [0.0, 1.0, 2.0, 3.0, 1.0]})
    y_train, y_test = residualize_targets(
        y,
        meta,
        np.asarray([0, 1, 2, 3], dtype=np.intp),
        np.asarray([4], dtype=np.intp),
        columns=["nuisance"],
    )
    assert np.allclose(y_train, 0.0)
    assert y_test.shape == (1,)


def test_missing_column_raises_error() -> None:
    meta = pd.DataFrame({"nuisance": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="missing"):
        residualize_targets(
            np.arange(3.0),
            meta,
            np.asarray([0, 1], dtype=np.intp),
            np.asarray([2], dtype=np.intp),
            columns=["unknown"],
        )


def test_reconstruct_staged_permutation_target_shifts_residuals_within_fold() -> None:
    y = np.asarray([10.0, 25.0, 30.0, 40.0], dtype=float)
    covariates = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float).reshape(-1, 1)
    train_idx = np.asarray([0, 1, 2], dtype=np.intp)
    test_idx = np.asarray([3], dtype=np.intp)
    perm_indices = np.asarray([1, 2, 0, 3], dtype=np.intp)
    y_perm = reconstruct_staged_permutation_target_for_fold(
        y=y,
        covariates=covariates,
        train=train_idx,
        test=test_idx,
        columns=["c"],
        permutation_indices=perm_indices,
    )
    assert y_perm.shape == y.shape
    assert not np.allclose(y_perm, y)
    np.testing.assert_allclose(y_perm, [15.0, 20.0, 30.0, 40.0])


def test_staged_preprocessor_transforms_features_and_target() -> None:
    rng = np.random.default_rng(42)
    n = 30
    nuisance = rng.normal(size=n)
    X = np.column_stack([nuisance * 3.0, rng.normal(size=n)])
    y = 5.0 + 2.0 * nuisance + rng.normal(scale=0.1, size=n)
    groups = np.repeat(["sub-01", "sub-02", "sub-03"], 10)
    meta = pd.DataFrame({"nuisance": nuisance})
    train_rows = np.arange(20, dtype=np.intp)
    test_rows = np.arange(20, 30, dtype=np.intp)

    preprocessor = fit_staged_residual_preprocessor(
        X=X,
        y=y,
        covariates=meta,
        groups=groups,
        rows=train_rows,
        columns=["nuisance"],
    )
    X_train_res = preprocessor.transform_features(X, meta, train_rows, groups)
    assert X_train_res.shape == (20, 2)
    assert not np.allclose(X_train_res, X[train_rows])
    np.testing.assert_allclose(X_train_res[:, 0], 0.0, atol=1e-10)

    y_test_trans = preprocessor.transform_target(y, meta, test_rows)
    assert y_test_trans.shape == (10,)
    y_test_inv = preprocessor.inverse_transform_target(y_test_trans)
    assert y_test_inv.shape == (10,)


def test_staged_imputation_rejects_excessive_subject_missingness() -> None:
    n_subjects = 30
    n_trials = 10
    n = n_subjects * n_trials
    X = np.ones((n, 2))
    groups = np.repeat([f"sub-{i:02d}" for i in range(n_subjects)], n_trials)
    # One subject with missing EEG is 10/300 = 3.3% feature missingness (under 5% limit),
    # but 100% subject missingness (exceeds 10% limit).
    X[groups == "sub-00"] = np.nan
    y = np.ones(n)
    meta = pd.DataFrame({"nuisance": np.arange(n, dtype=float)})
    train_rows = np.arange(n, dtype=np.intp)

    with pytest.raises(ValueError, match="Subject sub-00 has missingness"):
        fit_staged_residual_preprocessor(
            X=X,
            y=y,
            covariates=meta,
            groups=groups,
            rows=train_rows,
            columns=["nuisance"],
            config=PreprocessingConfig(max_subject_missingness=0.10),
        )


def test_a_constant_covariate_is_refused_even_when_its_value_is_inexact_in_binary() -> None:
    # 0.1 has no exact binary form, so centring twelve copies leaves rounding residue
    # rather than zeros. The rank check must still see a constant column instead of
    # rescaling the residue into one that looks informative.
    y = np.arange(16, dtype=float)
    covariates = np.full((16, 1), 0.1)
    with pytest.raises(ValueError, match="rank deficient"):
        residualize_targets(
            y,
            covariates,
            np.arange(12, dtype=np.intp),
            np.arange(12, 16, dtype=np.intp),
            columns=["c"],
        )


# s1 trains on 8 trials and is tested on 2; s2 is held out whole, with 8 trials.
SUBJECTS = np.repeat(np.array(["s1", "s2"], dtype=object), [10, 8])
NUISANCE = np.r_[np.arange(10.0) % 4, [5.0, 1.0, 3.0, 2.0, 4.0, 0.0, 2.0, 6.0]].reshape(-1, 1)
SPLIT_TRAIN = np.arange(8, dtype=np.intp)
SPLIT_TEST = np.arange(8, 18, dtype=np.intp)


def _within(values, nuisance=NUISANCE, columns=("n",)):
    return residualize_within_subjects(
        values, nuisance, SUBJECTS, SPLIT_TRAIN, SPLIT_TEST, columns=columns
    )


@pytest.mark.parametrize("shape", [(18,), (18, 2)])
def test_a_subject_with_training_rows_never_sees_its_held_out_values(shape) -> None:
    values = np.random.default_rng(0).normal(size=shape)
    changed = values.copy()
    changed[8:10] += 1e14
    np.testing.assert_array_equal(_within(values)[0], _within(changed)[0])


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("row", [0, 8])
def test_nonfinite_features_stay_missing_without_erasing_finite_residuals(missing, row) -> None:
    values = np.random.default_rng(2).normal(size=(18, 2))
    values[row, 1] = missing
    expected = values.copy()
    expected[row, 1] = np.nan

    actual = np.concatenate(_within(values))
    reference = np.concatenate(_within(expected))

    assert np.isnan(actual[row, 1])
    np.testing.assert_allclose(actual, reference, equal_nan=True)


def test_a_subject_held_out_whole_is_fitted_on_its_own_rows() -> None:
    # Its residual target is then orthogonal to its own nuisance design, whatever the others do.
    _, test = _within(np.random.default_rng(1).normal(size=18))
    design = np.column_stack([np.ones(8), NUISANCE[10:, 0]])
    np.testing.assert_allclose(design.T @ test[2:], 0.0, atol=1e-10)


def test_a_missing_feature_value_stays_missing_and_the_rest_is_residualized() -> None:
    X = np.random.default_rng(2).normal(size=(18, 2))
    X[11, 1] = np.nan
    _, test = _within(X)
    assert np.isnan(test[3, 1])
    finite = np.array([10, 12, 13, 14, 15, 16, 17])
    design = np.column_stack([np.ones(finite.size), NUISANCE[finite, 0]])
    np.testing.assert_allclose(design.T @ test[finite - 8, 1], 0.0, atol=1e-10)


def test_a_nuisance_column_constant_within_a_subject_is_projected_out_not_refused() -> None:
    # A subject may meet a single level of a nuisance factor; its residual is still defined.
    level = np.r_[np.arange(10.0) % 2, np.full(8, 3.0)]
    _, test = _within(
        np.random.default_rng(3).normal(size=18),
        np.column_stack([NUISANCE[:, 0], level]),
        columns=("n", "level"),
    )
    np.testing.assert_allclose(test[2:].sum(), 0.0, atol=1e-10)


def test_a_value_constant_within_a_subject_leaves_an_exact_zero_residual() -> None:
    # Rounding leaves about 1e-15 behind, which anything downstream reads as variation.
    X = np.random.default_rng(5).normal(size=(18, 2))
    X[:, 1] = 3.7
    train, test = _within(X)
    assert np.all(train[:, 1] == 0.0) and np.all(test[:, 1] == 0.0)
    train, test = _within(np.full(18, 2.5))
    assert np.all(train == 0.0) and np.all(test == 0.0)


def test_a_subject_left_without_residual_degrees_of_freedom_is_named() -> None:
    # Six nuisance terms fit s2's 8 trials all but exactly, so nothing is left to correlate.
    rng = np.random.default_rng(4)
    nuisance = rng.normal(size=(18, 6))
    columns = tuple(f"c{i}" for i in range(6))
    y = rng.normal(size=18)
    with pytest.raises(ValueError, match="Subject s2: .*fewer than 3 residual degrees"):
        residualize_within_subjects(
            y, nuisance, SUBJECTS, np.arange(10, dtype=np.intp), np.arange(10, 18), columns=columns
        )
