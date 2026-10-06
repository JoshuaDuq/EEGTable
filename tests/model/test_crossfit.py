from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eegtable.model.aggregate import fold_results, subject_level_r
from eegtable.model.crossfit import cross_fit_regression
from eegtable.model.estimators import ridge_pipeline
from eegtable.model.splits import InnerSplit, loso_folds, within_subject_folds
from eegtable.model.transformers import PreprocessingConfig

PIPE = Pipeline([("regressor", DummyRegressor(strategy="mean"))])
GRID = {"regressor__strategy": ["mean", "median"]}
GROUPS = np.repeat(["s1", "s2", "s3", "s4"], 8).astype(object)
RUNS = np.tile(np.repeat(["r1", "r2", "r3", "r4"], 2), 4).astype(object)
X = np.arange(GROUPS.size, dtype=float).reshape(-1, 1)
Y = np.arange(GROUPS.size, dtype=float)
BY_SUBJECT = InnerSplit(grouping="subject", n_splits=2)
BY_RUN = InnerSplit(grouping="run", n_splits=2)


def test_every_trial_is_predicted_exactly_once_across_folds() -> None:
    predictions = cross_fit_regression(
        loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=BY_SUBJECT, seed=0
    )
    rows = np.concatenate([p.rows for p in predictions])
    np.testing.assert_array_equal(np.sort(rows), np.arange(GROUPS.size))


def test_no_fold_is_fitted_on_the_subject_it_predicts() -> None:
    for prediction in cross_fit_regression(
        loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=BY_SUBJECT, seed=0
    ):
        assert len(set(GROUPS[prediction.rows])) == 1


def test_the_same_loop_serves_within_subject_folds_grouped_on_runs() -> None:
    # The only things that change between designs are the folds and the inner grouping.
    # Each run holds 3 trials, the fewest a validation run can score a correlation on.
    groups = np.repeat(["s1", "s2", "s3", "s4"], 12).astype(object)
    runs = np.tile(np.repeat(["r1", "r2", "r3", "r4"], 3), 4).astype(object)
    y = np.random.default_rng(0).normal(size=groups.size)
    folds = within_subject_folds(groups, runs, inner_splits=2)
    predictions = cross_fit_regression(
        folds, y.reshape(-1, 1), y, groups, PIPE, GRID, inner=BY_RUN, seed=0, runs=runs
    )
    assert all(p.subject is not None for p in predictions)


def test_within_subject_folds_cannot_be_grouped_by_subject() -> None:
    # Caught at the entry point, where the message can name the real problem, rather than
    # surfacing as "at least 2 groups" from inside tuning.
    folds = within_subject_folds(GROUPS, RUNS, inner_splits=2)
    with pytest.raises(ValueError, match="within-subject"):
        cross_fit_regression(folds, X, Y, GROUPS, PIPE, GRID, inner=BY_SUBJECT, seed=0)


def test_run_grouping_without_runs_is_refused() -> None:
    with pytest.raises(ValueError, match="runs"):
        cross_fit_regression(loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=BY_RUN, seed=0)


def test_a_failing_fold_raises_rather_than_being_dropped() -> None:
    # Safeguard 6: a fold that cannot be fitted is a fault, not a measurement. Dropping it
    # would silently compute the cohort result on a subset nobody chose.
    with pytest.raises(ValueError):
        cross_fit_regression(
            loso_folds(GROUPS),
            X,
            Y,
            GROUPS,
            PIPE,
            {"regressor__nonexistent": [1]},
            inner=BY_SUBJECT,
            seed=0,
        )


def test_results_are_ordered_by_fold_not_by_completion() -> None:
    predictions = cross_fit_regression(
        loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=BY_SUBJECT, seed=0, outer_n_jobs=2
    )
    assert [p.fold for p in predictions] == sorted(p.fold for p in predictions)


def test_cross_fit_applies_feature_harmonization() -> None:
    X_mod = np.column_stack([X, np.zeros(GROUPS.size)])
    X_mod[GROUPS != "s1", -1] = 1.0
    predictions = cross_fit_regression(
        loso_folds(GROUPS),
        X_mod,
        Y,
        GROUPS,
        PIPE,
        GRID,
        inner=BY_SUBJECT,
        seed=0,
        harmonization="intersection",
    )
    assert len(predictions) == 4


def test_cross_fit_applies_target_residualization() -> None:
    covariates = Y.reshape(-1, 1)
    predictions = cross_fit_regression(
        loso_folds(GROUPS),
        X,
        Y,
        GROUPS,
        PIPE,
        GRID,
        inner=BY_SUBJECT,
        seed=0,
        covariates=covariates,
        residualize_on=["c1"],
    )
    assert len(predictions) == 4
    # Residualized target has nuisance subtracted, so it must not equal raw Y
    raw_y_fold0 = Y[predictions[0].rows]
    assert not np.allclose(predictions[0].y_true, raw_y_fold0)
    assert np.allclose(predictions[0].y_true, 0.0, atol=1e-10)


@pytest.mark.parametrize("inplace_step", ["regressor", "scaler"])
def test_inner_candidates_receive_unmodified_split_data(inplace_step: str) -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(60, 3)) * [2, 3, 4] + 10
    covariates = rng.normal(size=(60, 1))
    target = values @ np.array([2, -1, 1]) + 2 * covariates[:, 0] + rng.normal(scale=0.2, size=60)
    groups = np.repeat(["a", "b", "c", "d", "e"], 12).astype(object)
    predictions = []
    for copy in (True, False):
        steps = [("regressor", Ridge(copy_X=copy if inplace_step == "regressor" else True))]
        if inplace_step == "scaler":
            steps.insert(0, ("scaler", StandardScaler(copy=copy)))
        predictions.append(
            cross_fit_regression(
                loso_folds(groups),
                values,
                target,
                groups,
                Pipeline(steps),
                {"regressor__alpha": [100.0, 0.001]},
                inner=BY_SUBJECT,
                seed=0,
                covariates=covariates,
                residualize_on=("nuisance",),
                scoring="neg_mean_squared_error",
            )
        )
    for copied, inplace in zip(*predictions, strict=True):
        assert copied.best_params == inplace.best_params
        np.testing.assert_allclose(copied.y_pred, inplace.y_pred)


def test_cross_fit_regression_rejects_residualize_on_without_covariates() -> None:
    with pytest.raises(ValueError, match="residualize_on, but covariates is None"):
        cross_fit_regression(
            loso_folds(GROUPS),
            X,
            Y,
            GROUPS,
            PIPE,
            GRID,
            inner=BY_SUBJECT,
            seed=0,
            covariates=None,
            residualize_on=["c1"],
        )


@pytest.mark.parametrize("n_covariates", [0, 1])
def test_a_training_subject_above_the_missingness_limit_fails_the_fold(
    n_covariates: int,
) -> None:
    # max_subject_missingness belongs to the pipeline's missingness step, but pipelines never
    # route groups to their steps, so the limit has to be checked on the fitted model.
    rng = np.random.default_rng(0)
    groups = np.repeat(["s1", "s2", "s3", "s4"], 10).astype(object)
    values = rng.normal(size=(40, 10 + n_covariates))
    # Where s1 trains, each of these features is 20% missing (kept), but 36% of s1's own
    # feature values are missing.
    values[:6, :6] = np.nan
    pipe = ridge_pipeline(
        PreprocessingConfig(max_subject_missingness=0.3), seed=0, n_covariates=n_covariates
    )
    with pytest.raises(ValueError, match="s1"):
        cross_fit_regression(
            loso_folds(groups),
            values,
            rng.normal(size=40),
            groups,
            pipe,
            {},
            inner=BY_SUBJECT,
            seed=0,
        )


def test_custom_fold_with_overlap_is_rejected() -> None:
    from eegtable.model.splits import Fold

    bad = Fold(
        index=1,
        train=np.array([0, 1, 2]),
        test=np.array([2, 3]),
    )

    with pytest.raises(ValueError, match="train/test overlap"):
        cross_fit_regression(
            [bad],
            X,
            Y,
            GROUPS,
            PIPE,
            {},
            inner=BY_SUBJECT,
            seed=0,
        )


def test_target_residualization_is_refitted_inside_inner_cv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eegtable.model.crossfit as module

    rng = np.random.default_rng(42)
    groups = np.repeat(["s1", "s2", "s3", "s4"], 8).astype(object)
    x = rng.normal(size=(32, 3))
    y = rng.normal(size=32)
    covariates = rng.normal(size=(32, 1))

    folds = loso_folds(groups)
    training_sizes: list[int] = []

    original = module.residualize_targets

    def record_fit(y_all, cov_all, train, test, *, columns):
        training_sizes.append(len(train))
        return original(
            y_all,
            cov_all,
            train,
            test,
            columns=columns,
        )

    monkeypatch.setattr(module, "residualize_targets", record_fit)

    module.cross_fit_regression(
        folds,
        x,
        y,
        groups,
        PIPE,
        GRID,
        inner=InnerSplit(grouping="subject", n_splits=2),
        seed=42,
        covariates=covariates,
        residualize_on=("nuisance",),
        scoring="neg_mean_squared_error",
    )

    outer_training_size = len(folds[0].train)

    assert training_sizes
    assert any(
        size < outer_training_size for size in training_sizes
    ), "Nuisance fitting never occurred inside the inner CV folds."


def test_inner_validation_targets_do_not_leak_into_nuisance_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eegtable.model.crossfit as module

    rng = np.random.default_rng(42)
    groups = np.repeat(["s1", "s2", "s3", "s4"], 8).astype(object)
    x = rng.normal(size=(32, 3))
    y1 = rng.normal(size=32)
    y2 = y1.copy()
    y2[groups == "s2"] += 500.0
    covariates = rng.normal(size=(32, 1))

    folds = [loso_folds(groups)[0]]  # fold where s1 is test, s2/s3/s4 are train

    recorded_inner_fits_y1: dict[tuple[int, ...], np.ndarray] = {}
    recorded_inner_fits_y2: dict[tuple[int, ...], np.ndarray] = {}

    original = module.residualize_targets

    def record_fit(store: dict[tuple[int, ...], np.ndarray]):
        def fit_wrapper(y_all, cov_all, train, test, *, columns):
            res_tr, res_te = original(y_all, cov_all, train, test, columns=columns)
            store[tuple(int(i) for i in train)] = res_tr.copy()
            return res_tr, res_te

        return fit_wrapper

    monkeypatch.setattr(module, "residualize_targets", record_fit(recorded_inner_fits_y1))
    module.cross_fit_regression(
        folds,
        x,
        y1,
        groups,
        PIPE,
        GRID,
        inner=InnerSplit(grouping="subject", n_splits=2),
        seed=42,
        covariates=covariates,
        residualize_on=("nuisance",),
        scoring="neg_mean_squared_error",
    )

    monkeypatch.setattr(module, "residualize_targets", record_fit(recorded_inner_fits_y2))
    module.cross_fit_regression(
        folds,
        x,
        y2,
        groups,
        PIPE,
        GRID,
        inner=InnerSplit(grouping="subject", n_splits=2),
        seed=42,
        covariates=covariates,
        residualize_on=("nuisance",),
        scoring="neg_mean_squared_error",
    )

    s2_indices = set(np.flatnonzero(groups == "s2"))
    # In inner CV splits where s2 was NOT in train (i.e. s2 was in validation),
    # the training residuals must be bit-for-bit identical between y1 and y2
    matches = [
        train_idx for train_idx in recorded_inner_fits_y1 if not s2_indices.intersection(train_idx)
    ]
    assert matches, "Expected at least one inner split where s2 was held out in validation"
    for train_idx in matches:
        np.testing.assert_array_equal(
            recorded_inner_fits_y1[train_idx],
            recorded_inner_fits_y2[train_idx],
        )


def test_covariates_that_still_carry_rejected_epochs_are_refused() -> None:
    # Covariates are indexed positionally, so a frame that kept the rejected epochs would
    # residualize each trial against another trial's nuisance values without any error.
    rng = np.random.default_rng(0)
    groups = np.repeat([f"s{i}" for i in range(4)], 10).astype(object)
    values = rng.normal(size=(40, 3))
    target = rng.normal(size=40)

    with pytest.raises(ValueError, match="covariates has 60 rows and X has 40"):
        cross_fit_regression(
            loso_folds(groups),
            values,
            target,
            groups,
            PIPE,
            {},
            inner=BY_SUBJECT,
            seed=0,
            covariates=rng.normal(size=(60, 1)),
            residualize_on=["age"],
        )


def test_run_grouping_under_cross_subject_folds_is_refused() -> None:
    # Run labels are shared across subjects, so a run-grouped inner split of a LOSO
    # training set keeps every subject on both sides and tunes for the wrong task.
    with pytest.raises(ValueError, match="cross-subject folds cannot be grouped by run"):
        cross_fit_regression(
            loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=BY_RUN, seed=0, runs=RUNS
        )


class _Column(BaseEstimator, RegressorMixin):
    # Predicts one input column as it is, so every candidate's inner scores are known ahead.
    def __init__(self, column: int = 0) -> None:
        self.column = column

    def fit(self, X, y):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float)[:, self.column]


def _offsets_or_tracking() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Column 0 is each subject's offset: a good pooled R^2 and no within-subject tracking.
    # Column 1 tracks every trial at the wrong scale: a hopeless R^2 and a subject r of 1.
    rng = np.random.default_rng(0)
    groups = np.repeat([f"s{i}" for i in range(6)], 10).astype(object)
    offset = np.repeat(np.arange(6) * 10.0, 10)
    signal = rng.normal(size=groups.size)
    return np.column_stack([offset, 100.0 * signal]), offset + signal, groups


def test_regression_tuning_selects_on_the_subject_level_r_it_is_evaluated_with() -> None:
    # Pooled R^2 rewards predicting each subject's offset, which the outer subject-level r
    # ignores; selecting on it picks the model that tracks nothing within a subject.
    X_, y, groups = _offsets_or_tracking()
    predictions = cross_fit_regression(
        loso_folds(groups),
        X_,
        y,
        groups,
        Pipeline([("regressor", _Column())]),
        {"regressor__column": [0, 1]},
        inner=BY_SUBJECT,
        seed=0,
    )
    assert [p.best_params["regressor__column"] for p in predictions] == [1] * 6


def test_an_explicit_scikit_learn_scorer_still_decides_the_selection() -> None:
    X_, y, groups = _offsets_or_tracking()
    predictions = cross_fit_regression(
        loso_folds(groups),
        X_,
        y,
        groups,
        Pipeline([("regressor", _Column())]),
        {"regressor__column": [0, 1]},
        inner=BY_SUBJECT,
        seed=0,
        scoring="r2",
    )
    assert [p.best_params["regressor__column"] for p in predictions] == [0] * 6


class _Blend(BaseEstimator, RegressorMixin):
    # Tracks column 1, blurred by column 2 at 1/k and by column 3 at k/1000, so the
    # blur is least at k = sqrt(1000) and the best k is known in advance.
    def __init__(self, k: float = 1.0) -> None:
        self.k = k

    def fit(self, X, y):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, 1] + X[:, 2] / self.k + self.k * X[:, 3] / 1000.0


def _blend_design() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_, y, groups = _offsets_or_tracking()
    noise = np.random.default_rng(1).normal(size=(groups.size, 2))
    return np.column_stack([X_[:, 0], y, noise]), y, groups


def test_a_search_that_settles_on_a_grid_edge_is_reported() -> None:
    X_, y, groups = _blend_design()
    with pytest.warns(
        UserWarning, match=r"regressor__k .* upper end of its grid \(10\.0\) in all 6"
    ):
        cross_fit_regression(
            loso_folds(groups),
            X_,
            y,
            groups,
            Pipeline([("regressor", _Blend())]),
            {"regressor__k": [1.0, 3.0, 10.0]},
            inner=BY_SUBJECT,
            seed=0,
        )


def test_an_interior_optimum_raises_no_grid_edge_warning() -> None:
    X_, y, groups = _blend_design()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        predictions = cross_fit_regression(
            loso_folds(groups),
            X_,
            y,
            groups,
            Pipeline([("regressor", _Blend())]),
            {"regressor__k": [1.0, 10.0, 30.0, 100.0, 1000.0]},
            inner=BY_SUBJECT,
            seed=0,
        )
    assert {p.best_params["regressor__k"] for p in predictions} == {30.0}


def _shared_stimulus_slopes() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Subjects differ in how steeply both the target and the feature follow the stimulus;
    # beyond the stimulus the feature tracks nothing of the target.
    rng = np.random.default_rng(0)
    groups = np.repeat([f"s{i}" for i in range(8)], 40).astype(object)
    stimulus = rng.choice([-2.0, -1.0, 0.0, 1.0, 2.0], size=groups.size)
    slope = np.repeat(np.linspace(0.2, 2.0, 8), 40)
    y = slope * stimulus + 0.3 * rng.normal(size=groups.size)
    feature = slope * stimulus + 0.3 * rng.normal(size=groups.size)
    return feature.reshape(-1, 1), y, groups, stimulus.reshape(-1, 1)


def _subject_r(predictions, groups) -> float:
    y_true, y_pred, labels, _, _ = fold_results(predictions, groups=groups)
    frame = pd.DataFrame({"subject_id": labels, "y_true": y_true, "y_pred": y_pred})
    return subject_level_r(frame).r


def test_a_pooled_nuisance_model_leaves_each_subjects_own_stimulus_slope_behind() -> None:
    # Removing the average slope from both sides leaves every subject's deviation from it in
    # both residuals, where it reads as trial-level tracking the feature does not do.
    feature, y, groups, stimulus = _shared_stimulus_slopes()
    pipeline = ridge_pipeline(PreprocessingConfig(deconfound=True), seed=0, n_covariates=1)
    predictions = cross_fit_regression(
        loso_folds(groups),
        np.column_stack([feature, stimulus]),
        y,
        groups,
        pipeline,
        {},
        inner=BY_SUBJECT,
        seed=0,
        covariates=stimulus,
        residualize_on=["stimulus"],
    )
    assert _subject_r(predictions, groups) > 0.5


def test_residualizing_within_subjects_removes_each_subjects_own_stimulus_response() -> None:
    feature, y, groups, stimulus = _shared_stimulus_slopes()
    predictions = cross_fit_regression(
        loso_folds(groups),
        feature,
        y,
        groups,
        ridge_pipeline(PreprocessingConfig(), seed=0),
        {},
        inner=BY_SUBJECT,
        seed=0,
        covariates=stimulus,
        residualize_on=["stimulus"],
        residualize_within="subject",
    )
    assert abs(_subject_r(predictions, groups)) < 0.15
