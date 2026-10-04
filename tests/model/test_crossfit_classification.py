from __future__ import annotations

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from eegtable.model.crossfit import cross_fit_classification
from eegtable.model.estimators import ensemble_pipeline, svm_pipeline
from eegtable.model.splits import InnerSplit, loso_folds, within_subject_folds
from eegtable.model.transformers import PreprocessingConfig

GROUPS = np.repeat(["s1", "s2", "s3", "s4"], 6).astype(object)
Y = np.tile([0, 1], GROUPS.size // 2).astype(np.intp)
X = np.random.default_rng(0).normal(size=(GROUPS.size, 3)) + Y[:, None]
PIPE = Pipeline([("classifier", LogisticRegression(max_iter=500))])
GRID = {"classifier__C": [0.1, 1.0]}
STRATIFIED = InnerSplit(grouping="subject", stratified=True, n_splits=2)


def test_probability_columns_are_pinned_to_the_recorded_classes() -> None:
    for prediction in cross_fit_classification(
        loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=STRATIFIED, seed=0
    ):
        assert prediction.y_prob is not None
        assert prediction.y_prob.shape[1] == len(prediction.classes)
        np.testing.assert_allclose(prediction.y_prob.sum(axis=1), 1.0)


def test_a_single_class_training_fold_raises() -> None:
    y_degenerate = np.where(GROUPS == "s1", 1, 0).astype(np.intp)
    fold_s1 = [f for f in loso_folds(GROUPS) if set(GROUPS[f.test]) == {"s1"}][:1]
    with pytest.raises(ValueError, match="only one class in training for subject s1"):
        cross_fit_classification(
            fold_s1,
            X,
            y_degenerate,
            GROUPS,
            PIPE,
            GRID,
            inner=STRATIFIED,
            seed=0,
        )


def test_an_estimator_without_predict_proba_reports_none_not_a_decision_function() -> None:
    pipe = Pipeline([("classifier", RidgeClassifier())])
    predictions = cross_fit_classification(
        loso_folds(GROUPS), X, Y, GROUPS, pipe, {}, inner=STRATIFIED, seed=0
    )
    assert all(p.y_prob is None for p in predictions)


def test_svm_preserves_held_out_decision_scores() -> None:
    pipeline = Pipeline([("classifier", SVC())])
    folds = loso_folds(GROUPS)
    predictions = cross_fit_classification(
        folds, X, Y, GROUPS, pipeline, {}, inner=STRATIFIED, seed=0
    )
    for fold, prediction in zip(folds, predictions, strict=True):
        reference = SVC().fit(X[fold.train], Y[fold.train])
        assert prediction.y_prob is None
        np.testing.assert_allclose(prediction.y_score, reference.decision_function(X[fold.test]))


def test_labels_stay_integers_through_the_fold_loop() -> None:
    for prediction in cross_fit_classification(
        loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=STRATIFIED, seed=0
    ):
        assert prediction.y_pred.dtype == np.intp


def test_within_subject_classification_serves_run_splits() -> None:
    groups = np.repeat(["s1", "s2"], 8).astype(object)
    runs = np.tile(np.repeat(["r1", "r2", "r3", "r4"], 2), 2).astype(object)
    y = np.tile([0, 1], groups.size // 2).astype(np.intp)
    x = np.ones((groups.size, 2))
    folds = within_subject_folds(groups, runs, inner_splits=2)
    inner_run = InnerSplit(grouping="run", stratified=True, n_splits=2)
    predictions = cross_fit_classification(
        folds, x, y, groups, PIPE, GRID, inner=inner_run, seed=0, runs=runs
    )
    assert len(predictions) > 0
    assert all(p.subject is not None for p in predictions)


def test_within_subject_classification_applies_harmonization() -> None:
    groups = np.repeat(["s1", "s2"], 8).astype(object)
    runs = np.tile(np.repeat(["r1", "r2", "r3", "r4"], 2), 2).astype(object)
    y = np.tile([0, 1], groups.size // 2).astype(np.intp)
    x = np.column_stack([np.ones(groups.size), np.zeros(groups.size)])
    x[groups == "s2", -1] = 1.0
    folds = within_subject_folds(groups, runs, inner_splits=2)
    inner_run = InnerSplit(grouping="run", stratified=True, n_splits=2)
    predictions = cross_fit_classification(
        folds,
        x,
        y,
        groups,
        PIPE,
        GRID,
        inner=inner_run,
        seed=0,
        runs=runs,
        harmonization="intersection",
    )
    assert len(predictions) > 0


def test_within_subject_classification_raises_when_training_fold_has_one_class() -> None:
    groups = np.repeat(["s1", "s2"], 8).astype(object)
    runs = np.tile(np.repeat(["r1", "r2", "r3", "r4"], 2), 2).astype(object)
    y = np.zeros(groups.size, dtype=np.intp)
    y[groups == "s2"] = 1
    folds = within_subject_folds(groups, runs, inner_splits=2)
    inner_run = InnerSplit(grouping="run", stratified=True, n_splits=2)
    with pytest.raises(ValueError, match="one class"):
        cross_fit_classification(
            folds[:1],
            np.ones((groups.size, 2)),
            y,
            groups,
            PIPE,
            GRID,
            inner=inner_run,
            seed=0,
            runs=runs,
        )


def test_labels_outside_zero_and_one_are_refused_before_any_fold_is_fitted() -> None:
    # A missing-response code such as -1 is a third class: sklearn would fit it happily
    # and only the metrics would notice, after every fold had been tuned.
    y_coded = Y.copy()
    y_coded[:2] = -1
    with pytest.raises(ValueError, match=r"labels coded 0/1, got \[-1, 0, 1\]"):
        cross_fit_classification(
            loso_folds(GROUPS), X, y_coded, GROUPS, PIPE, GRID, inner=STRATIFIED, seed=0
        )


@pytest.mark.parametrize("grid", [{}, {"classifier__C": [0.1, 1.0]}])
@pytest.mark.parametrize("probability", [True, np.bool_(True)])
def test_grouped_fitting_refuses_hidden_svm_probability_calibration(grid, probability) -> None:
    pipe = Pipeline([("classifier", SVC(probability=probability))])
    with pytest.raises(ValueError, match="internal trial-wise cross-validation"):
        cross_fit_classification(
            loso_folds(GROUPS), X, Y, GROUPS, pipe, grid, inner=STRATIFIED, seed=0
        )


def test_grouped_tuning_refuses_a_grid_that_enables_trial_wise_calibration() -> None:
    pipe = Pipeline([("classifier", SVC(probability=False))])
    with pytest.raises(ValueError, match="internal trial-wise cross-validation"):
        cross_fit_classification(
            loso_folds(GROUPS),
            X,
            Y,
            GROUPS,
            pipe,
            {"classifier__probability": [False, True]},
            inner=STRATIFIED,
            seed=0,
        )


def test_grouped_fitting_refuses_calibration_that_cannot_receive_split_groups() -> None:
    pipe = Pipeline([("classifier", CalibratedClassifierCV(LogisticRegression(), cv=2))])
    with pytest.raises(ValueError, match="group-disjoint calibration"):
        cross_fit_classification(
            loso_folds(GROUPS), X, Y, GROUPS, pipe, {}, inner=STRATIFIED, seed=0
        )


def test_a_soft_ensemble_with_an_uncalibrated_svm_is_refused_explicitly() -> None:
    ensemble = VotingClassifier(
        [("svm", SVC(probability=False)), ("lr", LogisticRegression())], voting="soft"
    )
    with pytest.raises(ValueError, match="Soft voting.*SVC"):
        cross_fit_classification(
            loso_folds(GROUPS),
            X,
            Y,
            GROUPS,
            Pipeline([("ensemble", ensemble)]),
            {},
            inner=STRATIFIED,
            seed=0,
        )


def test_soft_voting_without_internal_probability_cv_remains_supported() -> None:
    ensemble = VotingClassifier(
        [("lr", LogisticRegression()), ("rf", RandomForestClassifier(n_estimators=5))],
        voting="soft",
    )
    predictions = cross_fit_classification(
        loso_folds(GROUPS),
        X,
        Y,
        GROUPS,
        Pipeline([("ensemble", ensemble)]),
        {},
        inner=STRATIFIED,
        seed=0,
    )
    assert all(prediction.y_prob is not None for prediction in predictions)


@pytest.mark.parametrize("factory", [svm_pipeline, ensemble_pipeline])
def test_default_svm_and_ensemble_keep_grouped_predictions_without_calibration(factory) -> None:
    pipeline = factory(PreprocessingConfig(), seed=0)
    predictions = cross_fit_classification(
        loso_folds(GROUPS), X, Y, GROUPS, pipeline, {}, inner=STRATIFIED, seed=0
    )
    assert all(prediction.y_prob is None for prediction in predictions)
    assert all(set(prediction.y_pred) <= {0, 1} for prediction in predictions)
