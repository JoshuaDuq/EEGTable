from __future__ import annotations

import numpy as np
import pytest

from eegtable.model.metrics import (
    classification_metrics,
    regression_metrics,
    within_condition_metrics,
    within_subject_centered_metrics,
)


def test_a_single_class_fold_does_not_report_a_balanced_accuracy() -> None:
    # One class in the held-out subject makes balanced accuracy undefined. Reporting NaN
    # keeps the fold visible as unscored; reporting 0.5 would invent a chance result.
    result = classification_metrics(
        np.array([1, 1, 1], dtype=np.intp),
        np.array([1, 1, 0], dtype=np.intp),
        groups=np.array(["s1", "s1", "s1"], dtype=object),
    )
    assert np.isnan(result.balanced_accuracy)


def test_a_single_class_confusion_matrix_still_has_both_axes() -> None:
    # A 1x1 matrix would silently reindex downstream, turning "no negatives were seen"
    # into "no negatives exist".
    result = classification_metrics(
        np.array([1, 1], dtype=np.intp), np.array([1, 1], dtype=np.intp)
    )
    assert result.confusion.shape == (2, 2)


def test_auc_and_average_precision_use_unbounded_decision_scores() -> None:
    result = classification_metrics(
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 0, 1]),
        y_score=np.array([-8.0, 2.0, -1.0, 9.0]),
        groups=np.array(["s1", "s1", "s2", "s2"], dtype=object),
    )
    assert result.auc == 1.0
    assert result.average_precision == 1.0
    assert result.y_prob is None


@pytest.mark.parametrize("scores", [np.array([0.0]), np.array([0.0, np.nan])])
def test_classification_rejects_invalid_decision_scores(scores) -> None:
    with pytest.raises(ValueError, match="y_score"):
        classification_metrics(np.array([0, 1]), np.array([0, 1]), y_score=scores)


def test_classification_requires_a_subject_label_for_every_trial() -> None:
    with pytest.raises(ValueError, match="subject label for every trial"):
        classification_metrics(
            np.array([0, 1, 0, 1]),
            np.array([0, 1, 1, 0]),
            groups=np.array([1.0, 1.0, np.nan, np.nan]),
        )


@pytest.mark.parametrize("groups", [np.array([]), np.array([["s1"], ["s1"]])])
def test_classification_requires_aligned_group_labels(groups) -> None:
    with pytest.raises(ValueError, match="aligned 1-D arrays"):
        classification_metrics(np.array([0, 1]), np.array([0, 1]), groups=groups)


def test_classification_requires_one_label_per_trial() -> None:
    labels = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="aligned 1-D arrays"):
        classification_metrics(labels, labels)


def test_regression_metrics_report_subject_level_r() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    groups = np.array(["s1"] * 3 + ["s2"] * 3, dtype=object)
    summary, _ = regression_metrics(y_true, y_true.copy(), groups)
    assert summary["subject_level_r"] == pytest.approx(1.0, abs=1e-5)


def test_classification_primary_precision_recall_f1_are_subject_level() -> None:
    # s1 is perfect; s2 has one hit, one false alarm and two misses. Every subject
    # predicts at least one positive, so each metric is defined for both, and the
    # subject means (0.75, 2/3, 0.7) differ from the pooled values (2/3, 0.5, 4/7).
    y_true = np.array([0, 1, 0, 1, 1, 1], dtype=np.intp)
    y_pred = np.array([0, 1, 1, 0, 0, 1], dtype=np.intp)
    groups = np.array(["s1", "s1", "s2", "s2", "s2", "s2"], dtype=object)
    result = classification_metrics(y_true, y_pred, groups=groups)
    assert result.precision == pytest.approx(0.75)
    assert result.recall == pytest.approx(2.0 / 3.0)
    assert result.f1 == pytest.approx(0.7)


def test_group_classification_permutations_do_not_fallback_to_pooled_auc() -> None:
    # If subject-level AUC cannot be computed, mean_subject_auc remains NaN rather than
    # quietly falling back to pooled trials.
    y_true = np.array([0, 0, 1, 1], dtype=np.intp)
    y_pred = np.array([0, 0, 1, 1], dtype=np.intp)
    groups = np.array(["s1", "s1", "s2", "s2"], dtype=object)
    result = classification_metrics(y_true, y_pred, groups=groups)
    assert np.isnan(result.mean_subject_auc)


def test_within_subject_centered_metrics_computes_r2() -> None:
    target = np.array([1.0, 2.0, 10.0, 12.0])
    full = np.array([1.0, 2.0, 10.0, 12.0])
    nuis = np.array([1.5, 1.5, 11.0, 11.0])
    groups = np.array(["s1", "s1", "s2", "s2"], dtype=object)
    res = within_subject_centered_metrics(target, full, nuis, groups)
    assert res["within_subject_centered_full_r2"] == pytest.approx(1.0)


def test_within_condition_metrics_computes_r2() -> None:
    target = np.array([1.0, 2.0, 1.0, 2.0])
    full = np.array([1.0, 2.0, 1.0, 2.0])
    nuis = np.array([1.5, 1.5, 1.5, 1.5])
    groups = np.array(["s1", "s1", "s1", "s1"], dtype=object)
    conditions = np.array(["c1", "c1", "c1", "c1"])
    res = within_condition_metrics(target, full, nuis, groups, conditions)
    assert res["within_condition_centered_full_r2"] == pytest.approx(1.0)


def test_within_subject_centered_metrics_equal_subject_weighting() -> None:
    t_a = np.linspace(0.0, 10.0, 100)
    f_a = t_a.copy()
    n_a = np.full(100, 5.0)
    t_b = np.linspace(0.0, 10.0, 10)
    f_b = np.full(10, 5.0)
    n_b = np.full(10, 5.0)

    target = np.concatenate([t_a, t_b])
    full = np.concatenate([f_a, f_b])
    nuis = np.concatenate([n_a, n_b])
    groups = np.array(["sA"] * 100 + ["sB"] * 10, dtype=object)

    res = within_subject_centered_metrics(target, full, nuis, groups)
    assert res["within_subject_centered_full_r2"] == pytest.approx(0.5)


def test_classification_metrics_handles_2d_probabilities() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1], dtype=np.intp)
    y_pred = np.array([0, 1, 0, 1, 0, 1], dtype=np.intp)
    y_prob_2d = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.3, 0.7],
        ],
        dtype=float,
    )
    groups = np.array(["s1", "s1", "s2", "s2", "s3", "s3"], dtype=object)
    result = classification_metrics(y_true, y_pred, y_prob=y_prob_2d, groups=groups)
    assert result.auc == pytest.approx(1.0)
    assert result.average_precision == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [([1, 1, 2, 2], [1, 2, 2, 2]), ([0, 0, 1, 1], [0, 2, 1, 1])],
    ids=["event-codes", "prediction-out-of-range"],
)
def test_labels_other_than_zero_and_one_are_refused(y_true: list[int], y_pred: list[int]) -> None:
    # The confusion matrix, specificity and the positive class of precision and recall all
    # assume 0/1 coding; event codes 1/2 would be scored against the wrong classes.
    with pytest.raises(ValueError, match="0/1"):
        classification_metrics(np.array(y_true), np.array(y_pred))


def test_subject_averaged_scalars_and_pooled_confusion_differ_by_design() -> None:
    # Six correct trials from one subject and two wrong from another: pooled accuracy is 6/8,
    # subject-averaged is (1.0 + 0.0)/2. The confusion matrix reports the pooled counts.
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    groups = np.array(["s1"] * 6 + ["s2"] * 2, dtype=object)

    res = classification_metrics(y_true, y_pred, groups=groups)

    assert res.accuracy == pytest.approx(0.5)
    pooled = (res.confusion[0, 0] + res.confusion[1, 1]) / res.confusion.sum()
    assert pooled == pytest.approx(0.75)


@pytest.mark.parametrize("metric", ["within_subject", "within_condition"])
def test_centred_metrics_refuse_a_trial_without_a_subject_label(metric: str) -> None:
    # Dropping an unlabelled trial would quietly shrink the subject denominator, and the two
    # functions used to disagree: one skipped such trials, the other raised a TypeError on sort.
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    f = t.copy()
    n = np.zeros(6)
    groups = np.array(["s1", "s1", "s1", "s1", np.nan, np.nan], dtype=object)
    conditions = np.array(["a", "a", "b", "b", "a", "b"], dtype=object)

    with pytest.raises(ValueError, match="subject label for every trial"):
        if metric == "within_subject":
            within_subject_centered_metrics(t, f, n, groups)
        else:
            within_condition_metrics(t, f, n, groups, conditions)


def test_centred_metrics_refuse_two_dimensional_input() -> None:
    values = np.arange(6.0).reshape(3, 2)
    groups = np.array([["s1", "s1"], ["s1", "s1"], ["s2", "s2"]], dtype=object)

    with pytest.raises(ValueError, match="aligned 1-D arrays"):
        within_subject_centered_metrics(values, values, np.zeros((3, 2)), groups)


@pytest.mark.parametrize("scale", [1.0, 1e-7])
def test_centered_r2_is_invariant_to_measurement_units(scale) -> None:
    y = np.tile(np.arange(4.0), 2) * scale
    groups = np.repeat(["a", "b"], 4)
    conditions = np.tile(["c", "c", "d", "d"], 2)
    result = within_subject_centered_metrics(y, y, np.zeros(8), groups)
    assert result["within_subject_centered_full_r2"] == pytest.approx(1.0)
    result = within_condition_metrics(y, y, np.zeros(8), groups, conditions)
    assert result["within_condition_centered_full_r2"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([0.1, 0.9, np.nan, np.nan]),
        np.array([0.1, 0.9, np.inf, 0.2]),
        np.ones((4, 3)) / 3,
        np.array([0.1, 0.9, 0.1]),
    ],
)
def test_auc_rejects_invalid_probability_inputs(probabilities) -> None:
    with pytest.raises(ValueError, match="y_prob"):
        classification_metrics(
            np.array([0, 1, 0, 1]),
            np.array([0, 1, 1, 0]),
            y_prob=probabilities,
        )


def test_centered_r2_rejects_constant_targets_despite_roundoff() -> None:
    y = np.full(3, 0.1)
    groups = np.full(3, "s")
    result = within_subject_centered_metrics(y, y, np.zeros(3), groups)
    assert np.isnan(result["within_subject_centered_full_r2"])
    result = within_condition_metrics(y, y, np.zeros(3), groups, np.full(3, "c"))
    assert np.isnan(result["within_condition_centered_full_r2"])


def test_a_subject_without_positives_has_undefined_rather_than_zero_recall() -> None:
    # s2 holds no positive trials and predicts none: 0/0. It must drop out of the
    # subject mean the way balanced accuracy and AUC already do, not enter as zero
    # and halve a perfect recall.
    y_true = np.array([0, 1, 0, 1, 0, 0, 0], dtype=np.intp)
    y_pred = y_true.copy()
    groups = np.array(["s1"] * 4 + ["s2"] * 3, dtype=object)
    result = classification_metrics(y_true, y_pred, groups=groups)
    assert result.recall == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)
    assert result.accuracy == pytest.approx(1.0)
    assert np.isnan(result.per_subject["s2"]["recall"])


def test_regression_metrics_centre_the_subject_correlation_within_folds() -> None:
    # Two folds score s1 with offsets of opposite sign to its run means; without the fold
    # labels those offsets alone would make the subject correlation negative.
    y_true = np.array([1.0, 2.0, 3.0, 11.0, 12.0, 13.0, 1.0, 2.0, 4.0])
    y_pred = np.array([10.0, 11.0, 12.0, 0.0, 1.0, 2.0, 1.0, 3.0, 2.0])
    groups = np.array(["s1"] * 6 + ["s2"] * 3, dtype=object)
    folds = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    pooled, _ = regression_metrics(y_true, y_pred, groups)
    centred, per_subject = regression_metrics(y_true, y_pred, groups, folds=folds)
    assert pooled["subject_level_r"] < 0.0
    assert per_subject[0] == {"subject": "s1", "r": pytest.approx(1.0)}
    assert centred["subject_level_r"] > 0.9


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("side", ["target", "prediction"])
def test_regression_evaluation_cannot_drop_failed_trials(invalid, side) -> None:
    target, prediction = np.arange(4.0), np.arange(4.0)
    (target if side == "target" else prediction)[-1] = invalid
    with pytest.raises(ValueError, match="finite.*every trial"):
        regression_metrics(target, prediction)


@pytest.mark.parametrize("constant", [1.0, 0.1])
def test_constant_targets_keep_undefined_variance_scores_visible(constant) -> None:
    target = np.full(3, constant)
    perfect, _ = regression_metrics(target, target.copy())
    imperfect, _ = regression_metrics(target, target + np.arange(3.0))
    assert np.isnan(perfect["r2"])
    assert np.isnan(perfect["explained_variance"])
    assert imperfect["r2"] == -np.inf
    assert imperfect["explained_variance"] == -np.inf


@pytest.mark.parametrize("function", [within_subject_centered_metrics, within_condition_metrics])
def test_centered_evaluation_refuses_missing_predictions(function) -> None:
    target = np.arange(4.0)
    prediction = np.array([0.0, 1.0, 2.0, np.nan])
    arrays = [target, prediction, np.zeros(4), np.full(4, "s")]
    if function is within_condition_metrics:
        arrays.append(np.full(4, "c"))
    with pytest.raises(ValueError, match="finite.*every trial"):
        function(*arrays)


def test_regression_evaluation_refuses_multioutput_arrays() -> None:
    with pytest.raises(ValueError, match="aligned 1-D arrays"):
        regression_metrics(np.arange(6.0).reshape(3, 2), np.arange(6.0).reshape(3, 2))
