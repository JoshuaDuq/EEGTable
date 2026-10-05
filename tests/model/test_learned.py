from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from eegtable.csp import CommonSpatialPattern
from eegtable.microstates import MicrostateModel, microstate_coverage
from eegtable.model import learned
from eegtable.model.crossfit import FoldClassification, FoldPrediction
from eegtable.model.splits import Fold, InnerSplit, loso_folds, within_subject_folds
from eegtable.signal import Signal
from eegtable.spectra import Window

CHANNELS = ("F3", "F4", "P3", "P4")


def _signals(n_subjects=4, n_per_subject=8):
    rng = np.random.default_rng(4)
    n_epochs = n_subjects * n_per_subject
    labels = np.tile(np.arange(n_per_subject) % 2, n_subjects)
    data = rng.standard_normal((n_epochs, 4, 150))
    data[labels == 0, 0] *= 3.0
    data[labels == 1, 1] *= 3.0
    signal = Signal.from_arrays(
        data=data,
        times=np.arange(150) / 100.0,
        sfreq=100.0,
        ch_names=CHANNELS,
        row_ids=tuple((f"subject{i // n_per_subject}", i, "task") for i in range(n_epochs)),
    )
    groups = np.repeat(np.arange(n_subjects).astype(str), n_per_subject)
    return signal, labels, groups


def _microstate_signals():
    rng = np.random.default_rng(2)
    maps = np.array([[1.0, -1.0, 0.0, 0.0], [1.0, 1.0, -1.0, -1.0]])
    states = np.tile(np.repeat([0, 1], 25), 4)
    pulse = np.tile(1.0 + np.sin(np.linspace(0.0, np.pi, 25)), 8)
    data = maps[states].T[None] * pulse
    data = np.repeat(data, 24, axis=0) + rng.normal(0, 0.02, (24, 4, 200))
    return Signal.from_arrays(
        data=data,
        times=np.arange(200) / 100.0,
        sfreq=100.0,
        ch_names=CHANNELS,
        row_ids=tuple((f"subject{i // 6}", i, "task") for i in range(24)),
    )


def test_csp_transformer_reuses_existing_algorithm_and_clones_cleanly():
    signal, labels, _ = _signals()
    transformer = learned.CSPTransformer(ch_names=CHANNELS, n_components=2)
    transformer.fit(signal.data[:16], labels[:16])
    model = CommonSpatialPattern.fit(signal, labels, rows=np.arange(16), n_components=2)
    np.testing.assert_allclose(
        transformer.transform(signal.data[16:]), model.transform(signal, rows=np.arange(16, 32))
    )
    assert not hasattr(clone(transformer), "model_")


def test_microstate_transformer_only_fits_training_templates():
    signal = _microstate_signals()
    transformer = learned.MicrostateTransformer(
        ch_names=CHANNELS, sfreq=100.0, n_states=2, measures=("coverage",)
    ).fit(signal.data[:12])
    model = MicrostateModel.fit(signal, rows=np.arange(12), n_states=2)
    held_out = replace(
        signal,
        data=signal.data[12:],
        coverage=signal.coverage[12:],
        row_ids=signal.row_ids[12:],
    )
    expected = microstate_coverage(
        model.segment(held_out), windows=[Window("epoch", 0.0, 1.99)]
    ).values
    before = transformer.model_.templates.copy()
    np.testing.assert_allclose(transformer.transform(signal.data[12:]), expected)
    transformer.transform(-signal.data[12:])
    np.testing.assert_array_equal(before, transformer.model_.templates)


def test_microstate_transformer_explicit_reference_matches_feature_order():
    signal = _microstate_signals()
    model = MicrostateModel.fit(signal, rows=np.arange(12), n_states=2)
    reference = MicrostateModel.from_templates(
        model.templates[::-1], ch_names=CHANNELS, labels=("A", "B"), reference_name="study"
    )
    transformer = learned.MicrostateTransformer(
        ch_names=CHANNELS, sfreq=100.0, n_states=2, reference=reference, measures=("coverage",)
    ).fit(signal.data[:12])
    assert transformer.model_.labels == ("A", "B")
    np.testing.assert_allclose(transformer.model_.templates, model.templates[::-1])


def test_covariances_project_average_reference_into_training_rank():
    pytest.importorskip("pyriemann")
    from pyriemann.estimation import Covariances

    signal, _, _ = _signals()
    data = signal.data - signal.data.mean(axis=1, keepdims=True)
    transformer = learned.CovarianceTransformer(ch_names=CHANNELS, estimator="oas").fit(data[:16])
    assert transformer.rank_ == 3
    assert transformer.basis_.shape == (4, 3)
    expected = Covariances(estimator="oas").transform(
        np.einsum("cr,ect->ert", transformer.basis_, data[16:])
    )
    np.testing.assert_allclose(transformer.transform(data[16:]), expected)
    assert np.all(np.linalg.eigvalsh(expected) > 0)
    with pytest.raises(ValueError, match="training.*subspace"):
        transformer.transform(signal.data[16:])


def test_tangent_reference_is_training_only_and_prediction_is_batch_independent():
    pytest.importorskip("pyriemann")
    from pyriemann.tangentspace import TangentSpace

    signal, _, _ = _signals()
    transformer = learned.TangentSpaceTransformer(ch_names=CHANNELS).fit(signal.data[:16])
    covariances = transformer.covariances_.transform(signal.data[:16])
    reference = TangentSpace(metric="riemann", tsupdate=False).fit(covariances).reference_
    np.testing.assert_allclose(transformer.reference_, reference)
    before = transformer.reference_.copy()
    all_features = transformer.transform(signal.data[16:])
    one = transformer.transform(signal.data[16:17])
    np.testing.assert_allclose(one[0], all_features[0])
    transformer.transform(signal.data[16:] * 100.0)
    np.testing.assert_array_equal(before, transformer.reference_)
    assert all_features.shape == (16, 10)


def test_covariances_reject_singular_scm_and_zero_variance():
    pytest.importorskip("pyriemann")
    signal, _, _ = _signals()
    short = signal.data[:, :, :3]
    with pytest.raises(ValueError, match="positive definite"):
        learned.CovarianceTransformer(ch_names=CHANNELS, estimator="scm").fit_transform(short)
    with pytest.raises(ValueError, match="temporal variance"):
        learned.CovarianceTransformer(ch_names=CHANNELS).fit(np.zeros_like(signal.data))


def test_learned_pipeline_has_fold_fitted_feature_and_scaling_steps():
    transformer = learned.CSPTransformer(ch_names=CHANNELS, n_components=2)
    pipeline = learned.learned_pipeline(transformer, LogisticRegression())
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["features", "scale", "model"]
    assert isinstance(pipeline.named_steps["scale"], StandardScaler)


@pytest.mark.parametrize(
    "estimator,grid",
    [
        (LogisticRegression(), {"model__C": [0.1, 1.0]}),
        (
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            {"model__shrinkage": ["auto", 0.5]},
        ),
    ],
)
def test_signal_classification_produces_nested_held_out_predictions(estimator, grid):
    signal, labels, groups = _signals()
    folds = loso_folds(groups)
    pipeline = learned.learned_pipeline(
        learned.CSPTransformer(ch_names=CHANNELS, n_components=2), estimator
    )
    results = learned.cross_fit_signal_classification(
        folds,
        signal,
        labels,
        groups,
        pipeline,
        grid,
        inner=InnerSplit("subject", n_splits=3),
        seed=4,
    )
    assert all(isinstance(result, FoldClassification) for result in results)
    assert all(
        result.best_params[name] in candidates
        for result in results
        for name, candidates in grid.items()
    )
    assert all(result.y_prob.shape == (8, 2) for result in results)
    for result in results:
        np.testing.assert_allclose(result.y_prob[:, 1], expit(result.y_score))
    assert np.concatenate([result.rows for result in results]).size == labels.size
    assert np.mean(np.concatenate([result.y_true == result.y_pred for result in results])) > 0.9


def test_untuned_signal_classification_rejects_internal_svm_calibration():
    signal, labels, groups = _signals()
    pipeline = learned.learned_pipeline(
        learned.CSPTransformer(ch_names=CHANNELS, n_components=2), SVC(probability=True)
    )
    with pytest.raises(ValueError, match="internal trial-wise cross-validation"):
        learned.cross_fit_signal_classification(
            loso_folds(groups),
            signal,
            labels,
            groups,
            pipeline,
            {},
            inner=InnerSplit("subject", n_splits=3),
            seed=4,
        )


def test_signal_regression_produces_nested_held_out_predictions():
    pytest.importorskip("pyriemann")
    signal, _, groups = _signals()
    target = np.log(np.var(signal.data[:, 0], axis=-1))
    pipeline = learned.learned_pipeline(learned.TangentSpaceTransformer(ch_names=CHANNELS), Ridge())
    results = learned.cross_fit_signal_regression(
        loso_folds(groups),
        signal,
        target,
        groups,
        pipeline,
        {"model__alpha": [0.1, 1.0]},
        inner=InnerSplit("subject", n_splits=3),
        seed=4,
    )
    assert all(isinstance(result, FoldPrediction) for result in results)
    assert np.concatenate([result.rows for result in results]).size == target.size
    assert all(np.isfinite(result.y_pred).all() for result in results)


def test_signal_cross_fit_rejects_outer_group_overlap_before_fitting():
    signal, labels, groups = _signals()
    pipeline = learned.learned_pipeline(
        learned.CSPTransformer(ch_names=CHANNELS, n_components=2), LogisticRegression()
    )
    with pytest.raises(ValueError, match="subject overlap"):
        learned.cross_fit_signal_classification(
            [Fold(1, np.arange(16), np.arange(16, 32))],
            signal,
            labels,
            np.tile(np.arange(4).astype(str), 8),
            pipeline,
            {},
            inner=InnerSplit("subject"),
            seed=4,
        )


def test_signal_cross_fit_rejects_channel_reordering_before_fitting():
    signal, labels, groups = _signals()
    pipeline = learned.learned_pipeline(
        learned.CSPTransformer(ch_names=CHANNELS[::-1], n_components=2), LogisticRegression()
    )
    with pytest.raises(ValueError, match="channels.*order"):
        learned.cross_fit_signal_classification(
            loso_folds(groups),
            signal,
            labels,
            groups,
            pipeline,
            {},
            inner=InnerSplit("subject"),
            seed=4,
        )


def test_signal_cross_fit_refits_the_transformer_inside_inner_training_groups():
    signal, labels, groups = _signals()
    observed = []

    class AuditCSP(learned.CSPTransformer):
        def fit(self, X, y=None):
            observed.append(X.copy())
            return super().fit(X, y)

    pipeline = learned.learned_pipeline(
        AuditCSP(ch_names=CHANNELS, n_components=2), LogisticRegression()
    )
    folds = loso_folds(groups)
    learned.cross_fit_signal_classification(
        folds[:1],
        signal,
        labels,
        groups,
        pipeline,
        {"model__C": [0.1, 1.0]},
        inner=InnerSplit("subject", n_splits=3),
        seed=4,
    )
    assert len(observed) == 7  # two candidates x three inner fits + outer refit
    assert [len(data) for data in observed].count(16) == 6
    assert len(observed[-1]) == 24
    outer_test = signal.data[folds[0].test]
    for fitted_data in observed:
        assert not any(
            np.array_equal(row, held_out) for row in fitted_data for held_out in outer_test
        )
        fit_groups = {
            groups[i]
            for i, row in enumerate(signal.data)
            if any(np.array_equal(row, fitted) for fitted in fitted_data)
        }
        assert len(fit_groups) in (2, 3)


def test_changing_outer_test_labels_cannot_change_tuned_predictions():
    signal, labels, groups = _signals()
    fold = loso_folds(groups)[0]
    pipeline = learned.learned_pipeline(
        learned.CSPTransformer(ch_names=CHANNELS, n_components=2), LogisticRegression()
    )
    options = dict(inner=InnerSplit("subject", n_splits=3), seed=4)
    first = learned.cross_fit_signal_classification(
        [fold], signal, labels, groups, pipeline, {"model__C": [0.1, 1.0]}, **options
    )[0]
    changed = labels.copy()
    changed[fold.test] = 1 - changed[fold.test]
    second = learned.cross_fit_signal_classification(
        [fold], signal, changed, groups, pipeline, {"model__C": [0.1, 1.0]}, **options
    )[0]
    assert first.best_params == second.best_params
    np.testing.assert_array_equal(first.y_pred, second.y_pred)
    np.testing.assert_array_equal(first.y_prob, second.y_prob)


def test_within_subject_signal_cv_uses_disjoint_runs():
    signal, labels, groups = _signals(n_subjects=1, n_per_subject=32)
    runs = np.repeat(np.arange(4).astype(str), 8)
    folds = within_subject_folds(groups, runs, inner_splits=3, outer_splits=4)
    pipeline = learned.learned_pipeline(
        learned.CSPTransformer(ch_names=CHANNELS, n_components=2), LogisticRegression()
    )
    results = learned.cross_fit_signal_classification(
        folds,
        signal,
        labels,
        groups,
        pipeline,
        {"model__C": [1.0]},
        inner=InnerSplit("run", n_splits=3),
        seed=4,
        runs=runs,
    )
    assert len(results) == 4
    assert all(result.subject == "0" for result in results)


def test_microstates_are_refitted_in_inner_tuning_and_outer_refit():
    signal = _microstate_signals()
    groups = np.repeat(np.arange(4).astype(str), 6)
    labels = np.tile([0, 1], 12)
    observed = []

    class AuditMicrostates(learned.MicrostateTransformer):
        def fit(self, X, y=None):
            observed.append(len(X))
            return super().fit(X, y)

    pipeline = learned.learned_pipeline(
        AuditMicrostates(ch_names=CHANNELS, sfreq=100.0, n_states=2), LogisticRegression()
    )
    result = learned.cross_fit_signal_classification(
        loso_folds(groups)[:1],
        signal,
        labels,
        groups,
        pipeline,
        {"features__min_duration_ms": [0.0, 20.0]},
        inner=InnerSplit("subject", n_splits=3),
        seed=4,
    )[0]
    assert observed.count(12) == 6
    assert observed[-1] == 18
    assert result.y_pred.shape == (6,)
    assert result.best_params["features__min_duration_ms"] in (0.0, 20.0)


def test_cloning_keeps_shared_reference_templates_frozen():
    signal = _microstate_signals()
    reference = MicrostateModel.from_templates(
        np.array([[1.0, -1.0, 0.0, 0.0], [1.0, 1.0, -1.0, -1.0]]),
        ch_names=CHANNELS,
        labels=("A", "B"),
        reference_name="independent-study",
    )
    transformer = learned.MicrostateTransformer(
        ch_names=CHANNELS, sfreq=signal.sfreq, n_states=2, reference=reference
    )
    assert not clone(transformer).reference.templates.flags.writeable


def test_no_grid_still_rejects_refit_false():
    signal, labels, groups = _signals()
    pipeline = learned.learned_pipeline(
        learned.CSPTransformer(ch_names=CHANNELS, n_components=2), LogisticRegression()
    )
    with pytest.raises(ValueError, match="refit=False"):
        learned.cross_fit_signal_classification(
            loso_folds(groups),
            signal,
            labels,
            groups,
            pipeline,
            {},
            inner=InnerSplit("subject"),
            seed=4,
            refit=False,
        )


def test_signal_classification_never_rounds_invalid_prediction_labels():
    class InvalidClassifier(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            self.classes_ = np.array([0, 1])
            return self

        def predict(self, X):
            return np.full(len(X), 0.5)

    signal, labels, groups = _signals()
    pipeline = learned.learned_pipeline(
        learned.CSPTransformer(ch_names=CHANNELS, n_components=2), InvalidClassifier()
    )
    with pytest.raises(ValueError, match="integer.*predictions"):
        learned.cross_fit_signal_classification(
            loso_folds(groups)[:1],
            signal,
            labels,
            groups,
            pipeline,
            {},
            inner=InnerSplit("subject"),
            seed=4,
        )
