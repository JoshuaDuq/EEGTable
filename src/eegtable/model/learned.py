"""Spatial features learned inside grouped training folds."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from eegtable._validation import validate_names
from eegtable.csp import CommonSpatialPattern
from eegtable.microstates import (
    MicrostateModel,
    microstate_coverage,
    microstate_duration,
    microstate_occurrence,
    microstate_transitions,
)
from eegtable.model.crossfit import (
    FoldClassification,
    FoldPrediction,
    _decision_scores,
    _validate_and_resolve_inner_groups,
    _validate_binary_labels,
    _validate_outer_folds,
)
from eegtable.model.execution import run_folds
from eegtable.model.splits import Fold, InnerSplit
from eegtable.model.tuning import fit_untuned, tune
from eegtable.signal import Signal
from eegtable.spectra import Window

__all__ = [
    "CSPTransformer",
    "CovarianceTransformer",
    "MicrostateTransformer",
    "TangentSpaceTransformer",
    "cross_fit_signal_classification",
    "cross_fit_signal_regression",
    "learned_pipeline",
]

_Result = TypeVar("_Result", FoldClassification, FoldPrediction)
_Target = npt.NDArray[np.intp] | npt.NDArray[np.float64]


def _validate_epochs(X: npt.ArrayLike, ch_names: tuple[str, ...]) -> npt.NDArray[np.float64]:
    validate_names(ch_names, "Learned-feature channels")
    if np.iscomplexobj(X):
        raise ValueError("Learned features require real epoch data.")
    data = np.asarray(X, dtype=float)
    if data.ndim != 3 or data.shape[0] == 0 or data.shape[1] != len(ch_names):
        raise ValueError("Epoch data must have shape (n_epochs, fitted_channels, n_times).")
    if data.shape[1] < 2 or data.shape[2] < 2:
        raise ValueError("Learned features require at least two channels and two time samples.")
    if not np.isfinite(data).all():
        raise ValueError("Learned features require finite samples in every epoch.")
    if np.any(np.all(np.ptp(data, axis=-1) == 0.0, axis=1)):
        raise ValueError("Learned features require nonzero temporal variance in every epoch.")
    return data


def _make_signal(data: npt.ArrayLike, ch_names: tuple[str, ...], sfreq: float = 1.0) -> Signal:
    epochs = _validate_epochs(data, ch_names)
    return Signal.from_arrays(
        data=epochs,
        times=np.arange(epochs.shape[-1]) / sfreq,
        sfreq=sfreq,
        ch_names=ch_names,
        row_ids=tuple(("training-array", row, "epoch") for row in range(epochs.shape[0])),
    )


class CSPTransformer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Adapt the package's binary CSP to a fold-fitted scikit-learn pipeline."""

    def __init__(
        self,
        *,
        ch_names: tuple[str, ...],
        n_components: int = 4,
        regularization: float = 0.0,
    ) -> None:
        self.ch_names = ch_names
        self.n_components = n_components
        self.regularization = regularization

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> CSPTransformer:
        if y is None:
            raise ValueError("CSP fitting requires binary class labels.")
        self.model_ = CommonSpatialPattern.fit(
            _make_signal(X, self.ch_names),
            np.asarray(y),
            n_components=self.n_components,
            regularization=self.regularization,
        )
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        check_is_fitted(self, "model_")
        features = self.model_.transform(_make_signal(X, self.ch_names))
        if not np.isfinite(features).all():
            raise ValueError("CSP components must have finite, nonzero projected variance.")
        return features


_MICROSTATE_MEASURES = {
    "coverage": microstate_coverage,
    "occurrence": microstate_occurrence,
    "duration": microstate_duration,
    "transitions": microstate_transitions,
}


class MicrostateTransformer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Fit training templates, then summarize each epoch against the same states.

    ``reference`` identifies state correspondence across fits and must come from
    independent data. Coverage and occurrence are finite even for absent states;
    duration and transitions retain their scientifically undefined NaNs.
    """

    def __init__(
        self,
        *,
        ch_names: tuple[str, ...],
        sfreq: float,
        n_states: int = 4,
        measures: tuple[str, ...] = ("coverage", "occurrence"),
        min_duration_ms: float = 20.0,
        reference: MicrostateModel | None = None,
        random_state: int = 42,
    ) -> None:
        self.ch_names = ch_names
        self.sfreq = sfreq
        self.n_states = n_states
        self.measures = measures
        self.min_duration_ms = min_duration_ms
        self.reference = reference
        self.random_state = random_state

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> MicrostateTransformer:
        if not self.measures or len(set(self.measures)) != len(self.measures):
            raise ValueError("Microstate measures must be nonempty and unique.")
        unknown = set(self.measures).difference(_MICROSTATE_MEASURES)
        if unknown:
            raise ValueError(f"Unknown microstate measures: {sorted(unknown)}.")
        self.model_ = MicrostateModel.fit(
            _make_signal(X, self.ch_names, self.sfreq),
            n_states=self.n_states,
            random_state=self.random_state,
        )
        if self.reference is not None:
            self.model_ = self.model_.match_reference(self.reference)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        check_is_fitted(self, "model_")
        signal = _make_signal(X, self.ch_names, self.sfreq)
        segmentation = self.model_.segment(signal, min_duration_ms=self.min_duration_ms)
        window = Window("epoch", float(signal.times[0]), float(signal.times[-1]))
        return np.concatenate(
            [
                _MICROSTATE_MEASURES[measure](segmentation, windows=[window]).values
                for measure in self.measures
            ],
            axis=1,
        )


def _require_riemann() -> tuple[Any, Any]:
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
    except ImportError as exc:
        raise ImportError(
            "Covariance and tangent features require pyriemann; "
            "install it with: pip install eegtable[riemann]"
        ) from exc
    return Covariances, TangentSpace


def _validate_positive_definite(covariances: npt.NDArray[np.float64]) -> None:
    eigenvalues = np.linalg.eigvalsh(covariances)
    tolerance = eigenvalues[:, -1:] * covariances.shape[1] * np.finfo(float).eps
    if not np.isfinite(covariances).all() or np.any(eigenvalues <= tolerance):
        raise ValueError(
            "Covariances must be positive definite in the training subspace; "
            "use an explicit shrinkage estimator or longer epochs."
        )


class CovarianceTransformer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """pyRiemann covariance estimation in the measured training subspace.

    Rank and projection are estimated from training epochs only. Removed sensor
    dimensions are projected out before explicit shrinkage (default OAS). New
    recordings must retain the same sensor subspace and channel order.
    """

    def __init__(self, *, ch_names: tuple[str, ...], estimator: str = "oas") -> None:
        self.ch_names = ch_names
        self.estimator = estimator

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> CovarianceTransformer:
        covariances, _ = _require_riemann()
        data = _validate_epochs(X, self.ch_names)
        centred = data - data.mean(axis=-1, keepdims=True)
        scatter = np.einsum("ect,edt->cd", centred, centred)
        eigenvalues, eigenvectors = np.linalg.eigh(scatter)
        retained = eigenvalues > eigenvalues[-1] * len(self.ch_names) * np.finfo(float).eps
        self.basis_ = eigenvectors[:, retained]
        self.basis_.setflags(write=False)
        self.rank_ = int(retained.sum())
        if self.rank_ == 0:
            raise ValueError("Training epochs have no measured covariance rank.")
        self.estimation_ = covariances(estimator=self.estimator)
        self.transform(data)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        check_is_fitted(self, "estimation_")
        data = _validate_epochs(X, self.ch_names)
        centred = data - data.mean(axis=-1, keepdims=True)
        projected = np.einsum("cr,ect->ert", self.basis_, centred)
        recovered = np.einsum("cr,ert->ect", self.basis_, projected)
        residual = np.linalg.norm(centred - recovered, axis=(1, 2))
        magnitude = np.linalg.norm(centred, axis=(1, 2))
        if np.any(residual > np.sqrt(np.finfo(float).eps) * magnitude):
            raise ValueError("New epochs do not occupy the fitted training sensor subspace.")
        values = np.asarray(self.estimation_.transform(projected), dtype=float)
        _validate_positive_definite(values)
        return values


class TangentSpaceTransformer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Epoch covariance vectors relative to a training-only Riemannian mean.

    Test-batch reference updates are disabled: a prediction does not depend on
    which other held-out epochs are transformed alongside it.
    """

    def __init__(
        self, *, ch_names: tuple[str, ...], estimator: str = "oas", metric: str = "riemann"
    ) -> None:
        self.ch_names = ch_names
        self.estimator = estimator
        self.metric = metric

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> TangentSpaceTransformer:
        _, tangent_space = _require_riemann()
        self.covariances_ = CovarianceTransformer(
            ch_names=self.ch_names, estimator=self.estimator
        ).fit(X)
        self.tangent_ = tangent_space(metric=self.metric, tsupdate=False).fit(
            self.covariances_.transform(X)
        )
        self.reference_ = self.tangent_.reference_
        self.reference_.setflags(write=False)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        check_is_fitted(self, "tangent_")
        return np.asarray(self.tangent_.transform(self.covariances_.transform(X)), dtype=float)


def learned_pipeline(features: BaseEstimator, estimator: BaseEstimator) -> Pipeline:
    """Compose train-fitted spatial features, scaling, and a supplied estimator."""
    return Pipeline([("features", features), ("scale", StandardScaler()), ("model", estimator)])


def _validate_signal_pipeline(pipeline: Pipeline, signal: Signal) -> None:
    features = pipeline.steps[0][1]
    if not isinstance(
        features,
        (CSPTransformer, MicrostateTransformer, TangentSpaceTransformer, CovarianceTransformer),
    ):
        raise ValueError("Signal pipelines must start with a learned epoch transformer.")
    if features.ch_names != signal.ch_names:
        raise ValueError("Signal pipeline channels must match the recording in the same order.")
    if isinstance(features, MicrostateTransformer) and features.sfreq != signal.sfreq:
        raise ValueError("Microstate pipeline sampling frequency must match the signal.")


def _classification_prediction(
    fold: Fold,
    model: Pipeline,
    X: npt.NDArray[np.float64],
    y: _Target,
    parameters: dict[str, object],
) -> FoldClassification:
    classes = tuple(int(value) for value in model.classes_)
    if classes != (0, 1):
        raise ValueError(f"Fold {fold.index}: classification training requires both classes 0/1.")
    prediction = np.asarray(model.predict(X))
    if (
        prediction.shape != y.shape
        or not np.issubdtype(prediction.dtype, np.integer)
        or not np.isin(prediction, classes).all()
    ):
        raise ValueError(
            f"Fold {fold.index}: integer binary predictions are required for every row."
        )
    probability = (
        np.asarray(model.predict_proba(X), dtype=float) if hasattr(model, "predict_proba") else None
    )
    if probability is not None and (
        probability.shape != (len(X), 2) or not np.isfinite(probability).all()
    ):
        raise ValueError(f"Fold {fold.index}: predictions require finite binary probabilities.")
    return FoldClassification(
        fold=fold.index,
        subject=fold.subject,
        rows=fold.test.copy(),
        y_true=np.asarray(y, dtype=np.intp),
        y_pred=np.asarray(prediction, dtype=np.intp),
        y_prob=probability,
        classes=classes,
        best_params=parameters,
        y_score=_decision_scores(model, X),
    )


def _regression_prediction(
    fold: Fold,
    model: Pipeline,
    X: npt.NDArray[np.float64],
    y: _Target,
    parameters: dict[str, object],
) -> FoldPrediction:
    prediction = np.asarray(model.predict(X), dtype=float)
    if prediction.shape != y.shape or not np.isfinite(prediction).all():
        raise ValueError(f"Fold {fold.index}: regression predictions must be finite and aligned.")
    return FoldPrediction(
        fold=fold.index,
        subject=fold.subject,
        rows=fold.test.copy(),
        y_true=np.asarray(y, dtype=float),
        y_pred=prediction,
        best_params=parameters,
    )


def _cross_fit_signals(
    folds: Sequence[Fold],
    signal: Signal,
    y: _Target,
    groups: npt.NDArray[np.object_],
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    predict: Callable[
        [Fold, Pipeline, npt.NDArray[np.float64], _Target, dict[str, object]], _Result
    ],
    *,
    inner: InnerSplit,
    seed: int,
    runs: npt.NDArray[np.object_] | None,
    scoring: object,
    refit: str | bool | None,
    outer_n_jobs: int,
) -> tuple[_Result, ...]:
    if refit is False:
        raise ValueError("refit=False cannot return a fitted outer-fold model.")
    data = _validate_epochs(signal.data, signal.ch_names)
    target = cast(_Target, np.asarray(y))
    if target.shape != (len(data),) or not np.isfinite(target).all():
        raise ValueError("Targets must be finite with one value per signal epoch.")
    subjects = np.asarray(groups, dtype=object)
    blocks = None if runs is None else np.asarray(runs, dtype=object)
    for name, values in (("groups", subjects), ("runs", blocks)):
        if values is not None and (values.shape != target.shape or pd.isna(values).any()):
            raise ValueError(f"{name} must contain one nonmissing label per signal epoch.")
    _validate_outer_folds(folds, len(data), subjects, blocks)
    inner_groups = _validate_and_resolve_inner_groups(folds, inner, subjects, blocks)
    _validate_signal_pipeline(pipeline, signal)
    for parameters in ParameterGrid(dict(grid)):
        _validate_signal_pipeline(cast(Pipeline, clone(pipeline).set_params(**parameters)), signal)

    def fit_fold(fold: Fold) -> _Result:
        if grid:
            fitted = tune(
                pipeline,
                grid,
                data[fold.train],
                target[fold.train],
                inner_groups[fold.train],
                split=inner,
                seed=seed,
                fold=fold.index,
                scoring=scoring,
                refit=refit,
            )
            model, parameters = fitted.estimator, fitted.best_params
        else:
            model = fit_untuned(
                pipeline, data[fold.train], target[fold.train], seed=seed, fold=fold.index
            )
            parameters = {}
        return predict(fold, model, data[fold.test], target[fold.test], parameters)

    return tuple(run_folds(folds, fit_fold, outer_n_jobs=outer_n_jobs))


def cross_fit_signal_classification(
    folds: Sequence[Fold],
    signal: Signal,
    y: npt.NDArray[np.intp],
    groups: npt.NDArray[np.object_],
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    *,
    inner: InnerSplit,
    seed: int,
    runs: npt.NDArray[np.object_] | None = None,
    scoring: object = "balanced_accuracy",
    refit: str | bool | None = None,
    outer_n_jobs: int = 1,
) -> tuple[FoldClassification, ...]:
    """Nested grouped binary classification directly from epochs.

    Every candidate refits features and scaling inside each inner training fold.
    The chosen pipeline is refitted on the outer training rows and predicts only
    held-out rows. ``groups`` are subjects; within-subject folds also need ``runs``.
    Returned fold containers work with the existing classification metrics.
    """
    labels = np.asarray(y)
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Classification labels must be integers coded 0/1.")
    _validate_binary_labels(labels)
    return _cross_fit_signals(
        folds,
        signal,
        labels,
        groups,
        pipeline,
        grid,
        _classification_prediction,
        inner=inner,
        seed=seed,
        runs=runs,
        scoring=scoring,
        refit=refit,
        outer_n_jobs=outer_n_jobs,
    )


def cross_fit_signal_regression(
    folds: Sequence[Fold],
    signal: Signal,
    y: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_],
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    *,
    inner: InnerSplit,
    seed: int,
    runs: npt.NDArray[np.object_] | None = None,
    scoring: object = "neg_mean_squared_error",
    refit: str | bool | None = None,
    outer_n_jobs: int = 1,
) -> tuple[FoldPrediction, ...]:
    """Nested grouped regression with training-only microstate/tangent features.

    Tuning defaults to negative mean squared error; a standard scikit-learn
    scorer may be supplied. CSP is supervised by binary labels and cannot model
    continuous regression targets. Returns existing held-out fold containers.
    """
    if isinstance(pipeline.steps[0][1], CSPTransformer):
        raise ValueError("CSP requires binary classification labels; use unsupervised features.")
    return _cross_fit_signals(
        folds,
        signal,
        y,
        groups,
        pipeline,
        grid,
        _regression_prediction,
        inner=inner,
        seed=seed,
        runs=runs,
        scoring=scoring,
        refit=refit,
        outer_n_jobs=outer_n_jobs,
    )
