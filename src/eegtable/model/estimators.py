from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import sklearn
from packaging.version import parse as parse_version
from sklearn.compose import TransformedTargetRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
)
from sklearn.feature_selection import f_classif, f_regression
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from eegtable.model import _deps as _deps
from eegtable.model.transformers import PreprocessingConfig, base_preprocessing_steps

__all__ = [
    "elasticnet_grid",
    "elasticnet_pipeline",
    "ensemble_pipeline",
    "hist_gradient_boosting_classifier_grid",
    "hist_gradient_boosting_classifier_pipeline",
    "hist_gradient_boosting_grid",
    "hist_gradient_boosting_pipeline",
    "lda_grid",
    "lda_pipeline",
    "logistic_grid",
    "logistic_pipeline",
    "random_forest_classifier_grid",
    "random_forest_classifier_pipeline",
    "random_forest_grid",
    "random_forest_pipeline",
    "ridge_grid",
    "ridge_pipeline",
    "scaled_ridge_grid",
    "scaled_ridge_pipeline",
    "svm_grid",
    "svm_pipeline",
    "svr_grid",
    "svr_pipeline",
]


def _append_classification_resampler(
    steps: list[tuple[str, object]],
    *,
    resampler: str = "none",
    resampler_seed: int = 42,
) -> None:
    resampler_choice = resampler.strip().lower()
    if resampler_choice == "none":
        return
    if resampler_choice == "undersample":
        try:
            from imblearn.under_sampling import (  # type: ignore
                RandomUnderSampler,
            )
        except ImportError as err:
            msg = "imblearn is required for resampling 'undersample'."
            raise ImportError(msg) from err
        steps.append(("resampler", RandomUnderSampler(random_state=resampler_seed)))
    elif resampler_choice == "smote":
        try:
            from imblearn.over_sampling import SMOTE  # type: ignore
        except ImportError as err:
            msg = "imblearn is required for resampling 'smote'."
            raise ImportError(msg) from err
        steps.append(("resampler", SMOTE(random_state=resampler_seed)))
    else:
        msg = f"Unknown resampler '{resampler}'."
        raise ValueError(msg)


_LR_PENALTIES = ("l1", "l2", "elasticnet", "none")


def _get_lr_kwargs(penalty: str, l1_ratio: float | None = None) -> dict[str, Any]:
    # Handles penalty argument deprecation in scikit-learn >= 1.8.0. On those versions the
    # penalty is translated into l1_ratio/C rather than passed through, so an unrecognised
    # value would silently fall past every branch and fit a default L2 model instead of the
    # error older scikit-learn raised.
    if penalty not in _LR_PENALTIES:
        raise ValueError(f"penalty must be one of {_LR_PENALTIES}, got {penalty!r}.")

    kwargs: dict[str, Any] = {}
    if parse_version(sklearn.__version__) >= parse_version("1.8.0"):
        if penalty == "l2":
            kwargs["l1_ratio"] = 0.0
        elif penalty == "l1":
            kwargs["l1_ratio"] = 1.0
        elif penalty == "elasticnet":
            kwargs["l1_ratio"] = l1_ratio if l1_ratio is not None else 0.5
        elif penalty == "none":
            kwargs["C"] = float("inf")
    else:
        # scikit-learn 1.4 removed the string "none"; None has meant no penalty since 1.2.
        kwargs["penalty"] = None if penalty == "none" else penalty
        if penalty == "elasticnet":
            kwargs["l1_ratio"] = l1_ratio if l1_ratio is not None else 0.5
    return kwargs


def _assemble_pipeline(steps: list[tuple[str, object]], resampler: str) -> Pipeline:
    if resampler.strip().lower() != "none":
        try:
            from imblearn.pipeline import (  # type: ignore
                Pipeline as ImbPipeline,
            )

            return ImbPipeline(steps)
        except ImportError as err:
            msg = "imblearn is required for resampling pipeline."
            raise ImportError(msg) from err
    return Pipeline(steps)


class _UnitFreeElasticNet(ElasticNet):  # type: ignore[misc]
    # ElasticNet's L1 penalty does not scale with the target, so one alpha shrinks a rating in
    # points and the same rating in hundredths of a point differently. Fitting the training
    # fold's target in units of its own SD, and scaling the solution back, makes the grid mean
    # the same thing whatever the units.
    def fit(self, X: Any, y: Any, sample_weight: Any = None, check_input: bool = True) -> Any:
        target = np.asarray(y, dtype=np.float64)
        scale = float(np.std(target)) or 1.0
        fitted = super().fit(
            X, target / scale, sample_weight=sample_weight, check_input=check_input
        )
        self.coef_ = np.asarray(fitted.coef_) * scale
        self.intercept_ = fitted.intercept_ * scale
        return self


def elasticnet_pipeline(
    config: PreprocessingConfig,
    *,
    seed: int,
    n_covariates: int = 0,
    max_iter: int = 10000,
    tol: float = 1e-4,
    selection: str = "cyclic",
) -> Pipeline:
    steps = base_preprocessing_steps(
        config,
        include_scaling=True,
        n_covariates=n_covariates,
        score_func=f_regression,
    )
    steps.append(
        (
            "regressor",
            _UnitFreeElasticNet(
                random_state=seed,
                max_iter=max_iter,
                tol=tol,
                selection=selection,
            ),
        )
    )
    return Pipeline(steps)


class _TrainingScaledRidge(Ridge):  # type: ignore[misc]
    """Scale the penalty by the training row and retained column counts."""

    alpha: float

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
        alpha = self.alpha
        self.alpha_ = alpha * X.shape[0] * X.shape[1]
        self.alpha = self.alpha_
        try:
            return super().fit(X, y, sample_weight=sample_weight)
        finally:
            self.alpha = alpha


def ridge_pipeline(
    config: PreprocessingConfig,
    *,
    seed: int,
    n_covariates: int = 0,
) -> Pipeline:
    steps = base_preprocessing_steps(
        config,
        include_scaling=True,
        n_covariates=n_covariates,
        score_func=f_regression,
    )
    steps.append(("regressor", Ridge(random_state=seed)))
    return Pipeline(steps)


def scaled_ridge_pipeline(
    config: PreprocessingConfig, *, seed: int, n_covariates: int = 0
) -> Pipeline:
    """Fit Ridge with alpha multiplied by training rows × retained columns at each fit."""
    pipeline = ridge_pipeline(config, seed=seed, n_covariates=n_covariates)
    pipeline.set_params(regressor=_TrainingScaledRidge(random_state=seed))
    return pipeline


def random_forest_pipeline(
    config: PreprocessingConfig,
    *,
    seed: int,
    n_covariates: int = 0,
    n_estimators: int = 500,
    n_jobs: int = 1,
    bootstrap: bool = True,
) -> Pipeline:
    steps = base_preprocessing_steps(
        config,
        include_scaling=False,
        n_covariates=n_covariates,
        score_func=f_regression,
    )
    steps.append(
        (
            "rf",
            RandomForestRegressor(
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=seed,
                bootstrap=bootstrap,
            ),
        )
    )
    return Pipeline(steps)


def hist_gradient_boosting_pipeline(
    config: PreprocessingConfig, *, seed: int, n_covariates: int = 0
) -> Pipeline:
    """Fit histogram gradient boosting without an internal validation split."""
    steps = base_preprocessing_steps(
        config, include_scaling=False, n_covariates=n_covariates, score_func=f_regression
    )
    steps.append(("hgb", HistGradientBoostingRegressor(early_stopping=False, random_state=seed)))
    return Pipeline(steps)


def svr_pipeline(config: PreprocessingConfig, *, seed: int, n_covariates: int = 0) -> Pipeline:
    """Fit RBF-SVR with training-only feature and target standardization.

    Predictions retain the original target units. Tune the SVR through
    ``svr__regressor__`` parameters. The estimator is deterministic.
    """
    steps = base_preprocessing_steps(
        config, include_scaling=True, n_covariates=n_covariates, score_func=f_regression
    )
    steps.append(("svr", TransformedTargetRegressor(regressor=SVR(), transformer=StandardScaler())))
    return Pipeline(steps)


def svm_pipeline(
    config: PreprocessingConfig,
    *,
    seed: int,
    kernel: str = "rbf",
    n_covariates: int = 0,
    class_weight: str | dict[Any, Any] | None = "balanced",
    resampler: str = "none",
    resampler_seed: int = 42,
) -> Pipeline:
    steps = base_preprocessing_steps(
        config,
        include_scaling=True,
        n_covariates=n_covariates,
        score_func=f_classif,
    )
    _append_classification_resampler(steps, resampler=resampler, resampler_seed=resampler_seed)
    steps.append(
        (
            "svm",
            SVC(
                kernel=kernel,
                probability=False,
                random_state=seed,
                class_weight=class_weight,
            ),
        )
    )
    return _assemble_pipeline(steps, resampler)


def logistic_pipeline(
    config: PreprocessingConfig,
    *,
    seed: int,
    penalty: str = "l2",
    n_covariates: int = 0,
    max_iter: int = 1000,
    class_weight: str | dict[Any, Any] | None = "balanced",
    resampler: str = "none",
    resampler_seed: int = 42,
) -> Pipeline:
    solver = "saga" if penalty in ("l1", "elasticnet") else "lbfgs"
    steps = base_preprocessing_steps(
        config,
        include_scaling=True,
        n_covariates=n_covariates,
        score_func=f_classif,
    )
    _append_classification_resampler(steps, resampler=resampler, resampler_seed=resampler_seed)
    lr_kwargs = _get_lr_kwargs(penalty=penalty, l1_ratio=0.5 if penalty == "elasticnet" else None)
    steps.append(
        (
            "lr",
            LogisticRegression(
                solver=solver,
                max_iter=max_iter,
                random_state=seed,
                class_weight=class_weight,
                **lr_kwargs,
            ),
        )
    )
    return _assemble_pipeline(steps, resampler)


def random_forest_classifier_pipeline(
    config: PreprocessingConfig,
    *,
    seed: int,
    n_covariates: int = 0,
    n_estimators: int = 500,
    n_jobs: int = 1,
    class_weight: str | dict[Any, Any] | None = "balanced",
    resampler: str = "none",
    resampler_seed: int = 42,
) -> Pipeline:
    steps = base_preprocessing_steps(
        config,
        include_scaling=False,
        n_covariates=n_covariates,
        score_func=f_classif,
    )
    _append_classification_resampler(steps, resampler=resampler, resampler_seed=resampler_seed)
    steps.append(
        (
            "rf",
            RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=seed,
                class_weight=class_weight,
                n_jobs=n_jobs,
            ),
        )
    )
    return _assemble_pipeline(steps, resampler)


def hist_gradient_boosting_classifier_pipeline(
    config: PreprocessingConfig, *, seed: int, n_covariates: int = 0
) -> Pipeline:
    """Fit balanced histogram boosting without an internal validation split."""
    steps = base_preprocessing_steps(
        config, include_scaling=False, n_covariates=n_covariates, score_func=f_classif
    )
    steps.append(
        (
            "hgb",
            HistGradientBoostingClassifier(
                early_stopping=False, random_state=seed, class_weight="balanced"
            ),
        )
    )
    return Pipeline(steps)


def lda_pipeline(config: PreprocessingConfig, *, seed: int, n_covariates: int = 0) -> Pipeline:
    """Fit deterministic LDA with automatic covariance shrinkage and empirical priors."""
    steps = base_preprocessing_steps(
        config, include_scaling=True, n_covariates=n_covariates, score_func=f_classif
    )
    steps.append(("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")))
    return Pipeline(steps)


def ensemble_pipeline(
    config: PreprocessingConfig,
    *,
    seed: int,
    n_covariates: int = 0,
    svm_kernel: str = "rbf",
    lr_penalty: str = "l2",
    rf_n_estimators: int = 500,
    calibrate_ensemble: bool = False,
    resampler: str = "none",
    resampler_seed: int = 42,
) -> Pipeline:
    if calibrate_ensemble:
        raise ValueError(
            "Ensemble probabilities need group-disjoint calibration with preprocessing "
            "fitted inside each calibration split; calibrate_ensemble=True is unsupported."
        )
    svm = SVC(
        kernel=svm_kernel,
        probability=False,
        random_state=seed,
        class_weight="balanced",
    )
    solver = "saga" if lr_penalty in ("l1", "elasticnet") else "lbfgs"
    lr_kwargs = _get_lr_kwargs(
        penalty=lr_penalty,
        l1_ratio=0.5 if lr_penalty == "elasticnet" else None,
    )
    lr = LogisticRegression(
        solver=solver,
        max_iter=1000,
        random_state=seed,
        class_weight="balanced",
        **lr_kwargs,
    )
    rf = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        random_state=seed,
        class_weight="balanced",
        n_jobs=1,
    )

    estimators = [("svm", svm), ("lr", lr), ("rf", rf)]

    steps = base_preprocessing_steps(
        config,
        include_scaling=True,
        n_covariates=n_covariates,
        score_func=f_classif,
    )
    _append_classification_resampler(steps, resampler=resampler, resampler_seed=resampler_seed)
    steps.append(("ensemble", VotingClassifier(estimators=estimators, voting="hard")))
    return _assemble_pipeline(steps, resampler)


# The grids do not tune the variance threshold. It acts on unscaled features, whose units
# differ across a FeatureTable, so any value above zero selects features by unit; the
# pipelines keep it at 0.0, which removes only constant features.


def elasticnet_grid() -> dict[str, list[object]]:
    return {
        "regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "regressor__l1_ratio": [0.2, 0.5, 0.8],
    }


def scaled_ridge_grid() -> dict[str, list[object]]:
    """Dimensionless penalties for :func:`scaled_ridge_pipeline`."""
    return {"regressor__alpha": [10.0**power for power in range(-6, 2)]}


def ridge_grid(X: npt.ArrayLike) -> dict[str, list[object]]:
    # scikit-learn's Ridge does not divide its penalty by the number of trials, and the
    # eigenvalues of a standardized design's Gram matrix sum to n_trials * n_features. A fixed
    # grid therefore stops shrinking as the cohort grows: at 1,200 trials and 12,600 features
    # alpha = 100 barely touches any direction. Scaled by that sum, the grid runs from
    # effectively unpenalized to at most a tenth of a degree of freedom for any design.
    values = np.asarray(X, dtype=float)
    n_features = max(int(np.isfinite(values).any(axis=0).sum()), 1)
    scale = values.shape[0] * n_features
    return {"regressor__alpha": [float(scale * 10.0**power) for power in range(-6, 2)]}


def random_forest_grid() -> dict[str, list[object]]:
    return {
        "rf__max_depth": [5, 10, 20, None],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
    }


def svm_grid() -> dict[str, list[object]]:
    return {
        "svm__C": [0.1, 1.0, 10.0],
        "svm__gamma": ["scale", "auto"],
    }


def logistic_grid(*, penalty: str = "l2") -> dict[str, list[object]]:
    if penalty == "none":
        # C is the inverse penalty strength; searching it would refit this model as L2.
        return {}
    grid: dict[str, list[object]] = {"lr__C": [0.01, 0.1, 1.0, 10.0]}
    if penalty == "elasticnet":
        grid["lr__l1_ratio"] = [0.1, 0.5, 0.9]
    return grid


def random_forest_classifier_grid() -> dict[str, list[object]]:
    return random_forest_grid()


def hist_gradient_boosting_grid() -> dict[str, list[object]]:
    """Tune tree complexity and iteration count through grouped inner folds."""
    return {
        "hgb__max_iter": [100, 200],
        "hgb__max_leaf_nodes": [7, 15],
        "hgb__min_samples_leaf": [10, 20],
        "hgb__l2_regularization": [0.0, 1.0],
    }


def hist_gradient_boosting_classifier_grid() -> dict[str, list[object]]:
    return hist_gradient_boosting_grid()


def svr_grid() -> dict[str, list[object]]:
    """SVR penalties and epsilon are expressed in training-target standard deviations."""
    return {
        "svr__regressor__C": [0.1, 1.0, 10.0],
        "svr__regressor__gamma": ["scale", 0.01, 0.1],
        "svr__regressor__epsilon": [0.01, 0.1],
    }


def lda_grid() -> dict[str, list[object]]:
    """Compare automatic covariance shrinkage with fixed regularization strengths."""
    return {"lda__shrinkage": ["auto", 0.1, 0.5, 0.9]}
