from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from eegtable.model._nuisance import fit_coefficients

__all__ = [
    "Deconfounder",
    "DropAllNaNColumns",
    "MissingnessThreshold",
    "PreprocessingConfig",
    "ReplaceInfWithNaN",
    "SpatialFeatureSelector",
    "VarianceThreshold",
    "base_preprocessing_steps",
    "transform_feature_names",
    "validate_subject_missingness",
]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Settings of the training-fitted steps built by :func:`base_preprocessing_steps`.

    Parameters
    ----------
    max_feature_missingness : float, default 0.2
        Largest fraction of missing training values a feature may have and still
        be kept, in ``[0, 1]``.
    max_subject_missingness : float, default 0.5
        Largest fraction of missing cells, over the kept features, allowed for any
        one training subject, in ``[0, 1]``. Cross-fitting raises when a subject
        exceeds it rather than dropping the subject.
    feature_selection_percentile : float, optional
        Keep this percentile of features ranked by a univariate F statistic
        against the training target (``f_regression`` or ``f_classif``). None or
        100 skips selection.
    deconfound : bool, default False
        Regress the covariate columns out of the features, after which the
        estimator sees only the feature residuals. Ignored without covariates.
    pca_enabled : bool, default False
        Standardize the features and reduce them with PCA.
    pca_n_components : int or float, optional
        Number of components, or the fraction of variance to keep; None keeps 95%.
    pca_whiten : bool, default False
        Whiten the components.
    pca_svd_solver : str, default "auto"
        scikit-learn PCA solver.
    pca_random_state : int, default 42
        PCA random state, used by the randomized solvers.
    """

    max_feature_missingness: float = 0.2
    max_subject_missingness: float = 0.5
    feature_selection_percentile: float | None = None
    deconfound: bool = False
    pca_enabled: bool = False
    pca_n_components: int | float | None = None
    pca_whiten: bool = False
    pca_svd_solver: str = "auto"
    pca_random_state: int = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_feature_missingness <= 1.0:
            raise ValueError(
                f"max_feature_missingness must be in [0, 1], got {self.max_feature_missingness}"
            )
        if not 0.0 <= self.max_subject_missingness <= 1.0:
            raise ValueError(
                f"max_subject_missingness must be in [0, 1], got {self.max_subject_missingness}"
            )


def validate_subject_missingness(
    values: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_],
    *,
    maximum: float,
) -> None:
    """Raise when any group's fraction of NaN cells exceeds ``maximum``.

    Parameters
    ----------
    values : ndarray, shape (n_rows, n_features)
        The retained feature columns; at least one is required. Only NaN counts as
        missing.
    groups : ndarray, shape (n_rows,)
        Group label of each row, usually the subject.
    maximum : float
        Largest allowed fraction of NaN over all of a group's cells.
    """
    x_arr = np.asarray(values, dtype=float)
    groups_arr = np.asarray(groups)

    if x_arr.shape[1] == 0:
        raise ValueError("validate_subject_missingness requires at least one retained feature.")

    for grp in np.unique(groups_arr):
        mask = groups_arr == grp
        sub_x = x_arr[mask]
        sub_missingness = float(np.isnan(sub_x).mean())
        if sub_missingness > maximum:
            raise ValueError(
                f"Subject {grp} has missingness {sub_missingness:.2f} > maximum {maximum:.2f}."
            )


class ReplaceInfWithNaN(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Turn infinite values into NaN, so later steps treat them as missing. Stateless."""

    def fit(self, X: Any, y: Any = None) -> ReplaceInfWithNaN:
        _ = (X, y)
        return self

    def transform(self, X: Any) -> npt.NDArray[np.float64]:
        out = np.asarray(X, dtype=float).copy()
        out[np.isinf(out)] = np.nan
        return out

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        if input_features is None:
            raise ValueError("input_features is required for get_feature_names_out.")
        return list(input_features)


class DropAllNaNColumns(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Drop columns with fewer than ``min_finite`` finite values in the fitting rows.

    Parameters
    ----------
    min_finite : int, default 1
        Finite values a column needs to be kept. The default drops only columns
        with no finite value at all.
    """

    def __init__(self, min_finite: int = 1) -> None:
        self.min_finite = min_finite

    def fit(self, X: Any, y: Any = None) -> DropAllNaNColumns:
        _ = y
        x_arr = np.asarray(X, dtype=float)
        finite_counts = np.sum(np.isfinite(x_arr), axis=0)
        self.support_mask_ = finite_counts >= self.min_finite
        return self

    def transform(self, X: Any) -> npt.NDArray[np.float64]:
        x_arr = np.asarray(X, dtype=float)
        return x_arr[:, self.support_mask_]

    def get_support(self, indices: bool = False) -> npt.NDArray[Any]:
        if indices:
            return cast(npt.NDArray[np.intp], np.flatnonzero(self.support_mask_))
        return cast(npt.NDArray[np.bool_], self.support_mask_)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        if input_features is None:
            raise ValueError("input_features is required for get_feature_names_out.")
        return [f for f, keep in zip(input_features, self.support_mask_, strict=True) if keep]


class VarianceThreshold(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Drop columns whose variance in the fitting rows, ignoring NaN, is not above a threshold.

    A column without any finite value is dropped as well.

    Parameters
    ----------
    threshold : float, default 0.0
        Variance a column must exceed. The default removes only constant columns.
        A positive value compares variances of features measured in different
        units, so it selects by unit.
    """

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def fit(self, X: Any, y: Any = None) -> VarianceThreshold:
        _ = y
        x_arr = np.asarray(X, dtype=float)
        self.variances_ = np.nanvar(x_arr, axis=0)
        if self.threshold == 0.0:
            # A constant decimal can acquire a positive variance through mean rounding.
            spread = np.nanmax(x_arr, axis=0) - np.nanmin(x_arr, axis=0)
            self.variances_[spread == 0.0] = 0.0
        self.support_mask_ = self.variances_ > self.threshold
        return self

    def transform(self, X: Any) -> npt.NDArray[np.float64]:
        x_arr = np.asarray(X, dtype=float)
        return x_arr[:, self.support_mask_]

    def get_support(self, indices: bool = False) -> npt.NDArray[Any]:
        if indices:
            return cast(npt.NDArray[np.intp], np.flatnonzero(self.support_mask_))
        return cast(npt.NDArray[np.bool_], self.support_mask_)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        if input_features is None:
            raise ValueError("input_features is required for get_feature_names_out.")
        return [f for f, keep in zip(input_features, self.support_mask_, strict=True) if keep]


class MissingnessThreshold(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Keep the features whose NaN fraction in the fitting rows is within a limit.

    Parameters
    ----------
    max_feature_missingness : float, default 0.2
        Largest NaN fraction a kept feature may have.
    max_subject_missingness : float, default 0.5
        Largest NaN fraction over the kept features for any one group. ``fit``
        checks it only when given ``groups``, which scikit-learn pipelines do not
        pass; cross-fitting checks it on the fitted pipeline instead.
    """

    def __init__(
        self,
        max_feature_missingness: float = 0.2,
        max_subject_missingness: float = 0.5,
    ) -> None:
        self.max_feature_missingness = max_feature_missingness
        self.max_subject_missingness = max_subject_missingness

    def fit(self, X: Any, y: Any = None, groups: Any = None) -> MissingnessThreshold:
        _ = y
        x_arr = np.asarray(X, dtype=float)
        feat_missingness = np.isnan(x_arr).mean(axis=0)
        self.support_mask_ = feat_missingness <= self.max_feature_missingness
        if groups is not None and np.any(self.support_mask_):
            validate_subject_missingness(
                x_arr[:, self.support_mask_],
                groups,
                maximum=self.max_subject_missingness,
            )
        return self

    def transform(self, X: Any) -> npt.NDArray[np.float64]:
        x_arr = np.asarray(X, dtype=float)
        return x_arr[:, self.support_mask_]

    def get_support(self, indices: bool = False) -> npt.NDArray[Any]:
        if indices:
            return cast(npt.NDArray[np.intp], np.flatnonzero(self.support_mask_))
        return cast(npt.NDArray[np.bool_], self.support_mask_)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        if input_features is None:
            raise ValueError("input_features is required for get_feature_names_out.")
        return [f for f, keep in zip(input_features, self.support_mask_, strict=True) if keep]


class SpatialFeatureSelector(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Keep the features whose names contain any of the given region tokens.

    Matching is a case-insensitive substring test on the feature name, so a short
    token can match unintended names. ``Selection(space=...)`` when building the
    design matches the metadata exactly instead. ``fit`` raises when no feature
    matches.

    Parameters
    ----------
    allowed_regions : sequence of str
        Tokens to look for. Empty keeps every feature.
    feature_names : sequence of str, optional
        One name per column. Required when ``allowed_regions`` is set, unless ``X``
        is a DataFrame whose columns supply the names.
    """

    def __init__(
        self,
        allowed_regions: Sequence[str] = (),
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self.allowed_regions = tuple(allowed_regions)
        self.feature_names = tuple(feature_names) if feature_names is not None else None

    def fit(self, X: Any, y: Any = None) -> SpatialFeatureSelector:
        _ = y
        x_arr = np.asarray(X, dtype=float)
        n_features = x_arr.shape[1]

        if not self.allowed_regions:
            self.support_mask_ = np.ones(n_features, dtype=bool)
            return self

        names = self.feature_names
        if names is None and hasattr(X, "columns"):
            names = tuple(str(c) for c in X.columns)

        if names is None or len(names) != n_features:
            raise ValueError("SpatialFeatureSelector requires feature names when regions are set.")

        allowed_set = {r.strip().lower() for r in self.allowed_regions if r.strip()}
        # Match feature name containing any of the allowed region tokens
        support: list[bool] = []
        for name in names:
            name_lower = name.lower()
            matched = any(r in name_lower for r in allowed_set)
            support.append(matched)

        self.support_mask_ = np.array(support, dtype=bool)
        if not np.any(self.support_mask_):
            raise ValueError(f"No features matched allowed regions: {self.allowed_regions}")
        return self

    def transform(self, X: Any) -> npt.NDArray[np.float64]:
        x_arr = np.asarray(X, dtype=float)
        return x_arr[:, self.support_mask_]

    def get_support(self, indices: bool = False) -> npt.NDArray[Any]:
        if indices:
            return cast(npt.NDArray[np.intp], np.flatnonzero(self.support_mask_))
        return cast(npt.NDArray[np.bool_], self.support_mask_)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        if input_features is None:
            raise ValueError("input_features is required for get_feature_names_out.")
        return [f for f, keep in zip(input_features, self.support_mask_, strict=True) if keep]


class Deconfounder(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Replace features by their residuals on the trailing covariate columns.

    ``fit`` regresses every feature column on an intercept and the last
    ``n_covariates`` columns by least squares; the covariate design must be finite
    and of full rank. ``transform`` subtracts the fitted prediction and returns the
    feature residuals without the covariate columns. Expects finite input, as
    after imputation.

    Parameters
    ----------
    n_covariates : int, default 0
        Number of trailing covariate columns. 0 passes the input through.
    """

    def __init__(self, n_covariates: int = 0) -> None:
        self.n_covariates = int(n_covariates)

    def fit(self, X: Any, y: Any = None) -> Deconfounder:
        _ = y
        if self.n_covariates <= 0:
            return self

        x_arr = np.asarray(X, dtype=float)
        n_features = x_arr.shape[1] - self.n_covariates
        if n_features <= 0:
            raise ValueError(
                f"X has {x_arr.shape[1]} columns, but n_covariates={self.n_covariates}."
            )

        features = x_arr[:, :n_features]
        covariates = x_arr[:, n_features:]

        cov_design = np.column_stack([np.ones(len(x_arr)), covariates])
        self.coef_ = fit_coefficients(cov_design, features)
        return self

    def transform(self, X: Any) -> npt.NDArray[np.float64]:
        x_arr = np.asarray(X, dtype=float)
        if self.n_covariates <= 0:
            return x_arr

        n_features = x_arr.shape[1] - self.n_covariates
        features = x_arr[:, :n_features]
        covariates = x_arr[:, n_features:]

        cov_design = np.column_stack([np.ones(len(x_arr)), covariates])
        residuals = features - (cov_design @ self.coef_)
        return residuals

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        if input_features is None:
            raise ValueError("input_features is required for get_feature_names_out.")
        if self.n_covariates <= 0:
            return list(input_features)
        n_features = len(input_features) - self.n_covariates
        return list(input_features[:n_features])


class _CovariateImputer(SimpleImputer):  # type: ignore[misc]
    """Impute covariates without silently removing their fixed trailing columns."""

    def fit(self, X: Any, y: Any = None) -> _CovariateImputer:
        missing = np.flatnonzero(~np.isfinite(np.asarray(X, dtype=float)).any(axis=0))
        if missing.size:
            raise ValueError(
                f"Covariate columns {missing.tolist()} have no finite training values."
            )
        super().fit(X, y)
        return self


def base_preprocessing_steps(
    config: PreprocessingConfig,
    *,
    include_scaling: bool,
    n_covariates: int = 0,
    score_func: Callable[..., object] | None = None,
) -> list[tuple[str, object]]:
    """The training-fitted preprocessing steps shared by the pipeline factories.

    Feature steps, in order: infinities to NaN, dropping all-NaN columns, the
    missingness limit, median imputation and dropping constant columns, then
    optional percentile selection, scaling and PCA. With covariates, the trailing
    ``n_covariates`` columns get their own branch inside a ``ColumnTransformer``:
    infinities to NaN, most-frequent imputation, then scaling or passthrough,
    optionally followed by :class:`Deconfounder`. A covariate without any finite
    training value raises instead of being dropped by imputation.

    Parameters
    ----------
    config : PreprocessingConfig
        Missingness limits, feature selection, deconfounding and PCA settings.
    include_scaling : bool
        Standardize the features and covariates. PCA standardizes the features
        regardless.
    n_covariates : int, default 0
        Trailing covariate columns of ``X``.
    score_func : callable, optional
        Univariate score for percentile selection; None uses ``f_regression``.

    Returns
    -------
    list of (str, object)
        Named pipeline steps, to which the factories append an estimator.
    """
    # Choosing channels or ROIs is a fixed choice of columns, made with Selection(space=...)
    # when the design is built, so no spatial step is fitted here.
    feature_steps: list[tuple[str, object]] = [
        ("finite", ReplaceInfWithNaN()),
        ("drop_all_nan", DropAllNaNColumns()),
        (
            "missingness",
            MissingnessThreshold(
                max_feature_missingness=config.max_feature_missingness,
                max_subject_missingness=config.max_subject_missingness,
            ),
        ),
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold()),
    ]

    if (
        config.feature_selection_percentile is not None
        and config.feature_selection_percentile < 100.0
    ):
        from sklearn.feature_selection import SelectPercentile, f_regression

        chosen_score_func = score_func or f_regression
        feature_steps.append(
            (
                "k_best",
                SelectPercentile(
                    chosen_score_func,
                    percentile=config.feature_selection_percentile,
                ),
            )
        )

    if include_scaling or config.pca_enabled:
        feature_steps.append(("scaler", StandardScaler()))

    if config.pca_enabled:
        pca_components = 0.95 if config.pca_n_components is None else config.pca_n_components
        feature_steps.append(
            (
                "pca",
                PCA(
                    n_components=pca_components,
                    whiten=config.pca_whiten,
                    random_state=config.pca_random_state,
                    svd_solver=config.pca_svd_solver,
                ),
            )
        )

    if n_covariates > 0:
        cov_steps: list[tuple[str, object]] = [
            ("finite", ReplaceInfWithNaN()),
            ("impute", _CovariateImputer(strategy="most_frequent")),
        ]
        if include_scaling:
            cov_steps.append(("scaler", StandardScaler()))
        else:
            cov_steps.append(
                (
                    "passthrough",
                    FunctionTransformer(func=None, validate=False, feature_names_out="one-to-one"),
                )
            )

        def feature_idx(X: npt.NDArray[Any]) -> list[int]:
            return list(range(X.shape[1] - n_covariates))

        def cov_idx(X: npt.NDArray[Any]) -> list[int]:
            return list(range(X.shape[1] - n_covariates, X.shape[1]))

        preprocessor = ColumnTransformer(
            transformers=[
                ("eeg", Pipeline(feature_steps), feature_idx),
                ("cov", Pipeline(cov_steps), cov_idx),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        steps: list[tuple[str, object]] = [("preprocessing", preprocessor)]
        if config.deconfound:
            steps.append(("deconfound", Deconfounder(n_covariates=n_covariates)))
        return steps

    return feature_steps


def transform_feature_names(
    steps: Sequence[tuple[str, object]],
    feature_names: Sequence[str],
) -> list[str]:
    """Carry input feature names through fitted pipeline steps.

    A step with ``get_feature_names_out`` renames or drops columns; a step with
    only ``get_support`` drops the unsupported ones. Passthrough steps preserve
    names; any other step without a column mapping raises.

    Parameters
    ----------
    steps : sequence of (str, object)
        Fitted steps, e.g. ``pipeline.steps[:-1]``.
    feature_names : sequence of str
        Names of the columns entering the first step.

    Returns
    -------
    list of str
        Names of the columns leaving the last step. A step that derives new
        columns, such as PCA, reports its own names (``"pca0"``, ...).
    """
    # A step that cannot report its output names raises: keeping the input names would
    # silently attribute values to the wrong columns.
    names = list(feature_names)
    for _name, step in steps:
        if step is None or (isinstance(step, str) and step == "passthrough"):
            continue
        if hasattr(step, "get_feature_names_out"):
            names = list(step.get_feature_names_out(names))
        elif hasattr(step, "get_support"):
            support = step.get_support()
            names = [f for f, keep in zip(names, support, strict=True) if keep]
        else:
            raise ValueError(
                f"{type(step).__name__} must report output feature names or feature support "
                "to map transformed columns to input features."
            )
    return names


def _missingness_inputs(
    estimator: object, X: npt.NDArray[np.float64]
) -> Iterator[tuple[npt.NDArray[np.float64], MissingnessThreshold]]:
    # Each fitted MissingnessThreshold, with the data it saw when it was fitted.
    if isinstance(estimator, MissingnessThreshold):
        yield X, estimator
    elif isinstance(estimator, Pipeline):
        data = X
        for position, (_, step) in enumerate(estimator.steps):
            if step is None or step == "passthrough":
                continue
            yield from _missingness_inputs(step, data)
            if position == len(estimator.steps) - 1 or not hasattr(step, "transform"):
                break
            data = step.transform(data)
    elif isinstance(estimator, ColumnTransformer):
        for _, transformer, columns in estimator.transformers_:
            if not isinstance(transformer, str):
                selected = X[:, columns]
                if selected.size:
                    yield from _missingness_inputs(transformer, selected)


def _check_subject_missingness(
    model: object,
    X: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_],
) -> None:
    # Pipelines never route groups to their steps, so max_subject_missingness is applied to
    # the fitted model instead, against the columns each missingness step kept. Traversing
    # its preprocessing must not change input shared with other candidates or scorers.
    for seen, step in _missingness_inputs(model, np.array(X, dtype=np.float64, copy=True)):
        if np.any(step.support_mask_):
            validate_subject_missingness(
                seen[:, step.support_mask_], groups, maximum=step.max_subject_missingness
            )
