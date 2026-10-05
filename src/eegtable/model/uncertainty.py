from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline

from eegtable.model.splits import _GroupKFold
from eegtable.model.tuning import _validate_boosting_early_stopping, fit_untuned

__all__ = [
    "Method",
    "PredictionIntervals",
    "prediction_intervals",
]

Method = Literal["split", "cv_plus", "quantile"]


@dataclass(frozen=True)
class PredictionIntervals:
    lower: npt.NDArray[np.float64]
    upper: npt.NDArray[np.float64]
    alpha: float
    method: Method
    calibration_unit: Literal["trial", "subject"]


def _compute_conformal_quantile(residuals: npt.NDArray[np.float64], alpha: float) -> float:
    arr = np.sort(np.asarray(residuals, dtype=np.float64))
    if not np.all(np.isfinite(arr)):
        raise ValueError("Calibration scores must be finite; no observations may be dropped.")
    n_cal = len(arr)
    if n_cal == 0:
        raise ValueError("Calibration set cannot be empty.")
    k = int(np.ceil((n_cal + 1) * (1.0 - alpha)))
    if k > n_cal:
        return float("inf")
    return float(arr[k - 1])


def _order_stat_quantile(values: npt.NDArray[np.float64], alpha: float, *, tail: str) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    if not np.all(np.isfinite(arr)):
        raise ValueError("Calibration candidates must be finite; no observations may be dropped.")
    n = arr.size
    if n == 0:
        return np.nan
    if tail == "upper":
        k = int(np.ceil((1.0 - alpha) * (n + 1)))
        if k > n:
            return float("inf")
        if k <= 0:
            return float("-inf")
        return float(arr[k - 1])
    k = int(np.floor(alpha * (n + 1)))
    if k <= 0:
        return float("-inf")
    if k > n:
        return float("inf")
    return float(arr[k - 1])


def _split_conformal(
    model: Pipeline,
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    X_test: npt.NDArray[np.float64],
    alpha: float,
    seed: int,
    groups: npt.NDArray[np.object_] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    n = len(X_train)
    if n < 5:
        msg = "Split conformal requires at least 5 training samples."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    if groups is None:
        n_cal = min(max(int(0.2 * n), 2), n - 2)
        indices = rng.permutation(n)
        cal_idx = indices[:n_cal]
        train_idx = indices[n_cal:]
    else:
        groups_arr = np.asarray(groups, dtype=object)
        unique_groups = [g for g in pd.unique(groups_arr) if not pd.isna(g)]
        if len(unique_groups) < 2:
            msg = (
                f"Group-aware split conformal requires at least 2 unique groups, "
                f"got {len(unique_groups)}."
            )
            raise ValueError(msg)
        n_cal_groups = min(max(1, int(round(0.2 * len(unique_groups)))), len(unique_groups) - 1)
        test_size = float(n_cal_groups / len(unique_groups))
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        dummy_x = np.zeros((n, 1), dtype=np.float64)
        train_idx, cal_idx = next(splitter.split(dummy_x, y_train, groups=groups_arr))

    if len(train_idx) < 2 or len(cal_idx) < 2:
        msg = "Split conformal could not form valid train/calibration splits."
        raise ValueError(msg)

    model_proper = fit_untuned(model, X_train[train_idx], y_train[train_idx], seed=seed)

    residuals = np.abs(y_train[cal_idx] - model_proper.predict(X_train[cal_idx]))
    q_hat = _compute_conformal_quantile(residuals, alpha)

    y_test_pred = np.asarray(model_proper.predict(X_test), dtype=np.float64)
    if not np.all(np.isfinite(y_test_pred)):
        raise ValueError("Test predictions must be finite.")
    return y_test_pred - q_hat, y_test_pred + q_hat


def _get_cv_splits(
    cv_splits: int,
    seed: int,
    groups: npt.NDArray[np.object_] | None,
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
) -> list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
    n = len(X_train)
    if groups is not None:
        groups_arr = np.asarray(groups, dtype=object)
        unique_groups = [g for g in pd.unique(groups_arr) if not pd.isna(g)]
        n_unique = len(unique_groups)
        if n_unique >= cv_splits:
            splitter = _GroupKFold(n_splits=cv_splits)
            splits = splitter.split(X_train, y_train, groups=groups_arr)
        else:
            splitter_logo = LeaveOneGroupOut()
            splits = splitter_logo.split(X_train, y_train, groups=groups_arr)
    else:
        if cv_splits > n:
            msg = f"Cannot have cv_splits={cv_splits} greater than n_samples={n}."
            raise ValueError(msg)
        splitter_kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        splits = splitter_kf.split(X_train)

    return [(tr.astype(np.intp), val.astype(np.intp)) for tr, val in splits]


def _conformal_cv_plus(
    model: Pipeline,
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    X_test: npt.NDArray[np.float64],
    alpha: float,
    cv_splits: int,
    seed: int,
    groups: npt.NDArray[np.object_] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    splits = _get_cv_splits(cv_splits, seed, groups, X_train, y_train)
    n_test = len(X_test)
    lower_chunks: list[npt.NDArray[np.float64]] = []
    upper_chunks: list[npt.NDArray[np.float64]] = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        model_fold = fit_untuned(
            model, X_train[train_idx], y_train[train_idx], seed=seed, fold=fold
        )

        val_preds = np.asarray(model_fold.predict(X_train[val_idx]), dtype=np.float64)
        residuals = np.abs(y_train[val_idx] - val_preds)
        test_preds = np.asarray(model_fold.predict(X_test), dtype=np.float64)
        lower_chunks.append(test_preds[:, None] - residuals[None, :])
        upper_chunks.append(test_preds[:, None] + residuals[None, :])

    return _cv_plus_bounds(lower_chunks, upper_chunks, alpha, n_test)


def _cv_plus_bounds(
    lower_chunks: list[npt.NDArray[np.float64]],
    upper_chunks: list[npt.NDArray[np.float64]],
    alpha: float,
    n_test: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if not lower_chunks or not upper_chunks:
        msg = "CV+ calibration failed: no valid fold calibration chunks were produced."
        raise ValueError(msg)

    lower_candidates = np.concatenate(lower_chunks, axis=1)
    upper_candidates = np.concatenate(upper_chunks, axis=1)

    lower = np.full(n_test, np.nan, dtype=np.float64)
    upper = np.full(n_test, np.nan, dtype=np.float64)
    for i in range(n_test):
        lower[i] = _order_stat_quantile(lower_candidates[i, :], alpha, tail="lower")
        upper[i] = _order_stat_quantile(upper_candidates[i, :], alpha, tail="upper")

    return lower, upper


def _quantile_model(model: Pipeline, quantile: float, seed: int) -> Pipeline:
    # Quantile regression needs a quantile loss, so the model's final estimator is replaced;
    # its preprocessing (imputation, scaling, selection) is kept.
    regressor = GradientBoostingRegressor(loss="quantile", alpha=quantile, random_state=seed)
    if not isinstance(model, Pipeline):
        return Pipeline([("regressor", regressor)])
    quantile_model = cast(Pipeline, clone(model))
    quantile_model.set_params(**{quantile_model.steps[-1][0]: regressor})
    return quantile_model


def _conformal_quantile(
    model: Pipeline,
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    X_test: npt.NDArray[np.float64],
    alpha: float,
    cv_splits: int,
    seed: int,
    groups: npt.NDArray[np.object_] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # Conformalized quantile regression in CV+ form: each fold's quantile models are
    # calibrated on the trials they did not see, and those same models give the interval.
    # Refitting on all data after calibrating out of fold would lose the coverage guarantee.
    splits = _get_cv_splits(cv_splits, seed, groups, X_train, y_train)
    lower_chunks: list[npt.NDArray[np.float64]] = []
    upper_chunks: list[npt.NDArray[np.float64]] = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        low = fit_untuned(
            _quantile_model(model, alpha / 2.0, seed),
            X_train[train_idx],
            y_train[train_idx],
            seed=seed,
            fold=fold,
        )
        high = fit_untuned(
            _quantile_model(model, 1.0 - alpha / 2.0, seed),
            X_train[train_idx],
            y_train[train_idx],
            seed=seed,
            fold=fold,
        )

        y_val = y_train[val_idx]
        scores = np.maximum(
            np.asarray(low.predict(X_train[val_idx]), dtype=np.float64) - y_val,
            y_val - np.asarray(high.predict(X_train[val_idx]), dtype=np.float64),
        )
        lower_chunks.append(
            np.asarray(low.predict(X_test), dtype=np.float64)[:, None] - scores[None, :]
        )
        upper_chunks.append(
            np.asarray(high.predict(X_test), dtype=np.float64)[:, None] + scores[None, :]
        )

    return _cv_plus_bounds(lower_chunks, upper_chunks, alpha, len(X_test))


def prediction_intervals(
    model: Pipeline,
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    X_test: npt.NDArray[np.float64],
    *,
    alpha: float = 0.1,
    method: Method = "cv_plus",
    cv_splits: int = 5,
    seed: int = 42,
    groups: npt.NDArray[np.object_] | None = None,
) -> PredictionIntervals:
    """Compute intervals calibrated over trials.

    Split conformal targets coverage 1 - alpha for exchangeable trials. The
    CV+ methods use alpha in each tail, not a universal 1 - alpha guarantee.
    Group-disjoint fitting alone does not establish coverage for dependent
    trials or new subjects; calibration scores are still pooled over trials.
    ``seed`` controls both splits and stochastic pipeline fitting, without
    mutating the supplied model or the caller's global random state.
    """
    if not (0.0 < alpha < 1.0):
        msg = f"alpha must be between 0.0 and 1.0, got {alpha}."
        raise ValueError(msg)

    valid_methods: tuple[str, ...] = ("split", "cv_plus", "quantile")
    if method not in valid_methods:
        msg = f"Unknown method {method!r}. Expected one of: {valid_methods}."
        raise ValueError(msg)

    X_tr = np.asarray(X_train, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.float64)
    X_te = np.asarray(X_test, dtype=np.float64)

    if y_tr.ndim != 1 or not np.all(np.isfinite(y_tr)):
        raise ValueError("y_train must be a finite 1-D array.")

    if X_tr.ndim != 2 or X_te.ndim != 2:
        raise ValueError("X_train and X_test must be 2-D.")

    if len(X_tr) == 0 or len(y_tr) == 0 or len(X_te) == 0:
        msg = "Training and test arrays must not be empty."
        raise ValueError(msg)

    if len(y_tr) != len(X_tr):
        raise ValueError(f"y_train has {len(y_tr)} rows and X_train has {len(X_tr)}.")

    if groups is not None and len(groups) != len(X_tr):
        raise ValueError(f"groups has {len(groups)} rows and X_train has {len(X_tr)}.")

    if X_te.shape[1] != X_tr.shape[1]:
        raise ValueError(f"X_test has {X_te.shape[1]} columns and X_train has {X_tr.shape[1]}.")

    if groups is not None and method in ("split", "cv_plus"):
        _validate_boosting_early_stopping(model)

    # Scores remain one per trial, even with group-disjoint splits.
    cal_unit: Literal["trial", "subject"] = "trial"

    if method == "split":
        lower, upper = _split_conformal(model, X_tr, y_tr, X_te, alpha, seed, groups)
    elif method == "cv_plus":
        lower, upper = _conformal_cv_plus(model, X_tr, y_tr, X_te, alpha, cv_splits, seed, groups)
    else:
        lower, upper = _conformal_quantile(model, X_tr, y_tr, X_te, alpha, cv_splits, seed, groups)

    return PredictionIntervals(
        lower=lower,
        upper=upper,
        alpha=alpha,
        method=method,
        calibration_unit=cal_unit,
    )
