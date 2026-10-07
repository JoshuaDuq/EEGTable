from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer

from eegtable._validation import as_real_array

__all__ = [
    "pearsonr_scorer",
    "safe_pearsonr",
    "scoring_dict",
]


def safe_pearsonr(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    *,
    min_variance: float = 0.0,
) -> tuple[float, float]:
    """Pearson correlation that returns NaN for degenerate input instead of failing.

    Pairs with a non-finite value in either array are dropped. The result is
    ``(nan, nan)`` for arrays of different lengths, fewer than two finite pairs, or
    a constant array; otherwise ``r`` and the two-sided ``p`` of
    :func:`scipy.stats.pearsonr`, with ``r`` clipped to ``[-1, 1]``.

    Parameters
    ----------
    x, y : ndarray
        Paired values.
    min_variance : float, default 0.0
        Sample variance below which either array also counts as constant. The
        default applies no floor, so whether ``r`` is defined does not depend on
        the units of the data.

    Returns
    -------
    r, p : float
    """
    x_arr = as_real_array(x, "x")
    y_arr = as_real_array(y, "y")

    if len(x_arr) != len(y_arr) or len(x_arr) < 2:
        return np.nan, np.nan

    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(valid.sum()) < 2:
        return np.nan, np.nan

    x_valid, y_valid = x_arr[valid], y_arr[valid]
    if np.all(x_valid == x_valid[0]) or np.all(y_valid == y_valid[0]):
        return np.nan, np.nan
    if min_variance > 0.0 and (
        np.var(x_valid, ddof=1) < min_variance or np.var(y_valid, ddof=1) < min_variance
    ):
        return np.nan, np.nan

    r, p = pearsonr(x_valid, y_valid)
    if not (np.isfinite(r) and np.isfinite(p)):
        return np.nan, np.nan

    # Guard against floating-point overshoot past the theoretical bound [-1, 1].
    return float(np.clip(r, -1.0, 1.0)), float(p)


def _selection_pearsonr(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    yt = as_real_array(y_true, "y_true")
    yp = as_real_array(y_pred, "y_pred")
    if not np.isfinite(yt).all() or not np.isfinite(yp).all():
        raise ValueError("Model selection requires finite targets and predictions for every trial.")
    r, _ = safe_pearsonr(yt, yp)
    finite = np.isfinite(yt) & np.isfinite(yp)
    if np.isfinite(r) or int(finite.sum()) < 2:
        return r
    # A candidate whose predictions do not vary has no linear association with the target,
    # so it scores 0 and loses the search; NaN would make the non-finite-score guard abort
    # the whole fold. A target that does not vary is a data fault and stays undefined.
    if np.all(yp == yp[0]) and not np.all(yt == yt[0]):
        return 0.0
    return r


def pearsonr_scorer() -> Callable[..., float]:
    """Scikit-learn scorer of the Pearson ``r`` between targets and predictions.

    The correlation is pooled over all validation rows, so subject offsets enter
    it; :func:`subject_r_scorer` correlates within subjects instead. Constant
    predictions score 0, so such a candidate loses a search instead of aborting it;
    a constant target stays NaN. Non-finite targets or predictions raise.

    Returns
    -------
    callable
        Scorer for a ``scoring`` argument; greater is better.
    """
    scorer = make_scorer(_selection_pearsonr, greater_is_better=True)
    return cast(Callable[..., float], scorer)


def scoring_dict() -> dict[str, object]:
    """Multi-metric scoring of pooled Pearson ``r`` and negative mean squared error.

    Returns
    -------
    dict
        ``{"r": pearsonr_scorer(), "neg_mse": "neg_mean_squared_error"}``. Pass
        ``refit`` naming the metric that chooses the candidate.
    """
    return {"r": pearsonr_scorer(), "neg_mse": "neg_mean_squared_error"}
