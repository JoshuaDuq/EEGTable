from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from eegtable._validation import as_real_array
from eegtable.model.aggregate import AggregationConfig, subject_level_r
from eegtable.model.scoring import safe_pearsonr

__all__ = [
    "ClassificationResult",
    "classification_metrics",
    "regression_metrics",
    "within_condition_metrics",
    "within_subject_centered_metrics",
]

_DEFAULT_AGGREGATION_CONFIG = AggregationConfig()


@dataclass(frozen=True)
class ClassificationResult:
    """Scores from :func:`classification_metrics`.

    With ``groups``, every scalar score is the equal-weight mean over the subjects
    for which it is defined, while ``confusion`` stays pooled over trials, so
    accuracy recomputed from ``confusion`` is the pooled value, not ``accuracy``.

    Parameters
    ----------
    y_true, y_pred : ndarray of int
        The labels scored.
    y_prob, y_score : ndarray or None
        The probabilities and decision scores scored, if any.
    groups : ndarray or None
        Subject labels, or None for pooled scores.
    accuracy : float
        Fraction of trials classified correctly.
    balanced_accuracy : float
        Mean recall of the two classes; NaN, and left out of subject means, for a
        subject with one class.
    auc, average_precision : float
        Ranking scores from ``y_score``, else from the class-1 probability. NaN
        without either, or with a single class.
    f1, precision, recall : float
        For class 1. NaN where undefined, e.g. recall without positive trials.
    specificity : float
        Recall of class 0; NaN without negative trials.
    confusion : ndarray of int, shape (2, 2)
        Trial counts, rows true and columns predicted, in label order ``(0, 1)``.
    per_subject : mapping of str to mapping of str to float
        Each subject's scalar scores; empty without ``groups``.
    mean_subject_auc : float
        The subject-mean AUC, equal to ``auc`` with ``groups``; NaN without them.
    """

    # When `groups` is given, every scalar below is the mean over subjects, giving each subject
    # equal weight, while `confusion` stays pooled counts over all trials. The two therefore
    # disagree by design: accuracy recomputed from `confusion` is the trial-pooled value, not
    # `accuracy`. Report one or the other, not both as if they matched.
    y_true: npt.NDArray[np.intp]
    y_pred: npt.NDArray[np.intp]
    y_prob: npt.NDArray[np.float64] | None
    y_score: npt.NDArray[np.float64] | None
    groups: npt.NDArray[np.object_] | None
    accuracy: float
    balanced_accuracy: float
    auc: float
    average_precision: float
    f1: float
    precision: float
    recall: float
    specificity: float
    confusion: npt.NDArray[np.intp]
    per_subject: Mapping[str, Mapping[str, float]]
    mean_subject_auc: float


def _check_aligned(label: str, *arrays: npt.NDArray[np.generic]) -> None:
    if any(a.ndim != 1 for a in arrays) or len({a.shape for a in arrays}) != 1:
        raise ValueError(f"{label} metrics require aligned 1-D arrays.")


def _check_finite_predictions(label: str, *arrays: npt.NDArray[np.float64]) -> None:
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError(f"{label} metrics require finite targets and predictions for every trial.")


def _check_labelled(label: str, **named: npt.NDArray[np.object_]) -> None:
    # A trial with no subject or condition has no cell to be centred within, and inventing one
    # would change the statistic. Refuse it rather than dropping it from the denominator.
    for name, values in named.items():
        missing = int(np.sum(pd.isna(values)))
        if missing:
            raise ValueError(
                f"{label} metrics require a {name} label for every trial; "
                f"{missing} of {len(values)} have none."
            )


def _variance_scores(
    target: npt.NDArray[np.float64], prediction: npt.NDArray[np.float64]
) -> tuple[float, float]:
    # A constant decimal can acquire a nonzero centered norm through rounding.
    if np.all(target == target[0]):
        residual = target - prediction
        r2 = np.nan if np.array_equal(target, prediction) else -np.inf
        explained = np.nan if np.all(residual == residual[0]) else -np.inf
        return r2, explained
    return (
        float(r2_score(target, prediction, force_finite=False)),
        float(explained_variance_score(target, prediction, force_finite=False)),
    )


def _subset_classification_metrics(
    y_true: npt.NDArray[np.intp],
    y_pred: npt.NDArray[np.intp],
    y_prob: npt.NDArray[np.float64] | None = None,
    y_score: npt.NDArray[np.float64] | None = None,
) -> dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else np.nan
    # 0/0 is undefined, not zero: a held-out subject with no positive trials has no
    # recall, and one with no positive predictions has no precision. NaN lets the
    # subject mean drop them, as it already does for balanced accuracy and AUC;
    # zero would pull every subject-level mean toward zero by the share of such
    # subjects, and F1 would then move against balanced accuracy.
    f1 = float(f1_score(y_true, y_pred, zero_division=np.nan)) if len(y_true) > 0 else np.nan
    prec = (
        float(precision_score(y_true, y_pred, zero_division=np.nan)) if len(y_true) > 0 else np.nan
    )
    rec = float(recall_score(y_true, y_pred, zero_division=np.nan)) if len(y_true) > 0 else np.nan
    b_acc = (
        float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) >= 2 else np.nan
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, _fn, _tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan

    auc = np.nan
    ap = np.nan
    scores = y_score
    if scores is None and y_prob is not None:
        scores = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    if scores is not None and len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, scores))
        ap = float(average_precision_score(y_true, scores))

    return {
        "accuracy": acc,
        "balanced_accuracy": b_acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "auc": auc,
        "average_precision": ap,
    }


def classification_metrics(
    y_true: npt.NDArray[np.intp],
    y_pred: npt.NDArray[np.intp],
    *,
    y_prob: npt.NDArray[np.float64] | None = None,
    y_score: npt.NDArray[np.float64] | None = None,
    groups: npt.NDArray[np.object_] | None = None,
) -> ClassificationResult:
    """Binary classification scores, pooled over trials or averaged over subjects.

    Parameters
    ----------
    y_true, y_pred : ndarray of int, shape (n_trials,)
        Labels and predictions coded 0 and 1; any other label raises.
    y_prob : ndarray, optional
        Real probabilities in ``[0, 1]``, shape ``(n_trials, 2)`` in class order
        ``(0, 1)`` with each row summing to one, or ``(n_trials,)`` for class 1.
    y_score : ndarray, optional
        Finite decision scores, shape ``(n_trials,)``, larger for class 1. AUC and
        average precision use them in preference to ``y_prob``.
    groups : ndarray, optional
        Subject label of each trial; every trial needs one. Each scalar is then
        the equal-weight mean over the subjects for which it is defined.

    Returns
    -------
    ClassificationResult
    """
    # The confusion matrix, specificity and the positive class of precision and recall all
    # assume 0/1 coding, so any other coding is refused instead of scored against it.
    labels = np.unique(np.concatenate([np.ravel(y_true), np.ravel(y_pred)]))
    if not set(labels.tolist()) <= {0, 1}:
        msg = f"classification_metrics expects labels coded 0/1, got {labels.tolist()}."
        raise ValueError(msg)
    y_t = np.asarray(as_real_array(y_true, "y_true"), dtype=np.intp)
    y_p = np.asarray(as_real_array(y_pred, "y_pred"), dtype=np.intp)
    _check_aligned("Classification", y_t, y_p)
    if y_prob is not None:
        y_prob = as_real_array(y_prob, "y_prob")
        if y_prob.shape not in ((len(y_t),), (len(y_t), 2)):
            raise ValueError("y_prob must have shape (n_trials,) or (n_trials, 2).")
        if not np.isfinite(y_prob).all():
            raise ValueError("y_prob must be finite for every trial; no trials may be dropped.")
        if np.any((y_prob < 0.0) | (y_prob > 1.0)):
            raise ValueError("y_prob probabilities must be in [0, 1].")
        if y_prob.ndim == 2 and not np.allclose(y_prob.sum(axis=1), 1.0):
            raise ValueError("y_prob class probabilities must sum to one for every trial.")
    if y_score is not None:
        y_score = as_real_array(y_score, "y_score")
        if y_score.shape != (len(y_t),) or not np.isfinite(y_score).all():
            raise ValueError("y_score must be finite with shape (n_trials,).")
    cm = confusion_matrix(y_t, y_p, labels=[0, 1]).astype(np.intp)

    if groups is not None:
        groups_arr = np.asarray(groups)
        _check_aligned("Classification", y_t, groups_arr)
        _check_labelled("Classification", subject=groups_arr)
        per_subject: dict[str, Mapping[str, float]] = {}
        for subj in np.unique(groups_arr):
            mask = groups_arr == subj
            sub_prob = y_prob[mask] if y_prob is not None else None
            sub_score = y_score[mask] if y_score is not None else None
            per_subject[str(subj)] = _subset_classification_metrics(
                y_t[mask], y_p[mask], sub_prob, sub_score
            )

        def _mean_metric(key: str) -> float:
            vals = [m[key] for m in per_subject.values() if key in m and np.isfinite(m[key])]
            return float(np.mean(vals)) if vals else np.nan

        mean_auc = _mean_metric("auc")
        return ClassificationResult(
            y_true=y_t,
            y_pred=y_p,
            y_prob=y_prob,
            y_score=y_score,
            groups=groups_arr,
            accuracy=_mean_metric("accuracy"),
            balanced_accuracy=_mean_metric("balanced_accuracy"),
            auc=mean_auc,
            average_precision=_mean_metric("average_precision"),
            f1=_mean_metric("f1"),
            precision=_mean_metric("precision"),
            recall=_mean_metric("recall"),
            specificity=_mean_metric("specificity"),
            confusion=cm,
            per_subject=per_subject,
            mean_subject_auc=mean_auc,
        )

    global_m = _subset_classification_metrics(y_t, y_p, y_prob, y_score)
    return ClassificationResult(
        y_true=y_t,
        y_pred=y_p,
        y_prob=y_prob,
        y_score=y_score,
        groups=None,
        accuracy=global_m["accuracy"],
        balanced_accuracy=global_m["balanced_accuracy"],
        auc=global_m["auc"],
        average_precision=global_m["average_precision"],
        f1=global_m["f1"],
        precision=global_m["precision"],
        recall=global_m["recall"],
        specificity=global_m["specificity"],
        confusion=cm,
        per_subject={},
        mean_subject_auc=np.nan,
    )


def regression_metrics(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_] | None = None,
    *,
    config: AggregationConfig = _DEFAULT_AGGREGATION_CONFIG,
    folds: npt.NDArray[np.intp] | None = None,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Pooled regression scores and, with ``groups``, the subject-level correlation.

    Parameters
    ----------
    y_true, y_pred : ndarray, shape (n_trials,)
        Aligned targets and predictions, finite on every trial.
    groups : ndarray, optional
        Subject label of each trial; every trial needs one. Adds the Fisher-z mean
        of the within-subject correlations (:func:`subject_level_r`), which raises
        for a subject with fewer than three trials or an undefined correlation.
    config : AggregationConfig, optional
        Subject weighting of that mean. Intervals are not reported here; call
        :func:`subject_level_r` for them.
    folds : ndarray, optional
        Outer fold of each trial, as :func:`fold_results` returns them. Targets and
        predictions are then centred within each subject and fold before
        correlating, so differing fold-model offsets do not drive the score. Pass
        them whenever a subject is scored by several folds.

    Returns
    -------
    summary : dict of str to float
        ``pearson_r``, ``r2`` and ``explained_variance`` over all trials, ``n``,
        and ``subject_level_r``, also under the alias ``avg_subject_r_fisher_z``,
        NaN without ``groups``. A constant target leaves ``r2`` and
        ``explained_variance`` NaN or ``-inf``. Fewer than two trials give NaN
        scores.
    per_subject : list of dict
        ``{"subject": ..., "r": ...}`` for each subject; empty without ``groups``.
    """
    yt = as_real_array(y_true, "y_true")
    yp = as_real_array(y_pred, "y_pred")
    _check_aligned("Regression", yt, yp)
    _check_finite_predictions("Regression", yt, yp)
    if groups is not None:
        groups = np.asarray(groups)
        _check_aligned("Regression", yt, groups)
        _check_labelled("Regression", subject=groups)
    if folds is not None:
        folds = np.asarray(folds)
        _check_aligned("Regression", yt, folds)

    if len(yt) < 2:
        return {
            "pearson_r": np.nan,
            "subject_level_r": np.nan,
            "avg_subject_r_fisher_z": np.nan,
            "r2": np.nan,
            "explained_variance": np.nan,
            "n": float(len(yt)),
        }, []

    r_val, _ = safe_pearsonr(yt, yp)
    r2_val, ev_val = _variance_scores(yt, yp)

    summary: dict[str, float] = {
        "pearson_r": r_val,
        "r2": r2_val,
        "explained_variance": ev_val,
        "n": float(len(yt)),
        "subject_level_r": np.nan,
        "avg_subject_r_fisher_z": np.nan,
    }
    per_subject_list: list[dict[str, object]] = []

    if groups is not None:
        pred_df = pd.DataFrame({"subject_id": groups, "y_true": yt, "y_pred": yp})
        if folds is not None:
            # fold_results returns these ids; without them a subject scored by several fold
            # models (within-subject CV) has its r biased by the models' differing offsets.
            pred_df["fold"] = folds
        subj_r = subject_level_r(pred_df, config=config)
        # Two names for one number, kept because both are in use. subject_level_r is always the
        # Fisher-z mean, under equal and trial-count weighting alike, so they cannot diverge.
        # They are one result, not two: reporting both would double-count it.
        summary["subject_level_r"] = subj_r.r
        summary["avg_subject_r_fisher_z"] = subj_r.r
        for s, r in subj_r.per_subject:
            per_subject_list.append({"subject": s, "r": r})

    return summary, per_subject_list


def within_subject_centered_metrics(
    target: npt.NDArray[np.float64],
    full_prediction: npt.NDArray[np.float64],
    nuisance_prediction: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_],
) -> dict[str, float]:
    """Within-subject R² of a full and a nuisance-only prediction, and their difference.

    The target and both predictions are centred within each subject, so the scores
    reflect trial-level tracking rather than subject offsets. Each subject's
    ``1 - SS_res / SS_tot`` is averaged with equal weight; a subject with fewer than
    two trials or a constant target is skipped.

    Parameters
    ----------
    target : ndarray, shape (n_trials,)
        Finite target.
    full_prediction : ndarray, shape (n_trials,)
        Finite predictions of the full model.
    nuisance_prediction : ndarray, shape (n_trials,)
        Finite predictions of the nuisance-only model for the same trials.
    groups : ndarray, shape (n_trials,)
        Subject label of each trial; every trial needs one.

    Returns
    -------
    dict of str to float
        ``within_subject_centered_full_r2``, ``within_subject_centered_nuisance_r2``
        and ``within_subject_centered_delta_r2``, full minus nuisance. All NaN when
        no subject qualifies.
    """
    grp = np.asarray(groups)
    t = as_real_array(target, "target")
    f = as_real_array(full_prediction, "full_prediction")
    n = as_real_array(nuisance_prediction, "nuisance_prediction")
    _check_aligned("Within-subject prediction", grp, t, f, n)
    _check_finite_predictions("Within-subject prediction", t, f, n)
    _check_labelled("Within-subject prediction", subject=grp)

    full_scores: list[float] = []
    nuis_scores: list[float] = []

    for subj in pd.unique(grp):
        mask = grp == subj
        t_sub = t[mask]
        f_sub = f[mask]
        n_sub = n[mask]
        if len(t_sub) < 2 or np.all(t_sub == t_sub[0]):
            continue
        cent_t = t_sub - np.mean(t_sub)
        denom = float(cent_t @ cent_t)
        if denom == 0.0:
            continue
        cent_f = f_sub - np.mean(f_sub)
        cent_n = n_sub - np.mean(n_sub)
        res_f = cent_t - cent_f
        res_n = cent_t - cent_n
        full_scores.append(1.0 - float(res_f @ res_f) / denom)
        nuis_scores.append(1.0 - float(res_n @ res_n) / denom)

    if not full_scores:
        return {
            "within_subject_centered_full_r2": float("nan"),
            "within_subject_centered_nuisance_r2": float("nan"),
            "within_subject_centered_delta_r2": float("nan"),
        }

    full_r2 = float(np.mean(full_scores))
    nuis_r2 = float(np.mean(nuis_scores))
    return {
        "within_subject_centered_full_r2": full_r2,
        "within_subject_centered_nuisance_r2": nuis_r2,
        "within_subject_centered_delta_r2": full_r2 - nuis_r2,
    }


def _within_condition_cells(
    subject_mask: npt.NDArray[np.bool_],
    conditions: npt.NDArray[np.object_],
) -> list[npt.NDArray[np.intp]]:
    subject_rows = np.flatnonzero(subject_mask).astype(np.intp)
    cells: list[npt.NDArray[np.intp]] = []
    sub_conditions = conditions[subject_rows]
    for condition in np.unique(sub_conditions):
        cell_rows = subject_rows[sub_conditions == condition]
        if cell_rows.size >= 2:
            cells.append(cell_rows)
    return cells


def _center_within_cells(
    values: npt.NDArray[np.float64],
    cells: list[npt.NDArray[np.intp]],
) -> npt.NDArray[np.float64]:
    return np.concatenate([values[cell] - values[cell].mean() for cell in cells])


def within_condition_metrics(
    target: npt.NDArray[np.float64],
    full_prediction: npt.NDArray[np.float64],
    nuisance_prediction: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_],
    conditions: npt.NDArray[np.object_],
) -> dict[str, float]:
    """Within-condition R² of a full and a nuisance-only prediction, and their difference.

    As :func:`within_subject_centered_metrics`, but centring within each subject and
    condition cell, so condition means drive neither score. Cells with fewer than
    two trials are left out; a subject is skipped when the target is constant
    within every remaining cell.

    Parameters
    ----------
    target : ndarray, shape (n_trials,)
        Finite target.
    full_prediction : ndarray, shape (n_trials,)
        Finite predictions of the full model.
    nuisance_prediction : ndarray, shape (n_trials,)
        Finite predictions of the nuisance-only model for the same trials.
    groups : ndarray, shape (n_trials,)
        Subject label of each trial; every trial needs one.
    conditions : ndarray, shape (n_trials,)
        Condition label of each trial; every trial needs one.

    Returns
    -------
    dict of str to float
        ``within_condition_centered_full_r2``, ``..._nuisance_r2`` and
        ``..._delta_r2``, plus the numbers of subjects and trials scored
        (``..._n_subjects``, ``..._n_trials``). The scores are NaN when no subject
        qualifies.
    """
    grp = np.asarray(groups)
    t = as_real_array(target, "target")
    f = as_real_array(full_prediction, "full_prediction")
    n = as_real_array(nuisance_prediction, "nuisance_prediction")
    cond = np.asarray(conditions)

    _check_aligned("Within-condition prediction", grp, t, f, n, cond)
    _check_finite_predictions("Within-condition prediction", t, f, n)
    _check_labelled("Within-condition prediction", subject=grp, condition=cond)

    full_scores: list[float] = []
    nuis_scores: list[float] = []
    n_trials = 0

    for subj in pd.unique(grp):
        mask = grp == subj
        cells = _within_condition_cells(mask, cond)
        if not cells or all(np.all(t[cell] == t[cell[0]]) for cell in cells):
            continue
        centered_t = _center_within_cells(t, cells)
        denominator = float(centered_t @ centered_t)
        if denominator == 0.0:
            continue
        centered_f = _center_within_cells(f, cells)
        centered_n = _center_within_cells(n, cells)
        full_res = centered_t - centered_f
        nuis_res = centered_t - centered_n
        full_scores.append(1.0 - float(full_res @ full_res) / denominator)
        nuis_scores.append(1.0 - float(nuis_res @ nuis_res) / denominator)
        n_trials += int(sum(len(cell) for cell in cells))

    if not full_scores:
        return {
            "within_condition_centered_full_r2": float("nan"),
            "within_condition_centered_nuisance_r2": float("nan"),
            "within_condition_centered_delta_r2": float("nan"),
            "within_condition_centered_n_subjects": 0.0,
            "within_condition_centered_n_trials": 0.0,
        }

    full_r2 = float(np.mean(full_scores))
    nuis_r2 = float(np.mean(nuis_scores))
    return {
        "within_condition_centered_full_r2": full_r2,
        "within_condition_centered_nuisance_r2": nuis_r2,
        "within_condition_centered_delta_r2": full_r2 - nuis_r2,
        "within_condition_centered_n_subjects": float(len(full_scores)),
        "within_condition_centered_n_trials": float(n_trials),
    }
