from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from eegtable.model.scoring import safe_pearsonr

__all__ = [
    "AggregationConfig",
    "FoldResults",
    "SubjectLevelR",
    "bootstrap_mean_ci",
    "fold_results",
    "paired_signflip_p_value",
    "subject_level_errors",
    "subject_level_r",
    "subject_r_scorer",
]


@dataclass(frozen=True)
class AggregationConfig:
    """How per-subject scores are combined, and whether an interval is computed.

    Parameters
    ----------
    subject_weighting : {"equal", "trial_count"}, default "equal"
        ``"trial_count"`` weights Fisher-z correlations by ``n - 3`` and errors by
        ``n``, the subject's trial count; correlations then need more than three
        trials per subject.
    bootstrap_iterations : int, default 10000
        Subject resamples for ``ci_method="bootstrap"``.
    ci_method : {"none", "fixed_effects", "bootstrap"}, default "none"
        Interval over subjects. Any interval treats subject scores as independent,
        which scores held out from cross-subject folds are not: they share
        training data. Test those with :func:`permutation_test` instead.
        ``"fixed_effects"`` applies to correlations only.
    seed : int, default 42
        Bootstrap seed.
    """

    subject_weighting: Literal["equal", "trial_count"] = "equal"
    bootstrap_iterations: int = 10_000
    # An interval over subjects treats their scores as independent, which holds only when no
    # subject is scored by a model trained on another subject's data (within-subject folds).
    # Scores held out from cross-subject folds share training data and are positively
    # correlated: in a null simulation of leave-one-subject-out ridge, the t interval excluded
    # 0 in 13-19% of cohorts at a nominal 5%. So none is computed unless asked for, and those
    # scores are tested with permutation_test.
    ci_method: Literal["none", "fixed_effects", "bootstrap"] = "none"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.subject_weighting not in ("equal", "trial_count"):
            raise ValueError(
                f"subject_weighting must be 'equal' or 'trial_count', got {self.subject_weighting}"
            )
        if self.ci_method not in ("none", "fixed_effects", "bootstrap"):
            raise ValueError(
                "ci_method must be 'none', 'fixed_effects' or 'bootstrap', "
                f"got {self.ci_method!r}"
            )
        if self.bootstrap_iterations <= 0:
            raise ValueError(f"bootstrap_iterations must be > 0, got {self.bootstrap_iterations}")


_DEFAULT_CONFIG = AggregationConfig()


@dataclass(frozen=True)
class SubjectLevelR:
    """Result of :func:`subject_level_r`.

    Parameters
    ----------
    r : float
        Weighted mean of the subjects' Fisher-z correlations, transformed back to r.
    per_subject : tuple of (str, float)
        Each subject's correlation.
    ci_low, ci_high : float
        Interval bounds in r units; NaN unless ``AggregationConfig.ci_method``
        requests an interval.
    """

    r: float
    per_subject: tuple[tuple[str, float], ...]
    ci_low: float
    ci_high: float


def bootstrap_mean_ci(
    values: npt.NDArray[np.float64],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean of independent per-subject values.

    Resampling subjects treats them as independent, which scores held out from
    cross-subject folds are not (see :class:`AggregationConfig`); test those with
    :func:`~eegtable.model.permutation_test`.

    Parameters
    ----------
    values : ndarray, shape (n_subjects,)
        One finite value per subject, chosen before analysis. Pass Fisher-z values
        rather than correlations.
    iterations : int
        Bootstrap resamples.
    seed : int
        Resampling seed.

    Returns
    -------
    low, high : float
        2.5th and 97.5th percentiles of the resampled means. NaN for no subjects;
        the value itself for one.
    """
    vals = np.asarray(values, dtype=float)
    if vals.ndim != 1:
        raise ValueError("Bootstrap inference requires a 1-D vector of subject values.")
    if not np.isfinite(vals).all():
        raise ValueError("Bootstrap inference requires a finite value for every subject.")
    if vals.size == 0:
        return np.nan, np.nan
    if vals.size == 1:
        v = float(vals[0])
        return v, v

    rng = np.random.default_rng(seed)
    n = len(vals)
    boot_means = np.empty(iterations, dtype=float)
    for i in range(iterations):
        sample = rng.choice(vals, size=n, replace=True)
        boot_means[i] = float(np.mean(sample))
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def paired_signflip_p_value(
    differences: npt.NDArray[np.float64],
    *,
    iterations: int,
    seed: int,
) -> float:
    """Two-sided sign-flip p of a zero mean over independent per-subject differences.

    Finite-sample validity requires independent differences whose null distributions are
    symmetric about zero; zero mean alone is insufficient. Scores held out from
    cross-subject folds are not independent (see :class:`AggregationConfig`).

    Parameters
    ----------
    differences : ndarray, shape (n_subjects,)
        One finite paired difference per subject.
    iterations : int
        Random sign-flip draws.
    seed : int
        Seed of the draws.

    Returns
    -------
    float
        ``(b + 1) / (iterations + 1)``, where ``b`` counts draws whose absolute
        mean is at least the observed absolute mean. NaN for no subjects or no
        draws.
    """
    vals = np.asarray(differences, dtype=float)
    if vals.ndim != 1:
        raise ValueError("Sign-flip inference requires a 1-D vector of subject differences.")
    if not np.isfinite(vals).all():
        raise ValueError("Sign-flip inference requires a finite value for every subject.")
    if vals.size == 0 or iterations <= 0:
        return np.nan

    observed = float(abs(np.mean(vals)))
    threshold = observed - 100.0 * np.finfo(float).eps * observed
    rng = np.random.default_rng(seed)
    n = len(vals)
    count = 0
    for _ in range(iterations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n, replace=True)
        if float(abs(np.mean(vals * signs))) >= threshold:
            count += 1
    return float((count + 1) / (iterations + 1))


class FoldResults(NamedTuple):
    """Held-out predictions of every fold, concatenated in fold order.

    Parameters
    ----------
    y_true, y_pred : ndarray
        Targets and predictions.
    groups : list of str
        Group label of each prediction; empty when no record carried one.
    rows : list of int
        Design row of each prediction; empty when the records carry no rows.
    folds : list of int
        Fold index of each prediction, the ``folds`` that
        :func:`regression_metrics` takes.
    """

    y_true: npt.NDArray[np.float64]
    y_pred: npt.NDArray[np.float64]
    groups: list[str]
    rows: list[int]
    folds: list[int]


def fold_results(
    results: Sequence[object],
    groups: npt.NDArray[np.object_] | None = None,
) -> FoldResults:
    """Concatenate fold predictions in fold order and label them with their groups.

    Parameters
    ----------
    results : sequence of FoldPrediction, FoldClassification or dict
        Records with ``fold``, ``y_true`` and ``y_pred``, and optionally the test
        ``rows`` (``test_idx`` or ``rows`` for a dict) and a group label: a
        record's ``subject``, or a dict's ``groups``. Records are sorted by fold.
    groups : ndarray, optional
        The design's group labels, indexed by each record's rows. Pass them for
        group-disjoint folds, whose records carry no subject of their own.

    Returns
    -------
    FoldResults
        Raises when only some records carry group labels.
    """

    def _get_fold(r: object) -> int:
        if hasattr(r, "fold"):
            return int(getattr(r, "fold"))  # noqa: B009
        if isinstance(r, dict):
            return int(str(r["fold"]))
        raise TypeError(f"Unsupported fold result record type: {type(r)}")

    sorted_results = sorted(results, key=_get_fold)
    # The design's groups label rows by index; without them each record labels itself, and
    # a leave-one-subject-out prediction has no subject of its own to offer.
    group_source = None if groups is None else np.asarray(groups, dtype=object)

    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    groups_ordered: list[str] = []
    test_indices: list[int] = []
    fold_ids: list[int] = []
    labelled: list[bool] = []

    for record in sorted_results:
        fold = _get_fold(record)
        raw_groups: Sequence[object] | npt.NDArray[np.object_] | None
        if hasattr(record, "y_true") and hasattr(record, "y_pred"):
            raw_true = getattr(record, "y_true")  # noqa: B009
            raw_pred = getattr(record, "y_pred")  # noqa: B009
            raw_test = getattr(record, "rows", None)
            sub = getattr(record, "subject", None)
            raw_groups = [sub] * len(raw_true) if sub is not None else None
        elif isinstance(record, dict):
            raw_true = record["y_true"]
            raw_pred = record["y_pred"]
            raw_test = record.get("test_idx", record.get("rows"))
            raw_groups = record.get("groups")
        else:
            raise TypeError(f"Unsupported record type: {type(record)}")

        yt = [float(v) for v in raw_true]
        yp = [float(v) for v in raw_pred]
        if len(yt) != len(yp):
            msg = f"Fold {fold} has mismatched lengths: y_true ({len(yt)}) vs y_pred ({len(yp)})."
            raise ValueError(msg)

        if raw_test is not None:
            t_idx = [int(i) for i in raw_test]
            if len(t_idx) != len(yt):
                msg = (
                    f"Fold {fold} has mismatched test_idx length ({len(t_idx)}) "
                    f"vs y_true ({len(yt)})."
                )
                raise ValueError(msg)
        else:
            t_idx = []

        if group_source is not None and raw_test is not None:
            raw_groups = group_source[np.asarray(t_idx, dtype=np.intp)]
        if raw_groups is not None:
            g_arr = [str(g) for g in raw_groups]
            if len(g_arr) != len(yt):
                msg = (
                    f"Fold {fold} has mismatched groups length ({len(g_arr)}) "
                    f"vs y_true ({len(yt)})."
                )
                raise ValueError(msg)
            groups_ordered.extend(g_arr)
        labelled.append(raw_groups is not None)

        y_true_all.extend(yt)
        y_pred_all.extend(yp)
        test_indices.extend(t_idx)
        fold_ids.extend([fold] * len(yt))

    if any(labelled) and not all(labelled):
        msg = (
            "Some fold results carry subject labels and others do not, so the groups could "
            "not be aligned with y_true; pass groups= to label every row from the design."
        )
        raise ValueError(msg)

    return FoldResults(
        y_true=np.asarray(y_true_all, dtype=float),
        y_pred=np.asarray(y_pred_all, dtype=float),
        groups=groups_ordered,
        rows=test_indices,
        folds=fold_ids,
    )


_PREDICTION_COLUMNS = ("subject_id", "y_true", "y_pred")


def _validate_predictions(predictions: pd.DataFrame, function: str) -> None:
    missing = [column for column in _PREDICTION_COLUMNS if column not in predictions.columns]
    if missing:
        msg = (
            f"{function} needs one row per trial with columns "
            f"{list(_PREDICTION_COLUMNS)}; missing {missing}."
        )
        raise ValueError(msg)
    if predictions["subject_id"].isna().any():
        raise ValueError(f"{function} needs a subject_id label for every trial.")
    for column in ("y_true", "y_pred"):
        values = pd.to_numeric(predictions[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{function} needs finite targets and predictions for every trial.")


def subject_level_r(
    predictions: pd.DataFrame,
    *,
    config: AggregationConfig = _DEFAULT_CONFIG,
    undefined: Literal["raise", "zero"] = "raise",
) -> SubjectLevelR:
    """Correlate predictions with targets within each subject, then average.

    Parameters
    ----------
    predictions : DataFrame
        One row per trial, with a ``subject_id`` label and the ``y_true`` and
        ``y_pred`` values of that trial. An optional ``fold`` column names the
        outer fold that scored the trial; targets and predictions are then
        centred within each fold before correlating. Pass it whenever a subject
        can be scored by more than one fold, as in within-subject CV.
    config : AggregationConfig
        Subject weighting, confidence-interval method and bootstrap settings.
    undefined : {"raise", "zero"}
        What to do with a subject whose correlation is undefined because nothing
        varies. ``"raise"``, the default, refuses; ``"zero"`` scores it as no
        linear association.
    """
    _validate_predictions(predictions, "subject_level_r")
    if undefined not in ("raise", "zero"):
        raise ValueError(f"undefined must be 'raise' or 'zero', got {undefined!r}")
    has_folds = "fold" in predictions.columns
    if has_folds and predictions["fold"].isna().any():
        raise ValueError("subject_level_r needs a fold label for every trial when 'fold' is given.")
    per_subject: list[tuple[str, float]] = []
    valid_entries: list[tuple[float, int]] = []
    invalid_subjects: list[str] = []

    for subj, df_sub in predictions.groupby("subject_id", observed=True):
        yt = pd.to_numeric(df_sub["y_true"], errors="coerce").to_numpy(dtype=float)
        yp = pd.to_numeric(df_sub["y_pred"], errors="coerce").to_numpy(dtype=float)
        n_trials = len(yt)

        if n_trials < 3:
            invalid_subjects.append(f"{subj}: fewer than 3 finite predictions (got {n_trials})")
            continue

        if has_folds:
            # A subject scored by several fold models (within-subject CV) carries each model's
            # offset, and that offset -- the mean target of the other runs -- is
            # anti-correlated with the run it scores, which biases r below zero with no signal
            # at all. Centring within folds keeps only trial-level tracking; each extra fold
            # mean removed costs one degree of freedom.
            fold_labels = df_sub["fold"].to_numpy()
            yt = _center_within(yt, fold_labels)
            yp = _center_within(yp, fold_labels)
            n_folds = len(pd.unique(fold_labels))
            n_trials -= n_folds - 1
            # Two points left would correlate at exactly +1 or -1 whatever the data.
            if n_trials < 3:
                invalid_subjects.append(
                    f"{subj}: fewer than 3 trials left after centring within {n_folds} folds "
                    f"({len(yt)} trials)"
                )
                continue

        r, _ = safe_pearsonr(yt, yp)
        if undefined == "zero" and not np.isfinite(r):
            # Nothing varies to correlate with, which is no linear association.
            r = 0.0
        per_subject.append((str(subj), float(r)))
        if np.isfinite(r):
            valid_entries.append((float(r), n_trials))
        else:
            invalid_subjects.append(f"{subj}: non-finite correlation (degenerate subject)")

    if invalid_subjects:
        details = "; ".join(invalid_subjects)
        raise ValueError(f"Invalid subject-level correlation inputs: {details}")

    if config.ci_method == "bootstrap" and len(valid_entries) < 3:
        raise ValueError(
            f"Bootstrap confidence intervals need at least 3 subjects, got {len(valid_entries)}. "
            "Use ci_method='none'."
        )

    if not valid_entries:
        return SubjectLevelR(
            r=np.nan,
            per_subject=tuple(per_subject),
            ci_low=np.nan,
            ci_high=np.nan,
        )

    r_vals = np.array([r for r, _ in valid_entries], dtype=float)
    n_vals = np.array([n for _, n in valid_entries], dtype=int)

    # Correlations are averaged in Fisher z rather than in r because r is not additive.
    clipped_r = np.clip(r_vals, -0.999999, 0.999999)
    z_vals = np.arctanh(clipped_r)

    if config.subject_weighting == "trial_count":
        # var(z) = 1/(n-3), so a 3-trial subject has infinite variance and no weight. Flooring
        # it at 1.0 would give it the same weight as a 4-trial subject, fabricating precision
        # the data does not have and making the fixed-effects interval far too narrow.
        underpowered = int(np.sum(n_vals < 4))
        if underpowered:
            raise ValueError(
                f"Trial-count weighting needs more than 3 trials per subject for the Fisher-z "
                f"variance to be defined; {underpowered} of {len(n_vals)} subject(s) have 3 or "
                "fewer. Use subject_weighting='equal' or drop those subjects."
            )
        weights = n_vals - 3.0
    else:
        weights = np.ones_like(z_vals, dtype=float)

    sum_weights = float(np.sum(weights))
    norm_weights = weights / sum_weights
    mean_z = float(np.average(z_vals, weights=norm_weights))
    agg_r = float(np.tanh(mean_z))

    ci_low, ci_high = np.nan, np.nan
    if config.ci_method == "bootstrap":
        rng = np.random.default_rng(config.seed)
        n_sub = len(z_vals)
        boot_means = np.empty(config.bootstrap_iterations, dtype=float)
        for i in range(config.bootstrap_iterations):
            idx = rng.choice(n_sub, size=n_sub, replace=True)
            boot_z = z_vals[idx]
            if config.subject_weighting == "trial_count":
                boot_w = weights[idx]
                boot_means[i] = float(np.average(boot_z, weights=boot_w / np.sum(boot_w)))
            else:
                boot_means[i] = float(np.mean(boot_z))
        ci_low = float(np.tanh(np.percentile(boot_means, 2.5)))
        ci_high = float(np.tanh(np.percentile(boot_means, 97.5)))
    elif config.ci_method == "fixed_effects" and len(z_vals) > 1:
        if config.subject_weighting == "trial_count":
            # Known Fisher-z variance, so the normal quantile is the right multiplier.
            se = float(np.sqrt(1.0 / sum_weights))
            multiplier = 1.96
        else:
            # The between-subject SD is estimated from the subjects themselves, so the interval
            # needs Student's t. At 5 subjects the normal quantile gives a nominal 95% interval
            # that covers about 88%.
            se = float(np.std(z_vals, ddof=1) / np.sqrt(len(z_vals)))
            multiplier = float(stats.t.ppf(0.975, len(z_vals) - 1))
        if np.isfinite(se) and se > 0:
            delta = multiplier * se
            ci_low = float(np.tanh(mean_z - delta))
            ci_high = float(np.tanh(mean_z + delta))

    return SubjectLevelR(
        r=agg_r,
        per_subject=tuple(per_subject),
        ci_low=ci_low,
        ci_high=ci_high,
    )


@dataclass(frozen=True)
class _SubjectRScorer:
    config: AggregationConfig

    def __call__(
        self,
        estimator: object,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        groups: npt.NDArray[np.object_],
    ) -> float:
        y_pred = np.asarray(estimator.predict(X), dtype=np.float64)  # type: ignore[attr-defined]
        y_true = np.asarray(y, dtype=np.float64)
        frame = pd.DataFrame({"subject_id": groups, "y_true": y_true, "y_pred": y_pred})
        sizes = frame.groupby("subject_id", observed=True).size()
        if (sizes < 3).any():
            raise ValueError(
                f"Subject-level r needs at least 3 held-out trials per subject; "
                f"{sizes.idxmin()} has {sizes.min()}. Hold out more trials or pass scoring."
            )
        # A candidate that predicts a constant for a subject tracks nothing in it, so it
        # scores 0 there rather than aborting the search.
        config = replace(self.config, ci_method="none")
        return subject_level_r(frame, config=config, undefined="zero").r


def subject_r_scorer(config: AggregationConfig = _DEFAULT_CONFIG) -> _SubjectRScorer:
    """Score held-out predictions by their subject-level r, the statistic regression reports.

    Cross-fitting selects regression hyperparameters with it by default. It needs the subject
    of every validation row, so it works only through eegtable's own tuning loop.

    Parameters
    ----------
    config : AggregationConfig, optional
        Subject weighting; interval settings are ignored.

    Returns
    -------
    callable
        Called as ``scorer(estimator, X, y, groups)``. Raises when a validation
        subject has fewer than three trials; a subject whose correlation is
        undefined, as with constant predictions, scores 0.
    """
    return _SubjectRScorer(config)


def _center_within(
    values: npt.NDArray[np.float64], labels: npt.NDArray[np.generic]
) -> npt.NDArray[np.float64]:
    centered = values.copy()
    for label in pd.unique(labels):
        cell = labels == label
        # Shift first so a constant decimal remains exactly zero in every fold.
        shifted = values[cell] - values[cell][0]
        centered[cell] = shifted - shifted.mean(axis=0)
    return centered


def _weighted_mean(values: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]) -> float:
    return float(np.average(values, weights=weights / np.sum(weights)))


def _weighted_bootstrap_ci(
    values: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    # Subjects are resampled, carrying their weights, so the interval reflects the same
    # weighting as the point estimate.
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(iterations, dtype=float)
    for i in range(iterations):
        idx = rng.choice(n, size=n, replace=True)
        means[i] = _weighted_mean(values[idx], weights[idx])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def subject_level_errors(
    predictions: pd.DataFrame,
    *,
    config: AggregationConfig = _DEFAULT_CONFIG,
) -> dict[str, float]:
    """Mean absolute and root-mean-square error per subject, then averaged.

    Parameters
    ----------
    predictions : DataFrame
        One row per trial, with a ``subject_id`` label and the ``y_true`` and
        ``y_pred`` values of that trial.
    config : AggregationConfig
        Subject weighting and optional bootstrap intervals, which require at
        least three subjects. ``ci_method="fixed_effects"`` applies only to
        correlations and is rejected here.
    """
    _validate_predictions(predictions, "subject_level_errors")
    if config.ci_method == "fixed_effects":
        raise ValueError(
            "subject_level_errors does not support ci_method='fixed_effects'; "
            "use ci_method='none' or 'bootstrap'."
        )
    per_subject_mae: list[float] = []
    per_subject_rmse: list[float] = []
    per_subject_n: list[int] = []
    invalid_subjects: list[str] = []

    for subj, df_sub in predictions.groupby("subject_id", observed=True):
        yt = pd.to_numeric(df_sub["y_true"], errors="coerce").to_numpy(dtype=float)
        yp = pd.to_numeric(df_sub["y_pred"], errors="coerce").to_numpy(dtype=float)
        n_trials = len(yt)
        if n_trials < 1:
            invalid_subjects.append(f"{subj}: no finite predictions")
            continue
        err = yp - yt
        per_subject_mae.append(float(np.mean(np.abs(err))))
        per_subject_rmse.append(float(np.sqrt(np.mean(err**2))))
        per_subject_n.append(n_trials)

    if invalid_subjects:
        details = "; ".join(invalid_subjects)
        raise ValueError(f"Invalid subject-level error inputs: {details}")

    if config.ci_method == "bootstrap" and len(per_subject_mae) < 3:
        raise ValueError(
            f"Bootstrap confidence intervals need at least 3 subjects, got {len(per_subject_mae)}. "
            "Use ci_method='none'."
        )

    if not per_subject_mae:
        return {
            "mean_mae": np.nan,
            "mean_rmse": np.nan,
            "ci_low_mae": np.nan,
            "ci_high_mae": np.nan,
            "ci_low_rmse": np.nan,
            "ci_high_rmse": np.nan,
        }

    maes = np.array(per_subject_mae, dtype=float)
    rmses = np.array(per_subject_rmse, dtype=float)
    # One config must not weight the correlation by trial count while leaving the errors
    # equal-weighted; subject_level_r honours this field, so these have to as well.
    counts = np.array(per_subject_n, dtype=float)
    weights = counts if config.subject_weighting == "trial_count" else np.ones_like(counts)

    mean_mae = _weighted_mean(maes, weights)
    mean_rmse = _weighted_mean(rmses, weights)

    ci_low_mae, ci_high_mae = np.nan, np.nan
    ci_low_rmse, ci_high_rmse = np.nan, np.nan

    if config.ci_method == "bootstrap":
        ci_low_mae, ci_high_mae = _weighted_bootstrap_ci(
            maes, weights, iterations=config.bootstrap_iterations, seed=config.seed
        )
        ci_low_rmse, ci_high_rmse = _weighted_bootstrap_ci(
            rmses, weights, iterations=config.bootstrap_iterations, seed=config.seed
        )

    return {
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "ci_low_mae": ci_low_mae,
        "ci_high_mae": ci_high_mae,
        "ci_low_rmse": ci_low_rmse,
        "ci_high_rmse": ci_high_rmse,
    }
