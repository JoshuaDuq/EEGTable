from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.pipeline import Pipeline

from eegtable.model import _ridge_null
from eegtable.model.aggregate import AggregationConfig, subject_level_r
from eegtable.model.crossfit import (
    FoldPrediction,
    _cross_fit_engine,
    _default_scoring,
    _validate_and_resolve_inner_groups,
)
from eegtable.model.residualize import residualize_targets, residualize_within_subjects
from eegtable.model.splits import Fold, InnerSplit
from eegtable.model.tuning import FoldFitError

__all__ = [
    "NullConfig",
    "NullResult",
    "Scheme",
    "changed_fraction",
    "circular_shift_group",
    "is_permutation_valid_run",
    "permutation_test",
    "permute",
]

# "run_wise" is an alias for "within_subject_within_run", kept because the upstream pipeline
# calls this shuffle "runwise". Both shuffle labels within each run of each subject; neither
# exchanges whole runs. Run structure is paradigm-specific, so a run-block exchange is not
# offered rather than guessed at. The alias is pinned by a test.
Scheme = Literal[
    "within_subject",
    "run_wise",
    "within_subject_within_run",
    "circular_shift_within_run",
]


@dataclass(frozen=True)
class NullConfig:
    """Rearrangement scheme and number of draws for :func:`permutation_test`.

    Parameters
    ----------
    scheme : str, default "within_subject"
        ``"within_subject"`` shuffles targets among each subject's trials.
        ``"within_subject_within_run"`` shuffles within each run of each subject;
        ``"run_wise"`` is an alias of it. No scheme exchanges whole runs.
        ``"circular_shift_within_run"`` rotates each run's trials, ordered by trial
        index, by a random shift. A scheme states a rearrangement; whether the
        targets are exchangeable under it follows from the study design, not from
        the choice.
    n_permutations : int, default 1000
        Number of draws.
    min_retained_trials : int, default 8
        Fewest trials a run may have under circular shifts.
    """

    scheme: Scheme = "within_subject"
    n_permutations: int = 1000
    min_retained_trials: int = 8

    def __post_init__(self) -> None:
        valid: tuple[str, ...] = (
            "within_subject",
            "run_wise",
            "within_subject_within_run",
            "circular_shift_within_run",
        )
        if self.scheme not in valid:
            msg = f"Unknown scheme {self.scheme!r}. Expected one of: {valid}."
            raise ValueError(msg)
        if self.n_permutations <= 0:
            msg = f"n_permutations must be > 0, got {self.n_permutations}."
            raise ValueError(msg)
        if self.min_retained_trials < 1:
            msg = f"min_retained_trials must be >= 1, got {self.min_retained_trials}."
            raise ValueError(msg)


@dataclass(frozen=True)
class NullResult:
    """Outcome of :func:`permutation_test`.

    Parameters
    ----------
    p_value : float
        ``(b + 1) / (B + 1)`` for ``b`` of ``B`` null statistics at least as
        extreme as ``observed``, in the direction ``greater_is_better`` names.
    observed : float
        The supplied observed statistic, confirmed by recomputation.
    null : ndarray, shape (n_permutations,)
        Statistic of each refitted draw.
    changed_fractions : ndarray, shape (n_permutations,)
        Fraction of trials whose target value each draw's rearrangement changes.
    n_incomplete : int
        Draws that did not complete; always 0, because a failed draw raises.
    """

    p_value: float
    observed: float
    null: npt.NDArray[np.float64]
    changed_fractions: npt.NDArray[np.float64]
    n_incomplete: int


_DEFAULT_AGGREGATION = AggregationConfig()


def is_permutation_valid_run(retained: npt.NDArray[np.intp], min_retained: int = 8) -> bool:
    """Whether a run retains at least ``min_retained`` trials, all with finite indices."""
    # Cast to float, not int: an int cast turns NaN into a large negative integer, which then
    # passes any finiteness test and reports a run of missing counts as usable.
    arr = np.asarray(retained, dtype=np.float64)
    return bool(arr.size >= min_retained and np.all(np.isfinite(arr)))


def circular_shift_group(n_retained: int) -> tuple[int, ...]:
    """Shifts a circular draw chooses from for a run of ``n_retained`` trials.

    ``0`` to ``n_retained - 1``, so the identity shift is included; empty for no
    trials.
    """
    if n_retained <= 0:
        return ()
    return tuple(range(n_retained))


def changed_fraction(
    y_original: npt.NDArray[np.float64],
    y_permuted: npt.NDArray[np.float64],
) -> float:
    """Fraction of positions whose value differs between two aligned arrays.

    Positions where either value is non-finite are ignored; 0.0 when none remain.
    Arrays of different shapes raise.
    """
    orig = np.asarray(y_original, dtype=np.float64)
    perm = np.asarray(y_permuted, dtype=np.float64)
    if orig.shape != perm.shape:
        msg = f"y_original and y_permuted shape mismatch: {orig.shape} vs {perm.shape}."
        raise ValueError(msg)
    if len(orig) == 0:
        return 0.0
    finite = np.isfinite(orig) & np.isfinite(perm)
    if np.sum(finite) == 0:
        return 0.0
    return float(np.mean(orig[finite] != perm[finite]))


def permute(
    y: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_],
    runs: npt.NDArray[np.object_] | None = None,
    trial_indices: npt.NDArray[np.intp] | None = None,
    *,
    config: NullConfig,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Draw one rearrangement of ``y`` under a null scheme.

    Parameters
    ----------
    y : ndarray, shape (n_trials,)
        Values to rearrange.
    groups : ndarray, shape (n_trials,)
        Subject label of each trial. Every trial needs one, and every subject at
        least two trials: a lone trial would keep its value in every draw.
    runs : ndarray, optional
        Run label of each trial, required by the run-wise and circular schemes;
        every trial needs one.
    trial_indices : ndarray of int, optional
        Order of the trials within their run, required by circular shifts; finite
        integers, unique within each run.
    config : NullConfig
        Scheme and, for circular shifts, the shortest run allowed.
    rng : numpy.random.Generator
        Source of the draw.

    Returns
    -------
    ndarray
        ``y`` rearranged within the scheme's blocks. A run of one trial stays in
        place, and a circular shift can draw the identity.
    """
    values = np.asarray(y, dtype=np.float64)
    groups_arr = np.asarray(groups, dtype=object)
    if len(values) != len(groups_arr):
        msg = "Permutation values and groups must have the same length."
        raise ValueError(msg)

    source_indices = np.arange(len(values), dtype=np.intp)

    if config.scheme in ("run_wise", "within_subject_within_run", "circular_shift_within_run"):
        if runs is None:
            msg = f"Permutation scheme {config.scheme!r} requires run labels."
            raise ValueError(msg)
        runs_arr = np.asarray(runs)
        if len(runs_arr) != len(values):
            msg = (
                f"Permutation runs must have the same length as y when scheme is {config.scheme!r}."
            )
            raise ValueError(msg)
        n_unlabelled = int(np.sum(pd.isna(runs_arr)))
        if n_unlabelled:
            # Runs are paradigm-specific: a trial without a label has no run to be exchanged
            # within, and inventing one would change the hypothesis being tested.
            msg = (
                f"Permutation scheme {config.scheme!r} requires run labels for every trial; "
                f"{n_unlabelled} of {len(runs_arr)} have none."
            )
            raise ValueError(msg)
    else:
        runs_arr = None

    if config.scheme == "circular_shift_within_run":
        if trial_indices is None:
            msg = "circular_shift_within_run requires within-run trial indices."
            raise ValueError(msg)
        trial_arr = np.asarray(trial_indices, dtype=float)
        if len(trial_arr) != len(values):
            msg = (
                "Permutation trial indices must have the same length as y "
                "when scheme is 'circular_shift_within_run'."
            )
            raise ValueError(msg)
        if not np.all(np.isfinite(trial_arr)):
            msg = "circular_shift_within_run requires finite within-run trial indices."
            raise ValueError(msg)
        if trial_arr.ndim != 1 or np.any(trial_arr != np.floor(trial_arr)):
            raise ValueError("Circular shifts require 1-D integer trial indices.")
        trial_indices_arr = trial_arr.astype(np.intp)
    else:
        trial_indices_arr = None

    # A trial with no subject belongs to no exchangeable block, so it would keep its observed
    # target in every draw and pull the null toward the observed statistic. Refuse it, exactly
    # as a missing run label is refused above.
    n_unlabelled = int(np.sum(pd.isna(groups_arr)))
    if n_unlabelled:
        msg = (
            f"Permutation requires subject labels for every trial; "
            f"{n_unlabelled} of {len(groups_arr)} have none."
        )
        raise ValueError(msg)

    for subj in pd.unique(groups_arr):
        subj_mask = groups_arr == subj
        if np.sum(subj_mask) < 2:
            # Nothing to exchange this trial with, so it would carry its observed target into
            # every draw. Skipping the subject also skipped the per-run size checks below, so a
            # one-trial subject slipped past min_retained_trials that a seven-trial run fails.
            msg = (
                f"Subject {subj!r} has a single trial, which cannot be permuted and would keep "
                "its observed target in every draw; drop the subject before testing."
            )
            raise ValueError(msg)

        if config.scheme == "within_subject":
            subj_idx = np.where(subj_mask)[0]
            source_indices[subj_idx] = rng.permutation(source_indices[subj_idx])

        elif config.scheme in ("run_wise", "within_subject_within_run") and runs_arr is not None:
            # A subject with one run is still shuffled within it; skipping it would put its
            # real labels into every draw.
            subj_idx = np.flatnonzero(subj_mask)
            for r in pd.unique(runs_arr[subj_idx]):
                run_idx = subj_idx[runs_arr[subj_idx] == r]
                if len(run_idx) >= 2:
                    source_indices[run_idx] = rng.permutation(source_indices[run_idx])

        elif (
            config.scheme == "circular_shift_within_run"
            and runs_arr is not None
            and trial_indices_arr is not None
        ):
            # Missing run labels were refused above, so every run here is a real one.
            for r in pd.unique(runs_arr[subj_mask]):
                run_idx = np.where(subj_mask & (runs_arr == r))[0]
                if np.unique(trial_indices_arr[run_idx]).size != run_idx.size:
                    raise ValueError(
                        "Circular shifts require unique trial indices within each run."
                    )
                order = np.argsort(trial_indices_arr[run_idx], kind="stable")
                ordered_run_idx = run_idx[order]
                n_trials = len(ordered_run_idx)
                if n_trials < config.min_retained_trials:
                    msg = (
                        f"circular_shift_within_run requires runs with at "
                        f"least {config.min_retained_trials} retained trials, got {n_trials}."
                    )
                    raise ValueError(msg)
                shift_group = circular_shift_group(n_trials)
                shift = int(rng.choice(shift_group))
                source_indices[ordered_run_idx] = np.roll(source_indices[ordered_run_idx], shift)

    return values[source_indices]


def _prediction_statistic(
    predictions: Sequence[FoldPrediction],
    groups: npt.NDArray[np.object_],
    aggregation: AggregationConfig,
    metric_fn: (
        Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]],
            float,
        ]
        | None
    ),
) -> float:
    yt = np.concatenate([p.y_true for p in predictions])
    yp = np.concatenate([p.y_pred for p in predictions])

    if not np.all(np.isfinite(yt)) or not np.all(np.isfinite(yp)):
        raise ValueError("Permutation statistic requires finite predictions.")

    if metric_fn is not None:
        score = float(metric_fn(yt, yp))
    else:
        frame = pd.DataFrame(
            {
                "subject_id": np.concatenate([groups[p.rows] for p in predictions]),
                "fold": np.concatenate([np.full(p.rows.size, p.fold) for p in predictions]),
                "y_true": yt,
                "y_pred": yp,
            }
        )
        score = subject_level_r(
            frame,
            config=aggregation,
            undefined="zero",
        ).r

    if not np.isfinite(score):
        raise ValueError("Permutation statistic is undefined.")

    return score


def _freedman_lane_target(
    fold: Fold,
    *,
    y: npt.NDArray[np.float64],
    source: npt.NDArray[np.intp],
    covariates: npt.NDArray[np.float64] | None,
    groups: npt.NDArray[np.object_],
    columns: Sequence[str],
    within: Literal["subject"] | None,
) -> npt.NDArray[np.float64]:
    if covariates is None:
        raise ValueError(
            "Target residualization requested via residualize_on, but covariates is None."
        )
    rows = np.concatenate([fold.train, fold.test])
    in_fold = np.zeros(len(y), dtype=np.bool_)
    in_fold[rows] = True
    # The nuisance fit belongs to the fold, so its residuals can only be exchanged inside it.
    if not np.all(in_fold[source[rows]]):
        raise ValueError(
            f"Fold {fold.index}: the permutation moves trials outside the fold, whose nuisance "
            "fit cannot then be kept; use a scheme that exchanges trials within its runs."
        )
    residual = np.full(len(y), np.nan)
    if within is None:
        residual[fold.train], residual[fold.test] = residualize_targets(
            y, covariates, fold.train, fold.test, columns=columns
        )
    else:
        residual[fold.train], residual[fold.test] = residualize_within_subjects(
            y, covariates, groups, fold.train, fold.test, columns=columns
        )
    permuted = y.copy()
    permuted[rows] = y[rows] - residual[rows] + residual[source[rows]]
    return permuted


def permutation_test(
    folds: Sequence[Fold],
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_],
    runs: npt.NDArray[np.object_] | None,
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    observed: float,
    *,
    config: NullConfig,
    inner: InnerSplit,
    seed: int,
    outer_n_jobs: int = 1,
    harmonization: str | None = None,
    covariates: npt.NDArray[np.float64] | None = None,
    residualize_on: Sequence[str] = (),
    residualize_within: Literal["subject"] | None = None,
    scoring: object = None,
    refit: str | bool | None = None,
    metric_fn: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float] | None = None,
    greater_is_better: bool = True,
    trial_indices: npt.NDArray[np.intp] | None = None,
    aggregation: AggregationConfig = _DEFAULT_AGGREGATION,
) -> NullResult:
    """Refit the whole pipeline under permuted targets and compare the observed statistic.

    ``greater_is_better`` picks the tail, and it is not cosmetic. The default suits
    a correlation or an R^2, where a good model scores high. An error metric --
    ``mean_squared_error``, ``mean_absolute_error`` -- scores *low* when the model
    is good, so leaving the default in place counts the wrong tail and returns
    ``p`` near 1 for a strong effect and a small ``p`` for a worthless one. Pass
    ``greater_is_better=False`` for any metric where smaller is better. There is no
    way to infer the direction from an arbitrary callable, so it has to be declared.

    ``observed`` must come from the same folds, model, seed, aggregation and metric
    as this call; it is recomputed and a mismatch is an error rather than a silently
    invalid p-value.

    With ``residualize_on`` the null is Freedman-Lane (Freedman & Lane 1983; Winkler et
    al. 2014): each fold fits its nuisance model exactly as cross-fitting does, keeps
    that fit's prediction and permutes only its residuals, so a draw breaks the
    feature-target link and leaves the nuisance-target link in place. Permuting the raw
    target instead would break both, and the null would describe the wrong hypothesis.

    A draw that fails raises ``RuntimeError`` rather than being dropped from the null,
    and a scheme whose draws never change a target raises ``ValueError``.

    Parameters
    ----------
    folds, X, y, groups, pipeline, grid
        As for :func:`cross_fit_regression`, and the same as in the call that
        produced ``observed``.
    runs : ndarray or None
        Run label of each row; required by the run-wise and circular schemes and
        by within-subject folds.
    observed : float
        Finite statistic of the unpermuted fit; it is recomputed and must match.
    config : NullConfig
        Scheme and number of draws.
    inner : InnerSplit
        As for :func:`cross_fit_regression`.
    seed : int
        As for :func:`cross_fit_regression`; it also seeds the draws.
    outer_n_jobs, harmonization, covariates, residualize_within, scoring, refit
        As for :func:`cross_fit_regression`.
    residualize_on : sequence of str
        As for :func:`cross_fit_regression`. Every draw must then keep each
        fold's rows within that fold.
    metric_fn : callable, optional
        ``metric_fn(y_true, y_pred)`` over the held-out predictions of all folds,
        pooled. None uses subject-level ``r`` with targets and predictions centred
        within each fold, a subject whose correlation is undefined scoring 0.
    greater_is_better : bool, default True
        Tail of the test; see above.
    trial_indices : ndarray of int, optional
        Order of the trials within their run, required by
        ``"circular_shift_within_run"``.
    aggregation : AggregationConfig, optional
        Subject weighting of the default statistic; its interval is not computed.

    Returns
    -------
    NullResult
    """
    rng = np.random.default_rng(seed)
    groups_arr = np.asarray(groups, dtype=object)
    target = np.asarray(y, dtype=np.float64)
    # Each draw needs only the point estimate, so no interval is computed for it.
    null_aggregation = replace(aggregation, ci_method="none")
    # Drawn as source rows, in the random stream permuting y itself would use.
    sources: list[npt.NDArray[np.intp]] = []
    sampled_changed_fractions: list[float] = []

    for _ in range(config.n_permutations):
        rows = np.arange(len(target), dtype=np.float64)
        drawn = permute(rows, groups, runs, trial_indices, config=config, rng=rng)
        sources.append(drawn.astype(np.intp))
        sampled_changed_fractions.append(changed_fraction(target, target[sources[-1]]))

    changed_arr = np.asarray(sampled_changed_fractions, dtype=np.float64)
    if np.all(changed_arr == 0.0):
        msg = "Permutation scheme never changes any labels under this design."
        raise ValueError(msg)

    if not np.isfinite(observed):
        raise ValueError("observed statistic must be finite.")

    def fit_targets(
        source: npt.NDArray[np.intp] | None,
    ) -> tuple[FoldPrediction, ...]:
        fold_targets = None
        if source is not None and residualize_on:
            fold_targets = partial(
                _freedman_lane_target,
                y=target,
                source=source,
                covariates=covariates,
                groups=groups_arr,
                columns=residualize_on,
                within=residualize_within,
            )
        results = _cross_fit_engine(
            "regression",
            folds,
            X,
            target if source is None or residualize_on else target[source],
            groups,
            pipeline,
            grid,
            inner=inner,
            seed=seed,
            runs=runs,
            outer_n_jobs=outer_n_jobs,
            harmonization=harmonization,
            covariates=covariates,
            residualize_on=residualize_on,
            residualize_within=residualize_within,
            scoring=scoring,
            refit=refit,
            # The caller's own cross-fit already reported the grid; a null draw has no signal
            # to find, so its choices at the grid's ends are noise.
            warn_at_grid_edges=False,
            fold_targets=fold_targets,
        )
        return tuple(cast(list[FoldPrediction], results))

    observed_predictions = fit_targets(None)
    recomputed_observed = _prediction_statistic(
        observed_predictions,
        groups_arr,
        null_aggregation,
        metric_fn,
    )

    if not np.isclose(
        observed,
        recomputed_observed,
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError(
            "The supplied observed statistic differs from the statistic "
            "computed by this permutation procedure. Use the same folds, "
            "model, seed, aggregation, and metric for both."
        )

    selection = _default_scoring("regression", scoring)
    penalty = _ridge_null.ridge_penalty(pipeline, grid, selection, refit, metric_fn)
    if penalty is not None:
        try:
            null_arr = _ridge_null.ridge_null(
                folds,
                X,
                target,
                np.stack(sources),
                groups_arr,
                _validate_and_resolve_inner_groups(folds, inner, groups_arr, runs),
                pipeline,
                grid,
                penalty,
                inner=inner,
                seed=seed,
                scoring=selection,
                refit=refit,
                aggregation=null_aggregation,
                harmonization=harmonization,
                covariates=covariates,
                residualize_on=residualize_on,
                residualize_within=residualize_within,
                outer_n_jobs=outer_n_jobs,
            )
        except (FoldFitError, ValueError) as exc:
            raise RuntimeError(
                "The permutation draws failed. No p-value is returned because dropping "
                "a failed permutation could alter the null distribution."
            ) from exc
    else:
        null_scores: list[float] = []
        for b, source in enumerate(sources):
            try:
                predictions = fit_targets(source)
                score = _prediction_statistic(
                    predictions,
                    groups_arr,
                    null_aggregation,
                    metric_fn,
                )
            except (FoldFitError, ValueError) as exc:
                raise RuntimeError(
                    f"Permutation {b + 1} failed. No p-value is returned "
                    "because dropping a failed permutation could alter "
                    "the null distribution."
                ) from exc
            null_scores.append(score)
        null_arr = np.asarray(null_scores, dtype=np.float64)
    # Count numerical ties inclusively, as SciPy's permutation_test does: independently
    # computed statistics can round differently, including an identity permutation.
    tolerance = 100.0 * np.finfo(null_arr.dtype).eps * abs(observed)
    extreme = (
        null_arr >= observed - tolerance if greater_is_better else null_arr <= observed + tolerance
    )
    count_extreme = int(np.sum(extreme))
    p_value = float((count_extreme + 1) / (len(null_arr) + 1))

    return NullResult(
        p_value=p_value,
        observed=observed,
        null=null_arr,
        changed_fractions=changed_arr,
        n_incomplete=0,
    )
