from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from eegtable._validation import as_real_array
from eegtable.model import _deps as _deps
from eegtable.quality import QualityPolicy, _validate_descriptor_labels, apply_quality
from eegtable.table import FeatureMeta, FeatureTable, RowId

__all__ = [
    "Design",
    "Selection",
    "build_design",
    "compute_train_group_intersection_mask",
    "harmonize_fold",
    "select",
]


@dataclass(frozen=True)
class Selection:
    """Filter on feature metadata that chooses design columns.

    Each field lists accepted values of the :class:`~eegtable.FeatureMeta` field of
    the same name; an empty tuple places no restriction. A column is kept when it
    matches every restricted field, so values within a field are alternatives.

    Parameters
    ----------
    measure : tuple of str
        ``FeatureMeta.measure`` labels, such as ``"band_power"``. A label is not
        always the name of the function that produced the column.
    band : tuple of str
        Band names. A column without a band never matches a restricted band.
    space_kind : tuple of str
        Any of ``"channel"``, ``"roi"``, ``"global"``, ``"pair"`` and ``"state"``.
    window : tuple of str
        Window names. A column without a window never matches a restricted window.
    normalization : tuple of str
        Normalization labels, such as ``"log10"`` or ``"db"``.
    space : tuple of str
        Channel, ROI, pair or state names.
    exclude : Selection, optional
        Columns it matches are dropped from those the other fields keep. It must
        restrict at least one field.
    """

    measure: tuple[str, ...] = ()
    band: tuple[str, ...] = ()
    space_kind: tuple[str, ...] = ()
    window: tuple[str, ...] = ()
    normalization: tuple[str, ...] = ()
    space: tuple[str, ...] = ()
    # Columns matching this are dropped from those the fields above keep.
    exclude: Selection | None = None


@dataclass(frozen=True)
class Design:
    """Epoch features aligned with their targets, from :func:`build_design`.

    Rows follow the feature table's row order.

    Parameters
    ----------
    X : ndarray, shape (n_rows, n_columns)
        Feature columns followed by covariate columns. Missing feature values stay
        NaN for the training-fitted pipeline to handle; covariates are finite.
    y : ndarray, shape (n_rows,)
        Finite target.
    groups : ndarray of object, shape (n_rows,)
        Grouping label of each row, usually the subject.
    runs : ndarray of object or None
        Run label of each row, or None when no run column was named.
    row_ids : tuple of (str, int, str)
        ``(recording, epoch, event)`` identity of each row.
    column_names : tuple of str
        One name per column of ``X``: feature names, then covariate names.
    feature_columns : ndarray of int
        Positions of the feature columns in ``X``.
    covariate_columns : ndarray of int
        Positions of the covariate columns, which follow the features.
    coverage : ndarray, shape (n_rows, n_features)
        Coverage of the feature columns only.
    support : ndarray, shape (n_rows, n_features) or None
        Time-window support of the feature columns, or None for full support.
    flags : dict of str to ndarray of bool
        Per-cell flags of the feature columns, including ``quality_rejected`` when
        a quality policy was applied.
    meta : tuple of FeatureMeta
        Metadata of the feature columns; covariates have none.
    quality_ledger : DataFrame
        One row per cell the quality policy masked, with ``row``, ``feature``,
        ``reason`` and ``coverage``; empty without a policy.
    """

    X: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    groups: npt.NDArray[np.object_]
    runs: npt.NDArray[np.object_] | None
    row_ids: tuple[RowId, ...]
    column_names: tuple[str, ...]
    feature_columns: npt.NDArray[np.intp]
    covariate_columns: npt.NDArray[np.intp]
    coverage: npt.NDArray[np.float64]
    support: npt.NDArray[np.float64] | None
    flags: dict[str, npt.NDArray[np.bool_]]
    meta: tuple[FeatureMeta, ...]
    quality_ledger: pd.DataFrame

    @property
    def n_covariates(self) -> int:
        """Number of covariate columns, the ``n_covariates`` the pipeline factories take."""
        return int(self.covariate_columns.size)


_DEFAULT_SELECTION = Selection()


def _matches(meta: FeatureMeta, selection: Selection) -> bool:
    return (
        (not selection.measure or meta.measure in selection.measure)
        and (not selection.band or (meta.band is not None and meta.band.name in selection.band))
        and (not selection.space_kind or meta.space_kind in selection.space_kind)
        and (not selection.window or (meta.window is not None and meta.window in selection.window))
        and (not selection.normalization or meta.normalization in selection.normalization)
        and (not selection.space or meta.space in selection.space)
        and not (selection.exclude is not None and _matches(meta, selection.exclude))
    )


def select(table: FeatureTable, selection: Selection) -> FeatureTable:
    """Keep the columns of a table that a :class:`Selection` matches.

    Parameters
    ----------
    table : FeatureTable
        Epoch-row or trial-group-row table.
    selection : Selection
        Metadata filter. Raises when it matches no column, or when an ``exclude``
        at any depth restricts no field.

    Returns
    -------
    FeatureTable
        The matching columns with their coverage, flags and metadata. The input
        table itself is returned when every column matches.
    """
    exclusion = selection.exclude
    while exclusion is not None:
        # An exclusion that restricts nothing matches every column and would drop them all.
        if replace(exclusion, exclude=None) == _DEFAULT_SELECTION:
            raise ValueError(f"Selection.exclude restricts no field: {exclusion}.")
        exclusion = exclusion.exclude
    kept_indices = [i for i, meta in enumerate(table.meta) if _matches(meta, selection)]

    if not kept_indices:
        msg = f"Selection {selection} matched no columns in FeatureTable."
        raise ValueError(msg)

    if len(kept_indices) == len(table.meta):
        return table

    return table._columns(np.array(kept_indices, dtype=np.intp))


def _validate_precomputed_features(table: FeatureTable) -> None:
    """Reject learned features that must be fitted within predictive folds."""
    if any(
        meta.measure == "csp_log_power" or meta.computation.method == "csp_features"
        for meta in table.meta
    ):
        raise ValueError(
            "CSP must be fitted inside each training fold, including inner tuning; "
            "an assembled cross-fitted CSP table can leak test labels into training "
            "features even when the same folds are reused."
        )


def build_design(
    table: FeatureTable,
    targets: pd.DataFrame,
    *,
    target: str,
    groups: str = "subject_id",
    runs: str | None = None,
    covariates: Sequence[str] = (),
    selection: Selection = _DEFAULT_SELECTION,
    strict_covariates: bool = True,
    quality: QualityPolicy | None = None,
) -> Design:
    """Align an epoch feature table with a target frame.

    Rows are matched on ``(recording, epoch, event)``. Every table row needs exactly
    one target row and every target row one table row; the error names how many are
    left over on each side. To model a subset of epochs, cut the table to them
    with :meth:`~eegtable.FeatureTable.take` first.

    Parameters
    ----------
    table : FeatureTable
        Per-epoch features with ``row_ids``. Trial-group tables are refused (see
        :func:`eegtable.group.build_group_design`), as are precomputed CSP
        features, which must be fitted inside each training fold.
    targets : DataFrame
        One row per epoch with ``recording``, ``epoch`` and ``event`` columns, the
        target, the grouping column and any run or covariate columns.
    target : str
        Target column. Values are coerced to numbers and must all be finite.
    groups : str, default "subject_id"
        Column whose labels the outer folds keep disjoint. Labels must be
        nonmissing and nonempty.
    runs : str, optional
        Run label column, needed by within-subject folds and run-wise nulls.
        Labels must be nonmissing and nonempty.
    covariates : sequence of str
        Numeric columns appended to ``X`` after the features. Values must be
        finite; encode categorical covariates before calling. A name matching
        the target, ``"outcome"`` or ``"target"``, ignoring case, is refused as
        label leakage.
    selection : Selection, optional
        Feature columns to keep, applied before the quality policy. The default
        keeps every column.
    strict_covariates : bool, default True
        Raise when a requested covariate is absent from ``targets``. False drops
        absent covariates without notice.
    quality : QualityPolicy, optional
        Applied to the selected features; rejected cells become NaN and are
        listed in ``Design.quality_ledger``. Choose it before evaluating outcomes.

    Returns
    -------
    Design
    """
    if table.row_ids is None:
        msg = (
            "Modeling is per-epoch only; cross-trial/group-row FeatureTables "
            "have no row_ids and cannot be aligned to targets."
        )
        raise ValueError(msg)
    row_ids = table.row_ids

    if selection != _DEFAULT_SELECTION:
        table = select(table, selection)

    quality_ledger = pd.DataFrame(columns=["row", "feature", "reason", "coverage"])
    if quality is not None:
        quality_result = apply_quality(table, quality)
        table, quality_ledger = quality_result.table, quality_result.ledger

    _validate_precomputed_features(table)

    key_columns = ["recording", "epoch", "event"]
    for col in key_columns:
        if col not in targets.columns:
            msg = f"targets DataFrame missing required key column '{col}'."
            raise ValueError(msg)

    if target not in targets.columns:
        msg = f"Target column '{target}' missing from targets."
        raise ValueError(msg)

    if groups not in targets.columns:
        msg = f"Groups column '{groups}' missing from targets."
        raise ValueError(msg)

    if runs is not None and runs not in targets.columns:
        msg = f"Runs column '{runs}' missing from targets."
        raise ValueError(msg)
    _validate_descriptor_labels(targets, (groups, *((runs,) if runs is not None else ())))

    target_lower = target.strip().lower()
    leaking = [
        c for c in covariates if str(c).strip().lower() in {target_lower, "outcome", "target"}
    ]
    if leaking:
        msg = (
            "Covariates include the selected target, which would leak labels into predictors: "
            f"{leaking}. Target={target!r}."
        )
        raise ValueError(msg)

    missing_covs = [c for c in covariates if c not in targets.columns]
    if missing_covs:
        if strict_covariates:
            msg = f"Requested covariates missing from targets: {missing_covs}."
            raise ValueError(msg)
        active_covs = [c for c in covariates if c in targets.columns]
    else:
        active_covs = list(covariates)

    table_keys = list(row_ids)
    if len(set(table_keys)) != len(table_keys):
        msg = "FeatureTable contains duplicate row_ids; join must be one-to-one."
        raise ValueError(msg)

    target_keys = list(zip(targets["recording"], targets["epoch"], targets["event"], strict=True))
    if len(set(target_keys)) != len(target_keys):
        msg = "targets contains duplicate (recording, epoch, event) keys; join must be one-to-one."
        raise ValueError(msg)

    if set(table_keys) != set(target_keys):
        raise ValueError(_unmatched_rows(table_keys, target_keys))

    target_key_map = {k: i for i, k in enumerate(target_keys)}
    target_row_indices = [target_key_map[k] for k in table_keys]
    aligned_targets = targets.iloc[target_row_indices]

    raw = aligned_targets[target]
    coerced = pd.to_numeric(raw, errors="coerce")
    labels = raw[coerced.isna() & raw.notna()]
    if len(labels):
        # Coerced to NaN they would read as missing data, which they are not.
        examples = sorted({str(label) for label in labels})[:3]
        raise ValueError(
            f"Target column '{target}' holds text labels such as {examples}. Encode them as "
            "numbers first, for example 0 and 1 for a binary classification."
        )
    y = coerced.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(y)):
        msg = f"Target column '{target}' contains non-finite values."
        raise ValueError(msg)

    groups_arr = cast(npt.NDArray[np.object_], aligned_targets[groups].to_numpy(dtype=object))
    runs_arr = (
        cast(npt.NDArray[np.object_], aligned_targets[runs].to_numpy(dtype=object))
        if runs is not None
        else None
    )

    n_features = table.values.shape[1]
    feature_names = tuple(m.name for m in table.meta)

    if active_covs:
        cov_df = aligned_targets[active_covs].apply(pd.to_numeric, errors="coerce")
        cov_arr = cov_df.to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(cov_arr)):
            msg = "Covariates contain non-finite values."
            raise ValueError(msg)
        X = np.column_stack([table.values, cov_arr])
        column_names = feature_names + tuple(active_covs)
        feature_columns = np.arange(n_features, dtype=np.intp)
        covariate_columns = np.arange(n_features, n_features + len(active_covs), dtype=np.intp)
    else:
        X = table.values
        column_names = feature_names
        feature_columns = np.arange(n_features, dtype=np.intp)
        covariate_columns = np.empty(0, dtype=np.intp)

    return Design(
        X=X,
        y=y,
        groups=groups_arr,
        runs=runs_arr,
        row_ids=row_ids,
        column_names=column_names,
        feature_columns=feature_columns,
        covariate_columns=covariate_columns,
        coverage=table.coverage,
        flags=dict(table.flags),
        meta=table.meta,
        quality_ledger=quality_ledger,
        support=table.support,
    )


def _unmatched_rows(table_keys: Sequence[RowId], target_keys: Sequence[RowId]) -> str:
    def count(keys: list[RowId], row: str, other: str) -> str:
        one = len(keys) == 1
        return (
            f"{len(keys)} {row} row{'' if one else 's'} {'has' if one else 'have'} no {other} "
            f"row (first: {keys[0]})"
        )

    table_set, target_set = set(table_keys), set(target_keys)
    parts = [
        count(keys, row, other)
        for keys, row, other in (
            ([k for k in table_keys if k not in target_set], "feature", "target"),
            ([k for k in target_keys if k not in table_set], "target", "feature"),
        )
        if keys
    ]
    return (
        f"FeatureTable row_ids and targets keys do not match one-to-one: {'; '.join(parts)}. "
        "To model some of the rows, cut the table to them with FeatureTable.take."
    )


def compute_train_group_intersection_mask(
    X_train: npt.NDArray[np.float64],
    groups_train: npt.NDArray[np.object_] | Sequence[object],
) -> npt.NDArray[np.bool_]:
    """Columns with at least one finite value in every training group.

    Parameters
    ----------
    X_train : ndarray, shape (n_rows, n_features)
        Training rows only, so held-out rows do not decide which columns stay.
    groups_train : array-like, shape (n_rows,)
        Group label of each training row.

    Returns
    -------
    ndarray of bool, shape (n_features,)
        True for the columns to keep. Raises when no column qualifies.
    """
    X_arr = as_real_array(X_train, "X_train")
    groups_arr = np.asarray(groups_train)
    if X_arr.ndim != 2:
        msg = f"Expected 2D X_train, got shape={X_arr.shape}"
        raise ValueError(msg)
    if X_arr.shape[0] != len(groups_arr):
        msg = f"Length mismatch for X_train/groups_train: {X_arr.shape[0]} vs {len(groups_arr)}"
        raise ValueError(msg)
    n_features = X_arr.shape[1]
    if n_features == 0:
        return np.zeros(0, dtype=np.bool_)

    keep_mask = np.ones(n_features, dtype=np.bool_)
    unique_groups = np.unique(groups_arr)
    for grp in unique_groups:
        grp_mask = groups_arr == grp
        if not np.any(grp_mask):
            continue
        grp_has = np.any(np.isfinite(X_arr[grp_mask]), axis=0)
        keep_mask &= grp_has

    if not np.any(keep_mask):
        msg = "No features are finite for every training group."
        raise ValueError(msg)
    return keep_mask


def harmonize_fold(
    X_train: npt.NDArray[np.float64],
    X_test: npt.NDArray[np.float64],
    groups_train: npt.NDArray[np.object_] | Sequence[object],
    *,
    mode: str | None,
    n_covariates: int = 0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Apply one fold's column harmonization, decided on its training rows.

    Parameters
    ----------
    X_train, X_test : ndarray
        Training and held-out rows with the same columns.
    groups_train : array-like
        Group label of each training row.
    mode : {"intersection", "union_impute"} or None
        ``"intersection"`` keeps the feature columns that have at least one finite
        value in every training group (:func:`compute_train_group_intersection_mask`).
        ``"union_impute"``, also used for None, keeps every column and leaves
        missing values to the pipeline's imputer.
    n_covariates : int, default 0
        Trailing covariate columns, which are always kept.

    Returns
    -------
    X_train, X_test : ndarray
        The kept columns.
    keep : ndarray of bool
        Mask of the kept input columns. Raises when it would keep none.
    """
    mode_str = (mode or "union_impute").strip().lower()
    if mode_str not in ("intersection", "union_impute"):
        msg = f"Unknown harmonization mode: {mode!r}. Expected 'intersection' or 'union_impute'."
        raise ValueError(msg)
    Xtr = as_real_array(X_train, "X_train")
    Xte = as_real_array(X_test, "X_test")
    if Xtr.shape[1] != Xte.shape[1]:
        msg = f"X_train/X_test feature mismatch: {Xtr.shape[1]} vs {Xte.shape[1]}"
        raise ValueError(msg)

    if mode_str == "union_impute":
        keep = np.ones(Xtr.shape[1], dtype=np.bool_)
        return Xtr, Xte, keep

    Xtr_eeg = Xtr[:, :-n_covariates] if n_covariates > 0 else Xtr

    if Xtr_eeg.shape[1] > 0:
        keep_eeg = compute_train_group_intersection_mask(Xtr_eeg, np.asarray(groups_train))
    else:
        keep_eeg = np.zeros(0, dtype=np.bool_)

    keep = np.ones(Xtr.shape[1], dtype=np.bool_)
    if n_covariates > 0:
        keep[:-n_covariates] = keep_eeg
    else:
        keep = keep_eeg

    if keep.size == 0 or not np.any(keep):
        msg = "Fold-specific intersection harmonization removed all features."
        raise ValueError(msg)
    return Xtr[:, keep], Xte[:, keep], keep
