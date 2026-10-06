"""Balanced repeated-session reliability, with explicit sample identities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eegtable.quality import _validate_descriptor_labels
from eegtable.table import FeatureTable


def intraclass_reliability(
    table: FeatureTable,
    descriptors: pd.DataFrame,
    *,
    subject: str = "subject_id",
    session: str = "session",
) -> pd.DataFrame:
    """Compute single-measure ICC(2,1) agreement and ICC(3,1) consistency.

    Each row must be one subject/session estimate. Aggregate repeated epochs
    explicitly before calling this function. The two-way ANOVA formulas follow
    Shrout and Fleiss, as implemented by Pingouin's ``intraclass_corr``. Missing
    values, duplicate samples, unbalanced designs and constant features raise.

    Parameters
    ----------
    table : FeatureTable
        One row per subject and session; every value must be finite.
    descriptors : DataFrame
        Aligned one-to-one with the table rows.
    subject : str, default "subject_id"
        Descriptor column of subject labels; nonmissing and nonempty.
    session : str, default "session"
        Descriptor column of session labels. Every subject needs every session
        exactly once, with at least three subjects and two sessions.

    Returns
    -------
    DataFrame
        One row per feature: ``feature``, ``n_subjects``, ``n_sessions``,
        ``icc_absolute`` (ICC(2,1)) and ``icc_consistency`` (ICC(3,1)). Estimates
        can be negative; report them unclipped.
    """
    if len(descriptors) != table.n_rows:
        raise ValueError("Descriptors must align one-to-one with feature rows.")
    if subject == session or not {subject, session}.issubset(descriptors.columns):
        raise ValueError("Provide distinct subject and session descriptor columns.")
    keys = descriptors[[subject, session]]
    _validate_descriptor_labels(descriptors, (subject, session))
    if keys.duplicated().any():
        raise ValueError("Reliability found duplicate subject/session samples.")
    subjects, sessions = pd.unique(keys[subject]), pd.unique(keys[session])
    n, k = len(subjects), len(sessions)
    if len(keys) != n * k:
        raise ValueError("Reliability requires a complete balanced subject/session design.")
    if n < 3 or k < 2:
        raise ValueError("Reliability requires at least three subjects and two sessions.")
    if not np.isfinite(table.values).all():
        raise ValueError("Reliability requires finite values for every matched sample.")
    ordering = pd.MultiIndex.from_product([subjects, sessions])
    positions = pd.Series(np.arange(len(keys)), index=pd.MultiIndex.from_frame(keys))
    values = table.values[positions.loc[ordering].to_numpy()].reshape(n, k, -1)
    midpoint = values.min(axis=(0, 1)) / 2 + values.max(axis=(0, 1)) / 2
    values = values - midpoint
    scale = np.max(np.abs(values), axis=(0, 1))
    if np.any(scale == 0):
        raise ValueError("Reliability is undefined for constant or degenerate features.")
    # Center before rescaling to preserve small differences on large baselines.
    # The midpoint and normalized deviations avoid overflow in ANOVA squares.
    values = values / scale
    grand = values.mean(axis=(0, 1))
    subject_mean, session_mean = values.mean(axis=1), values.mean(axis=0)
    subject_ms = k * np.sum((subject_mean - grand) ** 2, axis=0) / (n - 1)
    session_ms = n * np.sum((session_mean - grand) ** 2, axis=0) / (k - 1)
    residual = values - subject_mean[:, None] - session_mean[None, :] + grand
    error_ms = np.sum(residual**2, axis=(0, 1)) / ((n - 1) * (k - 1))
    consistency_denominator = subject_ms + (k - 1) * error_ms
    absolute_denominator = consistency_denominator + k * (session_ms - error_ms) / n
    if np.any(consistency_denominator <= 0) or np.any(absolute_denominator <= 0):
        raise ValueError("Reliability is undefined for constant or degenerate features.")
    return pd.DataFrame(
        {
            "feature": table.names,
            "n_subjects": n,
            "n_sessions": k,
            "icc_absolute": (subject_ms - error_ms) / absolute_denominator,
            "icc_consistency": (subject_ms - error_ms) / consistency_denominator,
        }
    )
