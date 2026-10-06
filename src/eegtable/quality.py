"""Prespecified feature exclusions and summaries of the retained evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real

import numpy as np
import pandas as pd

from eegtable.table import FeatureTable


def _validate_descriptor_labels(descriptors: pd.DataFrame, columns: Sequence[str]) -> None:
    """Require complete scalar group identities while preserving numeric labels."""
    if len(set(descriptors.columns)) != len(descriptors.columns):
        raise ValueError("Descriptor columns must have unique names.")
    for name in columns:
        if name not in descriptors:
            raise ValueError(f"Grouping labels are missing: descriptor column {name!r}.")
        labels = descriptors[name]
        invalid = labels.isna().any() or any(
            (isinstance(value, str) and not value.strip())
            or (isinstance(value, Real) and not isfinite(value))
            for value in labels
        )
        if invalid:
            raise ValueError(
                f"Descriptor {name!r} requires nonmissing, nonempty, finite grouping labels."
            )


@dataclass(frozen=True)
class QualityPolicy:
    """Fixed thresholds; choose these before evaluating predictive performance."""

    min_coverage: float = 0.0
    rejected_flags: tuple[str, ...] = ()
    # Last, so policies built positionally keep their meaning. Only Morlet values rest on part
    # of their window; every other value has full support and always passes.
    min_support: float = 0.0

    def __post_init__(self) -> None:
        for name in ("min_coverage", "min_support"):
            threshold: float = getattr(self, name)
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, Real)
                or not isfinite(threshold)
                or not 0 <= threshold <= 1
            ):
                raise ValueError(f"{name} must be a finite fraction in [0, 1].")
        if not isinstance(self.rejected_flags, tuple) or any(
            not isinstance(name, str) or not name for name in self.rejected_flags
        ):
            raise TypeError("rejected_flags must be a tuple of nonempty flag names.")
        if len(set(self.rejected_flags)) != len(self.rejected_flags):
            raise ValueError("rejected_flags must be unique.")


@dataclass(frozen=True)
class QualityResult:
    table: FeatureTable
    ledger: pd.DataFrame


def apply_quality(table: FeatureTable, policy: QualityPolicy) -> QualityResult:
    """Mask rejected cells, retaining earlier quality exclusions in flags and the ledger."""
    unknown = set(policy.rejected_flags) - set(table.flags)
    if unknown:
        raise ValueError(f"Quality policy names unknown flags: {sorted(unknown)}.")
    reasons = {"low_coverage": table.coverage < policy.min_coverage}
    if table.support is not None:
        reasons["low_support"] = table.support < policy.min_support
    for name in policy.rejected_flags:
        existing = reasons.get(name)
        reasons[name] = table.flags[name] if existing is None else existing | table.flags[name]
    if "quality_rejected" in table.flags:
        reasons["quality_rejected"] = table.flags["quality_rejected"]
    rejected = np.logical_or.reduce(tuple(reasons.values()))
    values = table.values.copy()
    values[rejected] = np.nan
    records = []
    for row, column in zip(*np.nonzero(rejected), strict=True):
        record = {
            "row": int(row),
            "feature": table.meta[column].name,
            "reason": ";".join(name for name, mask in reasons.items() if mask[row, column]),
            "coverage": float(table.coverage[row, column]),
        }
        records.append(record)
    ledger = pd.DataFrame(records, columns=["row", "feature", "reason", "coverage"])
    flags = {**table.flags, "quality_rejected": rejected}
    return QualityResult(replace(table, values=values, flags=flags), ledger)


def feature_quality(table: FeatureTable) -> pd.DataFrame:
    """Summarize finite values, support and flags for each feature definition."""
    frame = pd.DataFrame(
        {
            "feature": table.names,
            "measure": [meta.measure for meta in table.meta],
            "n_rows": table.n_rows,
            "n_finite": np.isfinite(table.values).sum(axis=0),
            "missing_fraction": 1 - np.isfinite(table.values).mean(axis=0),
            "mean_coverage": table.coverage.mean(axis=0),
        }
    )
    for name, mask in table.flags.items():
        frame[f"flag_{name}_fraction"] = mask.mean(axis=0)
    return frame


def cohort_quality(
    table: FeatureTable, descriptors: pd.DataFrame, *, by: tuple[str, ...]
) -> pd.DataFrame:
    """Summarize all cells by explicit, aligned study/condition/session labels."""
    if len(descriptors) != table.n_rows:
        raise ValueError("Descriptors must have one aligned row per feature row.")
    if not by or len(set(by)) != len(by):
        raise ValueError("by must contain unique descriptor column names.")
    _validate_descriptor_labels(descriptors, by)
    records = []
    aligned = descriptors.reset_index(drop=True)
    grouped = aligned.groupby(list(by), sort=False, dropna=False, observed=True)
    for labels, positions in grouped.indices.items():
        labels = labels if isinstance(labels, tuple) else (labels,)
        values, coverage = table.values[positions], table.coverage[positions]
        record = dict(zip(by, labels, strict=True))
        record.update(
            n_rows=len(positions),
            n_cells=values.size,
            missing_fraction=float(1 - np.isfinite(values).mean()),
            mean_coverage=float(coverage.mean()),
        )
        for name, mask in table.flags.items():
            record[f"flag_{name}_fraction"] = float(mask[positions].mean())
        records.append(record)
    return pd.DataFrame(records)
