"""Prespecified quality exclusions preserve evidence through design construction."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from eegtable.bands import Band
from eegtable.model.design import build_design
from eegtable.table import ComputationSpec, FeatureMeta, FeatureTable


@pytest.fixture
def alpha_beta_table():
    return FeatureTable(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        coverage=np.ones((2, 2)),
        meta=tuple(
            FeatureMeta(
                measure="power",
                band=Band(name, lo, hi),
                space="C3",
                space_kind="channel",
                window="whole",
                normalization="raw",
                unit="V²",
                source="welch",
                window_bounds=(0.0, 1.0),
                computation=ComputationSpec.create("band_power"),
            )
            for name, lo, hi in (("alpha", 8.0, 13.0), ("beta", 13.0, 30.0))
        ),
        row_ids=(("recording", 0, "rest"), ("recording", 1, "task")),
    )


def test_quality_policy_masks_only_requested_cells(alpha_beta_table):
    from eegtable.quality import QualityPolicy, apply_quality

    table = replace(
        alpha_beta_table,
        coverage=np.array([[0.4, 1.0], [1.0, 1.0]]),
        flags={"edge_hit": np.array([[False, False], [False, True]])},
    )
    result = apply_quality(table, QualityPolicy(min_coverage=0.8, rejected_flags=("edge_hit",)))
    assert np.isnan(result.table.values[[0, 1], [0, 1]]).all()
    np.testing.assert_array_equal(result.table.values[[0, 1], [1, 0]], table.values[[0, 1], [1, 0]])
    np.testing.assert_array_equal(result.table.coverage, table.coverage)
    assert result.ledger["reason"].tolist() == ["low_coverage", "edge_hit"]
    assert np.isfinite(table.values).all()


def test_unknown_quality_flag_is_an_error(alpha_beta_table):
    from eegtable.quality import QualityPolicy, apply_quality

    with pytest.raises(ValueError, match="unknown"):
        apply_quality(alpha_beta_table, QualityPolicy(rejected_flags=("typo",)))


@pytest.mark.parametrize("value", [True, "0.5", [0.5], np.array([0.5]), np.nan, np.inf])
def test_quality_policy_requires_a_finite_scalar_fraction(value):
    from eegtable.quality import QualityPolicy

    with pytest.raises(ValueError, match="min_coverage"):
        QualityPolicy(min_coverage=value)


def test_design_retains_quality_and_feature_definitions(alpha_beta_table):
    from eegtable.quality import QualityPolicy

    table = replace(alpha_beta_table, coverage=np.array([[0.4, 1.0], [1.0, 1.0]]))
    targets = pd.DataFrame(table.row_ids, columns=["recording", "epoch", "event"])
    targets["target"] = [1.0, 2.0]
    targets["subject_id"] = ["01", "02"]
    design = build_design(table, targets, target="target", quality=QualityPolicy(min_coverage=0.8))
    assert np.isnan(design.X[0, 0])
    np.testing.assert_array_equal(design.coverage, table.coverage)
    assert design.meta == table.meta
    assert len(design.quality_ledger) == 1


def test_cohort_report_counts_cells_by_condition(alpha_beta_table):
    from eegtable.quality import cohort_quality, feature_quality

    targets = pd.DataFrame({"condition": ["rest", "task"]})
    table = replace(alpha_beta_table, values=np.array([[1.0, np.nan], [2.0, 3.0]]))
    summary = cohort_quality(table, targets, by=("condition",))
    assert summary["n_rows"].tolist() == [1, 1]
    assert summary["missing_fraction"].tolist() == [0.5, 0.0]
    assert feature_quality(table)["n_finite"].tolist() == [2, 1]


def test_report_preserves_all_group_keys_with_missing_labels(alpha_beta_table):
    from eegtable.quality import cohort_quality

    with pytest.raises(ValueError, match="missing"):
        cohort_quality(
            alpha_beta_table, pd.DataFrame({"condition": ["rest", None]}), by=("condition",)
        )


def test_input_low_coverage_flag_cannot_override_the_policy_threshold(alpha_beta_table):
    from eegtable.quality import QualityPolicy, apply_quality

    table = replace(
        alpha_beta_table,
        coverage=np.zeros_like(alpha_beta_table.coverage),
        flags={"low_coverage": np.zeros_like(alpha_beta_table.values, dtype=bool)},
    )
    result = apply_quality(table, QualityPolicy(min_coverage=0.5, rejected_flags=("low_coverage",)))
    assert np.isnan(result.table.values).all()
    assert result.table.flags["quality_rejected"].all()
    assert result.ledger.reason.tolist() == ["low_coverage"] * table.values.size


def test_reapplying_quality_preserves_prior_exclusion_evidence(alpha_beta_table):
    from eegtable.quality import QualityPolicy, apply_quality

    coverage = alpha_beta_table.coverage.copy()
    coverage[0, 0] = 0.2
    first = apply_quality(
        replace(alpha_beta_table, coverage=coverage), QualityPolicy(min_coverage=0.8)
    )
    repeated = apply_quality(first.table, QualityPolicy())
    np.testing.assert_array_equal(
        repeated.table.flags["quality_rejected"], first.table.flags["quality_rejected"]
    )
    assert repeated.ledger["row"].tolist() == [0]
    assert repeated.ledger.reason.tolist() == ["quality_rejected"]


@pytest.mark.parametrize("value", [None, "", " ", np.inf])
def test_design_rejects_missing_or_blank_group_identity(alpha_beta_table, value):
    targets = pd.DataFrame(alpha_beta_table.row_ids, columns=["recording", "epoch", "event"])
    targets["target"] = [1.0, 2.0]
    targets["subject_id"] = pd.Series(["a", value], dtype=object)
    with pytest.raises(ValueError, match="subject_id"):
        build_design(alpha_beta_table, targets, target="target")


def test_design_rejects_missing_run_identity(alpha_beta_table):
    targets = pd.DataFrame(alpha_beta_table.row_ids, columns=["recording", "epoch", "event"])
    targets["target"] = [1.0, 2.0]
    targets["subject_id"] = ["a", "b"]
    targets["run"] = ["one", None]
    with pytest.raises(ValueError, match="run"):
        build_design(alpha_beta_table, targets, target="target", runs="run")


def test_quality_policy_masks_cells_resting_on_too_little_of_their_window(alpha_beta_table):
    from eegtable.quality import QualityPolicy, apply_quality

    table = replace(alpha_beta_table, support=np.array([[0.2, 1.0], [1.0, 1.0]]))
    result = apply_quality(table, QualityPolicy(min_support=0.5))
    assert np.isnan(result.table.values[0, 0])
    assert np.isfinite(result.table.values[[0, 1, 1], [1, 0, 1]]).all()
    assert result.ledger["reason"].tolist() == ["low_support"]
