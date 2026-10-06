from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from eegtable.model.design import Selection, build_design, harmonize_fold, select
from eegtable.quality import QualityPolicy
from eegtable.table import FeatureMeta, FeatureTable


def test_every_selection_field_is_a_real_feature_meta_field() -> None:
    # Selection keys are metadata field names rather than name fragments, so a measure
    # that gets renamed upstream fails loudly here instead of selecting nothing.
    fields = set(Selection.__dataclass_fields__) - {"exclude"}
    assert fields <= set(FeatureMeta.__dataclass_fields__)


def test_excluding_a_measure_keeps_every_other_column(alpha_beta_table: FeatureTable) -> None:
    # Dropping one measure used to mean listing every other one.
    kept = select(alpha_beta_table, Selection(exclude=Selection(band=("beta",))))
    assert [m.band.name for m in kept.meta if m.band is not None] == ["alpha"]


def test_an_exclusion_applies_after_the_inclusion(alpha_beta_table: FeatureTable) -> None:
    selection = Selection(measure=("power",), exclude=Selection(band=("alpha",)))
    assert [m.band.name for m in select(alpha_beta_table, selection).meta if m.band] == ["beta"]


def test_an_exclusion_that_restricts_nothing_is_refused(alpha_beta_table: FeatureTable) -> None:
    # An empty Selection matches every column, so excluding it would exclude everything.
    with pytest.raises(ValueError, match="exclude"):
        select(alpha_beta_table, Selection(exclude=Selection()))


def test_unmatched_rows_are_counted_on_both_sides_and_named(
    alpha_beta_table: FeatureTable,
) -> None:
    # Excluded trials leave feature rows with no target. Say how many, and which, so the
    # table can be cut to the modeled rows rather than rebuilt by hand.
    targets = pd.DataFrame(
        {
            "recording": ["sub-01", "sub-03"],
            "epoch": [0, 0],
            "event": ["stim", "stim"],
            "pain": [1.0, 2.0],
            "subject_id": ["sub-01", "sub-03"],
        }
    )
    with pytest.raises(ValueError) as error:
        build_design(alpha_beta_table, targets, target="pain")
    message = str(error.value)
    assert "1 feature row has no target row" in message and "sub-02" in message
    assert "1 target row has no feature row" in message and "sub-03" in message
    assert "take" in message


def test_selecting_a_band_keeps_only_that_band(alpha_beta_table: FeatureTable) -> None:
    kept = select(alpha_beta_table, Selection(band=("alpha",)))
    assert {m.band.name for m in kept.meta if m.band is not None} == {"alpha"}


def test_selected_features_retain_support_for_quality_exclusions(alpha_beta_table) -> None:
    table = replace(alpha_beta_table, support=np.array([[0.2, 1.0], [0.9, 1.0]]))
    targets = pd.DataFrame(table.row_ids, columns=["recording", "epoch", "event"])
    targets["subject_id"] = targets.recording
    targets["pain"] = [1.0, 9.0]

    design = build_design(
        table,
        targets,
        target="pain",
        selection=Selection(band=("alpha",)),
        quality=QualityPolicy(min_support=0.5),
    )

    assert np.isnan(design.X[0, 0])
    assert design.X[1, 0] == table.values[1, 0]
    np.testing.assert_array_equal(design.support, [[0.2], [0.9]])
    assert design.quality_ledger.reason.tolist() == ["low_support"]


def test_an_empty_field_places_no_restriction(alpha_beta_table: FeatureTable) -> None:
    kept = select(alpha_beta_table, Selection())
    assert kept.values.shape == alpha_beta_table.values.shape


def test_selecting_a_space_keeps_only_those_channels(alpha_beta_table: FeatureTable) -> None:
    # Choosing channels or ROIs is a fixed choice of columns, so it belongs to the design
    # rather than to a fitted pipeline step, and it matches the metadata field exactly.
    meta = (alpha_beta_table.meta[0], replace(alpha_beta_table.meta[1], space="C4"))
    table = FeatureTable(
        values=alpha_beta_table.values,
        coverage=alpha_beta_table.coverage,
        meta=meta,
        row_ids=alpha_beta_table.row_ids,
    )
    kept = select(table, Selection(space=("C4",)))
    assert [m.space for m in kept.meta] == ["C4"]


def test_a_selection_that_matches_nothing_raises_rather_than_returning_empty(
    alpha_beta_table: FeatureTable,
) -> None:
    with pytest.raises(ValueError, match="no columns"):
        select(alpha_beta_table, Selection(band=("delta",)))


def test_the_join_uses_the_whole_row_id_not_the_epoch_index(
    alpha_beta_table: FeatureTable,
) -> None:
    # Two recordings both number their epochs from zero. Joining on the epoch index alone
    # matches sub-02's epoch 0 to sub-01's target and raises nothing.
    targets = pd.DataFrame(
        {
            "recording": ["sub-01", "sub-02"],
            "epoch": [0, 0],
            "event": ["stim", "stim"],
            "pain": [1.0, 9.0],
            "subject_id": ["sub-01", "sub-02"],
        }
    )
    design = build_design(alpha_beta_table, targets, target="pain")
    assert design.row_ids == (("sub-01", 0, "stim"), ("sub-02", 0, "stim"))
    np.testing.assert_array_equal(design.y, [1.0, 9.0])


def test_a_target_row_matching_two_feature_rows_is_refused(
    alpha_beta_table: FeatureTable,
) -> None:
    targets = pd.DataFrame(
        {
            "recording": ["sub-01", "sub-01"],
            "epoch": [0, 0],
            "event": ["stim", "stim"],
            "pain": [1.0, 2.0],
            "subject_id": ["sub-01", "sub-01"],
        }
    )
    with pytest.raises(ValueError, match="one-to-one"):
        build_design(alpha_beta_table, targets, target="pain")


def test_cross_trial_tables_are_explicitly_outside_the_modeling_boundary(
    alpha_beta_table: FeatureTable,
) -> None:
    grouped = FeatureTable(
        values=alpha_beta_table.values,
        coverage=alpha_beta_table.coverage,
        meta=alpha_beta_table.meta,
        row_labels=("left", "right"),
    )

    with pytest.raises(ValueError, match="per-epoch only"):
        build_design(grouped, pd.DataFrame(), target="pain")


def test_cross_trial_boundary_is_checked_before_feature_selection(
    alpha_beta_table: FeatureTable,
) -> None:
    grouped = FeatureTable(
        values=alpha_beta_table.values,
        coverage=alpha_beta_table.coverage,
        meta=alpha_beta_table.meta,
        row_labels=("left", "right"),
    )

    with pytest.raises(ValueError, match="per-epoch only"):
        build_design(
            grouped,
            pd.DataFrame(),
            target="pain",
            selection=Selection(measure=("absent",)),
        )


def test_covariate_columns_are_identified_not_merely_counted(
    alpha_beta_table: FeatureTable,
) -> None:
    targets = pd.DataFrame(
        {
            "recording": ["sub-01", "sub-02"],
            "epoch": [0, 0],
            "event": ["stim", "stim"],
            "pain": [1.0, 9.0],
            "age": [30.0, 40.0],
            "subject_id": ["sub-01", "sub-02"],
        }
    )
    design = build_design(alpha_beta_table, targets, target="pain", covariates=["age"])
    assert design.column_names[design.covariate_columns[0]] == "age"
    assert design.covariate_columns.size == 1


def test_covariates_including_target_is_rejected_as_leakage(
    alpha_beta_table: FeatureTable,
) -> None:
    targets = pd.DataFrame(
        {
            "recording": ["sub-01", "sub-02"],
            "epoch": [0, 0],
            "event": ["stim", "stim"],
            "pain": [1.0, 9.0],
            "subject_id": ["sub-01", "sub-02"],
        }
    )
    with pytest.raises(ValueError, match="leak"):
        build_design(alpha_beta_table, targets, target="pain", covariates=["pain"])


def test_missing_covariates_raises_when_strict(
    alpha_beta_table: FeatureTable,
) -> None:
    targets = pd.DataFrame(
        {
            "recording": ["sub-01", "sub-02"],
            "epoch": [0, 0],
            "event": ["stim", "stim"],
            "pain": [1.0, 9.0],
            "subject_id": ["sub-01", "sub-02"],
        }
    )
    with pytest.raises(ValueError, match="missing"):
        build_design(
            alpha_beta_table,
            targets,
            target="pain",
            covariates=["missing_col"],
            strict_covariates=True,
        )


def test_missing_covariates_dropped_when_not_strict(
    alpha_beta_table: FeatureTable,
) -> None:
    targets = pd.DataFrame(
        {
            "recording": ["sub-01", "sub-02"],
            "epoch": [0, 0],
            "event": ["stim", "stim"],
            "pain": [1.0, 9.0],
            "subject_id": ["sub-01", "sub-02"],
        }
    )
    design = build_design(
        alpha_beta_table,
        targets,
        target="pain",
        covariates=["missing_col"],
        strict_covariates=False,
    )
    assert design.covariate_columns.size == 0


def test_harmonize_fold_drops_features_missing_in_any_training_group() -> None:
    X_tr = np.array(
        [
            [1.0, 10.0],
            [2.0, np.nan],
            [3.0, 30.0],
            [4.0, 40.0],
        ],
        dtype=float,
    )
    X_te = np.array([[5.0, 50.0]], dtype=float)
    groups = np.array(["sub-01", "sub-01", "sub-02", "sub-02"], dtype=object)

    X_tr_h, X_te_h, keep = harmonize_fold(X_tr, X_te, groups, mode="intersection")
    assert np.array_equal(keep, [True, True])  # sub-01 has 10.0 in row 0, sub-02 has 30, 40

    # If sub-01 has all NaNs in column 1:
    X_tr[0, 1] = np.nan
    X_tr_h, X_te_h, keep = harmonize_fold(X_tr, X_te, groups, mode="intersection")
    assert np.array_equal(keep, [True, False])
    assert X_tr_h.shape == (4, 1)
    assert X_te_h.shape == (1, 1)


def test_harmonize_fold_rejects_empty_strict_intersection() -> None:
    X_tr = np.array(
        [
            [1.0, np.nan],
            [2.0, np.nan],
            [np.nan, 3.0],
            [np.nan, 4.0],
        ],
        dtype=float,
    )
    X_te = np.array([[1.0, 1.0]], dtype=float)
    groups = np.array(["sub-01", "sub-01", "sub-02", "sub-02"], dtype=object)

    with pytest.raises(ValueError, match="No features are finite for every training group"):
        harmonize_fold(X_tr, X_te, groups, mode="intersection")


def test_harmonize_fold_intersection_differs_from_union_impute() -> None:
    X_tr = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, 10.0], [4.0, 20.0]], dtype=float)
    X_te = np.array([[5.0, 30.0]], dtype=float)
    groups = np.array(["s1", "s1", "s2", "s2"], dtype=object)

    X_tr_int, X_te_int, keep_int = harmonize_fold(X_tr, X_te, groups, mode="intersection")
    assert np.array_equal(keep_int, [True, False])
    assert X_tr_int.shape == (4, 1)

    X_tr_union, X_te_union, keep_union = harmonize_fold(X_tr, X_te, groups, mode="union_impute")
    assert np.array_equal(keep_union, [True, True])
    assert X_tr_union.shape == (4, 2)


def test_harmonize_fold_rejects_unknown_mode() -> None:
    X_tr = np.ones((4, 2))
    X_te = np.ones((2, 2))
    groups = np.array(["s1", "s1", "s2", "s2"], dtype=object)
    with pytest.raises(ValueError, match="Unknown harmonization mode"):
        harmonize_fold(X_tr, X_te, groups, mode="intersecton")


def test_a_target_of_text_labels_is_named_as_such(alpha_beta_table: FeatureTable) -> None:
    # Text labels coerced to NaN used to be reported as non-finite values, which sent the
    # reader looking for missing data instead of an encoding.
    targets = pd.DataFrame(
        {
            "recording": ["sub-01", "sub-02"],
            "epoch": [0, 0],
            "event": ["stim", "stim"],
            "hand": ["left", "right"],
            "subject_id": ["sub-01", "sub-02"],
        }
    )
    with pytest.raises(ValueError, match="text labels.*'left'"):
        build_design(alpha_beta_table, targets, target="hand")
