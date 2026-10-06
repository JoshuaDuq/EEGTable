import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import eegtable.table as table_module
from eegtable.bands import Band
from eegtable.table import ComputationSpec, FeatureMeta, FeatureTable, concat

ALPHA = Band("alpha", 8.0, 13.0)


def _meta(space: str = "C3", measure: str = "power") -> FeatureMeta:
    return FeatureMeta(
        measure=measure,
        band=ALPHA,
        space=space,
        space_kind="channel",
        window="stim",
        normalization="log_ratio",
        unit="log10",
        source="morlet",
        window_bounds=(0.0, 1.0),
        computation=ComputationSpec.create("band_power", weighting="trapezoid"),
    )


def _table(spaces: tuple[str, ...] = ("C3", "C4")) -> FeatureTable:
    n = len(spaces)
    return FeatureTable(
        values=np.arange(3 * n, dtype=float).reshape(3, n),
        coverage=np.ones((3, n)),
        meta=tuple(_meta(s) for s in spaces),
    )


def test_name_is_derived_from_metadata_fields() -> None:
    assert _meta().name.startswith("eeg_power_alpha_c3_stim_log-ratio_p")


def test_parameter_hash_is_canonical_and_stable() -> None:
    left = ComputationSpec.create("burst", threshold=0.75, minimum_duration=0.1)
    right = ComputationSpec.create("burst", minimum_duration=0.1, threshold=0.75)

    assert left == right
    assert left.parameter_hash == right.parameter_hash


def test_numpy_non_finite_parameters_are_encoded_like_python_ones() -> None:
    # A NumPy infinity is still infinity: it takes the same token a Python one does,
    # rather than reaching json.dumps(allow_nan=False) raw and aborting the measure.
    spec = ComputationSpec.create("x", bounds=[np.float64(-np.inf), np.float64(np.inf)])
    assert spec.parameters == {"bounds": ["-Infinity", "Infinity"]}


def test_window_bounds_and_parameters_distinguish_feature_identifiers() -> None:
    base = _meta()
    shifted = replace(base, window_bounds=(0.25, 1.25))
    thresholded = replace(
        base,
        computation=ComputationSpec.create("band_power", weighting="trapezoid", threshold=0.75),
    )

    assert len({base.name, shifted.name, thresholded.name}) == 3


def test_dataframe_columns_are_the_canonical_names() -> None:
    df = _table().to_dataframe()
    assert list(df.columns) == [_meta("C3").name, _meta("C4").name]
    assert df.shape == (3, 2)


def test_select_filters_columns_by_metadata_not_by_string_matching() -> None:
    selected = _table().select(space="C4")
    assert len(selected.meta) == 1
    assert selected.meta[0].space == "C4"
    assert selected.values.shape == (3, 1)


def test_select_on_an_unknown_field_raises() -> None:
    with pytest.raises(ValueError, match="not a FeatureMeta field"):
        _table().select(channel="C4")


def test_duplicate_column_names_raise() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        FeatureTable(
            values=np.zeros((3, 2)),
            coverage=np.ones((3, 2)),
            meta=(_meta("C3"), _meta("C3")),
        )


def test_mismatched_meta_length_raises() -> None:
    with pytest.raises(ValueError, match="meta"):
        FeatureTable(values=np.zeros((3, 2)), coverage=np.ones((3, 2)), meta=(_meta("C3"),))


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -0.1, 1.1])
def test_coverage_must_be_a_finite_fraction(invalid: float) -> None:
    coverage = np.ones((3, 2))
    coverage[0, 0] = invalid

    with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
        FeatureTable(values=np.zeros((3, 2)), coverage=coverage, meta=(_meta("C3"), _meta("C4")))


def test_values_must_be_real() -> None:
    with pytest.raises(TypeError, match="real"):
        replace(_table(), values=np.full((3, 2), 1.0 + 2.0j))


@pytest.mark.parametrize("field", ["coverage", "support"])
def test_fraction_arrays_must_be_real(field: str) -> None:
    with pytest.raises(ValueError, match="real"):
        replace(_table(), **{field: np.full((3, 2), 0.5 + 2.0j)})


def test_flags_must_match_the_value_shape() -> None:
    with pytest.raises(ValueError, match="flag"):
        FeatureTable(
            values=np.zeros((3, 2)),
            coverage=np.ones((3, 2)),
            meta=(_meta("C3"), _meta("C4")),
            flags={"edge_hit": np.zeros((2, 2), dtype=bool)},
        )


def test_concat_joins_columns_and_preserves_flags() -> None:
    identities = tuple(("recording", epoch, "event") for epoch in range(3))
    left = FeatureTable(
        values=np.zeros((3, 1)),
        coverage=np.ones((3, 1)),
        meta=(_meta("C3"),),
        flags={"edge_hit": np.ones((3, 1), dtype=bool)},
        row_ids=identities,
    )
    right = FeatureTable(
        values=np.zeros((3, 1)),
        coverage=np.ones((3, 1)),
        meta=(_meta("C4"),),
        row_ids=identities,
    )
    joined = concat([left, right])
    assert joined.values.shape == (3, 2)
    assert joined.flags["edge_hit"].shape == (3, 2)
    assert joined.flags["edge_hit"][:, 0].all()
    assert not joined.flags["edge_hit"][:, 1].any()


def test_concat_refuses_to_align_mismatched_epoch_counts() -> None:
    small = FeatureTable(np.zeros((2, 1)), np.ones((2, 1)), (_meta("C3"),))
    with pytest.raises(ValueError, match="n_rows"):
        concat([small, _table(("C4",))])


def test_concat_refuses_unidentified_epoch_rows() -> None:
    with pytest.raises(ValueError, match="row_ids"):
        concat([_table(("C3",)), _table(("C4",))])


def test_concat_requires_exact_epoch_identity_and_order() -> None:
    identities = (("sub-01_task-rest", 4, "eyes-open"), ("sub-01_task-rest", 9, "eyes-closed"))
    left = FeatureTable(
        np.zeros((2, 1)),
        np.ones((2, 1)),
        (_meta("C3"),),
        row_ids=identities,
    )
    reordered = FeatureTable(
        np.zeros((2, 1)),
        np.ones((2, 1)),
        (_meta("C4"),),
        row_ids=tuple(reversed(identities)),
    )

    with pytest.raises(ValueError, match="row identities"):
        concat([left, reordered])


def test_concat_preserves_matching_epoch_identities() -> None:
    identities = (("sub-01_task-rest", 4, "eyes-open"), ("sub-01_task-rest", 9, "eyes-closed"))
    left = FeatureTable(np.zeros((2, 1)), np.ones((2, 1)), (_meta("C3"),), row_ids=identities)
    right = FeatureTable(np.zeros((2, 1)), np.ones((2, 1)), (_meta("C4"),), row_ids=identities)

    assert concat([left, right]).row_ids == identities


def test_concat_of_nothing_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        concat([])


def _identified_table(recording: str, *, start: float = 0.0) -> FeatureTable:
    return FeatureTable(
        values=np.arange(start, start + 6.0).reshape(3, 2),
        coverage=np.full((3, 2), 0.75),
        meta=(_meta("C3"), _meta("C4")),
        flags={"edge_hit": np.array([[True, False], [False, False], [False, True]])},
        row_ids=tuple((recording, epoch, "stim") for epoch in range(3)),
    )


def test_stack_rows_combines_compatible_epoch_tables() -> None:
    first = _identified_table("recording-01")
    second = replace(_identified_table("recording-02", start=6.0), flags={})

    stacked = table_module.stack_rows([first, second])

    np.testing.assert_array_equal(stacked.values, np.arange(12.0).reshape(6, 2))
    np.testing.assert_array_equal(stacked.coverage, np.full((6, 2), 0.75))
    assert stacked.meta == first.meta
    assert stacked.row_ids == first.row_ids + second.row_ids
    np.testing.assert_array_equal(stacked.flags["edge_hit"][:3], first.flags["edge_hit"])
    assert not stacked.flags["edge_hit"][3:].any()


def test_stack_rows_refuses_incompatible_feature_schemas() -> None:
    incompatible = replace(_identified_table("recording-02"), meta=(_meta("C3"), _meta("Pz")))

    with pytest.raises(ValueError, match="same ordered feature metadata"):
        table_module.stack_rows([_identified_table("recording-01"), incompatible])


def test_stack_rows_refuses_duplicate_row_identities() -> None:
    table = _identified_table("recording-01")

    with pytest.raises(ValueError, match="duplicate row_ids"):
        table_module.stack_rows([table, table])


def test_stack_rows_refuses_cross_trial_tables() -> None:
    grouped = FeatureTable(
        values=np.zeros((2, 2)),
        coverage=np.ones((2, 2)),
        meta=(_meta("C3"), _meta("C4")),
        row_labels=("left", "right"),
    )

    with pytest.raises(ValueError, match="per-epoch"):
        table_module.stack_rows([grouped])


def test_stack_rows_of_nothing_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        table_module.stack_rows([])


# --- row semantics --------------------------------------------------------------------


def test_rows_are_epochs_unless_labelled() -> None:
    assert _table().row_labels is None
    assert _table().n_rows == 3


def test_labelled_rows_index_the_dataframe() -> None:
    table = FeatureTable(
        values=np.zeros((2, 1)),
        coverage=np.ones((2, 1)),
        meta=(_meta("C3"),),
        row_labels=("rest", "task"),
    )
    assert list(table.to_dataframe().index) == ["rest", "task"]


def test_row_labels_must_match_the_row_count() -> None:
    with pytest.raises(ValueError, match="row_labels"):
        FeatureTable(
            values=np.zeros((2, 1)),
            coverage=np.ones((2, 1)),
            meta=(_meta("C3"),),
            row_labels=("only-one",),
        )


def test_concat_refuses_to_join_group_rows_to_epoch_rows() -> None:
    per_epoch = FeatureTable(np.zeros((2, 1)), np.ones((2, 1)), (_meta("C3"),))
    per_group = FeatureTable(
        np.zeros((2, 1)), np.ones((2, 1)), (_meta("C4"),), row_labels=("rest", "task")
    )
    with pytest.raises(ValueError, match="row semantics"):
        concat([per_epoch, per_group])


def test_select_preserves_row_labels() -> None:
    table = FeatureTable(
        values=np.zeros((2, 2)),
        coverage=np.ones((2, 2)),
        meta=(_meta("C3"), _meta("C4")),
        row_labels=("rest", "task"),
    )
    assert table.select(space="C4").row_labels == ("rest", "task")


def test_stack_rows_unions_columns_when_recordings_differ_in_channels() -> None:
    # Bad channels differ per recording, so each one measures its own channel set.
    first = FeatureTable(
        values=np.array([[1.0, 2.0]]),
        coverage=np.array([[1.0, 0.5]]),
        meta=(_meta("C3"), _meta("C4")),
        flags={"edge_hit": np.array([[True, False]])},
        row_ids=(("recording-01", 0, "stim"),),
    )
    second = FeatureTable(
        values=np.array([[3.0, 4.0]]),
        coverage=np.array([[0.25, 1.0]]),
        meta=(_meta("C3"), _meta("Pz")),
        row_ids=(("recording-02", 0, "stim"),),
    )

    stacked = table_module.stack_rows([first, second], columns="union")

    assert tuple(m.space for m in stacked.meta) == ("C3", "C4", "Pz")
    np.testing.assert_array_equal(
        stacked.values, np.array([[1.0, 2.0, np.nan], [3.0, np.nan, 4.0]])
    )
    np.testing.assert_array_equal(stacked.coverage, np.array([[1.0, 0.5, 0.0], [0.25, 0.0, 1.0]]))
    np.testing.assert_array_equal(
        stacked.flags["edge_hit"], np.array([[True, False, False], [False, False, False]])
    )
    assert stacked.row_ids == first.row_ids + second.row_ids


def test_stack_rows_union_of_identical_schemas_matches_the_strict_stack() -> None:
    tables = [_identified_table("recording-01"), _identified_table("recording-02", start=6.0)]

    union = table_module.stack_rows(tables, columns="union")
    strict = table_module.stack_rows(tables)

    assert union.meta == strict.meta
    np.testing.assert_array_equal(union.values, strict.values)
    np.testing.assert_array_equal(union.coverage, strict.coverage)


def test_stack_rows_refuses_an_unknown_columns_mode() -> None:
    with pytest.raises(ValueError, match="columns must be"):
        table_module.stack_rows([_identified_table("recording-01")], columns="outer")


# --- row and column subsets -------------------------------------------------------------


def test_take_returns_the_chosen_rows_in_order_with_their_identities_and_flags() -> None:
    table = _identified_table("recording-01")
    taken = table.take([2, 0])
    np.testing.assert_array_equal(taken.values, [[4.0, 5.0], [0.0, 1.0]])
    np.testing.assert_array_equal(taken.coverage, np.full((2, 2), 0.75))
    np.testing.assert_array_equal(taken.flags["edge_hit"], [[False, True], [True, False]])
    assert taken.row_ids == (("recording-01", 2, "stim"), ("recording-01", 0, "stim"))
    assert taken.meta == table.meta


def test_take_accepts_a_mask_and_keeps_group_row_labels() -> None:
    grouped = FeatureTable(
        values=np.arange(3.0).reshape(3, 1),
        coverage=np.ones((3, 1)),
        meta=(_meta("C3"),),
        row_labels=("left", "right", "both"),
    )
    taken = grouped.take(np.array([True, False, True]))
    assert taken.row_labels == ("left", "both")
    np.testing.assert_array_equal(taken.values, [[0.0], [2.0]])


@pytest.mark.parametrize("rows", [[0, 0], [3], [-1], np.array([True, False])])
def test_take_refuses_rows_that_repeat_or_do_not_exist(rows) -> None:
    # A repeated row would carry the same identity twice into a cohort join.
    with pytest.raises(ValueError, match="row"):
        _identified_table("recording-01").take(rows)


def test_drop_missing_keeps_columns_missing_in_at_most_the_given_fraction() -> None:
    values = np.array([[1.0, np.nan, np.nan], [2.0, 5.0, np.nan], [3.0, 6.0, np.inf]])
    table = FeatureTable(
        values=values, coverage=np.ones((3, 3)), meta=tuple(_meta(s) for s in ("C3", "C4", "Pz"))
    )
    assert [m.space for m in table.drop_missing(0.4).meta] == ["C3", "C4"]
    assert [m.space for m in table.drop_missing(0.0).meta] == ["C3"]
    with pytest.raises(ValueError, match="between 0 and 1"):
        table.drop_missing(1.5)


def test_an_incompatible_schema_is_described_by_the_columns_that_differ() -> None:
    # Diffing sidecars by hand is the alternative: name the columns and the recording.
    incompatible = replace(_identified_table("recording-02"), meta=(_meta("C3"), _meta("Pz")))
    with pytest.raises(ValueError) as error:
        table_module.stack_rows([_identified_table("recording-01"), incompatible])
    message = str(error.value)
    assert "recording-02" in message
    assert _meta("Pz").name in message and _meta("C4").name in message
    assert "1 column" in message


def test_select_matches_a_band_by_its_name() -> None:
    # The field holds a Band, so band="alpha" used to match nothing and return an empty table.
    assert _table().select(band="alpha").names == _table().names


def test_select_warns_when_a_requested_value_names_no_column() -> None:
    # The function is integrated_band_power; its columns are labelled "power". An empty
    # result here reads as "no such feature" rather than as the mistake it is.
    with pytest.warns(UserWarning, match="measure values"):
        assert _table().select(measure="band_power").meta == ()


def test_select_stays_quiet_when_known_values_simply_never_co_occur() -> None:
    table = FeatureTable(
        values=np.zeros((1, 2)),
        coverage=np.ones((1, 2)),
        meta=(_meta("C3"), _meta("C4", measure="peak")),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert table.select(space="C3", measure="peak").meta == ()


def test_a_join_on_epoch_aligns_features_with_metadata_after_dropped_epochs() -> None:
    # MNE keeps the original epoch numbers in its metadata index after a drop. A 0..n index
    # would pair epoch 4's features with epoch 0's metadata without any error.
    table = FeatureTable(
        values=np.array([[1.0], [2.0]]),
        coverage=np.ones((2, 1)),
        meta=(_meta("C3"),),
        row_ids=(("sub-01", 4, "stim"), ("sub-01", 7, "stim")),
    )
    metadata = pd.DataFrame({"rating": [10, 40, 70]}, index=[0, 4, 7])
    joined = table.to_dataframe().join(metadata, on="epoch")
    assert joined["rating"].tolist() == [40, 70]
    assert joined.index.names == ["recording", "epoch", "event"]


def test_to_long_gives_one_row_per_cell_with_its_identity_and_metadata() -> None:
    long = _identified_table("sub-01").to_long()
    assert len(long) == 6
    first, last = long.iloc[0], long.iloc[-1]
    assert (first["recording"], first["epoch"], first["event"]) == ("sub-01", 0, "stim")
    assert (first["space"], first["band"], first["window"], first["unit"]) == (
        "C3",
        "alpha",
        "stim",
        "log10",
    )
    assert (first["band_fmin"], first["band_fmax"]) == (8.0, 13.0)
    assert (first["window_tmin"], first["window_tmax"]) == (0.0, 1.0)
    assert (last["epoch"], last["space"], last["value"], last["coverage"]) == (2, "C4", 5.0, 0.75)
    assert first["feature"] == _meta("C3").name


def test_to_long_carries_each_flag_beside_the_cell_it_marks() -> None:
    long = _identified_table("sub-01").to_long()
    flagged = long.loc[long["edge_hit"], ["epoch", "space"]]
    assert flagged.values.tolist() == [[0, "C3"], [2, "C4"]]


def test_to_long_names_group_rows_by_their_label() -> None:
    table = FeatureTable(
        values=np.zeros((2, 1)),
        coverage=np.ones((2, 1)),
        meta=(_meta("C3"),),
        row_labels=("rest", "task"),
    )
    assert table.to_long()["group"].tolist() == ["rest", "task"]


def test_repr_summarizes_the_table_rather_than_printing_every_value() -> None:
    # The dataclass repr printed every array and record: 300 KB for 64 channels and 5 bands.
    text = repr(_identified_table("sub-01"))
    assert "3 epoch rows" in text and "2 columns" in text and "power" in text
    assert len(text) < 200


def test_support_rides_along_with_the_columns_and_rows_it_describes() -> None:
    table = replace(_identified_table("sub-01"), support=np.array([[0.5, 1.0]] * 3))
    selected = table.select(space="C4")
    assert selected.support is not None and selected.support.tolist() == [[1.0]] * 3
    taken = table.take([2])
    assert taken.support is not None and taken.support.tolist() == [[0.5, 1.0]]


@pytest.mark.parametrize("support", [np.ones((3, 1)), np.full((3, 2), 1.5)])
def test_support_must_be_a_fraction_shaped_like_the_values(support) -> None:
    with pytest.raises(ValueError, match="support"):
        replace(_identified_table("sub-01"), support=support)


def test_a_joined_table_without_support_reads_as_complete() -> None:
    # A Welch or time-domain column rests on the whole window it names.
    restricted = replace(_identified_table("sub-01"), support=np.full((3, 2), 0.25))
    other = replace(_identified_table("sub-01"), meta=(_meta("Pz"), _meta("Oz")))
    joined = concat([restricted, other])
    assert joined.support is not None
    np.testing.assert_array_equal(joined.support, [[0.25, 0.25, 1.0, 1.0]] * 3)


def test_a_column_a_recording_never_measured_has_no_support() -> None:
    first = replace(_identified_table("sub-01"), support=np.full((3, 2), 0.5))
    second = replace(
        _identified_table("sub-02"), meta=(_meta("C3"), _meta("Pz")), support=np.full((3, 2), 0.5)
    )
    stacked = table_module.stack_rows([first, second], columns="union")
    assert stacked.support is not None
    pz = stacked.names.index(_meta("Pz").name)
    assert stacked.support[:3, pz].tolist() == [0.0] * 3


def test_to_long_reports_complete_support_when_none_was_recorded() -> None:
    assert _identified_table("sub-01").to_long()["support"].tolist() == [1.0] * 6
