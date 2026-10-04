import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eegtable.io as io_module
from eegtable.bands import Band
from eegtable.io import read_table, write_table
from eegtable.provenance import file_hash
from eegtable.table import ComputationSpec, FeatureMeta, FeatureTable

ALPHA = Band("alpha", 8.0, 13.0)


def _resign(path: Path) -> None:
    """Re-sign a deliberately malformed fixture to exercise schema checks."""
    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["files"] = {name: file_hash(path.parent / name) for name in sidecar["files"]}
    sidecar_path.write_text(json.dumps(sidecar))


def _meta(**overrides: object) -> FeatureMeta:
    fields: dict[str, object] = {
        "measure": "power",
        "band": ALPHA,
        "space": "Cz",
        "space_kind": "channel",
        "window": "stim",
        "normalization": "log10",
        "unit": "log10(V^2/Hz)",
        "source": "welch",
        "window_bounds": (0.0, 1.0),
        "computation": ComputationSpec.create("welch", n_fft=512, window="hann"),
        "freq_resolution_hz": 0.5,
    }
    fields.update(overrides)
    return FeatureMeta(**fields)  # type: ignore[arg-type]


def _epoch_table() -> FeatureTable:
    return FeatureTable(
        # 1/3 needs all 17 significant digits, so a lossy float format cannot pass.
        values=np.array([[1.25, np.nan], [-3.5, 1.0 / 3.0], [2.0e-12, 7.0]]),
        coverage=np.array([[1.0, 0.0], [0.5, 1.0], [1.0, 0.75]]),
        meta=(
            _meta(),
            _meta(
                measure="slope",
                band=None,
                space="global",
                space_kind="global",
                window=None,
                window_bounds=None,
                normalization="raw",
                unit="a.u.",
                freq_resolution_hz=None,
            ),
        ),
        flags={"cog_fallback": np.array([[False, False], [True, False], [False, True]])},
        row_ids=(
            ("sub-01_task-test", 0, "left"),
            ("sub-01_task-test", 1, "right"),
            ("sub-01_task-test", 2, "left"),
        ),
    )


def _group_table() -> FeatureTable:
    return FeatureTable(
        values=np.array([[0.4], [0.9]]),
        coverage=np.array([[1.0], [0.5]]),
        meta=(_meta(measure="itpc", normalization="raw", unit="a.u."),),
        row_labels=("left", "right"),
    )


def _assert_same_table(actual: FeatureTable, expected: FeatureTable) -> None:
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.coverage, expected.coverage)
    assert actual.meta == expected.meta
    assert actual.row_labels == expected.row_labels
    assert actual.row_ids == expected.row_ids
    assert set(actual.flags) == set(expected.flags)
    for key, flag in expected.flags.items():
        np.testing.assert_array_equal(actual.flags[key], flag)


def test_epoch_table_round_trips_exactly(tmp_path) -> None:
    table = _epoch_table()
    write_table(table, tmp_path / "sub-01_features.tsv")

    _assert_same_table(read_table(tmp_path / "sub-01_features.tsv"), table)


def test_group_table_round_trips_with_its_row_labels(tmp_path) -> None:
    table = _group_table()
    write_table(table, tmp_path / "sub-01_crosstrial.tsv")

    _assert_same_table(read_table(tmp_path / "sub-01_crosstrial.tsv"), table)


def test_read_table_accepts_the_row_id_column_written_before_the_rename(tmp_path) -> None:
    table = _epoch_table()
    path = tmp_path / "sub-01_features.tsv"
    write_table(table, path)
    for file in (path, tmp_path / "sub-01_features_coverage.tsv"):
        file.write_text(file.read_text().replace("__eegtable_row_id", "__eegfeat_row_id", 1))
    _resign(path)

    _assert_same_table(read_table(path), table)


def test_write_returns_values_coverage_and_sidecar_paths(tmp_path) -> None:
    written = write_table(_epoch_table(), tmp_path / "sub-01_features.tsv")

    assert written == (
        tmp_path / "sub-01_features.tsv",
        tmp_path / "sub-01_features_coverage.tsv",
        tmp_path / "sub-01_features.json",
    )
    assert all(path.is_file() for path in written)


def test_bundle_publish_failure_restores_all_previous_files(tmp_path, monkeypatch) -> None:
    target = tmp_path / "sub-01_features.tsv"
    write_table(_epoch_table(), target)
    paths = (target, target.with_name(f"{target.stem}_coverage.tsv"), target.with_suffix(".json"))
    previous = {path: path.read_bytes() for path in paths}

    import eegtable.io as io

    real_replace = io.os.replace
    failed = False

    def fail_during_publish(source, destination):
        nonlocal failed
        destination = Path(destination)
        if not failed and destination == paths[1] and Path(source).parent != destination.parent:
            failed = True
            raise OSError("intentional publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(io.os, "replace", fail_during_publish)
    with pytest.raises(OSError, match="intentional"):
        write_table(_epoch_table(), target)
    assert all(path.read_bytes() == content for path, content in previous.items())


def test_values_file_leads_with_the_row_key_then_descriptors(tmp_path) -> None:
    rows = pd.DataFrame({"event": ["left", "right", "left"], "rating": [3, 5, 4]})
    write_table(_epoch_table(), tmp_path / "t.tsv", rows=rows)

    frame = pd.read_csv(tmp_path / "t.tsv", sep="\t")
    assert list(frame.columns[:3]) == ["epoch", "event", "rating"]
    assert frame["epoch"].tolist() == [0, 1, 2]
    assert frame["rating"].tolist() == [3, 5, 4]


def test_descriptor_columns_are_not_read_back_as_features(tmp_path) -> None:
    rows = pd.DataFrame({"event": ["left", "right", "left"], "rating": [3, 5, 4]})
    write_table(_epoch_table(), tmp_path / "t.tsv", rows=rows)

    assert read_table(tmp_path / "t.tsv").names == _epoch_table().names


def test_group_rows_are_keyed_by_their_labels(tmp_path) -> None:
    write_table(_group_table(), tmp_path / "t.tsv")

    frame = pd.read_csv(tmp_path / "t.tsv", sep="\t")
    assert frame.columns[0] == "group"
    assert frame["group"].tolist() == ["left", "right"]


def test_missing_values_are_written_as_bids_na(tmp_path) -> None:
    write_table(_epoch_table(), tmp_path / "t.tsv")

    second_row = (tmp_path / "t.tsv").read_text().splitlines()[1].split("\t")
    assert second_row[-1] == "n/a"


def test_sidecar_describes_every_column_with_its_band_bounds(tmp_path) -> None:
    write_table(_epoch_table(), tmp_path / "t.tsv", provenance={"input": "sub-01_epo.fif"})

    sidecar = json.loads((tmp_path / "t.json").read_text())
    power, slope = sidecar["columns"]
    assert power["name"] == _meta().name
    assert power["band"] == {"name": "alpha", "fmin": 8.0, "fmax": 13.0}
    assert slope["band"] is None
    assert sidecar["provenance"] == {"input": "sub-01_epo.fif"}


def test_reader_requires_the_descriptor_type_manifest(tmp_path) -> None:
    path = tmp_path / "features.tsv"
    write_table(_epoch_table(), path)
    sidecar = json.loads(path.with_suffix(".json").read_text())
    del sidecar["row_text_columns"]
    path.with_suffix(".json").write_text(json.dumps(sidecar))

    with pytest.raises(ValueError, match="descriptor type manifest.*regenerate"):
        read_table(path)


def test_rejects_a_path_that_is_not_tsv(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"\.tsv"):
        write_table(_epoch_table(), tmp_path / "t.csv")


def test_rejects_descriptor_rows_of_the_wrong_length(tmp_path) -> None:
    with pytest.raises(ValueError, match="3 rows"):
        write_table(_epoch_table(), tmp_path / "t.tsv", rows=pd.DataFrame({"event": ["a"]}))


@pytest.mark.parametrize("columns", [["rating", "rating"], ["", "rating"], [1, "rating"]])
def test_rejects_descriptor_names_that_cannot_round_trip(tmp_path, columns) -> None:
    rows = pd.DataFrame([[1, 10], [2, 20], [3, 30]], columns=columns)

    with pytest.raises(ValueError, match="descriptor columns"):
        write_table(_epoch_table(), tmp_path / "t.tsv", rows=rows)

    assert not list(tmp_path.iterdir())


def test_read_dataset_rejects_duplicate_descriptor_names_in_sidecar(tmp_path) -> None:
    path = tmp_path / "features.tsv"
    write_table(_epoch_table(), path, rows=pd.DataFrame({"rating": [1, 2, 3]}))
    sidecar = json.loads(path.with_suffix(".json").read_text())
    sidecar["row_columns"].append("rating")
    path.with_suffix(".json").write_text(json.dumps(sidecar))

    with pytest.raises(ValueError, match="descriptor columns"):
        io_module.read_dataset([path])


def test_rejects_a_descriptor_that_shadows_the_row_key(tmp_path) -> None:
    rows = pd.DataFrame({"epoch": [7, 8, 9]})
    with pytest.raises(ValueError, match="epoch"):
        write_table(_epoch_table(), tmp_path / "t.tsv", rows=rows)


def test_read_fails_when_the_values_file_lacks_a_described_column(tmp_path) -> None:
    write_table(_epoch_table(), tmp_path / "t.tsv")
    frame = pd.read_csv(tmp_path / "t.tsv", sep="\t", keep_default_na=False)
    missing_name = _epoch_table().names[1]
    frame.drop(columns=[missing_name]).to_csv(tmp_path / "t.tsv", sep="\t", index=False)
    _resign(tmp_path / "t.tsv")

    with pytest.raises(ValueError, match=missing_name):
        read_table(tmp_path / "t.tsv")


def test_read_dataset_stacks_tables_and_restores_aligned_targets(tmp_path) -> None:
    first = _epoch_table()
    second = FeatureTable(
        values=first.values + 10.0,
        coverage=first.coverage,
        meta=first.meta,
        flags=first.flags,
        row_ids=tuple(("sub-02_task-test", epoch, event) for _, epoch, event in first.row_ids),
    )
    paths = [tmp_path / "sub-01_features.tsv", tmp_path / "sub-02_features.tsv"]
    for path, table, ratings in zip(paths, (first, second), ([3, 5, 4], [2, 1, 0]), strict=True):
        rows = pd.DataFrame({"event": [event for _, _, event in table.row_ids], "rating": ratings})
        write_table(table, path, rows=rows)

    dataset = io_module.read_dataset(paths)

    assert dataset.table.row_ids == first.row_ids + second.row_ids
    np.testing.assert_array_equal(dataset.table.values, np.vstack([first.values, second.values]))
    assert list(dataset.targets.columns) == ["recording", "epoch", "event", "rating"]
    assert dataset.targets["recording"].tolist() == [
        "sub-01_task-test",
        "sub-01_task-test",
        "sub-01_task-test",
        "sub-02_task-test",
        "sub-02_task-test",
        "sub-02_task-test",
    ]
    assert dataset.targets["rating"].tolist() == [3, 5, 4, 2, 1, 0]


def test_read_dataset_refuses_descriptor_event_that_disagrees_with_row_identity(tmp_path) -> None:
    path = tmp_path / "features.tsv"
    rows = pd.DataFrame({"event": ["wrong", "right", "left"]})
    write_table(_epoch_table(), path, rows=rows)

    with pytest.raises(ValueError, match="event.*row_ids"):
        io_module.read_dataset([path])


def test_read_dataset_accepts_event_names_that_look_numeric(tmp_path) -> None:
    # mne.Epochs(raw, events) names its events "1", "2", ...; a TSV reader parses those as
    # numbers, and "01" as the number 1, yet they are the same identities the rows carry.
    table = replace(_epoch_table(), row_ids=(("rec", 0, "1"), ("rec", 1, "2"), ("rec", 2, "01")))
    path = tmp_path / "features.tsv"
    write_table(table, path, rows=pd.DataFrame({"event": ["1", "2", "01"]}))

    dataset = io_module.read_dataset([path])

    assert dataset.targets["event"].tolist() == ["1", "2", "01"]


@pytest.mark.parametrize("dtype", [object, "string", "category"])
def test_read_dataset_preserves_text_descriptors_that_look_numeric(tmp_path, dtype) -> None:
    path = tmp_path / "features.tsv"
    rows = pd.DataFrame(
        {
            "subject_id": pd.Series(["01", "1", "02"], dtype=dtype),
            "rating": [0.25, 1.5, 3.0],
        }
    )
    write_table(_epoch_table(), path, rows=rows)

    dataset = io_module.read_dataset([path])

    assert dataset.targets["subject_id"].tolist() == ["01", "1", "02"]
    assert dataset.targets["rating"].tolist() == [0.25, 1.5, 3.0]


def test_read_dataset_refuses_epoch_key_that_disagrees_with_row_identity(tmp_path) -> None:
    path = tmp_path / "features.tsv"
    write_table(_epoch_table(), path)
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    frame.loc[0, "epoch"] = 99
    frame.to_csv(path, sep="\t", index=False)
    _resign(path)

    with pytest.raises(ValueError, match="epoch.*row_ids"):
        io_module.read_dataset([path])


def test_read_dataset_refuses_fractional_epoch_key_instead_of_truncating_it(tmp_path) -> None:
    path = tmp_path / "features.tsv"
    write_table(_epoch_table(), path)
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    frame["epoch"] = frame["epoch"].astype(float)
    frame.loc[0, "epoch"] = 0.5
    frame.to_csv(path, sep="\t", index=False)
    _resign(path)

    with pytest.raises(ValueError, match="epoch.*row_ids"):
        io_module.read_dataset([path])


def test_read_dataset_refuses_cross_trial_tables(tmp_path) -> None:
    path = tmp_path / "crosstrial.tsv"
    write_table(_group_table(), path)

    with pytest.raises(ValueError, match="per-epoch"):
        io_module.read_dataset([path])


def test_read_dataset_refuses_a_missing_descriptor_column(tmp_path) -> None:
    path = tmp_path / "features.tsv"
    write_table(_epoch_table(), path, rows=pd.DataFrame({"rating": [3, 5, 4]}))
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    frame.drop(columns="rating").to_csv(path, sep="\t", index=False)
    _resign(path)

    with pytest.raises(ValueError, match="descriptor columns.*rating"):
        io_module.read_dataset([path])


def test_read_dataset_of_nothing_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        io_module.read_dataset([])


def test_reordered_coverage_rows_are_rejected(tmp_path) -> None:
    path = tmp_path / "features.tsv"
    write_table(_epoch_table(), path)

    coverage_path = tmp_path / "features_coverage.tsv"
    frame = pd.read_csv(coverage_path, sep="\t", keep_default_na=False)
    frame.iloc[::-1].to_csv(coverage_path, sep="\t", index=False)
    _resign(path)

    with pytest.raises(ValueError, match="row identities"):
        read_table(path)


def test_infinite_window_bounds_round_trip(tmp_path) -> None:
    table = FeatureTable(
        values=np.array([[1.0]]),
        coverage=np.array([[1.0]]),
        meta=(_meta(window="all", window_bounds=(-np.inf, np.inf)),),
        row_ids=(("sub-01_task-test", 0, "left"),),
    )
    path = tmp_path / "sub-01_features.tsv"
    write_table(table, path)
    restored = read_table(path)
    _assert_same_table(restored, table)
    assert restored.meta[0].window_bounds == (-np.inf, np.inf)


def test_read_dataset_unions_columns_across_recordings_that_measured_different_channels(
    tmp_path,
) -> None:
    first = _epoch_table()
    second = FeatureTable(
        values=first.values + 10.0,
        coverage=first.coverage,
        meta=(first.meta[0], _meta(space="Pz")),
        row_ids=tuple(("sub-02_task-test", epoch, event) for _, epoch, event in first.row_ids),
    )
    paths = [tmp_path / "sub-01_features.tsv", tmp_path / "sub-02_features.tsv"]
    for path, table in zip(paths, (first, second), strict=True):
        rows = pd.DataFrame({"event": [event for _, _, event in table.row_ids]})
        write_table(table, path, rows=rows)

    dataset = io_module.read_dataset(paths)

    spaces = tuple(m.space for m in dataset.table.meta)
    assert spaces == (first.meta[0].space, first.meta[1].space, "Pz")
    values = dataset.table.values
    assert np.all(np.isnan(values[3:, 1]))
    assert np.all(np.isnan(values[:3, 2]))
    np.testing.assert_array_equal(values[3:, 2], second.values[:, 1])
    assert len(dataset.targets) == 6


def test_reading_a_dataset_hashes_each_column_once(tmp_path, monkeypatch) -> None:
    # A column's name embeds a SHA-256 of its whole metadata record. Reading a 13,000-column
    # bundle recomputed it five times per column, most of read_dataset's time.
    import eegtable.table as table_module

    path = tmp_path / "features.tsv"
    write_table(_epoch_table(), path, rows=pd.DataFrame({"event": ["left", "right", "left"]}))
    real = table_module.hashlib.sha256
    calls: list[int] = []

    def counting(data: bytes = b"") -> object:
        calls.append(1)
        return real(data)

    monkeypatch.setattr(table_module.hashlib, "sha256", counting)
    io_module.read_dataset([path])
    assert len(calls) == len(_epoch_table().meta) + 2  # Two serialized payload checksums.


def test_a_macos_resource_file_is_named_instead_of_failing_to_decode(tmp_path) -> None:
    # macOS writes a "._" twin beside every file on an exFAT drive, and a glob picks it up.
    real = tmp_path / "sub-01_features.tsv"
    write_table(_epoch_table(), real, rows=pd.DataFrame({"event": ["left", "right", "left"]}))
    junk = b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        \xff\xfe"
    for suffix in (".tsv", ".json"):
        (tmp_path / f"._sub-01_features{suffix}").write_bytes(junk)
    with pytest.raises(ValueError, match=r"\._sub-01_features\.json.*AppleDouble"):
        io_module.read_dataset([real, tmp_path / "._sub-01_features.tsv"])


def test_a_sidecar_that_is_not_json_is_named(tmp_path) -> None:
    path = tmp_path / "features.tsv"
    write_table(_epoch_table(), path, rows=pd.DataFrame({"event": ["left", "right", "left"]}))
    path.with_suffix(".json").write_text("{ not json")
    with pytest.raises(ValueError, match=r"features\.json"):
        read_table(path)


def _as_written_by_eegfeat(path: Path) -> None:
    """Turn a bundle into one eegfeat wrote: no schema, no manifests, the old names."""
    for file in (path, path.with_name(f"{path.stem}_coverage.tsv")):
        file.write_text(file.read_text().replace("__eegtable_row_id", "__eegfeat_row_id", 1))
    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    for key in ("schema", "files", "row_text_columns"):
        del sidecar[key]
    sidecar["eegfeat_version"] = sidecar.pop("eegtable_version")
    sidecar_path.write_text(json.dumps(sidecar))


def test_a_bundle_written_by_eegfeat_reads_back_unchanged(tmp_path) -> None:
    # Event names that look numeric stay text, as eegfeat's reader kept them.
    table = replace(_epoch_table(), row_ids=(("rec", 0, "1"), ("rec", 1, "2"), ("rec", 2, "01")))
    path = tmp_path / "sub-01_features.tsv"
    write_table(table, path, rows=pd.DataFrame({"event": ["1", "2", "01"], "rating": [3, 5, 4]}))
    _as_written_by_eegfeat(path)

    _assert_same_table(read_table(path), table)
    dataset = io_module.read_dataset([path])
    np.testing.assert_array_equal(dataset.table.values, table.values)
    assert dataset.targets["event"].tolist() == ["1", "2", "01"]
    assert dataset.targets["rating"].tolist() == [3, 5, 4]


def test_a_bundle_written_by_eegfeat_still_needs_its_coverage_file(tmp_path) -> None:
    path = tmp_path / "sub-01_features.tsv"
    write_table(_epoch_table(), path, rows=pd.DataFrame({"event": ["left", "right", "left"]}))
    _as_written_by_eegfeat(path)
    path.with_name("sub-01_features_coverage.tsv").unlink()

    with pytest.raises(ValueError, match="legacy feature bundle file manifest"):
        read_table(path)
