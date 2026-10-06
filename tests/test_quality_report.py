from pathlib import Path

import pandas as pd
import pytest

from tests.test_io import _epoch_table


def test_quality_report_writes_cohort_feature_and_definition_evidence(tmp_path):
    from eegtable.report import write_quality_report

    table = _epoch_table()
    descriptors = pd.DataFrame({"condition": ["rest", "task", "rest"]})
    path = tmp_path / "quality.html"
    result = write_quality_report(table, descriptors, path, by=("condition",))
    assert Path(result) == path
    html = path.read_text(encoding="utf-8")
    assert "Cohort quality" in html
    assert "Feature quality" in html
    assert "Feature definitions" in html
    assert "code_sha256" in html


def _linked_bundle(tmp_path):
    from dataclasses import replace

    from eegtable.io import write_table
    from eegtable.provenance import identity

    provenance = {
        "retained": 3,
        "original_events": 5,
        "artifact": {"method": "ica", "evidence": {"suggested_exclude": [0, 2]}},
        "artifact_decision": {"exclude": [2], "apply": True, "fit_id": "reviewed"},
        "epoch_decision": {"exclude": [1, 3]},
    }
    path = tmp_path / "features.tsv"
    table = replace(
        _epoch_table(),
        row_ids=tuple(
            ("sub-01/eeg/rest_epo.fif", epoch, event) for _, epoch, event in _epoch_table().row_ids
        ),
    )
    write_table(
        table,
        path,
        provenance={
            "recording": "sub-01/eeg/rest_epo.fif",
            "n_epochs": 3,
            "upstream": {"identity": identity(provenance), "provenance": provenance},
        },
    )
    return path


def test_recording_quality_restores_observed_retention_and_artifact_decisions(tmp_path):
    import eegtable.report as reporting

    path = _linked_bundle(tmp_path)
    summary = reporting.recording_quality([path])
    assert summary.recording.tolist() == ["sub-01/eeg/rest_epo.fif"]
    assert summary.input_epochs.tolist() == [3]
    assert summary.upstream_original_events.tolist() == [5]
    assert summary.upstream_retained.tolist() == [3]
    assert summary.upstream_rejected.tolist() == [2]
    assert summary.proposed_artifact_exclusions.tolist() == [2]
    assert summary.artifact_method.tolist() == ["ica"]
    assert '"exclude": [2]' in summary.artifact_decision.item()
    assert '"exclude": [1, 3]' in summary.epoch_decision.item()


def test_recording_quality_marks_unavailable_upstream_evidence_missing(tmp_path):
    import eegtable.report as reporting
    from eegtable.io import write_table

    path = tmp_path / "features.tsv"
    write_table(
        _epoch_table(),
        path,
        provenance={"recording": "sub-01_task-test", "n_epochs": 3, "upstream": None},
    )
    summary = reporting.recording_quality([path])
    assert summary.input_epochs.item() == 3
    assert pd.isna(summary.upstream_retained.item())
    assert pd.isna(summary.artifact_decision.item())


def test_quality_report_includes_recording_retention_without_mutating_evidence(tmp_path):
    import numpy as np

    import eegtable.report as reporting
    from eegtable.quality import QualityPolicy

    table = _epoch_table()
    values, coverage = table.values.copy(), table.coverage.copy()
    summary = reporting.recording_quality([_linked_bundle(tmp_path)])
    path = tmp_path / "quality.html"
    reporting.write_quality_report(
        table,
        pd.DataFrame({"condition": ["rest", "task", "rest"]}),
        path,
        by=("condition",),
        recording_summary=summary,
        quality=QualityPolicy(0.8),
    )
    text = path.read_text(encoding="utf-8")
    assert "Recording and preprocessing quality" in text
    assert "upstream_original_events" in text
    assert "artifact_decision" in text
    np.testing.assert_array_equal(table.values, values)
    np.testing.assert_array_equal(table.coverage, coverage)


def test_recording_quality_checks_bundle_payloads_and_saved_upstream_identity(tmp_path):
    import json

    from eegtable.report import recording_quality

    path = _linked_bundle(tmp_path)
    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["provenance"]["upstream"]["provenance"]["retained"] = 1
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        recording_quality([path])
    path = _linked_bundle(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        recording_quality([path])


def test_recording_quality_deduplicates_consistent_evidence_and_rejects_conflicts(tmp_path):
    import json

    from eegtable.report import recording_quality

    path = _linked_bundle(tmp_path)
    assert len(recording_quality([path, path])) == 1
    other = tmp_path / "other"
    other.mkdir()
    alternate = _linked_bundle(other)
    sidecar_path = alternate.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["provenance"]["n_epochs"] = 4
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting"):
        recording_quality([path, alternate])


def test_report_cli_includes_saved_recording_evidence(tmp_path):
    from eegtable.runner.cli import main
    from tests.test_provenance import _extract

    recipe, _, _ = _extract(tmp_path)
    path = tmp_path / "quality.html"
    assert main(["report", str(recipe.path), str(path), "--by", "event"]) == 0
    assert "Recording and preprocessing quality" in path.read_text(encoding="utf-8")
