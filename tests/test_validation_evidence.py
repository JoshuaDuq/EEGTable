import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from eegtable.provenance import software_versions
from tests.validation.report import Row, write


def test_validation_runs_are_immutable_and_do_not_merge_old_claims(tmp_path):
    old = Row("old::claim", ("variance",), "formula", "", "old", "exact", "1", True, "old")
    new = Row("new::claim", ("variance",), "formula", "", "new", "exact", "2", True, "new")
    write([old], tmp_path)
    snapshots = list((tmp_path / "runs").glob("*/results.json"))
    assert len(snapshots) == 1
    original = snapshots[0].read_bytes()
    write([new], tmp_path)
    current = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert [row["nodeid"] for row in current["rows"]] == ["new::claim"]
    assert len(list((tmp_path / "runs").glob("*/results.json"))) == 2
    assert snapshots[0].read_bytes() == original
    assert len(current["code_sha256"]) == 64
    assert len(current["validation_sha256"]) == 64
    assert current["run_id"]


def test_validation_refuses_evidence_from_a_changed_implementation(tmp_path):
    row = Row("claim", ("variance",), "formula", "", "claim", "exact", "1", True, "now")
    with pytest.raises(ValueError, match="implementation changed"):
        write([row], tmp_path, code_sha256="0" * 64)
    assert not (tmp_path / "results.json").exists()


@pytest.mark.parametrize("phase", ["setup", "call", "teardown"])
def test_validation_records_failures_in_every_test_phase(tmp_path, phase):
    suite = tmp_path / "tests" / "validation"
    suite.mkdir(parents=True)
    source = Path(__file__).parent / "validation" / "conftest.py"
    (suite / "conftest.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (suite / "test_failure.py").write_text(
        "import pytest\n"
        f"PHASE = {phase!r}\n"
        "pytestmark = pytest.mark.validates(\n"
        "    'mean_psd', kind='formula', dataset='ssvep',\n"
        "    claim='fixture regression', criterion='completes successfully')\n"
        "@pytest.fixture\n"
        "def recording():\n"
        "    if PHASE == 'setup':\n"
        "        raise RuntimeError('recording unavailable')\n"
        "    yield\n"
        "    if PHASE == 'teardown':\n"
        "        raise RuntimeError('cleanup failed')\n"
        "def test_passes():\n"
        "    assert True\n"
        "def test_fails(recording):\n"
        "    if PHASE == 'call':\n"
        "        raise RuntimeError('comparison failed')\n",
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "EEGTABLE_DATASETS": "1",
        "MNE_DONTWRITE_HOME": "true",
        "PYTHONPATH": str(Path(__file__).parents[1]),
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(suite), "-q"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    payload = json.loads(
        (tmp_path / "docs" / "validation" / "results.json").read_text(encoding="utf-8")
    )
    rows = {row["nodeid"].rsplit("::", 1)[1]: row for row in payload["rows"]}
    assert set(rows) == {"test_passes", "test_fails"}
    assert rows["test_passes"]["passed"]
    assert not rows["test_fails"]["passed"]
    assert phase in rows["test_fails"]["observed"]


def test_validation_rejects_unknown_evidence_categories(tmp_path):
    row = Row("claim", ("mean_psd",), "unknown", "ssvep", "claim", "exact", "1", True, "now")
    with pytest.raises(ValueError, match="kind"):
        write([row], tmp_path)
    assert not (tmp_path / "results.json").exists()


def test_preprocessing_comparisons_appear_in_the_scorecard(tmp_path):
    row = Row(
        "claim",
        ("preprocessing",),
        "estimator",
        "erp_core",
        "matches MNE",
        "exact",
        "0",
        True,
        "now",
    )
    write([row], tmp_path)
    assert "``preprocessing``" in (tmp_path / "scorecard.inc").read_text(encoding="utf-8")


def test_validation_summary_counts_individual_parameterized_outcomes(tmp_path):
    passed = Row("claim[a]", ("mean_psd",), "formula", "ssvep", "claim", "exact", "1", True, "now")
    failed = Row("claim[b]", ("mean_psd",), "formula", "ssvep", "claim", "exact", "2", False, "now")
    write([passed, failed], tmp_path)
    summary = (tmp_path / "summary.inc").read_text(encoding="utf-8")
    assert "0 of 1 claims hold, 1 of 2 checks pass" in summary


def test_validation_refuses_evidence_from_changed_checks(tmp_path):
    row = Row("claim", ("mean_psd",), "formula", "ssvep", "claim", "exact", "1", True, "now")
    with pytest.raises(ValueError, match="checks changed"):
        write([row], tmp_path, validation_sha256="0" * 64)
    assert not (tmp_path / "results.json").exists()


def test_validation_records_all_installed_backend_versions(tmp_path):
    row = Row(
        "claim", ("preprocessing",), "estimator", "erp_core", "claim", "exact", "1", True, "now"
    )
    write([row], tmp_path)
    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert payload["versions"] == software_versions()
