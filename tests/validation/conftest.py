"""Opt-in validation against public MNE datasets.

These tests download real recordings from PhysioNet and the MNE servers and
take minutes to run, so they are skipped unless ``EEGTABLE_DATASETS=1`` is set.
Data lands wherever MNE keeps its datasets (``MNE_DATA``, default ``~/mne_data``)
and is reused on later runs.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path

import pytest

from eegtable.provenance import implementation_hash
from tests.validation import report
from tests.validation.loaders import Recording, load_eegbci, load_sleep, load_ssvep

ENABLE = "EEGTABLE_DATASETS"

EEGBCI_SUBJECTS = tuple(range(1, 21))
SLEEP_SUBJECTS = (0, 1)

# Optional extras: eegtable.model and the microstates need scikit-learn, the
# spectral connectivity estimators need mne-connectivity.
collect_ignore = []
if find_spec("sklearn") is None:
    collect_ignore += [
        "test_decoding.py",
        "test_microstates.py",
        "test_csp.py",
        "test_regression.py",
        "test_model_extras.py",
    ]
if find_spec("mne_connectivity") is None:
    collect_ignore += ["test_connectivity.py"]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "validates(*measures, kind, claim, criterion, dataset=None): what a validation test "
        "establishes, rendered into the docs scorecard",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    if os.environ.get(ENABLE) == "1":
        return
    skip = pytest.mark.skip(reason=f"set {ENABLE}=1 to download and run the dataset checks")
    here = os.path.dirname(__file__)
    for item in items:
        if os.path.commonpath([str(item.path), here]) == here:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def quiet_mne() -> Iterator[None]:
    import mne

    previous = mne.set_log_level("ERROR", return_old_level=True)
    yield
    mne.set_log_level(previous)


@pytest.fixture(scope="session")
def eegbci_recordings() -> list[Recording]:
    return [load_eegbci(subject) for subject in EEGBCI_SUBJECTS]


@pytest.fixture(scope="session")
def sleep_recordings() -> list[Recording]:
    return [load_sleep(subject) for subject in SLEEP_SUBJECTS]


@pytest.fixture(scope="session")
def ssvep_recording() -> Recording:
    return load_ssvep()


# --- results the docs render -----------------------------------------------------------

_OBSERVED: dict[str, str] = {}
_ROWS: dict[str, report.Row] = {}
_CODE_SHA256 = implementation_hash()
_VALIDATION_SHA256 = report.validation_hash()


@pytest.fixture
def record(request: pytest.FixtureRequest) -> Callable[[str], None]:
    """Store the observed value a test is about to assert on, for the results table."""

    def _record(observed: str) -> None:
        _OBSERVED[request.node.nodeid] = observed

    return _record


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Iterator[None]:
    outcome = yield
    marker = item.get_closest_marker("validates")
    if marker is None:
        return
    result = outcome.get_result()
    if result.skipped:
        return
    if call.when != "call" and result.passed:
        return
    previous = _ROWS.get(item.nodeid)
    observations = [previous.observed] if previous is not None and previous.observed else []
    observed = _OBSERVED.pop(item.nodeid, "")
    if observed:
        observations.append(observed)
    if call.excinfo is not None:
        message = " ".join(str(call.excinfo.value).split())
        observations.append(f"{call.when}: {call.excinfo.typename}: {message}")
    _ROWS[item.nodeid] = report.Row(
        nodeid=item.nodeid,
        measures=tuple(marker.args),
        kind=marker.kwargs["kind"],
        dataset=str(marker.kwargs.get("dataset") or getattr(item.module, "DATASET", "")),
        claim=marker.kwargs["claim"],
        criterion=marker.kwargs["criterion"],
        observed="; ".join(observations),
        passed=result.passed and (previous is None or previous.passed),
        run=datetime.now(UTC).replace(microsecond=0).isoformat(),
    )


def pytest_sessionfinish(session: pytest.Session) -> None:
    del session
    if not _ROWS:
        return
    report.write(
        _ROWS.values(),
        Path(__file__).parents[2] / "docs" / "validation",
        code_sha256=_CODE_SHA256,
        validation_sha256=_VALIDATION_SHA256,
    )
