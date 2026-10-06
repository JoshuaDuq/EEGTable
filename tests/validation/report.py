"""Write the validation results the docs render.

Every validation test declares what it validates with the ``validates`` marker,
and may record an observed value through the ``record`` fixture. Each run is
archived independently. Its observed rows replace ``docs/validation/results.json``
and three reStructuredText fragments included by the validation guide, so a
partial run cannot inherit claims from earlier evidence.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import eegtable
from eegtable.provenance import file_hash, identity, implementation_hash, software_versions

KINDS = ("formula", "estimator", "physiology", "decoding", "behaviour")
KIND_TITLES = {
    "formula": "Formula",
    "estimator": "Estimator",
    "physiology": "Physiology",
    "decoding": "Decoding",
    "behaviour": "Behaviour",
}
KIND_MEANINGS = {
    "formula": "equals an independent NumPy, SciPy or closed-form computation on the same data",
    "estimator": "agrees with a different estimator or library of the same quantity",
    "physiology": "recovers an effect the literature describes, at the level it is known to hold",
    "decoding": "a leakage-safe model finds a contrast strong enough that a correct pipeline must",
    "behaviour": "a documented refusal, identity, null rate or command-line behaviour holds",
}

# The modeling and runner entry points the scorecard reports on; the rest of
# eegtable.model are helpers reached through these.
MODEL_API = (
    "build_design",
    "within_subject_folds",
    "loso_folds",
    "cross_fit_classification",
    "cross_fit_regression",
    "classification_metrics",
    "regression_metrics",
    "subject_level_r",
    "permutation_test",
    "permutation_importance_over_folds",
    "prediction_intervals",
)
TOOLING = (
    "Spectra.from_spectrum",
    "Spectra.from_tfr",
    "Signal.from_epochs",
    "BandSignal.from_epochs",
    "stack_rows",
    "runner",
    "eegtable command",
    "preprocessing",
)

DATASET_TITLES = {
    "eegbci": "PhysioNet motor movement",
    "ssvep": "MNE SSVEP",
    "erp_core": "ERP CORE Flankers",
    "sleep": "Sleep-EDF",
}


@dataclass(frozen=True)
class Row:
    nodeid: str
    measures: tuple[str, ...]
    kind: str
    dataset: str
    claim: str
    criterion: str
    observed: str
    passed: bool
    run: str
    cases: int = 1


def feature_api() -> tuple[str, ...]:
    return tuple(
        name
        for name in eegtable.__all__
        if name[0].islower() and callable(getattr(eegtable, name)) and name != "concat"
    )


def validation_hash() -> str:
    """Identify the exact validation checks, loaders, and reporting code."""
    root = Path(__file__).parent
    return identity(
        {path.relative_to(root).as_posix(): file_hash(path) for path in sorted(root.rglob("*.py"))}
    )


def write(
    rows: Iterable[Row],
    root: Path,
    *,
    code_sha256: str | None = None,
    validation_sha256: str | None = None,
) -> None:
    """Archive this run independently and render only its observed claims."""
    code_hash = implementation_hash()
    if code_sha256 is not None and code_hash != code_sha256:
        raise ValueError(
            "Validation implementation changed during the run; evidence not published."
        )
    checks_hash = validation_hash()
    if validation_sha256 is not None and checks_hash != validation_sha256:
        raise ValueError("Validation checks changed during the run; evidence not published.")
    cases = list(rows)
    fresh = _collapse(cases)
    root.mkdir(parents=True, exist_ok=True)
    results = root / "results.json"
    merged = sorted((asdict(row) for row in fresh), key=lambda r: (r["dataset"], r["nodeid"]))
    stamp = datetime.now(UTC)
    run_id = stamp.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + code_hash[:12]
    payload = {
        "generated": stamp.isoformat(),
        "run_id": run_id,
        "code_sha256": code_hash,
        "validation_sha256": checks_hash,
        "checks": {"passed": sum(row.passed for row in cases), "total": len(cases)},
        "versions": software_versions(),
        "rows": merged,
    }
    archive = root / "runs" / run_id
    archive.mkdir(parents=True)
    text = json.dumps(payload, indent=1) + "\n"
    (archive / "results.json").write_text(text, encoding="utf-8")
    results.write_text(text, encoding="utf-8")
    (root / "summary.inc").write_text(_summary(payload), encoding="utf-8")
    (root / "scorecard.inc").write_text(_scorecard(merged), encoding="utf-8")
    (root / "results.inc").write_text(_results(merged), encoding="utf-8")


def _collapse(rows: Iterable[Row]) -> list[Row]:
    """One row per test function: parametrized cases share a claim and pool their values."""
    grouped: dict[str, list[Row]] = {}
    for row in rows:
        if row.kind not in KINDS:
            raise ValueError(f"Unknown validation kind {row.kind!r}; expected one of {KINDS}.")
        grouped.setdefault(row.nodeid.split("[")[0], []).append(row)
    out = []
    for nodeid, cases in grouped.items():
        first = cases[0]
        observed = "; ".join(case.observed for case in cases if case.observed)
        out.append(
            Row(
                nodeid=nodeid,
                measures=first.measures,
                kind=first.kind,
                dataset=first.dataset,
                claim=first.claim,
                criterion=first.criterion,
                observed=observed,
                passed=all(case.passed for case in cases),
                run=max(case.run for case in cases),
                cases=len(cases),
            )
        )
    return out


def _summary(payload: Mapping[str, Any]) -> str:
    rows = payload["rows"]
    checks = payload["checks"]["total"]
    passed_checks = payload["checks"]["passed"]
    passed = sum(r["passed"] for r in rows)
    datasets = sorted({r["dataset"] for r in rows if r["dataset"]})
    measures = {m for r in rows for m in r["measures"]}
    v = payload["versions"]
    sklearn_version = v.get("scikit-learn", "not installed")
    return (
        f"**{passed} of {len(rows)} claims hold, {passed_checks} of {checks} checks pass** "
        f"across {len(datasets)} public datasets, covering {len(measures)} public functions. "
        f"Last run {payload['generated'][:10]} on "
        f"Python {v['python']}, MNE {v['mne']}, NumPy {v['numpy']}, SciPy {v['scipy']}, "
        f"scikit-learn {sklearn_version}, eegtable {v['eegtable']}.\n"
    )


def _cell(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "–"
    failed = [r for r in rows if not r["passed"]]
    if failed:
        return f"✗ {len(failed)} of {len(rows)}"
    return f"✓ {len(rows)}"


def _scorecard(rows: list[dict[str, Any]]) -> str:
    by_measure: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for measure in row["measures"]:
            by_measure.setdefault(measure, []).append(row)

    groups = (
        ("Features", feature_api()),
        ("Modeling", MODEL_API),
        ("Containers, runner and command", TOOLING),
    )
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 35 13 13 13 13 13",
        "",
        "   * - Function",
        *(f"     - {KIND_TITLES[kind]}" for kind in KINDS),
    ]
    untested: list[str] = []
    for title, names in groups:
        lines += [f"   * - **{title}**", *("     - " for _ in KINDS)]
        for name in names:
            tested = by_measure.get(name, [])
            if not tested:
                untested.append(name)
                continue
            lines.append(f"   * - ``{name}``")
            lines += [f"     - {_cell([r for r in tested if r['kind'] == kind])}" for kind in KINDS]
    text = "\n".join(lines) + "\n"
    if untested:
        listed = ", ".join(f"``{name}``" for name in untested)
        text += f"\n**Covered by unit tests only, not yet by a real-data check:** {listed}.\n"
    return text


def _escape(text: str) -> str:
    return text.replace("*", "\\*").replace("|", "\\|")


def _results(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for dataset in sorted({r["dataset"] for r in rows}):
        title = DATASET_TITLES.get(dataset, dataset or "Several datasets")
        lines += [title, "^" * len(title), ""]
        lines += [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 30 26 30 14",
            "",
            "   * - Claim",
            "     - Criterion",
            "     - Observed",
            "     - Result",
        ]
        for row in [r for r in rows if r["dataset"] == dataset]:
            functions = ", ".join(f"``{m}``" for m in row["measures"])
            verdict = "pass" if row["passed"] else "**fail**"
            lines += [
                f"   * - {_escape(row['claim'])} ({functions})",
                f"     - {_escape(row['criterion'])}",
                f"     - {_escape(row['observed']) or '–'}",
                f"     - {verdict}",
            ]
        lines.append("")
    return "\n".join(lines)
