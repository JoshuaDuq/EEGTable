"""Headless MNE quality reports without modifying numerical results."""

from __future__ import annotations

from html import escape
from typing import Any

import mne  # type: ignore[import-untyped]

from .pipeline import PreprocessingResult, StageData, result_from_state
from .provenance import canonical_json


def _provenance(report: Any, provenance: Any) -> None:
    report.add_html("<pre>" + escape(canonical_json(provenance)) + "</pre>", title="Provenance")


def build_report(result: PreprocessingResult) -> Any:
    """MNE report of a preprocessing result: provenance, ledgers and final epochs.

    Parameters
    ----------
    result : PreprocessingResult
        From :func:`~eegtable.preprocessing.preprocess` or a completed workflow.

    Returns
    -------
    mne.Report
        Unsaved; the result is not modified.
    """
    report = mne.Report(title="EEG preprocessing quality control")
    _provenance(report, result.provenance)
    report.add_html(result.events.to_html(index=False), title="Event and rejection ledger")
    if result.repairs is not None:
        report.add_html(result.repairs.to_html(index=False), title="Per-epoch channel repairs")
    report.add_epochs(result.epochs.copy(), title="Final epochs", psd=True)
    return report


def build_artifact_report(model: Any, raw: Any) -> Any:
    report = mne.Report(title=f"{model.method.upper()} artifact review")
    report.add_html(
        "<pre>" + escape(canonical_json(model.evidence)) + "</pre>", title="Training and scores"
    )
    if model.method == "ica":
        # Every component; Report.add_ica shows only the first 20 by default.
        report.add_ica(
            model.model,
            title="All fitted ICA components",
            inst=raw.copy(),
            picks=list(range(model.model.n_components_)),
        )
    elif model.method == "ssp":
        report.add_projs(info=raw.info, projs=model.model, title="All candidate EEG projectors")
    else:
        report.add_html(
            "<pre>" + escape(str(model.model.coef_)) + "</pre>", title="Regression coefficients"
        )
    return report


def build_checkpoint_report(state: StageData) -> Any:
    """MNE report of what a checkpoint holds.

    Epochs are reported as by :func:`build_report`; a fitted artifact operator
    with its evidence and all of its components, projectors or coefficients;
    otherwise the continuous data with its provenance.

    Parameters
    ----------
    state : StageData
        A checkpoint's ``state``.

    Returns
    -------
    mne.Report
        Unsaved.
    """
    # What the checkpoint holds: epochs once they exist, a fitted operator before, else raw.
    if state.epochs is not None:
        return build_report(result_from_state(state))
    if state.artifact is not None:
        return build_artifact_report(state.artifact, state.raw)
    report = mne.Report(title="Continuous data")
    _provenance(report, state.provenance)
    report.add_raw(state.raw.copy(), title="Raw", psd=True)
    return report
