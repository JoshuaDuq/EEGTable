"""MNE example datasets: every preprocessing stage equals the direct MNE call.

ERP CORE's Flankers recording (30 EEG, 3 EOG with real blinks, 1024 Hz, montage)
runs through the checkpointed workflow with a bipolar VEOG, notch, band-pass,
seeded ICA reviewed on its EOG scores, event epochs, threshold rejection,
average reference, decimation, and baseline. Each checkpoint is compared with
the same MNE operation applied by hand to its parent checkpoint, so a hidden
operation anywhere in the chain would show. An EEGBCI eyes-open run covers the
automated tools: PyPREP candidates, autoreject, and spline interpolation.
Equivalence says nothing about whether these recipes are good ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
from mne.datasets import eegbci, erp_core

from eegtable.preprocessing import load_config, open_workflow, read_checkpoint, run_until
from eegtable.preprocessing.config import read_yaml
from eegtable.preprocessing.execution import Workflow
from eegtable.preprocessing.review import save_review

ERP_FILE = "ERP-CORE_Subject-001_Task-Flankers_eeg.fif"
ERP_EVENTS = {
    "stimulus/compatible/target_left": 1,
    "stimulus/compatible/target_right": 2,
    "stimulus/incompatible/target_left": 3,
    "stimulus/incompatible/target_right": 4,
}
PHYSIOLOGY = ("edge", "bad")
ATOL = 1e-15


def _yaml_events() -> str:
    return ", ".join(f'"{name}": {code}' for name, code in ERP_EVENTS.items())


@pytest.fixture(scope="module")
def erp_workflow(tmp_path_factory: pytest.TempPathFactory) -> Workflow:
    root = tmp_path_factory.mktemp("erp_core")
    source = mne.io.read_raw_fif(erp_core.data_path(update_path=False) / ERP_FILE, preload=True)
    source.crop(0, 240).save(root / "sub-001_raw.fif", fmt="double")
    config = root / "preprocessing.yaml"
    config.write_text(
        "input: {path: sub-001_raw.fif}\n"
        "output: {directory: preprocessed, name: sub-001}\n"
        "workflow: {raw_review: disabled}\n"
        "channels:\n"
        "  bipolar: [{name: VEOG, anode: VEOG_lower, cathode: FP2, type: eog}]\n"
        "crop: {tmin: 10.0, tmax: 230.0}\n"
        "filter: {l_freq: 0.1, h_freq: 30.0, notch_freqs: [60.0]}\n"
        "artifact:\n"
        "  method: ica\n"
        "  ica: {n_components: 15, eog_channels: [VEOG], reject: {eeg: 0.0005}}\n"
        "epochs:\n"
        f"  kind: events\n  events: {{source: annotations, event_id: {{{_yaml_events()}}}}}\n"
        "  tmin: -0.2\n  tmax: 0.8\n  baseline: [-0.2, 0.0]\n"
        "rejection: {method: thresholds, reject: {eeg: 0.00015}}\n"
        "reference: {channels: average}\n"
        "sampling: {method: decimate, factor: 4}\n"
    )
    workflow = open_workflow(load_config(config))
    pending = run_until(workflow)
    assert pending.state == "needs-review"
    template = read_yaml(pending.steps[-1].path)
    # Review chooses the component with the strongest VEOG score; a real review would look.
    scores = np.asarray(
        read_checkpoint(workflow, "fit-artifact").state.artifact.evidence["scores"]["VEOG"]
    )
    template["exclude"] = [int(np.argmax(np.abs(scores)))]
    decision = root / "artifact.yaml"
    decision.write_text(json.dumps(template))
    save_review(workflow, "artifact", decision)
    assert run_until(workflow).state == "completed"
    return workflow


def _raw(workflow: Workflow, stage: str):
    return read_checkpoint(workflow, stage).state.raw


def _epochs(workflow: Workflow, stage: str):
    return read_checkpoint(workflow, stage).state.epochs


def _same_epochs(actual, expected) -> None:
    np.testing.assert_array_equal(actual.events, expected.events)
    assert actual.drop_log == expected.drop_log
    np.testing.assert_allclose(actual.get_data(), expected.get_data(), rtol=0, atol=ATOL)


@pytest.mark.validates(
    "preprocessing",
    kind="estimator",
    dataset="erp_core",
    claim="prepare and crop-raw are set_bipolar_reference and Raw.crop",
    criterion="identical samples, first_samp, and channel order",
)
def test_prepare_and_crop_match_mne(erp_workflow: Workflow) -> None:
    loaded, prepared, cropped = (_raw(erp_workflow, s) for s in ("load", "prepare", "crop-raw"))
    expected = mne.set_bipolar_reference(
        loaded, "VEOG_lower", "FP2", ch_name="VEOG", drop_refs=False
    ).set_channel_types({"VEOG": "eog"})
    assert prepared.ch_names == expected.ch_names
    np.testing.assert_allclose(prepared.get_data(), expected.get_data(), rtol=0, atol=ATOL)
    expected = prepared.copy().crop(10.0, 230.0)
    assert cropped.first_samp == expected.first_samp == 10 * 1024
    np.testing.assert_allclose(cropped.get_data(), expected.get_data(), rtol=0, atol=ATOL)


@pytest.mark.validates(
    "preprocessing",
    kind="estimator",
    dataset="erp_core",
    claim="notch and filter are Raw.notch_filter and Raw.filter on physiology channels only",
    criterion="samples within 1e-15 V; 60 Hz bin reduced, 20 Hz bin within 0.5%",
)
def test_notch_and_filter_match_mne(erp_workflow: Workflow) -> None:
    cropped, notched, filtered = (_raw(erp_workflow, s) for s in ("crop-raw", "notch", "filter"))
    picks = [
        n for n, k in zip(cropped.ch_names, cropped.get_channel_types(), strict=True) if k != "stim"
    ]
    expected = cropped.copy().notch_filter([60.0], picks=picks, skip_by_annotation=PHYSIOLOGY)
    np.testing.assert_allclose(notched.get_data(), expected.get_data(), rtol=0, atol=ATOL)
    expected = notched.copy().filter(0.1, 30.0, picks=picks, skip_by_annotation=PHYSIOLOGY)
    np.testing.assert_allclose(filtered.get_data(), expected.get_data(), rtol=0, atol=ATOL)

    def bin_power(inst, frequency: float) -> float:
        spectrum = inst.compute_psd(picks=["C3"], tmin=20, tmax=200, n_fft=8192)
        return float(spectrum.get_data()[0, np.argmin(np.abs(spectrum.freqs - frequency))])

    # This recording carries little line noise; the notch must still only touch 60 Hz.
    assert bin_power(notched, 60.0) < bin_power(cropped, 60.0)
    assert abs(bin_power(notched, 20.0) / bin_power(cropped, 20.0) - 1) < 0.005


@pytest.mark.validates(
    "preprocessing",
    kind="estimator",
    dataset="erp_core",
    claim="epoch is mne.Epochs on the acquisition grid with baseline, proj, and decim off",
    criterion="identical events, drop log, and samples; events past the crop dropped, not shifted",
)
def test_epochs_match_mne(erp_workflow: Workflow) -> None:
    filtered = _raw(erp_workflow, "filter")
    events = read_checkpoint(erp_workflow, "events").state.events
    expected = mne.Epochs(
        filtered,
        events.events,
        ERP_EVENTS,
        tmin=-0.2,
        tmax=0.8,
        baseline=None,
        proj=False,
        preload=True,
        picks=filtered.ch_names,
        reject_by_annotation=True,
        on_missing="ignore",
    )
    _same_epochs(_epochs(erp_workflow, "epoch"), expected)
    ledger = pd.read_csv(erp_workflow.config.output.directory / "sub-001_events.tsv", sep="\t")
    assert len(ledger) == len(events.events)
    np.testing.assert_array_equal(ledger["original_sample"], events.original_samples)
    assert (~ledger["retained"]).sum() > 0


@pytest.mark.validates(
    "preprocessing",
    kind="estimator",
    dataset="erp_core",
    claim="apply-artifact is ICA.apply of the saved fit with the reviewed exclusion",
    criterion="samples within 1e-15 V; mean |corr(FP1, VEOG)| across epochs down by a quarter",
)
def test_reviewed_ica_matches_mne_and_removes_blinks(erp_workflow: Workflow) -> None:
    fit = read_checkpoint(erp_workflow, "fit-artifact").state.artifact
    decision = read_checkpoint(erp_workflow, "review-artifact").state.reviewed.decision
    before, after = _epochs(erp_workflow, "epoch"), _epochs(erp_workflow, "apply-artifact")
    expected = fit.model.copy().apply(before.copy(), exclude=decision["exclude"])
    np.testing.assert_allclose(after.get_data(), expected.get_data(), rtol=0, atol=ATOL)
    assert abs(fit.evidence["scores"]["VEOG"][decision["exclude"][0]]) > 0.5

    def blink_coupling(epochs) -> float:
        fp1, veog = epochs.get_data(picks=["FP1"])[:, 0], epochs.get_data(picks=["VEOG"])[:, 0]
        return float(
            np.mean([abs(np.corrcoef(a, b)[0, 1]) for a, b in zip(fp1, veog, strict=True)])
        )

    assert blink_coupling(after) < 0.75 * blink_coupling(before)
    np.testing.assert_array_equal(after.get_data(picks=["VEOG"]), before.get_data(picks=["VEOG"]))


@pytest.mark.validates(
    "preprocessing",
    kind="estimator",
    dataset="erp_core",
    claim="reject, reference, resample, crop-epochs, baseline are drop_bad, set_eeg_reference, "
    "decimate, crop, apply_baseline",
    criterion="identical events and drop log; samples within 1e-15 V",
)
def test_epoch_stages_match_mne(erp_workflow: Workflow) -> None:
    chain = (
        ("reject", lambda e: e.drop_bad(reject={"eeg": 150e-6})),
        ("reference", lambda e: e.set_eeg_reference("average", projection=False)),
        ("resample", lambda e: e.decimate(4)),
        ("crop-epochs", lambda e: e.crop(-0.2, 0.8)),
        ("baseline", lambda e: e.apply_baseline((-0.2, 0.0))),
    )
    previous = _epochs(erp_workflow, "apply-artifact")
    for stage, operation in chain:
        actual = _epochs(erp_workflow, stage)
        _same_epochs(actual, operation(previous.copy()))
        previous = actual
    assert previous.info["sfreq"] == 256.0
    exported = mne.read_epochs(
        erp_workflow.config.output.directory / "sub-001_epo.fif", preload=True, proj=False
    )
    _same_epochs(exported, previous)


@pytest.mark.validates(
    "preprocessing",
    kind="estimator",
    dataset="eegbci",
    claim="detect-bads, reject, and interpolate are PyPREP NoisyChannels, autoreject "
    "fit/transform, and interpolate_bads with the same seeds",
    criterion="identical candidate lists, retained epochs, repair labels, and samples",
)
def test_automated_tools_match_their_libraries(tmp_path: Path) -> None:
    autoreject = pytest.importorskip("autoreject")
    pyprep = pytest.importorskip("pyprep")
    path = eegbci.load_data(1, [1], update_path=False)[0]
    raw = mne.io.read_raw_edf(path, preload=True)
    eegbci.standardize(raw)
    raw.set_montage("standard_1005")
    raw.save(tmp_path / "sub-01_raw.fif", fmt="double")
    config = tmp_path / "preprocessing.yaml"
    config.write_text(
        "input: {path: sub-01_raw.fif}\n"
        "output: {directory: preprocessed, name: sub-01}\n"
        "workflow: {raw_review: disabled}\n"
        "channels: {bads: [T8], interpolate_bads: true}\n"
        "bad_channels:\n  method: pyprep\n  methods: [flat, deviation, correlation]\n"
        "  random_state: 42\n"
        "filter: {l_freq: 1.0, h_freq: 40.0}\n"
        "epochs: {kind: fixed, duration: 2.0}\n"
        "rejection: {method: autoreject, n_interpolate: [1, 4], consensus: [0.5, 1.0], cv: 4}\n"
    )
    workflow = open_workflow(load_config(config))
    assert run_until(workflow).state == "completed"

    prepared = _raw(workflow, "prepare")
    good = [n for n in prepared.ch_names if n not in prepared.info["bads"]]
    detector = pyprep.NoisyChannels(
        prepared.copy().pick(good), random_state=42, do_detrend=True, reject_by_annotation="omit"
    )
    detector.find_bad_by_nan_flat()
    detector.find_bad_by_deviation()
    detector.find_bad_by_correlation()
    candidates = read_checkpoint(workflow, "detect-bads").state.candidates
    # PyPREP returns a set as a list, in an order that changes with each process's hash
    # seed; the candidates are listed in the recording's channel order.
    assert candidates["bads"] == [name for name in good if name in detector.get_bads()]

    epochs = _epochs(workflow, "epoch")
    model = autoreject.AutoReject(
        n_interpolate=[1, 4], consensus=[0.5, 1.0], cv=4, random_state=42, picks=good, n_jobs=1
    ).fit(epochs.copy())
    log = model.get_reject_log(epochs.copy())
    expected = model.transform(epochs.copy(), reject_log=log)
    rejected = _epochs(workflow, "reject")
    _same_epochs(rejected, expected)
    repairs = pd.read_csv(workflow.config.output.directory / "sub-01_repairs.tsv", sep="\t")
    interpolated = log.labels[:, [epochs.ch_names.index(n) for n in good]] == 2
    assert (repairs["label"] == "interpolated").sum() == int(interpolated.sum())

    expected = rejected.copy().interpolate_bads(reset_bads=True, method={"eeg": "spline"})
    np.testing.assert_allclose(
        _epochs(workflow, "interpolate").get_data(), expected.get_data(), rtol=0, atol=ATOL
    )
    assert _epochs(workflow, "interpolate").info["bads"] == []
