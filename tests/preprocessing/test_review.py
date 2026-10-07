import json
from dataclasses import replace

import pytest

from eegtable.preprocessing import open_workflow, read_checkpoint, run_until
from eegtable.preprocessing import review as review_module
from eegtable.preprocessing.config import WorkflowSettings, read_yaml
from eegtable.preprocessing.review import ReviewCancelled, save_review

from .test_execution import config_for


def _pending_workflow(raw, tmp_path):
    config = replace(config_for(raw, tmp_path), workflow=WorkflowSettings(raw_review="required"))
    workflow = open_workflow(config)
    assert run_until(workflow, "review-raw").state == "needs-review"
    return workflow


def test_viewer_decision_is_validated_and_applied(raw, tmp_path, monkeypatch):
    workflow = _pending_workflow(raw, tmp_path)
    spans = [{"onset": 5.0, "duration": 1.0, "description": "BAD_manual"}]
    monkeypatch.setattr(
        review_module, "viewer_decision", lambda stage, state: {"bads": ["C3"], "spans": spans}
    )
    save_review(workflow, "raw")
    assert run_until(workflow, "review-raw").state == "completed"
    reviewed = read_checkpoint(workflow, "review-raw").state.raw
    assert reviewed.info["bads"] == ["C3"]
    assert reviewed.annotations.onset[0] == 5.0 + reviewed.first_time
    assert read_checkpoint(workflow, "detect-bads").state.raw.info["bads"] == []


def test_cancelled_viewer_saves_nothing(raw, tmp_path, monkeypatch):
    workflow = _pending_workflow(raw, tmp_path)

    def cancel(stage, state):
        raise ReviewCancelled("cancelled")

    monkeypatch.setattr(review_module, "viewer_decision", cancel)
    with pytest.raises(ReviewCancelled):
        save_review(workflow, "raw")
    assert not (workflow.workspace / "decisions" / "review-raw.yaml").exists()


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda d: d.update(parent_id="0" * 64), "parent_id"),
        (lambda d: d.update(spans=[]), "pending"),
        (lambda d: d.update(bads=["missing"], spans=[]), "missing"),
    ],
)
def test_invalid_headless_decisions_are_rejected(raw, tmp_path, edit, match):
    workflow = _pending_workflow(raw, tmp_path)
    decision = read_yaml(workflow.workspace / "decisions" / "review-raw.pending.yaml")
    edit(decision)
    path = tmp_path / "decision.yaml"
    path.write_text(json.dumps(decision), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        save_review(workflow, "raw", path)


def test_suggested_raw_review_applies_candidates(raw, tmp_path):
    import mne

    from eegtable.preprocessing.config import AmplitudeSettings, AnnotationSettings

    # One channel ten times larger than the rest is the only amplitude candidate.
    data = raw.get_data()
    data[0] *= 10.0
    loud = mne.io.RawArray(data, raw.info.copy(), first_samp=raw.first_samp)
    base = config_for(loud, tmp_path)
    config = replace(
        base,
        workflow=WorkflowSettings(raw_review="required"),
        processing=replace(
            base.processing,
            annotations=AnnotationSettings(amplitude=AmplitudeSettings(peak={"eeg": 5e-5})),
        ),
    )
    workflow = open_workflow(config)
    assert run_until(workflow, "review-raw").state == "needs-review"
    candidates = read_checkpoint(workflow, "detect-bads").state.candidates
    assert candidates["bads"] == ["Fp1"]
    save_review(workflow, "raw", suggested=True)
    assert run_until(workflow, "review-raw").state == "completed"
    reviewed = read_checkpoint(workflow, "review-raw").state.raw
    assert reviewed.info["bads"] == candidates["bads"]
    assert read_checkpoint(workflow, "review-raw").state.provenance["raw_decision"]["bads"] == (
        candidates["bads"]
    )
    with pytest.raises(ValueError, match="exclusive"):
        save_review(workflow, "raw", tmp_path / "x.yaml", suggested=True)


def test_cropped_raw_gate_keeps_acquisition_duration_and_span_clock(raw, tmp_path):
    from eegtable.preprocessing.config import CropSettings
    from eegtable.preprocessing.review import gate_view

    base = config_for(raw, tmp_path)
    workflow = open_workflow(
        replace(
            base,
            processing=replace(base.processing, crop=CropSettings(4.0, 28.0)),
            workflow=WorkflowSettings(raw_review="required"),
        )
    )
    assert run_until(workflow, "review-raw").state == "needs-review"
    gate = gate_view(workflow, "review-raw")
    assert gate["duration"] == raw.n_times / raw.info["sfreq"]
    span = {"onset": 26.0, "duration": 1.0, "description": "BAD_manual"}
    assert span["onset"] + span["duration"] <= gate["duration"]
    decision = tmp_path / "decision.yaml"
    decision.write_text(
        json.dumps({"parent_id": gate["parent_id"], "bads": [], "spans": [span]}),
        encoding="utf-8",
    )
    save_review(workflow, "raw", decision)
    assert run_until(workflow, "review-raw").state == "completed"
    reviewed = read_checkpoint(workflow, "review-raw").state.raw
    assert reviewed.annotations.onset.tolist() == [raw.first_time + span["onset"]]
    assert reviewed.annotations.duration.tolist() == [span["duration"]]


def test_suggested_artifact_review_takes_detector_union(mixture, tmp_path):
    from eegtable.preprocessing.config import ArtifactSettings, ICASettings

    base = config_for(mixture, tmp_path)
    artifact = ArtifactSettings(
        "ica", ICASettings(n_components=4, eog_channels=("VEOG",), ecg_channel="ECG"), "average"
    )
    workflow = open_workflow(replace(base, processing=replace(base.processing, artifact=artifact)))
    assert run_until(workflow, "review-artifact").state == "needs-review"
    evidence = read_checkpoint(workflow, "fit-artifact").state.artifact.evidence
    save_review(workflow, "artifact", suggested=True)
    assert run_until(workflow, "review-artifact").state == "completed"
    decision = read_checkpoint(workflow, "review-artifact").state.reviewed.decision
    assert decision["exclude"] == evidence["suggested_exclude"]


def test_pending_file_is_commented_yaml_that_flagless_review_reads(raw, tmp_path):
    workflow = _pending_workflow(raw, tmp_path)
    pending = workflow.workspace / "decisions" / "review-raw.pending.yaml"
    text = pending.read_text(encoding="utf-8")
    assert text.startswith("#") and "bads: null" in text and "spans: null" in text
    pending.write_text(
        text.replace("bads: null", "bads: [C3]").replace("spans: null", "spans: []"),
        encoding="utf-8",
    )
    save_review(workflow, "raw")
    assert run_until(workflow, "review-raw").state == "completed"
    assert read_checkpoint(workflow, "review-raw").state.raw.info["bads"] == ["C3"]


def test_flagless_review_without_viewer_names_the_pending_file(raw, tmp_path, monkeypatch):
    workflow = _pending_workflow(raw, tmp_path)

    def missing(stage, state):
        raise ModuleNotFoundError("Install eegtable[preprocessing-gui]")

    monkeypatch.setattr(review_module, "viewer_decision", missing)
    with pytest.raises(ModuleNotFoundError, match="preprocessing-gui.*review-raw.pending.yaml"):
        save_review(workflow, "raw")


def test_review_of_a_disabled_gate_is_refused(raw, tmp_path):
    workflow = open_workflow(
        replace(config_for(raw, tmp_path), workflow=WorkflowSettings(raw_review="disabled"))
    )
    run_until(workflow, "reject")
    with pytest.raises(ValueError, match="review-epochs.*disabled"):
        save_review(workflow, "epochs", suggested=True)


def test_viewer_raw_review_reports_only_new_spans_in_acquisition_time(raw):
    import mne

    from eegtable.preprocessing.review import new_bad_spans

    # The viewer copy starts with the spans already on the record; only additions are the decision.
    before = raw.copy()
    before.annotations.append(5.0 + raw.first_time, 1.0, "BAD_manual")
    after = before.copy()
    after.annotations.append(9.0 + raw.first_time, 0.5, "BAD_blink")
    after.annotations.append(1.0 + raw.first_time, 0.5, "stimulus")
    original_first_samp = raw.first_samp - 250
    spans = new_bad_spans(before, after, original_first_samp)
    assert spans == [{"onset": 10.0, "duration": 0.5, "description": "BAD_blink"}]
    assert isinstance(spans[0]["onset"], float)
    assert isinstance(mne.Annotations([0], [1], ["x"]), mne.Annotations)


class _Plottable:
    def __init__(self):
        self.calls = []

    def copy(self):
        return self

    def plot(self, block):
        self.calls.append(("plot", block))

    def plot_components(self, **kwargs):
        self.calls.append(("components", kwargs))

    def plot_sources(self, raw, block):
        self.calls.append(("sources", raw, block))


@pytest.fixture
def qt_stubbed(monkeypatch):
    import contextlib

    import mne

    monkeypatch.setattr(review_module, "require", lambda package, extra: None)
    monkeypatch.setattr(mne.viz, "use_browser_backend", lambda name: contextlib.nullcontext())


def test_viewer_shows_ica_sources_for_a_fitted_checkpoint(qt_stubbed):
    from types import SimpleNamespace

    model = _Plottable()
    model.n_components_ = 3
    raw = _Plottable()
    state = SimpleNamespace(
        raw=raw, epochs=None, artifact=SimpleNamespace(method="ica", model=model)
    )
    review_module.open_viewer(state)
    # No picks: MNE pages the topomaps 20 to a figure, and one figure holding every
    # component is metres tall on a montage of any size, with nothing to scroll it.
    assert model.calls == [("components", {}), ("sources", raw, True)]
    assert raw.calls == []


def test_interactive_ica_review_pages_the_topomaps(qt_stubbed, monkeypatch):
    from types import SimpleNamespace

    # Same reason as open_viewer: one figure per component count is unscrollable.
    monkeypatch.setattr(review_module, "_confirm_choices", lambda title, labels: [])
    model = _Plottable()
    model.n_components_ = 3
    model.exclude = [1]
    raw = _Plottable()
    state = SimpleNamespace(
        raw=raw, epochs=None, artifact=SimpleNamespace(method="ica", model=model, fit_id="fit")
    )
    decision = review_module.viewer_decision("review-artifact", state)
    assert model.calls == [("components", {}), ("sources", raw, True)]
    assert decision == {"fit_id": "fit", "exclude": [1]}


@pytest.mark.parametrize("excluded", [[], [1]])
def test_ica_review_saves_viewer_choices_on_a_disposable_copy(qt_stubbed, monkeypatch, excluded):
    from types import SimpleNamespace

    import mne

    model = mne.preprocessing.ICA()
    model.n_components_ = 3
    monkeypatch.setattr(mne.preprocessing.ICA, "plot_components", lambda self: None)
    monkeypatch.setattr(
        mne.preprocessing.ICA,
        "plot_sources",
        lambda self, raw, block: setattr(self, "exclude", excluded.copy()),
    )
    monkeypatch.setattr(review_module, "_confirm_choices", lambda title, labels: [])
    state = SimpleNamespace(
        raw=_Plottable(), artifact=SimpleNamespace(method="ica", model=model, fit_id="fit")
    )

    decision = review_module.viewer_decision("review-artifact", state)

    assert decision == {"fit_id": "fit", "exclude": excluded}
    assert model.exclude == []


def test_viewer_shows_epochs_before_raw_without_an_ica(qt_stubbed):
    from types import SimpleNamespace

    raw, epochs = _Plottable(), _Plottable()
    review_module.open_viewer(SimpleNamespace(raw=raw, epochs=epochs, artifact=None))
    assert epochs.calls == [("plot", True)] and raw.calls == []
    review_module.open_viewer(SimpleNamespace(raw=raw, epochs=None, artifact=None))
    assert raw.calls == [("plot", True)]


def test_viewer_shows_epochs_after_ica_application(qt_stubbed):
    from types import SimpleNamespace

    model, epochs = _Plottable(), _Plottable()
    state = SimpleNamespace(
        raw=None, epochs=epochs, artifact=SimpleNamespace(method="ica", model=model)
    )
    review_module.open_viewer(state)
    assert epochs.calls == [("plot", True)]
    assert model.calls == []
