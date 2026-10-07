import pytest

from eegtable.preprocessing.config import ICASettings


@pytest.mark.parametrize("channels", [None, ["Fp1", "Fp2", "C3"]])
def test_ordinary_ica_does_not_require_electrode_geometry(mixture, channels):
    from eegtable.preprocessing.checks import validate_processing
    from eegtable.preprocessing.config import (
        ArtifactSettings,
        FixedEpochSettings,
        ProcessingSettings,
    )
    from eegtable.preprocessing.ica import fit_ica

    raw = mixture.copy()
    if channels is not None:
        raw.pick(channels)
    raw.set_montage(None)
    settings = ICASettings(n_components=3)
    validate_processing(
        raw, ProcessingSettings(FixedEpochSettings(2), artifact=ArtifactSettings("ica", settings))
    )

    model = fit_ica(raw, settings)

    assert model.model.n_components_ == 3


def test_iclabel_still_requires_electrode_geometry(mixture):
    from eegtable.preprocessing.checks import validate_processing
    from eegtable.preprocessing.config import (
        ArtifactSettings,
        FilterSettings,
        FixedEpochSettings,
        ICLabelSettings,
        ProcessingSettings,
    )
    from eegtable.preprocessing.ica import fit_ica

    raw = mixture.copy().filter(None, 100).set_montage(None)
    settings = ICASettings(method="infomax", iclabel=ICLabelSettings())
    processing = ProcessingSettings(
        FixedEpochSettings(2),
        artifact=ArtifactSettings("ica", settings, "average"),
        filter=FilterSettings(l_freq=1, h_freq=100),
    )

    with pytest.raises(ValueError, match="montage"):
        validate_processing(raw, processing)
    with pytest.raises(ValueError, match="montage"):
        fit_ica(raw, settings)


def test_exact_ica_review_and_no_refit(mixture):
    import numpy as np

    from eegtable.preprocessing.artifacts import apply_artifact, review_artifact
    from eegtable.preprocessing.ica import fit_ica

    model = fit_ica(mixture, ICASettings(n_components=4))
    reviewed = review_artifact(model, {"fit_id": model.fit_id, "exclude": []})
    result = apply_artifact(mixture, reviewed)
    np.testing.assert_array_equal(result.get_data(), mixture.get_data())
    with pytest.raises(ValueError, match="fit_id"):
        review_artifact(model, {"fit_id": "stale", "exclude": []})
    with pytest.raises(ValueError, match="outside"):
        review_artifact(model, {"fit_id": model.fit_id, "exclude": [4]})


def test_rank_follows_reference(mixture):
    from eegtable.preprocessing.artifacts import reference_artifact_data
    from eegtable.preprocessing.ica import fit_ica

    referenced = reference_artifact_data(mixture, "average")
    assert fit_ica(referenced, ICASettings()).evidence["rank"] == 7
    with pytest.raises(ValueError, match="usable rank"):
        fit_ica(referenced, ICASettings(n_components=8))


def test_reviewed_exclusion_removes_injected_artifact(raw):
    import mne
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    from eegtable.preprocessing.artifacts import apply_artifact, review_artifact
    from eegtable.preprocessing.ica import fit_ica

    # Seven neural sources plus one sparse artifact source in eight channels.
    rng = np.random.default_rng(7)
    neural = rng.normal(size=(8, 7)) @ (rng.laplace(size=(7, raw.n_times)) * 1e-6)
    artifact = np.zeros(raw.n_times)
    artifact[rng.choice(raw.n_times, 40, replace=False)] = 1.0
    artifact = gaussian_filter1d(artifact, 8) * 4e-5
    weights = np.linspace(1.0, 0.2, 8)
    data = np.vstack([neural + np.outer(weights, artifact), raw.get_data()[8:]])
    contaminated = mne.io.RawArray(data, raw.info.copy(), first_samp=raw.first_samp)
    model = fit_ica(contaminated, ICASettings())
    sources = model.model.get_sources(contaminated).get_data()
    # Choosing by correlation with the injected source is for the test only.
    scores = [abs(np.corrcoef(source, artifact)[0, 1]) for source in sources]
    chosen = int(np.argmax(scores))
    assert scores[chosen] > 0.9
    cleaned = apply_artifact(
        contaminated, review_artifact(model, {"fit_id": model.fit_id, "exclude": [chosen]})
    )
    fp1 = cleaned.get_data(picks=["Fp1"])[0]
    assert abs(np.corrcoef(fp1, artifact)[0, 1]) < 0.2
    assert np.corrcoef(fp1, neural[0])[0, 1] > 0.95


def test_detector_suggestions_are_evidence_only(mixture):
    import numpy as np

    from eegtable.preprocessing.artifacts import reference_artifact_data
    from eegtable.preprocessing.ica import fit_ica

    referenced = reference_artifact_data(mixture, "average")
    model = fit_ica(referenced, ICASettings(eog_channels=("VEOG",), ecg_channel="ECG"))
    evidence = model.evidence
    assert set(evidence["suggested"]) == {"VEOG", "ECG"}
    union = sorted({i for found in evidence["suggested"].values() for i in found})
    assert evidence["suggested_exclude"] == union
    assert all(isinstance(index, int) for index in union)
    # Suggestions are MNE's own verdicts on the same training copy.
    training = referenced.copy().filter(1.0, None)
    expected, _ = model.model.find_bads_eog(training, ch_name="VEOG")
    assert evidence["suggested"]["VEOG"] == [int(i) for i in expected]
    assert "iclabel" not in evidence
    np.testing.assert_array_equal(model.model.exclude, [])


@pytest.mark.parametrize("method", ["infomax", "picard"])
def test_ica_methods_follow_settings(mixture, method):
    from eegtable.preprocessing.ica import fit_ica

    if method == "picard":
        pytest.importorskip("picard")
    model = fit_ica(mixture, ICASettings(method=method, n_components=4, max_iter=2000))
    assert model.model.method == method
    assert model.model.fit_params["extended"] is True
    if method == "picard":
        assert model.model.fit_params["ortho"] is False


def test_infomax_converged_by_weight_change_is_accepted(mixture):
    import numpy as np

    from eegtable.preprocessing.ica import fit_ica

    # This fit stops by weight change after about 110 steps, and MNE then reports max_iter.
    first = fit_ica(mixture, ICASettings(method="infomax", n_components=2, max_iter=500))
    longer = fit_ica(mixture, ICASettings(method="infomax", n_components=2, max_iter=1000))
    assert first.model.n_iter_ == 500
    # A higher limit changes nothing, so the first fit converged before its limit.
    np.testing.assert_array_equal(first.model.unmixing_matrix_, longer.model.unmixing_matrix_)


@pytest.mark.parametrize("method", ["fastica", "picard"])
def test_ica_that_reaches_its_limit_raises(mixture, method):
    from eegtable.preprocessing.ica import fit_ica

    if method == "picard":
        pytest.importorskip("picard")
    with pytest.raises(ValueError, match="did not converge"):
        fit_ica(mixture, ICASettings(method=method, n_components=4, max_iter=1))


def test_iclabel_requires_extended_infomax_and_average_reference():
    from eegtable.preprocessing.config import ArtifactSettings, ICLabelSettings

    with pytest.raises(ValueError, match="infomax or picard"):
        ICASettings(iclabel=ICLabelSettings())
    with pytest.raises(ValueError, match="reference: average"):
        ArtifactSettings("ica", ICASettings(method="infomax", iclabel=ICLabelSettings()), None)
    with pytest.raises(ValueError, match="keep"):
        ICLabelSettings(keep=("cortex",))


def test_iclabel_uses_onnx_to_label_every_component(mixture, monkeypatch):
    pytest.importorskip("mne_icalabel")
    import mne_icalabel.iclabel

    from eegtable.preprocessing.artifacts import reference_artifact_data
    from eegtable.preprocessing.config import FilterSettings, ICLabelSettings
    from eegtable.preprocessing.ica import fit_ica
    from eegtable.preprocessing.raw import filter_raw

    classify = mne_icalabel.iclabel.iclabel_label_components

    def classify_with_onnx(*args, **kwargs):
        assert kwargs.get("backend") == "onnx"
        return classify(*args, **kwargs)

    monkeypatch.setattr(mne_icalabel.iclabel, "iclabel_label_components", classify_with_onnx)
    # ICLabel's training band, 1-100 Hz: fit_ica adds the 1 Hz high-pass.
    lowpassed = filter_raw(mixture, FilterSettings(h_freq=100.0))
    referenced = reference_artifact_data(lowpassed, "average")
    settings = ICASettings(
        method="infomax", n_components=4, max_iter=2000, iclabel=ICLabelSettings(threshold=0.5)
    )
    evidence = fit_ica(referenced, settings).evidence["iclabel"]
    assert len(evidence["labels"]) == 4
    assert evidence["probabilities"].shape == (4, 7)
    assert set(evidence["labels"]) <= set(evidence["classes"])
    # The suggestion rule: winning label outside keep with confidence at or above threshold.
    expected = [
        index
        for index, (label, row) in enumerate(
            zip(evidence["labels"], evidence["probabilities"], strict=True)
        )
        if label not in ("brain", "other") and row.max() >= 0.5
    ]
    assert evidence["suggested"] == expected


def test_iclabel_requires_its_training_band(raw):
    import pytest

    from eegtable.preprocessing.checks import validate_processing
    from eegtable.preprocessing.config import (
        ArtifactSettings,
        FilterSettings,
        FixedEpochSettings,
        ICASettings,
        ICLabelSettings,
        ProcessingSettings,
    )

    ica = ICASettings(method="infomax", iclabel=ICLabelSettings())
    artifact = ArtifactSettings("ica", ica, "average")
    # ICLabel was trained on 1-100 Hz data; a 250 Hz recording without a low-pass is 125 Hz wide.
    with pytest.raises(ValueError, match="100 Hz"):
        validate_processing(raw, ProcessingSettings(FixedEpochSettings(2), artifact=artifact))
    with pytest.raises(ValueError, match="100 Hz"):
        validate_processing(
            raw,
            ProcessingSettings(
                FixedEpochSettings(2), artifact=artifact, filter=FilterSettings(h_freq=40)
            ),
        )
    validate_processing(
        raw,
        ProcessingSettings(
            FixedEpochSettings(2), artifact=artifact, filter=FilterSettings(h_freq=100)
        ),
    )


@pytest.mark.parametrize("highpass", [0.5, 2.0])
def test_iclabel_rejects_an_actual_training_band_outside_one_to_one_hundred_hz(raw, highpass):
    pytest.importorskip("mne_icalabel")
    from eegtable.preprocessing.config import FilterSettings, ICLabelSettings
    from eegtable.preprocessing.ica import fit_ica
    from eegtable.preprocessing.raw import filter_raw

    filtered = filter_raw(raw, FilterSettings(l_freq=highpass, h_freq=100.0))
    settings = ICASettings(method="infomax", l_freq=highpass, iclabel=ICLabelSettings())
    with pytest.raises(ValueError, match="1-100 Hz"):
        fit_ica(filtered, settings)


@pytest.mark.parametrize("reference", [None, ("C3",), "inactive-average"])
def test_iclabel_rejects_training_without_common_average_reference(mixture, reference):
    pytest.importorskip("mne_icalabel")
    from eegtable.preprocessing.config import FilterSettings, ICLabelSettings
    from eegtable.preprocessing.ica import fit_ica
    from eegtable.preprocessing.raw import filter_raw

    training = filter_raw(mixture, FilterSettings(l_freq=1.0, h_freq=100.0))
    if reference == "inactive-average":
        training.set_eeg_reference("average", projection=True)
    elif reference is not None:
        training.set_eeg_reference(list(reference), projection=False)
    settings = ICASettings(
        method="infomax", n_components=4, max_iter=2000, iclabel=ICLabelSettings()
    )
    with pytest.raises(ValueError, match="common average reference"):
        fit_ica(training, settings)


def test_iclabel_accepts_an_applied_average_projector_with_bad_channels(mixture):
    pytest.importorskip("mne_icalabel")
    from eegtable.preprocessing.config import FilterSettings, ICLabelSettings
    from eegtable.preprocessing.ica import fit_ica
    from eegtable.preprocessing.raw import filter_raw

    training = filter_raw(mixture, FilterSettings(l_freq=1.0, h_freq=100.0))
    training.info["bads"] = ["O2"]
    training.set_eeg_reference("average", projection=True).apply_proj()
    settings = ICASettings(
        method="infomax", n_components=4, max_iter=2000, iclabel=ICLabelSettings()
    )
    model = fit_ica(training, settings)
    assert model.model.ch_names == [name for name in mixture.ch_names[:8] if name != "O2"]
    assert model.evidence["iclabel"]["probabilities"].shape == (4, 7)


def test_iclabel_accepts_average_reference_after_single_precision_storage(mixture, tmp_path):
    import mne

    pytest.importorskip("mne_icalabel")
    from eegtable.preprocessing.config import FilterSettings, ICLabelSettings
    from eegtable.preprocessing.ica import fit_ica
    from eegtable.preprocessing.raw import filter_raw

    training = filter_raw(mixture, FilterSettings(l_freq=1.0, h_freq=100.0))
    training.set_eeg_reference("average", projection=False)
    path = tmp_path / "referenced_raw.fif"
    training.save(path, fmt="single", verbose=False)
    restored = mne.io.read_raw_fif(path, preload=True, verbose=False)
    settings = ICASettings(
        method="infomax", n_components=4, max_iter=2000, iclabel=ICLabelSettings()
    )
    model = fit_ica(restored, settings)
    assert model.evidence["iclabel"]["probabilities"].shape == (4, 7)


def test_iclabel_ignores_what_a_bad_span_holds(mixture):
    import mne
    import numpy as np

    pytest.importorskip("mne_icalabel")
    from eegtable.preprocessing.artifacts import reference_artifact_data
    from eegtable.preprocessing.config import FilterSettings, ICLabelSettings
    from eegtable.preprocessing.ica import fit_ica
    from eegtable.preprocessing.raw import filter_raw

    # Large noise in the EEG from 10 s to 15 s, which a BAD span covers in both copies.
    noise = np.zeros((mixture.info["nchan"], mixture.n_times))
    noise[:8, 2500:3750] = np.random.default_rng(0).normal(scale=1e-4, size=(8, 1250))
    settings = ICASettings(
        method="infomax", n_components=4, max_iter=2000, iclabel=ICLabelSettings()
    )
    fits = []
    for data in (mixture.get_data(), mixture.get_data() + noise):
        raw = mne.io.RawArray(data, mixture.info, first_samp=mixture.first_samp, verbose=False)
        raw.annotations.append(raw.first_time + 10.0, 5.0, "BAD_movement")
        lowpassed = filter_raw(raw, FilterSettings(h_freq=100.0))
        fits.append(fit_ica(reference_artifact_data(lowpassed, "average"), settings))
    clean, noisy = fits
    # The fit skips the span; MNE then orders the components by variance over every sample.
    unmixing = noisy.model.unmixing_matrix_
    order = [
        np.flatnonzero((unmixing == row).all(axis=1))[0] for row in clean.model.unmixing_matrix_
    ]
    np.testing.assert_array_equal(unmixing[order], clean.model.unmixing_matrix_)
    np.testing.assert_allclose(
        noisy.evidence["iclabel"]["probabilities"][order],
        clean.evidence["iclabel"]["probabilities"],
    )
