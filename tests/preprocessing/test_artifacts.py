import numpy as np
import pytest

from eegtable.preprocessing.artifacts import (
    apply_artifact,
    fit_eog_regression,
    reference_artifact_data,
    review_artifact,
)
from eegtable.preprocessing.config import RegressionSettings


@pytest.mark.parametrize("change", ["drop-auxiliary", "resample"])
def test_ica_accepts_changes_outside_its_spatial_inputs(mixture, change):
    from eegtable.preprocessing.config import ICASettings
    from eegtable.preprocessing.ica import fit_ica

    model = fit_ica(mixture, ICASettings(n_components=4))
    reviewed = review_artifact(model, {"fit_id": model.fit_id, "exclude": [0]})
    application = mixture.copy()
    if change == "drop-auxiliary":
        application.drop_channels(["VEOG", "ECG", "STI"])
    else:
        application.resample(125)

    actual = apply_artifact(application, reviewed)
    expected = model.model.copy().apply(application.copy(), exclude=[0])

    np.testing.assert_array_equal(actual.get_data(), expected.get_data())


@pytest.mark.parametrize(
    "change", ["missing", "reordered", "type", "bad", "reference", "projector"]
)
def test_ica_rejects_changes_to_its_spatial_inputs(mixture, change):
    import mne

    from eegtable.preprocessing.config import ICASettings
    from eegtable.preprocessing.ica import fit_ica

    model = fit_ica(mixture, ICASettings(n_components=4))
    reviewed = review_artifact(model, {"fit_id": model.fit_id, "exclude": [0]})
    application = mixture.copy()
    if change == "missing":
        application.drop_channels(["Fp1"])
    elif change == "reordered":
        application.reorder_channels(application.ch_names[::-1])
    elif change == "type":
        application.set_channel_types({"Fp1": "eog"})
    elif change == "bad":
        application.info["bads"] = ["Fp1"]
    elif change == "reference":
        application.set_eeg_reference("average")
    else:
        application.add_proj(mne.compute_proj_raw(application, n_eeg=1))

    with pytest.raises(ValueError, match="incompatible"):
        apply_artifact(application, reviewed)


def test_regression_matches_mne_and_attenuates(raw):
    from scipy.ndimage import gaussian_filter1d

    predictor = np.zeros(raw.n_times)
    predictor[500::750] = 1e-3
    predictor = gaussian_filter1d(predictor, 10)
    raw.apply_function(lambda values: predictor, picks=["VEOG"])
    raw.apply_function(lambda values: values + 4 * predictor, picks=["Fp1"])
    referenced = reference_artifact_data(raw, "average")
    model = fit_eog_regression(referenced, RegressionSettings(("VEOG",)))
    actual = apply_artifact(
        referenced, review_artifact(model, {"fit_id": model.fit_id, "apply": True})
    )
    expected = model.model.apply(referenced.copy())
    np.testing.assert_allclose(actual.get_data(), expected.get_data(), atol=1e-18)
    np.testing.assert_array_equal(
        actual.get_data(picks=["VEOG"]), referenced.get_data(picks=["VEOG"])
    )
    assert np.std(actual.get_data(picks=["Fp1"])) < np.std(referenced.get_data(picks=["Fp1"]))


def test_ssp_matches_mne_and_excludes_existing_projectors(raw):
    import mne
    from scipy.ndimage import gaussian_filter1d

    from eegtable.preprocessing.artifacts import fit_ssp
    from eegtable.preprocessing.config import SSPSettings

    blink = np.zeros(raw.n_times)
    blink[600::900] = 3e-4
    blink = gaussian_filter1d(blink, 12)
    raw.apply_function(lambda values: values + blink, picks=["VEOG"])
    raw.apply_function(lambda values: values + 0.3 * blink, picks=["Fp1", "Fp2"])
    raw.add_proj(mne.compute_proj_raw(raw, n_eeg=1, verbose=False), remove_existing=True)
    model = fit_ssp(raw, SSPSettings(n_eeg=1, eog_channels=("VEOG",), reject=None))
    assert len(model.model) == 1
    reviewed = review_artifact(model, {"fit_id": model.fit_id, "include": [0]})
    actual = apply_artifact(raw, reviewed)
    expected = raw.copy().add_proj(model.model).apply_proj()
    np.testing.assert_allclose(actual.get_data(), expected.get_data(), atol=1e-18)
    assert np.std(actual.get_data(picks=["Fp1"])) < np.std(raw.get_data(picks=["Fp1"]))
    assert (
        apply_artifact(raw, review_artifact(model, {"fit_id": model.fit_id, "include": []}))
        .get_data()
        .tolist()
        == raw.get_data().tolist()
    )
