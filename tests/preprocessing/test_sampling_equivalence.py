import numpy as np
import pytest

from eegtable.preprocessing.config import FilterSettings, FixedEpochSettings, ResamplingSettings
from eegtable.preprocessing.epochs import make_epochs
from eegtable.preprocessing.events import resolve_events
from eegtable.preprocessing.sampling import crop_epochs, resample_epochs


@pytest.mark.parametrize("target,padding", [(200.0, 0.5), (125.0, 0.52)])
def test_resampling_matches_direct_mne(raw, target, padding):
    settings = FixedEpochSettings(2, padding=padding)
    epochs = make_epochs(raw, resolve_events(raw, settings), settings)
    actual = resample_epochs(epochs, ResamplingSettings(target, padding), FilterSettings())
    expected = epochs.copy().resample(
        target, method="polyphase", window=("kaiser", 5.0), pad="reflect", n_jobs=1
    )
    np.testing.assert_allclose(actual.get_data(), expected.get_data(), atol=1e-18)
    np.testing.assert_array_equal(actual.events, epochs.events)
    assert np.isclose(actual.times, 0).any()
    cropped = crop_epochs(actual, 0, 1.996)
    assert cropped.times[0] == 0
    assert abs(cropped.times[-1] - 1.996) * target <= 0.5 + 1e-6


def test_misaligned_padding_fails(raw):
    settings = FixedEpochSettings(2, padding=0.5)
    epochs = make_epochs(raw, resolve_events(raw, settings), settings)
    with pytest.raises(ValueError, match="origin"):
        resample_epochs(epochs, ResamplingSettings(125.0, 0.5), FilterSettings())


@pytest.mark.parametrize("tmin,tmax", [(0.001, 0.997), (0.0019, 0.9941)])
def test_sampling_preflight_uses_mne_rounded_epoch_grid(raw, tmin, tmax):
    from eegtable.preprocessing import preprocess
    from eegtable.preprocessing.config import EventEpochSettings, EventSettings, ProcessingSettings

    events = EventSettings("stim", {"stimulus": 1}, stim_channel="STI", shortest_event=1)
    settings = EventEpochSettings(events, tmin, tmax, padding=0.08)
    sampling = ResamplingSettings(125.0, settings.padding)
    epochs = make_epochs(raw, resolve_events(raw, events), settings)
    expected = crop_epochs(resample_epochs(epochs, sampling, FilterSettings()), tmin, tmax)

    actual = preprocess(raw, ProcessingSettings(settings, sampling=sampling)).epochs

    np.testing.assert_array_equal(actual.times, expected.times)
    np.testing.assert_array_equal(actual.get_data(), expected.get_data())
