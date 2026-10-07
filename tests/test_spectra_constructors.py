"""Spectra computed from epochs in one call, recording their own settings."""

import mne
import numpy as np
import pytest

import eegtable as ef
from eegtable import spectra as spectra_module
from eegtable.spectra import Spectra, Window
from tests.synthetic import make_epochs

ALPHA = next(band for band in ef.BANDS_STANDARD if band.name == "alpha")
# Equally long, as multitaper needs: its frequency grid follows the window length.
WINDOWS = [Window("base", -0.5, 0.0), Window("stim", 0.5, 1.0)]
RECIPE_WINDOWS = {"base": [-0.5, 0.0], "stim": [0.5, 1.0]}
POWER = [{"measure": "integrated_band_power", "bands": ["alpha"]}]


@pytest.mark.parametrize(
    ("method", "settings"),
    [
        ("welch", {"fmin": 4.0, "fmax": 30.0, "n_overlap": 40}),
        ("multitaper", {"fmin": 4.0, "fmax": 30.0, "bandwidth": 3.0}),
    ],
)
def test_a_psd_constructor_names_its_columns_as_the_runner_does(method, settings) -> None:
    # One estimator behind both paths: the same settings give the same columns, value for
    # value, so a notebook result and a batch result can be stacked together.
    epochs = make_epochs()
    spectra = getattr(Spectra, method)(epochs, WINDOWS, recording="sub-01", **settings)
    api = ef.integrated_band_power(spectra, bands=[ALPHA])

    recipe = {
        "windows": RECIPE_WINDOWS,
        "spectra": {"method": method, **settings},
        "features": POWER,
    }
    runner = ef.extract(epochs, recipe, recording="sub-01").epochs

    assert runner is not None
    assert api.names == runner.names
    np.testing.assert_allclose(api.values, runner.values, rtol=1e-12)


@pytest.mark.parametrize("method", ["welch", "multitaper"])
def test_psd_constructor_records_the_effective_sampling_rate(method) -> None:
    estimates = []
    for sfreq in (100.0, 200.0):
        times = np.arange(int(4 * sfreq)) / sfreq
        epochs = mne.EpochsArray(
            np.sin(2 * np.pi * 10 * times)[None, None],
            mne.create_info(["C3"], sfreq, "eeg"),
            verbose=False,
        )
        estimates.append(getattr(Spectra, method)(epochs, recording="r"))

    assert estimates[0].computation != estimates[1].computation
    for estimate, sfreq in zip(estimates, (100.0, 200.0), strict=True):
        assert estimate.computation.parameters["sfreq_hz"] == sfreq


def test_default_welch_grid_is_recorded_when_another_window_shortens_its_segments() -> None:
    epochs = make_epochs(tmin=0.0)
    long = Window("long", 0.0, 1.5)
    alone = Spectra.welch(epochs, [long], recording="r")
    together = Spectra.welch(epochs, [long, Window("short", 0.0, 0.5)], recording="r")

    assert not np.array_equal(alone.freqs, together.freqs)
    assert alone.computation != together.computation


def test_the_morlet_constructor_is_the_tfr_route_without_restating_its_settings() -> None:
    # from_tfr needs n_cycles and the original sampling rate restated, because MNE keeps
    # neither on the TFR; the constructor computes the TFR and passes both itself.
    epochs = make_epochs()
    freqs = np.array([8.0, 10.0, 12.0])
    spectra = Spectra.morlet(epochs, WINDOWS, recording="sub-01", freqs=freqs, n_cycles=3.0)

    tfr = epochs.compute_tfr(
        "morlet", freqs=freqs, n_cycles=3.0, average=False, return_itc=False, verbose="error"
    )
    manual = Spectra.from_tfr(
        tfr, WINDOWS, recording="sub-01", n_cycles=3.0, sfreq=250.0, zero_mean=True
    )

    np.testing.assert_allclose(spectra.data, manual.data)
    assert spectra.computation == manual.computation


def test_morlet_decimation_that_changes_values_changes_column_identity() -> None:
    epochs = make_epochs()
    settings = dict(recording="sub-01", freqs=np.array([8.0, 10.0, 12.0]), n_cycles=3.0)
    tables = [
        ef.mean_tfr_power(
            Spectra.morlet(epochs, WINDOWS, decim=decim, **settings),
            bands=[ef.Band("alpha", 8.0, 12.0)],
        )
        for decim in (1, 4)
    ]

    assert not np.array_equal(tables[0].values, tables[1].values)
    assert set(tables[0].names).isdisjoint(tables[1].names)


def test_morlet_time_offsets_with_the_same_sample_count_have_distinct_identities() -> None:
    epochs = make_epochs(seconds=2.004)
    settings = dict(freqs=np.array([8.0, 10.0, 12.0]), n_cycles=3.0)
    tfrs = [
        epochs.compute_tfr(
            "morlet", decim=decim, average=False, return_itc=False, verbose=False, **settings
        )
        for decim in (4, slice(1, None, 4))
    ]
    spectra = [
        Spectra.from_tfr(
            tfr, WINDOWS, recording="sub-01", n_cycles=3.0, sfreq=250.0, zero_mean=True
        )
        for tfr in tfrs
    ]

    assert tfrs[0].times.size == tfrs[1].times.size
    assert not np.array_equal(spectra[0].data, spectra[1].data, equal_nan=True)
    assert spectra[0].computation != spectra[1].computation


def test_a_constructor_keeps_good_eeg_channels_by_default() -> None:
    epochs = make_epochs(bads=["Pz"])
    spectra = Spectra.welch(epochs, recording="sub-01", fmin=1.0, fmax=40.0)
    assert "Pz" not in spectra.ch_names and len(spectra.ch_names) == 4
    assert epochs.info["bads"] == ["Pz"] and "Pz" in epochs.ch_names


def test_an_infinite_window_is_measured_as_the_epochs_own_span() -> None:
    # As the runner does: the column records where the data begins and ends.
    spectra = Spectra.welch(make_epochs(), recording="sub-01", fmin=1.0, fmax=40.0)
    window = spectra.windows[0]
    assert (window.name, window.tmin, window.tmax) == ("all", -0.5, 1.5)


@pytest.mark.parametrize("method", ["welch", "multitaper"])
@pytest.mark.parametrize(
    ("fmin", "fmax"),
    [
        (np.nan, 40.0),
        (1.0, np.nan),
        (-np.inf, 40.0),
        (1.0, np.inf),
        (-0.1, 40.0),
        (1.0, 126.0),
        (10.1, 10.0),
        (10.0, 10.0),
    ],
)
def test_psd_constructors_refuse_invalid_frequency_bounds(method, fmin, fmax) -> None:
    with pytest.raises(ValueError, match="fmin.*fmax"):
        getattr(Spectra, method)(make_epochs(), recording="sub-01", fmin=fmin, fmax=fmax)


@pytest.mark.parametrize("method", ["welch", "multitaper"])
def test_psd_constructors_refuse_complex_epochs(method) -> None:
    epochs = make_epochs().apply_hilbert(envelope=False)
    with pytest.raises(TypeError, match="real"):
        getattr(Spectra, method)(epochs, recording="sub-01")


def test_morlet_constructor_refuses_complex_epochs() -> None:
    epochs = make_epochs().apply_hilbert(envelope=False)
    with pytest.raises(TypeError, match="real"):
        Spectra.morlet(
            epochs,
            [Window("all", -np.inf, np.inf)],
            recording="sub-01",
            freqs=np.array([8.0, 10.0, 12.0]),
            n_cycles=3.0,
        )


@pytest.mark.parametrize("field", ["freqs", "n_cycles"])
def test_morlet_constructor_refuses_complex_wavelet_settings(field) -> None:
    settings = {"freqs": np.array([8.0, 10.0, 12.0]), "n_cycles": np.full(3, 3.0)}
    settings[field] = np.asarray(settings[field], dtype=complex) + 1j
    with pytest.raises(TypeError, match="real"):
        Spectra.morlet(
            make_epochs(),
            [Window("all", -np.inf, np.inf)],
            recording="sub-01",
            **settings,
        )


def test_tfr_conversion_requires_the_zero_mean_declaration() -> None:
    tfr = make_epochs().compute_tfr(
        "morlet",
        freqs=np.array([8.0, 10.0, 12.0]),
        n_cycles=1.0,
        zero_mean=True,
        average=False,
        return_itc=False,
        verbose=False,
    )
    with pytest.raises(TypeError, match="zero_mean"):
        Spectra.from_tfr(
            tfr,
            [Window("all", -np.inf, np.inf)],
            recording="sub-01",
            n_cycles=1.0,
            sfreq=250.0,
        )


def test_tfr_zero_mean_settings_with_different_power_have_distinct_names() -> None:
    tables = []
    for zero_mean in (False, True):
        tfr = make_epochs().compute_tfr(
            "morlet",
            freqs=np.array([8.0, 10.0, 12.0]),
            n_cycles=1.0,
            zero_mean=zero_mean,
            average=False,
            return_itc=False,
            verbose=False,
        )
        spectra = Spectra.from_tfr(
            tfr,
            [Window("all", -np.inf, np.inf)],
            recording="sub-01",
            n_cycles=1.0,
            sfreq=250.0,
            zero_mean=zero_mean,
        )
        assert spectra.computation.parameters["zero_mean"] is zero_mean
        tables.append(ef.mean_tfr_power(spectra, bands=[ef.Band("alpha", 8.0, 12.0)]))
    assert not np.array_equal(tables[0].values, tables[1].values)
    assert set(tables[0].names).isdisjoint(tables[1].names)


@pytest.mark.parametrize("zero_mean", [None, 0, 1, "False"])
def test_tfr_conversion_requires_a_boolean_zero_mean(zero_mean) -> None:
    with pytest.raises(TypeError, match="zero_mean.*bool"):
        Spectra.from_tfr(
            object(),
            [Window("all", -np.inf, np.inf)],
            recording="sub-01",
            n_cycles=3.0,
            sfreq=250.0,
            zero_mean=zero_mean,
        )


def test_tfr_conversion_refuses_complex_cycle_counts() -> None:
    with pytest.raises(TypeError, match="real"):
        Spectra.from_tfr(
            object(),
            [Window("all", -np.inf, np.inf)],
            recording="sub-01",
            n_cycles=np.array([3.0 + 1j]),
            sfreq=250.0,
            zero_mean=True,
        )


_WELCH_ARGUMENTS = {
    "sfreq": 250.0,
    "window": Window("all", 0.0, 2.0),
    "fmin": 1.0,
    "fmax": 40.0,
    "n_fft": 250,
    "n_overlap": None,
    "statistic": "mean",
    "n_jobs": 1,
}


def _whole_and_in_blocks(monkeypatch, data):
    whole = spectra_module.welch_psd(data, **_WELCH_ARGUMENTS)
    # One epoch per call, as a large recording would be cut, to keep MNE off its slow path.
    monkeypatch.setattr(spectra_module, "_WELCH_BLOCK_BYTES", data[0].nbytes)
    return whole, spectra_module.welch_psd(data, **_WELCH_ARGUMENTS)


def test_welch_in_blocks_of_epochs_gives_the_numbers_of_one_call(monkeypatch) -> None:
    data = np.random.default_rng(0).normal(size=(6, 3, 501))
    (whole, freqs), (blocks, block_freqs) = _whole_and_in_blocks(monkeypatch, data)
    # macOS's FFT rounds the last bit differently with the batch size, so values agree to
    # rounding there rather than bit for bit.
    np.testing.assert_allclose(whole, blocks, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(freqs, block_freqs)


@pytest.mark.parametrize("missing", [np.nan, np.inf])
def test_a_channel_with_missing_samples_is_missing_whatever_else_is_estimated(
    monkeypatch, missing
) -> None:
    # MNE estimates around NaN that every row of a call shares, so an epoch's estimate
    # used to depend on the epochs estimated with it, and an infinity was not missing.
    data = np.random.default_rng(1).normal(size=(4, 3, 501))
    data[0, :, 100:150] = missing
    (whole, _), (blocks, _) = _whole_and_in_blocks(monkeypatch, data)
    alone, _ = spectra_module.welch_psd(data[:1], **_WELCH_ARGUMENTS)
    assert np.isnan(whole[0]).all() and np.isnan(alone).all()
    np.testing.assert_allclose(whole[1:], blocks[1:], rtol=1e-12, atol=0)
    assert np.isfinite(whole[1:]).all()
