import warnings

import numpy as np
import pytest

from eegtable.signal import Signal
from eegtable.spectra import (
    Spectra,
    Window,
    gradient_weights,
    support_restricted_mask,
    trapezoid_weights,
)
from eegtable.table import ComputationSpec
from eegtable.temporal import variance


def test_trapezoid_weights_sum_to_the_frequency_span() -> None:
    for freqs in (np.linspace(1.0, 45.0, 89), np.logspace(0.0, 2.0, 40)):
        assert trapezoid_weights(freqs).sum() == pytest.approx(freqs[-1] - freqs[0])


def test_trapezoid_weighting_recovers_an_analytic_integral_on_a_log_grid() -> None:
    # A plain mean cannot do this: the log grid over-samples low frequencies.
    freqs = np.logspace(np.log10(1.0), np.log10(100.0), 40)
    power = 2.0 * freqs  # integral over [1, 100] is 100^2 - 1^2 = 9999
    weights = trapezoid_weights(freqs)
    assert (power * weights).sum() == pytest.approx(9999.0, rel=1e-3)


def test_single_frequency_gets_unit_weight() -> None:
    assert trapezoid_weights(np.array([10.0])).tolist() == [1.0]
    assert gradient_weights(np.array([10.0])).tolist() == [1.0]


def test_the_two_weightings_differ_only_at_the_endpoints() -> None:
    freqs = np.linspace(1.0, 10.0, 10)
    trap, grad = trapezoid_weights(freqs), gradient_weights(freqs)
    np.testing.assert_allclose(trap[1:-1], grad[1:-1])
    assert trap[0] == pytest.approx(grad[0] / 2.0)
    assert trap[-1] == pytest.approx(grad[-1] / 2.0)


def test_gradient_weights_are_uniform_on_a_uniform_grid() -> None:
    # This is what makes a flat spectrum have exactly maximal entropy.
    assert len(set(gradient_weights(np.linspace(1.0, 10.0, 10)).round(12))) == 1


def test_window_requires_ordered_bounds() -> None:
    with pytest.raises(ValueError):
        Window("stim", 5.0, 5.0)
    with pytest.raises(ValueError):
        Window("", 0.0, 1.0)
    assert Window("all", -np.inf, np.inf).name == "all"


def _one_channel(n_times: int = 5) -> Signal:
    return Signal.from_arrays(
        data=np.arange(float(n_times)).reshape(1, 1, n_times),
        times=np.linspace(0.0, 1.0, n_times),
        ch_names=("C3",),
        sfreq=n_times - 1.0,
        row_ids=(("r", 0, "e"),),
    )


def test_integer_and_float_window_bounds_name_the_same_column() -> None:
    # 0 and 0.0 are different JSON tokens: kept as given, one window would hash to two
    # column names and the same feature would land in two columns of a cohort table.
    by_int = variance([_one_channel()], windows=[Window("post", 0, 1)], include_global=False)
    by_float = variance([_one_channel()], windows=[Window("post", 0.0, 1.0)], include_global=False)
    assert by_int.names == by_float.names


def test_numpy_infinite_window_bounds_name_a_column() -> None:
    window = Window("all", *np.array([-np.inf, np.inf]))
    table = variance([_one_channel()], windows=[window], include_global=False)
    assert table.meta[0].window_bounds == (-np.inf, np.inf)


def test_from_epochs_spectrum_gains_a_singleton_window_axis() -> None:
    mne = pytest.importorskip("mne")
    info = mne.create_info(["C3", "C4"], 200.0, "eeg")
    epochs = mne.EpochsArray(
        np.random.RandomState(0).randn(5, 2, 400) * 1e-6, info, tmin=-1.0, verbose="ERROR"
    )
    spectrum = epochs.compute_psd(
        "multitaper", fmin=2.0, fmax=40.0, normalization="full", verbose="ERROR"
    )
    spectra = Spectra.from_spectrum(
        spectrum,
        recording="test",
        estimator_parameters={"bandwidth": None, "normalization": "full"},
    )
    assert spectra.data.ndim == 4
    assert spectra.data.shape[0] == 5
    assert spectra.data.shape[1] == 2
    assert spectra.data.shape[2] == 1
    assert spectra.ch_names == ("C3", "C4")
    assert spectra.windows[0].name == "all"
    assert spectra.source == "multitaper"
    assert spectra.coverage.shape == spectra.data.shape
    assert (spectra.coverage == 1.0).all()


def test_from_continuous_spectrum_gains_epoch_and_window_axes() -> None:
    mne = pytest.importorskip("mne")
    info = mne.create_info(["C3", "C4"], 200.0, "eeg")
    raw = mne.io.RawArray(np.random.RandomState(0).randn(2, 4000) * 1e-6, info, verbose="ERROR")
    spectrum = raw.compute_psd("welch", fmin=2.0, fmax=40.0, n_fft=400, verbose="ERROR")
    spectra = Spectra.from_spectrum(spectrum, recording="test", estimator_parameters={"n_fft": 400})
    assert spectra.data.shape[0] == 1
    assert spectra.data.shape[2] == 1


@pytest.mark.parametrize("epoched", [False, True])
def test_from_spectrum_preserves_explicitly_retained_channels(epoched: bool) -> None:
    import mne

    info = mne.create_info(["C3", "C4", "VEOG"], 200.0, ["eeg", "eeg", "eog"])
    info["bads"] = ["C4"]
    data = np.random.default_rng(0).normal(size=(2, 3, 800)) * 1e-6
    source = (
        mne.EpochsArray(data, info, verbose=False)
        if epoched
        else mne.io.RawArray(data[0], info, verbose=False)
    )
    spectrum = source.compute_psd(
        "welch", picks=source.ch_names, exclude=(), n_fft=200, verbose=False
    )

    spectra = Spectra.from_spectrum(spectrum, recording="test", estimator_parameters={})

    expected = spectrum.get_data(picks=spectrum.ch_names)
    if not epoched:
        expected = expected[np.newaxis]
    assert spectra.ch_names == tuple(spectrum.ch_names)
    np.testing.assert_array_equal(spectra.data[:, :, 0], expected)
    assert spectrum.ch_names == ["C3", "C4", "VEOG"]
    assert spectrum.info["bads"] == ["C4"]


def _welch_spectrum(**parameters: object):
    import mne

    info = mne.create_info(["C3"], 160.0, "eeg")
    data = np.random.default_rng(0).normal(size=(2, 1, 800)) * 1e-6
    epochs = mne.EpochsArray(data, info, verbose="ERROR")
    return epochs.compute_psd("welch", verbose="ERROR", **parameters)


def test_from_spectrum_rejects_a_parameter_the_estimator_does_not_take() -> None:
    # A typo would otherwise enter the column identity as though it were a setting.
    spectrum = _welch_spectrum(n_fft=160)
    with pytest.raises(ValueError, match="nfft"):
        Spectra.from_spectrum(spectrum, recording="test", estimator_parameters={"nfft": 160})


def test_from_spectrum_rejects_an_n_fft_its_frequency_grid_contradicts() -> None:
    # 160 Hz over 160 points gives 1 Hz bins; n_fft = 320 would have given 0.5 Hz ones.
    spectrum = _welch_spectrum(n_fft=160)
    with pytest.raises(ValueError, match="n_fft"):
        Spectra.from_spectrum(spectrum, recording="test", estimator_parameters={"n_fft": 320})


@pytest.mark.parametrize(("declared", "key"), [({"fmin": 2.0}, "fmin"), ({"fmax": 30.0}, "fmax")])
def test_from_spectrum_rejects_a_frequency_range_its_axis_contradicts(declared, key) -> None:
    spectrum = _welch_spectrum(n_fft=160, fmin=1.0, fmax=45.0)
    with pytest.raises(ValueError, match=key):
        Spectra.from_spectrum(
            spectrum, recording="test", estimator_parameters={"n_fft": 160, **declared}
        )


def test_from_spectrum_refuses_execution_settings_that_would_split_columns() -> None:
    spectrum = _welch_spectrum(n_fft=160)
    with pytest.raises(ValueError, match="n_jobs"):
        Spectra.from_spectrum(
            spectrum, recording="test", estimator_parameters={"n_fft": 160, "n_jobs": 2}
        )


def test_from_spectrum_accepts_declarations_that_match_the_spectrum() -> None:
    parameters = {
        "fmin": 1.0,
        "fmax": 45.0,
        "n_fft": 160,
        "n_per_seg": 160,
        "n_overlap": 80,
        "window": "hamming",
        "average": "mean",
        "remove_dc": True,
    }
    spectrum = _welch_spectrum(**parameters)
    spectra = Spectra.from_spectrum(
        spectrum, recording="test", estimator_parameters={"method": "welch", **parameters}
    )
    assert spectra.source == "welch"


@pytest.mark.parametrize("parameters", [{}, {"normalization": "length"}])
def test_multitaper_spectrum_requires_explicit_density_normalization(parameters) -> None:
    import mne

    epochs = mne.EpochsArray(
        np.random.default_rng(0).normal(size=(2, 1, 400)),
        mne.create_info(["Cz"], 100.0, "eeg"),
        verbose=False,
    )
    spectrum = epochs.compute_psd("multitaper", verbose=False)
    with pytest.raises(ValueError, match="normalization.*full"):
        Spectra.from_spectrum(spectrum, recording="test", estimator_parameters=parameters)


def test_complex_fourier_coefficients_cannot_be_interpreted_as_power() -> None:
    from types import SimpleNamespace

    spectrum = SimpleNamespace(
        get_data=lambda **kwargs: np.ones((1, 3), dtype=complex) * (1 + 2j),
        freqs=np.array([1.0, 2.0, 3.0]),
        ch_names=["Cz"],
        method="welch",
    )
    with pytest.raises(ValueError, match="complex.*power"):
        Spectra.from_spectrum(spectrum, recording="test", estimator_parameters={})


@pytest.mark.parametrize("field", ["data", "freqs"])
def test_spectral_arrays_reject_complex_power_and_frequency_axes(field: str) -> None:
    data = np.ones((1, 1, 1, 3))
    freqs = np.array([8.0, 10.0, 13.0])
    if field == "data":
        data = data + 100.0j
    else:
        freqs = freqs + 100.0j
    with warnings.catch_warnings(), pytest.raises(TypeError, match="real"):
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        Spectra(
            data=data,
            freqs=freqs,
            ch_names=("Cz",),
            windows=(Window("all", -np.inf, np.inf),),
            coverage=np.ones(data.shape),
            source="provided",
            representation="psd",
            support=np.ones(data.shape),
            row_ids=(("test", 0, "event"),),
            computation=ComputationSpec.create("provided"),
        )


def test_non_finite_input_lowers_coverage_rather_than_raising() -> None:
    data = np.ones((2, 2, 1, 4))
    data[0, 0, 0, 1] = np.nan
    spectra = Spectra(
        data=data,
        freqs=np.array([1.0, 2.0, 3.0, 4.0]),
        ch_names=("C3", "C4"),
        windows=(Window("all", -np.inf, np.inf),),
        coverage=np.isfinite(data).astype(float),
        source="test",
        representation="psd",
        support=np.ones(data.shape),
        row_ids=(("test", 0, "event"), ("test", 1, "event")),
        computation=ComputationSpec.create("test"),
    )
    assert spectra.coverage[0, 0, 0, 1] == 0.0
    assert spectra.coverage.sum() == 15.0


def test_shape_and_axis_mismatches_raise() -> None:
    good = dict(
        data=np.ones((2, 2, 1, 4)),
        freqs=np.array([1.0, 2.0, 3.0, 4.0]),
        ch_names=("C3", "C4"),
        windows=(Window("all", -np.inf, np.inf),),
        coverage=np.ones((2, 2, 1, 4)),
        source="test",
        representation="psd",
        support=np.ones((2, 2, 1, 4)),
        row_ids=(("test", 0, "event"), ("test", 1, "event")),
        computation=ComputationSpec.create("test"),
    )
    with pytest.raises(ValueError, match="ch_names"):
        Spectra(**{**good, "ch_names": ("C3",)})
    with pytest.raises(ValueError, match="freqs"):
        Spectra(**{**good, "freqs": np.array([1.0, 2.0])})
    with pytest.raises(ValueError, match="ascending"):
        Spectra(**{**good, "freqs": np.array([4.0, 3.0, 2.0, 1.0])})
    with pytest.raises(ValueError, match="coverage"):
        Spectra(**{**good, "coverage": np.ones((2, 2, 1, 3))})
    with pytest.raises(ValueError, match="support"):
        Spectra(**{**good, "support": np.ones((2, 2, 1, 3))})


def test_repeated_window_names_cannot_make_baseline_selection_ambiguous() -> None:
    data = np.ones((1, 1, 2, 3))
    with pytest.raises(ValueError, match="window names.*unique"):
        Spectra(
            data=data,
            freqs=np.array([8.0, 10.0, 13.0]),
            ch_names=("C3",),
            windows=(Window("baseline", -1.0, -0.5), Window("baseline", -0.5, 0.0)),
            coverage=np.ones_like(data),
            source="test",
            representation="psd",
            support=np.ones_like(data),
            row_ids=(("test", 0, "event"),),
            computation=ComputationSpec.create("test"),
        )


@pytest.mark.parametrize("field", ["coverage", "support"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -0.1, 1.1])
def test_spectral_fractions_must_be_finite_and_bounded(field: str, invalid: float) -> None:
    good = dict(
        data=np.ones((1, 1, 1, 2)),
        freqs=np.array([1.0, 2.0]),
        ch_names=("C3",),
        windows=(Window("all", -np.inf, np.inf),),
        coverage=np.ones((1, 1, 1, 2)),
        source="test",
        representation="psd",
        support=np.ones((1, 1, 1, 2)),
        row_ids=(("test", 0, "event"),),
        computation=ComputationSpec.create("test"),
    )
    values = good[field].copy()
    values[0, 0, 0, 0] = invalid

    with pytest.raises(ValueError, match=rf"{field} must contain finite values in \[0, 1\]"):
        Spectra(**{**good, field: values})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("freqs", np.array([1.0, np.nan]), "finite"),
        ("freqs", np.array([-1.0, 2.0]), "non-negative"),
        ("data", np.array([[[[-1.0, 1.0]], [[1.0, 1.0]]]]), "negative power"),
        ("ch_names", ("C3", "C3"), "unique"),
    ],
)
def test_spectral_domain_invariants_raise(field: str, value: object, message: str) -> None:
    good = dict(
        data=np.ones((1, 2, 1, 2)),
        freqs=np.array([1.0, 2.0]),
        ch_names=("C3", "C4"),
        windows=(Window("all", -np.inf, np.inf),),
        coverage=np.ones((1, 2, 1, 2)),
        source="test",
        representation="psd",
        support=np.ones((1, 2, 1, 2)),
        row_ids=(("test", 0, "event"),),
        computation=ComputationSpec.create("test"),
    )

    with pytest.raises(ValueError, match=message):
        Spectra(**{**good, field: value})


def test_spectra_require_nonempty_axes() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Spectra(
            data=np.empty((1, 1, 1, 0)),
            freqs=np.empty(0),
            ch_names=("C3",),
            windows=(Window("all", -np.inf, np.inf),),
            coverage=np.empty((1, 1, 1, 0)),
            source="test",
            representation="psd",
            support=np.empty((1, 1, 1, 0)),
            row_ids=(("test", 0, "event"),),
            computation=ComputationSpec.create("test"),
        )


def _toy_tfr(n_epochs: int = 4, method: str = "morlet", decim: int = 1):
    mne = pytest.importorskip("mne")
    info = mne.create_info(["C3", "C4"], 200.0, "eeg")
    rng = np.random.RandomState(0)
    epochs = mne.EpochsArray(rng.randn(n_epochs, 2, 800) * 1e-6, info, tmin=-2.0, verbose="ERROR")
    return epochs.compute_tfr(
        method,
        freqs=np.array([8.0, 10.0, 12.0]),
        n_cycles=3.0,
        decim=decim,
        return_itc=False,
        verbose="ERROR",
    )


def test_from_tfr_produces_one_spectrum_per_window() -> None:
    windows = (Window("base", -2.0, -1.0), Window("stim", 0.0, 1.0))
    spectra = Spectra.from_tfr(_toy_tfr(), windows, recording="test", n_cycles=3.0, sfreq=200.0)
    assert spectra.data.shape == (4, 2, 2, 3)
    assert tuple(w.name for w in spectra.windows) == ("base", "stim")
    assert spectra.source == "morlet"


def test_from_tfr_preserves_explicitly_retained_channels() -> None:
    import mne

    info = mne.create_info(["C3", "C4", "VEOG"], 200.0, ["eeg", "eeg", "eog"])
    info["bads"] = ["C4"]
    epochs = mne.EpochsArray(
        np.random.default_rng(0).normal(size=(2, 3, 800)) * 1e-6, info, verbose=False
    )
    tfr = epochs.compute_tfr(
        "morlet", freqs=[10.0], n_cycles=3.0, picks=epochs.ch_names, verbose=False
    )
    window = Window("middle", 1.0, 3.0)

    spectra = Spectra.from_tfr(tfr, [window], recording="test", n_cycles=3.0, sfreq=200.0)

    mask = support_restricted_mask(tfr.times, tfr.freqs, window, 3.0)[0]
    expected = tfr.get_data(picks=tfr.ch_names)[:, :, 0, mask].mean(axis=-1) / 200.0
    assert spectra.ch_names == tuple(tfr.ch_names)
    np.testing.assert_allclose(spectra.data[:, :, 0, 0], expected)
    assert tfr.ch_names == ["C3", "C4", "VEOG"]
    assert tfr.info["bads"] == ["C4"]


def test_from_tfr_window_mean_equals_a_manual_mean_over_the_time_mask() -> None:
    tfr = _toy_tfr()
    window = Window("stim", 0.0, 1.0)
    spectra = Spectra.from_tfr(tfr, (window,), recording="test", n_cycles=3.0, sfreq=200.0)
    times = np.asarray(tfr.times)
    mask = support_restricted_mask(times, np.asarray(tfr.freqs), window, 3.0)
    # Divided by the sampling rate: MNE's power is sfreq times the density.
    data = np.asarray(tfr.get_data()) / 200.0
    expected = np.stack(
        [data[:, :, index, row].mean(axis=2) for index, row in enumerate(mask)], axis=2
    )
    np.testing.assert_allclose(spectra.data[:, :, 0, :], expected)


@pytest.mark.parametrize(
    ("tmin", "tmax"),
    [(-np.inf, np.inf), (-np.inf, 1.0), (-1.0, np.inf)],
)
def test_from_tfr_limits_wavelet_support_to_available_times(tmin: float, tmax: float) -> None:
    tfr = _toy_tfr()
    window = Window("requested", tmin, tmax)
    intersection = Window("available", max(tmin, tfr.times[0]), min(tmax, tfr.times[-1]))
    spectra = Spectra.from_tfr(
        tfr, (window, intersection), recording="test", n_cycles=3.0, sfreq=200.0
    )

    np.testing.assert_allclose(spectra.data[:, :, 0], spectra.data[:, :, 1])
    np.testing.assert_array_equal(spectra.coverage[:, :, 0], spectra.coverage[:, :, 1])
    np.testing.assert_array_equal(spectra.support[:, :, 0], spectra.support[:, :, 1])
    assert np.all(spectra.support < 1.0)


def _noise_tfr(burst: bool):
    # White noise in 20 epochs, optionally with a 0.2 s burst at ten times the noise
    # amplitude inside the window: 3% of it, or up to 9% once the wavelet smears it.
    # A median moves with the contaminated fraction too, by about 14% at 9%.
    mne = pytest.importorskip("mne")
    sfreq = 200.0
    data = np.random.RandomState(1).randn(20, 1, 1600) * 1e-6
    if burst:
        data[:, :, 600:640] *= 10.0
    epochs = mne.EpochsArray(data, mne.create_info(["C3"], sfreq, "eeg"), verbose="ERROR")
    tfr = epochs.compute_tfr(
        "morlet",
        freqs=np.array([10.0, 20.0, 40.0]),
        n_cycles=7.0,
        return_itc=False,
        verbose="ERROR",
    )
    return tfr, sfreq


def _reduced(burst: bool, statistic: str) -> np.ndarray:
    tfr, sfreq = _noise_tfr(burst)
    spectra = Spectra.from_tfr(
        tfr,
        (Window("stim", 1.0, 7.0),),
        recording="test",
        n_cycles=7.0,
        sfreq=sfreq,
        statistic=statistic,
    )
    return spectra.data[:, 0, 0, :].mean(axis=0)


def test_from_tfr_median_estimates_the_mean_density_of_gaussian_noise() -> None:
    # Power of Gaussian noise is exponential at each time point, so its median is ln 2
    # of its mean; without that correction the median would read 31% low.
    np.testing.assert_allclose(_reduced(False, "median"), _reduced(False, "mean"), rtol=0.1)


def test_from_tfr_median_ignores_a_brief_burst_that_the_mean_absorbs() -> None:
    # Each statistic against its own burst-free value, so the median's sampling error
    # on clean data does not count as the burst's effect.
    assert np.all(_reduced(True, "mean") > 4.0 * _reduced(False, "mean"))
    assert np.all(_reduced(True, "median") < 1.25 * _reduced(False, "median"))


def test_from_tfr_records_a_median_reduction_in_its_computation() -> None:
    tfr = _toy_tfr()
    window = (Window("stim", 0.0, 1.0),)
    mean = Spectra.from_tfr(tfr, window, recording="test", n_cycles=3.0, sfreq=200.0)
    median = Spectra.from_tfr(
        tfr, window, recording="test", n_cycles=3.0, sfreq=200.0, statistic="median"
    )
    assert median.computation.parameters["window_statistic"] == "median"
    assert "window_statistic" not in mean.computation.parameters


def test_from_tfr_rejects_an_unknown_window_statistic() -> None:
    with pytest.raises(ValueError, match="statistic"):
        Spectra.from_tfr(
            _toy_tfr(),
            (Window("stim", 0.0, 1.0),),
            recording="test",
            n_cycles=3.0,
            sfreq=200.0,
            statistic="mode",
        )


def test_from_tfr_refuses_an_already_baselined_tfr() -> None:
    tfr = _toy_tfr().apply_baseline((-2.0, -1.0), mode="logratio", verbose="ERROR")
    with pytest.raises(ValueError, match="already baseline"):
        Spectra.from_tfr(
            tfr, (Window("stim", 0.0, 1.0),), recording="test", n_cycles=3.0, sfreq=200.0
        )


def test_from_tfr_refuses_a_tfr_that_is_not_morlet() -> None:
    # The support mask and the recorded method assume Morlet wavelets; a multitaper TFR,
    # as in MNE's ERDS example, has another support and would be labelled morlet.
    with pytest.raises(ValueError, match="multitaper"):
        Spectra.from_tfr(
            _toy_tfr(method="multitaper"),
            (Window("stim", 0.0, 1.0),),
            recording="test",
            n_cycles=3.0,
            sfreq=200.0,
        )


def test_from_tfr_rejects_a_window_outside_the_time_axis() -> None:
    with pytest.raises(ValueError, match="no samples"):
        Spectra.from_tfr(
            _toy_tfr(), (Window("late", 30.0, 40.0),), recording="test", n_cycles=3.0, sfreq=200.0
        )


def test_from_tfr_rejects_a_finite_window_bound_past_the_time_axis() -> None:
    # The epochs end at 1.995 s. Clipped, the column would claim 0-3 s and measure 0-1.995 s.
    with pytest.raises(ValueError, match="reaches outside"):
        Spectra.from_tfr(
            _toy_tfr(), (Window("stim", 0.0, 3.0),), recording="test", n_cycles=3.0, sfreq=200.0
        )


def test_from_tfr_accepts_a_window_ending_where_a_decimated_axis_stops_short() -> None:
    # Decimating by 4 keeps the sample at 1.98 s but not the epochs' last one at 1.995 s, so a
    # window ending at the epochs' end lies within one step of the axis and is still data.
    tfr = _toy_tfr(decim=4)
    spectra = Spectra.from_tfr(
        tfr, (Window("stim", 0.0, 1.995),), recording="test", n_cycles=3.0, sfreq=200.0
    )
    assert spectra.data.shape[2] == 1


def test_from_tfr_requires_at_least_one_window() -> None:
    with pytest.raises(ValueError, match="at least one window"):
        Spectra.from_tfr(_toy_tfr(), (), recording="test", n_cycles=3.0, sfreq=200.0)


def test_from_tfr_rejects_complex_output() -> None:
    tfr = _toy_tfr()
    complex_tfr = type(
        "T",
        (),
        {
            "baseline": None,
            "times": np.asarray(tfr.times),
            "freqs": np.asarray(tfr.freqs),
            "ch_names": list(tfr.ch_names),
            "method": "morlet",
            "get_data": lambda self, **kwargs: np.asarray(tfr.get_data(**kwargs), dtype=complex),
        },
    )()
    with pytest.raises(ValueError, match="complex"):
        Spectra.from_tfr(
            complex_tfr,
            (Window("stim", 0.0, 1.0),),
            recording="test",
            n_cycles=3.0,
            sfreq=200.0,
        )


def test_support_restriction_narrows_low_frequencies_more_than_high_ones() -> None:
    times = np.linspace(-2.0, 2.0, 401)
    freqs = np.array([4.0, 40.0])
    mask = support_restricted_mask(times, freqs, Window("stim", 0.0, 1.0), n_cycles=6.0)
    # MNE extends to five Gaussian standard deviations, where
    # sigma_t = n_cycles / (2*pi*f).
    assert mask.shape == (2, 401)
    assert mask[0].sum() < mask[1].sum()
    expected_half_support = 5.0 * 6.0 / (2.0 * np.pi * 40.0)
    assert times[mask[1]].min() == pytest.approx(expected_half_support, abs=0.01)
    assert times[mask[1]].max() == pytest.approx(1.0 - expected_half_support, abs=0.01)


@pytest.mark.parametrize("n_cycles", [0.0, -3.0, np.nan, np.inf, np.array([3.0, -1.0])])
def test_morlet_support_requires_finite_positive_cycles(n_cycles) -> None:
    with pytest.raises(ValueError, match="n_cycles.*finite.*positive"):
        support_restricted_mask(
            np.linspace(-2.0, 2.0, 401),
            np.array([4.0, 40.0]),
            Window("all", -np.inf, np.inf),
            n_cycles,
        )


@pytest.mark.parametrize("n_cycles", [np.array([3.0]), np.ones((2, 1))])
def test_morlet_support_requires_a_scalar_or_one_cycle_count_per_frequency(n_cycles) -> None:
    with pytest.raises(ValueError, match="n_cycles.*scalar.*frequency"):
        support_restricted_mask(
            np.linspace(-2.0, 2.0, 401), np.array([4.0, 40.0]), Window("stim", 0.0, 1.0), n_cycles
        )


@pytest.mark.parametrize("frequency", [0.0, -1.0, np.nan, np.inf])
def test_morlet_support_requires_finite_positive_frequencies(frequency) -> None:
    with pytest.raises(ValueError, match="freqs.*finite.*positive"):
        support_restricted_mask(
            np.linspace(-2.0, 2.0, 401),
            np.array([frequency, 40.0]),
            Window("all", -np.inf, np.inf),
            3.0,
        )


@pytest.mark.parametrize("n_cycles", [0.0, -3.0])
def test_from_tfr_refuses_invalid_cycles_for_a_whole_epoch_window(n_cycles) -> None:
    with pytest.raises(ValueError, match="n_cycles.*positive"):
        Spectra.from_tfr(
            _toy_tfr(),
            (Window("all", -np.inf, np.inf),),
            recording="test",
            n_cycles=n_cycles,
            sfreq=200.0,
        )


def test_a_frequency_whose_support_never_fits_drops_out_entirely() -> None:
    times = np.linspace(-2.0, 2.0, 401)
    freqs = np.array([1.0, 40.0])
    # at 1 Hz half support is 3 s, far wider than the 1 s window
    mask = support_restricted_mask(times, freqs, Window("stim", 0.0, 1.0), n_cycles=6.0)
    assert not mask[0].any()
    assert mask[1].any()


def test_frequencies_drop_out_of_a_window_individually() -> None:
    # MNE's five-sigma half supports are approximately 0.298 / 0.239 / 0.199 s.
    # A window of half-width 0.22 s therefore holds only 12 Hz.
    tfr = _toy_tfr()
    spectra = Spectra.from_tfr(
        tfr, (Window("narrow", 0.0, 0.44),), recording="test", n_cycles=3.0, sfreq=200.0
    )
    assert np.isnan(spectra.data[:, :, 0, 0]).all()
    assert np.isnan(spectra.data[:, :, 0, 1]).all()
    assert np.isfinite(spectra.data[:, :, 0, 2]).all()
    assert (spectra.coverage[:, :, 0, :2] == 0.0).all()
    assert (spectra.coverage[:, :, 0, 2] > 0.0).all()


def test_a_window_narrower_than_every_wavelet_raises() -> None:
    # Every half-support exceeds this window's half-width, so nothing survives.
    # That is a specification error, not a data condition: the message says so.
    tfr = _toy_tfr()
    with pytest.raises(ValueError, match="retains no coefficients"):
        Spectra.from_tfr(
            tfr, (Window("tiny", 0.0, 0.1),), recording="test", n_cycles=3.0, sfreq=200.0
        )


def test_from_tfr_requires_the_morlet_cycle_count() -> None:
    with pytest.raises(TypeError, match="n_cycles"):
        Spectra.from_tfr(_toy_tfr(), (Window("stim", 0.0, 1.0),), recording="test", sfreq=200.0)


def test_from_tfr_requires_the_original_sampling_rate() -> None:
    # MNE reports the decimated rate as the TFR's own and keeps no record of the
    # original, so reading it off the object would scale a decimated TFR wrongly.
    with pytest.raises(TypeError, match="sfreq"):
        Spectra.from_tfr(_toy_tfr(), (Window("stim", 0.0, 1.0),), recording="test", n_cycles=3.0)


def test_from_tfr_expresses_morlet_power_as_a_density_independent_of_sampling_rate() -> None:
    # MNE's unit-energy wavelets return sfreq times the one-sided PSD, so the same
    # sine resampled to twice the rate reports twice the raw power. Dividing by the
    # rate makes both agree with each other and with a Welch density.
    mne = pytest.importorskip("mne")
    amplitude = 2e-6
    densities = []
    for sfreq in (200.0, 400.0):
        times = np.arange(0.0, 8.0, 1.0 / sfreq)
        data = (amplitude * np.sin(2.0 * np.pi * 10.0 * times))[np.newaxis, np.newaxis, :]
        info = mne.create_info(["C3"], sfreq, "eeg")
        epochs = mne.EpochsArray(data, info, tmin=0.0, verbose="ERROR")
        tfr = epochs.compute_tfr(
            "morlet", freqs=np.array([10.0]), n_cycles=7.0, return_itc=False, verbose="ERROR"
        )
        spectra = Spectra.from_tfr(
            tfr, (Window("mid", 2.0, 6.0),), recording="test", n_cycles=7.0, sfreq=sfreq
        )
        densities.append(spectra.data.item())
    np.testing.assert_allclose(densities[0], densities[1], rtol=1e-3)
    # A line of power A^2/2 seen through a wavelet of bandwidth sigma_f = f / n_cycles
    # spreads over roughly that many hertz, so the density is of order A^2 / 2 / sigma_f.
    assert 0.1 * amplitude**2 < densities[0] < amplitude**2


def test_support_fraction_is_distinct_from_finite_coverage() -> None:
    tfr = _toy_tfr()
    spectra = Spectra.from_tfr(
        tfr, (Window("stim", 0.0, 1.0),), recording="test", n_cycles=3.0, sfreq=200.0
    )

    assert (spectra.coverage == 1.0).all()
    assert (spectra.support < 1.0).all()
    assert (spectra.support > 0.0).all()


def test_event_immediately_outside_window_cannot_affect_retained_coefficients() -> None:
    mne = pytest.importorskip("mne")
    sfreq = 200.0
    info = mne.create_info(["C3"], sfreq, "eeg")
    data = np.zeros((2, 1, 401))
    times = -1.0 + np.arange(data.shape[-1]) / sfreq
    data[1, 0, np.flatnonzero(times < 0.0)[-1]] = 1.0
    epochs = mne.EpochsArray(data, info, tmin=-1.0, verbose="ERROR")
    tfr = epochs.compute_tfr(
        "morlet",
        freqs=np.array([10.0]),
        n_cycles=3.0,
        return_itc=False,
        verbose="ERROR",
    )

    spectra = Spectra.from_tfr(
        tfr, (Window("target", 0.0, 0.6),), recording="test", n_cycles=3.0, sfreq=200.0
    )

    np.testing.assert_allclose(spectra.data[1], spectra.data[0], atol=1e-14)


def test_from_spectrum_accepts_method_in_estimator_parameters() -> None:
    mne = pytest.importorskip("mne")
    info = mne.create_info(["C3"], 200.0, "eeg")
    epochs = mne.EpochsArray(
        np.random.RandomState(0).randn(2, 1, 400) * 1e-6, info, tmin=-1.0, verbose="ERROR"
    )
    spectrum = epochs.compute_psd("welch", fmin=2.0, fmax=40.0, verbose="ERROR")
    spectra = Spectra.from_spectrum(
        spectrum,
        recording="test",
        estimator_parameters={"method": "welch", "fmin": 2.0, "fmax": 40.0},
    )
    assert spectra.source == "welch"
    assert spectra.computation.method == "welch"


def test_from_spectrum_refuses_a_declared_method_the_spectrum_contradicts() -> None:
    # The density check keys on the method; a declaration that overrode the object would
    # let MNE's length-normalized multitaper PSD pass as a Welch density.
    spectrum = _epochs_for_hashing().compute_psd("multitaper", fmax=40.0, verbose=False)
    with pytest.raises(ValueError, match="multitaper"):
        Spectra.from_spectrum(spectrum, recording="r", estimator_parameters={"method": "welch"})


def test_from_spectrum_takes_the_declared_method_when_the_spectrum_does_not_know_it() -> None:
    mne = pytest.importorskip("mne")
    info = mne.create_info(["C3"], 100.0, "eeg")
    freqs = np.arange(1.0, 6.0)
    array = mne.time_frequency.EpochsSpectrumArray(
        np.ones((2, 1, 5)),
        info,
        freqs,
        events=np.array([[0, 0, 1], [100, 0, 1]]),
        event_id={"a": 1},
    )
    spectra = Spectra.from_spectrum(array, recording="r", estimator_parameters={"method": "welch"})
    assert spectra.source == "welch"


def test_array_spectra_without_event_id_name_events_by_code_as_mne_epochs_does() -> None:
    mne = pytest.importorskip("mne")
    info = mne.create_info(["C3"], 100.0, "eeg")
    array = mne.time_frequency.EpochsSpectrumArray(
        np.ones((2, 1, 5)), info, np.arange(1.0, 6.0), events=np.array([[0, 0, 1], [100, 0, 2]])
    )
    spectra = Spectra.from_spectrum(array, recording="r", estimator_parameters={"method": "welch"})
    assert spectra.row_ids == (("r", 0, "1"), ("r", 1, "2"))


def _epochs_for_hashing(sfreq: float = 500.0):
    mne = pytest.importorskip("mne")
    rng = np.random.default_rng(0)
    info = mne.create_info(["C3", "C4"], sfreq, "eeg")
    return mne.EpochsArray(
        rng.standard_normal((4, 2, int(4 * sfreq))) * 1e-5, info, tmin=0.0, verbose=False
    )


def _declared(spectrum) -> Spectra:
    # The same declaration in both arms: what varies is the data, not the dict. It names no
    # bounds, which a narrower arm would contradict and so be refused before it is hashed.
    return Spectra.from_spectrum(
        spectrum,
        recording="r",
        estimator_parameters={"method": "welch"},
    )


def test_a_different_frequency_axis_is_a_different_computation() -> None:
    epochs = _epochs_for_hashing()
    coarse = _declared(epochs.compute_psd(method="welch", fmin=1.0, fmax=45.0, verbose=False))
    narrow = _declared(epochs.compute_psd(method="welch", fmin=1.0, fmax=30.0, verbose=False))
    assert coarse.computation.parameter_hash != narrow.computation.parameter_hash


def test_a_different_sampling_rate_is_a_different_computation() -> None:
    slow = _declared(
        _epochs_for_hashing(250.0).compute_psd(method="welch", fmin=1.0, fmax=45.0, verbose=False)
    )
    fast = _declared(
        _epochs_for_hashing(500.0).compute_psd(method="welch", fmin=1.0, fmax=45.0, verbose=False)
    )
    assert slow.computation.parameter_hash != fast.computation.parameter_hash


def test_the_same_computation_still_hashes_the_same() -> None:
    epochs = _epochs_for_hashing()
    first = _declared(epochs.compute_psd(method="welch", fmin=1.0, fmax=45.0, verbose=False))
    again = _declared(epochs.compute_psd(method="welch", fmin=1.0, fmax=45.0, verbose=False))
    assert first.computation.parameter_hash == again.computation.parameter_hash
