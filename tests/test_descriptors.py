import numpy as np
import pytest
from scipy.signal import find_peaks, welch

from eegtable.bands import Band
from eegtable.descriptors import (
    _smooth,
    peak_frequency,
    spectral_bandwidth,
    spectral_centroid,
    spectral_edge,
    spectral_entropy,
)
from eegtable.spectra import Spectra, Window
from eegtable.table import ComputationSpec

ALPHA = Band("alpha", 8.0, 13.0)

# The corrections are on by default. These keywords select the bare
# interpolated argmax, which is what the tests immediately below are about.
PLAIN = {"aperiodic_adjusted": False, "smoothing_hz": 0.0, "min_prominence": 0.0}


def _spectra(power: np.ndarray, freqs: np.ndarray) -> Spectra:
    data = power.reshape(1, 1, 1, freqs.size)
    return Spectra(
        data=data,
        freqs=freqs,
        ch_names=("C3",),
        windows=(Window("all", -np.inf, np.inf),),
        coverage=np.isfinite(data).astype(float),
        source="test",
        representation="psd",
        support=np.ones(data.shape),
        row_ids=(("test", 0, "event"),),
        computation=ComputationSpec.create("test"),
    )


def _gaussian(freqs: np.ndarray, centre: float, width: float = 0.8) -> np.ndarray:
    return np.exp(-0.5 * ((freqs - centre) / width) ** 2)


def test_recovers_a_known_peak_to_better_than_the_bin_spacing() -> None:
    freqs = np.arange(8.0, 13.0, 0.5)
    table = peak_frequency(
        _spectra(_gaussian(freqs, 10.2), freqs), band=ALPHA, include_global=False, **PLAIN
    )
    assert table.values.item() == pytest.approx(10.2, abs=0.1)


@pytest.mark.parametrize("centre", [9.5, 10.0, 10.25])
def test_peak_interpolation_uses_actual_frequency_spacing(centre) -> None:
    freqs = np.array([8.0, 10.0, 11.0, 12.0])
    power = 10.0 - (freqs - centre) ** 2
    table = peak_frequency(_spectra(power, freqs), band=ALPHA, include_global=False, **PLAIN)
    assert table.values.item() == pytest.approx(centre)


def test_peak_on_a_band_edge_is_flagged() -> None:
    freqs = np.arange(8.0, 13.0, 0.5)
    rising = _spectra(np.linspace(1.0, 5.0, freqs.size), freqs)
    table = peak_frequency(rising, band=ALPHA, include_global=False, **PLAIN)
    assert table.flags["edge_hit"].all()


def test_an_interior_peak_is_not_flagged() -> None:
    freqs = np.arange(8.0, 13.0, 0.5)
    table = peak_frequency(
        _spectra(_gaussian(freqs, 10.2), freqs), band=ALPHA, include_global=False, **PLAIN
    )
    assert not table.flags["edge_hit"].any()


def test_resolution_is_reported_on_the_column() -> None:
    freqs = np.arange(8.0, 13.0, 0.5)
    table = peak_frequency(
        _spectra(_gaussian(freqs, 10.2), freqs), band=ALPHA, include_global=False, **PLAIN
    )
    assert table.meta[0].freq_resolution_hz == pytest.approx(0.5)
    assert table.meta[0].unit == "Hz"


def test_a_grid_too_coarse_for_an_interior_peak_raises() -> None:
    # A sparse grid with fewer than 3 bins in the band is below the
    # definition domain of the estimator and must not silently return a number.
    freqs = np.array([8.0, 11.0, 14.0])
    with pytest.raises(ValueError, match="3 bins"):
        peak_frequency(
            _spectra(np.array([1.0, 2.0, 1.0]), freqs), band=ALPHA, include_global=False, **PLAIN
        )


def test_all_nan_input_yields_nan_without_raising() -> None:
    freqs = np.arange(8.0, 13.0, 0.5)
    table = peak_frequency(
        _spectra(np.full(freqs.size, np.nan), freqs), band=ALPHA, include_global=False, **PLAIN
    )
    assert np.isnan(table.values).all()


UNIFORM = np.arange(8.0, 13.0, 0.25)


def test_centroid_of_a_flat_band_is_its_midpoint() -> None:
    table = spectral_centroid(
        _spectra(np.ones(UNIFORM.size), UNIFORM), band=ALPHA, include_global=False
    )
    assert table.values.item() == pytest.approx((UNIFORM[0] + UNIFORM[-1]) / 2.0)


def test_centroid_follows_where_the_power_is() -> None:
    table = spectral_centroid(
        _spectra(_gaussian(UNIFORM, 9.0, 0.3), UNIFORM), band=ALPHA, include_global=False
    )
    assert table.values.item() == pytest.approx(9.0, abs=0.1)


def test_bandwidth_of_a_flat_band_matches_a_uniform_distribution() -> None:
    # The discrete population std of n equally spaced bins, h*sqrt((n^2-1)/12).
    # NOT the continuous span/sqrt(12): for n = 20 those differ by 5%, because the
    # estimator sums over bins rather than integrating over the band.
    table = spectral_bandwidth(
        _spectra(np.ones(UNIFORM.size), UNIFORM), band=ALPHA, include_global=False
    )
    expected = np.sqrt(np.mean((UNIFORM - UNIFORM.mean()) ** 2))
    assert table.values.item() == pytest.approx(expected, rel=1e-12)


def test_bandwidth_is_smaller_for_a_narrower_peak() -> None:
    narrow = spectral_bandwidth(
        _spectra(_gaussian(UNIFORM, 10.5, 0.2), UNIFORM), band=ALPHA, include_global=False
    )
    wide = spectral_bandwidth(
        _spectra(_gaussian(UNIFORM, 10.5, 1.0), UNIFORM), band=ALPHA, include_global=False
    )
    assert narrow.values.item() < wide.values.item()


def test_entropy_of_a_flat_band_is_exactly_one() -> None:
    table = spectral_entropy(
        _spectra(np.ones(UNIFORM.size), UNIFORM), band=ALPHA, include_global=False
    )
    assert table.values.item() == pytest.approx(1.0)


def test_entropy_of_a_single_occupied_bin_is_zero() -> None:
    power = np.zeros(UNIFORM.size)
    power[5] = 1.0
    table = spectral_entropy(_spectra(power, UNIFORM), band=ALPHA, include_global=False)
    assert table.values.item() == pytest.approx(0.0, abs=1e-12)


def test_entropy_rejects_a_nonuniform_frequency_grid() -> None:
    freqs = np.array([8.0, 8.5, 9.25, 10.5, 12.0])

    with pytest.raises(ValueError, match="interpolate.*uniform-Hz grid"):
        spectral_entropy(_spectra(np.ones(freqs.size), freqs), band=ALPHA)


def test_edge_at_half_of_a_flat_band_is_near_its_midpoint() -> None:
    table = spectral_edge(
        _spectra(np.ones(UNIFORM.size), UNIFORM),
        band=ALPHA,
        percentile=0.5,
        include_global=False,
    )
    midpoint = (UNIFORM[0] + UNIFORM[-1]) / 2.0
    assert abs(table.values.item() - midpoint) <= 0.25


def test_edge_returns_a_grid_frequency_because_it_does_not_interpolate() -> None:
    table = spectral_edge(
        _spectra(np.ones(UNIFORM.size), UNIFORM),
        band=ALPHA,
        percentile=0.5,
        include_global=False,
    )
    assert table.values.item() in set(UNIFORM.tolist())


@pytest.mark.parametrize("trailing", [0.0, np.nan, np.inf])
def test_full_spectral_edge_stops_at_the_last_bin_with_power(trailing) -> None:
    freqs = np.arange(8.0, 12.0, 0.5)
    power = np.r_[np.full(6, 0.1), trailing, trailing]
    table = spectral_edge(_spectra(power, freqs), band=ALPHA, percentile=1.0, include_global=False)
    assert table.values.item() == freqs[5]


@pytest.mark.parametrize("percentile", [0.0, -0.1, 1.5, np.nan])
def test_an_out_of_range_percentile_raises(percentile: float) -> None:
    with pytest.raises(ValueError, match="percentile"):
        spectral_edge(
            _spectra(np.ones(UNIFORM.size), UNIFORM),
            band=ALPHA,
            percentile=percentile,
            include_global=False,
        )


def test_an_empty_band_yields_nan_for_every_descriptor() -> None:
    dead = _spectra(np.zeros(UNIFORM.size), UNIFORM)
    for fn in (spectral_centroid, spectral_bandwidth, spectral_entropy):
        assert np.isnan(fn(dead, band=ALPHA, include_global=False).values).all()


# --- the aperiodic-adjusted estimator ------------------------------------------------

WIDE = np.arange(2.0, 45.0, 0.25)


def _wide(power: np.ndarray) -> Spectra:
    data = power.reshape(1, 1, 1, WIDE.size)
    return Spectra(
        data=data,
        freqs=WIDE,
        ch_names=("C3",),
        windows=(Window("all", -np.inf, np.inf),),
        coverage=np.isfinite(data).astype(float),
        source="test",
        representation="psd",
        support=np.ones(data.shape),
        row_ids=(("test", 0, "event"),),
        computation=ComputationSpec.create("test"),
    )


def _alpha_on_a_slope(exponent: float = -3.5, amplitude: float = 0.35) -> np.ndarray:
    background = 10.0 * WIDE**exponent
    return background * (1.0 + amplitude * np.exp(-0.5 * ((WIDE - 10.5) / 0.6) ** 2))


def test_a_raw_argmax_reports_the_band_edge_on_a_steep_spectrum() -> None:
    # The failure the adjustment exists to fix, pinned so it stays visible.
    table = peak_frequency(_wide(_alpha_on_a_slope()), band=ALPHA, include_global=False, **PLAIN)
    assert table.values.item() == pytest.approx(8.0)
    assert table.flags["edge_hit"].all()


def test_the_adjustment_recovers_the_oscillation_the_raw_argmax_missed() -> None:
    table = peak_frequency(_wide(_alpha_on_a_slope()), band=ALPHA, include_global=False)
    assert table.values.item() == pytest.approx(10.5, abs=0.5)
    assert not table.flags["edge_hit"].any()


def test_smoothing_rejects_a_single_bin_spike() -> None:
    clean = 10.0 * WIDE**-1.5 * (1.0 + 0.8 * np.exp(-0.5 * ((WIDE - 11.0) / 0.5) ** 2))
    spiked = clean.copy()
    spiked[int(np.argmin(np.abs(WIDE - 8.75)))] *= 6.0
    fooled = peak_frequency(_wide(spiked), band=ALPHA, smoothing_hz=0.0, include_global=False)
    steady = peak_frequency(_wide(spiked), band=ALPHA, include_global=False)
    assert fooled.values.item() == pytest.approx(8.75)
    assert abs(steady.values.item() - 11.0) < abs(fooled.values.item() - 11.0)


@pytest.mark.parametrize("smoothing_hz", [1.0, 1.5, 2.0])
def test_smoothing_does_not_move_a_peak_that_sits_on_a_bin(smoothing_hz: float) -> None:
    # Windows of 2, 3 and 4 bins on this 0.5 Hz grid. An even window that is not
    # centred shifts the spectrum half a bin up and reported this sine at 10.25 Hz.
    sfreq = 250.0
    times = np.arange(int(60 * sfreq)) / sfreq
    noise = 0.05 * np.random.default_rng(0).standard_normal(times.size)
    freqs, power = welch(np.sin(2 * np.pi * 10.0 * times) + noise, fs=sfreq, nperseg=500)
    table = peak_frequency(
        _spectra(power, freqs), band=ALPHA, smoothing_hz=smoothing_hz, include_global=False
    )
    assert table.values.item() == pytest.approx(10.0, abs=0.05)


def test_smoothing_bandwidth_is_constant_in_hertz_on_a_log_grid() -> None:
    freqs = np.geomspace(1.0, 100.0, 120)
    values = np.zeros((1, 1, 1, freqs.size))
    centre = int(np.argmin(np.abs(freqs - 75.0)))
    values[..., centre] = 1.0
    smoothed = _smooth(values, freqs, smoothing_hz=10.0).ravel()
    affected = freqs[smoothed > 0.0]
    assert affected.max() - affected.min() <= 10.0 + np.max(np.diff(freqs))


@pytest.mark.parametrize("scale", [1.0, 1e-24])
def test_the_prominence_guard_does_not_depend_on_the_power_unit(scale: float) -> None:
    # Scaled into the range of a source estimate in A.m, every bin fell under the old
    # absolute floor, the log spectrum went flat, and every cell took the fallback.
    power = scale * 10.0 * WIDE**-1.0 * (1.0 + 2.0 * np.exp(-0.5 * ((WIDE - 10.5) / 0.6) ** 2))
    table = peak_frequency(_wide(power), band=ALPHA, aperiodic_adjusted=False, include_global=False)
    assert not table.flags["no_peak"].any()
    assert table.values.item() == pytest.approx(10.5, abs=0.1)


def test_a_spectrum_with_no_oscillation_has_no_peak() -> None:
    table = peak_frequency(_wide(10.0 * WIDE**-2.2), band=ALPHA, include_global=False)
    assert table.flags["no_peak"].all()
    assert not table.flags["edge_hit"].any()
    assert np.isnan(table.values.item())


@pytest.mark.parametrize("freqs", [np.array([1.0, 2.0, 10.0]), np.geomspace(1.0, 10.0, 30)])
def test_a_flat_spectrum_on_a_nonuniform_grid_has_no_peak(freqs) -> None:
    power = np.ones_like(freqs)
    table = peak_frequency(
        _spectra(power, freqs),
        band=Band("wide", 1.0, 11.0),
        aperiodic_adjusted=False,
        smoothing_hz=0.0,
        include_global=False,
    )
    assert table.flags["no_peak"].item()
    assert np.isnan(table.values.item())


def test_disabling_the_prominence_guard_returns_the_flagged_argmax() -> None:
    table = peak_frequency(
        _wide(10.0 * WIDE**-2.2), band=ALPHA, min_prominence=0.0, include_global=False
    )
    assert not table.flags["no_peak"].any()
    assert np.isfinite(table.values).all()


def test_the_measure_name_records_whether_the_adjustment_ran() -> None:
    spectra = _wide(_alpha_on_a_slope())
    adjusted = peak_frequency(spectra, band=ALPHA, include_global=False)
    plain = peak_frequency(spectra, band=ALPHA, include_global=False, **PLAIN)
    assert adjusted.meta[0].measure == "peak_freq_adjusted"
    assert plain.meta[0].measure == "peak_freq"


def test_the_source_records_the_whitening() -> None:
    table = peak_frequency(_wide(_alpha_on_a_slope()), band=ALPHA, include_global=False)
    assert table.meta[0].source == "test+aperiodic_ratio"


def test_without_interpolation_the_result_is_a_grid_frequency() -> None:
    table = peak_frequency(
        _wide(_alpha_on_a_slope(amplitude=2.0)),
        band=ALPHA,
        interpolate=False,
        min_prominence=0.0,
        include_global=False,
    )
    assert table.values.item() in set(WIDE.tolist())


def test_fit_range_without_the_adjustment_raises() -> None:
    with pytest.raises(ValueError, match="fit_range applies only"):
        peak_frequency(
            _wide(_alpha_on_a_slope()),
            band=ALPHA,
            aperiodic_adjusted=False,
            fit_range=(2.0, 40.0),
            include_global=False,
        )


@pytest.mark.parametrize("key", ["smoothing_hz", "min_prominence"])
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_tuning_values_raise(key: str, value: float) -> None:
    with pytest.raises(ValueError, match=key):
        peak_frequency(_wide(_alpha_on_a_slope()), band=ALPHA, include_global=False, **{key: value})


def test_peak_prominence_matches_scipy_on_log_power() -> None:
    freqs = np.arange(8.0, 13.0, 0.25)
    log_power = 3.0 - 0.5 * (freqs - 8.0)
    log_power[8] += 0.4
    power = 10.0**log_power
    indices, _ = find_peaks(log_power, prominence=0.2)
    expected = freqs[indices[np.argmax(power[indices])]]
    table = peak_frequency(
        _spectra(power, freqs),
        band=ALPHA,
        aperiodic_adjusted=False,
        smoothing_hz=0.0,
        min_prominence=0.2,
        interpolate=False,
        include_global=False,
    )
    assert table.values.item() == expected


@pytest.mark.parametrize("min_prominence", [0.0, 0.1])
def test_zero_power_has_no_peak(min_prominence: float) -> None:
    table = peak_frequency(
        _spectra(np.zeros(UNIFORM.size), UNIFORM),
        band=ALPHA,
        aperiodic_adjusted=False,
        min_prominence=min_prominence,
        include_global=False,
    )
    assert np.isnan(table.values.item())
    assert table.flags["no_peak"].item()


def test_smoothing_preserves_a_missing_frequency_bin() -> None:
    power = np.ones((1, 1, 1, UNIFORM.size))
    power[..., 5] = np.nan
    smoothed = _smooth(power, UNIFORM, smoothing_hz=1.0)
    assert np.isnan(smoothed[..., 5]).all()
    np.testing.assert_allclose(smoothed[..., np.arange(UNIFORM.size) != 5], 1.0)


@pytest.mark.parametrize("gap", [np.nan, np.inf])
def test_missing_bins_do_not_supply_peak_prominence(gap: float) -> None:
    freqs = np.arange(8.0, 13.0, 0.5)
    power = np.array([1.0, 2.0, gap, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    table = peak_frequency(
        _spectra(power, freqs),
        band=ALPHA,
        aperiodic_adjusted=False,
        smoothing_hz=0.0,
        include_global=False,
    )
    assert np.isnan(table.values.item())
    assert table.flags["no_peak"].item()


def test_smoothing_and_prominence_use_the_same_power_spectrum() -> None:
    freqs = np.arange(8.0, 13.0, 0.25)
    power = np.array(
        [
            16.0,
            5.0,
            3.0,
            6.0,
            8.0,
            16.0,
            9.0,
            2.0,
            7.0,
            12.0,
            16.0,
            14.0,
            19.0,
            4.0,
            17.0,
            2.0,
            11.0,
            6.0,
            4.0,
            13.0,
        ]
    )
    # Independent trapezoid smoothing for interior bins, over a 1 Hz window.
    smoothed = np.convolve(power, [0.125, 0.25, 0.25, 0.25, 0.125], mode="valid")
    candidates, _ = find_peaks(np.log10(smoothed), prominence=0.1)
    expected = freqs[2:-2][candidates[np.argmax(smoothed[candidates])]]
    table = peak_frequency(
        _spectra(power, freqs),
        band=ALPHA,
        aperiodic_adjusted=False,
        smoothing_hz=1.0,
        interpolate=False,
        include_global=False,
    )
    assert table.values.item() == expected
