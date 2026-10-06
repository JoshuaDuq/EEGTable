import json
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from eegtable.bands import Band
from eegtable.power import integrated_band_power, mean_psd, mean_tfr_power, periodic_power
from eegtable.spectra import Spectra, Window
from eegtable.table import ComputationSpec, FeatureTable

ALPHA = Band("alpha", 8.0, 13.0)


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


def test_flat_power_returns_that_value() -> None:
    freqs = np.arange(8.0, 13.5, 0.5)
    table = mean_psd(
        _spectra(np.full(freqs.size, 3.0), freqs), bands=(ALPHA,), include_global=False
    )
    np.testing.assert_allclose(table.values, 3.0)


def test_band_value_is_the_width_weighted_mean_not_the_bin_mean() -> None:
    # For P(f) = 2f the trapezoid rule is exact, so the width-weighted mean over
    # [f0, f1] is (f1^2 - f0^2) / (f1 - f0) = f0 + f1 on any grid at all. A plain
    # bin mean on a log grid is not, because it over-weights low frequencies.
    # Compare against that closed form rather than against np.trapezoid, which
    # does not exist below numpy 2.0 while this package supports numpy 1.26.
    # The grid sits strictly inside the band: np.logspace does not reproduce its
    # own endpoints exactly, so a grid ending on a band bound lands on the wrong
    # side of it by ~1e-15 and silently gains or loses a bin.
    freqs = np.r_[8.0, np.logspace(np.log10(8.5), np.log10(12.5), 9), 13.0]
    power = 2.0 * freqs
    table = mean_psd(_spectra(power, freqs), bands=(ALPHA,), include_global=False)
    assert table.values.item() == pytest.approx(ALPHA.fmin + ALPHA.fmax, rel=1e-12)
    assert table.values.item() != pytest.approx(power.mean(), rel=1e-4)


def test_integration_uses_the_exact_requested_boundaries() -> None:
    freqs = np.array([7.0, 9.0, 12.0, 14.0])
    power = 2.0 * freqs
    table = integrated_band_power(_spectra(power, freqs), bands=(ALPHA,), include_global=False)
    expected = ALPHA.fmax**2 - ALPHA.fmin**2
    assert table.values.item() == pytest.approx(expected, rel=1e-12)


def test_integrated_power_is_stable_across_linear_and_log_grids() -> None:
    linear = np.linspace(7.0, 14.0, 31)
    logarithmic = np.geomspace(7.0, 14.0, 47)
    results = [
        integrated_band_power(
            _spectra(3.0 * axis + 2.0, axis), bands=(ALPHA,), include_global=False
        ).values.item()
        for axis in (linear, logarithmic)
    ]
    expected = 1.5 * (ALPHA.fmax**2 - ALPHA.fmin**2) + 2.0 * (ALPHA.fmax - ALPHA.fmin)
    np.testing.assert_allclose(results, expected, rtol=1e-12)


def test_nan_frequencies_are_excluded_rather_than_poisoning_the_band() -> None:
    freqs = np.arange(8.0, 13.5, 0.5)
    power = np.full(freqs.size, 4.0)
    power[2] = np.nan
    table = mean_psd(_spectra(power, freqs), bands=(ALPHA,), include_global=False)
    assert table.values.item() == pytest.approx(4.0)
    assert table.coverage.item() < 1.0


def test_an_all_nan_band_yields_nan_and_zero_coverage() -> None:
    freqs = np.arange(8.0, 13.5, 0.5)
    table = mean_psd(
        _spectra(np.full(freqs.size, np.nan), freqs), bands=(ALPHA,), include_global=False
    )
    assert np.isnan(table.values).all()
    assert table.coverage.item() == 0.0


def test_unit_and_normalization_are_recorded() -> None:
    freqs = np.arange(8.0, 13.5, 0.5)
    table = mean_psd(
        _spectra(np.full(freqs.size, 3.0), freqs), bands=(ALPHA,), include_global=False
    )
    assert table.meta[0].normalization == "raw"
    assert table.meta[0].unit == "V^2/Hz"
    assert table.meta[0].measure == "mean_psd"


def test_psd_and_tfr_reductions_reject_the_wrong_representation() -> None:
    freqs = np.arange(8.0, 13.5, 0.5)
    psd = _spectra(np.ones(freqs.size), freqs)
    tfr = replace(psd, representation="time_frequency_power")
    with pytest.raises(ValueError, match="requires spectral representation"):
        mean_psd(tfr, bands=(ALPHA,))
    with pytest.raises(ValueError, match="requires spectral representation"):
        integrated_band_power(tfr, bands=(ALPHA,))
    assert np.isfinite(mean_tfr_power(tfr, bands=(ALPHA,)).values).all()


@pytest.mark.parametrize(
    ("reduction", "representation"),
    [
        (integrated_band_power, "psd"),
        (mean_psd, "psd"),
        (mean_tfr_power, "time_frequency_power"),
    ],
)
def test_percent_is_change_from_the_baseline_window(
    reduction: Callable[..., FeatureTable], representation: str
) -> None:
    # Flat power of 2 in the baseline and 3 in the stimulus: (3 - 2) / 2 = +50%,
    # whether the band value is an integral or a mean.
    freqs = np.arange(8.0, 13.5, 0.5)
    data = np.stack([np.full(freqs.size, 2.0), np.full(freqs.size, 3.0)])[np.newaxis, np.newaxis]
    spectra = replace(
        _spectra(np.ones(freqs.size), freqs),
        data=data,
        windows=(Window("baseline", -1.0, 0.0), Window("stimulus", 0.0, 1.0)),
        coverage=np.ones(data.shape),
        support=np.ones(data.shape),
        representation=representation,
    )
    table = reduction(
        spectra, bands=(ALPHA,), include_global=False, baseline="baseline", normalize="percent"
    )
    stimulus = table.select(window="stimulus")
    np.testing.assert_allclose(stimulus.values, 50.0)
    assert stimulus.meta[0].normalization == "percent"
    assert stimulus.meta[0].unit == "%"


def _tfr_baseline_and_stimulus(stimulus_over_baseline: list[float], scale: float) -> Spectra:
    # Morlet power of an EEG channel near 1e-12 V^2/Hz; the baseline window holds 2e-12,
    # and each epoch's stimulus window the given multiple of it.
    freqs = np.arange(8.0, 13.5, 0.5)
    ratios = np.array(stimulus_over_baseline)
    data = scale * 2e-12 * np.ones((ratios.size, 1, 2, freqs.size))
    data[:, :, 1] *= ratios[:, np.newaxis, np.newaxis]
    return replace(
        _spectra(np.ones(freqs.size), freqs),
        data=data,
        windows=(Window("baseline", -1.0, 0.0), Window("stimulus", 0.0, 1.0)),
        coverage=np.ones(data.shape),
        support=np.ones(data.shape),
        representation="time_frequency_power",
        row_ids=tuple(("test", index, "event") for index in range(ratios.size)),
    )


@pytest.mark.parametrize(
    ("normalize", "expected"),
    [
        ("log_ratio", np.log10([0.5, 1.5, 3.0])),
        ("db", 10.0 * np.log10([0.5, 1.5, 3.0])),
        ("percent", [-50.0, 50.0, 200.0]),
    ],
)
@pytest.mark.parametrize("scale", [1.0, 1e-12])
def test_normalized_tfr_power_does_not_depend_on_the_power_unit(
    normalize: str, expected: list[float], scale: float
) -> None:
    # eLORETA source power in A.m arrives near 1e-21, under the old absolute floor of
    # 1e-20: every trial came out the same constant, and a variance filter downstream
    # found no usable feature.
    spectra = _tfr_baseline_and_stimulus([0.5, 1.5, 3.0], scale)
    table = mean_tfr_power(
        spectra, bands=(ALPHA,), include_global=False, baseline="baseline", normalize=normalize
    )
    np.testing.assert_allclose(table.values[:, 0], expected, rtol=1e-12)


def test_log10_tfr_power_moves_by_the_log_of_the_unit_and_keeps_its_name() -> None:
    native, rescaled = (
        mean_tfr_power(
            _tfr_baseline_and_stimulus([0.5, 1.5, 3.0], scale),
            bands=(ALPHA,),
            include_global=False,
            normalize="log10",
        )
        for scale in (1.0, 1e-12)
    )
    np.testing.assert_allclose(rescaled.values, native.values - 12.0, rtol=1e-12)
    # The floor follows the data, so nothing about it belongs in a column's identity.
    assert rescaled.names == native.names


@pytest.mark.parametrize("normalize", ["log_ratio", "db", "percent"])
def test_baseline_quality_is_carried_into_normalized_power(normalize):
    freqs = np.arange(8.0, 14.0)
    data = np.ones((1, 1, 2, freqs.size))
    coverage = np.ones_like(data)
    coverage[:, :, 0] = 0.25
    spectra = replace(
        _spectra(np.ones(freqs.size), freqs),
        data=data,
        windows=(Window("base", -1.0, 0.0), Window("stim", 0.0, 1.0)),
        coverage=coverage,
        support=np.ones_like(data),
        flags={"insufficient_support": np.array([[[True, False]]])},
    )
    table = integrated_band_power(
        spectra, bands=(ALPHA,), baseline="base", normalize=normalize, include_global=False
    )
    assert table.coverage.item() == pytest.approx(0.25)
    assert table.flags["insufficient_support"].item()
    np.testing.assert_array_equal(spectra.flags["insufficient_support"], [[[True, False]]])


def test_normalized_feature_identity_includes_baseline_bounds():
    freqs = np.arange(8.0, 14.0)
    data = np.ones((1, 1, 2, freqs.size))
    spectra = replace(
        _spectra(np.ones(freqs.size), freqs),
        data=data,
        windows=(Window("base", -1.0, 0.0), Window("stim", 0.0, 1.0)),
        coverage=np.ones_like(data),
        support=np.ones_like(data),
    )
    longer_baseline = replace(spectra, windows=(Window("base", -2.0, 0.0), spectra.windows[1]))
    tables = [
        integrated_band_power(
            source, bands=(ALPHA,), baseline="base", normalize="db", include_global=False
        )
        for source in (spectra, longer_baseline)
    ]
    assert tables[0].names != tables[1].names


# --- periodic_power ------------------------------------------------------------------

# geomspace keeps the 40 Hz bin exact, which decides whether the half-open fit range keeps it.
FIT_FREQS = np.geomspace(2.0, 40.0, 60)
BACKGROUND = 10.0 * FIT_FREQS**-1.7
AGAINST_BASELINE = {
    "bands": (ALPHA,),
    "include_global": False,
    "baseline": "baseline",
    "normalize": "log_ratio",
}


def _alpha_peak(height: float) -> np.ndarray:
    return 1.0 + height * np.exp(-0.5 * ((FIT_FREQS - 10.0) / 1.0) ** 2)


def _baseline_and_stimulus(baseline: np.ndarray, stimulus: np.ndarray) -> Spectra:
    data = np.stack([baseline, stimulus])[np.newaxis, np.newaxis]
    return replace(
        _spectra(baseline, FIT_FREQS),
        data=data,
        windows=(Window("baseline", -1.0, 0.0), Window("stimulus", 0.0, 1.0)),
        coverage=np.ones(data.shape),
        support=np.ones(data.shape),
    )


def test_a_pure_power_law_has_unit_periodic_power() -> None:
    table = periodic_power(_spectra(BACKGROUND, FIT_FREQS), bands=(ALPHA,), include_global=False)
    np.testing.assert_allclose(table.values, 1.0, rtol=1e-6)


def test_a_broadband_gain_moves_band_power_but_not_periodic_power() -> None:
    # The whole spectrum four times higher around the same oscillation: band power rises
    # by the gain, while the power above the fitted 1/f component does not move.
    peaked = BACKGROUND * _alpha_peak(1.5)
    spectra = _baseline_and_stimulus(peaked, 4.0 * peaked)
    band = mean_psd(spectra, **AGAINST_BASELINE).values.item()
    assert band == pytest.approx(np.log10(4.0), rel=1e-12)
    periodic = periodic_power(spectra, **AGAINST_BASELINE).values.item()
    assert periodic == pytest.approx(0.0, abs=1e-12)


def test_an_oscillation_masked_by_a_broadband_rise_is_recovered() -> None:
    # Alpha falls while every frequency triples: band power reports a rise, the power
    # above the 1/f component the fall.
    spectra = _baseline_and_stimulus(
        BACKGROUND * _alpha_peak(1.5), 3.0 * BACKGROUND * _alpha_peak(0.75)
    )
    assert mean_psd(spectra, **AGAINST_BASELINE).values.item() > 0.0
    assert periodic_power(spectra, **AGAINST_BASELINE).values.item() < 0.0


def test_periodic_power_is_the_same_for_a_psd_and_time_frequency_power() -> None:
    psd = _spectra(BACKGROUND * _alpha_peak(1.5), FIT_FREQS)
    tfr = replace(psd, representation="time_frequency_power")
    np.testing.assert_array_equal(
        periodic_power(psd, bands=(ALPHA,)).values, periodic_power(tfr, bands=(ALPHA,)).values
    )


def test_a_cell_whose_aperiodic_fit_fails_is_withheld_and_flagged() -> None:
    power = np.full(FIT_FREQS.size, np.nan)
    power[:4] = 1.0
    table = periodic_power(_spectra(power, FIT_FREQS), bands=(ALPHA,), include_global=False)
    assert np.isnan(table.values).all()
    assert table.flags["aperiodic_fit_failed"].all()


@pytest.mark.parametrize(
    ("normalize", "unit"), [("raw", "ratio to the aperiodic fit"), ("db", "dB")]
)
def test_periodic_power_is_named_with_its_unit_and_its_fit(normalize: str, unit: str) -> None:
    spectra = _baseline_and_stimulus(BACKGROUND, BACKGROUND)
    table = periodic_power(
        spectra,
        bands=(ALPHA,),
        include_global=False,
        baseline="baseline" if normalize != "raw" else None,
        normalize=normalize,
        fit_range=(3.0, 30.0),
    )
    meta = table.meta[0]
    assert (meta.measure, meta.unit, meta.normalization) == ("periodic_power", unit, normalize)
    fit = json.loads(meta.computation.parameters_json)["input_computation"]
    assert (fit["method"], fit["parameters"]["fit_range"]) == ("aperiodic_ratio", [3.0, 30.0])


def test_tfr_power_carries_how_much_of_its_window_the_wavelets_supported() -> None:
    # Only coefficients whose whole wavelet fits the window are averaged. How much of the
    # window that leaves must reach the table, or a value resting on a sliver reads as sound.
    spectra = _tfr_baseline_and_stimulus([1.0, 2.0], scale=1.0)
    support = np.ones(spectra.data.shape)
    support[:, :, 1] = 0.25
    table = mean_tfr_power(replace(spectra, support=support), bands=[ALPHA], include_global=False)
    stimulus = [i for i, m in enumerate(table.meta) if m.window == "stimulus"]
    assert table.support is not None
    np.testing.assert_allclose(table.support[:, stimulus], 0.25)


def test_a_baseline_normalized_value_rests_on_the_less_supported_window() -> None:
    spectra = _tfr_baseline_and_stimulus([1.0, 2.0], scale=1.0)
    support = np.ones(spectra.data.shape)
    support[:, :, 0] = 0.1
    table = mean_tfr_power(
        replace(spectra, support=support),
        bands=[ALPHA],
        include_global=False,
        baseline="baseline",
        normalize="db",
    )
    assert table.support is not None
    np.testing.assert_allclose(table.support, 0.1)


def test_psd_power_records_no_support_restriction() -> None:
    freqs = np.linspace(0.0, 40.0, 41)
    table = integrated_band_power(_spectra(np.ones(41), freqs), bands=[ALPHA], include_global=False)
    assert table.support is None


def test_epochs_passed_for_spectra_name_the_constructors() -> None:
    # The epochs object reached a missing attribute deep inside; say what to build instead.
    from tests.synthetic import make_epochs

    with pytest.raises(TypeError, match="Spectra.welch"):
        integrated_band_power(make_epochs(), bands=[ALPHA])  # type: ignore[arg-type]


def test_an_unknown_normalization_lists_the_known_ones() -> None:
    freqs = np.linspace(0.0, 40.0, 41)
    with pytest.raises(ValueError, match="log10"):
        integrated_band_power(_spectra(np.ones(41), freqs), bands=[ALPHA], normalize="relative")  # type: ignore[arg-type]
