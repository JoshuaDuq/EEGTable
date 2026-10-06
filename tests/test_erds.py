import numpy as np
import pytest
from scipy.stats import linregress

import eegtable as ef
from eegtable.bands import Band
from eegtable.erds import (
    ErdsScale,
    erd_duration,
    erd_magnitude,
    erds_mean,
    erds_onset_latency,
    erds_peak_latency,
    erds_rebound_latency,
    erds_slope,
    ers_duration,
    ers_magnitude,
)
from eegtable.signal import BandSignal
from eegtable.spectra import Window

ERDS_FUNCTIONS = {
    "erds_mean": erds_mean,
    "erds_slope": erds_slope,
    "erd_magnitude": erd_magnitude,
    "erd_duration": erd_duration,
    "ers_magnitude": ers_magnitude,
    "ers_duration": ers_duration,
    "erds_peak_latency": erds_peak_latency,
    "erds_onset_latency": erds_onset_latency,
    "erds_rebound_latency": erds_rebound_latency,
}


def erds(signals, **kwargs):
    """Test-only: every ERDS measure in one table.

    The library exposes one function per measure so a caller asks for exactly what
    they want. These tests assert relationships across measures, so they rebuild
    the combined view here rather than in the public namespace.
    """
    return ef.concat([fn(signals, **kwargs) for fn in ERDS_FUNCTIONS.values()])


ALPHA = Band("alpha", 8.0, 13.0)
SFREQ = 100.0
BASE = Window("base", -1.0, -1.0 / SFREQ)
STIM = Window("stim", 0.0, 1.0)


def _signal(envelope: np.ndarray) -> BandSignal:
    times = np.arange(envelope.shape[-1]) / SFREQ - 1.0
    return BandSignal.from_arrays(
        analytic=envelope.astype(complex),
        times=times,
        ch_names=("C3",),
        band=ALPHA,
        sfreq=SFREQ,
        row_ids=tuple(("test", index, "event") for index in range(envelope.shape[0])),
    )


def _step(baseline_amp: float, active_amp: float, n: int = 201) -> BandSignal:
    envelope = np.full((1, 1, n), baseline_amp)
    envelope[:, :, n // 2 :] = active_amp
    return _signal(envelope)


@pytest.mark.parametrize("function", list(ERDS_FUNCTIONS.values()))
@pytest.mark.parametrize("missing", [np.nan, np.inf])
def test_erds_coverage_includes_missing_baseline_samples(function, missing) -> None:
    envelope = np.ones((1, 1, 201))
    envelope[:, :, :90] = missing
    envelope[:, :, 100:] = np.sqrt(2.0)
    table = function([_signal(envelope)], baseline=BASE, windows=[STIM], include_global=False)
    assert table.coverage.item() == pytest.approx(0.1)


def test_erds_limits_channel_coverage_before_spatial_aggregation() -> None:
    signal = _step(1.0, np.sqrt(2.0))
    analytic = np.repeat(signal.analytic, 2, axis=1)
    coverage = np.ones(analytic.shape)
    coverage[:, 0, :100] = 0.25
    coverage[:, 1, 100:] = 0.5
    paired = BandSignal.from_arrays(
        analytic=analytic,
        times=signal.times,
        ch_names=("C3", "C4"),
        band=ALPHA,
        sfreq=SFREQ,
        row_ids=signal.row_ids,
        coverage=coverage,
    )
    table = erds_mean(
        [paired],
        baseline=BASE,
        windows=[STIM],
        groups={"central": ("C3", "C4")},
        include_global=True,
    )
    np.testing.assert_allclose(table.coverage, 0.375)
    np.testing.assert_allclose(table.values, 10.0 * np.log10(2.0))


@pytest.mark.parametrize(
    "measure", [fn for name, fn in ERDS_FUNCTIONS.items() if name != "erds_onset_latency"]
)
def test_non_onset_measures_accept_a_band_starting_at_zero(measure) -> None:
    from dataclasses import replace

    signal = replace(_step(1.0, 2.0), band=Band("lowpass", 0.0, 4.0))
    result = measure([signal], baseline=BASE, windows=[STIM], include_global=False)
    assert np.isfinite(result.values).all()


def test_onset_requires_the_full_minimum_duration() -> None:
    envelope = np.ones((1, 1, 201))
    envelope[:, :, 110:120] = 2.0
    result = erds_onset_latency(
        [_signal(envelope)],
        baseline=BASE,
        windows=[STIM],
        min_duration_ms=104.0,
        include_global=False,
    )
    assert np.isnan(result.values).all()


def test_onset_accepts_an_exact_duration_despite_roundoff() -> None:
    envelope = np.ones((1, 1, 201))
    envelope[:, :, 110:124] = 2.0
    result = erds_onset_latency(
        [_signal(envelope)],
        baseline=BASE,
        windows=[STIM],
        min_duration_ms=140.0,
        include_global=False,
    )
    assert result.values.item() == pytest.approx(0.1)


def test_a_halved_power_gives_minus_fifty_percent() -> None:
    # amplitude 1 -> power 1 in baseline; amplitude sqrt(0.5) -> power 0.5 active
    table = erds(
        [_step(1.0, np.sqrt(0.5))],
        baseline=BASE,
        windows=[STIM],
        include_global=False,
        normalize="percent",
    )
    assert table.select(measure="erds_mean").values.item() == pytest.approx(-50.0)


def test_the_same_signal_against_its_own_baseline_is_zero() -> None:
    table = erds([_step(1.0, 1.0)], baseline=BASE, windows=[STIM], include_global=False)
    assert table.select(measure="erds_mean").values.item() == pytest.approx(0.0)


def test_db_and_percent_agree_on_a_doubling() -> None:
    signal = _step(1.0, np.sqrt(2.0))
    pct = (
        erds(
            [signal],
            baseline=BASE,
            windows=[STIM],
            include_global=False,
            normalize="percent",
        )
        .select(measure="erds_mean")
        .values.item()
    )
    db = (
        erds(
            [signal],
            baseline=BASE,
            windows=[STIM],
            include_global=False,
            normalize="db",
        )
        .select(measure="erds_mean")
        .values.item()
    )
    assert pct == pytest.approx(100.0)
    assert db == pytest.approx(10.0 * np.log10(2.0))


def test_a_wholly_negative_trace_has_full_erd_duration_and_no_ers() -> None:
    table = erds(
        [_step(1.0, np.sqrt(0.5))],
        baseline=BASE,
        windows=[STIM],
        include_global=False,
        normalize="percent",
    )
    assert table.select(measure="erd_magnitude").values.item() == pytest.approx(50.0)
    assert table.select(measure="erd_duration").values.item() == pytest.approx(101 / SFREQ)
    assert table.select(measure="ers_magnitude").values.item() == 0.0
    assert table.select(measure="ers_duration").values.item() == 0.0


def test_slope_recovers_a_known_linear_ramp() -> None:
    n = 201
    envelope = np.ones((1, 1, n))
    # power ramps from 1.0 to 2.0 across the 1 s active window -> +100%/s in percent
    envelope[:, :, n // 2 :] = np.sqrt(np.linspace(1.0, 2.0, n - n // 2))
    table = erds(
        [_signal(envelope)],
        baseline=BASE,
        windows=[STIM],
        include_global=False,
        normalize="percent",
    )
    assert table.select(measure="erds_slope").values.item() == pytest.approx(100.0, rel=0.02)


@pytest.mark.parametrize("offset", [0.0, 100_000.0])
@pytest.mark.parametrize("gap", [None, np.nan, np.inf])
@pytest.mark.parametrize("normalize", ["percent", "db"])
def test_slope_matches_scipy_with_shifted_times_and_missing_samples(offset, gap, normalize) -> None:
    sfreq = 1000.0
    times = offset + np.arange(300) / sfreq
    power = np.ones(300)
    power[100:] += times[100:] - times[100]
    if gap is not None:
        power[150:160] = gap
    signal = BandSignal.from_arrays(
        analytic=np.sqrt(power).astype(complex).reshape(1, 1, -1),
        times=times,
        ch_names=("C3",),
        band=ALPHA,
        sfreq=sfreq,
        row_ids=(("test", 0, "event"),),
    )
    table = erds_slope(
        [signal],
        baseline=Window("base", times[0], times[99]),
        windows=[Window("stim", times[100], times[-1])],
        include_global=False,
        normalize=normalize,
    )
    finite = np.isfinite(power[100:])
    active = power[100:][finite]
    trace = 100 * (active - 1) if normalize == "percent" else 10 * np.log10(active)
    reference = linregress(times[100:][finite], trace).slope
    assert table.values.item() == pytest.approx(reference, rel=1e-10)
    assert table.meta[0].unit == ("%/s" if normalize == "percent" else "dB/s")


@pytest.mark.parametrize("n_finite", [0, 1, 2, 3])
def test_slope_requires_three_finite_samples(n_finite) -> None:
    envelope = np.ones((1, 1, 201))
    envelope[:, :, 100:] = np.nan
    envelope[:, :, 100 : 100 + n_finite] = np.sqrt(1 + np.arange(n_finite) / SFREQ)
    table = erds_slope(
        [_signal(envelope)],
        baseline=BASE,
        windows=[STIM],
        normalize="percent",
        include_global=False,
    )
    if n_finite < 3:
        assert np.isnan(table.values.item())
    else:
        assert table.values.item() == pytest.approx(100.0)


def test_an_unusable_baseline_yields_nan_rather_than_an_enormous_ratio() -> None:
    envelope = np.zeros((1, 1, 201))
    envelope[:, :, 100:] = 1.0
    table = erds([_signal(envelope)], baseline=BASE, windows=[STIM], include_global=False)
    assert np.isnan(table.select(measure="erds_mean").values).all()


def test_erds_without_a_baseline_window_is_impossible_to_call() -> None:
    with pytest.raises(TypeError):
        erds([_step(1.0, 1.0)], windows=[STIM])  # type: ignore[call-arg]


def test_an_unknown_normalization_raises() -> None:
    with pytest.raises(ValueError, match="normalize"):
        erds([_step(1.0, 1.0)], baseline=BASE, windows=[STIM], normalize="log10")  # type: ignore[arg-type]


def test_peak_latency_finds_the_largest_excursion() -> None:
    n = 201
    envelope = np.ones((1, 1, n))
    envelope[:, :, 150] = np.sqrt(3.0)  # t = +0.5 s
    table = erds([_signal(envelope)], baseline=BASE, windows=[STIM], include_global=False)
    assert table.select(measure="erds_peak_latency").values.item() == pytest.approx(0.5)


def test_onset_latency_is_the_first_crossing_of_the_baseline_variability() -> None:
    n = 201
    rng = np.random.RandomState(0)
    envelope = np.ones((1, 1, n))
    envelope[:, :, :100] += rng.normal(0.0, 0.01, (1, 1, 100))
    envelope[:, :, 130:] = np.sqrt(2.0)  # steps at t = +0.3 s
    table = erds_onset_latency(
        [_signal(envelope)],
        baseline=BASE,
        windows=[STIM],
        include_global=False,
        min_duration_ms=100.0,
    )
    assert table.values.item() == pytest.approx(0.3, abs=0.02)


def test_onset_is_independent_of_the_reported_normalization_scale() -> None:
    n = 201
    rng = np.random.RandomState(22)
    envelope = np.ones((1, 1, n))
    envelope[:, :, :100] += rng.normal(0.0, 0.03, (1, 1, 100))
    envelope[:, :, 140:] = np.sqrt(1.25)
    signal = _signal(envelope)
    percent = erds_onset_latency(
        [signal], baseline=BASE, windows=[STIM], normalize="percent", include_global=False
    )
    db = erds_onset_latency(
        [signal], baseline=BASE, windows=[STIM], normalize="db", include_global=False
    )
    np.testing.assert_allclose(percent.values, db.values, equal_nan=True)


def test_a_trace_that_never_crosses_has_no_onset() -> None:
    table = erds_onset_latency(
        [_step(1.0, 1.0)],
        baseline=BASE,
        windows=[STIM],
        include_global=False,
        min_duration_ms=100.0,
    )
    assert np.isnan(table.values).all()


def test_rebound_is_the_largest_value_after_the_peak() -> None:
    n = 201
    envelope = np.ones((1, 1, n))
    envelope[:, :, 120] = np.sqrt(0.1)  # deep ERD at +0.2 s, the largest excursion
    envelope[:, :, 180] = np.sqrt(1.5)  # smaller ERS at +0.8 s
    table = erds([_signal(envelope)], baseline=BASE, windows=[STIM], include_global=False)
    assert table.select(measure="erds_peak_latency").values.item() == pytest.approx(0.2)
    assert table.select(measure="erds_rebound_latency").values.item() == pytest.approx(0.8)


def test_a_peak_at_the_window_end_leaves_no_rebound() -> None:
    n = 201
    envelope = np.ones((1, 1, n))
    envelope[:, :, -1] = np.sqrt(5.0)
    table = erds([_signal(envelope)], baseline=BASE, windows=[STIM], include_global=False)
    assert np.isnan(table.select(measure="erds_rebound_latency").values).all()


def test_an_enormous_response_is_reported_but_flagged() -> None:
    # A 0.1 uV baseline against a 100 uV response is an ERDS of order 1e8 percent.
    # Suspicious, but a measurement: it is reported, and the flag says so.
    n = 201
    envelope = np.full((1, 1, n), 1e-7)
    envelope[:, :, n // 2 :] = 1e-4
    table = erds_mean(
        [_signal(envelope)],
        baseline=BASE,
        windows=[STIM],
        include_global=False,
        normalize="percent",
    )
    assert table.values.item() == pytest.approx(1e6 * 100.0 - 100.0)
    assert table.flags["baseline_extreme_ratio"].all()
    assert not table.flags["baseline_degenerate"].any()


def test_a_healthy_channel_is_not_withheld_by_the_guard() -> None:
    n = 201
    envelope = np.full((1, 1, n), 1e-5)
    envelope[:, :, n // 2 :] = 1e-5 * np.sqrt(0.5)
    table = erds_mean(
        [_signal(envelope)],
        baseline=BASE,
        windows=[STIM],
        include_global=False,
        normalize="percent",
    )
    assert table.values.item() == pytest.approx(-50.0)
    assert not table.flags["baseline_extreme_ratio"].any()
    assert not table.flags["baseline_degenerate"].any()


def test_a_low_amplitude_band_is_not_withheld_for_being_small() -> None:
    # The guard used to be an absolute 1e-12 V^2, which discards a sub-microvolt
    # gamma envelope: ordinary EEG, not a fault. The result must depend only on the
    # ratio, so the same data in volts and in microvolts has to agree.
    n = 201
    envelope = np.full((1, 1, n), 5e-7)  # 0.5 uV: power 2.5e-13, under the old floor
    envelope[:, :, n // 2 :] = 5e-7 * np.sqrt(0.5)
    volts = erds_mean(
        [_signal(envelope)],
        baseline=BASE,
        windows=[STIM],
        include_global=False,
        normalize="percent",
    )
    micro = erds_mean(
        [_signal(envelope * 1e6)],
        baseline=BASE,
        windows=[STIM],
        include_global=False,
        normalize="percent",
    )
    assert volts.values.item() == pytest.approx(-50.0)
    assert volts.values.item() == pytest.approx(micro.values.item())
    assert not volts.flags["baseline_degenerate"].any()


@pytest.mark.parametrize(
    ("normalize", "expected"), [("percent", -50.0), ("db", 10.0 * np.log10(0.5))]
)
def test_a_source_estimate_in_ampere_metres_is_not_flattened(
    normalize: ErdsScale, expected: float
) -> None:
    # eLORETA amplitude near 3e-11 A.m: power 9e-22, under the old absolute floor of
    # 1e-20, so decibels read 0 and percent was measured against the floor.
    n = 201
    envelope = np.full((1, 1, n), 3e-11)
    envelope[:, :, n // 2 :] = 3e-11 * np.sqrt(0.5)
    table = erds_mean(
        [_signal(envelope)],
        baseline=BASE,
        windows=[STIM],
        include_global=False,
        normalize=normalize,
    )
    assert table.values.item() == pytest.approx(expected)


def test_a_baseline_with_no_signal_at_all_is_withheld_and_flagged() -> None:
    # A channel that dropped out through the baseline window cannot anchor a ratio
    # at any scale, so it is NaN and says why.
    n = 201
    envelope = np.full((1, 1, n), 1e-5)
    envelope[:, :, : n // 2] = 0.0
    table = erds_mean([_signal(envelope)], baseline=BASE, windows=[STIM], include_global=False)
    assert np.isnan(table.values).all()
    assert table.flags["baseline_degenerate"].all()


# --- the split public surface ---------------------------------------------------------


@pytest.mark.parametrize(("measure", "function"), list(ERDS_FUNCTIONS.items()))
def test_each_function_returns_exactly_its_own_measure(measure, function) -> None:
    table = function(
        [_step(1.0, np.sqrt(0.5))], baseline=BASE, windows=[STIM], include_global=False
    )
    assert table.values.shape == (1, 1)
    assert [m.measure for m in table.meta] == [measure]


def test_the_functions_agree_with_the_combined_view() -> None:
    signals = [_step(1.0, np.sqrt(0.5))]
    combined = erds(signals, baseline=BASE, windows=[STIM], include_global=False)
    for measure, function in ERDS_FUNCTIONS.items():
        alone = function(signals, baseline=BASE, windows=[STIM], include_global=False)
        np.testing.assert_allclose(
            alone.values, combined.select(measure=measure).values, equal_nan=True
        )


def test_erds_labels_do_not_collide_with_other_measures() -> None:
    # Selection and aggregate_by identify a column by its measure label alone, so a
    # label ERDS shares with another measure silently mixes the two: the 1/f exponent
    # with an ERDS time-course slope, or a waveform peak latency with an ERDS one.
    freqs = np.geomspace(2.0, 40.0, 60)
    power = (10.0 * freqs**-1.7).reshape(1, 1, 1, freqs.size)
    spectra = ef.Spectra(
        data=power,
        freqs=freqs,
        ch_names=("C3",),
        windows=(Window("all", -np.inf, np.inf),),
        coverage=np.ones(power.shape),
        source="test",
        representation="psd",
        support=np.ones(power.shape),
        row_ids=(("test", 0, "event"),),
        computation=ef.ComputationSpec.create("test"),
    )
    signal = _step(1.0, np.sqrt(0.5))
    series = ef.Signal.from_arrays(
        data=np.abs(signal.analytic),
        times=signal.times,
        ch_names=("C3",),
        sfreq=SFREQ,
        row_ids=signal.row_ids,
    )

    erds_labels = {
        m.measure
        for function in ERDS_FUNCTIONS.values()
        for m in function([signal], baseline=BASE, windows=[STIM], include_global=False).meta
    }
    other_labels = {m.measure for m in ef.aperiodic(spectra, include_global=False).meta} | {
        m.measure for m in ef.peak_latency([series], windows=[STIM], include_global=False).meta
    }

    assert not erds_labels & other_labels


def test_a_single_sample_excursion_is_not_an_onset() -> None:
    # One sample beyond a baseline SD is met by chance on essentially every trial;
    # only an excursion that persists for min_duration_ms counts.
    n = 201
    rng = np.random.RandomState(3)
    envelope = np.ones((1, 1, n))
    envelope[:, :, :100] += rng.normal(0.0, 0.01, (1, 1, 100))
    envelope[:, :, 120] = np.sqrt(2.0)  # a lone spike at t = +0.2 s
    envelope[:, :, 150:] = np.sqrt(2.0)  # the sustained step at t = +0.5 s
    table = erds_onset_latency(
        [_signal(envelope)],
        baseline=BASE,
        windows=[STIM],
        include_global=False,
        min_duration_ms=100.0,
    )
    assert table.values.item() == pytest.approx(0.5, abs=0.02)
    single = erds_onset_latency(
        [_signal(envelope)],
        baseline=BASE,
        windows=[STIM],
        include_global=False,
        min_duration_ms=0.0,
    )
    assert single.values.item() == pytest.approx(0.2, abs=0.02)


def test_an_excursion_shorter_than_min_duration_gives_no_onset() -> None:
    n = 201
    envelope = np.ones((1, 1, n))
    envelope[:, :, 120:125] = np.sqrt(2.0)  # 50 ms at 100 Hz
    table = erds_onset_latency(
        [_signal(envelope)],
        baseline=BASE,
        windows=[STIM],
        include_global=False,
        min_duration_ms=100.0,
    )
    assert np.isnan(table.values).all()


def test_the_default_scale_is_decibels() -> None:
    table = erds([_step(1.0, np.sqrt(0.5))], baseline=BASE, windows=[STIM], include_global=False)
    assert table.select(measure="erds_mean").values.item() == pytest.approx(10.0 * np.log10(0.5))
    assert all(meta.normalization == "db" for meta in table.meta)


def test_onset_persistence_defaults_to_six_cycles_of_the_band_edge() -> None:
    # ALPHA starts at 8 Hz, so six cycles is 0.75 s: a step lasting 0.8 s is an onset and one
    # lasting 0.7 s is not, on the same one-second window.
    n = 201
    long_step = np.ones((1, 1, n))
    long_step[:, :, 120:] = np.sqrt(2.0)  # from +0.2 s to the end: 0.8 s
    short_step = np.ones((1, 1, n))
    short_step[:, :, 130:] = np.sqrt(2.0)  # from +0.3 s to the end: 0.7 s
    found = erds_onset_latency(
        [_signal(long_step)], baseline=BASE, windows=[STIM], include_global=False
    )
    missing = erds_onset_latency(
        [_signal(short_step)], baseline=BASE, windows=[STIM], include_global=False
    )
    assert found.values.item() == pytest.approx(0.2, abs=0.02)
    assert np.isnan(missing.values).all()
    # Milliseconds, when given, replace the cycle rule.
    forced = erds_onset_latency(
        [_signal(short_step)],
        baseline=BASE,
        windows=[STIM],
        include_global=False,
        min_duration_ms=100.0,
    )
    assert forced.values.item() == pytest.approx(0.3, abs=0.02)
    recorded = found.meta[0].computation.record()["parameters"]["parameters"]
    assert recorded["onset_min_duration_cycles"] == 6.0
    assert recorded["onset_min_duration_ms"] is None


def test_a_band_from_zero_hertz_needs_an_explicit_persistence() -> None:
    signal = _step(1.0, np.sqrt(2.0))
    from dataclasses import replace

    from_zero = replace(signal, band=Band("dc", 0.0, 4.0))
    with pytest.raises(ValueError, match="min_duration_ms"):
        erds_onset_latency([from_zero], baseline=BASE, windows=[STIM], include_global=False)
    table = erds_onset_latency(
        [from_zero], baseline=BASE, windows=[STIM], include_global=False, min_duration_ms=100.0
    )
    assert np.isfinite(table.values).all()
