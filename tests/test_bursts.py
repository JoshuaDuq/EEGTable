import numpy as np
import pytest

import eegtable as ef
from eegtable.bands import Band
from eegtable.bursts import (
    burst_amplitude,
    burst_count,
    burst_duration,
    burst_rate,
    fraction_above_threshold,
)
from eegtable.signal import BandSignal
from eegtable.spectra import Window

BURST_FUNCTIONS = {
    "count": burst_count,
    "rate": burst_rate,
    "duration_mean": burst_duration,
    "amp_mean": burst_amplitude,
    "fraction_above": fraction_above_threshold,
}


def burst_features(signals, **kwargs):
    """Test-only: every burst measure in one table. See the note in test_erds.py."""
    return ef.concat([fn(signals, **kwargs) for fn in BURST_FUNCTIONS.values()])


BETA = Band("beta", 13.0, 30.0)
SFREQ = 100.0
# The toy envelopes differ in length; the window runs to the end of each.
WINDOW = Window("stim", 0.0, np.inf)


def _signal(envelope: np.ndarray) -> BandSignal:
    times = np.arange(envelope.shape[-1]) / SFREQ
    return BandSignal.from_arrays(
        analytic=envelope.astype(complex),
        times=times,
        ch_names=("C3",),
        band=BETA,
        sfreq=SFREQ,
        row_ids=tuple(("test", index, "event") for index in range(envelope.shape[0])),
    )


def test_square_wave_gives_exact_burst_measures() -> None:
    # 100 samples at 100 Hz = 1.0 second
    envelope = np.zeros((1, 1, 100))
    envelope[:, :, 10:30] = 1.0  # 20 samples = 0.20 s, amp 1.0
    envelope[:, :, 50:80] = 2.0  # 30 samples = 0.30 s, amp 2.0

    sig = _signal(envelope)
    table = burst_features(
        [sig],
        windows=[WINDOW],
        threshold=np.array([[0.5]]),
        min_duration_ms=100.0,
        include_global=False,
    )

    assert table.select(measure="count").values.item() == pytest.approx(2.0)
    assert table.select(measure="rate").values.item() == pytest.approx(2.0)
    assert table.select(measure="duration_mean").values.item() == pytest.approx(0.25)
    assert table.select(measure="amp_mean").values.item() == pytest.approx(1.5)
    assert table.select(measure="fraction_above").values.item() == pytest.approx(0.5)


def test_duration_filter_discards_short_bursts() -> None:
    envelope = np.zeros((1, 1, 100))
    envelope[:, :, 10:30] = 1.0  # 20 samples = 200 ms
    envelope[:, :, 50:80] = 2.0  # 30 samples = 300 ms

    sig = _signal(envelope)
    # min_duration 250 ms discards the 200 ms burst but keeps the 300 ms burst
    table = burst_features(
        [sig],
        windows=[WINDOW],
        threshold=np.array([[0.5]]),
        min_duration_ms=250.0,
        include_global=False,
    )

    assert table.select(measure="count").values.item() == pytest.approx(1.0)
    assert table.select(measure="rate").values.item() == pytest.approx(1.0)
    assert table.select(measure="duration_mean").values.item() == pytest.approx(0.3)
    assert table.select(measure="amp_mean").values.item() == pytest.approx(2.0)
    assert table.select(measure="fraction_above").values.item() == pytest.approx(0.5)


@pytest.mark.parametrize("minimum, expected", [(100.0, 1.0), (104.0, 0.0), (106.0, 0.0)])
def test_minimum_burst_duration_is_not_rounded_down(minimum, expected) -> None:
    envelope = np.zeros((1, 1, 100))
    envelope[:, :, 10:20] = 1.0
    table = burst_count(
        [_signal(envelope)],
        windows=[WINDOW],
        threshold=np.array([[0.5]]),
        min_duration_ms=minimum,
        include_global=False,
    )
    assert table.values.item() == expected


def test_no_burst_survives_duration_filter() -> None:
    envelope = np.zeros((1, 1, 100))
    envelope[:, :, 10:30] = 1.0
    envelope[:, :, 50:80] = 2.0

    sig = _signal(envelope)
    table = burst_features(
        [sig],
        windows=[WINDOW],
        threshold=np.array([[0.5]]),
        min_duration_ms=400.0,
        include_global=False,
    )

    assert table.select(measure="count").values.item() == pytest.approx(0.0)
    assert table.select(measure="rate").values.item() == pytest.approx(0.0)
    assert np.isnan(table.select(measure="duration_mean").values).all()
    assert np.isnan(table.select(measure="amp_mean").values).all()
    assert table.select(measure="fraction_above").values.item() == pytest.approx(0.5)


def test_no_sample_above_threshold() -> None:
    envelope = np.full((1, 1, 100), 0.1)
    sig = _signal(envelope)
    table = burst_features(
        [sig],
        windows=[WINDOW],
        threshold=np.array([[0.5]]),
        min_duration_ms=50.0,
        include_global=False,
    )

    assert table.select(measure="count").values.item() == pytest.approx(0.0)
    assert table.select(measure="rate").values.item() == pytest.approx(0.0)
    assert np.isnan(table.select(measure="duration_mean").values).all()
    assert np.isnan(table.select(measure="amp_mean").values).all()
    assert table.select(measure="fraction_above").values.item() == pytest.approx(0.0)


def test_pulses_touching_window_edges_are_closed_at_those_ends() -> None:
    envelope = np.zeros((1, 1, 100))
    envelope[:, :, :20] = 1.0  # touches start: [0:20] -> 20 samples = 0.20 s
    envelope[:, :, 80:] = 1.0  # touches end: [80:100] -> 20 samples = 0.20 s

    sig = _signal(envelope)
    table = burst_features(
        [sig],
        windows=[WINDOW],
        threshold=np.array([[0.5]]),
        min_duration_ms=100.0,
        include_global=False,
    )

    assert table.select(measure="count").values.item() == pytest.approx(2.0)
    assert table.select(measure="duration_mean").values.item() == pytest.approx(0.20)
    assert table.select(measure="fraction_above").values.item() == pytest.approx(0.40)


def test_count_is_monotonically_non_increasing_in_min_duration() -> None:
    rng = np.random.RandomState(42)
    envelope = np.abs(rng.randn(2, 2, 500))
    times = np.arange(500) / SFREQ
    sig = BandSignal.from_arrays(
        analytic=envelope.astype(complex),
        times=times,
        ch_names=("C3", "Cz"),
        band=BETA,
        sfreq=SFREQ,
        row_ids=tuple(("test", index, "event") for index in range(envelope.shape[0])),
    )

    durations = [10.0, 50.0, 100.0, 200.0]
    counts = [
        burst_features(
            [sig],
            windows=[Window("stim", 0.0, 4.0)],
            threshold=0.75,
            min_duration_ms=d,
            include_global=False,
        )
        .select(measure="count")
        .values
        for d in durations
    ]

    for i in range(len(durations) - 1):
        assert (counts[i] >= counts[i + 1]).all()


def test_fraction_above_is_unchanged_by_min_duration_filter() -> None:
    rng = np.random.RandomState(42)
    envelope = np.abs(rng.randn(1, 1, 300))
    sig = _signal(envelope)
    win = Window("stim", 0.0, 2.5)

    frac_short = (
        burst_features(
            [sig], windows=[win], threshold=0.8, min_duration_ms=20.0, include_global=False
        )
        .select(measure="fraction_above")
        .values
    )

    frac_long = (
        burst_features(
            [sig], windows=[win], threshold=0.8, min_duration_ms=200.0, include_global=False
        )
        .select(measure="fraction_above")
        .values
    )

    np.testing.assert_allclose(frac_short, frac_long)


def test_percentile_threshold_is_computed_per_trial_without_cross_trial_leakage() -> None:
    # Epoch 1 has scale 1.0, Epoch 2 has scale 10.0
    envelope = np.zeros((2, 1, 100))
    envelope[0, 0, :50] = 1.0
    envelope[0, 0, 50:] = 2.0  # median is 1.5, 50% above median
    envelope[1, 0, :50] = 10.0
    envelope[1, 0, 50:] = 20.0  # median is 15.0, 50% above median

    times = np.arange(100) / SFREQ
    sig = BandSignal.from_arrays(
        analytic=envelope.astype(complex),
        times=times,
        ch_names=("C3",),
        band=BETA,
        sfreq=SFREQ,
        row_ids=tuple(("test", index, "event") for index in range(envelope.shape[0])),
    )
    table = burst_features(
        [sig],
        windows=[WINDOW],
        threshold=0.5,
        min_duration_ms=100.0,
        include_global=False,
    )
    # Both epochs should see their own 50th percentile and detect the same burst pattern
    np.testing.assert_allclose(
        table.select(measure="count").values[0],
        table.select(measure="count").values[1],
    )


def test_non_finite_trace_returns_nan_measures() -> None:
    envelope = np.full((1, 1, 100), np.nan)
    sig = _signal(envelope)
    table = burst_features(
        [sig],
        windows=[WINDOW],
        threshold=0.5,
        include_global=False,
    )
    for m in ("count", "rate", "duration_mean", "amp_mean", "fraction_above"):
        assert np.isnan(table.select(measure=m).values).all()


def test_invalid_threshold_raises() -> None:
    sig = _signal(np.ones((1, 1, 50)))
    with pytest.raises(ValueError, match="threshold"):
        burst_features([sig], windows=[WINDOW], threshold=0.0)
    with pytest.raises(ValueError, match="threshold"):
        burst_features([sig], windows=[WINDOW], threshold=1.0)
    with pytest.raises(ValueError, match="threshold"):
        burst_features([sig], windows=[WINDOW], threshold=-0.5)


def test_invalid_min_duration_raises() -> None:
    sig = _signal(np.ones((1, 1, 50)))
    with pytest.raises(ValueError, match="min_duration_ms"):
        burst_features([sig], windows=[WINDOW], min_duration_ms=-1.0)


def test_threshold_array_shape_mismatch_raises() -> None:
    sig = _signal(np.ones((2, 1, 50)))
    with pytest.raises(ValueError, match="broadcast"):
        burst_features([sig], windows=[WINDOW], threshold=np.ones((5, 5)))


# --- threshold calibration window ----------------------------------------------------

BASELINE = Window("base", 0.0, 0.95)
ACTIVE = Window("active", 1.0, 2.0)


def _stimulus_response() -> BandSignal:
    # A variable baseline, then a sustained response carrying three bursts. The
    # windows deliberately do not touch: bounds are inclusive at both ends.
    n = 201
    envelope = np.empty((1, 1, n))
    envelope[:, :, :100] = np.linspace(1.0, 6.0, 100)
    envelope[:, :, 100:] = 3.0
    for start in (110, 140, 170):
        envelope[:, :, start : start + 15] = 9.0
    return _signal(envelope)


def test_the_threshold_is_calibrated_on_the_baseline_when_one_is_given() -> None:
    # Calibrating on the analysis window lets the stimulus response raise the very
    # threshold used to detect it: here the active window's 75th percentile is 9.0,
    # exactly the burst amplitude, so nothing exceeds it and every burst is lost.
    signal = _stimulus_response()
    on_baseline = burst_features(
        [signal],
        windows=[ACTIVE],
        baseline=BASELINE,
        threshold=0.75,
        min_duration_ms=100.0,
        include_global=False,
    )
    on_active = burst_features(
        [signal],
        windows=[ACTIVE],
        threshold=0.75,
        min_duration_ms=100.0,
        include_global=False,
    )
    assert on_baseline.select(measure="count").values.item() == 3.0
    assert on_active.select(measure="count").values.item() == 0.0


def test_coverage_is_limited_by_the_samples_the_threshold_was_calibrated_on() -> None:
    # A quantile taken from a quarter of the baseline cannot stand behind full coverage.
    envelope = np.abs(_stimulus_response().analytic)
    envelope[:, :, :72] = np.nan  # 72 of the 96 baseline samples
    kwargs = {"windows": [ACTIVE], "baseline": BASELINE, "include_global": False}
    calibrated = burst_features([_signal(envelope)], threshold=0.75, **kwargs)
    absolute = burst_features([_signal(envelope)], threshold=np.array([[2.0]]), **kwargs)
    np.testing.assert_allclose(calibrated.coverage, 0.25)
    np.testing.assert_allclose(absolute.coverage, 1.0)


def test_the_calibration_window_is_recorded_in_the_unit() -> None:
    signal = _stimulus_response()
    with_baseline = burst_features(
        [signal], windows=[ACTIVE], baseline=BASELINE, threshold=0.75, include_global=False
    )
    without = burst_features([signal], windows=[ACTIVE], threshold=0.75, include_global=False)
    absolute = burst_features(
        [signal], windows=[ACTIVE], threshold=np.array([[2.0]]), include_global=False
    )
    assert with_baseline.meta[0].unit.endswith("(quantile 0.75 of baseline)")
    assert without.meta[0].unit.endswith("(quantile 0.75 of analysis windows)")
    assert absolute.meta[0].unit.endswith("(absolute)")


def test_an_absolute_threshold_ignores_the_baseline_argument() -> None:
    signal = _stimulus_response()
    with_baseline = burst_features(
        [signal],
        windows=[ACTIVE],
        baseline=BASELINE,
        threshold=np.array([[5.0]]),
        min_duration_ms=100.0,
        include_global=False,
    )
    without = burst_features(
        [signal],
        windows=[ACTIVE],
        threshold=np.array([[5.0]]),
        min_duration_ms=100.0,
        include_global=False,
    )
    np.testing.assert_array_equal(with_baseline.values, without.values)


def test_a_baseline_window_outside_the_time_axis_raises() -> None:
    with pytest.raises(ValueError, match="no samples"):
        burst_features(
            [_stimulus_response()],
            windows=[ACTIVE],
            baseline=Window("nope", 50.0, 60.0),
            threshold=0.75,
            include_global=False,
        )


# --- the split public surface ---------------------------------------------------------


@pytest.mark.parametrize(("measure", "function"), list(BURST_FUNCTIONS.items()))
def test_each_burst_function_returns_exactly_its_own_measure(measure, function) -> None:
    table = function(
        [_stimulus_response()],
        windows=[ACTIVE],
        baseline=BASELINE,
        threshold=0.75,
        min_duration_ms=100.0,
        include_global=False,
    )
    assert table.values.shape == (1, 1)
    assert [m.measure for m in table.meta] == [measure]


def test_the_burst_functions_agree_with_the_combined_view() -> None:
    signals = [_stimulus_response()]
    kw = dict(
        windows=[ACTIVE],
        baseline=BASELINE,
        threshold=0.75,
        min_duration_ms=100.0,
        include_global=False,
    )
    combined = burst_features(signals, **kw)
    for measure, function in BURST_FUNCTIONS.items():
        alone = function(signals, **kw)
        np.testing.assert_allclose(
            alone.values, combined.select(measure=measure).values, equal_nan=True
        )
