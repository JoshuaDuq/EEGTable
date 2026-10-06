import numpy as np
import pytest

from eegtable.bands import Band
from eegtable.signal import BandSignal, Signal
from eegtable.spectra import Window
from eegtable.temporal import (
    area_under_curve,
    mean_amplitude,
    peak_amplitude,
    peak_latency,
    peak_to_peak,
    variance,
    zero_crossing_rate,
)

SFREQ = 100.0
# Each toy series is shorter than any finite window would fit; run to its end.
WINDOW = Window("all", 0.0, np.inf)


def _signal(data: np.ndarray) -> Signal:
    times = np.arange(data.shape[-1]) / SFREQ
    return Signal.from_arrays(
        data=data,
        times=times,
        ch_names=("C3",),
        sfreq=SFREQ,
        row_ids=tuple(("test", index, "event") for index in range(data.shape[0])),
    )


@pytest.mark.parametrize("gap", [np.nan, np.inf, -np.inf])
def test_zero_crossings_do_not_bridge_missing_samples(gap) -> None:
    values = np.array([-1.0, -1.0, gap, 0.0, 1.0, -1.0]).reshape(1, 1, -1)
    table = zero_crossing_rate([_signal(values)], windows=[WINDOW], include_global=False)
    assert table.values.item() == pytest.approx(SFREQ / values.size)


def test_variance_of_a_known_series() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(1, 1, 5)
    table = variance([_signal(values)], windows=[WINDOW], include_global=False)
    assert table.values.item() == pytest.approx(np.var(values))


def test_peak_to_peak_is_max_minus_min() -> None:
    values = np.array([-2.0, 1.0, 7.0, 0.0]).reshape(1, 1, 4)
    table = peak_to_peak([_signal(values)], windows=[WINDOW], include_global=False)
    assert table.values.item() == pytest.approx(9.0)


def test_mean_amplitude_of_a_known_series() -> None:
    values = np.array([1.0, 2.0, 3.0]).reshape(1, 1, 3)
    table = mean_amplitude([_signal(values)], windows=[WINDOW], include_global=False)
    assert table.values.item() == pytest.approx(2.0)


def test_area_under_a_constant_is_height_times_span() -> None:
    n = 201
    values = np.full((1, 1, n), 3.0)
    table = area_under_curve([_signal(values)], windows=[WINDOW], include_global=False)
    assert table.values.item() == pytest.approx(3.0 * 2.0)


def test_area_of_a_symmetric_signal_cancels() -> None:
    n = 201
    times = np.arange(n) / SFREQ
    values = np.sin(2 * np.pi * 1.0 * times).reshape(1, 1, n)
    table = area_under_curve([_signal(values)], windows=[WINDOW], include_global=False)
    assert table.values.item() == pytest.approx(0.0, abs=1e-12)


def test_area_skips_a_gap_instead_of_bridging_it() -> None:
    # A straight-line interpolation across the gap would add area; skipping adds none.
    n = 201
    values = np.full((1, 1, n), 2.0)
    values[:, :, 50:150] = np.nan
    table = area_under_curve([_signal(values)], windows=[WINDOW], include_global=False)
    contiguous = 2.0 * (49 / SFREQ) + 2.0 * ((n - 1 - 150) / SFREQ)
    assert table.values.item() == pytest.approx(contiguous)


def test_polarity_selects_which_extremum_is_found() -> None:
    values = np.array([0.0, -5.0, 0.0, 3.0, 0.0]).reshape(1, 1, 5)
    signal = _signal(values)
    positive = peak_amplitude([signal], windows=[WINDOW], polarity="positive", include_global=False)
    negative = peak_amplitude([signal], windows=[WINDOW], polarity="negative", include_global=False)
    absolute = peak_amplitude([signal], windows=[WINDOW], polarity="absolute", include_global=False)
    assert positive.values.item() == pytest.approx(3.0)
    assert negative.values.item() == pytest.approx(-5.0)
    assert absolute.values.item() == pytest.approx(-5.0)


def test_peak_latency_reports_the_time_not_the_index() -> None:
    values = np.array([0.0, 0.0, 9.0, 0.0]).reshape(1, 1, 4)
    table = peak_latency(
        [_signal(values)], windows=[WINDOW], polarity="positive", include_global=False
    )
    assert table.values.item() == pytest.approx(2 / SFREQ)


def test_polarity_is_not_inferred_from_the_window_name() -> None:
    # Polarity is explicit; verify window names (e.g. "noxious") do not influence polarity.
    values = np.array([0.0, -5.0, 0.0, 3.0, 0.0]).reshape(1, 1, 5)
    noxious = Window("noxious", 0.0, np.inf)
    table = peak_amplitude(
        [_signal(values)], windows=[noxious], polarity="positive", include_global=False
    )
    assert table.values.item() == pytest.approx(3.0)


def test_prominence_restricts_the_search_to_interior_local_peaks() -> None:
    # On a monotonic ramp the bare argmax lands on the final sample, which is not a
    # peak at all. Prominence confines the search to genuine local maxima. It does
    # not reject narrow spikes: an isolated tall sample is highly prominent.
    n = 201
    values = np.linspace(0.0, 5.0, n)
    values[50] = 4.0
    signal = _signal(values.reshape(1, 1, n))
    bare = peak_latency([signal], windows=[WINDOW], polarity="positive", include_global=False)
    prominent = peak_latency(
        [signal], windows=[WINDOW], polarity="positive", prominence=1.0, include_global=False
    )
    assert bare.values.item() == pytest.approx(2.0)
    assert prominent.values.item() == pytest.approx(0.5)


@pytest.mark.parametrize("values", [[0.0, 1.0, 2.0], [0.0, 1.0, 0.0], [np.nan] * 3])
@pytest.mark.parametrize("measure", [peak_amplitude, peak_latency])
def test_prominence_without_a_qualifying_peak_is_undefined(values, measure) -> None:
    table = measure(
        [_signal(np.array(values).reshape(1, 1, -1))],
        windows=[WINDOW],
        polarity="positive",
        prominence=2.0,
        include_global=False,
    )
    assert np.isnan(table.values.item())


@pytest.mark.parametrize("gap", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("polarity", ["positive", "negative", "absolute"])
def test_prominence_does_not_use_missing_samples_as_peak_bases(gap, polarity) -> None:
    sign = -1.0 if polarity == "negative" else 1.0
    values = np.array([0.0, 10.0 * sign, 0.0, gap, sign, gap, 0.0, 5.0 * sign, 0.0])
    signal = _signal(values.reshape(1, 1, -1))
    kwargs = dict(windows=[WINDOW], polarity=polarity, prominence=2.0, include_global=False)

    amplitude = peak_amplitude([signal], **kwargs)
    latency = peak_latency([signal], **kwargs)

    assert amplitude.values.item() == pytest.approx(10.0 * sign)
    assert latency.values.item() == pytest.approx(1.0 / SFREQ)


def test_prominence_keeps_original_latency_after_a_gap() -> None:
    values = np.array([0.0, 2.0, 0.0, np.nan, 0.0, 10.0, 0.0]).reshape(1, 1, -1)
    table = peak_latency(
        [_signal(values)],
        windows=[WINDOW],
        polarity="positive",
        prominence=1.0,
        include_global=False,
    )
    assert table.values.item() == pytest.approx(5.0 / SFREQ)


def test_measures_work_on_a_band_envelope_too() -> None:
    n = 201
    envelope = np.full((1, 1, n), 2.0, dtype=complex)
    band_signal = BandSignal.from_arrays(
        analytic=envelope,
        times=np.arange(n) / SFREQ,
        ch_names=("C3",),
        band=Band("beta", 13.0, 30.0),
        sfreq=SFREQ,
        row_ids=(("test", 0, "event"),),
    )
    table = mean_amplitude([band_signal], windows=[WINDOW], include_global=False)
    assert table.values.item() == pytest.approx(2.0)
    assert table.meta[0].band is not None


def test_non_finite_samples_are_excluded_rather_than_poisoning_the_window() -> None:
    values = np.array([1.0, np.nan, 3.0]).reshape(1, 1, 3)
    table = mean_amplitude([_signal(values)], windows=[WINDOW], include_global=False)
    assert table.values.item() == pytest.approx(2.0)
    assert table.coverage.item() < 1.0


def test_a_wholly_non_finite_window_yields_nan() -> None:
    values = np.full((1, 1, 3), np.nan)
    for fn in (variance, peak_to_peak, mean_amplitude, area_under_curve):
        assert np.isnan(fn([_signal(values)], windows=[WINDOW], include_global=False).values).all()


@pytest.mark.parametrize("bad", ["neg", "pos", "N", ""])
def test_an_unknown_polarity_raises(bad: str) -> None:
    values = np.zeros((1, 1, 3))
    with pytest.raises(ValueError, match="polarity"):
        peak_amplitude([_signal(values)], windows=[WINDOW], polarity=bad)  # type: ignore[arg-type]


def test_a_non_positive_prominence_raises() -> None:
    values = np.zeros((1, 1, 3))
    with pytest.raises(ValueError, match="prominence"):
        peak_amplitude([_signal(values)], windows=[WINDOW], prominence=0.0)
