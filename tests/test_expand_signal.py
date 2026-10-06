from collections.abc import Callable

import numpy as np
import pytest

import eegtable as ef
from eegtable._expand import expand_signal
from eegtable.bands import Band
from eegtable.signal import BandSignal
from eegtable.spectra import Window
from eegtable.table import FeatureTable

ALPHA, BETA = Band("alpha", 8.0, 13.0), Band("beta", 13.0, 30.0)
EARLY, LATE = Window("early", -1.0, 0.0), Window("late", 0.0, 1.0)
SFREQ = 100.0


def _signal(band: Band, value: float) -> BandSignal:
    times = np.arange(400) / SFREQ - 1.0
    analytic = np.full((3, 2, 400), value + 0j)
    return BandSignal.from_arrays(
        analytic=analytic,
        times=times,
        ch_names=("C3", "C4"),
        band=band,
        sfreq=SFREQ,
        row_ids=tuple(("test", index, "event") for index in range(analytic.shape[0])),
    )


def _mean_kernel(
    signal: BandSignal, trace: np.ndarray, times: np.ndarray, mask: np.ndarray
) -> dict[str, np.ndarray]:
    del signal, times, mask
    return {"mean": trace.mean(axis=2), "count": np.full(trace.shape[:2], trace.shape[2], float)}


def _run(**kw: object) -> object:
    defaults = dict(
        signals=[_signal(ALPHA, 2.0), _signal(BETA, 4.0)],
        trace_of=lambda s: s.envelope,
        kernel=_mean_kernel,
        units={"mean": "a.u.", "count": "samples"},
        windows=[EARLY, LATE],
        groups=None,
        include_global=False,
        mode="raw",
        parameters={},
    )
    return expand_signal(**{**defaults, **kw})  # type: ignore[arg-type]


def test_columns_are_the_product_of_bands_measures_spaces_and_windows() -> None:
    table = _run()
    # 2 bands x 2 measures x 2 channels x 2 windows
    assert table.values.shape == (3, 16)  # type: ignore[attr-defined]
    assert {m.measure for m in table.meta} == {"mean", "count"}  # type: ignore[attr-defined]
    assert {m.band.name for m in table.meta if m.band} == {"alpha", "beta"}  # type: ignore[attr-defined]


def test_each_measure_carries_its_own_unit() -> None:
    table = _run()
    assert {m.unit for m in table.select(measure="count").meta} == {"samples"}  # type: ignore[attr-defined]
    assert {m.unit for m in table.select(measure="mean").meta} == {"a.u."}  # type: ignore[attr-defined]


def test_values_come_from_the_kernel_per_band() -> None:
    table = _run()
    np.testing.assert_allclose(table.select(measure="mean", band=ALPHA).values, 2.0)  # type: ignore[attr-defined]
    np.testing.assert_allclose(table.select(measure="mean", band=BETA).values, 4.0)  # type: ignore[attr-defined]


def test_windows_slice_the_time_axis() -> None:
    table = _run()
    # 1 s windows at 100 Hz, inclusive of both bounds
    np.testing.assert_allclose(table.select(measure="count").values, 101.0)  # type: ignore[attr-defined]


def test_groups_aggregate_the_channel_axis() -> None:
    table = _run(groups={"central": ["C3", "C4"]}, include_global=True)
    assert {m.space for m in table.meta} == {"central", "global"}  # type: ignore[attr-defined]


def test_signal_roi_identity_records_members_independent_of_order() -> None:
    assert _run(groups={"central": ["C3"]}).names != _run(groups={"central": ["C4"]}).names
    assert (
        _run(groups={"central": ["C3", "C4"]}).names == _run(groups={"central": ["C4", "C3"]}).names
    )


def test_a_window_outside_the_time_axis_raises() -> None:
    with pytest.raises(ValueError, match="no samples"):
        _run(windows=[Window("late", 30.0, 40.0)])


def test_a_window_reaching_past_the_time_axis_raises() -> None:
    # Truncated, it would measure 0-2.99 s under a name and bounds claiming 0-5 s, so a typo
    # or a recording with shorter epochs would pass unnoticed.
    with pytest.raises(ValueError, match="reaches outside"):
        _run(windows=[Window("late", 0.0, 5.0)])


def test_a_window_may_end_where_the_last_sample_ends() -> None:
    # The last sample, at 2.99 s, stands for the interval up to 3 s: a window written to the
    # end of the data, 0 to 3 s, is whole, as 0 to 30 s is for a 30 s sleep epoch.
    table = _run(windows=[Window("late", 0.0, 3.0)])
    np.testing.assert_allclose(table.select(measure="count").values, 300.0)  # type: ignore[attr-defined]


def test_an_infinite_window_bound_runs_to_the_edge_of_the_axis() -> None:
    table = _run(windows=[Window("late", 0.0, np.inf)])
    np.testing.assert_allclose(table.select(measure="count").values, 300.0)  # type: ignore[attr-defined]


def test_signals_disagreeing_on_the_time_axis_raise() -> None:
    other = _signal(BETA, 4.0)
    shifted = BandSignal.from_arrays(
        analytic=other.analytic,
        times=other.times + 5.0,
        ch_names=other.ch_names,
        band=other.band,
        sfreq=other.sfreq,
        row_ids=other.row_ids,
    )
    with pytest.raises(ValueError, match="same time axis"):
        _run(signals=[_signal(ALPHA, 2.0), shifted])


def test_signals_disagreeing_on_channels_raise() -> None:
    other = _signal(BETA, 4.0)
    renamed = BandSignal.from_arrays(
        analytic=other.analytic,
        times=other.times,
        ch_names=("Cz", "Pz"),
        band=other.band,
        sfreq=other.sfreq,
        row_ids=other.row_ids,
    )
    with pytest.raises(ValueError, match="same channels"):
        _run(signals=[_signal(ALPHA, 2.0), renamed])


def test_an_empty_signal_sequence_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _run(signals=[])


def _reordered(signal: BandSignal) -> BandSignal:
    """The same data under reversed epoch identities: same shape, different trials."""
    return BandSignal.from_arrays(
        analytic=signal.analytic,
        times=signal.times,
        ch_names=signal.ch_names,
        band=signal.band,
        sfreq=signal.sfreq,
        row_ids=tuple(reversed(signal.row_ids)),
    )


def test_signals_disagreeing_on_row_identity_raise() -> None:
    with pytest.raises(ValueError, match="row identities"):
        _run(signals=[_signal(ALPHA, 2.0), _reordered(_signal(BETA, 4.0))])


def test_signals_disagreeing_on_sampling_frequency_raise() -> None:
    other = _signal(BETA, 4.0)
    resampled = BandSignal.from_arrays(
        analytic=other.analytic,
        times=other.times[0] + np.arange(other.times.size) / (other.sfreq * 2.0),
        ch_names=other.ch_names,
        band=other.band,
        sfreq=other.sfreq * 2.0,
        row_ids=other.row_ids,
    )
    with pytest.raises(ValueError, match="sampling frequency"):
        _run(signals=[_signal(ALPHA, 2.0), resampled])


def test_misalignment_is_caught_before_the_kernel_runs() -> None:
    calls: list[str] = []

    def spy(signal: BandSignal, trace: np.ndarray, times: np.ndarray) -> dict[str, np.ndarray]:
        calls.append("called")
        return _mean_kernel(signal, trace, times)

    with pytest.raises(ValueError, match="row identities"):
        _run(signals=[_signal(ALPHA, 2.0), _reordered(_signal(BETA, 4.0))], kernel=spy)
    assert calls == []


Measure = Callable[[list[BandSignal]], FeatureTable]


@pytest.mark.parametrize(
    "compute",
    [
        pytest.param(lambda signals: ef.itpc(signals, windows=[LATE]), id="itpc"),
        pytest.param(lambda signals: ef.ppc(signals, windows=[LATE]), id="ppc"),
        pytest.param(lambda signals: ef.burst_count(signals, windows=[LATE]), id="burst_count"),
        pytest.param(
            lambda signals: ef.erds_mean(signals, baseline=EARLY, windows=[LATE]), id="erds_mean"
        ),
        pytest.param(
            lambda signals: ef.envelope_correlation(signals, windows=[LATE]),
            id="envelope_correlation",
        ),
    ],
)
def test_every_multi_signal_measure_rejects_misaligned_trials(compute: Measure) -> None:
    signals = [_signal(ALPHA, 2.0), _reordered(_signal(BETA, 4.0))]
    with pytest.raises(ValueError, match="row identities"):
        compute(signals)


def test_a_single_signal_outside_a_list_is_named() -> None:
    # Measures take a sequence of signals, one per band; a bare one used to fail on indexing.
    with pytest.raises(TypeError, match=r"\[signal\]"):
        ef.variance(_signal(ALPHA, 1.0), windows=[EARLY])  # type: ignore[arg-type]
