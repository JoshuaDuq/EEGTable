import logging
import re
import warnings

import mne
import numpy as np
import pytest

from eegtable.bands import BANDS_STANDARD, Band
from eegtable.signal import BandSignal, Signal, TimeSeries, _required_filter_length
from eegtable.spectra import Window
from eegtable.temporal import variance

BETA = Band("beta", 13.0, 30.0)


def _signal(analytic: np.ndarray, sfreq: float = 100.0) -> BandSignal:
    times = np.arange(analytic.shape[-1]) / sfreq
    return BandSignal.from_arrays(
        analytic=analytic,
        times=times,
        ch_names=("C3", "C4"),
        band=BETA,
        sfreq=sfreq,
        row_ids=tuple(("test", index, "event") for index in range(analytic.shape[0])),
    )


def test_envelope_phase_and_power_are_views_of_the_analytic_signal() -> None:
    analytic = np.array([[[1 + 1j, 0 + 2j], [3 + 0j, -1 - 1j]]])
    signal = _signal(analytic)
    np.testing.assert_allclose(signal.envelope, np.abs(analytic))
    np.testing.assert_allclose(signal.phase, np.angle(analytic))
    np.testing.assert_allclose(signal.power, np.abs(analytic) ** 2)


def test_power_is_the_squared_envelope_exactly() -> None:
    signal = _signal(np.array([[[3 + 4j, 5 + 12j], [1 + 0j, 0 + 1j]]]))
    np.testing.assert_allclose(signal.power, signal.envelope**2)
    np.testing.assert_allclose(signal.envelope[0, 0], [5.0, 13.0])


def test_coverage_defaults_to_where_the_input_was_finite() -> None:
    analytic = np.ones((1, 2, 3), dtype=complex)
    analytic[0, 1, 2] = np.nan
    signal = _signal(analytic)
    assert signal.coverage[0, 1, 2] == 0.0
    assert signal.coverage.sum() == 5.0


def test_shape_mismatches_raise() -> None:
    good = dict(
        analytic=np.ones((2, 2, 4), dtype=complex),
        times=np.arange(4) / 100.0,
        ch_names=("C3", "C4"),
        band=BETA,
        sfreq=100.0,
        row_ids=(("test", 0, "event"), ("test", 1, "event")),
    )
    with pytest.raises(ValueError, match="ch_names"):
        BandSignal.from_arrays(**{**good, "ch_names": ("C3",)})
    with pytest.raises(ValueError, match="times"):
        BandSignal.from_arrays(**{**good, "times": np.arange(3) / 100.0})
    with pytest.raises(ValueError, match="ascending"):
        BandSignal.from_arrays(**{**good, "times": np.array([3.0, 2.0, 1.0, 0.0])})
    with pytest.raises(ValueError, match="3-D"):
        BandSignal.from_arrays(**{**good, "analytic": np.ones((2, 4), dtype=complex)})


def test_a_real_valued_input_raises_rather_than_silently_losing_phase() -> None:
    with pytest.raises(TypeError, match="complex"):
        BandSignal.from_arrays(
            analytic=np.ones((1, 2, 4)),
            times=np.arange(4) / 100.0,
            ch_names=("C3", "C4"),
            band=BETA,
            sfreq=100.0,
            row_ids=(("test", 0, "event"),),
        )


def test_a_broadband_signal_cannot_discard_complex_data() -> None:
    with pytest.raises(TypeError, match="real"):
        Signal.from_arrays(
            data=np.ones((1, 2, 4), dtype=complex) * (1.0 + 2.0j),
            times=np.arange(4) / 100.0,
            ch_names=("C3", "C4"),
            sfreq=100.0,
            row_ids=(("test", 0, "event"),),
        )


def test_non_positive_sfreq_raises() -> None:
    with pytest.raises(ValueError, match="sfreq"):
        BandSignal.from_arrays(
            analytic=np.ones((1, 2, 4), dtype=complex),
            times=np.arange(4),
            ch_names=("C3", "C4"),
            band=BETA,
            sfreq=0.0,
            row_ids=(("test", 0, "event"),),
        )


ALPHA = Band("alpha", 8.0, 13.0)


def _epochs(freq_hz: float, n_epochs: int = 3, sfreq: float = 200.0, dur: float = 4.0):
    n = int(sfreq * dur)
    t = np.arange(n) / sfreq
    wave = np.cos(2 * np.pi * freq_hz * t)
    data = np.tile(wave, (n_epochs, 2, 1))
    info = mne.create_info(["C3", "C4"], sfreq, "eeg")
    return mne.EpochsArray(data, info, tmin=-1.0, verbose="ERROR")


@pytest.mark.parametrize("container", [Signal, BandSignal])
def test_epoch_constructors_reject_complex_samples(container) -> None:
    sfreq = 200.0
    times = np.arange(800) / sfreq
    analytic = np.exp(2j * np.pi * 10.0 * times)[None, None, :]
    epochs = mne.EpochsArray(analytic, mne.create_info(["C3"], sfreq, "eeg"), verbose=False)
    arguments = {"band": ALPHA} if container is BandSignal else {}
    with pytest.raises(TypeError, match="real"):
        container.from_epochs(epochs, recording="test", **arguments)


@pytest.mark.parametrize("parameter", ["pad_sec", "pad_cycles"])
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_padding_settings_cannot_be_silently_ignored(parameter, value) -> None:
    with pytest.raises(ValueError, match=parameter):
        BandSignal.from_epochs(_epochs(10.0), ALPHA, recording="test", **{parameter: value})


def test_a_sine_inside_the_band_yields_a_flat_envelope_at_its_amplitude() -> None:
    signal = BandSignal.from_epochs(_epochs(10.0), ALPHA, recording="test")
    interior = signal.envelope[:, :, 100:-100]
    np.testing.assert_allclose(interior, 1.0, rtol=0.05)


def test_a_sine_outside_the_band_is_attenuated() -> None:
    inside = (
        BandSignal.from_epochs(_epochs(10.0), ALPHA, recording="test")
        .envelope[:, :, 100:-100]
        .mean()
    )
    outside = (
        BandSignal.from_epochs(_epochs(40.0), ALPHA, recording="test")
        .envelope[:, :, 100:-100]
        .mean()
    )
    assert outside < inside / 20.0


def test_padding_protects_the_edges() -> None:
    padded = BandSignal.from_epochs(_epochs(10.0), ALPHA, recording="test")
    unpadded = BandSignal.from_epochs(
        _epochs(10.0), ALPHA, recording="test", pad_sec=0.0, pad_cycles=0.0
    )
    edge = slice(0, 20)
    assert abs(padded.envelope[0, 0, edge].mean() - 1.0) < abs(
        unpadded.envelope[0, 0, edge].mean() - 1.0
    )


def test_shape_times_and_band_are_carried_through() -> None:
    epochs = _epochs(10.0)
    signal = BandSignal.from_epochs(epochs, ALPHA, recording="test")
    assert signal.analytic.shape == (3, 2, len(epochs.times))
    np.testing.assert_allclose(signal.times, epochs.times)
    assert signal.ch_names == ("C3", "C4")
    assert signal.band is ALPHA
    assert signal.sfreq == 200.0


def test_a_band_above_nyquist_raises() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        BandSignal.from_epochs(_epochs(10.0), Band("vhf", 90.0, 150.0), recording="test")


def test_coverage_reflects_the_filtered_output_not_the_raw_input() -> None:
    # One bad input sample propagates through the FIR convolution and the Hilbert
    # transform and destroys the whole epoch. Coverage must say so, or a caller
    # filtering on coverage keeps a column that is entirely NaN.
    epochs = _epochs(10.0)
    data = np.asarray(epochs.get_data())
    data[0, 0, data.shape[2] // 2] = np.nan
    dirty = mne.EpochsArray(data, epochs.info, tmin=epochs.tmin, verbose="ERROR")
    signal = BandSignal.from_epochs(dirty, ALPHA, recording="test")
    assert not np.isfinite(signal.envelope[0, 0]).any()
    assert signal.coverage[0, 0].max() == 0.0
    assert signal.coverage[1, 0].min() == 1.0


def test_parallel_filtering_leaves_feature_identity_unchanged() -> None:
    # n_jobs only spreads MNE's filtering over workers. Were it part of the identity, a
    # recording run with --n-jobs 2 would stack beside one run serially as disjoint
    # columns holding identical values.
    epochs = _epochs(10.0)
    serial, parallel = (
        variance(
            [BandSignal.from_epochs(epochs, ALPHA, recording="test", n_jobs=n_jobs)],
            windows=[Window("task", 0.0, 2.0)],
        )
        for n_jobs in (1, 2)
    )
    np.testing.assert_array_equal(serial.values, parallel.values)
    assert serial.names == parallel.names


# --- the raw Signal container ---------------------------------------------------------


def test_signal_wraps_epochs_without_transforming_them() -> None:
    epochs = _epochs(10.0)
    signal = Signal.from_epochs(epochs, recording="sub-01_task-test")
    np.testing.assert_allclose(signal.data, np.asarray(epochs.get_data()))
    np.testing.assert_allclose(signal.times, epochs.times)
    assert signal.ch_names == ("C3", "C4")
    assert signal.sfreq == 200.0
    assert signal.band is None
    assert signal.source == "signal"
    assert signal.row_ids == (
        ("sub-01_task-test", 0, "1"),
        ("sub-01_task-test", 1, "1"),
        ("sub-01_task-test", 2, "1"),
    )


def test_from_epochs_selects_good_eeg_channels_by_default() -> None:
    mne = pytest.importorskip("mne")
    info = mne.create_info(["C3", "C4", "EOG"], 100.0, ["eeg", "eeg", "eog"])
    info["bads"] = ["C4"]
    epochs = mne.EpochsArray(np.ones((2, 3, 20)), info, verbose="ERROR")

    signal = Signal.from_epochs(epochs, recording="test")

    assert signal.ch_names == ("C3",)
    assert signal.data.shape == (2, 1, 20)


def test_both_containers_satisfy_the_time_series_protocol() -> None:
    epochs = _epochs(10.0)
    assert isinstance(Signal.from_epochs(epochs, recording="test"), TimeSeries)
    assert isinstance(BandSignal.from_epochs(epochs, ALPHA, recording="test"), TimeSeries)


def test_a_band_signal_reports_its_band_and_a_raw_signal_does_not() -> None:
    epochs = _epochs(10.0)
    assert BandSignal.from_epochs(epochs, ALPHA, recording="test").band is ALPHA
    assert Signal.from_epochs(epochs, recording="test").band is None


def test_signal_coverage_marks_non_finite_samples() -> None:
    data = np.ones((1, 2, 5))
    data[0, 1, 3] = np.nan
    signal = Signal.from_arrays(
        data=data,
        times=np.arange(5) / 100.0,
        ch_names=("C3", "C4"),
        sfreq=100.0,
        row_ids=(("test", 0, "event"),),
    )
    assert signal.coverage[0, 1, 3] == 0.0
    assert signal.coverage.sum() == 9.0


def test_signal_shape_mismatches_raise() -> None:
    good = dict(
        data=np.ones((2, 2, 4)),
        times=np.arange(4) / 100.0,
        ch_names=("C3", "C4"),
        sfreq=100.0,
        row_ids=(("test", 0, "event"), ("test", 1, "event")),
    )
    with pytest.raises(ValueError, match="ch_names"):
        Signal.from_arrays(**{**good, "ch_names": ("C3",)})
    with pytest.raises(ValueError, match="times"):
        Signal.from_arrays(**{**good, "times": np.arange(3) / 100.0})
    with pytest.raises(ValueError, match="3-D"):
        Signal.from_arrays(**{**good, "data": np.ones((2, 4))})
    with pytest.raises(ValueError, match="sfreq"):
        Signal.from_arrays(**{**good, "sfreq": 0.0})


@pytest.mark.parametrize("container", [Signal, BandSignal])
@pytest.mark.parametrize("times", [np.arange(4) / 200.0, np.array([0.0, 0.01, 0.025, 0.03])])
def test_time_series_reject_timestamps_inconsistent_with_sampling_rate(
    container: type[Signal] | type[BandSignal], times: np.ndarray
) -> None:
    arguments = {
        "analytic" if container is BandSignal else "data": np.ones(
            (1, 2, 4), dtype=complex if container is BandSignal else float
        ),
        "times": times,
        "ch_names": ("C3", "C4"),
        "sfreq": 100.0,
        "row_ids": (("test", 0, "event"),),
    }
    if container is BandSignal:
        arguments["band"] = BETA

    with pytest.raises(ValueError, match="uniformly spaced.*sfreq"):
        container.from_arrays(**arguments)


@pytest.mark.parametrize("container", [Signal, BandSignal])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -0.1, 1.1])
def test_time_series_coverage_must_be_a_finite_fraction(
    container: type[Signal] | type[BandSignal], invalid: float
) -> None:
    values = np.ones((1, 2, 4), dtype=complex if container is BandSignal else float)
    arguments = {
        "analytic" if container is BandSignal else "data": values,
        "times": np.arange(4) / 100.0,
        "ch_names": ("C3", "C4"),
        "sfreq": 100.0,
        "coverage": np.array([[[invalid, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]]),
        "row_ids": (("test", 0, "event"),),
    }
    if container is BandSignal:
        arguments["band"] = BETA

    with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
        container.from_arrays(**arguments)


@pytest.mark.parametrize("container", [Signal, BandSignal])
def test_time_series_require_nonempty_axes_and_unique_channels(
    container: type[Signal] | type[BandSignal],
) -> None:
    values = np.ones((1, 2, 0), dtype=complex if container is BandSignal else float)
    arguments = {
        "analytic" if container is BandSignal else "data": values,
        "times": np.empty(0),
        "ch_names": ("C3", "C3"),
        "sfreq": 100.0,
        "row_ids": (("test", 0, "event"),),
    }
    if container is BandSignal:
        arguments["band"] = BETA

    with pytest.raises(ValueError, match="non-empty"):
        container.from_arrays(**arguments)


def test_channel_names_must_be_nonempty_strings() -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        Signal.from_arrays(
            data=np.ones((1, 2, 4)),
            times=np.arange(4) / 100.0,
            ch_names=("C3", 7),
            sfreq=100.0,
            row_ids=(("test", 0, "event"),),
        )


def test_a_band_reaching_exactly_nyquist_raises() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        BandSignal.from_epochs(_epochs(10.0), Band("hf", 80.0, 100.0), recording="test")


# The bands and epoch lengths a caller actually combines. Every cell used to be
# untested; delta on short epochs raised MNE's own "filter_length, if a string"
# error, and alpha on very short epochs silently filtered with a truncated kernel.
_DURATIONS = (0.5, 1.0, 2.0, 4.0, 10.0)


@pytest.mark.parametrize("band", BANDS_STANDARD, ids=lambda b: b.name)
@pytest.mark.parametrize("dur", _DURATIONS)
def test_every_band_and_epoch_length_either_filters_cleanly_or_refuses(
    band: Band, dur: float
) -> None:
    epochs = _epochs(10.0, dur=dur)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            signal = BandSignal.from_epochs(epochs, band, recording="test")
    except ValueError as error:
        # The only acceptable failure is our own, and it has to say what to change.
        assert band.name in str(error)
        assert "at least" in str(error)
        return
    assert signal.analytic.shape == (3, 2, len(epochs.times))
    assert np.isfinite(signal.analytic).all()


@pytest.mark.parametrize("band", BANDS_STANDARD, ids=lambda b: b.name)
def test_the_refusal_names_an_epoch_length_that_actually_works(band: Band) -> None:
    sfreq = 200.0
    try:
        BandSignal.from_epochs(_epochs(10.0, sfreq=sfreq, dur=0.5), band, recording="test")
    except ValueError as error:
        samples = int(re.search(r"at least (\d+) samples", str(error)).group(1))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            signal = BandSignal.from_epochs(
                _epochs(10.0, sfreq=sfreq, dur=samples / sfreq), band, recording="test"
            )
        assert np.isfinite(signal.analytic).all()
        # Tight, not merely sufficient: one sample fewer must still be refused,
        # or the advice sends the caller to collect more data than they need.
        with pytest.raises(ValueError, match="at least"):
            BandSignal.from_epochs(
                _epochs(10.0, sfreq=sfreq, dur=(samples - 1) / sfreq), band, recording="test"
            )


@pytest.mark.parametrize("sfreq", (128.0, 200.0, 250.0, 500.0))
@pytest.mark.parametrize("band", BANDS_STANDARD, ids=lambda b: b.name)
def test_required_filter_length_matches_what_mne_designs(band: Band, sfreq: float) -> None:
    # The refusal is only trustworthy if it predicts MNE's own 'auto' length exactly.
    if band.fmax >= sfreq / 2.0:
        pytest.skip("band above Nyquist at this sampling rate")
    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _Capture()
    logger = logging.getLogger("mne")
    logger.addHandler(handler)
    previous = mne.set_log_level("INFO", return_old_level=True)
    try:
        mne.filter.filter_data(
            np.zeros((1, 20000)), sfreq, band.fmin, band.fmax, filter_length="auto", verbose="info"
        )
    finally:
        logger.removeHandler(handler)
        mne.set_log_level(previous)

    designed = int(
        re.search(r"(\d+) samples", [r for r in records if "Filter length:" in r][-1]).group(1)
    )
    assert _required_filter_length(band, sfreq) == designed


def test_delta_on_short_epochs_no_longer_raises_mnes_internal_error() -> None:
    # Regression: two-second epochs at 250 Hz used to fail with MNE's
    # 'filter_length, if a string, must be a human-readable time' error.
    epochs = _epochs(2.0, sfreq=250.0, dur=2.0)
    signal = BandSignal.from_epochs(epochs, Band("delta", 1.0, 4.0), recording="test")
    assert signal.analytic.shape == (3, 2, 500)
    assert np.isfinite(signal.analytic).all()
