import numpy as np
import pytest

from eegtable import complexity
from eegtable.signal import Signal
from eegtable.spectra import Window


def _signal(values):
    values = np.asarray(values, dtype=float)
    return Signal.from_arrays(
        data=values.reshape(1, 1, -1),
        times=np.arange(values.size) / 100.0,
        ch_names=("C3",),
        sfreq=100.0,
        row_ids=(("complexity", 0, "event"),),
    )


def _measure(name, values, **parameters):
    assert hasattr(complexity, name), f"{name} is not implemented"
    return getattr(complexity, name)(
        [_signal(values)],
        windows=(Window("all", -np.inf, np.inf),),
        include_global=False,
        **parameters,
    )


def test_permutation_entropy_matches_antropy_and_preserves_identity():
    pytest.importorskip("antropy")
    import antropy

    values = np.random.default_rng(4).normal(size=500)
    table = _measure("permutation_entropy", values, order=4, delay=2)
    assert table.values.item() == pytest.approx(
        antropy.perm_entropy(values, order=4, delay=2, normalize=True)
    )
    assert table.row_ids == (("complexity", 0, "event"),)
    assert table.meta[0].unit == "a.u."
    assert table.coverage.item() == 1.0


def test_lempel_ziv_uses_explicit_median_binary_symbols():
    pytest.importorskip("antropy")
    import antropy

    values = np.random.default_rng(5).normal(size=200)
    table = _measure("lempel_ziv_complexity", values, symbolization="median")
    symbols = (values >= np.median(values)).astype(np.uint32)
    assert table.values.item() == pytest.approx(antropy.lziv_complexity(symbols, normalize=True))
    assert table.meta[0].computation.parameters["parameters"]["symbolization"] == "median"


def test_lempel_ziv_mean_symbolization_changes_the_actual_sequence():
    pytest.importorskip("antropy")
    import antropy

    values = np.r_[np.arange(100), np.repeat(1000.0, 5)]
    table = _measure("lempel_ziv_complexity", values, symbolization="mean")
    symbols = (values >= np.mean(values)).astype(np.uint32)
    assert table.values.item() == pytest.approx(antropy.lziv_complexity(symbols, normalize=True))


def test_detrended_fluctuation_matches_antropy():
    pytest.importorskip("antropy")
    import antropy

    values = np.random.default_rng(6).normal(size=1000)
    table = _measure("detrended_fluctuation", values)
    assert table.values.item() == pytest.approx(antropy.detrended_fluctuation(values))


@pytest.mark.parametrize(
    "name", ["permutation_entropy", "lempel_ziv_complexity", "detrended_fluctuation"]
)
def test_extended_complexity_rejects_missing_samples(name):
    pytest.importorskip("antropy")
    values = np.random.default_rng(7).normal(size=200)
    values[50] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _measure(name, values)


@pytest.mark.parametrize(
    "name,parameters",
    [
        ("permutation_entropy", {"order": 1}),
        ("permutation_entropy", {"delay": 0}),
        ("lempel_ziv_complexity", {"symbolization": "auto"}),
    ],
)
def test_extended_complexity_rejects_invalid_parameters(name, parameters):
    pytest.importorskip("antropy")
    with pytest.raises(ValueError):
        _measure(name, np.arange(200), **parameters)


@pytest.mark.parametrize(
    "name,values",
    [
        ("permutation_entropy", np.arange(2)),
        ("lempel_ziv_complexity", np.ones(200)),
        ("detrended_fluctuation", np.arange(40)),
        ("detrended_fluctuation", np.ones(200)),
    ],
)
def test_extended_complexity_rejects_unsupported_windows(name, values):
    pytest.importorskip("antropy")
    with pytest.raises(ValueError):
        _measure(name, values)


def test_dfa_refuses_windows_too_short_for_two_block_sizes():
    # Below 58 samples AntroPy fits a single block size and returns a slope of exactly 0.
    pytest.importorskip("antropy")
    values = np.cumsum(np.random.default_rng(3).normal(size=58))
    for n_samples in (50, 57):
        with pytest.raises(ValueError, match="at least 58 samples"):
            _measure("detrended_fluctuation", values[:n_samples])
    assert _measure("detrended_fluctuation", values).values.item() > 0.5


def test_dfa_accepts_multichannel_windows_with_noncontiguous_storage():
    pytest.importorskip("antropy")
    import antropy

    assert hasattr(complexity, "detrended_fluctuation")
    values = np.random.default_rng(20).normal(size=(2, 2, 1000))
    signal = Signal.from_arrays(
        data=values,
        times=np.arange(1000) / 100,
        sfreq=100,
        ch_names=("C3", "C4"),
        row_ids=(("dfa", 0, "event"), ("dfa", 1, "event")),
    )
    window = Window("middle", 1.0, 8.0)
    table = complexity.detrended_fluctuation([signal], windows=(window,), include_global=False)
    expected = np.array(
        [
            [
                antropy.detrended_fluctuation(values[epoch, channel, 100:801].copy())
                for channel in range(2)
            ]
            for epoch in range(2)
        ]
    )
    np.testing.assert_allclose(table.values, expected)
