from dataclasses import replace
from importlib.util import find_spec

import numpy as np
import pytest

from eegtable.bands import Band
from eegtable.signal import Signal
from eegtable.spectra import Window, band_integration_weights

SFREQ = 200.0
ALPHA = Band("alpha", 8.0, 13.0)
FACTORS = (1.1, 1.3, 1.5)
WINDOW = Window("all", -np.inf, np.inf)


def _signal(values=None):
    rng = np.random.default_rng(10)
    times = np.arange(2000) / SFREQ
    values = (
        rng.normal(size=times.size) + 2 * np.sin(2 * np.pi * 10 * times)
        if values is None
        else values
    )
    return Signal.from_arrays(
        data=np.asarray(values).reshape(1, 1, -1),
        times=times[: len(values)],
        ch_names=("C3",),
        sfreq=SFREQ,
        row_ids=(("irasa", 0, "event"),),
    )


def _compute(signal, **parameters):
    assert find_spec("eegtable.irasa") is not None, "IRASA is not implemented"
    from eegtable.irasa import irasa

    return irasa(
        [signal],
        windows=(WINDOW,),
        bands=(ALPHA,),
        hset=FACTORS,
        include_global=False,
        **parameters,
    )


def test_irasa_matches_neurodsp_aperiodic_fit_and_component_integrals():
    pytest.importorskip("neurodsp")
    from neurodsp.aperiodic import compute_irasa, fit_irasa

    signal = _signal()
    freqs, aperiodic, periodic = compute_irasa(
        signal.data[0, 0],
        SFREQ,
        f_range=(2, 40),
        hset=FACTORS,
        nperseg=400,
        noverlap=200,
        avg_type="mean",
        window="hann",
    )
    fit_mask = (freqs >= 2) & (freqs < 40)
    offset, slope = fit_irasa(freqs[fit_mask], aperiodic[fit_mask])
    weights = band_integration_weights(freqs, ALPHA.fmin, ALPHA.fmax)
    table = _compute(signal)
    for name, expected in [
        ("irasa_offset", offset),
        ("irasa_slope", slope),
        ("irasa_aperiodic_power", weights @ aperiodic),
        ("irasa_periodic_power", weights @ periodic),
    ]:
        assert table.select(measure=name).values.item() == pytest.approx(expected)
    assert table.row_ids == signal.row_ids
    assert table.select(measure="irasa_aperiodic_power").meta[0].band == ALPHA
    assert table.select(measure="irasa_offset").meta[0].band is None
    assert table.coverage.min() == 1.0


def test_irasa_preserves_signed_residuals_and_total_power():
    pytest.importorskip("neurodsp")
    from neurodsp.spectral import compute_spectrum

    values = np.random.default_rng(1).normal(size=2000)
    band = Band("beta", 13, 30)
    signal = _signal(values)
    assert find_spec("eegtable.irasa") is not None, "IRASA is not implemented"
    from eegtable.irasa import irasa

    table = irasa([signal], windows=(WINDOW,), bands=(band,), hset=FACTORS, include_global=False)
    freqs, power = compute_spectrum(values, SFREQ, nperseg=400, noverlap=200)
    expected = band_integration_weights(freqs, band.fmin, band.fmax) @ power
    measured = table.select(measure="irasa_aperiodic_power").values.item()
    measured += table.select(measure="irasa_periodic_power").values.item()
    assert measured == pytest.approx(expected)


@pytest.mark.parametrize("band", [Band("low", 2.1, 4.0), Band("high", 38.0, 39.9)])
def test_irasa_retains_bins_bracketing_non_grid_aligned_band_boundaries(band):
    pytest.importorskip("neurodsp")
    from neurodsp.aperiodic import compute_irasa, fit_irasa

    from eegtable.irasa import irasa

    signal = _signal()
    fit_range = (2.1, 39.9)
    freqs, aperiodic, periodic = compute_irasa(
        signal.data[0, 0],
        SFREQ,
        hset=FACTORS,
        nperseg=400,
        noverlap=200,
        avg_type="mean",
        window="hann",
    )
    weights = band_integration_weights(freqs, band.fmin, band.fmax)
    fit_mask = (freqs >= fit_range[0]) & (freqs < fit_range[1])
    offset, slope = fit_irasa(freqs[fit_mask], aperiodic[fit_mask])
    table = irasa(
        [signal],
        windows=(WINDOW,),
        bands=(band,),
        fit_range=fit_range,
        hset=FACTORS,
        include_global=False,
    )
    expected = {
        "irasa_offset": offset,
        "irasa_slope": slope,
        "irasa_aperiodic_power": aperiodic @ weights,
        "irasa_periodic_power": periodic @ weights,
    }
    for measure, value in expected.items():
        assert table.select(measure=measure).values.item() == pytest.approx(value)
    total = expected["irasa_aperiodic_power"] + expected["irasa_periodic_power"]
    measured = table.select(measure="irasa_aperiodic_power").values.item()
    measured += table.select(measure="irasa_periodic_power").values.item()
    assert measured == pytest.approx(total)


@pytest.mark.parametrize("boundary", ["passband", "nyquist"])
def test_irasa_validates_resampling_support_of_bracketing_bins(boundary):
    signal = _signal()
    fit_range = (2.1, 39.9)
    if boundary == "passband":
        signal = replace(signal, passband=(1.35, 70.0))
    else:
        fit_range = (2.1, 66.6)
    with pytest.raises(ValueError, match=boundary if boundary == "passband" else "Nyquist"):
        _compute(signal, fit_range=fit_range)


@pytest.mark.parametrize("kind", ["nan", "short", "nyquist", "passband", "constant"])
def test_irasa_refuses_unsupported_signals(kind):
    pytest.importorskip("neurodsp")
    signal = _signal()
    if kind == "nan":
        signal.data[0, 0, 40] = np.nan
    if kind == "short":
        signal = _signal(np.arange(300))
    if kind == "nyquist":
        signal = replace(signal, sfreq=100.0, times=signal.times * 2)
    if kind == "passband":
        signal = replace(signal, passband=(2.0, 45.0))
    if kind == "constant":
        signal = _signal(np.ones(2000))
    parameters = {"fit_range": (2, 40)}
    with pytest.raises(ValueError):
        _compute(signal, **parameters)


@pytest.mark.parametrize("parameters", [{"segment_seconds": 0}, {"fit_range": (0, 40)}])
def test_irasa_refuses_invalid_parameters(parameters):
    with pytest.raises(ValueError):
        _compute(_signal(), **parameters)


def test_irasa_refuses_integer_resampling_factors():
    assert find_spec("eegtable.irasa") is not None, "IRASA is not implemented"
    from eegtable.irasa import irasa

    with pytest.raises(ValueError, match="hset"):
        irasa([_signal()], windows=(WINDOW,), bands=(ALPHA,), hset=(1.0, 2.0))


@pytest.mark.parametrize("factor", [1.00001, 1.99999])
def test_irasa_refuses_factors_that_round_to_integers(factor):
    from eegtable.irasa import irasa

    with pytest.raises(ValueError, match="hset"):
        irasa([_signal()], windows=(WINDOW,), bands=(ALPHA,), hset=(factor,))


def test_irasa_band_powers_pair_up_in_a_band_ratio():
    # The band is already part of each column's identity; repeating it among the shared
    # computation parameters kept theta and beta columns from ever matching.
    pytest.importorskip("neurodsp")
    from eegtable.derived import band_ratio
    from eegtable.irasa import irasa

    bands = (Band("theta", 4.0, 8.0), Band("beta", 13.0, 30.0))
    table = irasa([_signal()], windows=(WINDOW,), bands=bands, hset=FACTORS, include_global=False)
    power = table.select(measure="irasa_aperiodic_power")
    ratio = band_ratio(power, "theta", "beta")
    expected = power.select(band="theta").values / power.select(band="beta").values
    np.testing.assert_allclose(ratio.values, expected)
