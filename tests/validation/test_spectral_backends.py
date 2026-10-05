"""Spectral backend agreement on the existing SSVEP and motor datasets.

Direct specparam and NeuroDSP calls validate integration and feature selection.
SciPy quadrature and Welch separately check signed IRASA power conservation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.signal import welch

import eegtable as ef
from tests.validation.loaders import Recording, load_eegbci

PEAK_BANDS = (ef.Band("drive", 11.0, 16.0), ef.Band("harmonic", 23.0, 31.0))
IRASA_BANDS = (
    ef.Band("low", 2.1, 4.4),
    ef.Band("mu", 8.2, 12.8),
    ef.Band("beta", 13.2, 29.8),
    ef.Band("high", 38.6, 39.9),
)
IRASA_WINDOW = ef.Window("segment", -0.5, 3.5)
IRASA_FACTORS = (1.1, 1.3, 1.5, 1.7, 1.9)
IRASA_FIT = (2.1, 39.9)


def _integrate(freqs: np.ndarray, power: np.ndarray, band: ef.Band) -> np.ndarray:
    inside = (freqs > band.fmin) & (freqs < band.fmax)
    grid = np.r_[band.fmin, freqs[inside], band.fmax]
    return np.array(
        [trapezoid(np.interp(grid, freqs, trace), grid) for trace in power.reshape(-1, freqs.size)]
    ).reshape(power.shape[:-1])


@pytest.fixture(scope="module")
def ssvep_spectra(ssvep_recording: Recording) -> ef.Spectra:
    parameters = {
        "method": "welch",
        "fmin": 1.0,
        "fmax": 45.0,
        "tmin": 1.0,
        "tmax": 20.0,
        "n_fft": 2000,
        "n_per_seg": 2000,
        "n_overlap": 1000,
        "picks": ["Oz"],
    }
    spectrum = ssvep_recording.epochs.compute_psd(**parameters)
    return ef.Spectra.from_spectrum(
        spectrum, recording=ssvep_recording.name, estimator_parameters=parameters
    )


@pytest.mark.validates(
    "spectral_parameterization",
    kind="estimator",
    dataset="ssvep",
    claim="SSVEP fixed/knee model parameters, diagnostics and selected peaks match specparam",
    criterion="rtol=1e-10, atol=1e-12 on every Oz trial; absent peaks are NaN and flagged",
)
@pytest.mark.parametrize("mode", ["fixed", "knee"])
def test_spectral_parameterization_matches_specparam(
    ssvep_spectra: ef.Spectra, mode: str, record: Callable[[str], None]
) -> None:
    specparam = pytest.importorskip("specparam")
    settings = {
        "aperiodic_mode": mode,
        "peak_width_limits": (0.5, 8.0),
        "max_n_peaks": 5,
        "min_peak_height": 0.1,
        "peak_threshold": 2.0,
    }
    spectra = ssvep_spectra
    table = ef.spectral_parameterization(
        spectra, bands=PEAK_BANDS, fit_range=(2.0, 40.0), include_global=False, **settings
    )
    inside = (spectra.freqs >= 2.0) & (spectra.freqs < 40.0)
    expected = np.full_like(table.values, np.nan)
    fields = {
        "specparam_offset": "offset",
        "specparam_exponent": "exponent",
        "specparam_knee": "knee",
    }
    if mode == "fixed":
        fields.pop("specparam_knee")
    peak_fields = ("specparam_peak_cf", "specparam_peak_height", "specparam_peak_width")
    parameters = (*fields, "specparam_n_peaks", "specparam_r_squared", "specparam_error")
    expected_columns = {(measure, None, "Oz", "all") for measure in parameters}
    expected_columns.update(
        (measure, band, "Oz", "all") for measure in peak_fields for band in PEAK_BANDS
    )
    assert len(table.meta) == len(expected_columns)
    assert {
        (meta.measure, meta.band, meta.space, meta.window) for meta in table.meta
    } == expected_columns
    for epoch in range(table.n_rows):
        model = specparam.SpectralModel(
            **settings, periodic_mode="gaussian", debug=True, verbose=False
        )
        model.fit(spectra.freqs[inside], spectra.data[epoch, 0, 0, inside])
        peaks = model.get_params("periodic")
        for column, meta in enumerate(table.meta):
            if meta.measure in fields:
                expected[epoch, column] = model.get_params("aperiodic", fields[meta.measure])
            elif meta.measure == "specparam_n_peaks":
                expected[epoch, column] = model.results.n_peaks
            elif meta.measure == "specparam_r_squared":
                expected[epoch, column] = model.get_metrics("gof", "rsquared")
            elif meta.measure == "specparam_error":
                expected[epoch, column] = model.get_metrics("error", "mae")
            else:
                assert meta.band is not None
                candidates = peaks[(peaks[:, 0] >= meta.band.fmin) & (peaks[:, 0] < meta.band.fmax)]
                if len(candidates):
                    strongest = candidates[np.argmax(candidates[:, 1])]
                    expected[epoch, column] = strongest[peak_fields.index(meta.measure)]
    finite = np.isfinite(expected)
    error = float(np.max(np.abs(table.values[finite] - expected[finite])))
    record(
        f"{mode}: {table.n_rows} Oz trials, {finite.sum()} finite parameters, "
        f"{np.isnan(expected).sum()} absent peak parameters; max absolute error {error:.2e}"
    )
    np.testing.assert_allclose(table.values, expected, rtol=1e-10, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(table.flags["spectral_no_peak"], np.isnan(expected))
    assert table.row_ids == spectra.row_ids


@pytest.fixture(scope="module")
def motor_signal() -> ef.Signal:
    recording = load_eegbci(1)
    conditions = recording.metadata["condition"].to_numpy()
    rows = np.concatenate(
        [np.flatnonzero(conditions == condition)[:2] for condition in ("rest", "left", "right")]
    )
    assert rows.size == 6
    epochs = recording.epochs[np.sort(rows)].copy().pick(["C3", "C4"])
    return ef.Signal.from_epochs(epochs, recording=recording.name)


@pytest.fixture(scope="module")
def irasa_reference(motor_signal: ef.Signal) -> tuple[ef.FeatureTable, dict[str, np.ndarray]]:
    neurodsp = pytest.importorskip("neurodsp.aperiodic")
    signal = motor_signal
    table = ef.irasa(
        [signal],
        windows=[IRASA_WINDOW],
        bands=IRASA_BANDS,
        fit_range=IRASA_FIT,
        hset=IRASA_FACTORS,
        segment_seconds=2.0,
        include_global=False,
    )
    inside = (signal.times >= IRASA_WINDOW.tmin) & (signal.times <= IRASA_WINDOW.tmax)
    traces = signal.data[..., inside]
    reference = {name: np.empty(traces.shape[:2]) for name in ("irasa_offset", "irasa_slope")}
    for band in IRASA_BANDS:
        for component in ("aperiodic", "periodic"):
            reference[f"irasa_{component}_power/{band.name}"] = np.empty(traces.shape[:2])
    for cell in np.ndindex(traces.shape[:2]):
        freqs, aperiodic, periodic = neurodsp.compute_irasa(
            traces[cell],
            signal.sfreq,
            hset=IRASA_FACTORS,
            thresh=None,
            nperseg=320,
            noverlap=160,
            avg_type="mean",
            window="hann",
        )
        fit = (freqs >= IRASA_FIT[0]) & (freqs < IRASA_FIT[1])
        offset, slope = neurodsp.fit_irasa(freqs[fit], aperiodic[fit])
        reference["irasa_offset"][cell] = offset
        reference["irasa_slope"][cell] = slope
        for band in IRASA_BANDS:
            reference[f"irasa_aperiodic_power/{band.name}"][cell] = _integrate(
                freqs, aperiodic, band
            )
            reference[f"irasa_periodic_power/{band.name}"][cell] = _integrate(freqs, periodic, band)
    return table, reference


@pytest.mark.validates(
    "irasa",
    kind="estimator",
    dataset="eegbci",
    claim="Motor-task IRASA aperiodic offset and slope agree with direct NeuroDSP fits",
    criterion="rtol=1e-10, atol=1e-12 on six trials at C3 and C4",
)
def test_irasa_fit_matches_neurodsp(
    motor_signal: ef.Signal,
    irasa_reference: tuple[ef.FeatureTable, dict[str, np.ndarray]],
    record: Callable[[str], None],
) -> None:
    table, reference = irasa_reference
    errors = []
    for measure in ("irasa_offset", "irasa_slope"):
        values = table.select(measure=measure).values
        errors.append(float(np.max(np.abs(values - reference[measure]))))
        np.testing.assert_allclose(values, reference[measure], rtol=1e-10, atol=1e-12)
    record(f"six trials, two channels; max absolute offset/slope error {max(errors):.2e}")
    assert table.row_ids == motor_signal.row_ids


@pytest.mark.validates(
    "irasa",
    kind="estimator",
    dataset="eegbci",
    claim="Motor-task IRASA signed component integrals agree with NeuroDSP and SciPy quadrature",
    criterion="rtol=1e-10, atol=1e-24 including off-grid edges near both fit-range boundaries",
)
@pytest.mark.parametrize("band", IRASA_BANDS, ids=lambda band: band.name)
def test_irasa_component_integrals_match_neurodsp(
    irasa_reference: tuple[ef.FeatureTable, dict[str, np.ndarray]],
    band: ef.Band,
    record: Callable[[str], None],
) -> None:
    table, reference = irasa_reference
    errors = []
    for component in ("aperiodic", "periodic"):
        measure = f"irasa_{component}_power"
        expected = reference[f"{measure}/{band.name}"]
        values = table.select(measure=measure, band=band).values
        errors.append(float(np.max(np.abs(values - expected))))
        np.testing.assert_allclose(values, expected, rtol=1e-10, atol=1e-24)
    record(f"{band.name}: max absolute component integral error {max(errors):.2e} V^2")


@pytest.mark.validates(
    "irasa",
    kind="formula",
    dataset="eegbci",
    claim="Signed IRASA components sum to an independently computed SciPy Welch band integral",
    criterion="rtol=1e-10, atol=1e-24 in four bands on all selected motor trials/channels",
)
def test_irasa_components_preserve_welch_power(
    motor_signal: ef.Signal,
    irasa_reference: tuple[ef.FeatureTable, dict[str, np.ndarray]],
    record: Callable[[str], None],
) -> None:
    signal = motor_signal
    table, _ = irasa_reference
    inside = (signal.times >= IRASA_WINDOW.tmin) & (signal.times <= IRASA_WINDOW.tmax)
    freqs, power = welch(
        signal.data[..., inside],
        fs=signal.sfreq,
        window="hann",
        nperseg=320,
        noverlap=160,
        detrend="constant",
        scaling="density",
        average="mean",
        axis=-1,
    )
    errors = []
    for band in IRASA_BANDS:
        expected = _integrate(freqs, power, band)
        actual = table.select(measure="irasa_aperiodic_power", band=band).values
        actual = actual + table.select(measure="irasa_periodic_power", band=band).values
        errors.append(float(np.max(np.abs(actual - expected) / expected)))
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-24)
    record(
        f"four bands, six trials, two channels; max relative total-power error {max(errors):.2e}"
    )
