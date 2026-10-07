"""The Morlet path: wavelet power as a density comparable to a Welch PSD.

:meth:`eegtable.Spectra.from_tfr` divides MNE's unit-energy wavelet power by the
sampling rate and calls the result a density in V²/Hz. If that is right, the band
mean of the wavelet power equals the band mean of a Welch PSD of the same window
up to the wavelet's smoothing, and it does not move when the recording is
resampled. Both are checked on the motor data.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import eegtable as ef
from tests.validation.loaders import Recording

MU = ef.Band("mu", 8.0, 13.0)
BETA = ef.Band("beta", 13.0, 30.0)
WINDOW = ef.Window("movement", 0.5, 3.5)
FREQS = np.arange(6.0, 31.0, 1.0)
N_CYCLES = FREQS / 2.0

DATASET = "eegbci"


def _morlet_band_means(recording: Recording, epochs: object) -> ef.FeatureTable:
    sfreq = float(epochs.info["sfreq"])  # type: ignore[attr-defined]
    tfr = epochs.compute_tfr(  # type: ignore[attr-defined]
        "morlet", freqs=FREQS, n_cycles=N_CYCLES, use_fft=True, return_itc=False, average=False
    )
    spectra = ef.Spectra.from_tfr(
        tfr, [WINDOW], recording=recording.name, n_cycles=N_CYCLES, sfreq=sfreq, zero_mean=True
    )
    return ef.mean_tfr_power(spectra, bands=[MU, BETA], include_global=False)


def _welch_band_means(recording: Recording, epochs: object) -> ef.FeatureTable:
    sfreq = float(epochs.info["sfreq"])  # type: ignore[attr-defined]
    parameters = {
        "method": "welch",
        "fmin": 1.0,
        "fmax": min(40.0, sfreq / 2.0 - 1.0),
        "tmin": WINDOW.tmin,
        "tmax": WINDOW.tmax,
        "n_fft": int(2 * sfreq),
        "n_per_seg": int(2 * sfreq),
        "n_overlap": int(sfreq),
    }
    spectrum = epochs.compute_psd(**parameters)  # type: ignore[attr-defined]
    spectra = ef.Spectra.from_spectrum(
        spectrum, recording=recording.name, estimator_parameters=parameters
    )
    return ef.mean_psd(spectra, bands=[MU, BETA], include_global=False)


@pytest.fixture(scope="module")
def first(eegbci_recordings: list[Recording]) -> Recording:
    return eegbci_recordings[0]


@pytest.mark.validates(
    "Spectra.from_tfr",
    "mean_tfr_power",
    kind="estimator",
    claim="Wavelet power divided by the sampling rate matches a Welch PSD of the same window",
    criterion="median ratio within 0.9 to 1.1; log correlation above 0.95",
)
@pytest.mark.parametrize("band", [MU, BETA], ids=lambda band: band.name)
def test_wavelet_density_matches_welch(
    first: Recording, band: ef.Band, record: Callable[[str], None]
) -> None:
    morlet = _morlet_band_means(first, first.epochs).select(band=band).values
    welch = _welch_band_means(first, first.epochs).select(band=band).values
    ratio = morlet / welch
    record(
        f"{band.name}: median ratio {np.median(ratio):.2f}, log correlation "
        f"{np.corrcoef(np.log(morlet.ravel()), np.log(welch.ravel()))[0, 1]:.2f}"
    )
    assert 0.9 < np.median(ratio) < 1.1, np.median(ratio)
    assert np.corrcoef(np.log(morlet.ravel()), np.log(welch.ravel()))[0, 1] > 0.95


@pytest.mark.validates(
    "Spectra.from_tfr",
    "mean_tfr_power",
    kind="behaviour",
    claim="The wavelet density does not change when the recording is resampled",
    criterion="median ratio at 80 against 160 Hz within 2 percent of one",
)
def test_wavelet_density_does_not_depend_on_the_sampling_rate(
    first: Recording, record: Callable[[str], None]
) -> None:
    """The same recording at half the rate must report the same density."""
    original = _morlet_band_means(first, first.epochs).values
    resampled = _morlet_band_means(first, first.epochs.copy().resample(80.0)).values
    ratio = resampled / original
    record(f"median ratio 80 Hz against 160 Hz: {np.median(ratio):.3f}")
    assert abs(np.median(ratio) - 1.0) < 0.02, np.median(ratio)
    assert np.percentile(np.abs(ratio - 1.0), 90) < 0.1


@pytest.mark.validates(
    "Spectra.from_tfr",
    kind="behaviour",
    claim="Support is partial where wavelets overrun the window while coverage stays full",
    criterion="support at most 1 and below 1 somewhere; coverage exactly 1",
)
def test_support_is_partial_where_wavelets_overrun_the_window(first: Recording) -> None:
    """Support is a statement about the wavelets, coverage about finite data."""
    tfr = first.epochs.compute_tfr(
        "morlet", freqs=FREQS, n_cycles=N_CYCLES, use_fft=True, return_itc=False, average=False
    )
    spectra = ef.Spectra.from_tfr(
        tfr, [WINDOW], recording=first.name, n_cycles=N_CYCLES, sfreq=160.0, zero_mean=True
    )
    assert spectra.support.max() <= 1.0
    assert spectra.support.min() < 1.0, "a 3 s window cannot be fully supported at every bin"
    assert spectra.coverage.min() == 1.0
