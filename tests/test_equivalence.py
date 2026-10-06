import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import eegtable as ef
from eegtable.table import ComputationSpec

_ERDS_FUNCTIONS = {
    "mean": ef.erds_mean,
    "slope": ef.erds_slope,
    "erd_magnitude": ef.erd_magnitude,
    "erd_duration": ef.erd_duration,
    "ers_magnitude": ef.ers_magnitude,
    "ers_duration": ef.ers_duration,
    "peak_latency": ef.erds_peak_latency,
    "onset_latency": ef.erds_onset_latency,
}

_BURST_FUNCTIONS = {
    "count": ef.burst_count,
    "rate": ef.burst_rate,
    "duration_mean": ef.burst_duration,
    "amp_mean": ef.burst_amplitude,
    "fraction_above": ef.fraction_above_threshold,
}

FIXTURES = Path(__file__).parent / "fixtures"
pytestmark = pytest.mark.skipif(
    not (FIXTURES / "manifest.json").exists(), reason="fixtures not generated"
)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def spectra_npz(manifest: dict[str, Any]) -> Any:
    return np.load(FIXTURES / f"spectra_{manifest['subject']}.npz", allow_pickle=False)


@pytest.fixture(scope="module")
def reference(manifest: dict[str, Any]) -> Any:
    return np.load(FIXTURES / f"reference_{manifest['subject']}.npz", allow_pickle=False)


def _tfr_spectra(spectra_npz: Any, manifest: dict[str, Any]) -> ef.Spectra:
    windows = tuple(
        ef.Window(name, bounds[0], bounds[1]) for name, bounds in manifest["windows"].items()
    )
    data = spectra_npz["tfr"].astype(float)
    times = spectra_npz["tfr_times"]
    freqs = spectra_npz["tfr_freqs"]
    stacked = []
    coverage = []
    for window in windows:
        mask = (times >= window.tmin) & (times <= window.tmax)
        stacked.append(np.nanmean(data[:, :, :, mask], axis=3))
        coverage.append(np.isfinite(data[:, :, :, mask]).mean(axis=3))
    return ef.Spectra(
        data=np.stack(stacked, axis=2),
        freqs=freqs,
        ch_names=tuple(spectra_npz["ch_names"].tolist()),
        windows=windows,
        coverage=np.stack(coverage, axis=2),
        source="morlet",
        representation="time_frequency_power",
        support=np.ones_like(np.stack(stacked, axis=2)),
        row_ids=tuple(("fixture", index, "event") for index in range(data.shape[0])),
        computation=ComputationSpec.create("fixture-morlet"),
    )


def _psd_spectra(reference: Any) -> ef.Spectra:
    data = reference["psd"].astype(float)[:, :, np.newaxis, :]
    return ef.Spectra(
        data=data,
        freqs=reference["psd_freqs"],
        ch_names=tuple(f"ch{i}" for i in range(data.shape[1])),
        windows=(ef.Window("all", -np.inf, np.inf),),
        coverage=np.isfinite(data).astype(float),
        source="welch",
        representation="psd",
        support=np.ones(data.shape),
        row_ids=tuple(("fixture", index, "event") for index in range(data.shape[0])),
        computation=ComputationSpec.create("fixture-welch"),
    )


@pytest.mark.parametrize("band_name", ["delta", "theta", "alpha", "beta", "gamma"])
def test_mean_tfr_power_is_finite_for_the_reference_recording(
    spectra_npz: Any, manifest: dict[str, Any], band_name: str
) -> None:
    band = ef.Band(band_name, manifest["bands"][band_name][0], manifest["bands"][band_name][1])
    spectra = _tfr_spectra(spectra_npz, manifest)
    table = ef.mean_tfr_power(spectra, bands=(band,), include_global=False)
    assert np.isfinite(table.values).all()
    assert all(meta.measure == "mean_tfr_power" for meta in table.meta)


@pytest.mark.parametrize(
    ("measure", "fn"),
    [
        ("centroid", ef.spectral_centroid),
        ("bandwidth", ef.spectral_bandwidth),
        ("entropy", ef.spectral_entropy),
    ],
)
@pytest.mark.parametrize("band_name", ["theta", "alpha", "beta"])
def test_descriptors_match_the_reference(
    reference: Any, manifest: dict[str, Any], measure: str, fn: Any, band_name: str
) -> None:
    band = ef.Band(band_name, manifest["bands"][band_name][0], manifest["bands"][band_name][1])
    got = fn(_psd_spectra(reference), band=band, include_global=False).values
    np.testing.assert_allclose(got, reference[f"{measure}__{band_name}"], rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("band_name", ["theta", "alpha", "beta"])
def test_spectral_edge_matches_the_reference(
    reference: Any, manifest: dict[str, Any], band_name: str
) -> None:
    band = ef.Band(band_name, manifest["bands"][band_name][0], manifest["bands"][band_name][1])
    got = ef.spectral_edge(
        _psd_spectra(reference), band=band, percentile=0.95, include_global=False
    ).values
    np.testing.assert_allclose(got, reference[f"edge__{band_name}"], rtol=0.0, atol=1e-9)


def test_aperiodic_matches_the_reference(reference: Any) -> None:
    # The reference pipeline counts its first fit as an iteration and drops its last
    # rejection without refitting, so its three iterations are two refit rounds here.
    table = ef.aperiodic(
        _psd_spectra(reference), fit_range=(2.0, 40.0), include_global=False, max_iterations=2
    )
    np.testing.assert_allclose(
        table.select(measure="slope").values,
        reference["aperiodic_slope"],
        rtol=1e-6,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        table.select(measure="offset").values,
        reference["aperiodic_offset"],
        rtol=1e-6,
        atol=1e-9,
    )


def _band_signal(
    spectra_npz: Any, reference: Any, manifest: dict[str, Any], band_name: str
) -> ef.BandSignal:
    band = ef.Band(band_name, manifest["bands"][band_name][0], manifest["bands"][band_name][1])
    analytic = reference[f"analytic__{band_name}"]
    sfreq = float(spectra_npz["sfreq"][0])
    n_times = analytic.shape[-1]
    times = np.arange(n_times) / sfreq - 7.0
    return ef.BandSignal.from_arrays(
        analytic=analytic,
        times=times,
        ch_names=tuple(spectra_npz["ch_names"].tolist()),
        band=band,
        sfreq=sfreq,
        row_ids=tuple(("fixture", index, "event") for index in range(analytic.shape[0])),
    )


@pytest.mark.parametrize("band_name", ["alpha", "beta"])
def test_band_signal_envelope_matches_the_reference(
    spectra_npz: Any, reference: Any, manifest: dict[str, Any], band_name: str
) -> None:
    signal = _band_signal(spectra_npz, reference, manifest, band_name)
    expected = reference[f"envelope__{band_name}"]
    np.testing.assert_allclose(signal.envelope, expected, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("band_name", ["alpha", "beta"])
@pytest.mark.parametrize(
    "measure",
    [
        "mean",
        "slope",
        "erd_magnitude",
        "erd_duration",
        "ers_magnitude",
        "ers_duration",
        "peak_latency",
        "onset_latency",
    ],
)
def test_erds_matches_the_reference(
    spectra_npz: Any, reference: Any, manifest: dict[str, Any], band_name: str, measure: str
) -> None:
    signal = _band_signal(spectra_npz, reference, manifest, band_name)
    base = ef.Window("base", manifest["windows"]["base"][0], manifest["windows"]["base"][1])
    stim = ef.Window("stim", manifest["windows"]["stim"][0], manifest["windows"]["stim"][1])
    # The reference pipeline's onset is the first single-sample crossing, which fires
    # on every trial. eegtable now requires the excursion to persist by default; the
    # equivalence holds for the persistence-free setting, a deliberate divergence.
    kwargs = {"min_duration_ms": 0.0} if measure == "onset_latency" else {}
    # The reference pipeline reports percent; eegtable's default is now decibels.
    table = _ERDS_FUNCTIONS[measure](
        [signal],
        baseline=base,
        windows=[stim],
        include_global=False,
        normalize="percent",
        **kwargs,
    )
    got = table.values
    expected = reference[f"erds_{measure}__{band_name}"]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("band_name", ["alpha", "beta"])
@pytest.mark.parametrize(
    "measure",
    ["count", "rate", "duration_mean", "amp_mean", "fraction_above"],
)
def test_bursts_matches_the_reference(
    spectra_npz: Any, reference: Any, manifest: dict[str, Any], band_name: str, measure: str
) -> None:
    signal = _band_signal(spectra_npz, reference, manifest, band_name)
    stim = ef.Window("stim", manifest["windows"]["stim"][0], manifest["windows"]["stim"][1])
    base = ef.Window("base", manifest["windows"]["base"][0], manifest["windows"]["base"][1])
    table = _BURST_FUNCTIONS[measure](
        [signal],
        windows=[stim],
        baseline=base,
        threshold=0.75,
        min_duration_ms=100.0,
        include_global=False,
    )
    got = table.values
    expected = reference[f"burst_{measure}__{band_name}"]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-10)
