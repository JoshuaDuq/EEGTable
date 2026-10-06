"""PhysioNet EEG Motor Movement/Imagery: sensorimotor desynchronization.

Hand movement suppresses mu and beta power over the sensorimotor strip, more so
over the hemisphere contralateral to the moving hand. Twenty subjects, three runs
of real fist movement each. The checks are at the group level because single
subjects in this dataset are known to vary, including a few who show no
desynchronization at all, and they read a cluster of electrodes around each hand
area rather than one electrode because the focus differs between subjects.

The physiology is read on the decibel scale, which is now the library default.
Single-trial percent change is right-skewed: a trial whose baseline happens to be
quiet reports hundreds of percent, and a handful of those dominate a mean over
forty-five trials. The log ratio is symmetric, and on it every one of the twenty
subjects shows the effect. A test below records the skew that motivated the default.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import trapezoid
from scipy.stats import ttest_1samp

import eegtable as ef
from eegtable.io import read_dataset
from eegtable.runner import load_recipe, run
from tests.validation.loaders import Recording

MU = ef.Band("mu", 8.0, 13.0)
BETA = ef.Band("beta", 13.0, 30.0)
BASELINE = ef.Window("baseline", -1.0, 0.0)
MOVEMENT = ef.Window("movement", 0.5, 3.5)
# Left and right hand areas: the central electrode with its four nearest neighbours.
HAND = {
    "left_hemisphere": ["FC3", "C5", "C3", "C1", "CP3"],
    "right_hemisphere": ["FC4", "C2", "C4", "C6", "CP4"],
}

WELCH = {"fmin": 1.0, "fmax": 40.0, "tmin": 0.5, "tmax": 3.5, "n_fft": 320, "n_per_seg": 320}

DATASET = "eegbci"


@pytest.fixture(scope="module")
def first(eegbci_recordings: list[Recording]) -> Recording:
    return eegbci_recordings[0]


@pytest.fixture(scope="module")
def welch(first: Recording) -> tuple[ef.Spectra, np.ndarray, np.ndarray]:
    spectrum = first.epochs.compute_psd("welch", **WELCH)
    spectra = ef.Spectra.from_spectrum(
        spectrum, recording=first.name, estimator_parameters={"method": "welch", **WELCH}
    )
    return spectra, spectrum.get_data(), spectrum.freqs


@pytest.fixture(scope="module")
def hand_erds(eegbci_recordings: list[Recording]) -> pd.DataFrame:
    """Per-trial mu and beta ERDS in dB over each hand area for every movement trial."""
    frames = []
    for recording in eegbci_recordings:
        moving = recording.metadata["moving"].to_numpy() == 1
        for band in (MU, BETA):
            signal = ef.BandSignal.from_epochs(recording.epochs, band, recording=recording.name)
            table = ef.erds_mean(
                [signal],
                baseline=BASELINE,
                windows=[MOVEMENT],
                groups=HAND,
                include_global=False,
                normalize="db",
            )
            values = pd.DataFrame(table.values, columns=[m.space for m in table.meta])
            values["subject"] = recording.name
            values["band"] = band.name
            values["condition"] = recording.metadata["condition"].to_numpy()
            frames.append(values[moving])
    return pd.concat(frames, ignore_index=True)


def _per_subject(frame: pd.DataFrame, band: ef.Band, column: str) -> np.ndarray:
    return frame[frame["band"] == band.name].groupby("subject")[column].mean().to_numpy()


# --- agreement with direct computation on the same real data -----------------------------


@pytest.mark.validates(
    "integrated_band_power",
    "mean_psd",
    "Spectra.from_spectrum",
    kind="formula",
    claim="Band power is the trapezoid integral of the Welch PSD; mean PSD is that per Hz",
    criterion="relative error below 1e-12",
)
def test_band_power_is_the_trapezoid_integral_of_the_psd(
    welch: tuple[ef.Spectra, np.ndarray, np.ndarray],
) -> None:
    spectra, psd, freqs = welch
    inside = (freqs >= MU.fmin) & (freqs <= MU.fmax)
    reference = trapezoid(psd[:, :, inside], freqs[inside], axis=-1)

    power = ef.integrated_band_power(spectra, bands=[MU], include_global=False)
    np.testing.assert_allclose(power.values, reference, rtol=1e-12)

    mean = ef.mean_psd(spectra, bands=[MU], include_global=False)
    np.testing.assert_allclose(mean.values, reference / (MU.fmax - MU.fmin), rtol=1e-12)


@pytest.mark.validates(
    "peak_frequency",
    kind="formula",
    claim="With every correction off, the peak is the argmax bin",
    criterion="exact equality",
)
def test_uncorrected_peak_frequency_is_the_argmax(
    welch: tuple[ef.Spectra, np.ndarray, np.ndarray],
) -> None:
    spectra, psd, freqs = welch
    inside = (freqs >= MU.fmin) & (freqs < MU.fmax)
    reference = freqs[inside][np.argmax(psd[:, :, inside], axis=-1)]

    peak = ef.peak_frequency(
        spectra,
        band=MU,
        aperiodic_adjusted=False,
        smoothing_hz=0.0,
        min_prominence=0.0,
        interpolate=False,
        include_global=False,
    )
    np.testing.assert_array_equal(peak.values, reference)


@pytest.mark.validates(
    "variance",
    "Signal.from_epochs",
    kind="formula",
    claim="Variance is NumPy's over the window",
    criterion="relative error below 1e-12",
)
def test_variance_matches_numpy(first: Recording) -> None:
    signal = ef.Signal.from_epochs(first.epochs, recording=first.name)
    table = ef.variance([signal], windows=[MOVEMENT], include_global=False)

    times = first.epochs.times
    inside = (times >= MOVEMENT.tmin) & (times <= MOVEMENT.tmax)
    reference = np.var(first.epochs.get_data(picks="eeg")[:, :, inside], axis=-1)
    np.testing.assert_allclose(table.values, reference, rtol=1e-12)


@pytest.mark.validates(
    "erds_mean",
    "BandSignal.from_epochs",
    kind="estimator",
    claim="Hilbert ERDS ranks channels and trials like MNE's multitaper time-frequency ERDS",
    criterion="correlation above 0.75 by channel and by trial-channel cell, every subject",
)
def test_hilbert_erds_agrees_with_multitaper_tfr(
    eegbci_recordings: list[Recording], record: Callable[[str], None]
) -> None:
    """Two estimators of the same quantity should rank channels and trials alike."""
    freqs = np.arange(MU.fmin, MU.fmax + 0.5, 1.0)
    correlations: list[tuple[float, float]] = []
    for recording in eegbci_recordings:
        signal = ef.BandSignal.from_epochs(recording.epochs, MU, recording=recording.name)
        # Percent, to match MNE's apply_baseline(mode="percent") below.
        own = ef.erds_mean(
            [signal],
            baseline=BASELINE,
            windows=[MOVEMENT],
            include_global=False,
            normalize="percent",
        ).values

        tfr = recording.epochs.compute_tfr(
            "multitaper",
            freqs=freqs,
            n_cycles=freqs / 2.0,
            use_fft=True,
            return_itc=False,
            average=False,
            decim=2,
        )
        tfr.crop(BASELINE.tmin, MOVEMENT.tmax)
        tfr.apply_baseline((BASELINE.tmin, BASELINE.tmax), mode="percent")
        window = (tfr.times >= MOVEMENT.tmin) & (tfr.times <= MOVEMENT.tmax)
        reference = tfr.get_data()[..., window].mean(axis=(2, 3)) * 100.0

        by_channel = np.corrcoef(own.mean(axis=0), reference.mean(axis=0))[0, 1]
        by_cell = np.corrcoef(own.ravel(), reference.ravel())[0, 1]
        correlations.append((by_channel, by_cell))
        assert by_channel > 0.75, (recording.name, by_channel)
        assert by_cell > 0.75, (recording.name, by_cell)
    lowest = np.min(correlations, axis=0)
    highest = np.max(correlations, axis=0)
    record(
        f"r by channel {lowest[0]:.2f} to {highest[0]:.2f}; by cell {lowest[1]:.2f} to "
        f"{highest[1]:.2f} over {len(correlations)} subjects"
    )


# --- known physiology ------------------------------------------------------------------


@pytest.mark.validates(
    "erds_mean",
    kind="physiology",
    claim="Hand movement desynchronizes mu and beta over the hand areas",
    criterion="per-subject mean dB below zero; one-sided t-test p below 0.05 across 20 subjects",
)
@pytest.mark.parametrize("band", [MU, BETA], ids=lambda band: band.name)
def test_movement_desynchronizes_sensorimotor_rhythms(
    hand_erds: pd.DataFrame, band: ef.Band, record: Callable[[str], None]
) -> None:
    per_trial = hand_erds.assign(hand=hand_erds[list(HAND)].mean(axis=1))
    per_subject = _per_subject(per_trial, band, "hand")

    result = ttest_1samp(per_subject, 0.0, alternative="less")
    record(
        f"{band.name}: mean {per_subject.mean():.1f} dB, range {per_subject.min():.1f} to "
        f"{per_subject.max():.1f}, negative in {int((per_subject < 0).sum())} of "
        f"{per_subject.size}, p = {result.pvalue:.0e}"
    )
    assert per_subject.mean() < 0.0, per_subject.round(1)
    assert result.pvalue < 0.05, (per_subject.round(1), result.pvalue)


@pytest.mark.validates(
    "erds_mean",
    kind="behaviour",
    claim="Single-trial percent ERDS is right-skewed, which is why dB is the default",
    criterion="trial mean above trial median in at least 75 percent of subjects",
)
def test_percent_change_is_right_skewed_on_single_trials(
    eegbci_recordings: list[Recording], record: Callable[[str], None]
) -> None:
    """Why the checks above read decibels: the trial mean of percent change sits above
    its median in most subjects, pulled up by trials with a quiet baseline."""
    above = 0
    for recording in eegbci_recordings:
        moving = recording.metadata["moving"].to_numpy() == 1
        signal = ef.BandSignal.from_epochs(recording.epochs, MU, recording=recording.name)
        percent = ef.erds_mean(
            [signal],
            baseline=BASELINE,
            windows=[MOVEMENT],
            groups=HAND,
            include_global=False,
            normalize="percent",
        ).values[moving]
        above += int(np.mean(percent) > np.median(percent))
    record(f"mean above median in {above} of {len(eegbci_recordings)} subjects")
    assert above >= 0.75 * len(eegbci_recordings), above


# --- the batch runner on real epochs files -------------------------------------------------

RECIPE = """
[inputs]
root = "epochs"
pattern = "**/*_epo.fif"

[output]
root = "features"

[bands]
mu = [8.0, 13.0]
beta = [13.0, 30.0]

[windows]
baseline = [-1.0, 0.0]
movement = [0.5, 3.5]

[rois]
hand = ["C3", "C4"]

[[features]]
measure = "erds_mean"
baseline = "baseline"
spatial = ["rois"]

[[features]]
measure = "integrated_band_power"
normalize = "log10"
spatial = ["channels"]
"""


@pytest.mark.validates(
    "runner",
    "erds_mean",
    "integrated_band_power",
    kind="formula",
    claim="The runner's TSV bundle holds the API's numbers",
    criterion="relative error below 1e-9",
)
def test_runner_reproduces_the_api_on_real_recordings(
    eegbci_recordings: list[Recording], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kept = eegbci_recordings[:2]
    for recording in kept:
        path = tmp_path / "epochs" / recording.name / f"{recording.name}_task-motor_epo.fif"
        path.parent.mkdir(parents=True)
        # Double precision, so what the runner reads is exactly what the API saw;
        # MNE's default single precision would perturb every value at 1e-7.
        recording.epochs.save(path, fmt="double", overwrite=True, verbose="error")
    (tmp_path / "recipe.toml").write_text(RECIPE, encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    result = run(load_recipe(Path("recipe.toml")))
    assert result.ok, [r.label for r in result.failed]

    dataset = read_dataset(sorted((tmp_path / "features").rglob("*_features.tsv")))
    assert dataset.table.n_rows == sum(len(r.epochs) for r in kept)
    assert {"subject", "run", "condition", "moving"} <= set(dataset.targets.columns)

    # The runner's ERDS ROI columns must be the API's numbers, not merely similar ones.
    written = dataset.table.select(measure="erds_mean", space="hand")
    expected = ef.stack_rows(
        [
            ef.erds_mean(
                [
                    ef.BandSignal.from_epochs(r.epochs, band, recording=r.name)
                    for band in (MU, BETA)
                ],
                baseline=BASELINE,
                windows=[MOVEMENT],
                groups={"hand": ["C3", "C4"]},
                include_global=False,
            )
            for r in kept
        ],
        columns="identical",
    )
    assert written.values.shape == expected.values.shape
    by_band = {m.band.name: i for i, m in enumerate(written.meta) if m.band is not None}
    for i, meta in enumerate(expected.meta):
        assert meta.band is not None
        np.testing.assert_allclose(
            written.values[:, by_band[meta.band.name]], expected.values[:, i], rtol=1e-9
        )


@pytest.mark.validates(
    "BandSignal.from_epochs",
    kind="behaviour",
    claim="The loaded epochs match the published recording",
    criterion="160 Hz, 64 channels, 45 movement trials",
)
def test_epochs_are_what_the_tutorials_describe(first: Recording) -> None:
    assert first.epochs.info["sfreq"] == 160.0
    assert len(first.epochs.ch_names) == 64
    assert {"C3", "Cz", "C4"} <= set(first.epochs.ch_names)
    counts = first.metadata["condition"].value_counts()
    assert counts["left"] + counts["right"] == 45
    assert mne.pick_types(first.epochs.info, eeg=True).size == 64
