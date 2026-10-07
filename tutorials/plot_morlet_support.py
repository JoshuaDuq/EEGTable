"""
Morlet Power: Temporal Support and Feature Quality
====================================================

Which part of a time window can a wavelet coefficient describe? Simulate an
alpha amplitude increase, verify window-averaged Morlet power against MNE, and
distinguish temporal support from numerical coverage before normalizing power.

This seeded simulation works offline with
``python -m pip install -e ".[docs]"``. The response is planted to explain the
measurement; its magnitude is not an empirical EEG finding.
"""

# %%
# Simulate a change in oscillation amplitude
# ------------------------------------------
# Twenty-four trials on Cz span -2 to 2 s at 200 Hz. A smooth transition raises
# the amplitude of a 10 Hz carrier from 2 to 6 µV around time zero. Independent
# phases, trial gains, and sensor noise provide variation. These sensor signals
# use an assumed acquisition reference; no source model is implied.
#
# MNE expects volts and arrays with shape ``(epochs, channels, samples)``.
# Keep the voltage data and time-frequency power uncorrected: power baseline
# normalization will occur after temporal and frequency reduction.

import matplotlib.pyplot as plt
import mne
import numpy as np

import eegtable as ef

rng = np.random.default_rng(53)
sampling_rate = 200.0
times = np.arange(-400, 401) / sampling_rate
n_trials = 24
amplitude = 2e-6 + 4e-6 * (1.0 + np.tanh(times / 0.08)) / 2.0
trial_gains = rng.lognormal(0.0, 0.15, (n_trials, 1))
phases = rng.uniform(-np.pi, np.pi, (n_trials, 1))
data = trial_gains * amplitude * np.sin(2.0 * np.pi * 10.0 * times + phases)
data += rng.normal(0.0, 0.6e-6, data.shape)
epochs = mne.EpochsArray(
    data[:, None, :],
    mne.create_info(["Cz"], sampling_rate, "eeg"),
    event_id={"response": 1},
    tmin=-2.0,
    baseline=None,
    verbose=False,
)

# %%
# Compute power before reducing time
# ----------------------------------
# Four cycles balance temporal and frequency smoothing for this demonstration;
# this is not a universal cycle count. No decimation is used. Every trial keeps
# its own power estimate, rather than transforming an evoked average.
# MNE's `time-frequency tutorial
# <https://mne.tools/stable/auto_tutorials/time-freq/20_sensors_time_frequency.html>`_
# explains the distinction between averaged power and phase-locked responses.
#
# MNE uses Morlet wavelets with energy 2. EEGTable divides coefficient power by
# the original sampling rate to report a wavelet-smoothed density in V²/Hz.
# This scaling does not make the estimate identical to a Welch PSD.

frequencies = np.arange(6.0, 31.0)
n_cycles = 4.0
baseline = ef.Window("baseline", -1.8, -0.2)
response = ef.Window("response", 0.2, 1.8)
windows = [baseline, response]
tfr = epochs.compute_tfr(
    "morlet",
    freqs=frequencies,
    n_cycles=n_cycles,
    zero_mean=True,
    output="power",
    average=False,
    return_itc=False,
    decim=1,
    verbose=False,
)
spectra = ef.Spectra.from_tfr(
    tfr,
    windows,
    recording="simulated-morlet",
    n_cycles=n_cycles,
    sfreq=sampling_rate,
    zero_mean=True,
)

# %%
# Verify which coefficients belong to each window
# ------------------------------------------------
# MNE truncates each wavelet at five Gaussian standard deviations on each side,
# where ``sigma_t = n_cycles / (2 * pi * frequency)``. EEGTable conservatively
# requires the entire support to fit inside both the measurement window and the
# recorded epoch. A coefficient centered inside a window can still depend on
# samples outside it. Lower frequencies need larger margins at this cycle count.
#
# Compute these masks independently and compare both the retained mean power
# and the retained fraction of requested time points. All coefficients here are
# finite: coverage is one even when only part of the window has valid support.

power = tfr.get_data()
half_support = 5.0 * n_cycles / (2.0 * np.pi * frequencies)
support_masks = []
for index, window in enumerate(windows):
    lower = max(window.tmin, tfr.times[0]) + half_support
    upper = min(window.tmax, tfr.times[-1]) - half_support
    supported = (tfr.times >= lower[:, None]) & (tfr.times <= upper[:, None])
    requested = (tfr.times >= window.tmin) & (tfr.times <= window.tmax)
    assert supported.any(axis=1).all()
    expected = np.stack(
        [power[:, :, row, mask].mean(axis=-1) for row, mask in enumerate(supported)]
    )
    np.testing.assert_allclose(
        spectra.data[:, :, index], expected.transpose(1, 2, 0) / sampling_rate
    )
    fraction = supported.sum(axis=1) / requested.sum()
    np.testing.assert_allclose(
        spectra.support[:, :, index], np.broadcast_to(fraction, (n_trials, 1, len(frequencies)))
    )
    support_masks.append(supported)
np.testing.assert_array_equal(spectra.coverage, np.ones_like(spectra.coverage))

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), layout="constrained")
density = power.mean(axis=(0, 1)) / sampling_rate * 1e12
image = axes[0].pcolormesh(tfr.times, frequencies, density, shading="auto")
axes[0].set(xlabel="Time (s)", ylabel="Frequency (Hz)", title="Mean simulated power")
fig.colorbar(image, ax=axes[0], label="Wavelet density (µV²/Hz)")
axes[1].pcolormesh(
    tfr.times, frequencies, support_masks[1], shading="auto", cmap="Greys", vmin=0, vmax=1
)
axes[1].axvline(response.tmin, color="tab:red", linestyle="--")
axes[1].axvline(response.tmax, color="tab:red", linestyle="--")
axes[1].set(xlabel="Time (s)", ylabel="Frequency (Hz)", title="Response support: black = retained")
plt.show()

# %%
# Normalize the band mean and verify its order of operations
# ----------------------------------------------------------
# ``mean_tfr_power`` averages the supported power over time and then averages
# over frequency with piecewise-linear band weights. Normalization takes
# ``10 * log10(response_mean / baseline_mean)`` within each trial/channel.
# It does not average a pointwise dB trace; taking a logarithm before averaging
# generally changes the result. :func:`~eegtable.erds_mean` instead averages a
# per-sample dB trace from Hilbert band power. These are different estimands,
# computed with different spectral estimators.
#
# Each output row is still one trial. Baseline is consumed by normalization,
# and the smaller support/coverage of the two windows accompanies the result.

bands = [ef.Band("theta", 6.0, 8.0), ef.Band("alpha", 8.0, 13.0), ef.Band("beta", 13.0, 30.0)]
raw_power = ef.mean_tfr_power(spectra, bands=bands, include_global=False)
normalized = ef.mean_tfr_power(
    spectra, bands=bands, baseline="baseline", normalize="db", include_global=False
)
for column, feature in enumerate(normalized.meta):
    base_column = next(
        index
        for index, meta in enumerate(raw_power.meta)
        if meta.band == feature.band and meta.window == "baseline"
    )
    response_column = next(
        index
        for index, meta in enumerate(raw_power.meta)
        if meta.band == feature.band and meta.window == "response"
    )
    expected = 10.0 * np.log10(
        raw_power.values[:, response_column] / raw_power.values[:, base_column]
    )
    np.testing.assert_allclose(normalized.values[:, column], expected)
    expected_support = np.minimum(
        raw_power.support[:, base_column], raw_power.support[:, response_column]
    )
    np.testing.assert_allclose(normalized.support[:, column], expected_support)
print(
    normalized.to_long()[["epoch", "band", "window", "unit", "value", "coverage", "support"]].head()
)

# %%
# Apply a stated quality rule and retain its evidence
# ---------------------------------------------------
# For this teaching example, require at least 60% temporal support and complete
# numerical coverage. This threshold is chosen to illustrate the distinction,
# not to recommend a research cutoff. Choose a study's window, cycle count and
# quality policy before examining outcomes; report exclusions by condition.
# The policy applies to each feature's frequency-weighted band support. The
# figure below shows individual frequency-bin support; some alpha bins fall
# below 60% even though the band's weighted support passes. The baseline and
# response curves coincide because their windows have equal durations.
#
# Low-support values become NaN while row identities, metadata, and measured
# support remain intact. A ledger records the reason for every rejected cell.
# The original table remains available for inspection and is not modified.

quality = ef.apply_quality(normalized, ef.QualityPolicy(min_coverage=1.0, min_support=0.60))
rejected = normalized.support < 0.60
np.testing.assert_array_equal(np.isnan(quality.table.values), rejected)
assert quality.table.row_ids == normalized.row_ids
np.testing.assert_allclose(quality.table.support, normalized.support)
assert np.isfinite(normalized.values).all()
assert set(quality.ledger["reason"]) == {"low_support"}
print(quality.ledger[["row", "reason", "coverage"]].head().to_string(index=False))
print(f"Rejected {len(quality.ledger)} of {normalized.values.size} trial/feature cells.")

fig, ax = plt.subplots(figsize=(6.5, 3.3), layout="constrained")
for index, window in enumerate(windows):
    ax.plot(frequencies, spectra.support[0, 0, index], label=window.name)
ax.axhline(0.60, color="0.5", linestyle="--", label="Illustrative minimum support")
ax.set(xlabel="Frequency (Hz)", ylabel="Retained fraction of window", ylim=(0.0, 1.0))
ax.legend()
plt.show()

# %%
# Interpret support and choose an appropriate measurement
# -------------------------------------------------------
# Full coverage establishes numerical finiteness of supported coefficients;
# it does not establish artifact-free EEG. Full wavelet support limits temporal
# mixing across window boundaries, but it does not eliminate smoothing or make
# adjacent coefficients independent. Wider windows retain a larger fraction;
# fewer cycles trade frequency resolution for shorter temporal support.
#
# Neither a finite feature nor a support threshold establishes physiological
# specificity. Inspect waveforms and time-frequency maps, and retain quality
# evidence with the feature table. The planted alpha carrier also contributes
# to nearby frequency bins through wavelet smoothing. Changes in the theta or
# beta band here do not establish a separate oscillation in either band.
# See :doc:`/methods/spectral` for the density
# scaling and reduction definitions, :doc:`/guides/cohorts` for quality policies,
# and :doc:`plot_motor_erds` for a Hilbert-power ERD measurement on public EEG.
