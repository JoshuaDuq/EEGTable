"""
Detecting Beta Bursts with a Baseline Threshold
===============================================

Band power alone cannot describe whether an oscillation appears as brief events
or as sustained activity. Simulate both patterns, inspect their Hilbert
envelopes, and measure the events that survive a baseline threshold and a
minimum duration.

This example is fully simulated and requires no download. Its envelopes are
designed to illustrate the detector, not to reproduce a biological EEG process.
"""

# %%
# Create brief and sustained beta activity
# ----------------------------------------
# Each of 24 trials has one central channel sampled at 200 Hz, from -2 to 3 s.
# Two Gaussian amplitude pulses describe the brief condition; a smooth plateau
# describes the sustained condition. Both modulate a 20 Hz carrier above a
# constant 3 µV background oscillation. Independent noise and carrier phases
# provide trial variation. Arrays passed to MNE are in volts and have dimensions
# ``(epochs, channels, samples)``, following MNE's `array construction tutorial
# <https://mne.tools/stable/auto_tutorials/simulation/10_array_objs.html>`_.

import matplotlib.pyplot as plt
import mne
import numpy as np

import eegtable as ef

rng = np.random.default_rng(17)
sampling_rate = 200.0
times = np.arange(-400, 601) / sampling_rate
trials_per_condition = 12
labels = np.repeat(["brief", "sustained"], trials_per_condition)
brief = 3e-6 + 12e-6 * (
    np.exp(-0.5 * ((times - 0.65) / 0.10) ** 2) + np.exp(-0.5 * ((times - 1.45) / 0.10) ** 2)
)
sustained = 3e-6 + 9e-6 * (np.tanh((times - 0.35) / 0.08) - np.tanh((times - 1.75) / 0.08)) / 2.0
envelopes = np.repeat(np.stack([brief, sustained]), trials_per_condition, axis=0)
phases = rng.uniform(0.0, 2.0 * np.pi, size=(len(labels), 1))
data = envelopes * np.sin(2.0 * np.pi * 20.0 * times + phases)
data += rng.normal(0.0, 0.4e-6, data.shape)
events = np.column_stack(
    [
        np.arange(len(labels)) * len(times),
        np.zeros(len(labels), int),
        np.repeat([1, 2], trials_per_condition),
    ]
)
epochs = mne.EpochsArray(
    data[:, None, :],
    mne.create_info(["C3"], sampling_rate, "eeg"),
    events=events,
    event_id={"brief": 1, "sustained": 2},
    tmin=-2.0,
    baseline=None,
    verbose=False,
)

# %%
# Inspect the filtered envelope and threshold
# -------------------------------------------
# ``BandSignal.from_epochs`` bandpasses 13–30 Hz, applies the Hilbert transform,
# and removes its reflect padding. Padding reduces edge distortion but supplies
# no new recorded samples; keep baseline and measurement windows in the epoch
# interior. The detector calibrates its 95th percentile separately for each
# trial/channel on the baseline, before the simulated response.

signal = ef.BandSignal.from_epochs(epochs, ef.Band("beta", 13.0, 30.0), recording="simulated-beta")
baseline = ef.Window("baseline", -1.5, -0.5)
response = ef.Window("response", 0.2, 1.9)
baseline_mask = (signal.times >= baseline.tmin) & (signal.times <= baseline.tmax)
thresholds = np.quantile(signal.envelope[:, 0, baseline_mask], 0.95, axis=-1)

fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True, layout="constrained")
for ax, row in zip(axes, [0, trials_per_condition], strict=True):
    ax.plot(times, data[row] * 1e6, color="0.8", linewidth=0.7, label="EEG")
    ax.plot(times, signal.envelope[row, 0] * 1e6, label="Beta envelope")
    ax.axhline(thresholds[row] * 1e6, color="tab:red", linestyle="--", label="Threshold")
    ax.axvspan(baseline.tmin, baseline.tmax, color="0.9")
    ax.axvspan(response.tmin, response.tmax, color="tab:green", alpha=0.08)
    ax.set(title=labels[row], ylabel="Amplitude (µV)")
axes[0].legend(loc="upper right")
axes[-1].set_xlabel("Time (s)")
plt.show()

# %%
# Count events and measure their durations
# ----------------------------------------
# A burst is a contiguous above-threshold interval lasting at least 150 ms.
# Count is zero when no interval qualifies; mean duration is undefined when
# there are no qualifying bursts. Rate divides count by the sampled window
# duration. The result has one row per trial, suitable for later within-person
# summaries. See :doc:`/methods/dynamics` for boundary and missing-value rules.

counts = ef.burst_count(
    [signal],
    windows=[response],
    baseline=baseline,
    threshold=0.95,
    min_duration_ms=150.0,
    include_global=False,
)
durations = ef.burst_duration(
    [signal],
    windows=[response],
    baseline=baseline,
    threshold=0.95,
    min_duration_ms=150.0,
    include_global=False,
)
table = ef.concat([counts, durations])
long = table.to_long()
print(long.groupby(["event", "measure"])["value"].agg(["mean", "min", "max"]))
np.testing.assert_array_equal(counts.values[:trials_per_condition, 0], 2.0)
np.testing.assert_array_equal(counts.values[trials_per_condition:, 0], 1.0)

# %%
# Verify duration from sampled threshold crossings
# ------------------------------------------------
# Pad with false values so events touching either analysis-window edge still
# have a start and stop. A run of N above-threshold samples lasts N / sfreq,
# including one sampling interval beyond the difference of its sample times.
# Only runs meeting the 150 ms duration requirement enter the mean.

response_mask = (signal.times >= response.tmin) & (signal.times <= response.tmax)
minimum_samples = int(np.ceil(0.150 * signal.sfreq))
for row, threshold in enumerate(thresholds):
    above = signal.envelope[row, 0, response_mask] > threshold
    crossings = np.diff(np.concatenate([[False], above, [False]]).astype(int))
    starts = np.flatnonzero(crossings == 1)
    stops = np.flatnonzero(crossings == -1)
    lengths = stops - starts
    retained_lengths = lengths[lengths >= minimum_samples]
    assert len(retained_lengths) == counts.values[row, 0]
    np.testing.assert_allclose(durations.values[row, 0], retained_lengths.mean() / signal.sfreq)

# %%
# Check sensitivity to the threshold
# ----------------------------------
# There is no universal burst threshold. Repeat a prespecified range and inspect
# whether the description changes. Increasing the threshold can shorten or
# split events; a minimum-duration rule can then remove them. Choose settings
# using the measurement definition and data quality, before testing outcomes.

quantiles = [0.75, 0.90, 0.95, 0.99]
mean_counts = []
for quantile in quantiles:
    detected = ef.burst_count(
        [signal],
        windows=[response],
        baseline=baseline,
        threshold=quantile,
        min_duration_ms=150.0,
        include_global=False,
    )
    mean_counts.append(detected.values[:, 0].reshape(2, trials_per_condition).mean(axis=1))

fig, ax = plt.subplots(figsize=(6.0, 3.3), layout="constrained")
for index, condition in enumerate(["brief", "sustained"]):
    ax.plot(quantiles, np.asarray(mean_counts)[:, index], "o-", label=condition)
ax.set(xlabel="Baseline envelope quantile", ylabel="Mean bursts per trial")
ax.legend()
plt.show()

# %%
# What the detector measures
# --------------------------
# It recovers two brief events and one sustained event at the stated settings,
# even though both conditions have increased beta amplitude. These are detected
# envelope excursions, not proof of distinct neural generators. Filtering,
# waveform shape, noise, threshold, and minimum duration all affect the result.
# Events crossing a measurement-window edge have only their visible duration
# counted; do not interpret that value as their full lifetime.
#
# Next, :doc:`plot_phase_consistency` changes phase while holding amplitude fixed.
