"""
Measuring ERPs Without Losing Trial Identities
==============================================

How do a component's mean amplitude and peak latency describe individual
trials? Simulate a positive ERP, remove two known artifacts, and extract
measurements in a prespecified time window while retaining the original trials.

This standalone simulation needs no download. Install
``python -m pip install -e ".[docs]"`` to run it. The planted component illustrates
measurement choices; its shape and condition effects are not empirical results.
"""

# %%
# Simulate trials with amplitude and latency variation
# ----------------------------------------------------
# Sixty alternating standard/target trials contain a positive Gaussian component
# around 300/340 ms, with amplitudes of approximately 4/8 µV. Trial jitter and
# independent noise make the measured peaks imperfect estimates of the planted
# latencies. Three parietal channels have different gains at an assumed common
# reference; these are sensor waveforms, not a source simulation.
#
# Arrays passed to MNE have shape ``(trials, channels, samples)`` and use volts.
# Metadata records trial numbers before rejection. A fixed seed makes the
# simulation and its artifact locations reproducible.

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

import eegtable as ef

rng = np.random.default_rng(41)
sampling_rate = 250.0
times = np.arange(-50, 201) / sampling_rate
codes = np.tile([1, 2], 30)
conditions = np.where(codes == 1, "standard", "target")
latencies = np.where(codes == 1, 0.30, 0.34) + rng.normal(0.0, 0.012, len(codes))
amplitudes = np.where(codes == 1, 4e-6, 8e-6) * rng.lognormal(0.0, 0.12, len(codes))
component = amplitudes[:, None] * np.exp(-0.5 * ((times - latencies[:, None]) / 0.045) ** 2)
data = component[:, None, :] * np.array([0.8, 1.0, 1.2])[None, :, None]
data += rng.normal(0.0, 0.7e-6, data.shape)
data += rng.normal(0.0, 0.5e-6, (len(codes), 3, 1))
artifact_trials = [4, 5]
data[artifact_trials, :, np.argmin(np.abs(times - 0.45))] += 180e-6
metadata = pd.DataFrame({"trial_number": np.arange(len(codes)), "planted_latency": latencies})
events = np.column_stack([np.arange(len(codes)) * len(times), np.zeros(len(codes), int), codes])
epochs = mne.EpochsArray(
    data,
    mne.create_info(["P3", "Pz", "P4"], sampling_rate, "eeg"),
    events=events,
    event_id={"standard": 1, "target": 2},
    metadata=metadata,
    tmin=-0.2,
    baseline=None,
    verbose=False,
)

# %%
# Correct the baseline and reject the planted artifacts
# -----------------------------------------------------
# Baseline correction subtracts each trial/channel's mean from -200 to 0 ms.
# It is a voltage correction, distinct from the power normalization in the ERD
# tutorial. Peak-to-peak rejection examines the entire epoch at a fixed 100 µV
# threshold, selected here to remove the two planted spikes. This demonstration
# does not replace reviewed preprocessing on real recordings.
#
# MNE retains original epoch numbers in ``selection`` and updates metadata when
# trials are dropped; see its `metadata tutorial
# <https://mne.tools/stable/auto_tutorials/epochs/30_epochs_metadata.html>`_.

epochs.apply_baseline((-0.2, 0.0), verbose=False)
epochs.drop_bad(reject={"eeg": 100e-6}, verbose=False)
np.testing.assert_array_equal(
    epochs.selection, np.setdiff1d(np.arange(len(codes)), artifact_trials)
)
np.testing.assert_array_equal(epochs.metadata["trial_number"], epochs.selection)
print(f"Retained {len(epochs)} of {len(codes)} original trials.")

# %%
# Inspect averages in an independently chosen window
# --------------------------------------------------
# The 220–420 ms window covers the simulated component by construction. For a
# research hypothesis, choose the window and channels from prior work or an
# independent localizer before examining the condition effect. MNE's `ERP
# tutorial <https://mne.tools/stable/auto_tutorials/evoked/30_eeg_erp.html>`_
# discusses mean amplitude and peak measurements in this setting.
#
# Condition averages show the waveform; each feature row below still represents
# one retained trial. The band of color marks the measurement window, not a
# confidence interval or a significant cluster.

window = ef.Window("component", 0.22, 0.42)
groups = {"parietal": ["P3", "Pz", "P4"]}
evokeds = {condition: epochs[condition].average() for condition in ["standard", "target"]}
fig, ax = plt.subplots(figsize=(7.0, 3.4), layout="constrained")
for condition, evoked in evokeds.items():
    ax.plot(evoked.times * 1e3, evoked.data.mean(axis=0) * 1e6, label=condition)
ax.axvspan(window.tmin * 1e3, window.tmax * 1e3, color="0.9")
ax.axhline(0.0, color="0.6", linewidth=0.8)
ax.set(xlabel="Time (ms)", ylabel="Mean parietal voltage (µV)", title="Simulated ERPs")
ax.legend()
plt.show()

# %%
# Extract mean amplitude and positive peak latency
# ------------------------------------------------
# Time windows include both endpoint samples. The ROI averages channel-level
# features: for mean amplitude this equals averaging voltages before extraction.
# For peak latency it averages the three channel peak times, which can differ
# from the peak time of an averaged ROI waveform. ``polarity="positive"`` locates
# the largest voltage in the window; it does not require an interior local peak.
#
# Tables retain volts and seconds. Check the formulas independently, then verify
# that trial-averaged mean amplitude equals the corresponding MNE evoked value.

signal = ef.Signal.from_epochs(epochs, recording="simulated-erp")
mean_amplitude = ef.mean_amplitude([signal], windows=[window], groups=groups, include_global=False)
peak_latency = ef.peak_latency(
    [signal], windows=[window], groups=groups, polarity="positive", include_global=False
)
mask = (signal.times >= window.tmin) & (signal.times <= window.tmax)
window_data = signal.data[:, :, mask]
np.testing.assert_allclose(mean_amplitude.values[:, 0], window_data.mean(axis=(1, 2)))
channel_peak_times = signal.times[mask][np.argmax(window_data, axis=-1)]
np.testing.assert_allclose(peak_latency.values[:, 0], channel_peak_times.mean(axis=1))
for condition, evoked in evokeds.items():
    trial_mask = epochs.events[:, 2] == epochs.event_id[condition]
    np.testing.assert_allclose(
        mean_amplitude.values[trial_mask, 0].mean(), evoked.data[:, mask].mean()
    )

# %%
# Join retained trials to their original metadata
# -----------------------------------------------
# Join on original epoch identity rather than pairing post-rejection values with
# the first 58 rows of the original metadata. The latter silently assigns some
# features to the wrong trials. ``validate="one_to_one"`` makes duplicates an
# error. The event label already lives in the feature table's row identity.

table = ef.concat([mean_amplitude, peak_latency])
wide = table.to_dataframe().join(epochs.metadata, on="epoch", validate="one_to_one")
np.testing.assert_array_equal(wide.index.get_level_values("epoch"), wide["trial_number"])
summary = pd.DataFrame(
    {
        "Mean amplitude (µV)": wide[mean_amplitude.names[0]] * 1e6,
        "Peak latency (ms)": wide[peak_latency.names[0]] * 1e3,
    }
)
print(summary.groupby(level="event").mean().round(3))
print(table.to_long()[["epoch", "event", "measure", "unit", "value"]].head())

retained_conditions = conditions[epochs.selection]
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), layout="constrained")
axes[0].boxplot(
    [mean_amplitude.values[retained_conditions == label, 0] * 1e6 for label in evokeds],
    tick_labels=list(evokeds),
)
axes[0].set(ylabel="Mean amplitude (µV)", title="One value per retained trial")
axes[1].scatter(wide["planted_latency"] * 1e3, peak_latency.values[:, 0] * 1e3, alpha=0.6)
axes[1].plot([220, 420], [220, 420], color="0.5", linestyle="--")
axes[1].set(xlabel="Planted latency (ms)", ylabel="Measured ROI peak latency (ms)")
plt.show()

# %%
# Interpret the measurements
# ---------------------------
# Mean amplitude describes a fixed portion of the waveform. Positive peak
# latency also depends on noise, sampling rate, window bounds and the choice of
# extremum. The single-trial estimates need not equal the planted component
# parameters, and peak latency of an evoked average is a different estimand from
# mean single-trial peak latency. Inspect waveforms and boundary peaks before
# drawing component-level conclusions.
#
# These trials come from one simulated recording. Their distributions describe
# the simulation; population inference needs participant-level replication and
# a model of repeated observations. Continue with :doc:`plot_session_reliability`
# to distinguish stable ordering from agreement across sessions, or
# :doc:`/guides/preprocessing` for real-data preparation.
