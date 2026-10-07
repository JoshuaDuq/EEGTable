"""
Sensor Connectivity: Shared Signals and Phase Lags
==================================================

Can a large sensor-connectivity value distinguish a shared instantaneous signal
from a stable phase offset? Compare phase-locking value (PLV) and weighted phase
lag index (wPLI) on three explicitly constructed scenarios, then check EEGTable's
results against MNE-Connectivity with identical spectral settings.

This seeded simulation works offline. Install
``python -m pip install -e ".[docs,connectivity]"`` to run it. Sensor correlations
in this example illustrate estimators, not anatomical or causal connections.
"""

# %%
# Construct random-phase, shared and phase-lagged signals
# -------------------------------------------------------
# Each scenario has 80 four-second trials on C3 and C4 at 128 Hz. Within each
# trial both channels oscillate at the same randomly selected 9–11 Hz frequency.
# Across trials, the phase difference is random, zero, or fixed at 60 degrees.
# The random-phase control has no consistent phase offset; its shared carrier
# frequency means the two complete signals are not statistically independent.
# Independent sensor noise prevents the zero-lag case from being numerically
# degenerate. The shared case represents one possible instantaneous mixing
# mechanism; no neural interaction is needed to construct it.
#
# Signals are already at an assumed common reference. Re-referencing these two
# channels to their own average would force them to be exact negatives and
# radically change the example. Select a scientifically justified reference on
# the full montage when working with recordings.

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne_connectivity import spectral_connectivity_epochs

import eegtable as ef

rng = np.random.default_rng(43)
sampling_rate = 128.0
times = np.arange(512) / sampling_rate
trials_per_scenario = 80
scenario_names = ["random-phase", "shared", "phase-lagged"]
labels = np.repeat(scenario_names, trials_per_scenario)
frequencies = rng.uniform(9.0, 11.0, (len(labels), 1))
phases = rng.uniform(-np.pi, np.pi, (len(labels), 1))
phase_differences = np.concatenate(
    [
        rng.uniform(-np.pi, np.pi, trials_per_scenario),
        np.zeros(trials_per_scenario),
        np.full(trials_per_scenario, np.pi / 3.0),
    ]
)
carrier_phase = 2.0 * np.pi * frequencies * times + phases
data = 5e-6 * np.stack(
    [np.sin(carrier_phase), np.sin(carrier_phase + phase_differences[:, None])], axis=1
)
data += rng.normal(0.0, 0.3e-6, data.shape)
events = np.column_stack(
    [
        np.arange(len(labels)) * len(times),
        np.zeros(len(labels), int),
        np.repeat([1, 2, 3], trials_per_scenario),
    ]
)
epochs = mne.EpochsArray(
    data,
    mne.create_info(["C3", "C4"], sampling_rate, "eeg"),
    events=events,
    event_id=dict(zip(scenario_names, [1, 2, 3], strict=True)),
    baseline=None,
    verbose=False,
)
signal = ef.Signal.from_epochs(epochs, recording="simulated-connectivity")

fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.4), sharex=True, layout="constrained")
for index, (ax, scenario) in enumerate(zip(axes, scenario_names, strict=True)):
    trial = index * trials_per_scenario
    for channel, name in enumerate(signal.ch_names):
        ax.plot(times, signal.data[trial, channel] * 1e6, label=name)
    ax.set(title=scenario, ylabel="Voltage (µV)", xlim=(0.0, 0.5))
axes[0].legend(loc="upper right", ncols=2)
axes[-1].set_xlabel("Time (s)")
plt.show()

# %%
# Estimate across trials with explicit frequency boundaries
# ---------------------------------------------------------
# Fourier estimation uses a Hann taper on each complete four-second epoch,
# giving a 0.25 Hz grid. No bandpass/Hilbert step precedes spectral connectivity.
# The selected band is [8.5, 11.5) Hz: EEGTable excludes the upper edge, averages
# connectivity values over retained bins, and returns one row per trial group.
# This is a mean of per-frequency estimates, not one estimate of pooled spectra.
# Group labels are sorted in the output; use their identities when matching
# scenarios rather than assuming the order of first appearance in the epochs.
#
# PLV measures consistency of phase differences, including differences near
# zero. wPLI emphasizes the sign-consistent imaginary cross-spectrum. See
# MNE-Connectivity's `estimator definitions
# <https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html>`_
# and its `comparison of coherency methods
# <https://mne.tools/mne-connectivity/stable/auto_examples/compare_coherency_methods.html>`_.

band = ef.Band("alpha_demo", 8.5, 11.5)
window = ef.Window("full_epoch", 0.0, times[-1])
tables = {}
for method in ["plv", "wpli"]:
    tables[method] = ef.spectral_connectivity(
        signal,
        method=method,
        bands=[band],
        windows=[window],
        trials=labels,
        mode="fourier",
    )
table = ef.concat(list(tables.values()))
assert table.row_ids is None
assert table.row_labels == tuple(sorted(scenario_names))
print(table.to_long()[["group", "measure", "space", "unit", "value", "coverage"]])

# %%
# Verify the wrapper against MNE-Connectivity
# -------------------------------------------
# MNE-Connectivity includes both requested frequency boundaries. Request its
# unreduced spectrum, explicitly select the same half-open band, and compare the
# arithmetic mean with EEGTable. Specify just the channel pair to make its axis
# unambiguous. PLV and wPLI magnitudes do not depend on which channel comes first.
# This checks delegation and band reduction; it does not independently validate
# the underlying estimator's statistical assumptions.

for row, scenario in enumerate(table.row_labels):
    reference_plv, reference_wpli = spectral_connectivity_epochs(
        signal.data[labels == scenario],
        method=["plv", "wpli"],
        indices=([0], [1]),
        sfreq=sampling_rate,
        mode="fourier",
        fmin=band.fmin,
        fmax=band.fmax,
        faverage=False,
        verbose=False,
    )
    for method, reference in zip(["plv", "wpli"], [reference_plv, reference_wpli], strict=True):
        reference_frequencies = np.asarray(reference.freqs)
        selected = (reference_frequencies >= band.fmin) & (reference_frequencies < band.fmax)
        expected = reference.get_data()[0, selected].mean()
        np.testing.assert_allclose(tables[method].values[row, 0], expected, atol=1e-12)

# %%
# Compare the scenarios without treating estimates as trials
# ----------------------------------------------------------
# Shared and phase-lagged signals have higher PLV than the random-phase control.
# PLV weights each frequency bin equally, including low-power bins where phase
# is noisy; the band mean need not approach one even with a fixed signal offset.
# A small wPLI for the shared case reflects weak sign-consistent imaginary
# cross-spectra; it does not establish absence of a biological interaction.
# Finite-sample bias can yield nonzero values for random-phase or zero-lag signals.
# The three output rows are three pooled estimates, not 240 independent samples.

fig, ax = plt.subplots(figsize=(6.5, 3.5), layout="constrained")
positions = np.arange(len(scenario_names))
for offset, method in zip([-0.18, 0.18], ["plv", "wpli"], strict=True):
    scenario_values = tables[method].to_dataframe().loc[scenario_names].iloc[:, 0]
    ax.bar(positions + offset, scenario_values, width=0.36, label=method.upper())
ax.set(
    xticks=positions,
    xticklabels=scenario_names,
    ylim=(0.0, 1.05),
    ylabel="Band-mean connectivity",
    title=f"{trials_per_scenario} trials per pooled estimate",
)
ax.legend()
plt.show()

# %%
# Interpret sensor connectivity cautiously
# ----------------------------------------
# Neither estimator identifies a direct connection, anatomical source, direction
# or cause. Shared inputs, reference choice, spectral signal-to-noise ratio and
# trial dependence all affect the estimate. Rejecting zero-lag contributions also
# discards genuine zero-lag interactions. wPLI is not a universal leakage cure.
#
# Keep trial counts, duration, frequency band and estimator settings comparable
# across recordings. Two trials satisfy the API's computational minimum but
# provide little information about stability. A population analysis needs one
# justified estimate per participant/condition and inference at that level;
# avoid broadcasting a group estimate into its contributing epoch rows.
# :doc:`/guides/cohorts` explains group-table export, and
# :doc:`/methods/connectivity` defines ROI reduction and other connectivity scales.
