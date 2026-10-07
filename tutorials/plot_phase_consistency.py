"""
Phase Consistency Across Trials: ITPC and PPC
=============================================

Two conditions can have the same oscillation amplitude while differing in
phase alignment across trials. Simulate that distinction and compare
inter-trial phase coherence (ITPC) with pairwise phase consistency (PPC).

These are cross-trial estimates: each output row represents a trial group,
not a single epoch. The simulation also shows why a positive ITPC under random
phase does not by itself establish phase locking.
"""

# %%
# Keep amplitude fixed and vary phase
# -----------------------------------
# Forty trials share a 10 Hz oscillation with phase zero; another forty have
# independent uniform phases. Both have 5 µV amplitude and the same independent
# noise distribution. The fixed random seed makes the demonstration reproducible.
# Epochs span -2 to 2 s, leaving a generous margin around the analysis window.

import matplotlib.pyplot as plt
import mne
import numpy as np

import eegtable as ef

rng = np.random.default_rng(23)
sampling_rate = 200.0
times = np.arange(-400, 401) / sampling_rate
trials_per_condition = 40
labels = np.repeat(["aligned", "random"], trials_per_condition)
phases = np.concatenate(
    [np.zeros(trials_per_condition), rng.uniform(-np.pi, np.pi, trials_per_condition)]
)
data = 5e-6 * np.sin(2.0 * np.pi * 10.0 * times + phases[:, None])
data += rng.normal(0.0, 0.5e-6, data.shape)
events = np.column_stack(
    [
        np.arange(len(labels)) * len(times),
        np.zeros(len(labels), int),
        np.repeat([1, 2], trials_per_condition),
    ]
)
epochs = mne.EpochsArray(
    data[:, None, :],
    mne.create_info(["Cz"], sampling_rate, "eeg"),
    events=events,
    event_id={"aligned": 1, "random": 2},
    tmin=-2.0,
    baseline=None,
    verbose=False,
)
signal = ef.BandSignal.from_epochs(epochs, ef.Band("alpha", 8.0, 13.0), recording="simulated-phase")
window = ef.Window("interior", -0.5, 0.5)
mask = (signal.times >= window.tmin) & (signal.times <= window.tmax)

# %%
# Visualize phase vectors at the epoch origin
# -------------------------------------------
# Each arrow represents one unit-length phase vector at time zero. Amplitude is
# removed by taking ``exp(1j * phase)``. ITPC first averages these vectors across
# trials at every latency, takes their magnitude, and then averages over time.
# Averaging each trial across time first would measure a different quantity.

origin = np.argmin(np.abs(signal.times))
fig, axes = plt.subplots(
    1, 2, figsize=(6.5, 3.2), subplot_kw={"projection": "polar"}, layout="constrained"
)
for ax, condition in zip(axes, ["aligned", "random"], strict=True):
    angles = signal.phase[labels == condition, 0, origin]
    for angle in angles:
        ax.plot([angle, angle], [0.0, 1.0], color="tab:blue", alpha=0.25)
    mean_vector = np.exp(1j * angles).mean()
    ax.plot([np.angle(mean_vector)] * 2, [0.0, abs(mean_vector)], color="tab:red", linewidth=3)
    ax.set(title=condition, ylim=(0.0, 1.0), yticks=[])
plt.show()

# %%
# Extract one row per condition and verify the formulas
# -----------------------------------------------------
# ITPC lies between zero and one. PPC estimates squared population phase locking
# without ITPC-squared's finite-sample bias under independent identically
# distributed trial phases. A finite-sample PPC can be negative, and neither
# quantity eliminates uncertainty due to having only a finite number of trials.
# This identity for complete data checks the library against NumPy directly.

coherence = ef.itpc([signal], windows=[window], trials=labels, include_global=False)
consistency = ef.ppc([signal], windows=[window], trials=labels, include_global=False)
table = ef.concat([coherence, consistency])
print(table.to_long()[["group", "measure", "space", "value", "coverage"]])

for row, condition in enumerate(coherence.row_labels):
    vectors = np.exp(1j * signal.phase[labels == condition, 0][:, mask])
    locking = np.abs(vectors.mean(axis=0))
    count = len(vectors)
    direct_ppc = (count * locking**2 - 1.0) / (count - 1.0)
    np.testing.assert_allclose(coherence.values[row, 0], locking.mean())
    np.testing.assert_allclose(consistency.values[row, 0], direct_ppc.mean())
assert coherence.values[0, 0] > 0.95
assert coherence.values[1, 0] < 0.3

# %%
# Inspect the effect of trial count
# ---------------------------------
# Keep the first 5, 10, 20, or 40 trials from each condition. These are nested
# subsets of one simulation, not independent replicates or a confidence interval.
# Random-phase ITPC is usually positive; PPC fluctuates around zero rather than
# sharing that positive bias. Real comparisons need balanced trial counts or
# an inference procedure that accounts for trial counts and dependence.

trial_counts = [5, 10, 20, 40]
estimates = {"itpc": [], "ppc": []}
for count in trial_counts:
    rows = np.concatenate([np.arange(count), trials_per_condition + np.arange(count)])
    subset = ef.BandSignal.from_epochs(
        epochs[rows], ef.Band("alpha", 8.0, 13.0), recording="simulated-phase"
    )
    for measure, function in [("itpc", ef.itpc), ("ppc", ef.ppc)]:
        result = function([subset], windows=[window], trials=labels[rows], include_global=False)
        estimates[measure].append(result.values[:, 0])

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True, layout="constrained")
for ax, measure in zip(axes, ["itpc", "ppc"], strict=True):
    for index, condition in enumerate(["aligned", "random"]):
        ax.plot(trial_counts, np.asarray(estimates[measure])[:, index], "o-", label=condition)
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set(title=measure.upper(), xlabel="Trials per condition", xticks=trial_counts)
axes[0].set_ylabel("Phase consistency (a.u.)")
axes[1].legend()
plt.show()

# %%
# Preserve the sample unit
# ------------------------
# ``table.row_labels`` contains the two condition groups, and ``row_ids`` is
# absent. Do not repeat a group estimate on forty epoch rows and analyze it as
# forty independent observations. For a cohort, compute comparable groups within
# each participant and use the trial-group workflow in :doc:`/guides/cohorts`.
# ITPC measures event-relative consistency, not connectivity between sensors.
#
# See :doc:`/methods/connectivity` for definitions and Vinck et al. (2010),
# `The pairwise phase consistency: a bias-free measure of rhythmic neuronal
# synchronization <https://doi.org/10.1016/j.neuroimage.2010.01.073>`_.
