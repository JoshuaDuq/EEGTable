"""
Fitting Microstate Templates and Segmenting New Epochs
======================================================

Fit spatial templates on explicit training epochs, freeze them, and summarize
the labels assigned to separate epochs. Known sensor maps make reference
matching, polarity invariance, and temporal measures directly checkable.

This offline simulation demonstrates an estimator, not evidence for biological
microstates. Install ``python -m pip install -e ".[docs,microstates]"``.
"""

# %%
# Plant identifiable sensor maps
# ------------------------------
# These four maps are functions of standard montage coordinates, not cortical
# sources. Demeaning gives an average reference; unit Euclidean norms make map
# similarity a spatial correlation. Their names identify this simulation only.
# An unmatched fit has arbitrary ``state1`` onward labels. Canonical A-D labels
# or brain network interpretations require an independently justified reference.

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.signal import find_peaks

import eegtable as ef

rng = np.random.default_rng(43)
sampling_rate = 200.0  # Hz: one sample spans 5 ms.
channel_names = (
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "C3",
    "Cz",
    "C4",
    "P3",
    "Pz",
    "P4",
    "O1",
    "Oz",
    "O2",
)
info = mne.create_info(channel_names, sampling_rate, "eeg")
info.set_montage(mne.channels.make_standard_montage("colin27_1020"))
positions = np.array([channel["loc"][:3] for channel in info["chs"]])
horizontal, anterior = positions[:, :2].T
planted_maps = np.stack([horizontal, anterior, horizontal * anterior, horizontal**2 - anterior**2])
planted_maps -= planted_maps.mean(axis=1, keepdims=True)
planted_maps /= np.linalg.norm(planted_maps, axis=1, keepdims=True)
state_names = tuple(f"planted-{index + 1}" for index in range(4))
reference = ef.MicrostateModel.from_templates(
    planted_maps,
    ch_names=channel_names,
    labels=state_names,
    reference_name="coordinate-maps-seed-43",
)

# %%
# Simulate visits and global field power peaks
# --------------------------------------------
# Each epoch has 32 visits lasting 90-130 ms. Their positive amplitude envelopes
# generate GFP peaks; random signs deliberately invert the maps across visits.
# The coefficient is 12 microvolts on a unit map, with independent 0.15 microvolt
# sensor noise. Arrays are in volts, times in seconds, and sampling rate in Hz.
# No temporal filter is applied to these synthetic map switches.
#
# `Pycrostates' GFP tutorial
# <https://pycrostates.readthedocs.io/en/stable/generated/auto_tutorials/preprocessing/00_extract_gfp_peaks.html>`_
# motivates peak sampling. Here GFP is the population standard deviation across
# sensors at each sample, and peak prominence is specified in volts.

visit_lengths = np.tile([18, 22, 24, 26], 8)
visit_order = np.tile([0, 1, 2, 3, 0, 2, 1, 3], 4)
n_epochs = 12
n_times = int(visit_lengths.sum())
times = np.arange(n_times) / sampling_rate
data = np.empty((n_epochs, len(channel_names), n_times))
truth = np.empty((n_epochs, n_times), dtype=int)
for epoch in range(n_epochs):
    start = 0
    for visit, length in enumerate(visit_lengths):
        state = (visit_order[visit] + epoch) % len(state_names)
        stop = start + length
        envelope = 1.0 + 0.5 * np.sin(np.linspace(0.0, np.pi, length))
        polarity = rng.choice([-1.0, 1.0])
        data[epoch, :, start:stop] = 12e-6 * polarity * planted_maps[state, :, None] * envelope
        truth[epoch, start:stop] = state
        start = stop
data += rng.normal(0.0, 0.15e-6, data.shape)
data -= data.mean(axis=1, keepdims=True)
row_ids = tuple(("sim-maps", epoch, "map-sequence") for epoch in range(n_epochs))
signal = ef.Signal.from_arrays(
    data=data,
    times=times,
    ch_names=channel_names,
    sfreq=sampling_rate,
    row_ids=row_ids,
    computation=ef.ComputationSpec.create("sensor-map-simulation", seed=43, unit="V"),
)
gfp = signal.data[0].std(axis=0)
peaks, _ = find_peaks(gfp, distance=2, prominence=0.35e-6)
assert set(truth[0, peaks]) == set(range(4))
print(f"Epoch 0: {len(peaks)} qualifying GFP peaks across all four planted maps")

# %%
# Fit only training epochs, then freeze and match
# -----------------------------------------------
# Eight training epochs contribute peaks, initialization, and template updates.
# Four held-out epochs contribute none of these. Matching uses the maps defined
# before any epoch was generated, never the held-out data. The assignment is
# one-to-one by absolute correlation; an inverted map has the same identity.
# Template updates preserve peak-map amplitude, giving maps weight proportional
# to GFP² rather than equal weight after unit normalization. The final reported
# templates are unit vectors; that convention does not remove fitting weights.
# This split tests recovery under the same simulation generator. It is not
# independent participant or population validation.

training_rows = np.arange(8)
held_out_rows = np.arange(8, n_epochs)
fitted = ef.MicrostateModel.fit(
    signal,
    rows=training_rows,
    n_states=4,
    random_state=43,
    min_peak_distance_ms=10.0,
    max_peaks_per_epoch=40,
    peak_prominence=0.35e-6,
)
assert fitted.computation.parameters["fit_rows"] == [list(row_ids[i]) for i in training_rows]
model = fitted.match_reference(reference)
assert model.labels == state_names
assert not model.templates.flags.writeable
similarity = np.abs(np.sum(model.templates * reference.templates, axis=1))
assert np.all(similarity > 0.99)
held_out = ef.Signal.from_arrays(
    data=signal.data[held_out_rows],
    times=times,
    ch_names=channel_names,
    sfreq=sampling_rate,
    row_ids=tuple(row_ids[i] for i in held_out_rows),
    computation=ef.ComputationSpec.create(
        "simulation-held-out", rows=held_out_rows.tolist(), input=signal.computation.record()
    ),
)
assert set(row_ids[i] for i in training_rows).isdisjoint(held_out.row_ids)
segmentation = model.segment(held_out, min_duration_ms=20.0)
np.testing.assert_array_equal(segmentation.states, truth[held_out_rows])
assert segmentation.row_ids == held_out.row_ids
print(f"Matched training-template correlations: {similarity.round(4)}")
print(f"Held-out GFP²-weighted explained variance: {segmentation.global_explained_variance:.3f}")

fig, axes = plt.subplots(2, 4, figsize=(9.0, 4.4), layout="constrained")
for index, name in enumerate(state_names):
    # Template sign is arbitrary: align it only for this comparison plot.
    orientation = np.sign(model.templates[index] @ reference.templates[index])
    for row, topography in enumerate(
        [reference.templates[index], orientation * model.templates[index]]
    ):
        mne.viz.plot_topomap(
            topography,
            info,
            axes=axes[row, index],
            show=False,
            names=channel_names,
            cmap="RdBu_r",
            vlim=(-0.6, 0.6),
        )
        axes[row, index].set_title(f"{'Planted' if row == 0 else 'Training fit'}: {name}")
fig.suptitle("Unit sensor maps; color is dimensionless and polarity is arbitrary")
plt.show()

# %%
# Check frozen assignments and their polarity invariance
# ------------------------------------------------------
# Independently calculate absolute spatial correlations. This low-noise example
# has no short misassignments, so the 20 ms smoothing leaves those labels intact.
# In real data smoothing changes durations and transitions and must be reported.
# Inverting every voltage sample must give exactly the same state sequence.
# Global explained variance is the mean squared correlation with each assigned
# template, weighted by GFP² over all held-out samples. Verify that formula as
# well as the labels; a high value alone does not validate a biological model.

centered_maps = held_out.data - held_out.data.mean(axis=1, keepdims=True)
unit_maps = centered_maps / np.linalg.norm(centered_maps, axis=1, keepdims=True)
correlations = np.abs(np.einsum("kc,ect->ekt", model.templates, unit_maps))
np.testing.assert_array_equal(segmentation.states, correlations.argmax(axis=1))
assigned_correlations = np.take_along_axis(correlations, segmentation.states[:, None, :], axis=1)[
    :, 0
]
gfp_squared = held_out.data.var(axis=1, ddof=0)
expected_gev = (gfp_squared * assigned_correlations**2).sum() / gfp_squared.sum()
np.testing.assert_allclose(segmentation.global_explained_variance, expected_gev)
inverted = ef.Signal.from_arrays(
    data=-held_out.data,
    times=times,
    ch_names=channel_names,
    sfreq=sampling_rate,
    row_ids=held_out.row_ids,
    computation=ef.ComputationSpec.create(
        "polarity-inversion", input=held_out.computation.record()
    ),
)
np.testing.assert_array_equal(
    model.segment(inverted, min_duration_ms=20.0).states, segmentation.states
)

palette = ListedColormap(["#4477AA", "#EE6677", "#228833", "#CCBB44"])
fig, axes = plt.subplots(2, 1, figsize=(9.0, 3.9), layout="constrained")
axes[0].plot(times, gfp * 1e6, color="0.25")
axes[0].scatter(times[peaks], gfp[peaks] * 1e6, color="#AA3377", s=16)
axes[0].set(ylabel="GFP (µV)", title="Training epoch 0: peak maps used for fitting")
axes[1].imshow(
    np.stack([truth[held_out_rows[0]], segmentation.states[0]]),
    aspect="auto",
    extent=(times[0], times[-1] + 1.0 / sampling_rate, 1.5, -0.5),
    cmap=palette,
    norm=BoundaryNorm(np.arange(5) - 0.5, palette.N),
    interpolation="nearest",
)
axes[1].set(yticks=[0, 1], yticklabels=["Planted", "Frozen fit"], xlabel="Time (s)")
axes[1].set_title("Held-out epoch 8: color indexes planted-1 through planted-4")
plt.show()

# %%
# Summarize visits within each epoch
# ----------------------------------
# Native measures keep epoch boundaries separate. Coverage counts samples;
# duration averages contiguous visit lengths in milliseconds. A visit cut by
# an epoch or analysis window is counted at its observed length. Transitions
# count successive visits, omit self transitions, and normalize per source.
# The `Pycrostates segmentation tutorial
# <https://pycrostates.readthedocs.io/en/stable/generated/auto_tutorials/segmentation/00_segmentation.html>`_
# discusses these distinct temporal summaries; its implementation conventions
# need not be identical to the native definitions checked here.

window = ef.Window("full-epoch", float(times[0]), float(times[-1]))
coverage = ef.microstate_coverage(segmentation, windows=[window])
duration = ef.microstate_duration(segmentation, windows=[window])
transitions = ef.microstate_transitions(segmentation, windows=[window])
assert coverage.row_ids == duration.row_ids == transitions.row_ids == held_out.row_ids
np.testing.assert_allclose(coverage.values.sum(axis=1), 1.0)
manual_duration = np.empty_like(duration.values)
manual_transitions = np.empty((held_out.n_epochs, 4, 4))
for epoch, states in enumerate(segmentation.states):
    boundaries = np.r_[0, np.flatnonzero(np.diff(states)) + 1, states.size]
    visits = states[boundaries[:-1]]
    lengths = np.diff(boundaries)
    for state in range(4):
        manual_duration[epoch, state] = lengths[visits == state].mean() * 1000 / sampling_rate
        np.testing.assert_allclose(coverage.values[epoch, state], np.mean(states == state))
    counts = np.zeros((4, 4))
    np.add.at(counts, (visits[:-1], visits[1:]), 1)
    assert np.all(counts.sum(axis=1) > 0)
    manual_transitions[epoch] = counts / counts.sum(axis=1, keepdims=True)
np.testing.assert_allclose(duration.values, manual_duration)
for column, meta in enumerate(transitions.meta):
    source, target = (state_names.index(name) for name in meta.space.split("-to-"))
    np.testing.assert_allclose(transitions.values[:, column], manual_transitions[:, source, target])
np.testing.assert_allclose(manual_transitions.sum(axis=2), 1.0)
print(f"Mean held-out coverage: {coverage.values.mean(axis=0).round(3)}")
print(f"Mean observed visit duration (ms): {duration.values.mean(axis=0).round(1)}")

fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.1), layout="constrained")
axes[0].bar(state_names, coverage.values.mean(axis=0), color=palette.colors)
axes[0].set(ylabel="Fraction of samples", ylim=(0, 0.4), title="Mean epoch coverage")
axes[1].bar(state_names, duration.values.mean(axis=0), color=palette.colors)
axes[1].set(ylabel="Mean visit length (ms)", title="Mean epoch duration")
image = axes[2].imshow(manual_transitions.mean(axis=0), vmin=0, vmax=1, cmap="viridis")
axes[2].set(
    xticks=range(4),
    yticks=range(4),
    xticklabels=state_names,
    yticklabels=state_names,
    xlabel="To",
    ylabel="From",
    title="Mean epoch transitions",
)
for axis in axes:
    axis.tick_params(axis="x", rotation=45)
fig.colorbar(image, ax=axes[2], label="Conditional probability")
plt.show()

# %%
# Interpret the fitted representation
# -----------------------------------
# Four states were prespecified because the generator contains four maps. Neither
# high explained variance nor accurate recovery establishes a real EEG state
# count. Real analyses must justify filtering, bad-channel handling, reference,
# rank, peak selection, noise/artifact rejection, and duration smoothing. An
# average reference reduces sensor rank by one; it cannot recover missing spatial
# information. Filtering can smear switching boundaries and change GFP peaks.
#
# The model stores channel order, template values, fit-row identities, seed, peak
# settings, and reference matching in ``computation``. Preserve those identities
# together with preprocessing provenance. Native coverage is compositional: its
# four columns sum to one. Durations are undefined if a state is absent, and a
# transition row is undefined if its source is never left; this simulation avoids
# those cases and does not replace missing values. For prediction in a cohort,
# refit templates within each training split, use an independent label reference,
# and keep held-out participants out of template and parameter selection.
