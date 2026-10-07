"""
Cycle Shape Beyond Alpha Power
==============================

Two 10 Hz waveforms can share a period and peak-to-trough amplitude while having
different rise and decay durations. Measure this distinction on smooth broadband
signals, inspect ByCycle's extrema, and verify EEGTable's windowed summaries
against the individual-cycle reference table.

This deterministic, offline simulation requires ``eegtable[cycles]``. The
dependency is imported directly so missing or incompatible installations raise
their normal errors.

From the repository root, install plotting and cycle-analysis dependencies with::

    python -m pip install -e ".[docs,cycles]"
"""

# %%
# Generate smooth symmetric and asymmetric cycles
# -----------------------------------------------
# Six trials each contain one channel and eight seconds sampled at 1000 Hz.
# A sinusoid is symmetric. Adding a phase-locked 20 Hz harmonic changes the shape
# of its 10 Hz cycle. Both components lie well below the 500 Hz Nyquist frequency;
# no corners or high-frequency noise create artificial extrema. Each trace is
# scaled to a sampled peak-to-trough range of 20 µV. Equal range and period do not
# imply equal spectral power.
#
# The three explicit starting phases in each condition change cycle alignment
# with the window. They are deterministic repetitions, not independent subjects.
# Values passed to ``Signal`` are in volts; the time axis is in seconds.

import matplotlib.pyplot as plt
import numpy as np
from bycycle.features import compute_features

import eegtable as ef

sampling_rate = 1000.0
times = np.arange(8000) / sampling_rate
phases = np.array([0.2, 0.8, 1.4])
conditions = ["symmetric", "asymmetric"]
labels = np.repeat(conditions, len(phases))
angle = 2.0 * np.pi * 10.0 * times + phases[:, None]
symmetric = np.cos(angle)
asymmetric = symmetric + 0.3 * np.cos(2.0 * angle + np.pi / 3.0)
data = np.concatenate([symmetric, asymmetric])
data *= 20e-6 / np.ptp(data, axis=-1, keepdims=True)
signal = ef.Signal.from_arrays(
    data=data[:, None, :],
    times=times,
    ch_names=("C3",),
    sfreq=sampling_rate,
    row_ids=tuple(("simulated-waveforms", row, str(label)) for row, label in enumerate(labels)),
    computation=ef.ComputationSpec.create(
        "synthetic-harmonic-waveforms",
        fundamental_hz=10.0,
        harmonic_fraction=0.3,
        harmonic_phase_rad=np.pi / 3.0,
        starting_phases_rad=phases.tolist(),
        peak_to_trough_volts=20e-6,
    ),
)
alpha = ef.Band("alpha", 8.0, 13.0)
window = ef.Window("interior", 1.03, 6.97)
tiny_window = ef.Window("no_complete_cycle", 2.0, 2.005)
thresholds = {
    "amp_fraction_threshold": 0.0,
    "amp_consistency_threshold": 0.5,
    "period_consistency_threshold": 0.5,
    "monotonicity_threshold": 0.8,
    "min_n_cycles": 3,
}

# %%
# Locate alpha cycles and measure the broadband waveform
# ------------------------------------------------------
# Pass the original broadband ``Signal`` to ``cycle_features``. ByCycle's 8–13 Hz
# narrowband filter locates intervals in which to search for extrema; the extrema
# voltages and waveform durations come from the broadband input. Filtering away
# the 20 Hz harmonic before this step would change the shape being measured.
#
# EEGTable fixes ``center_extrema='peak'``: each cycle runs from its preceding
# trough, through a positive peak, to its next trough. It uses the ``cycles``
# consistency detector and a locator of three cycles at the lower band edge
# (3/8 s here). Burst labels are assigned on the whole epoch, before selecting
# complete cycles inside the prespecified interior window.
#
# Thresholds require consistent neighboring amplitudes and periods, mostly
# monotonic flanks, and at least three consecutive qualifying cycles. The zero
# amplitude-fraction threshold supplies no positive amplitude cutoff. These
# settings suit this smooth demonstration; choose them from measurement goals
# and data quality before studying experimental outcomes.

all_windows = ef.cycle_features(
    [signal],
    windows=[window, tiny_window],
    bands=[alpha],
    burst_thresholds=thresholds,
    include_global=False,
)
table = all_windows.select(window=window.name)
assert table.row_ids == signal.row_ids
assert all(meta.band == alpha for meta in table.meta)
assert np.isfinite(table.values).all()
assert not table.flags["cycle_no_complete_cycles"].any()
assert not table.flags["cycle_no_burst"].any()
parameters = table.meta[0].computation.parameters["parameters"]
assert parameters["center_extrema"] == "peak"
assert parameters["burst_method"] == "cycles"
assert parameters["locator_filter_cycles"] == 3
print(
    table.to_long()[["epoch", "event", "measure", "unit", "value"]]
    .pivot(index=["epoch", "event"], columns="measure", values="value")
    .round(6)
)

# %%
# Check complete-cycle selection against ByCycle directly
# -------------------------------------------------------
# Use identical settings on one trace per condition. ByCycle returns duration
# fields in **samples**; EEGTable converts them to seconds. ``volt_amp`` is the
# mean of the rising and falling voltage excursions, in volts here. For a pure
# sinusoid of 10 µV peak amplitude, this is 20 µV, rather than 10 µV or RMS.
#
# A cycle counts only if **both troughs** lie inside the actual sampled window
# bounds. Count all such cycles, but average shape fields only over those marked
# ``is_burst``. ``cycle_burst_count`` counts burst-labelled cycles, not separate
# burst runs; ``cycle_burst_fraction`` is their fraction of complete cycles,
# rather than the fraction of time spent oscillating.
#
# These settings and fields follow the official `ByCycle algorithm tutorial
# <https://bycycle-tools.github.io/bycycle/auto_tutorials/plot_2_bycycle_algorithm.html>`_
# and `compute_features API
# <https://bycycle-tools.github.io/bycycle/generated/bycycle.features.compute_features.html>`_.

inside = (times >= window.tmin) & (times <= window.tmax)
sampled_bounds = (times[inside][0], times[inside][-1])
shape_fields = {
    "cycle_period": ("period", 1.0 / sampling_rate),
    "cycle_rise_time": ("time_rise", 1.0 / sampling_rate),
    "cycle_decay_time": ("time_decay", 1.0 / sampling_rate),
    "cycle_amplitude": ("volt_amp", 1.0),
    "cycle_rise_decay_symmetry": ("time_rdsym", 1.0),
    "cycle_peak_trough_symmetry": ("time_ptsym", 1.0),
}
references = {}
for row in [0, 3]:
    frame = compute_features(
        data[row],
        sampling_rate,
        (alpha.fmin, alpha.fmax),
        center_extrema="peak",
        burst_method="cycles",
        threshold_kwargs=thresholds.copy(),
        find_extrema_kwargs={"filter_kwargs": {"n_cycles": 3}},
        return_samples=True,
    )
    complete = frame[
        (times[frame.sample_last_trough.to_numpy(dtype=int)] >= sampled_bounds[0])
        & (times[frame.sample_next_trough.to_numpy(dtype=int)] <= sampled_bounds[1])
    ]
    bursting = complete[complete.is_burst]
    assert len(bursting) > 0
    references[row] = complete
    for measure, (field, conversion) in shape_fields.items():
        np.testing.assert_allclose(
            table.select(measure=measure).values[row, 0],
            bursting[field].mean() * conversion,
            rtol=1e-12,
            atol=1e-12,
        )
    assert table.select(measure="cycle_count").values[row, 0] == len(complete)
    assert table.select(measure="cycle_burst_count").values[row, 0] == len(bursting)
    np.testing.assert_allclose(
        table.select(measure="cycle_burst_fraction").values[row, 0],
        len(bursting) / len(complete),
    )
    assert not table.flags["cycle_no_complete_cycles"][row].any()
    assert not table.flags["cycle_no_burst"][row].any()

fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True, sharey=True, layout="constrained")
display = (times >= 2.0) & (times <= 2.32)
for ax, row, condition in zip(axes, [0, 3], conditions, strict=True):
    ax.plot(times[display], data[row, display] * 1e6, color="0.25", label="Broadband voltage")
    frame = references[row]
    peaks = frame.sample_peak.to_numpy(dtype=int)
    troughs = np.unique(
        np.concatenate(
            [
                frame.sample_last_trough.to_numpy(dtype=int),
                frame.sample_next_trough.to_numpy(dtype=int),
            ]
        )
    )
    for samples, marker, color, label in [
        (peaks, "o", "tab:red", "Peak"),
        (troughs, "v", "tab:blue", "Trough"),
    ]:
        samples = samples[display[samples]]
        ax.scatter(
            times[samples], data[row, samples] * 1e6, marker=marker, color=color, label=label
        )
    ax.set(title=condition, ylabel="Voltage (µV)")
axes[0].legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("Time (s)")
plt.show()

# %%
# Interpret symmetry with a stated polarity
# -----------------------------------------
# Rise-decay symmetry is ``rise / period``: the rising flank goes from the
# preceding voltage trough to the positive peak. Values below 0.5 describe a
# shorter rise than decay. Peak-trough symmetry is ``peak / (peak + trough)``,
# using the peak duration and the preceding trough duration between flank voltage
# midpoints. Its support can start before the cycle's first trough. It describes
# a different aspect of shape; neither measure is spectral phase.
#
# Reversing channel polarity exchanges positive and negative shapes and changes
# these conventions. Referencing, noise, filtering, and sample resolution also
# affect extrema and symmetry. Here one sample is 1 ms, or 0.01 of a 100 ms cycle.
# The assertions allow one sample of symmetry discretization error for the sine.

periods = table.select(measure="cycle_period").values[:, 0]
amplitudes = table.select(measure="cycle_amplitude").values[:, 0]
rise_symmetry = table.select(measure="cycle_rise_decay_symmetry").values[:, 0]
peak_symmetry = table.select(measure="cycle_peak_trough_symmetry").values[:, 0]
np.testing.assert_allclose(periods, 0.1, atol=1.0 / sampling_rate)
np.testing.assert_allclose(amplitudes, 20e-6, rtol=1e-10)
np.testing.assert_allclose(rise_symmetry[:3], 0.5, atol=0.01)
np.testing.assert_allclose(peak_symmetry[:3], 0.5, atol=0.01)
assert np.all(rise_symmetry[3:] < 0.4)
assert np.all(peak_symmetry[3:] < 0.45)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True, layout="constrained")
for ax, values, title in zip(
    axes,
    [rise_symmetry, peak_symmetry],
    ["Rise / period", "Peak / (peak + trough)"],
    strict=True,
):
    for index, condition in enumerate(conditions):
        offsets = np.linspace(-0.08, 0.08, len(phases))
        ax.scatter(index + offsets, values[labels == condition], label=condition)
    ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1.0)
    ax.set(title=title, xticks=[0, 1], xticklabels=conditions, ylim=(0.3, 0.55))
axes[0].set_ylabel("Symmetry fraction")
plt.show()

# %%
# Keep cycle selection, missingness, and sample identity explicit
# ---------------------------------------------------------------
# The tiny second window contains no complete trough-to-trough cycle. Counts
# are zero, shape summaries and burst fraction are undefined, and both flags
# are true. If complete cycles exist but none meet the burst rule, only burst
# count and fraction are zero; shape remains NaN and ``cycle_no_burst`` is true.
# Finite, nonconstant broadband samples throughout each epoch are required.

empty = all_windows.select(window=tiny_window.name)
np.testing.assert_array_equal(empty.select(measure="cycle_count").values, 0.0)
np.testing.assert_array_equal(empty.select(measure="cycle_burst_count").values, 0.0)
assert np.isnan(empty.select(measure="cycle_amplitude").values).all()
assert np.isnan(empty.select(measure="cycle_burst_fraction").values).all()
assert empty.flags["cycle_no_complete_cycles"].all()
assert empty.flags["cycle_no_burst"].all()

# %%
# Each result row remains one trial, with recording, epoch, and condition
# identity. Individual cycles are nested observations, not extra independent
# participants. The detector labels these sustained simulated oscillations as
# burst cycles; that label does not establish a neural event or biological source.
#
# The 20 Hz component is explicitly locked to the 10 Hz fundamental. Its spectral
# peak reflects this waveform's harmonic content; it does not demonstrate an
# independent beta oscillation. Real source separation requires additional
# evidence. See :doc:`/methods/cycles` and the `ByCycle burst detector API
# <https://bycycle-tools.github.io/bycycle/generated/bycycle.burst.detect_bursts_cycles.html>`_
# for detector definitions and threshold interpretation.
