"""
Signal Regularity Across Time Scales
====================================

Sample entropy describes whether matching short patterns remain similar one
sample later. Compare a sinusoid, correlated noise, and white noise at the same
standard deviation, then examine how coarse-graining and the tolerance definition
change that comparison.

This offline simulation demonstrates estimator behavior. Its three trials per
condition are illustrative realizations, not evidence for a clinical biomarker.

From the repository root, install the plotting dependencies with::

    python -m pip install -e ".[docs]"
"""

# %%
# Construct equally scaled signals
# --------------------------------
# Each trial contains 720 samples at 200 Hz: 3.6 s of sampled data, with sample
# times from 0 to 3.595 s. Arrays have dimensions ``(trials, channels, samples)``
# and amplitudes in volts. Each complete trace is centered and scaled to 5 µV
# standard deviation (``ddof=0``); standardization preserves its temporal order.
# The AR(1) process uses coefficient 0.9 and a discarded 200-sample warm-up. It
# supplies temporal correlation, rather than a model of a particular EEG source.

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter
from scipy.spatial.distance import pdist

import eegtable as ef

rng = np.random.default_rng(37)
sampling_rate = 200.0
sample_count = 720
times = np.arange(sample_count) / sampling_rate
trials_per_condition = 3
conditions = ["sine", "AR(1) noise", "white noise"]
labels = np.repeat(conditions, trials_per_condition)
phases = rng.uniform(0.0, 2.0 * np.pi, (trials_per_condition, 1))
sine = np.sin(2.0 * np.pi * 10.0 * times + phases)
innovations = rng.normal(size=(trials_per_condition, sample_count + 200))
correlated = lfilter([1.0], [1.0, -0.9], innovations, axis=-1)[:, 200:]
white = rng.normal(size=(trials_per_condition, sample_count))
data = np.concatenate([sine, correlated, white])
data -= data.mean(axis=-1, keepdims=True)
data *= 5e-6 / data.std(axis=-1, keepdims=True)
signal = ef.Signal.from_arrays(
    data=data[:, None, :],
    times=times,
    ch_names=("Cz",),
    sfreq=sampling_rate,
    row_ids=tuple(("simulated-complexity", row, str(label)) for row, label in enumerate(labels)),
    computation=ef.ComputationSpec.create(
        "synthetic-sine-ar1-white",
        seed=37,
        sine_frequency_hz=10.0,
        ar_coefficient=0.9,
        standard_deviation_volts=5e-6,
    ),
)
window = ef.Window("complete_trace", times[0], times[-1])
order = 2
tolerance_fraction = 0.2
scales = [1, 2, 3, 4, 6]

fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.5), sharex=True, layout="constrained")
display = times <= 0.6
for ax, row, condition in zip(axes, [0, 3, 6], conditions, strict=True):
    ax.plot(times[display], data[row, display] * 1e6, linewidth=1.0)
    ax.set(title=condition, ylabel="Voltage (µV)")
axes[-1].set_xlabel("Time (s)")
plt.show()

# %%
# Define a match and check the counts independently
# -------------------------------------------------
# With embedding order m=2, compare two-sample templates and their three-sample
# extensions. Both comparisons use the same N−m starting positions. A match
# requires **every** absolute coordinate difference to be strictly below the
# tolerance; overlapping templates are allowed, but self-matches are excluded.
# Count each unordered pair once. Sample entropy is ``−ln(long / short)`` in
# nats, using the natural logarithm, and is not normalized to [0, 1].
#
# The independent check uses SciPy's condensed pairwise distances, which exclude
# the diagonal, instead of EEGTable's chunked comparisons. The convention follows
# the short-series Chebyshev implementation described in `AntroPy's sample
# entropy documentation
# <https://raphaelvallat.com/antropy/generated/antropy.sample_entropy.html>`_.


def count_template_matches(values, embedding_order, tolerance):
    """Count strict Chebyshev matches on shared, complete embeddings."""
    templates = sliding_window_view(values, embedding_order + 1)
    templates = templates[np.isfinite(templates).all(axis=1)]
    short_count = np.count_nonzero(
        pdist(templates[:, :embedding_order], metric="chebyshev") < tolerance
    )
    long_count = np.count_nonzero(pdist(templates, metric="chebyshev") < tolerance)
    return short_count, long_count


entropy = ef.sample_entropy(
    [signal],
    windows=[window],
    order=order,
    r=tolerance_fraction,
    include_global=False,
)
selected_row = 6
selected_trace = data[selected_row]
original_tolerance = tolerance_fraction * selected_trace.std(ddof=0)
short_count, long_count = count_template_matches(selected_trace, order, original_tolerance)
assert 0 < long_count <= short_count
reference_entropy = -np.log(long_count / short_count)
np.testing.assert_allclose(entropy.values[selected_row, 0], reference_entropy, atol=1e-12)
assert np.isfinite(entropy.values).all()
assert entropy.values[labels == "white noise"].mean() > entropy.values[labels == "sine"].mean()
assert entropy.row_ids == signal.row_ids
assert entropy.meta[0].unit == "nats"
print(f"White-noise matches: {long_count} extensions / {short_count} short templates")
print(entropy.to_long()[["epoch", "event", "measure", "space", "unit", "value"]])

# %%
# Keep the tolerance fixed or recompute it at each scale
# ------------------------------------------------------
# At scale τ, average nonoverlapping blocks of τ samples and discard any trailing
# incomplete block. A block covers τ/200 s; the coarse series is sampled at
# 200/τ Hz. Order remains two coarse samples, so its physical lag increases with
# scale. This boxcar averaging and decimation changes the spectrum and does not
# provide a sharply isolated frequency band.
#
# ``original_sd`` fixes the tolerance at 0.2 × the original window SD: here 1 µV.
# ``scale_sd`` uses 0.2 × the SD of each coarse series. For white noise, averaging
# reduces SD, so the latter generally retains a stricter voltage criterion.
# These definitions answer different questions. Each scale also leaves fewer
# templates: scale six has only 120 coarse samples, making estimates less stable.

fixed = ef.multiscale_entropy(
    [signal],
    windows=[window],
    scales=scales,
    order=order,
    r=tolerance_fraction,
    tolerance_mode="original_sd",
    include_global=False,
)
rescaled = ef.multiscale_entropy(
    [signal],
    windows=[window],
    scales=scales,
    order=order,
    r=tolerance_fraction,
    tolerance_mode="scale_sd",
    include_global=False,
)
for table in [fixed, rescaled]:
    np.testing.assert_allclose(table.select(measure="mse01").values, entropy.values)
    assert np.isfinite(table.values).all()
    assert table.row_ids == signal.row_ids
    assert all(meta.unit == "nats" for meta in table.meta)

for column, scale in enumerate(scales):
    coarse_count = sample_count // scale
    coarse = selected_trace[: coarse_count * scale].reshape(coarse_count, scale).mean(axis=1)
    scale_tolerance = tolerance_fraction * coarse.std(ddof=0)
    for table, tolerance in [(fixed, original_tolerance), (rescaled, scale_tolerance)]:
        short_count, long_count = count_template_matches(coarse, order, tolerance)
        assert 0 < long_count <= short_count
        np.testing.assert_allclose(
            table.values[selected_row, column], -np.log(long_count / short_count), atol=1e-12
        )
    print(
        f"Scale {scale}: {1000 * scale / sampling_rate:.0f} ms blocks, "
        f"{sampling_rate / scale:.1f} Hz, {coarse_count} samples; "
        f"tolerances {original_tolerance * 1e6:.2f} / {scale_tolerance * 1e6:.2f} µV"
    )

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharey=True, layout="constrained")
block_widths = 1000.0 * np.asarray(scales) / sampling_rate
for ax, table, mode in zip(axes, [fixed, rescaled], ["original_sd", "scale_sd"], strict=True):
    for condition, color in zip(conditions, ["tab:blue", "tab:orange", "tab:green"], strict=True):
        values = table.values[labels == condition]
        for trial in values:
            ax.plot(block_widths, trial, color=color, alpha=0.2, linewidth=0.8)
        ax.plot(block_widths, values.mean(axis=0), "o-", color=color, label=condition)
    ax.set(title=mode, xlabel="Coarse block width (ms)", xticks=block_widths)
axes[0].set_ylabel("Sample entropy (nats)")
axes[1].legend(fontsize=9)
plt.show()

# %%
# Verify amplitude invariance and preserve undefined values
# ---------------------------------------------------------
# Multiplying every voltage by the same positive factor scales the SD-derived
# tolerance by that factor. All match decisions should therefore be unchanged,
# including under either multiscale definition. This is a controlled scaling
# check, not invariance to additive noise, quantization, filtering, or changes in
# sampling rate. The thin curves above are individual simulated trials; they do
# not constitute a confidence interval.

scaled_signal = ef.Signal.from_arrays(
    data=7.0 * signal.data,
    times=times,
    ch_names=signal.ch_names,
    sfreq=sampling_rate,
    row_ids=signal.row_ids,
)
scaled_entropy = ef.sample_entropy(
    [scaled_signal], windows=[window], order=order, r=tolerance_fraction, include_global=False
)
np.testing.assert_allclose(scaled_entropy.values, entropy.values, atol=1e-12)
for table, mode in [(fixed, "original_sd"), (rescaled, "scale_sd")]:
    scaled_mse = ef.multiscale_entropy(
        [scaled_signal],
        windows=[window],
        scales=scales,
        order=order,
        r=tolerance_fraction,
        tolerance_mode=mode,
        include_global=False,
    )
    np.testing.assert_allclose(scaled_mse.values, table.values, atol=1e-12)
    assert table.meta[0].computation.parameters["parameters"]["tolerance_mode"] == mode

# %%
# Interpret results at a consistent sample unit
# ---------------------------------------------
# Each table row is one trial; scales are correlated features of that trial.
# Compare recordings at matched sampling rates, window lengths, embedding order,
# preprocessing, tolerance mode, and data quality. More irregular noise can have
# higher entropy than a regular waveform without being biologically preferable.
# Entropy alone cannot distinguish useful signal from measurement noise.
#
# A constant trace has zero SD and therefore no strict matches: its entropy is
# NaN. No short-template matches or too few complete embeddings also produce NaN;
# short matches with no matching extension produce positive infinity. At large
# scales, an unembeddable series is NaN. Do not silently replace these with zero.

constant_times = np.arange(20) / sampling_rate
constant_signal = ef.Signal.from_arrays(
    data=np.zeros((1, 1, len(constant_times))),
    times=constant_times,
    ch_names=("Cz",),
    sfreq=sampling_rate,
    row_ids=(("undefined-example", 0, "zero_trace"),),
)
constant_entropy = ef.sample_entropy(
    [constant_signal],
    windows=[ef.Window("constant", constant_times[0], constant_times[-1])],
    order=order,
    r=tolerance_fraction,
    include_global=False,
)
assert np.isnan(constant_entropy.values).all()
np.testing.assert_array_equal(constant_entropy.coverage, 1.0)
print(constant_entropy.to_long()[["event", "measure", "value", "coverage"]])

# %%
# Preserve gaps in the time axis
# ------------------------------
#
# Native entropy tolerates gaps by using only complete consecutive embeddings;
# its SD uses finite samples. A coarse block containing a gap becomes missing.
# Gaps are never deleted to join samples across time. Missingness changes the
# available templates and potentially the estimate. Coverage describes finite
# input samples, rather than guaranteeing that entropy is defined.
#
# See :doc:`/methods/complexity` for the package definitions and Costa et al.,
# `Multiscale entropy analysis of complex physiologic time series
# <https://doi.org/10.1103/PhysRevLett.89.068102>`_ for classical coarse-graining.
