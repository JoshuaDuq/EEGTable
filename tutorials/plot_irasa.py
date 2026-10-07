"""
IRASA with a Signed Periodic Residual
=====================================

Separate a broadband background from a simulated alpha oscillator, retain
signed residuals, and verify EEGTable's fitted parameters and band integrals
against direct NeuroDSP calculations with identical settings.

This tutorial runs offline. From the repository root install its dependencies
with ``python -m pip install -e ".[docs,irasa]"``. NeuroDSP is required; an absent
dependency raises an error. All recordings here are seeded simulations.
"""

# %%
# Simulate broadband samples in volts
# -----------------------------------
# Six 12-second trials contain Gaussian white noise with a standard deviation
# of 5 µV. Three also contain a 10 Hz sinusoid with a peak amplitude of 7 µV.
# These amplitudes illustrate the method and make no claim about typical EEG.
# A white-noise background has a flat expected PSD; finite realizations do not
# have exactly zero fitted slope. Independent realizations and oscillator phases
# are controlled by one NumPy random generator.
#
# MNE's `array construction tutorial
# <https://mne.tools/stable/auto_tutorials/simulation/10_array_objs.html>`_
# establishes the ``(epochs, channels, samples)`` convention. No bandpass filter
# is applied: IRASA needs broadband inputs, including frequencies reached by
# resampling. The sample unit is V, densities will be V²/Hz, and band integrals
# will be V².

import matplotlib.pyplot as plt
import mne
import numpy as np
from neurodsp.aperiodic import compute_irasa, fit_irasa
from neurodsp.spectral import compute_spectrum
from scipy.integrate import trapezoid

import eegtable as ef

rng = np.random.default_rng(42)
sampling_rate = 200.0
times = np.arange(2400) / sampling_rate
labels = np.repeat(["noise", "noise + alpha"], 3)
noise = rng.normal(0.0, 5e-6, size=(6, times.size))
phases = rng.uniform(0.0, 2.0 * np.pi, size=(6, 1))
amplitudes = np.repeat([0.0, 7e-6], 3)[:, None]
data = noise + amplitudes * np.sin(2.0 * np.pi * 10.0 * times + phases)
events = np.column_stack([np.arange(6) * times.size, np.zeros(6, int), np.repeat([1, 2], 3)])
epochs = mne.EpochsArray(
    data[:, None, :],
    mne.create_info(["Oz"], sampling_rate, "eeg"),
    events=events,
    event_id={"noise": 1, "noise + alpha": 2},
    baseline=None,
    verbose=False,
)
signal = ef.Signal.from_epochs(epochs, recording="simulated-irasa")
assert signal.data.shape == (6, 1, times.size)
assert np.isfinite(signal.data).all()

# %%
# Decompose each trial with explicit settings
# -------------------------------------------
# The `NeuroDSP IRASA tutorial
# <https://neurodsp-tools.github.io/neurodsp/auto_tutorials/aperiodic/plot_IRASA.html>`_
# explains paired up/down resampling: it shifts narrowband peaks and estimates
# the background from the median of geometric-mean spectra. EEGTable delegates
# this step to NeuroDSP and uses Hann-window Welch densities with an arithmetic
# mean and 50% overlap. Two-second segments give a 0.5 Hz grid here. Five factors
# keep this demonstration short; their suitability should be checked for the
# peak widths and spectral structure of a research dataset.
#
# Fit the aperiodic component in the half-open interval ``[2.1, 39.9)`` Hz.
# These edges intentionally fall between bins. Band integrals interpolate exact
# boundaries and need outward bracketing bins, including those just outside the
# fit interval. Alongside alpha and beta, a broadband integral over these same
# non-grid edges checks that support. No periodic threshold is applied: the
# periodic component remains the signed difference between the total PSD and
# estimated background.

window = ef.Window("all", 0.0, float(times[-1]))
fit_range = (2.1, 39.9)
bands = [
    ef.Band("alpha", 8.0, 13.0),
    ef.Band("beta", 13.0, 30.0),
    ef.Band("broadband", *fit_range),
]
resampling_factors = (1.1, 1.3, 1.5, 1.7, 1.9)
segment_seconds = 2.0
segment_samples = round(segment_seconds * sampling_rate)
table = ef.irasa(
    [signal],
    windows=[window],
    bands=bands,
    fit_range=fit_range,
    hset=resampling_factors,
    segment_seconds=segment_seconds,
    include_global=False,
)
assert table.row_ids == signal.row_ids
assert np.isfinite(table.values).all()
assert (table.coverage == 1.0).all()
component_measures = ["irasa_aperiodic_power", "irasa_periodic_power"]
assert all(meta.unit == "V^2" for meta in table.meta if meta.measure in component_measures)
print(
    table.to_long()[["epoch", "event", "measure", "band", "unit", "value"]]
    .tail(8)
    .to_string(index=False)
)

# %%
# Reproduce a selected trial directly with NeuroDSP
# -------------------------------------------------
# The `compute_irasa API
# <https://neurodsp-tools.github.io/neurodsp/generated/neurodsp.aperiodic.irasa.compute_irasa.html>`_
# accepts the Welch settings below. Request the full axis first, then keep the
# same bracketing support as EEGTable. Passing a trimmed frequency interval to
# the backend prematurely could discard a bin needed for boundary integration.
# Its `fit_irasa API
# <https://neurodsp-tools.github.io/neurodsp/generated/neurodsp.aperiodic.irasa.fit_irasa.html>`_
# fits a line in log-log space; pass exactly the same half-open fit mask.
#
# Verify the additive identity against a separate Welch spectrum. Negative
# residual bins are valid estimates. Clipping them would raise component power
# and break ``total = aperiodic + periodic`` for the original decomposition.

selected_trial = 5
trace = signal.data[selected_trial, 0]
welch_settings = {
    "nperseg": segment_samples,
    "noverlap": segment_samples // 2,
    "avg_type": "mean",
    "window": "hann",
}
full_frequencies, full_aperiodic, full_periodic = compute_irasa(
    trace,
    sampling_rate,
    f_range=None,
    hset=resampling_factors,
    thresh=None,
    **welch_settings,
)
welch_frequencies, full_total = compute_spectrum(trace, sampling_rate, **welch_settings)
np.testing.assert_array_equal(welch_frequencies, full_frequencies)
np.testing.assert_allclose(full_total, full_aperiodic + full_periodic, rtol=1e-12, atol=1e-24)

left = np.searchsorted(full_frequencies, fit_range[0], side="right") - 1
right = np.searchsorted(full_frequencies, fit_range[1], side="left") + 1
integration_support = slice(left, right)
frequencies = full_frequencies[integration_support]
aperiodic_density = full_aperiodic[integration_support]
periodic_density = full_periodic[integration_support]
total_density = full_total[integration_support]
fit_mask = (frequencies >= fit_range[0]) & (frequencies < fit_range[1])
reference_offset, reference_slope = fit_irasa(frequencies[fit_mask], aperiodic_density[fit_mask])
for measure, expected in [("irasa_offset", reference_offset), ("irasa_slope", reference_slope)]:
    actual = table.select(measure=measure).values[selected_trial, 0]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
assert np.any(periodic_density < 0.0)

fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.2), sharex=True, layout="constrained")
axes[0].semilogy(frequencies, total_density * 1e12, label="Total Welch PSD")
axes[0].semilogy(frequencies, aperiodic_density * 1e12, label="IRASA background")
axes[0].set(ylabel="PSD (µV²/Hz)", title=f"Selected simulated trial {selected_trial}")
axes[0].legend()
axes[1].plot(frequencies, periodic_density * 1e12, color="C2", label="Signed periodic residual")
axes[1].axhline(0.0, color="0.4", linewidth=0.8)
axes[1].set_yscale("symlog", linthresh=0.02)
axes[1].set(xlabel="Frequency (Hz)", ylabel="Residual (µV²/Hz; symlog)", xlim=fit_range)
axes[1].legend()
for ax in axes:
    ax.axvspan(8.0, 13.0, color="0.9", zorder=0)
plt.show()

# %%
# Check component integrals with independent quadrature
# -----------------------------------------------------
# Interpolate each density to both exact band edges, then use SciPy's trapezoidal
# integral. This independently checks the wrapper's quadrature and the V² units.
# Additivity must also hold after integration because integration is linear.
# The signed beta residual in this selected trial is negative; preserve it.

for band in bands:
    interior = (frequencies > band.fmin) & (frequencies < band.fmax)
    integration_axis = np.concatenate([[band.fmin], frequencies[interior], [band.fmax]])
    reference_integrals = []
    for measure, density in zip(
        component_measures, [aperiodic_density, periodic_density], strict=True
    ):
        reference = trapezoid(np.interp(integration_axis, frequencies, density), integration_axis)
        actual = table.select(measure=measure, band=band.name).values[selected_trial, 0]
        np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-24)
        reference_integrals.append(reference)
    total_integral = trapezoid(
        np.interp(integration_axis, frequencies, total_density), integration_axis
    )
    np.testing.assert_allclose(sum(reference_integrals), total_integral, rtol=1e-10, atol=1e-24)

alpha_background = table.select(measure="irasa_aperiodic_power", band="alpha").values[:, 0]
alpha_periodic = table.select(measure="irasa_periodic_power", band="alpha").values[:, 0]
beta_periodic = table.select(measure="irasa_periodic_power", band="beta").values[:, 0]
assert beta_periodic[selected_trial] < 0.0
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), layout="constrained")
trials = np.arange(len(labels))
for values, marker, name in [
    (alpha_background + alpha_periodic, "o", "Total"),
    (alpha_background, "s", "Aperiodic"),
    (alpha_periodic, "^", "Signed periodic"),
]:
    axes[0].plot(trials, values * 1e12, marker, label=name)
axes[0].set(
    xlabel="Simulated trial", ylabel="Alpha-band power (µV²)", title="Component band integrals"
)
axes[0].legend(fontsize="small")
axes[1].plot(trials, beta_periodic * 1e12, "o", color="C2")
axes[1].axhline(0.0, color="0.4", linewidth=0.8)
axes[1].set(
    xlabel="Simulated trial",
    ylabel="Signed beta residual power (µV²)",
    title="Residuals can have either sign",
)
for ax in axes:
    ax.set_xticks(trials)
    ax.axvline(2.5, color="0.7", linestyle="--")
    ax.text(0.03, 0.83, "Noise", transform=ax.transAxes, va="top")
    ax.text(0.58, 0.83, "Noise + alpha", transform=ax.transAxes, va="top")
plt.show()

# %%
# Respect the resampling boundaries
# ---------------------------------
# At the largest factor 1.9, the retained 2–40 Hz support reaches approximately
# 1.05–76 Hz. Both limits are inside this unfiltered recording's 0–100 Hz
# passband and below its 100 Hz Nyquist frequency. In general, check the expanded
# bracketing support, not only ``fit_range * max(hset)``. A filtered recording
# needs a passband wide enough for all resampled frequencies. Every window also
# needs at least one complete Welch segment after the largest downsampling;
# gaps and constant traces are refused rather than repaired.
#
# IRASA assumes the background remains suitable for resampling. Broad or
# overlapping peaks, nonstationary oscillations, knees, filter transitions, and
# finite sampling can affect the separation and fitted line. A negative residual
# is an estimation discrepancy, not negative physiological energy. The finite
# white-noise realizations explain why the fitted slopes vary around zero; this
# example validates numerical agreement without demanding exact recovery of the
# sinusoid's planted variance. Simulation trials are the unit of these plots,
# not independent participants or evidence for a population effect.
#
# See Wen and Liu (2016), `Separating Fractal and Oscillatory Components in the
# Power Spectrum of Neurophysiological Signal
# <https://doi.org/10.1007/s10548-015-0448-0>`_, for the method, and
# :doc:`plot_aperiodic_background` for a direct power-law fit and a dimensionless
# background ratio.
