"""
Band Power and the Aperiodic Background
=======================================

An increase in alpha-band power need not come from an alpha oscillation. Build
analytic spectra with known backgrounds, compare ordinary regression with
positive peak rejection, and inspect a dimensionless ratio to the fitted
background.

This tutorial runs offline. Install its plotting dependencies from the
repository root with ``python -m pip install -e ".[docs]"``. These are exact
spectral curves, not estimated PSDs or empirical EEG recordings.
"""

# %%
# Construct three interpretable spectra
# -------------------------------------
# A decreasing power law follows ``P(f) = 10**offset * f**slope``. Frequencies
# below are numerical values in Hz and densities are in V²/Hz. Our slope is
# -1.7: an exponent defined by ``1/f**exponent`` would be +1.7. The three rows
# describe a pure background, that background multiplied by two, and the first
# background plus a Gaussian peak centered at 10 Hz.
#
# MNE's `EpochsSpectrumArray documentation
# <https://mne.tools/stable/generated/mne.time_frequency.EpochsSpectrumArray.html>`_
# describes how to represent precomputed spectral arrays. Its rows here are
# synthetic cases, not participant measurements. Declare the analytic formula
# when adapting this array; calling it Welch would misrepresent its provenance.
# No time-domain samples, spectral leakage, or estimator variance are simulated.

import matplotlib.pyplot as plt
import mne
import numpy as np

import eegtable as ef

frequencies = np.arange(1.0, 45.25, 0.25)
known_slope = -1.7
known_offset = -10.0
background = 10.0**known_offset * frequencies**known_slope
alpha_peak = 2e-11 * np.exp(-0.5 * ((frequencies - 10.0) / 0.7) ** 2)
densities = np.stack([background, 2.0 * background, background + alpha_peak])
case_names = ["background", "double background", "background + alpha peak"]
events = np.column_stack([np.arange(3) * 1000, np.zeros(3, int), [1, 2, 3]])
mne_spectrum = mne.time_frequency.EpochsSpectrumArray(
    densities[:, None, :],
    mne.create_info(["Oz"], 200.0, "eeg"),
    frequencies,
    events=events,
    event_id=dict(zip(case_names, [1, 2, 3], strict=True)),
    verbose=False,
)
spectra = ef.Spectra.from_spectrum(
    mne_spectrum,
    recording="analytic-background-cases",
    estimator_parameters={
        "method": "analytic-power-law-plus-gaussian",
        "slope": known_slope,
        "offsets": [known_offset, known_offset + np.log10(2.0), known_offset],
        "peak_amplitudes_v2_per_hz": [0.0, 0.0, 2e-11],
        "peak_center_hz": 10.0,
        "peak_sd_hz": 0.7,
    },
)
assert spectra.data.shape == (3, 1, 1, frequencies.size)
assert np.isfinite(spectra.data).all()
assert (spectra.data > 0.0).all()

# %%
# Measure the whole alpha-band density
# ------------------------------------
# Integrated band power includes every component within 8–13 Hz. The integral
# has units V²; the plot converts it to µV². EEGTable integrates piecewise-linear
# spectra between exact band edges. That numerical integral approximates the
# integral of our continuous curves, with accuracy set by the 0.25 Hz grid.
#
# Doubling the background must double its band integral, even though this row
# has no peak. Adding a peak also raises the integral. Those two mechanisms
# cannot be distinguished from a band-power value alone.

alpha = ef.Band("alpha", 8.0, 13.0)
power = ef.integrated_band_power(spectra, bands=[alpha], include_global=False)
assert all(meta.unit == "V^2" for meta in power.meta)
assert power.row_ids == spectra.row_ids
np.testing.assert_allclose(power.values[1], 2.0 * power.values[0], rtol=1e-12)
assert power.values[2, 0] > power.values[0, 0]

fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), layout="constrained")
for density, name in zip(densities, case_names, strict=True):
    axes[0].loglog(frequencies, density * 1e12, label=name)
axes[0].axvspan(alpha.fmin, alpha.fmax, color="0.9", zorder=0)
axes[0].set(xlabel="Frequency (Hz)", ylabel="PSD (µV²/Hz)", title="Analytic spectra")
axes[0].legend(fontsize="small")
axes[1].bar(np.arange(3), power.values[:, 0] * 1e12, color=["C0", "C1", "C2"])
axes[1].set(
    xticks=np.arange(3),
    xticklabels=["background", "double\nbackground", "+ alpha\npeak"],
    ylabel="Alpha-band power (µV²)",
    title="Different causes of increased band power",
)
plt.show()

# %%
# Fit a background and verify known parameters
# --------------------------------------------
# ``aperiodic`` fits ``log10(P) = offset + slope * log10(f)`` over the half-open
# interval ``[2, 40)`` Hz. It first uses ordinary least squares, then rejects
# positive residuals exceeding a threshold based on the median absolute
# deviation and refits, for at most three rounds here. Negative residuals stay
# in the fit. This asymmetric rejection targets peaks above a background; it
# does not fit Gaussian peak parameters or a knee.
#
# The offset describes the fitted density at 1 Hz in the chosen units, even
# though 1 Hz lies outside this fit interval. Converting densities from V²/Hz
# to µV²/Hz adds 12 to the offset and leaves the slope unchanged. Keep input
# units consistent when comparing fitted offsets.

fit_settings = {"fit_range": (2.0, 40.0), "peak_rejection_z": 2.5, "max_iterations": 3}
fits = ef.aperiodic(spectra, include_global=False, **fit_settings)
slopes = fits.select(measure="slope").values[:, 0]
offsets = fits.select(measure="offset").values[:, 0]
assert np.isfinite(fits.values).all()
np.testing.assert_allclose(slopes[:2], known_slope, atol=1e-10, rtol=0.0)
np.testing.assert_allclose(
    offsets[:2], [known_offset, known_offset + np.log10(2.0)], atol=1e-10, rtol=0.0
)

fit_mask = (frequencies >= 2.0) & (frequencies < 40.0)
ordinary_slope, ordinary_offset = np.polyfit(
    np.log10(frequencies[fit_mask]), np.log10(densities[2, fit_mask]), 1
)
assert abs(slopes[2] - known_slope) < abs(ordinary_slope - known_slope)
print(f"Known slope: {known_slope:.4f}")
print(f"Peak case, ordinary regression: {ordinary_slope:.4f}")
print(f"Peak case, positive peak rejection: {slopes[2]:.4f}")
print(fits.to_long()[["event", "measure", "value", "unit"]])

# %%
# Inspect the ratio to the fitted background
# ------------------------------------------
# ``aperiodic_ratio`` divides the measured density by its fitted background.
# A value of one matches that background; two means twice its fitted density
# at that frequency. This dimensionless ratio is distinct from a subtracted
# periodic PSD in V²/Hz. It cannot be passed to ``integrated_band_power`` as
# physical EEG power. A ratio below one indicates density below the fitted
# curve, and the DC bin, if present, has no defined power-law ratio.
#
# The two pure power laws flatten to one. For the peaked row, compare the fitted
# ratio with the ratio to the known background; their agreement depends on fit
# accuracy. We assert improved slope estimation above, not exact recovery of
# the added peak or a biological interpretation of its amplitude.

ratio = ef.aperiodic_ratio(spectra, **fit_settings)
assert ratio.representation == "aperiodic_ratio"
assert np.isfinite(ratio.data).all()
np.testing.assert_allclose(ratio.data[:2], 1.0, rtol=1e-10, atol=0.0)
assert ratio.data[2, 0, 0, np.argmin(np.abs(frequencies - 10.0))] > 1.0

ordinary_background = 10.0**ordinary_offset * frequencies**ordinary_slope
rejected_background = 10.0 ** offsets[2] * frequencies ** slopes[2]
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), layout="constrained")
axes[0].loglog(frequencies, densities[2] * 1e12, color="0.6", label="With alpha peak")
axes[0].loglog(frequencies, background * 1e12, label="Known background")
axes[0].loglog(frequencies, ordinary_background * 1e12, "--", label="Ordinary fit")
axes[0].loglog(frequencies, rejected_background * 1e12, ":", label="Peak-rejecting fit")
axes[0].set(xlabel="Frequency (Hz)", ylabel="PSD (µV²/Hz)", title="Influence of a positive peak")
axes[0].legend(fontsize="small")
for row, name in enumerate(case_names):
    axes[1].plot(frequencies, ratio.data[row, 0, 0], label=name)
axes[1].plot(frequencies, densities[2] / background, "k--", label="Known-background ratio")
axes[1].axhline(1.0, color="0.6", linewidth=0.8)
axes[1].set(
    xlim=(2.0, 40.0),
    xlabel="Frequency (Hz)",
    ylabel="PSD / fitted background",
    title="Dimensionless ratios",
)
axes[1].legend(fontsize="small")
plt.show()

# %%
# Choose the model and range deliberately
# ---------------------------------------
# Each retained bin has equal regression weight. A linear frequency grid gives
# more weight per log-frequency interval at high frequencies than a log-spaced
# grid. Changing the grid or fit range can therefore change the result. Peak
# location and width also matter: the bias direction shown here is specific to
# this example, and broad peaks may not be rejected successfully.
#
# Use a range supported by the spectrum and the recording's passband. Filter
# roll-off, line noise, knees, and spectral curvature can invalidate a straight
# power-law model. The ratio is evaluated across the full frequency axis, so
# values outside the fit range are extrapolations. The fit interval must contain
# at least five frequency-axis bins; a coarser grid raises an error. On a valid
# grid, a cell with fewer than five finite positive powers has missing fit
# parameters. ``aperiodic_ratio`` also marks that cell's missing ratios with
# ``aperiodic_fit_failed`` rather than substituting the original spectrum.
# The reported R² describes retained bins and does not establish that a complete
# periodic/aperiodic decomposition is correct.
#
# Next, :doc:`plot_irasa` demonstrates a resampling-based decomposition with a
# signed periodic residual. MNE's `Spectrum tutorial
# <https://mne.tools/stable/auto_tutorials/time-freq/10_spectrum_class.html>`_
# covers estimating spectra from actual recordings.
