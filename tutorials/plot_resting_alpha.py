"""
Resting Alpha Power with Eyes Open and Closed
=============================================

Does occipital alpha power differ between two resting recordings? Extract one
value per four-second segment of public EEG, inspect its units and provenance,
and save a feature table without losing the segment identities.

This tutorial needs internet access on its first run. The simulation tutorials
work offline. Install the documentation dependencies with
``python -m pip install -e ".[docs]"`` from the repository root.
"""

# %%
# Prepare two resting recordings
# ------------------------------
# EEGBCI run 1 is eyes open and run 2 is eyes closed. These are separate,
# approximately one-minute recordings from participant 1, not randomized trials.
# MNE downloads them once and caches them in its configured data directory.
# Its `Spectrum tutorial
# <https://mne.tools/stable/auto_tutorials/time-freq/10_spectrum_class.html>`_
# explains the spectral objects used below.
#
# Keep EEG in volts. Apply an average reference across the recorded EEG channels
# before selecting the occipital ROI. Filtering and annotation rejection here
# demonstrate data preparation; a research analysis also needs reviewed bad
# channels and artifact handling, as described in :doc:`/guides/preprocessing`.

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import eegbci

import eegtable as ef
from eegtable.io import read_table, write_table

conditions = {"eyes-open": 1, "eyes-closed": 2}
paths = eegbci.load_data(1, list(conditions.values()), update_path=False, verbose=False)
epoch_sets = []
for path, (condition, code) in zip(paths, conditions.items(), strict=True):
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    eegbci.standardize(raw)
    raw.filter(1.0, 40.0, verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    events = mne.make_fixed_length_events(
        raw, id=code, start=2.0, stop=raw.times[-1] - 2.0, duration=4.0
    )
    epochs = mne.Epochs(
        raw,
        events,
        event_id={condition: code},
        tmin=0.0,
        tmax=4.0 - 1.0 / raw.info["sfreq"],
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )
    # Rejection has consumed the annotations; MNE cannot retain them when
    # concatenating epochs from separate runs.
    epochs.set_annotations(None)
    epoch_sets.append(epochs.pick(["O1", "Oz", "O2"]))

epochs = mne.concatenate_epochs(epoch_sets, verbose=False)
print(epochs)

# %%
# Estimate a PSD for every segment
# --------------------------------
# Two-second Hann segments with 50% overlap give a 0.5 Hz frequency grid.
# Welch averages the periodograms within each epoch; it does not average across
# epochs. EEG PSD density has units V²/Hz. The assertion checks that EEGTable and
# MNE return the same densities with the same estimator settings. EEGTable keeps
# one extra bin at each frequency boundary so band integration can interpolate
# exact edges; request those same bins in the reference calculation.

segment_samples = round(2.0 * epochs.info["sfreq"])
overlap_samples = segment_samples // 2
frequency_spacing = epochs.info["sfreq"] / segment_samples
spectra = ef.Spectra.welch(
    epochs,
    recording="S001-rest",
    fmin=1.0,
    fmax=40.0,
    n_fft=segment_samples,
    n_overlap=overlap_samples,
)
mne_spectrum = epochs.compute_psd(
    method="welch",
    fmin=1.0 - frequency_spacing,
    fmax=40.0 + frequency_spacing,
    n_fft=segment_samples,
    n_per_seg=segment_samples,
    n_overlap=overlap_samples,
    window="hann",
    average="mean",
    remove_dc=True,
    verbose=False,
)
np.testing.assert_array_equal(spectra.freqs, mne_spectrum.freqs)
np.testing.assert_allclose(spectra.data[:, :, 0, :], mne_spectrum.get_data())

fig, ax = plt.subplots(figsize=(7.0, 3.4), layout="constrained")
for condition, code in conditions.items():
    density = spectra.data[epochs.events[:, 2] == code, :, 0, :].mean(axis=(0, 1))
    ax.semilogy(spectra.freqs, density * 1e12, label=condition)
ax.axvspan(8.0, 13.0, color="0.9", label="alpha band")
ax.set(xlabel="Frequency (Hz)", ylabel="PSD (µV²/Hz)", title="Occipital resting EEG")
ax.legend()
plt.show()

# %%
# Integrate alpha power and inspect its metadata
# ----------------------------------------------
# Integrating 8–13 Hz turns density into power in V². The occipital ROI averages
# the three channels' powers, rather than averaging their voltages first.
# ``include_global=False`` avoids a second, identical average over this ROI-only
# input. Convert to µV² only for the figure; the table retains its recorded unit.
# Numerical coverage describes finite input, not the absence of EEG artifacts.

alpha = ef.Band("alpha", 8.0, 13.0)
table = ef.integrated_band_power(
    spectra,
    bands=[alpha],
    groups={"occipital": ["O1", "Oz", "O2"]},
    include_global=False,
)
long = table.to_long()
print(long[["epoch", "event", "space", "band", "unit", "value", "coverage"]].head())
print(long.groupby("event")["value"].agg(["mean", "median", "count"]))

fig, ax = plt.subplots(figsize=(5.5, 3.4), layout="constrained")
ax.boxplot(
    [long.loc[long["event"] == condition, "value"] * 1e12 for condition in conditions],
    tick_labels=list(conditions),
)
ax.set(ylabel="Alpha power (µV²)", title="One value per four-second segment")
plt.show()

# %%
# Save and restore the complete table
# -----------------------------------
# The TSV holds values and row identities. Companion files retain coverage and
# feature definitions. A temporary directory keeps this runnable example from
# overwriting an analysis; in your project, replace it with your output folder.

with TemporaryDirectory() as directory:
    output = Path(directory) / "sub-01_task-rest_features.tsv"
    write_table(table, output)
    restored = read_table(output)
    np.testing.assert_allclose(restored.values, table.values)
    np.testing.assert_array_equal(restored.coverage, table.coverage)
    assert restored.row_ids == table.row_ids
    assert restored.meta == table.meta
    print(f"Restored {len(restored.values)} segments with matching values and metadata.")

# %%
# Interpret this comparison
# -------------------------
# Alpha can be more prominent with eyes closed, but inspect the spectrum rather
# than assuming that result for every participant. Segments within one recording
# are temporally dependent, and condition is confounded with recording order.
# These boxplots are descriptive; treating segments as independent participants
# would overstate evidence for a population effect. A cohort analysis should
# compare conditions within participants and retain participant/run identities.
#
# Next, try :doc:`plot_beta_bursts` for a time-resolved amplitude measure, or
# :doc:`/guides/tables` for cohort export.
#
# The public recordings are described by Schalk et al. (2004),
# `BCI2000: a general-purpose brain-computer interface system
# <https://doi.org/10.1109/TBME.2004.827072>`_, and distributed through the
# `PhysioNet EEG Motor Movement/Imagery Dataset
# <https://physionet.org/content/eegmmidb/1.0.0/>`_.
