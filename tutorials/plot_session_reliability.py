"""
Repeated-Session Reliability: Agreement and Consistency
=======================================================

Can a feature preserve participants' relative differences while changing its
level between sessions? Extract alpha power from a repeated-session simulation,
explicitly average trials within sessions, and compare absolute-agreement
ICC(2,1) with consistency ICC(3,1).

This example works offline with ``python -m pip install -e ".[docs]"``.
The session gain is planted to explain the statistics, not to benchmark EEG
reliability. Every participant contributes both sessions with the same settings.
"""

# %%
# Simulate a balanced repeated-session cohort
# -------------------------------------------
# Twelve participants each contribute twelve four-second trials at two sessions.
# O1 has stable alpha amplitude apart from random session/trial variation. O2
# has the same underlying participant effect, with a multiplicative gain of 1.6
# in session two. This could represent a measurement change rather than biology.
# No average reference is applied to this two-channel demonstration.
#
# All signals are in volts. Welch uses two-second Hann segments with 50% overlap.
# Feature extraction first integrates 8–13 Hz power in V², then takes log10.
# A gain in amplitude therefore becomes approximately an additive shift of
# ``2 * log10(1.6)`` in log power. Noise makes that relation approximate.

from dataclasses import replace

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

import eegtable as ef

rng = np.random.default_rng(47)
sampling_rate = 128.0
times = np.arange(512) / sampling_rate
participants = [f"sub-{number:02d}" for number in range(1, 13)]
sessions = ["one", "two"]
trials_per_session = 12
band = ef.Band("alpha", 8.0, 13.0)
trial_tables = []
sample_records = []
for participant in participants:
    participant_amplitude = 5e-6 * rng.lognormal(0.0, 0.20)
    for session in sessions:
        recording = f"{participant}_ses-{session}"
        amplitude = participant_amplitude * rng.lognormal(0.0, 0.03)
        trial_amplitudes = amplitude * rng.lognormal(0.0, 0.10, (trials_per_session, 1, 1))
        channel_gains = np.array([1.0, 1.6 if session == "two" else 1.0])
        phases = rng.uniform(-np.pi, np.pi, (trials_per_session, 2, 1))
        data = (
            trial_amplitudes
            * channel_gains[None, :, None]
            * np.sin(2.0 * np.pi * 10.0 * times + phases)
        )
        data += rng.normal(0.0, 0.3e-6, data.shape)
        epochs = mne.EpochsArray(
            data,
            mne.create_info(["O1", "O2"], sampling_rate, "eeg"),
            event_id={"rest": 1},
            baseline=None,
            verbose=False,
        )
        spectra = ef.Spectra.welch(
            epochs, recording=recording, fmin=1.0, fmax=40.0, n_fft=256, n_overlap=128
        )
        trial_tables.append(
            ef.integrated_band_power(spectra, bands=[band], normalize="log10", include_global=False)
        )
        sample_records.append(
            {"subject_id": participant, "session": session, "recording": recording}
        )

# %%
# Define the session measurement explicitly
# -----------------------------------------
# Each session measurement is the arithmetic mean of its twelve log10 powers.
# It equals the log10 of the geometric mean of the original powers, not the
# log10 of their arithmetic mean. Prespecify this aggregation and use it in both
# sessions. Finite data are required here; a missing trial value must surface
# before aggregation rather than disappear inside pandas' default mean.
#
# Group rows identify session estimates without pretending they are epochs.
# Record the aggregation and trial count in each feature's computation metadata.
# The ICC below is single-measure reliability of this session estimate, rather
# than reliability of a single epoch or of an average across both sessions.

cohort = ef.stack_rows(trial_tables, columns="identical")
assert np.isfinite(cohort.values).all()
np.testing.assert_array_equal(cohort.coverage, 1.0)
session_values = np.stack([table.values.mean(axis=0) for table in trial_tables])
session_metadata = tuple(
    replace(
        feature,
        computation=ef.ComputationSpec.create(
            "session_mean",
            input_computation=feature.computation.record(),
            n_trials=trials_per_session,
            aggregation="arithmetic_mean_of_log10_power",
        ),
    )
    for feature in cohort.meta
)
descriptors = pd.DataFrame(sample_records)
session_table = ef.FeatureTable(
    values=session_values,
    coverage=np.ones_like(session_values),
    meta=session_metadata,
    row_labels=tuple(descriptors["recording"]),
)
assert session_table.row_labels == tuple(descriptors["recording"])
print(f"{cohort.n_rows} epochs become {session_table.n_rows} subject/session measurements.")
print(session_table.to_long()[["group", "space", "unit", "value"]].head())

# %%
# Inspect equality as well as relative differences
# ------------------------------------------------
# Each point is one participant. The identity line indicates equal values at
# both sessions. The O2 points can preserve relative participant differences
# while lying away from that line; correlation alone would miss a common offset.

paired = session_table.values.reshape(len(participants), len(sessions), -1)
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.7), layout="constrained")
for column, (ax, feature) in enumerate(zip(axes, session_table.meta, strict=True)):
    values = paired[:, :, column]
    limits = [values.min() - 0.05, values.max() + 0.05]
    ax.scatter(values[:, 0], values[:, 1])
    ax.plot(limits, limits, color="0.5", linestyle="--")
    ax.set(
        xlim=limits,
        ylim=limits,
        xlabel="Session one: mean log10(power / 1 V²)",
        ylabel="Session two: mean log10(power / 1 V²)",
        title=feature.space,
        aspect="equal",
    )
plt.show()

# %%
# Estimate ICC and verify a two-session ANOVA identity
# ----------------------------------------------------
# Absolute agreement penalizes session offsets. Consistency accounts for fixed
# session mean differences. Both estimates depend on between-participant
# variability, so they describe reliability in this simulated population rather
# than an intrinsic property of alpha power. Negative ICC estimates are possible
# and should be reported without clipping.
#
# The definitions follow the two-way ANOVA framework documented by
# `Pingouin <https://pingouin-stats.org/generated/pingouin.intraclass_corr.html>`_.
# For exactly two sessions, subject and error mean squares can also be computed
# from the variance of participants' pair means and pair differences. This
# alternate calculation verifies both reported ICCs without another dependency.

reliability = ef.intraclass_reliability(session_table, descriptors)
reliability["space"] = [feature.space for feature in session_table.meta]
print(reliability[["space", "n_subjects", "n_sessions", "icc_absolute", "icc_consistency"]])

differences = paired[:, 1] - paired[:, 0]
subject_mean_square = 2.0 * paired.mean(axis=1).var(axis=0, ddof=1)
error_mean_square = differences.var(axis=0, ddof=1) / 2.0
session_mean_square = len(participants) * differences.mean(axis=0) ** 2 / 2.0
numerator = subject_mean_square - error_mean_square
denominator = subject_mean_square + error_mean_square
np.testing.assert_allclose(reliability["icc_consistency"], numerator / denominator)
np.testing.assert_allclose(
    reliability["icc_absolute"],
    numerator / (denominator + 2.0 * (session_mean_square - error_mean_square) / len(participants)),
)

# %%
# Check what a fixed session offset changes
# -----------------------------------------
# Remove exactly the planted log-power offset from O2 in session two. This is a
# simulation diagnostic, not a recommended data correction. The consistency
# estimate must remain unchanged because all participants receive the same
# subtraction; absolute agreement generally changes. Do not choose corrections
# by optimizing an ICC on the data used to report reliability.

offset = 2.0 * np.log10(1.6)
corrected_values = session_table.values.copy()
corrected_values[descriptors["session"] == "two", 1] -= offset
corrected = ef.intraclass_reliability(replace(session_table, values=corrected_values), descriptors)
np.testing.assert_allclose(corrected["icc_consistency"], reliability["icc_consistency"])
print(f"Planted O2 shift: {offset:.3f} log10 units.")
print(f"O2 agreement after removing the planted shift: {corrected.loc[1, 'icc_absolute']:.3f}")

# %%
# Design and report a reliability analysis
# ----------------------------------------
# This API requires one row per subject/session, a complete balanced design,
# finite values, at least three participants and at least two sessions. Duplicate
# samples, missing sessions and constant features raise errors. Resolve such
# designs explicitly; do not silently fill values or discard unmatched subjects.
# Three participants are a computational minimum, not a sample-size guideline.
#
# ICC(2,1) uses a random-session agreement formulation; ICC(3,1) treats session
# effects as fixed for consistency. Choose the interpretation to match the study
# and report the measurement unit, feature definition, trial aggregation, number
# of participants/sessions, and any session offset. EEGTable returns point
# estimates here; it does not provide confidence intervals. A real reliability
# study also needs justified uncertainty estimates and measurement equivalence
# across sessions. See :doc:`/guides/cohorts` for quality policies and cohort
# descriptors, and :doc:`plot_grouped_modeling` for participant-disjoint prediction.
