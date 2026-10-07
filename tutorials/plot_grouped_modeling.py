"""
Predicting Trial Conditions in Held-Out Participants
====================================================

Extract single-trial spectral predictors, align them with their labels, and
evaluate a classifier on participants excluded from training. Use nested
participant-disjoint validation to select regularization without consulting
the held-out participant.

This is a seeded simulation with a deliberately planted condition effect.
Its scores teach the workflow and do not estimate performance on real EEG.
Install ``python -m pip install -e ".[docs,model]"`` to run it.
"""

# %%
# Simulate a small cohort
# -----------------------
# Eight participants each contribute thirty balanced trials. Condition 1 raises
# the amplitude of a 10 Hz oscillation by a factor of 1.5; participant-specific
# gains and independent trial variation overlap the distributions. A 20 Hz
# component has no condition effect. Every array is in volts and every trial
# has explicit recording, epoch, event, participant, and target labels.
#
# Welch settings and feature definitions are fixed before evaluation. These
# per-trial transformations learn nothing from other participants, so extraction
# can happen before splitting. Learned spatial filters or microstates would need
# the workflow in :doc:`/guides/learned_features` instead.

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

import eegtable as ef
from eegtable import model as efm

rng = np.random.default_rng(31)
sampling_rate = 128.0
times = np.arange(512) / sampling_rate
trials_per_participant = 30
bands = [ef.Band("alpha", 8.0, 13.0), ef.Band("beta", 13.0, 30.0)]
tables = []
target_frames = []
for participant in range(1, 9):
    recording = f"sim-{participant:02d}"
    labels = rng.permutation(np.tile([0, 1], trials_per_participant // 2))
    participant_gain = rng.lognormal(0.0, 0.2)
    amplitudes = 5e-6 * participant_gain * (1.0 + 0.5 * labels)
    amplitudes *= rng.lognormal(0.0, 0.18, trials_per_participant)
    phases = rng.uniform(0.0, 2.0 * np.pi, (trials_per_participant, 2, 1))
    alpha_wave = amplitudes[:, None, None] * np.sin(2.0 * np.pi * 10.0 * times + phases)
    beta_wave = 3e-6 * participant_gain * np.sin(2.0 * np.pi * 20.0 * times + phases)
    data = alpha_wave + beta_wave + rng.normal(0.0, 1e-6, alpha_wave.shape)
    events = np.column_stack(
        [
            np.arange(trials_per_participant) * len(times),
            np.zeros(trials_per_participant, int),
            labels + 1,
        ]
    )
    epochs = mne.EpochsArray(
        data,
        mne.create_info(["C3", "C4"], sampling_rate, "eeg"),
        events=events,
        event_id={"condition-0": 1, "condition-1": 2},
        baseline=None,
        verbose=False,
    )
    spectra = ef.Spectra.welch(
        epochs, recording=recording, fmin=1.0, fmax=40.0, n_fft=256, n_overlap=128
    )
    features = ef.integrated_band_power(
        spectra, bands=bands, normalize="log10", include_global=False
    )
    tables.append(features)
    targets = features.to_dataframe().index.to_frame(index=False)
    targets["participant"] = recording
    targets["condition"] = labels
    target_frames.append(targets)

cohort = ef.stack_rows(tables, columns="identical")
targets = pd.concat(target_frames, ignore_index=True)
print(f"{len(cohort.values)} trials, {len(tables)} participants, {len(cohort.meta)} predictors")

# %%
# Align targets by identity rather than row position
# --------------------------------------------------
# ``build_design`` joins on ``(recording, epoch, event)`` and refuses unmatched or
# duplicated identities. Shuffle the target frame to demonstrate that its order
# does not determine the pairing. Group labels determine the evaluation unit.
# Both conditions occur within every participant; participants are never encoded
# as predictors here.

design = efm.build_design(
    cohort,
    targets.sample(frac=1.0, random_state=31),
    target="condition",
    groups="participant",
)
np.testing.assert_array_equal(design.y, targets["condition"].to_numpy())
folds = efm.loso_folds(design.groups)
for fold in folds:
    assert set(design.groups[fold.train]).isdisjoint(design.groups[fold.test])

# %%
# Fit preprocessing and tune inside training participants
# -------------------------------------------------------
# The logistic pipeline fits imputation and feature scaling on training data.
# Each outer fold holds out one participant; three grouped inner folds select
# inverse regularization strength using only the other seven participants.
# A two-value demonstration grid limits runtime; a real study should prespecify
# a search range that suits its scientific question and data size.
#
# This follows scikit-learn's guidance on `grouped cross-validation
# <https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data>`_.
# Randomly splitting trials would answer a different prediction question because
# trials from the same participant could occur on both sides of a split.

pipeline = efm.logistic_pipeline(efm.PreprocessingConfig(), seed=31)
results = efm.cross_fit_classification(
    folds,
    design.X,
    design.y.astype(int),
    design.groups,
    pipeline,
    {"lr__C": [0.1, 1.0]},
    inner=efm.InnerSplit(grouping="subject", n_splits=3),
    seed=31,
    harmonization="intersection",
    scoring="balanced_accuracy",
)

# %%
# Assemble and score only held-out predictions
# --------------------------------------------
# Fold outputs contain their original design-row indices. Put predictions back
# into that order and check that every trial was tested exactly once. The
# positive-class probability column is class 1; these model probabilities have
# not undergone a separate calibration procedure.

tested_rows = np.concatenate([result.rows for result in results])
np.testing.assert_array_equal(np.sort(tested_rows), np.arange(len(design.y)))
predictions = np.empty(len(design.y), dtype=int)
probabilities = np.empty(len(design.y))
for result in results:
    assert result.classes == (0, 1)
    predictions[result.rows] = result.y_pred
    probabilities[result.rows] = result.y_prob[:, 1]

scores = efm.classification_metrics(
    design.y.astype(int), predictions, y_prob=probabilities, groups=design.groups
)
per_participant = pd.DataFrame.from_dict(scores.per_subject, orient="index")
print(per_participant[["balanced_accuracy", "auc"]].round(3))
print(f"Participant-mean balanced accuracy: {scores.balanced_accuracy:.3f}")
print(f"Participant-mean AUC: {scores.auc:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), layout="constrained")
axes[0].bar(per_participant.index, per_participant["balanced_accuracy"])
axes[0].axhline(0.5, color="0.5", linestyle="--")
axes[0].set(ylim=(0.0, 1.05), ylabel="Balanced accuracy", title="Held-out participants")
axes[0].tick_params(axis="x", rotation=60)
axes[1].boxplot([probabilities[design.y == label] for label in [0, 1]], tick_labels=["0", "1"])
axes[1].set(xlabel="True condition", ylabel="Predicted probability of condition 1")
plt.show()

# %%
# Interpret performance at the correct level
# ------------------------------------------
# The summary weights participants equally. A pooled confusion matrix instead
# counts trials, giving more weight to participants who contribute more trials.
# The condition effect was built into this simulation; above-chance scores are
# a demonstration, not evidence of a biomarker.
#
# Outer-fold scores share training participants and are not independent. Do not
# apply a one-sample t-test to these eight scores as though they were independent
# experiments. Population inference needs a justified null procedure that repeats
# the complete fitting and tuning workflow. Keep the prediction question,
# grouping, feature definition, and model search fixed before inspecting results.
# See :doc:`/guides/modeling` for inference, nulls, and model export.
