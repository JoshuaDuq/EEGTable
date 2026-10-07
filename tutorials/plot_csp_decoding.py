"""
Decoding Covariance Contrasts with Training-Only CSP
====================================================

Learn common spatial patterns within participant-disjoint training folds, then
fit scaling and a classifier on those same training coordinates. Evaluate only
the predictions of participants excluded from every learned step.

The covariance effect is deliberately planted in this offline simulation.
Its scores illustrate the workflow, not a biomarker. Install
``python -m pip install -e ".[docs,model]"`` to run it.
"""

# %%
# Simulate two conditions in two small cohorts
# --------------------------------------------
# Four participants in each cohort contribute 24 balanced trials. Condition 0
# emphasizes one sensor mixing map; condition 1 emphasizes another. Independent
# phases, trial variation, participant-specific mixing/gain, and cohort-specific
# sensor noise make these trials distinct. These are sensor mixtures, not a
# forward model or a motor imagery experiment. Amplitudes are in volts.

import matplotlib.pyplot as plt
import mne
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import eegtable as ef
from eegtable import model as efm

rng = np.random.default_rng(47)
sampling_rate = 128.0
times = np.arange(512) / sampling_rate
channel_names = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2")
info = mne.create_info(channel_names, sampling_rate, "eeg")
info.set_montage(mne.channels.make_standard_montage("colin27_1020"))
positions = np.array([channel["loc"][:2] for channel in info["chs"]])
mixing_maps = positions.T.copy()
mixing_maps -= mixing_maps.mean(axis=1, keepdims=True)
mixing_maps /= np.linalg.norm(mixing_maps, axis=1, keepdims=True)
trials_per_participant = 24
data_blocks, label_blocks, participant_blocks, row_ids = [], [], [], []
label_by_identity = {}
for participant in range(8):
    cohort = "a" if participant < 4 else "b"
    identity = f"cohort-{cohort}-p{participant + 1:02d}"
    labels = rng.permutation(np.tile([0, 1], trials_per_participant // 2))
    participant_maps = mixing_maps + rng.normal(0.0, 0.035, mixing_maps.shape)
    participant_maps -= participant_maps.mean(axis=1, keepdims=True)
    participant_maps /= np.linalg.norm(participant_maps, axis=1, keepdims=True)
    phases = rng.uniform(0, 2 * np.pi, (trials_per_participant, 2, 1))
    waves = np.sin(2 * np.pi * np.array([10.0, 18.0])[None, :, None] * times + phases)
    amplitudes = 4e-6 * np.ones((trials_per_participant, 2))
    amplitudes[np.arange(trials_per_participant), labels] *= 1.8
    amplitudes *= rng.lognormal(0.0, 0.28, amplitudes.shape)
    participant_gain = rng.lognormal(0.0, 0.2)
    samples = participant_gain * np.einsum(
        "sc,nst->nct", participant_maps, amplitudes[..., None] * waves
    )
    noise_scale = 0.8e-6 if cohort == "a" else 1.2e-6
    samples += rng.normal(0.0, noise_scale, samples.shape)
    data_blocks.append(samples)
    label_blocks.append(labels)
    participant_blocks.extend([identity] * trials_per_participant)
    for trial, label in enumerate(labels):
        row_id = (identity, trial, f"condition-{label}")
        row_ids.append(row_id)
        label_by_identity[row_id] = int(label)
data = np.concatenate(data_blocks)
labels = np.concatenate(label_blocks)
participants = np.array(participant_blocks, dtype=object)
row_ids = tuple(row_ids)
assert np.isfinite(data).all()

# %%
# Apply fixed transforms independently before splitting
# -----------------------------------------------------
# Average reference, the 8-24 Hz band, and the 0.5-3.5 s crop are prespecified.
# MNE filters each epoch along time without concatenating participants or trials.
# Filtering the complete four-second epochs before cropping limits edge effects;
# half-second margins are a demonstration choice, not a guarantee for real data.
# A zero-phase filter is noncausal and unsuitable for claims of online latency.
#
# These fixed per-epoch operations learn no sample or label statistics from the
# cohort. Data-driven artifact correction, band/window selection, or a learned
# reference would instead belong inside training partitions. Follow the
# `MNE filter API <https://mne.tools/stable/generated/mne.filter.filter_data.html>`_
# when choosing padding and filter settings for a real recording.

data -= data.mean(axis=1, keepdims=True)
filtered = mne.filter.filter_data(
    data,
    sampling_rate,
    8.0,
    24.0,
    method="iir",
    phase="zero",
    iir_params={"order": 4, "ftype": "butter", "output": "sos"},
    verbose=False,
)
inside = (times >= 0.5) & (times <= 3.5)
cropped_data = filtered[..., inside]
assert np.isfinite(cropped_data).all()
signal = ef.Signal(
    data=cropped_data,
    times=times[inside],
    ch_names=channel_names,
    sfreq=sampling_rate,
    coverage=np.ones_like(cropped_data),
    row_ids=row_ids,
    computation=ef.ComputationSpec.create(
        "fixed-simulation-preprocessing",
        seed=47,
        unit="V",
        reference="average",
        band_hz=[8.0, 24.0],
        filter="zero-phase-butterworth-order-4",
        crop_s=[0.5, 3.5],
    ),
    passband=(8.0, 24.0),
)
np.testing.assert_array_equal(labels, [label_by_identity[row] for row in signal.row_ids])
np.testing.assert_array_equal(participants, [row[0] for row in signal.row_ids])

# %%
# Fit the full learned chain separately in every fold
# ---------------------------------------------------
# CSP is supervised: its spatial filters see training labels. Following the
# `MNE CSP example
# <https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html>`_,
# spatial filtering belongs with the classifier inside validation. Here explicit
# folds make native ``Signal`` identities and participant exclusion visible.
# `Grouped cross-validation
# <https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data>`_
# defines the evaluation unit: all trials of one participant are held out together.
# Both cohorts remain represented in training; this does not test a new cohort.
#
# Two components, 0.1 covariance shrinkage, and logistic C=1 are fixed before
# looking at scores. There is no tuning. Selecting any of these using the outer
# scores would require nested participant-disjoint validation, with CSP, scaling,
# and classifier refitted on every inner training split as well.

folds = efm.loso_folds(participants)
predictions = np.empty(labels.size, dtype=int)
probabilities = np.empty(labels.size)
test_count = np.zeros(labels.size, dtype=int)
fold_features = np.empty((labels.size, 2))  # Used only for a descriptive API check below.
scores, fitted_models = [], []
for fold in folds:
    assert set(participants[fold.train]).isdisjoint(participants[fold.test])
    assert set(labels[fold.train]) == set(labels[fold.test]) == {0, 1}
    csp = ef.CommonSpatialPattern.fit(
        signal,
        labels,
        rows=fold.train,
        n_components=2,
        regularization=0.1,
    )
    assert csp.computation.parameters["n_fit_epochs"] == len(fold.train)
    frozen_filters = csp.filters.copy()
    training_features = csp.transform(signal, rows=fold.train)
    test_features = csp.transform(signal, rows=fold.test)
    assert np.isfinite(training_features).all() and np.isfinite(test_features).all()
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", random_state=47),
    )
    classifier.fit(training_features, labels[fold.train])
    np.testing.assert_allclose(
        classifier.named_steps["standardscaler"].mean_,
        training_features.mean(axis=0),
    )
    np.testing.assert_array_equal(classifier.classes_, [0, 1])
    predictions[fold.test] = classifier.predict(test_features)
    probabilities[fold.test] = classifier.predict_proba(test_features)[:, 1]
    test_count[fold.test] += 1
    fold_features[fold.test] = test_features
    np.testing.assert_array_equal(csp.filters, frozen_filters)
    scores.append(balanced_accuracy_score(labels[fold.test], predictions[fold.test]))
    fitted_models.append(csp)
np.testing.assert_array_equal(test_count, np.ones(labels.size, dtype=int))
print(f"Participant-mean held-out balanced accuracy: {np.mean(scores):.3f}")
for fold, score in zip(folds, scores, strict=True):
    print(f"{participants[fold.test[0]]}: {score:.3f}")

# %%
# Verify what the native features mean
# ------------------------------------
# Native CSP averages trace-normalized, temporally centered epoch covariances
# within each training class. Its transform is log relative variance over the
# retained components, not log absolute power. Raising the component count changes
# this denominator. This differs from the usual MNE estimator settings described
# in the `MNE CSP API <https://mne.tools/stable/generated/mne.decoding.CSP.html>`_.
# A default-estimator comparison would therefore not check the same calculation.
# Instead, project held-out epochs directly and verify the native transformation.

first_fold, first_model = folds[0], fitted_models[0]
projected = first_model.filters @ signal.data[first_fold.test]
component_variance = projected.var(axis=-1, ddof=0)
expected = np.log(component_variance / component_variance.sum(axis=1, keepdims=True))
np.testing.assert_allclose(fold_features[first_fold.test], expected, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(np.exp(expected).sum(axis=1), 1.0)

# %%
# Keep fold-specific coordinates descriptive
# ------------------------------------------
# ``csp_features`` produces one held-out row per epoch under the supplied split.
# Check row identity and equality with the transforms above. This table is useful
# for auditing, but each fold has different fitted filters and thus different
# coordinates. Component labels indicate ordered variance contrasts, not shared
# anatomical locations. Do not concatenate them as common predictors for another
# cross-validation: other folds' training fits can encode the current test labels.
# The predictions above always use one CSP fit for both training and test rows.

descriptive = ef.csp_features(
    signal,
    labels,
    folds=folds,
    n_components=2,
    regularization=0.1,
)
assert descriptive.row_ids == signal.row_ids
np.testing.assert_allclose(descriptive.values, fold_features, rtol=1e-12, atol=1e-12)
assert descriptive.meta[0].computation.parameters["cross_fitted"]
print(f"Descriptive CSP split identity: {descriptive.meta[0].computation.parameters['folds']}")

# %%
# Inspect training patterns and held-out predictions
# --------------------------------------------------
# Plot forward patterns from one training fold, not spatial filter coefficients.
# Pattern sign and scale are arbitrary; sensor interpolation is a display aid,
# not source localization. Averaging patterns across folds requires an explicit
# alignment convention. Average reference removes one rank dimension; shrinkage
# acts within the measured subspace and cannot restore removed information.

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), layout="constrained")
limit = np.abs(first_model.patterns).max()
for component, axis in enumerate(axes):
    pattern = first_model.patterns[component]
    mne.viz.plot_topomap(
        pattern,
        info,
        axes=axis,
        show=False,
        names=channel_names,
        cmap="RdBu_r",
        vlim=(-limit, limit),
    )
    axis.set_title(f"Component {component + 1}; λ={first_model.eigenvalues[component]:.2f}")
fig.suptitle(f"Training forward patterns; excludes {participants[first_fold.test[0]]}")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), layout="constrained")
axes[0].bar(np.arange(1, 9), scores, color=["#4477AA"] * 4 + ["#AA3377"] * 4)
axes[0].axhline(0.5, color="0.5", linestyle="--")
axes[0].set(
    xticks=range(1, 9),
    xlabel="Held-out participant (cohorts a, b)",
    ylabel="Balanced accuracy",
    ylim=(0, 1.05),
)
axes[1].boxplot([probabilities[labels == label] for label in [0, 1]], tick_labels=["0", "1"])
axes[1].set(xlabel="True condition", ylabel="Held-out probability of condition 1", ylim=(0, 1))
plt.show()

# %%
# Match inference to the prediction question
# ------------------------------------------
# Scores assess a new participant from these two simulated cohorts. They do not
# establish transport to another acquisition system, cohort, or population.
# The condition effect was planted; above-chance accuracy is a demonstration.
# Participants receive equal weight, whereas pooling unequal trial counts would
# change that weighting. Outer folds share training participants and their scores
# are not independent replicates. Scientific inference needs a justified null
# procedure that repeats the complete learned chain and respects the study's
# participant/condition structure. Report reference, rank, artifact handling,
# filter/window choices, covariance normalization, regularization, and split
# identities together with the classifier settings.
