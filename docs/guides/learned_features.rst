Learned spatial features
========================

CSP filters, microstate templates, sensor subspaces, and tangent-space references
are estimated from epochs. Fit them inside each training split when evaluating
prediction, including the inner splits used to select hyperparameters.
A table assembled from separately cross-fitted CSP features remains
unsafe for classifier cross-validation, because other folds' feature fits can
encode the current test labels; ``build_design`` continues to reject it.

Install ``eegtable[model]`` for pipelines and ``eegtable[riemann]`` for pyRiemann
covariance and tangent-space estimation. These APIs accept finite real epochs,
with the same sensor names, sensor order, sampling frequency, and preprocessing
across recordings. Each evaluator takes one :class:`~eegtable.Signal`, with data
shaped ``(epochs, channels, times)`` and one target and group label per epoch.
It requires at least two channels and two time samples, with nonzero temporal
variance in every epoch. Choose a prespecified frequency band and crop the
analysis window on the ``Signal``. Selecting those settings from held-out
performance requires an additional training-only selection procedure.

The :doc:`CSP tutorial </auto_tutorials/plot_csp_decoding>` fits spatial filters
and a classifier inside each participant-disjoint split. The
:doc:`microstate tutorial </auto_tutorials/plot_microstate_templates>` separates
template fitting from assignment on held-out epochs.

Signal classification supports only integer labels coded ``0`` and ``1``.
Signal regression accepts one finite continuous target per epoch. Backend
support for multiclass classification or multioutput regression does not
extend these evaluators' contract. These pipelines are Python APIs; the
current YAML recipes accept precomputed feature tables (:doc:`model_recipes`).

Training fits and new recordings
--------------------------------

``MicrostateModel.fit`` learns templates exclusively from the selected rows.
``model.segment`` assigns a new recording without changing those templates.
The existing ``segment`` convenience function still fits and segments its input
in one call, with the same polarity-invariant algorithm and output measures.

.. code-block:: python

   from eegtable import MicrostateModel, microstate_coverage, Window

   model = MicrostateModel.fit(training_signal, n_states=4)
   held_out = model.segment(new_signal, min_duration_ms=20.0)
   coverage = microstate_coverage(
       held_out, windows=[Window("rest", 0.0, 2.0)]
   )

``fit(signal, rows=training_rows)`` also permits selecting training rows from a
larger signal. The fitting step validates and reads only those epochs. A new
recording must have exactly the fitted channels in order; reorder or reconcile
the montage explicitly before use.

State identities and external references
----------------------------------------

Unmatched clusters are ``state1`` onward. Their numbers are arbitrary and do not
claim canonical A-D identities. Name a reference set explicitly, then match a
training fit to it by a one-to-one assignment that maximizes total absolute
spatial correlation:

.. code-block:: python

   reference = MicrostateModel.from_templates(
       published_maps,
       ch_names=training_signal.ch_names,
       labels=("A", "B", "C", "D"),
       reference_name="study, montage, and template version",
   )
   matched = model.match_reference(reference)
   held_out = matched.segment(new_signal)

The maps are normalized, copied, and frozen. Matching reorders the fitted maps
and records the assignment, correlations, training computation, and reference
identity. A reference may itself be used to segment every recording with
``reference.segment(signal)``; this shares exactly the same fixed maps across
recordings. A matched training fit instead retains the maps learned in that
training set. For predictive evaluation, choose the reference independently of
all held-out subjects and runs. Matching does not make a reference estimated
from the complete evaluation cohort safe.

Nested grouped classification
-----------------------------

Use ``CSPTransformer`` or ``MicrostateTransformer`` as the first step of a
scikit-learn pipeline. ``learned_pipeline`` adds training-fitted scaling and
the supplied estimator. Hyperparameter names use ``features__``, ``scale__``,
and ``model__``.

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from eegtable.model import (
       CSPTransformer,
       InnerSplit,
       cross_fit_signal_classification,
       learned_pipeline,
       loso_folds,
   )

   pipeline = learned_pipeline(
       CSPTransformer(ch_names=signal.ch_names, n_components=2),
       LogisticRegression(max_iter=1000),
   )
   predictions = cross_fit_signal_classification(
       loso_folds(subjects),
       signal,
       labels,  # integer labels coded 0 and 1
       subjects,
       pipeline,
       {"features__regularization": [0.0, 0.1], "model__C": [0.1, 1.0]},
       inner=InnerSplit("subject", n_splits=3),
       seed=42,
   )

Every inner training split refits the spatial features, scaling, and classifier.
After tuning, the selected pipeline refits on the outer training rows and
predicts only the held-out rows. Classification tunes balanced accuracy by
default. Results are the existing ``FoldClassification`` containers: original
row indices, observed labels, predictions, probabilities and decision scores
when available, classes, and selected parameters. An empty grid performs an
untuned outer fit. All returned predictions must be finite and aligned.

Probabilities follow ``FoldClassification.classes == (0, 1)``. SVMs expose
decision scores with ``y_prob=None``; those scores can supply AUC and average
precision without probability calibration (:doc:`modeling`). Preserve score
and probability arrays when combining folds; ``fold_results`` collects labels
and predictions but does not collect either of those arrays.

For microstates, replace the feature step with:

.. code-block:: python

   from eegtable.model import MicrostateTransformer

   features = MicrostateTransformer(
       ch_names=signal.ch_names,
       sfreq=signal.sfreq,
       n_states=4,
       measures=("coverage", "occurrence"),
       reference=reference,
       random_state=42,
   )

Templates are fitted inside each training split, then matched to the supplied
independent reference. Coverage and occurrence are defined even when a state
is absent. ``duration`` and ``transitions`` are available as additional measures,
but retain undefined values as NaN; a compatible, explicitly chosen downstream
missing-value strategy is required for estimators that cannot accept them.

Outer folds are checked for subject overlap, repeated test observations,
invalid indices, and run overlap within a subject. ``within_subject_folds``
requires ``runs`` and ``InnerSplit("run")``. Cross-subject folds require
``InnerSplit("subject")``. Grouped calibration restrictions are shared with the
tabular modeling API; an SVC must have ``probability=False``.

Covariance and tangent-space models
-----------------------------------

``CovarianceTransformer`` delegates epoch covariance estimation to pyRiemann.
It estimates the measured sensor subspace using training data only, removes
null dimensions such as the average-reference dimension, then computes
covariance matrices in that fixed basis. OAS shrinkage is explicit and enabled
by default. ``estimator="scm"`` uses empirical covariance and fails when epochs
are too short or singular. Shrinkage never recreates removed sensor dimensions.
New recordings outside the training sensor subspace are rejected.

``TangentSpaceTransformer`` adds pyRiemann's tangent mapping. Its default
reference is the Riemannian mean of training covariances, and ``tsupdate=False`` prevents
the held-out batch from updating that reference. Transforming a single epoch
or including it in a batch gives the same vector.

.. code-block:: python

   from sklearn.linear_model import Ridge
   from eegtable.model import TangentSpaceTransformer, cross_fit_signal_regression

   pipeline = learned_pipeline(
       TangentSpaceTransformer(ch_names=signal.ch_names, estimator="oas"),
       Ridge(),
   )
   predictions = cross_fit_signal_regression(
       loso_folds(subjects), signal, continuous_target, subjects,
       pipeline, {"model__alpha": [0.1, 1.0, 10.0]},
       inner=InnerSplit("subject", n_splits=3), seed=42,
   )

Regression returns existing ``FoldPrediction`` containers and tunes negative
mean squared error by default. Supply a standard scikit-learn scorer when a
different selection objective is needed. Microstate features also support
regression. CSP requires binary class labels.

Covariance matrices may instead be passed directly to a pyRiemann classifier:

.. code-block:: python

   from pyriemann.classification import MDM
   from sklearn.pipeline import Pipeline
   from eegtable.model import CovarianceTransformer

   pipeline = Pipeline([
       ("covariance", CovarianceTransformer(ch_names=signal.ch_names)),
       ("model", MDM(metric="riemann")),
   ])

Use the same signal-level classification evaluator and a ``model__`` grid.
Matrix-valued covariances do not pass through ``StandardScaler``;
``learned_pipeline`` is for vector-valued CSP, microstate, and tangent features.

Package examples and methods
----------------------------

The pipelines follow the established fit-inside-cross-validation pattern in
`MNE's CSP motor imagery example
<https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html>`_.
Covariance and tangent mappings delegate to `pyRiemann's documented estimators
<https://pyriemann.readthedocs.io/en/latest/auto_examples/biosignal-mi/plot_single.html>`_
and `training-reference tangent-space API
<https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.tangentspace.TangentSpace.html>`_.
Separate microstate fitting, prediction, and explicit state matching agree with
the workflows exposed by `Pycrostates ModKMeans
<https://pycrostates.readthedocs.io/en/stable/api/generated/pycrostates.cluster.ModKMeans.html>`_.
