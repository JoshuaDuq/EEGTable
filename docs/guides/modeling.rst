Predictive Modeling
===================

.. raw:: html

   <p class="hero-lede">
     Evaluate EEG predictors with group-disjoint outer folds and training-only
     preprocessing and tuning. Choose the sample unit, prediction question,
     and scoring statistic before fitting.
   </p>

``eegtable.model`` is optional.

- **Install**: ``pip install "eegtable[model]"``.
- **SHAP**: needs ``pip install "eegtable[importance]"``.
- **Held-out permutation importance**: included in ``model``.

Inputs
------

Choose the input that matches the measurement's sample unit:

.. rubric:: Modeling workflows

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Samples
     - Design or evaluator
     - Use
   * - Individual epochs
     - :func:`~eegtable.model.build_design`
     - Precomputed epoch features aligned by ``(recording, epoch, event)``.
   * - Trial groups
     - :func:`~eegtable.group.build_group_design`
     - Native group features aligned by ``(recording, group)``.
   * - Signal epochs
     - ``cross_fit_signal_classification`` / ``cross_fit_signal_regression``
     - Spatial features learned again inside every training split.

The examples below use epoch features. :doc:`model_recipes` covers both table
row kinds and audited result bundles; :doc:`learned_features` covers signal
pipelines. Array regression evaluates one continuous target. Classification
evaluators and metrics support binary labels coded ``0`` and ``1``; multiclass,
multilabel, and multioutput evaluation are not provided.

**Feature table**
   The :class:`~eegtable.FeatureTable` must carry ``row_ids`` of
   ``(recording, epoch, event)`` for every row.

**Target frame**
   The frame passed to :func:`eegtable.model.build_design` must contain:

   - the same ``recording``, ``epoch``, and ``event`` columns,
   - the target column,
   - a grouping column such as ``subject_id``.

**Rejected tables**
   ``build_design`` rejects cross-trial tables (:ref:`concepts-row-kinds`).

   - ITPC, PPC, envelope correlation, every ``spectral_connectivity`` method
     including wPLI, and their graph summaries have one row per trial group.
   - Copying a group value onto its epochs repeats one number across rows.
   - Keep these as group samples with :func:`eegtable.group.read_group_dataset`
     and :func:`eegtable.group.build_group_design`. The number of contributing
     trials is a descriptor, not the number of independent model samples.
   - Precomputed CSP features are also rejected. Fit CSP inside the signal
     pipeline, including inner tuning (:doc:`learned_features`).

Building a cohort
-----------------

Stack per-epoch tables into one cohort table and align them with the targets.

Tables in memory
~~~~~~~~~~~~~~~~

Compute the same measures per recording and stack the per-epoch tables.

- **Stacking**: :func:`eegtable.stack_rows` keeps input order, values, coverage,
  flags, and metadata, and rejects duplicate row identities.
- **Differing columns**: recordings drop different bad channels, so their
  columns differ.
- ``columns="union"``: keeps every column any recording measured. A column a
  recording did not measure is NaN with zero coverage.
- ``harmonization="intersection"``: each fold then drops columns that some
  training subject lacks.
- **Alignment**: the target frame uses the same ``recording``, ``epoch``, and
  ``event`` keys. :func:`eegtable.model.build_design` aligns rows on those keys.

.. code-block:: python

   import pandas as pd
   import eegtable as ef
   import eegtable.model as efm

   selection = efm.Selection(
       band=("theta", "alpha"),
       space_kind=("channel",),
   )
   selected_tables = [efm.select(table, selection) for table in tables_by_recording]
   cohort_table = ef.stack_rows(selected_tables, columns="union")

   # Each frame must contain recording, epoch, event, reaction_time, and subject_id.
   targets = pd.concat(target_frames, ignore_index=True)
   design = efm.build_design(
       cohort_table,
       targets,
       target="reaction_time",
       groups="subject_id",
   )

Runner output
~~~~~~~~~~~~~

Pass the ``*_features.tsv`` paths to :func:`eegtable.io.read_dataset`.

.. code-block:: python

   import eegtable.model as efm
   from eegtable.io import read_dataset

   dataset = read_dataset(
       [
           "derivatives/eegtable/sub-01_task-rest_features.tsv",
           "derivatives/eegtable/sub-02_task-rest_features.tsv",
       ]
   )
   design = efm.build_design(
       dataset.table,
       dataset.targets,
       target="reaction_time",
       groups="subject_id",
   )

- **Columns**: ``read_dataset`` stacks the union of the feature columns. Pass
  ``columns="identical"`` to require one schema; a mismatch names the recording
  and the columns that differ.
- **Per-recording fits**: a measure fitted to each recording, such as its
  microstate templates, carries that fit's identity in its feature definitions.
  Independently fitted templates do not establish shared state identities.
  Exclude those columns with ``Selection(exclude=...)`` or use fold-fitted
  features or a fixed independent reference (:doc:`learned_features`).
- **Descriptors**: descriptor columns are those named in each JSON sidecar. The
  canonical key columns are rebuilt from the stored ``row_ids``, and a
  descriptor that disagrees with those identities is rejected.
- **Targets and groups**: not read from filenames. Put them in the descriptor
  rows when writing the tables.

Matching rows and subsets
~~~~~~~~~~~~~~~~~~~~~~~~~

``build_design`` requires every feature row to have a target row and every
target row a feature row, and says how many are left over on each side.

To model a subset of the epochs, such as the trials that remain after
exclusions, cut the table first.

- :meth:`~eegtable.FeatureTable.take` cuts the table to the given rows.
- Choose excluded trials before evaluation using the study's quality criteria.
- Keep missingness-based feature selection inside the training folds. Calling
  :meth:`~eegtable.FeatureTable.drop_missing` on the full cohort uses held-out
  observations to select columns even though it does not use the target.
  The modeling pipeline fits its missingness limit on each training split.

.. code-block:: python

   position = {row_id: i for i, row_id in enumerate(dataset.table.row_ids)}
   keys = zip(kept["recording"], kept["epoch"], kept["event"])
   table = dataset.table.take([position[key] for key in keys])

Selecting features and covariates
---------------------------------

:class:`eegtable.model.Selection` filters on metadata fields, not on substrings
of the generated column name. The fields are ``measure``, ``band``,
``space_kind``, ``window``, ``normalization``, and ``space``. Numeric
covariates are appended to ``X`` after the feature columns and listed in
``Design.covariate_columns``.

Targets and requested covariates must be finite numeric values; group and run
labels must be present and nonempty. Encode categorical covariates explicitly
before building the design. Any fitted encoding or data-dependent choice must
respect the training split. Use ``design.meta`` and ``design.column_names``
after selection, so metadata and arrays stay aligned.

``measure`` matches ``FeatureMeta.measure``, the label stored on the column.
That label is not always the function name or the recipe key.

- :func:`eegtable.integrated_band_power` labels columns ``"band_power"``.
- :func:`eegtable.peak_frequency` labels them ``"peak_freq_adjusted"``.
- :func:`eegtable.aperiodic` emits ``"slope"``, ``"offset"``, and
  ``"r_squared"``.
- The labels in a cohort are ``sorted({m.measure for m in cohort_table.meta})``.

.. code-block:: python

   selection = efm.Selection(
       measure=("band_power",),
       band=("alpha", "beta"),
       space_kind=("channel",),
       normalization=("log10",),
   )
   design = efm.build_design(
       cohort_table,
       targets,
       target="reaction_time",
       groups="subject_id",
       runs="run",
       covariates=("age", "sex_code"),
       selection=selection,
   )

``exclude`` takes a second ``Selection``. The columns it matches are dropped
from those the other fields keep, so leaving one measure out does not mean
listing every other one.

.. code-block:: python

   microstates = ("coverage", "duration", "occurrence", "transition")
   selection = efm.Selection(exclude=efm.Selection(measure=microstates))

- **Missing covariate**: raises. ``strict_covariates=False`` drops requested
  covariates that are absent from the target frame.
- **Target as covariate**: the target column cannot also be a covariate.

Group-disjoint cross-fitting
----------------------------

The outer split evaluates the complete fitting and tuning procedure. The group
defines the intended generalization: new participants for subject-disjoint
folds, or new runs of an observed participant for within-subject folds. Keep
related participants or recordings together when the study design requires it.

- :func:`eegtable.model.loso_folds` holds out groups.
- :func:`eegtable.model.within_subject_folds` holds out runs within each subject
  and requires a run label on every row.
- Inner tuning uses :class:`eegtable.model.InnerSplit`.

Every inner split refits preprocessing and the estimator on its training rows.
The selected pipeline is then refitted on the outer training rows. This follows
scikit-learn's `nested cross-validation example
<https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html>`_
and `grouped split documentation
<https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data>`_.

Grouped fitting rejects pipeline components with their own ``cv`` parameter,
including ``RidgeCV``, ``ElasticNetCV``, stacking, and ``RFECV``. Their internal
splits do not receive this workflow's group labels, and preprocessing outside
the component would be fitted before its validation split. Use an estimator
without internal CV and choose its parameters through the grouped inner grid.
The same restriction applies to grouped prediction intervals, including any
preprocessing retained by the quantile method.

Group separation does not by itself establish independence or a valid null.

.. code-block:: python

   config = efm.PreprocessingConfig(
       max_feature_missingness=0.2,
       max_subject_missingness=0.5,
   )
   pipeline = efm.scaled_ridge_pipeline(
       config,
       seed=42,
       n_covariates=design.n_covariates,
   )
   folds = efm.loso_folds(design.groups)
   inner = efm.InnerSplit(grouping="subject", n_splits=5)

   predictions = efm.cross_fit_regression(
       folds,
       design.X,
       design.y,
       design.groups,
       pipeline,
       efm.scaled_ridge_grid(),
       inner=inner,
       seed=42,
       harmonization="intersection",
   )

The array API limits the effective inner split count to the available training
groups, with a minimum of two. The recipe runner instead enforces the requested
count exactly. Outer folds reject overlapping groups, repeated test rows, and
invalid indices. Within-subject folds must carry ``Fold.subject``, have disjoint
runs, and use ``InnerSplit(grouping="run")`` with ``runs=design.runs``.
``ordered_runs=True`` evaluates later runs using earlier runs; it leaves the
first training runs untested. Array evaluators do not require that every input
row is tested, so inspect the returned row indices.

Pipelines
~~~~~~~~~

- **Regression**: ``scaled_ridge_pipeline``, ``ridge_pipeline``,
  ``elasticnet_pipeline``, ``random_forest_pipeline``,
  ``hist_gradient_boosting_pipeline``, and ``svr_pipeline``.
- **Classification**: ``svm_pipeline``, ``logistic_pipeline``,
  ``random_forest_classifier_pipeline``,
  ``hist_gradient_boosting_classifier_pipeline``, ``lda_pipeline``,
  and ``ensemble_pipeline``.
  :func:`eegtable.model.classification_metrics` requires labels in ``{0, 1}``.

Histogram gradient boosting
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The regression and classification factories use scikit-learn's
`histogram gradient boosting
<https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting>`_.
They retain the existing training-fitted quality/missingness policies and
median imputation; native backend support for NaN does not bypass those
policies. Scaling is omitted unless PCA is configured. Classification uses
balanced class weights estimated from each training fit.

``early_stopping=False`` prevents an internal random validation split from
mixing participants or runs. Grouped fitting rejects pipelines and grid
candidates that enable it, including ``early_stopping="auto"``. Select
``hgb__max_iter`` through the grouped inner folds instead.
``hist_gradient_boosting_grid`` and
``hist_gradient_boosting_classifier_grid`` share a compact grid over
iteration count, leaf count, minimum leaf size, and L2 regularization.
Group-aware prediction intervals apply the same restriction for split and
CV+ fitting. Ungrouped fitting can use the backend's early stopping.

Support vector regression
~~~~~~~~~~~~~~~~~~~~~~~~~

``svr_pipeline`` uses RBF-SVR with training-fitted feature scaling and a
`TransformedTargetRegressor
<https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html>`_
that standardizes the target inside every inner fit and outer refit.
Predictions are converted back to the original target units. The
``svr_grid`` parameters use ``svr__regressor__C``,
``svr__regressor__gamma``, and ``svr__regressor__epsilon``; penalties and
epsilon refer to standardized training targets, not raw rating units.
SVR is deterministic. Its fitting cost grows more than quadratically with
the number of samples, so evaluate its runtime before a large nested search.
See `scikit-learn's SVR documentation
<https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html>`_.

Shrinkage linear discriminant analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``lda_pipeline`` uses ``LinearDiscriminantAnalysis(solver="lsqr",
shrinkage="auto")`` after training-fitted preprocessing and scaling.
``lda_grid`` compares automatic shrinkage with fixed strengths. LDA is
deterministic, estimates class priors from its training labels, and provides
probabilities and decision scores. Shrinkage regularizes covariance
estimation; it does not remove LDA's shared class-covariance assumption.
The covariance matrix grows with the square of the retained feature count.
See `scikit-learn's shrinkage guidance
<https://scikit-learn.org/stable/modules/lda_qda.html#shrinkage-and-covariance-estimator>`_
and `MNE's CSP/LDA example
<https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html>`_.
Signal-level CSP followed by LDA uses ``learned_pipeline`` with a supplied
``LinearDiscriminantAnalysis`` estimator (:doc:`learned_features`).

Ridge grid
~~~~~~~~~~

:func:`~eegtable.model.scaled_ridge_pipeline` and
:func:`~eegtable.model.scaled_ridge_grid` use dimensionless penalties. At every
inner fit and outer refit, the estimator multiplies the selected alpha by the
training row count and the column count after preprocessing. No held-out rows
or full-cohort feature counts enter this scale. The fitted regressor's
``alpha_`` records the effective penalty; ``alpha`` remains the grid value.
This is the default in :doc:`model_recipes`.

:func:`~eegtable.model.ridge_grid` constructs ordinary alpha values from the
supplied array's row and column counts. It does not adapt to later fold-local
column drops. Prefer the scaled pipeline for a penalty scale determined at
each training fit. The amount of shrinkage also depends on the training design's
spectrum; sample and feature counts alone cannot establish it. See
`scikit-learn's Ridge objective
<https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_.

Tuning statistic
~~~~~~~~~~~~~~~~

Choose a tuning statistic that matches the scientific question and report it
alongside the held-out metrics. The defaults differ between entry points.

- **Tabular regression default**: each inner validation split uses subject-level
  ``r`` (:func:`~eegtable.model.subject_r_scorer`).
- **Interpretation**: subject-level correlation describes within-subject
  trial variation. Pooled error or ``R²`` also reflects between-subject offsets.
  Correlation does not assess absolute prediction error or calibration.
- **Requirement**: the scorer needs at least 3 held-out trials per validation
  subject. Degenerate correlations, including constant predictions or targets,
  contribute zero during selection. Within-subject association requires target
  variation; the recipe's ``subject_r`` option checks for it explicitly.
  For one sample per participant or a target constant within participants,
  explicitly use an error scorer such as ``"neg_mean_squared_error"``.
- **Override**: pass ``scoring`` to select on something else.
- **Other defaults**: tabular classification uses the estimator's default
  score (accuracy for the supplied classifiers). Signal pipelines and recipes
  use negative mean squared error for regression and balanced accuracy for
  classification.
- **Warning**: when every fold chooses the same end of a numeric grid,
  tabular cross-fitting warns that the search may have stopped short. An endpoint
  may also represent a meaningful limit, such as no penalty or an empty model.

Preprocessing
~~~~~~~~~~~~~

Preprocessing is fit on the training rows of the fold.

- **Steps**: the pipeline replaces infinities, drops all-NaN columns, applies the
  feature missingness limit, imputes, and removes constant columns. Cross-fitting
  checks the retained features against each training subject's missingness limit;
  excess missingness raises rather than discarding that subject.
  It can also select features, scale, deconfound, or reduce dimension with PCA.
- ``harmonization="intersection"``: keeps features that have at least one
  finite value in every training group.
- ``harmonization="union_impute"``: keeps the full feature union.
- **Target residualization**: ``covariates=...`` and ``residualize_on=...`` on
  the cross-fitting call, so the nuisance model is fit inside each outer
  training split, including each inner tuning split.
- **Scaling**: those least-squares fits scale the training design before
  solving. The numerical rank cutoff then does not depend on the units of the
  covariates.

Within-subject residualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default nuisance model is pooled: one fit over the training subjects, which
removes only the average nuisance effect. Each subject's own departure from it,
such as a steeper response to the stimulus, stays in the residual, and a
feature that follows the stimulus the same way then appears to track the
target within subjects.

``residualize_within="subject"`` fits every subject's own nuisance model
instead.

- It applies the model to the features as well as to the target. The features'
  residuals use no target.
- A subject with training rows is fitted on them alone, including the scale
  used to remove numerical rounding from residuals. Non-finite feature values
  remain missing.
- A held-out subject, as in leave-one-subject-out folds, has no training rows.
  Its nuisance fits therefore use its held-out feature and target batches
  separately. The predictive estimator receives no held-out targets, but the
  residual outcome and transformed test features depend on that subject's batch.
- This evaluates within-subject association after nuisance removal. It is not
  a training-only transform for prospective prediction of an unseen raw outcome.
  Use pooled training-fitted residualization when that is the required procedure.
- :func:`~eegtable.model.residualize_within_subjects` is the same step for a fold
  loop of your own.

.. code-block:: python

   residualized_predictions = efm.cross_fit_regression(
       folds,
       design.X,
       design.y,
       design.groups,
       pipeline,
       efm.scaled_ridge_grid(),
       inner=inner,
       seed=42,
       harmonization="intersection",
       covariates=nuisance,  # one row per design row
       residualize_on=("run", "stimulus_temp"),
       residualize_within="subject",
   )

Evaluation
----------

Predictions are mapped back to the design groups, then scored.

- :func:`eegtable.model.fold_results` returns predictions in fold order and maps
  them back to the design groups.
- :func:`eegtable.model.regression_metrics` returns Pearson ``r``, ``R²``,
  explained variance, and subject-level correlation. The default subject-level
  correlation averages Fisher ``z`` with equal weight per subject.
  Pass the fold IDs from ``fold_results``: within-subject correlation then
  centers targets and predictions separately within each subject and fold,
  preventing differences between fitted intercepts from driving the score.
- :func:`eegtable.model.classification_metrics` returns accuracy, balanced
  accuracy, AUC, average precision, F1, precision, recall, specificity, and the
  confusion matrix.

Classification details
~~~~~~~~~~~~~~~~~~~~~~

- SVM pipelines use ``probability=False``. Their predictions and decision scores
  remain available; cross-fitting returns ``y_prob=None`` and the raw scores
  in ``y_score``. Pass those scores to ``classification_metrics(y_score=...)``
  for AUC and average precision. Scores rank the positive class and are not
  calibrated probabilities. When scores and probabilities are both supplied,
  rank metrics use the scores.
- The built-in SVM/logistic/random-forest ensemble uses hard votes and returns
  ``y_prob=None``. ``calibrate_ensemble=True`` raises because calibration needs
  group-disjoint splits with preprocessing fitted inside those splits.
- Grouped fitting rejects ``SVC(probability=True)`` and
  ``CalibratedClassifierCV``: their internal calibration is not routed the
  subject or run groups used for evaluation. A requested soft ensemble with an
  SVM also raises. Explicit soft ensembles of logistic regression and random
  forests remain supported.
- With ``groups``, each scalar is the equal-weight mean over the subjects for
  which it is defined.
- A subject with one class is left out of the balanced-accuracy, AUC, and
  average-precision means. Undefined precision, recall, F1, and specificity
  remain NaN in the per-subject results and do not enter their scalar means.
  Inspect the supporting subjects for each metric.
- The confusion matrix stays pooled over trials, so accuracy recomputed from it
  can differ from the reported ``accuracy``. Call without ``groups`` for pooled
  scalar metrics.

``FoldClassification.classes`` defines the probability column order. For these
binary evaluators it is ``(0, 1)``. ``classification_metrics`` accepts either
``(n_samples, 2)`` probabilities in that order or a one-dimensional probability
of class ``1``; ``y_score`` is one finite decision score per sample, with larger
values indicating class ``1``. Logistic regression and random forests expose
probabilities, but this workflow does not assess their calibration. Estimator
capabilities do not extend the evaluator to multiclass targets.

The `scikit-learn calibration guide
<https://scikit-learn.org/stable/modules/calibration.html>`_ requires independent
classifier-training and calibration data. `SVC probability estimates
<https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_ use
internal five-fold cross-validation that cannot receive subject or run groups.

Defined values
~~~~~~~~~~~~~~

- Subject-level regression summaries require a ``subject_id`` for every trial;
  missing labels raise instead of silently dropping trials during grouping.
- Pearson correlation and centered ``R²`` use no absolute variance floor, so
  whether they are defined does not depend on the units of the target.
- Targets and predictions used for regression evaluation, subject summaries,
  centered scores, and model selection must be finite on every trial. Failed
  predictions raise instead of dropping trials from the reported scores.
- Pooled metrics with constant targets leave ``R²`` and explained variance
  undefined: their
  ``NaN`` or ``-inf`` values remain visible rather than being replaced with
  a finite score. Classification probabilities must also be finite.
- ``regression_metrics(groups=...)`` also requests a subject correlation. It
  raises if a subject has fewer than three trials or an undefined correlation.
  For group-constant targets, use pooled regression metrics and
  :func:`~eegtable.model.subject_level_errors`, or the recipe runner's error
  summaries. ``subject_level_r(undefined="zero")`` is an explicit alternative
  for degenerate correlations; it still requires three trials per subject.

See scikit-learn's `R² documentation
<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html>`_
and `explained-variance documentation
<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html>`_
for constant-target scores and the ``force_finite`` option.

.. code-block:: python

   import numpy as np

   result = efm.fold_results(
       predictions,
       groups=design.groups,
   )
   metrics, per_subject = efm.regression_metrics(
       result.y_true,
       result.y_pred,
       groups=np.asarray(result.groups, dtype=object),
       folds=np.asarray(result.folds),
   )
   print(metrics["subject_level_r"])

Dependence between subjects
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Held-out scores from cross-subject folds are not independent. Every fold model
is trained on the other subjects, so two subjects' scores share most of their
training data.

- **Default**: :func:`~eegtable.model.subject_level_r` therefore reports no
  interval unless ``AggregationConfig.ci_method`` asks for one.
- **When to ask**: an interval over subjects needs independent subject
  statistics and the interval method's other assumptions. Separate
  within-subject fits avoid shared training data across participants; they do
  not establish independence in a clustered study design.
- **Testing instead**: test cross-subject scores with
  :func:`~eegtable.model.permutation_test`, which refits the whole procedure
  under the null.

Independent statistics
~~~~~~~~~~~~~~~~~~~~~~

:func:`eegtable.model.bootstrap_mean_ci` and
:func:`eegtable.model.paired_signflip_p_value` take a pre-specified vector of
independent subject-level statistics, which leave-one-subject-out scores are
not.

- ``bootstrap_mean_ci`` is a 95% percentile interval of the mean. Pass
  Fisher-:math:`z` values, not correlations.
- ``paired_signflip_p_value`` is a two-sided test of zero mean on paired
  differences and returns :math:`(b + 1)/(B + 1)`. Finite-sample validity also
  requires each subject's null difference to be symmetric about zero;
  independence and zero mean alone do not establish this.
- Both functions require finite values for every included subject. Exclude
  subjects using pre-specified study criteria before constructing the vector.

Permutation nulls
-----------------

:func:`eegtable.model.permutation_test` refits the full cross-fitting procedure
on each draw. It is for regression and refits
:func:`~eegtable.model.cross_fit_regression`.
The supplied ``observed`` statistic is recomputed and must match the same
folds, pipeline, grid, seed, harmonization, nuisance settings, metric, and
aggregation. Selection of features or analysis settings outside this procedure
is not included in the null.

Schemes
~~~~~~~

``NullConfig.scheme`` accepts ``"within_subject"``,
``"within_subject_within_run"``, and ``"circular_shift_within_run"``.

- ``"run_wise"`` is an alias of ``"within_subject_within_run"``, the name used
  for this shuffle in the upstream pipeline (``runwise``).
- Both of those schemes shuffle labels inside each run of each subject. No
  scheme exchanges whole runs.
- Run structure differs by paradigm. Run-aware schemes require ``runs``.
- Circular shifts also require finite integer trial indices
  (``trial_indices``) that are unique inside each subject and run.
- Circular shifts also require at least ``min_retained_trials`` trials per run
  (default 8).
- A draw that fails to fit raises. Failed draws are not dropped from the null.
- A subject with only one sample cannot be permuted. If every sampled draw
  leaves the target unchanged, the test raises. In particular, within-subject
  shuffling cannot test a participant-level target constant across that
  participant's epochs. No between-subject target shuffle is provided here.

These schemes specify rearrangements, not proof of their validity. Under the
null, the joint label or residual distribution must be invariant under the
chosen rearrangements. Run labels alone do not make temporally dependent
trials exchangeable. Circular shifts require invariance under cyclic shifts
of the retained, ordered trial sequence; ordinary stationarity alone does not
establish that condition.

Tail
~~~~

``greater_is_better`` chooses the tail. The default is ``True``.

- Set it to ``False`` when a smaller value is the better score, as with
  ``mean_squared_error``.
- Left at the default, a strong effect on an error metric returns
  :math:`p \approx 1`, and only a model that scores worse than the permuted
  refits returns a small :math:`p`.
- The smallest attainable Monte Carlo :math:`p` is
  :math:`1/(B + 1)` for :math:`B` draws. Ties count as at least as extreme;
  the null :math:`p` distribution can be conservative and discrete.
- The direction is an argument. It is not inferred from ``metric_fn``.

Nuisance covariates
~~~~~~~~~~~~~~~~~~~

With ``residualize_on``, pooled or within subjects, the implementation uses a
Freedman-Lane residual reconstruction (Freedman & Lane, 1983; Winkler et al., 2014).

- Each fold fits its nuisance model exactly as cross-fitting does, keeps that
  fit's prediction, and permutes only its residuals.
- A draw rearranges the residual component around the fitted nuisance
  prediction. This targets association beyond the specified nuisance model;
  its validity still depends on the nuisance model and residual exchangeability.
- Permuting the raw target also rearranges its nuisance-associated component
  and tests a different hypothesis.
- The residuals are exchanged within the scheme's blocks, which must stay inside
  each fold.

Batched ridge draws
~~~~~~~~~~~~~~~~~~~

A ridge pipeline fits permutation targets together when all of these hold:

- its preprocessing never sees the target, as in the default
  ``ridge_pipeline``, including any ``ColumnTransformer`` remainder;
- it is tuned on the subject-level ``r`` and scored with it.
- its regressor is unconstrained ``Ridge`` with an intercept and an ``auto``,
  ``cholesky`` or ``svd`` solver; only the scalar penalty is tuned, and every
  candidate penalty is finite and strictly positive.

Each fold and inner split transforms its features once. Its configured
`scikit-learn Ridge solver
<https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_
then fits batches of permutation targets as multiple outputs for each penalty.
Target residualization and the Freedman-Lane rebuild are linear in the target.
The resulting null matches individual refits up to numerical roundoff. Very
small penalties and nearly dependent predictors amplify that roundoff.
``alpha=0`` raises for batched inference: with dependent predictors, fitting
multiple targets together can change both the null statistic and the p-value.
This restriction does not change ordinary cross-fitting.

- **Full refitting**: other pipelines, custom container subclasses, a
  target-driven step such as ``feature_selection_percentile``, a ``metric_fn``
  or another ``scoring`` refit every draw.
  ``scaled_ridge_pipeline`` also takes this path; the batched path requires
  the ordinary scikit-learn ``Ridge`` class.

.. code-block:: python

   null = efm.permutation_test(
       folds,
       design.X,
       design.y,
       design.groups,
       design.runs,
       pipeline,
       efm.scaled_ridge_grid(),
       metrics["subject_level_r"],
       config=efm.NullConfig(
           scheme="within_subject",
           n_permutations=1000,
       ),
       inner=inner,
       seed=42,
       harmonization="intersection",
   )
   print(null.p_value)

Univariate screen
-----------------

:func:`eegtable.model.univariate_screen` asks which single features track the
target within subjects.

- **Statistic**: each subject's correlation is taken over its own trials,
  averaged across subjects in Fisher :math:`z`, and tested with a one-sample
  :math:`t`.
- **Independence**: no predictive model shares training data across subjects.
  Independence of their statistics must still follow from the study design.
- ``p_fwer``: flipping the sign of a subject's :math:`z` for every feature at
  once keeps the dependence between features. The largest :math:`|t|` over those
  flips gives the max-statistic adjustment ``p_fwer``. A randomization
  interpretation requires independent subject vectors and a joint null
  distribution invariant under sign reversal. Independence alone does not
  establish that symmetry.
- ``q``: the Benjamini-Hochberg adjustment of ``p``.
  Its usual FDR interpretation requires valid input p-values and appropriate
  dependence assumptions; arbitrary correlated EEG features do not establish them.
- ``residualize_on``: removes each subject's own nuisance design from both sides
  first.
- Every trial needs a subject label; missing labels raise.

Statistics use centered sums of squares so identical or nearly identical
subject effects do not produce negative variances through numerical
cancellation. If both the mean effect and its between-subject variance are
zero, ``t``, ``p``, ``q`` and ``p_fwer`` are undefined (NaN). Such features do
not enter the Benjamini-Hochberg adjustment.

.. code-block:: python

   screen = efm.univariate_screen(
       design.X[:, design.feature_columns],
       design.y,
       design.groups,
       feature_names=[m.name for m in design.meta],
   )
   print(screen.sort_values("p_fwer").head())

This screen is a separate association analysis. Selecting predictors from a
full-cohort screen and then reporting cross-validation on that same cohort
uses the held-out outcomes. Predictive feature selection belongs inside the
training pipeline.

Conformal intervals
-------------------

:func:`eegtable.model.prediction_intervals` returns bounds for ``"split"``,
``"cv_plus"``, or ``"quantile"`` conformal calibration.

- **Groups**: with ``groups``, the fitting split and the calibration split are
  group-disjoint.
- **Pooling**: calibration scores are still pooled over trials, so a participant
  with more trials contributes more scores.
- **No new-participant guarantee**: the procedure does not give a
  distribution-free coverage guarantee for a new participant.
- **Result fields**: ``lower``, ``upper``, ``alpha``, ``method``, and
  ``calibration_unit``.
- **Not stored**: the coverage realized on a test set.

Coverage guarantees
~~~~~~~~~~~~~~~~~~~

- ``"split"`` uses a finite-sample upper quantile of absolute calibration
  residuals. The usual marginal :math:`1 - \alpha` coverage statement requires
  exchangeable calibration and test trials relative to the training-fitted
  model (Lei et al., 2018).
- ``"cv_plus"`` combines each validation residual with predictions from that
  validation fold's training model. This implementation uses :math:`\alpha`
  in each tail. Its theoretical lower bound is weaker than
  :math:`1 - \alpha`, with finite-fold corrections under the conditions in
  Barber et al. (2021).
- ``"quantile"`` uses quantile regressors and conformalized residual scores
  in CV+ form (Romano et al., 2019). It replaces the pipeline's final estimator
  with quantile gradient boosting while retaining its preprocessing.
- Group-disjoint folds do not make the pooled calibration trials exchangeable.
  The trial-level bounds do not establish coverage for a new participant or
  coverage conditional on a participant, feature value, or condition.
- Hyperparameters are used as supplied; this function does not run the nested
  tuning procedure. Select them without using calibration or test outcomes.
- Insufficient calibration samples for the requested tail can yield infinite
  endpoints. They remain visible in the result.
- A non-finite calibration score or prediction raises.

Example
~~~~~~~

The test rows must not be used for fitting or calibration, or no coverage
statement applies to them. The example holds out the first subject.

.. code-block:: python

   held_out = design.groups == design.groups[0]
   intervals = efm.prediction_intervals(
       pipeline,
       design.X[~held_out],
       design.y[~held_out],
       design.X[held_out],
       alpha=0.10,
       method="cv_plus",
       cv_splits=5,
       seed=42,
       groups=design.groups[~held_out],
   )
   lower, upper = intervals.lower, intervals.upper

Importance
----------

The ``*_over_folds`` helpers refit the specified tuning and fitting procedure,
then explain its held-out predictions. Match the folds, pipeline, grid, seed,
scoring, harmonization, and nuisance settings used for evaluation.

- :func:`eegtable.model.permutation_importance_over_folds` computes held-out
  permutation importance. Pass the feature names that were selected so a score
  can be matched after fold-local column drops.
- :func:`eegtable.model.shap_importance_over_folds` computes SHAP values.
- SHAP explanations after PCA or covariate deconfounding cannot be assigned to
  individual input features by the feature-name mapping and raise an error.
  Permutation importance perturbs input columns before the pipeline transforms
  them and supports these models.

The example below fits EEG features only, even if ``design`` has covariates.
To explain a model that includes covariates:

- fit on a pipeline with the same ``n_covariates``;
- pass ``design.column_names``;
- keep only ``design.feature_columns`` before aggregating, because covariates
  have no ``FeatureMeta`` record and :func:`~eegtable.model.aggregate_by`
  refuses them.

.. code-block:: python

   feature_pipeline = efm.scaled_ridge_pipeline(config, seed=42)
   importance = efm.permutation_importance_over_folds(
       folds,
       design.X[:, design.feature_columns],
       design.y,
       design.groups,
       feature_pipeline,
       efm.scaled_ridge_grid(),
       inner=inner,
       feature_names=[m.name for m in design.meta],
       seed=42,
       harmonization="intersection",
   )
   by_band = efm.aggregate_by(importance, design.meta, field="band")

The reported value is the mean decrease in the held-out score when that
feature is permuted.

- ``scoring=None``: the score is the one the model was selected on, the
  subject-level ``r`` for regressors and accuracy for classifiers.

Importance describes the fitted predictor and the chosen perturbation. It
does not establish a causal effect or a standalone physiological biomarker.
Correlated features can share predictive information and mask each other's
permutation importance, as illustrated in `scikit-learn's correlated-feature example
<https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html>`_.

References
----------

* Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018).
  *Distribution-free predictive inference for regression*. Journal of the
  American Statistical Association, 113(523), 1094--1111.
  `doi:10.1080/01621459.2017.1307116
  <https://doi.org/10.1080/01621459.2017.1307116>`__.
* Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021).
  *Predictive inference with the jackknife+*. The Annals of Statistics, 49(1),
  486--507. `doi:10.1214/20-AOS1965 <https://doi.org/10.1214/20-AOS1965>`__.
* Romano, Y., Patterson, E., & Candès, E. J. (2019). *Conformalized quantile
  regression*. Advances in Neural Information Processing Systems, 32.
* Freedman, D., & Lane, D. (1983). *A nonstochastic interpretation of reported
  significance levels*. Journal of Business & Economic Statistics, 1(4),
  292--298.
* Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols,
  T. E. (2014). *Permutation inference for the general linear model*.
  NeuroImage, 92, 381--397. `doi:10.1016/j.neuroimage.2014.01.060
  <https://doi.org/10.1016/j.neuroimage.2014.01.060>`__.
