Predictive modeling
===================

Install ``eegtable[model]`` for the modeling APIs. Workflows are described in
:doc:`/guides/modeling`, :doc:`/guides/learned_features`, and
:doc:`/guides/model_recipes`.

The evaluators support one continuous regression target or binary
classification labels coded ``0`` and ``1``. Multiclass, multilabel, and
multioutput evaluation are not provided. Epoch and trial-group samples have
distinct design builders; learned spatial features are fitted from signal
epochs inside the training pipeline.

Epoch and group designs
-----------------------

``build_design`` accepts epoch-row tables. Native trial-group samples use
``eegtable.group`` and keep their original sample unit. Neither design builder
accepts precomputed CSP features for predictive evaluation.

.. autoclass:: eegtable.model.Selection
   :members:

.. autoclass:: eegtable.model.Design
   :members:

.. autofunction:: eegtable.model.select

.. autofunction:: eegtable.model.build_design

.. autofunction:: eegtable.model.harmonize_fold

.. autofunction:: eegtable.model.compute_train_group_intersection_mask

.. autofunction:: eegtable.group.read_group_dataset

.. autofunction:: eegtable.group.build_group_design

Preprocessing
-------------

Fit these transformations inside the training splits. Pipeline factories below
compose missingness, imputation, scaling, and optional selection/PCA steps from
``PreprocessingConfig``.

.. autoclass:: eegtable.model.PreprocessingConfig
   :members:

.. autoclass:: eegtable.model.ReplaceInfWithNaN
   :members:

.. autoclass:: eegtable.model.DropAllNaNColumns
   :members:

.. autoclass:: eegtable.model.MissingnessThreshold
   :members:

.. autoclass:: eegtable.model.VarianceThreshold
   :members:

.. autoclass:: eegtable.model.SpatialFeatureSelector
   :members:

.. autoclass:: eegtable.model.Deconfounder
   :members:

.. autofunction:: eegtable.model.base_preprocessing_steps

.. autofunction:: eegtable.model.validate_subject_missingness

.. autofunction:: eegtable.model.transform_feature_names

Grouped splits
--------------

Subject-disjoint outer folds use subject-disjoint inner tuning. Within-subject
folds hold out runs and require run-disjoint inner tuning. The grouping variable
specifies what must remain disjoint.

.. autoclass:: eegtable.model.Fold
   :members:

.. autoclass:: eegtable.model.InnerSplit
   :members:

.. autofunction:: eegtable.model.loso_folds

.. autofunction:: eegtable.model.within_subject_folds

.. autofunction:: eegtable.model.inner_cv

.. autofunction:: eegtable.model.inner_cv_splits

.. autofunction:: eegtable.model.run_aware_cv

.. autofunction:: eegtable.model.run_aware_inner_cv

Regression estimators and grids
-------------------------------

.. autofunction:: eegtable.model.scaled_ridge_pipeline

.. autofunction:: eegtable.model.scaled_ridge_grid

.. autofunction:: eegtable.model.ridge_pipeline

.. autofunction:: eegtable.model.ridge_grid

.. autofunction:: eegtable.model.elasticnet_pipeline

.. autofunction:: eegtable.model.elasticnet_grid

.. autofunction:: eegtable.model.random_forest_pipeline

.. autofunction:: eegtable.model.random_forest_grid

.. autofunction:: eegtable.model.hist_gradient_boosting_pipeline

.. autofunction:: eegtable.model.hist_gradient_boosting_grid

.. autofunction:: eegtable.model.svr_pipeline

.. autofunction:: eegtable.model.svr_grid

Classification estimators and grids
-----------------------------------

The supplied SVM and ensemble use uncalibrated scores and hard voting,
respectively. Grouped fitting rejects hidden SVM probability calibration and
``CalibratedClassifierCV``. Probability availability does not imply calibration.

.. autofunction:: eegtable.model.logistic_pipeline

.. autofunction:: eegtable.model.logistic_grid

.. autofunction:: eegtable.model.svm_pipeline

.. autofunction:: eegtable.model.svm_grid

.. autofunction:: eegtable.model.random_forest_classifier_pipeline

.. autofunction:: eegtable.model.random_forest_classifier_grid

.. autofunction:: eegtable.model.hist_gradient_boosting_classifier_pipeline

.. autofunction:: eegtable.model.hist_gradient_boosting_classifier_grid

.. autofunction:: eegtable.model.lda_pipeline

.. autofunction:: eegtable.model.lda_grid

.. autofunction:: eegtable.model.ensemble_pipeline

Tabular cross-fitting and tuning
--------------------------------

.. autofunction:: eegtable.model.cross_fit_regression

.. autofunction:: eegtable.model.cross_fit_classification

.. autoclass:: eegtable.model.FoldPrediction
   :members:

.. autoclass:: eegtable.model.FoldClassification
   :members:

.. autofunction:: eegtable.model.tune

.. autofunction:: eegtable.model.fit_untuned

.. autoclass:: eegtable.model.TunedFit
   :members:

.. autoclass:: eegtable.model.FoldFitError
   :members:

Learned signal features
-----------------------

See :doc:`/guides/learned_features` for nested fits from a ``Signal``.
Covariance and tangent estimation additionally require ``eegtable[riemann]``.

.. autoclass:: eegtable.model.CSPTransformer
   :members:

.. autoclass:: eegtable.model.MicrostateTransformer
   :members:

.. autoclass:: eegtable.model.CovarianceTransformer
   :members:

.. autoclass:: eegtable.model.TangentSpaceTransformer
   :members:

.. autofunction:: eegtable.model.learned_pipeline

.. autofunction:: eegtable.model.cross_fit_signal_classification

.. autofunction:: eegtable.model.cross_fit_signal_regression

Metrics and model selection scores
----------------------------------

Regression metrics take aligned one-dimensional finite arrays. Classification
metrics require labels coded ``0`` and ``1``. Preserve the class-column order
and supporting groups when passing scores or probabilities.

.. autofunction:: eegtable.model.regression_metrics

.. autofunction:: eegtable.model.classification_metrics

.. autoclass:: eegtable.model.ClassificationResult
   :members:

.. autofunction:: eegtable.model.within_subject_centered_metrics

.. autofunction:: eegtable.model.within_condition_metrics

.. autofunction:: eegtable.model.subject_r_scorer

.. autofunction:: eegtable.model.pearsonr_scorer

.. autofunction:: eegtable.model.safe_pearsonr

.. autofunction:: eegtable.model.scoring_dict

Prediction aggregation
----------------------

Pass fold IDs when a subject is predicted by multiple fitted models. Intervals
and tests over subject statistics require their stated independence and null
assumptions; group-disjoint fitting alone does not establish them.

.. autofunction:: eegtable.model.fold_results

.. autoclass:: eegtable.model.FoldResults
   :members:

.. autoclass:: eegtable.model.AggregationConfig
   :members:

.. autofunction:: eegtable.model.subject_level_r

.. autoclass:: eegtable.model.SubjectLevelR
   :members:

.. autofunction:: eegtable.model.subject_level_errors

.. autofunction:: eegtable.model.bootstrap_mean_ci

.. autofunction:: eegtable.model.paired_signflip_p_value

Nuisance models and residualization
-----------------------------------

Pooled nuisance fits use training rows. Subject-specific residualization fits
a subject with no training rows on its held-out batch; this defines a residual
association estimand rather than a training-only transform of new raw outcomes.

.. autofunction:: eegtable.model.residualize_targets

.. autofunction:: eegtable.model.residualize_within_subjects

.. autofunction:: eegtable.model.fit_nuisance_model

.. autoclass:: eegtable.model.FoldNuisanceFit
   :members:

.. autofunction:: eegtable.model.fit_staged_residual_preprocessor

.. autoclass:: eegtable.model.StagedResidualPreprocessor
   :members:

.. autofunction:: eegtable.model.reconstruct_staged_permutation_target_for_fold

Permutation nulls and association screening
-------------------------------------------

``permutation_test`` refits tabular regression. Justify the rearrangement from
the study's null and dependence structure; the schemes do not establish
exchangeability. ``univariate_screen`` is a separate association analysis,
not a full-cohort predictive feature selector.

.. autofunction:: eegtable.model.permutation_test

.. autoclass:: eegtable.model.NullConfig
   :members:

.. autoclass:: eegtable.model.NullResult
   :members:

.. autodata:: eegtable.model.Scheme

.. autofunction:: eegtable.model.permute

.. autofunction:: eegtable.model.circular_shift_group

.. autofunction:: eegtable.model.changed_fraction

.. autofunction:: eegtable.model.univariate_screen

Prediction intervals
--------------------

Calibration scores are pooled over trials even with group-disjoint splits.
These functions do not establish distribution-free coverage for a new group.

.. autofunction:: eegtable.model.prediction_intervals

.. autoclass:: eegtable.model.PredictionIntervals
   :members:

Feature importance
------------------

The fold helpers refit the supplied evaluation procedure. Importance describes
the predictor and perturbation; it does not establish a causal effect.
SHAP additionally requires ``eegtable[importance]``.

.. autoclass:: eegtable.model.Importance
   :members:

.. autofunction:: eegtable.model.permutation_importance

.. autofunction:: eegtable.model.permutation_importance_over_folds

.. autofunction:: eegtable.model.shap_importance

.. autofunction:: eegtable.model.shap_importance_over_folds

.. autofunction:: eegtable.model.aggregate_by

Reproducible fold execution
---------------------------

.. autofunction:: eegtable.model.run_folds

.. autofunction:: eegtable.model.seeded

.. autofunction:: eegtable.model.set_random_seeds

.. autofunction:: eegtable.model.inner_n_jobs

.. autofunction:: eegtable.model.should_parallelize

Recipe loading and result bundles
---------------------------------

These runner APIs implement :doc:`/guides/model_recipes` and consume epoch or
group feature tables. Learned signal pipelines and inferential analyses use
the Python APIs above.

.. autofunction:: eegtable.runner.model_recipe.load_model_recipe

.. autoclass:: eegtable.runner.model_recipe.ModelRecipe
   :members:

.. autofunction:: eegtable.runner.model_run.check_model

.. autoclass:: eegtable.runner.model_run.ModelCheck
   :members:

.. autofunction:: eegtable.runner.model_run.run_model

.. autoclass:: eegtable.runner.model_run.ModelResult
   :members:
