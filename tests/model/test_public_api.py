from __future__ import annotations

import types

import eegtable.model as model

EXPECTED = {
    "CSPTransformer",
    "MicrostateTransformer",
    "CovarianceTransformer",
    "TangentSpaceTransformer",
    "learned_pipeline",
    "cross_fit_signal_classification",
    "cross_fit_signal_regression",
    "AggregationConfig",
    "ClassificationResult",
    "Deconfounder",
    "Design",
    "DropAllNaNColumns",
    "Fold",
    "FoldClassification",
    "FoldFitError",
    "FoldNuisanceFit",
    "FoldPrediction",
    "FoldResults",
    "Importance",
    "InnerSplit",
    "MissingnessThreshold",
    "NullConfig",
    "NullResult",
    "PredictionIntervals",
    "PreprocessingConfig",
    "ReplaceInfWithNaN",
    "Scheme",
    "Selection",
    "SpatialFeatureSelector",
    "StagedResidualPreprocessor",
    "SubjectLevelR",
    "TunedFit",
    "VarianceThreshold",
    "aggregate_by",
    "base_preprocessing_steps",
    "bootstrap_mean_ci",
    "build_design",
    "changed_fraction",
    "circular_shift_group",
    "classification_metrics",
    "compute_train_group_intersection_mask",
    "cross_fit_classification",
    "cross_fit_regression",
    "elasticnet_grid",
    "elasticnet_pipeline",
    "ensemble_pipeline",
    "fit_nuisance_model",
    "fit_staged_residual_preprocessor",
    "fit_untuned",
    "fold_results",
    "harmonize_fold",
    "inner_cv",
    "inner_cv_splits",
    "inner_n_jobs",
    "logistic_grid",
    "logistic_pipeline",
    "loso_folds",
    "paired_signflip_p_value",
    "pearsonr_scorer",
    "permutation_importance",
    "permutation_importance_over_folds",
    "permutation_test",
    "permute",
    "prediction_intervals",
    "random_forest_classifier_grid",
    "random_forest_classifier_pipeline",
    "random_forest_grid",
    "random_forest_pipeline",
    "reconstruct_staged_permutation_target_for_fold",
    "regression_metrics",
    "residualize_targets",
    "residualize_within_subjects",
    "ridge_grid",
    "ridge_pipeline",
    "scaled_ridge_grid",
    "scaled_ridge_pipeline",
    "run_aware_cv",
    "run_aware_inner_cv",
    "run_folds",
    "safe_pearsonr",
    "scoring_dict",
    "select",
    "seeded",
    "set_random_seeds",
    "shap_importance",
    "shap_importance_over_folds",
    "should_parallelize",
    "subject_level_errors",
    "subject_level_r",
    "subject_r_scorer",
    "svm_grid",
    "svm_pipeline",
    "transform_feature_names",
    "tune",
    "univariate_screen",
    "validate_subject_missingness",
    "within_condition_metrics",
    "within_subject_centered_metrics",
    "within_subject_folds",
}


def test_public_namespace_is_exactly_the_documented_surface() -> None:
    assert set(model.__all__) == EXPECTED


def test_every_exported_name_resolves() -> None:
    for name in model.__all__:
        assert getattr(model, name) is not None


def test_no_private_name_is_exported() -> None:
    assert not [n for n in model.__all__ if n.startswith("_")]


def test_the_only_private_attributes_are_submodules_or_deps() -> None:
    private = [n for n in dir(model) if n.startswith("_") and not n.startswith("__")]
    assert all(isinstance(getattr(model, n), types.ModuleType) for n in private)
