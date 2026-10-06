import types

import eegtable

EXPECTED = {
    "spectral_parameterization",
    "extract",
    "signal_measure",
    "spectral_measure",
    "irasa",
    "cycle_features",
    "permutation_entropy",
    "lempel_ziv_complexity",
    "detrended_fluctuation",
    "pac_surrogates",
    "spectral_connectivity_time",
    "MicrostateModel",
    "QualityPolicy",
    "QualityResult",
    "apply_quality",
    "cohort_quality",
    "feature_quality",
    "intraclass_reliability",
    "BANDS_STANDARD",
    "Band",
    "CommonSpatialPattern",
    "csp_features",
    "check_passband",
    "passband_fraction",
    "BandSignal",
    "FeatureMeta",
    "FeatureTable",
    "MicrostateSegmentation",
    "Signal",
    "Spectra",
    "Window",
    "__version__",
    "aperiodic",
    "aperiodic_ratio",
    "area_under_curve",
    "asymmetry",
    "band_ratio",
    "burst_amplitude",
    "burst_count",
    "burst_duration",
    "burst_rate",
    "clustering_coefficient",
    "ComputationSpec",
    "concat",
    "envelope_correlation",
    "erd_duration",
    "erd_magnitude",
    "erds_mean",
    "erds_onset_latency",
    "erds_peak_latency",
    "erds_rebound_latency",
    "erds_slope",
    "ers_duration",
    "ers_magnitude",
    "fraction_above_threshold",
    "global_efficiency",
    "itpc",
    "integrated_band_power",
    "mean_amplitude",
    "mean_psd",
    "mean_tfr_power",
    "microstate_coverage",
    "microstate_duration",
    "microstate_occurrence",
    "microstate_transitions",
    "multiscale_entropy",
    "pac",
    "peak_amplitude",
    "peak_frequency",
    "peak_latency",
    "peak_to_peak",
    "periodic_power",
    "ppc",
    "sample_entropy",
    "segment",
    "spectral_bandwidth",
    "spectral_centroid",
    "spectral_edge",
    "spectral_entropy",
    "stack_rows",
    "hjorth_complexity",
    "hjorth_mobility",
    "amplitude_quantile",
    "kurtosis",
    "line_length",
    "root_mean_square",
    "skewness",
    "zero_crossing_rate",
    "higuchi_fractal_dimension",
    "variance",
    "spectral_connectivity",
    "wpli",
}


def test_public_namespace_is_exactly_the_documented_surface() -> None:
    assert set(eegtable.__all__) == EXPECTED


def test_every_exported_name_resolves() -> None:
    for name in eegtable.__all__:
        assert getattr(eegtable, name) is not None


def test_no_private_name_is_exported() -> None:
    assert not [n for n in eegtable.__all__ if n.startswith("_") and n != "__version__"]


def test_the_only_private_attributes_are_submodules() -> None:
    # Importing eegtable._expand binds the name `_expand` on the package. That is
    # how Python packages work, not an export. What must not appear is a private
    # *value*: a helper function or constant that escaped into the namespace.
    private = [n for n in dir(eegtable) if n.startswith("_") and not n.startswith("__")]
    assert all(isinstance(getattr(eegtable, n), types.ModuleType) for n in private)
