"""Labelled EEG feature extraction and modeling for MNE objects.

`eegtable` turns MNE objects and NumPy arrays into labelled feature tables.
Feature metadata keeps channels, bands, windows, and normalizations as
structured fields rather than fragments of column names.
"""

from __future__ import annotations

from eegtable.aperiodic import aperiodic, aperiodic_ratio
from eegtable.bands import BANDS_STANDARD, Band, check_passband, passband_fraction
from eegtable.bursts import (
    burst_amplitude,
    burst_count,
    burst_duration,
    burst_rate,
    fraction_above_threshold,
)
from eegtable.complexity import (
    detrended_fluctuation,
    higuchi_fractal_dimension,
    lempel_ziv_complexity,
    multiscale_entropy,
    permutation_entropy,
    sample_entropy,
)
from eegtable.connectivity import (
    clustering_coefficient,
    envelope_correlation,
    global_efficiency,
    spectral_connectivity,
    spectral_connectivity_time,
    wpli,
)
from eegtable.csp import CommonSpatialPattern, csp_features
from eegtable.custom import signal_measure, spectral_measure
from eegtable.cycles import cycle_features
from eegtable.derived import asymmetry, band_ratio
from eegtable.descriptors import (
    peak_frequency,
    spectral_bandwidth,
    spectral_centroid,
    spectral_edge,
    spectral_entropy,
)
from eegtable.erds import (
    erd_duration,
    erd_magnitude,
    erds_mean,
    erds_onset_latency,
    erds_peak_latency,
    erds_rebound_latency,
    erds_slope,
    ers_duration,
    ers_magnitude,
)
from eegtable.extraction import extract
from eegtable.irasa import irasa
from eegtable.microstates import (
    MicrostateModel,
    MicrostateSegmentation,
    microstate_coverage,
    microstate_duration,
    microstate_occurrence,
    microstate_transitions,
    segment,
)
from eegtable.phase import itpc, pac, pac_surrogates, ppc
from eegtable.power import integrated_band_power, mean_psd, mean_tfr_power, periodic_power
from eegtable.quality import (
    QualityPolicy,
    QualityResult,
    apply_quality,
    cohort_quality,
    feature_quality,
)
from eegtable.reliability import intraclass_reliability
from eegtable.signal import BandSignal, Signal
from eegtable.spectra import Spectra, Window
from eegtable.spectral_model import spectral_parameterization
from eegtable.table import ComputationSpec, FeatureMeta, FeatureTable, concat, stack_rows
from eegtable.temporal import (
    amplitude_quantile,
    area_under_curve,
    hjorth_complexity,
    hjorth_mobility,
    kurtosis,
    line_length,
    mean_amplitude,
    peak_amplitude,
    peak_latency,
    peak_to_peak,
    root_mean_square,
    skewness,
    variance,
    zero_crossing_rate,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "spectral_parameterization",
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
    "__version__",
    "aperiodic",
    "aperiodic_ratio",
    "area_under_curve",
    "asymmetry",
    "Band",
    "CommonSpatialPattern",
    "csp_features",
    "check_passband",
    "passband_fraction",
    "band_ratio",
    "BANDS_STANDARD",
    "BandSignal",
    "burst_amplitude",
    "burst_count",
    "burst_duration",
    "burst_rate",
    "clustering_coefficient",
    "concat",
    "ComputationSpec",
    "envelope_correlation",
    "spectral_connectivity",
    "erd_duration",
    "erd_magnitude",
    "erds_mean",
    "extract",
    "signal_measure",
    "spectral_measure",
    "erds_onset_latency",
    "erds_peak_latency",
    "erds_rebound_latency",
    "erds_slope",
    "ers_duration",
    "ers_magnitude",
    "FeatureMeta",
    "FeatureTable",
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
    "MicrostateSegmentation",
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
    "Signal",
    "Spectra",
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
    "Window",
    "wpli",
]
