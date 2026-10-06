<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/JoshuaDuq/EEGTable/main/assets/branding/eegtable-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/JoshuaDuq/EEGTable/main/assets/branding/eegtable-logo.svg">
    <img src="https://raw.githubusercontent.com/JoshuaDuq/EEGTable/main/assets/branding/eegtable-logo.svg" width="402" height="96" alt="EEGTable">
  </picture>
</h1>

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/JoshuaDuq/EEGTable/blob/main/LICENSE)
[![MNE-Python ≥ 1.10](https://img.shields.io/badge/mne--python-≥1.10-blue.svg)](https://mne.tools/stable/)
[![Docs](https://img.shields.io/badge/docs-Sphinx-blue.svg)](https://joshuaduq.github.io/EEGTable/)

EEGTable extracts labelled EEG features from MNE objects and supports reproducible cohort analysis, regression, and binary classification. Spectral, temporal, connectivity, complexity, cycle, and microstate measures return structured `FeatureTable` objects.

Each table retains feature definitions, physical units, computation parameters, row identities, finite-input coverage, and quality flags. Modeling supports group-disjoint evaluation and training-fold preprocessing and tuning.

**Current version:** `0.1.0.dev0` (development release). Install from source and record the source revision used in an analysis.

[Documentation](https://joshuaduq.github.io/EEGTable/) · [Methods](https://joshuaduq.github.io/EEGTable/methods/index.html) · [API reference](https://joshuaduq.github.io/EEGTable/api/index.html) · [Validation evidence](https://joshuaduq.github.io/EEGTable/guides/validation.html)

## Installation

Python 3.11 or newer is required. From a macOS or Linux terminal:

```bash
git clone https://github.com/JoshuaDuq/EEGTable.git
cd EEGTable
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[model]"
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. The core package requires NumPy ≥ 1.26, SciPy ≥ 1.11, pandas ≥ 2.0, and MNE ≥ 1.10. The `model` extra adds modeling and YAML recipes; it is optional for core feature extraction.

| Extra | Capability |
| :--- | :--- |
| `model` | Estimator pipelines, grouped evaluation, metrics, nulls, and uncertainty |
| `connectivity` | Spectral connectivity, wPLI, and per-epoch time-frequency connectivity via MNE-Connectivity |
| `microstates` | Microstate template fitting and segmentation via scikit-learn |
| `spectral-model` | Fixed/knee spectral parameterization via specparam 2.0 |
| `irasa` | Aperiodic/oscillatory separation via NeuroDSP |
| `cycles` | Cycle waveform and burst features via ByCycle; keeps pandas below 3.0, which ByCycle 1.2.0 does not support |
| `complexity` | Permutation entropy, Lempel–Ziv complexity, and DFA via AntroPy |
| `pac` | PAC surrogate inference via Tensorpac |
| `riemann` | Covariance and tangent-space modeling via pyRiemann |
| `bids` | Native BIDS discovery and loading via MNE-BIDS |
| `importance` | SHAP explanations; permutation importance is included in `model` |
| `preprocessing` | Raw-to-epochs workflow; requires MNE ≥ 1.13.2 |
| `preprocessing-auto` | PyPREP, ICLabel, Picard, and autoreject integrations |
| `preprocessing-gui` | MNE Qt viewers for interactive review |
| `docs`, `dev` | Documentation build; tests, typing, and lint |

Combine extras for the intended workflow, for example `python -m pip install -e ".[preprocessing,preprocessing-auto,model]"`. See [Installation](https://joshuaduq.github.io/EEGTable/install.html) for complete dependencies and platform instructions.

## Quick start

This example uses a preprocessed epochs file with at least one second of data per epoch, sampling frequency above 90 Hz, and the EEG channels `C3`, `Cz`, and `C4`. MNE represents EEG amplitudes in volts. Choose bands, windows, references, and estimator settings for your study.

```python
import mne
import eegtable as ef

from eegtable.io import write_table

epochs = mne.read_epochs("sub-01_epo.fif", preload=True)

alpha = ef.Band("alpha", 8.0, 13.0)
rois = {"central": ["C3", "Cz", "C4"]}
spectra = ef.Spectra.welch(
    epochs, recording="sub-01", fmin=1.0, fmax=45.0, n_fft=round(epochs.info["sfreq"])
)
features = ef.integrated_band_power(spectra, bands=[alpha], groups=rois)

frame = features.to_dataframe()  # indexed by recording, epoch and event
if epochs.metadata is not None:
    frame = frame.join(epochs.metadata, on="epoch")
long = features.to_long()
paths = write_table(features, "sub-01_features.tsv", rows=epochs.metadata)
```

Raw integrated PSD power has units of V². `normalize="log10"` is available for log-scaled power. `Spectra.welch` keeps good EEG channels, computes the spectra and records the settings that produced them in every column's identity; it is the estimator `eegtable run` uses, so the same settings name the same columns. `to_dataframe` indexes rows by recording, epoch and event, so joining `epochs.metadata` on `epoch` keeps each row with its own trial; `to_long` gives one row per value, the layout mixed models read.

The same in one call, from a recipe: `ef.extract(epochs, {"features": [{"measure": "integrated_band_power", "bands": ["alpha"]}]}, recording="sub-01")`.

The [Quick start](https://joshuaduq.github.io/EEGTable/quickstart.html) adds Morlet power, bursts, and ERD/ERS. [Tables and files](https://joshuaduq.github.io/EEGTable/guides/tables.html) explains selection, serialization, and stacking.

## Data and provenance

A feature name contains six readable fields and a digest of the complete column specification:

```text
eeg_<measure>_<band>_<space>_<window>_<normalization>_p<12-hex-digest>
```

Structured metadata is authoritative; `FeatureTable.select()` uses that metadata. The digest also distinguishes estimator settings, frequency bounds, ROI membership, and other computation parameters that are absent from the readable fields.

- **Epoch rows** retain `(recording, original epoch index, event)` identities.
- **Trial-group rows** describe estimates across trials, such as ITPC and epoch-averaged connectivity. They are written separately and have a dedicated `eegtable.group` design interface.
- **Coverage** measures finite-input availability. It does not measure artifact removal or signal quality. Morlet temporal support is recorded separately on `Spectra`.
- **Files** contain values and coverage TSVs plus a JSON sidecar. Newly written schema 2 bundles include descriptor types and payload checksums. Extraction provenance records resolved settings, input identities, software/source identities, and available preprocessing evidence.

See [Data concepts](https://joshuaduq.github.io/EEGTable/concepts.html) and [Reproducible cohorts](https://joshuaduq.github.io/EEGTable/guides/cohorts.html).

## Command-line workflows

Apply a TOML extraction recipe to preprocessed epochs files:

```bash
eegtable init recipe.toml
eegtable check recipe.toml
eegtable run recipe.toml
eegtable status recipe.toml
eegtable report recipe.toml quality.html --by recording
```

`init` also provides `--template task` and `--template resting`. `check` validates the recipe and computes the first recording without writing feature bundles. See the [Runner guide](https://joshuaduq.github.io/EEGTable/guides/runner.html).

Preprocess raw recordings with a YAML recipe and explicit review policies:

```bash
eegtable preprocess init preprocessing.yaml --mode events
eegtable preprocess check preprocessing.yaml
eegtable preprocess run preprocessing.yaml
```

See [Preprocessing](https://joshuaduq.github.io/EEGTable/guides/preprocessing.html) for review decisions, checkpoint verification, and event/fixed-length workflows, and [Native BIDS input](https://joshuaduq.github.io/EEGTable/guides/bids.html) for BIDS recordings. The optional [`tui/`](https://github.com/JoshuaDuq/EEGTable/tree/main/tui) preprocessing interface requires Go 1.24 or newer.

## Modeling

`eegtable.model` supports regression and binary classification with group-disjoint outer folds and optional inner tuning. Ridge, elastic net, random forest, histogram gradient boosting, SVR, logistic regression, SVM classification, shrinkage LDA, and ensembles are available. Preprocessing is fitted within training folds; nested tuning uses training groups. SVR standardizes targets within each training fit and returns predictions in their original units. Histogram boosting disables internal early stopping and tunes iteration counts through grouped folds. Permutation inference must repeat the chosen fitting procedure under a null whose exchangeability assumptions match the study.

Training-fitted CSP, microstate, covariance, and tangent-space transforms support learned features. Fixed quality policies retain coverage, flags, and exclusion evidence. Predicting new participants, predicting new runs of known participants, and estimating within-participant associations require different split and scoring choices.

```bash
eegtable model init model.yaml
eegtable model check model.yaml
eegtable model run model.yaml
```

Model recipes support participant-level targets and explicit within-participant correlation scoring. Runs include a dummy baseline and, when covariates are supplied, a covariate-only baseline on the same held-out rows, with machine-readable comparisons. The starter uses Ridge penalties scaled inside each training fit. SVM outputs retain decision scores for AUC; these are not calibrated probabilities.

[Predictive modeling](https://joshuaduq.github.io/EEGTable/guides/modeling.html) · [Learned features](https://joshuaduq.github.io/EEGTable/guides/learned_features.html) · [Model recipes](https://joshuaduq.github.io/EEGTable/guides/model_recipes.html)

## Methods and scientific evidence

| Family | Selected entry points |
| :--- | :--- |
| Power and spectral shape | `integrated_band_power`, `mean_tfr_power`, `peak_frequency`, `spectral_entropy`, `aperiodic` |
| Spectral separation | `spectral_parameterization`, `irasa`, `periodic_power` |
| Time domain | `variance`, `peak_amplitude`, `peak_latency`, `area_under_curve`, `hjorth_mobility`, `hjorth_complexity` |
| Bursts and ERD/ERS | `burst_rate`, `burst_duration`, `erds_mean`, `erds_onset_latency`, `erds_rebound_latency` |
| Phase and connectivity | `itpc`, `ppc`, `pac`, `pac_surrogates`, `envelope_correlation`, `spectral_connectivity`, `spectral_connectivity_time`, `wpli` |
| Complexity | `sample_entropy`, `multiscale_entropy`, `higuchi_fractal_dimension`, `permutation_entropy`, `lempel_ziv_complexity`, `detrended_fluctuation` |
| Cycles and microstates | `cycle_features`, `MicrostateModel`, `segment`, `microstate_coverage`, `microstate_transitions` |
| Spatial and graph summaries | `CommonSpatialPattern`, `csp_features`, `global_efficiency`, `clustering_coefficient` |
| Quality and reliability | `QualityPolicy`, `apply_quality`, `feature_quality`, `cohort_quality`, `intraclass_reliability` |

[Method definitions](https://joshuaduq.github.io/EEGTable/methods/index.html) describe formulas, units, assumptions, and missing-value behavior. [Public-dataset validation](https://joshuaduq.github.io/EEGTable/guides/validation.html) distinguishes numerical agreement, estimator comparisons, physiological effects, and decoding. Its saved results identify the tested source and environment; they do not establish every method's validity for every study or automatically cover subsequent code changes.

For publications, record the EEGTable version and source revision, dependency versions, preprocessing, reference, estimator settings, bands/windows/ROIs, exclusions, split units, tuning, null scheme, and random seeds. Retain recipes, sidecars, and validation snapshots with the analysis. Cite the underlying methods and packages used; references appear in the method guides.

The [`examples/`](https://github.com/JoshuaDuq/EEGTable/tree/main/examples) directory contains reproducible simulated outputs. [`paradigm_specific/`](https://github.com/JoshuaDuq/EEGTable/tree/main/paradigm_specific) contains study-specific raw-to-BIDS scripts outside the public package API.

## Development and documentation

```bash
python -m pip install -e ".[dev,docs,model,connectivity,microstates,importance,preprocessing,preprocessing-auto,bids,spectral-model,irasa,cycles,complexity,pac,riemann]"
python -m pytest
python -m ruff check src tests
python -m black --check src tests
python -m mypy
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

Public-dataset checks are opt-in and download recordings on first use:

```bash
EEGTABLE_DATASETS=1 python -m pytest tests/validation -ra
```

Each dataset run writes its own evidence snapshot and updates the documentation's generated validation tables. See the [Validation guide](https://joshuaduq.github.io/EEGTable/guides/validation.html) for dependencies and execution details.

## Contributing and citing

Bug reports, fixes and new measures are welcome. [CONTRIBUTING](https://github.com/JoshuaDuq/EEGTable/blob/main/CONTRIBUTING.md) covers setup, the checks a pull request must pass, and how to add a measure. Changes are listed in the [changelog](https://github.com/JoshuaDuq/EEGTable/blob/main/CHANGELOG.md).

To cite EEGTable, use [CITATION.cff](https://github.com/JoshuaDuq/EEGTable/blob/main/CITATION.cff) (GitHub's "Cite this repository"), alongside the methods and packages your analysis relied on.

## License

MIT. See [LICENSE](https://github.com/JoshuaDuq/EEGTable/blob/main/LICENSE).
