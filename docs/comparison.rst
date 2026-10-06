EEGTable and Related Tools
==========================

.. rst-class:: hero-lede

   What EEGTable adds, what it borrows, and when another tool fits better.

EEGTable is built for one kind of question: what each trial's EEG looks like in
named time windows, relative to a baseline, in a table whose every column says
exactly what it measured. Most of its formulas also exist elsewhere; many of its
measures call those packages directly. What it adds is the layer around them:

- **Event-related windows and baselines** as first-class inputs. These include
  single-trial ERD/ERS magnitudes and latencies, burst thresholds calibrated on
  each trial's baseline, and Morlet power restricted to coefficients the window
  fully supports.
- **Per-column identity and provenance.** Units, bands, windows, ROI
  membership and every computation parameter are recorded, and each column is
  named from a digest of all of them. Finite-input coverage, support and flags
  sit beside every value.
- **Reproducible cohorts.** TOML recipes produce the same columns for every
  recording. Status tracks which results are current, and a public-dataset
  scorecard records what each measure was checked against.

Tools EEGTable builds on
------------------------

**MNE-Python** computes the spectra, time-frequency representations and filters
EEGTable starts from. Use MNE directly for averaged measures, such as the ERP
peak and area functions in ``mne.stats.erp``, or for the time courses
themselves, such as a baselined TFR from ``apply_baseline``. EEGTable reduces
such representations to per-trial values in named windows.

**specparam, NeuroDSP, ByCycle, AntroPy, Tensorpac, MNE-Connectivity,
scikit-learn and pyRiemann** compute specific measures that EEGTable wraps:
spectral parameterization, IRASA, cycle features, entropy and fractal measures,
PAC surrogates, connectivity, microstate clustering and covariance features.
EEGTable records their versions in each output's provenance. Use them directly
for options it does not expose.

Tools for other jobs
--------------------

**MNE-Features** and **EEGDash's feature bank** compute many features over
fixed-length windows, for machine learning across large datasets. They suit that
job well. When windows are locked to events and compared with a baseline,
EEGTable's window and baseline model fits better.

**MNE-BIDS-Pipeline** runs preprocessing, time-frequency analysis and decoding
from BIDS datasets end to end. Its epochs can be EEGTable's input. EEGTable's
own preprocessing workflow is optional and experimental; see :doc:`stability`.

**NeuroKit2** covers physiological signals broadly, and **YASA** sleep
staging and sleep-specific events. Use them for those domains.

**julearn** and scikit-learn provide general leakage-safe cross-validation.
:mod:`eegtable.model` adds what trial-level EEG tables need on top:
participant-level targets from trial rows, exclusions that respect coverage,
nulls that respect run structure, and CSP or microstate templates fitted inside
each training fold.

When not to use EEGTable
------------------------

- Continuous recordings without events, such as sleep or resting-state
  monitoring, scored epoch by epoch: a dedicated tool, or MNE directly, is
  simpler.
- Source-space analysis, or real-time decoding.
- A single measure on a single recording, where an MNE call answers the
  question directly.
